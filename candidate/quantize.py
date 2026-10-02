import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from analyze import load_images, miou
from optimize import NetOpt, load_net

HERE = Path(__file__).parent
FIG = HERE / "figures"


# calibration data

def smooth_background(rng, n=128):
    """Low-frequency colored background: random coarse grid upsampled with bicubic."""
    grid = torch.from_numpy(rng.random((1, 3, rng.integers(3, 9), rng.integers(3, 9)))).float()
    return F.interpolate(grid, size=(n, n), mode="bicubic", align_corners=False)[0].clamp(0, 1)


def shapes_image(rng, n=128):
    """Random image in the style of the task data: smooth background, 1-4 flat-colored
    circles / rectangles / triangles, light pixel noise."""
    bg = (smooth_background(rng, n).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.integers(1, 5)):
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        cx, cy = (int(v) for v in rng.integers(10, n - 10, 2))
        r = int(rng.integers(6, 25))
        kind = rng.integers(3)
        if kind == 0:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            continue
        if kind == 1:
            h = r * rng.uniform(0.5, 1.5)
            corners = [(-r, -h), (r, -h), (r, h), (-r, h)]
        else:
            corners = [(r * np.cos(k * 2 * np.pi / 3), r * np.sin(k * 2 * np.pi / 3)) for k in range(3)]
        a = rng.random() * 2 * np.pi
        pts = [(cx + dx * np.cos(a) - dy * np.sin(a), cy + dx * np.sin(a) + dy * np.cos(a)) for dx, dy in corners]
        draw.polygon(pts, fill=color)
    x = np.asarray(img, np.float32) / 255 + rng.normal(0, 0.02, (n, n, 3))
    return torch.from_numpy(np.clip(x, 0, 1).astype(np.float32)).permute(2, 0, 1)


def calibration_set(kind, num=32, seed=0):
    """uniform: U[0,1] per pixel; gauss: N(0.45, 0.25) clipped; shapes: synthetic scenes."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return torch.from_numpy(rng.random((num, 3, 128, 128), dtype=np.float32))
    if kind == "gauss":
        return torch.from_numpy(np.clip(rng.normal(0.45, 0.25, (num, 3, 128, 128)), 0, 1).astype(np.float32))
    if kind == "shapes":
        return torch.stack([shapes_image(rng) for _ in range(num)])
    raise ValueError(kind)


# fake quantization

def fake_quant_act(x, lo, hi, bits):
    """Asymmetric per-tensor: [lo, hi] -> integers 0 .. 2^bits - 1 -> back to float."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)                 # zero must be representable exactly
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    zero = round(-lo / scale)
    q = torch.clamp(torch.round(x / scale) + zero, 0, levels)
    return (q - zero) * scale


def fake_quant_weight(w, bits, per_channel, axis=0):
    """Symmetric: [-max|w|, max|w|] -> integers -(2^(b-1)-1) .. 2^(b-1)-1 -> back to float."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        amax = w.abs().amax(dim=[d for d in range(w.ndim) if d != axis], keepdim=True)
    else:
        amax = w.abs().max()
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def conv_layers(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))}


def out_axis(mod):
    return 1 if isinstance(mod, nn.ConvTranspose2d) else 0   # ConvTranspose weight: [in, out, k, k]


@torch.no_grad()
def calibrate(model, data, batch=8, q=0.9999):
    """Input range of every conv: global min/max, mean per-batch percentiles, per-channel max."""
    stats = {}

    def hook(name):
        def fn(mod, inputs):
            x = inputs[0]
            s = stats.setdefault(name, {"min": float("inf"), "max": float("-inf"), "plo": [], "phi": [], "ch_max": None})
            s["min"], s["max"] = min(s["min"], x.min().item()), max(s["max"], x.max().item())
            flat = x.flatten()
            s["plo"].append(torch.quantile(flat, 1 - q).item())
            s["phi"].append(torch.quantile(flat, q).item())
            ch = x.amax(dim=(0, 2, 3))
            s["ch_max"] = ch if s["ch_max"] is None else torch.maximum(s["ch_max"], ch)
        return fn

    handles = [m.register_forward_pre_hook(hook(n)) for n, m in conv_layers(model).items()]
    for i in range(0, len(data), batch):
        model(data[i:i + batch])
    for h in handles:
        h.remove()
    return {n: {"minmax": (s["min"], s["max"]),
                "percentile": (float(np.mean(s["plo"])), float(np.mean(s["phi"]))),
                "ch_max": s["ch_max"]} for n, s in stats.items()}


@torch.no_grad()
def quantized_copy(model, ranges, bits, per_channel=True, layers=None, calib="minmax",
                   act=True, weights=True, layer_bits=None):
    """Copy of the model with fake-quantized weights and conv-input activations.
    layers: quantize only these convs (None = all); layer_bits: per-layer bit override."""
    qm = copy.deepcopy(model)
    for name, mod in conv_layers(qm).items():
        if layers is not None and name not in layers:
            continue
        b = (layer_bits or {}).get(name, bits)
        if weights:
            mod.weight.data = fake_quant_weight(mod.weight.data, b, per_channel, out_axis(mod))
        if act:
            lo, hi = ranges[name][calib]
            mod.register_forward_pre_hook(
                lambda m, inp, lo=lo, hi=hi, b=b: (fake_quant_act(inp[0], lo, hi, b),) + inp[1:])
    return qm


@torch.no_grad()
def equalize(model, name_a, name_b):
    """Cross-layer equalization for conv_a -> ReLU -> conv_b (Nagel et al., 2019).
    Channel k: W_a[k] /= s_k, b_a[k] /= s_k, W_b[:, k] *= s_k, s_k = sqrt(r_a[k] / r_b[k]).
    Exact in FP32 because ReLU(z / s) = ReLU(z) / s for s > 0."""
    qm = copy.deepcopy(model)
    mods = dict(qm.named_modules())
    a, b = mods[name_a], mods[name_b]
    s = torch.sqrt(a.weight.abs().amax(dim=(1, 2, 3)) / b.weight.abs().amax(dim=(0, 2, 3)))
    a.weight /= s.view(-1, 1, 1, 1)
    if a.bias is not None:
        a.bias /= s
    b.weight *= s.view(1, -1, 1, 1)
    return qm


# metrics and printing

def compare(ref, out, gt):
    d = (out - ref).abs()
    return {"max": d.max().item(), "mean": d.mean().item(),
            "cos": F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item(),
            "argmax": (out.argmax(1) == ref.argmax(1)).float().mean().item(),
            "miou": miou(out.argmax(1), gt)}


def header(title):
    print(f"\n=== {title} ===")
    print(f"{'':<40}{'max|d|':>8}{'mean|d|':>9}{'cos':>10}{'argmax':>10}{'mIoU':>8}")


def row(tag, r):
    print(f"{tag:<40}{r['max']:>8.3f}{r['mean']:>9.4f}{r['cos']:>10.5f}{r['argmax']:>10.4%}{r['miou']:>8.4f}")


# experiments

def exp_calibration(models, refs, imgs, gt):
    """Which random distribution to calibrate on."""
    header("1. calibration distribution (INT8 / INT16, per-channel weights)")
    for kind in ["uniform", "gauss", "shapes"]:
        data = calibration_set(kind)
        for mname, model in models.items():
            ranges = calibrate(model, data)
            for bits in [16, 8]:
                row(f"{kind:<8} {mname:<10} INT{bits}", compare(refs[mname], quantized_copy(model, ranges, bits)(imgs), gt))

    print("\nconv input max: calibration sets vs real images (optimized model)")
    opt = models["optimized"]
    cols = {k: calibrate(opt, calibration_set(k)) for k in ["uniform", "gauss", "shapes"]}
    cols["real"] = calibrate(opt, imgs)
    print(f"{'layer':<8}" + "".join(f"{k:>10}" for k in cols))
    for name in conv_layers(opt):
        print(f"{name:<8}" + "".join(f"{cols[k][name]['minmax'][1]:>10.2f}" for k in cols))


def exp_main(models, refs, ranges, imgs, gt):
    header("2. FP32 vs INT16 vs INT8 (calibration: shapes)")
    for mname, model in models.items():
        row(f"{mname:<10} FP32", compare(refs[mname], model(imgs), gt))
        for bits in [16, 8]:
            for pc in [False, True]:
                tag = f"{mname:<10} INT{bits} w {'per-channel' if pc else 'per-tensor'}"
                row(tag, compare(refs[mname], quantized_copy(model, ranges[mname], bits, pc)(imgs), gt))


def exp_sensitivity(models, refs, ranges, imgs, gt):
    """Quantize one layer at a time (INT8), the rest stays FP32."""
    results = {}
    for mname, model in models.items():
        print(f"\n=== 3. per-layer INT8 sensitivity: {mname} ===")
        print(f"{'layer':<12}{'act range':>18}{'act: mean|d|':>14}{'argmax':>9}"
              f"{'w per-tensor':>14}{'w per-channel':>15}")
        rows = []
        for name in conv_layers(model):
            r_act = compare(refs[mname], quantized_copy(model, ranges[mname], 8, layers={name}, weights=False)(imgs), gt)
            r_wpt = compare(refs[mname], quantized_copy(model, ranges[mname], 8, False, {name}, act=False)(imgs), gt)
            r_wpc = compare(refs[mname], quantized_copy(model, ranges[mname], 8, True, {name}, act=False)(imgs), gt)
            lo, hi = ranges[mname][name]["minmax"]
            print(f"{name:<12}{f'[{lo:.2f}, {hi:.2f}]':>18}{r_act['mean']:>14.4f}{r_act['argmax']:>9.4f}"
                  f"{r_wpt['mean']:>14.4f}{r_wpc['mean']:>15.4f}")
            rows.append((name, r_act["mean"], r_wpt["mean"], r_wpc["mean"]))
        results[mname] = rows
    plot_sensitivity(results, FIG / "sensitivity.png")


def exp_fixes(models, refs, ranges, imgs, gt):
    header("4. fixes for INT8 (per-channel weights unless noted)")
    pairs = {"original": ("b6", "b7.conv"), "optimized": ("c6", "c7")}
    data = calibration_set("shapes")
    for mname, model in models.items():
        a, b = pairs[mname]
        r = ranges[mname]
        row(f"{mname:<10} baseline", compare(refs[mname], quantized_copy(model, r, 8)(imgs), gt))
        row(f"{mname:<10} percentile calibration",
            compare(refs[mname], quantized_copy(model, r, 8, calib="percentile")(imgs), gt))
        row(f"{mname:<10} {b} in INT16 (mixed)",
            compare(refs[mname], quantized_copy(model, r, 8, layer_bits={b: 16})(imgs), gt))
        eq = equalize(model, a, b)
        print(f"{'':<10} CLE {a}->{b}: FP32 max|d| = {(eq(imgs) - refs[mname]).abs().max().item():.2e}")
        r_eq = calibrate(eq, data)
        row(f"{mname:<10} CLE", compare(refs[mname], quantized_copy(eq, r_eq, 8)(imgs), gt))
        row(f"{mname:<10} CLE, per-tensor weights", compare(refs[mname], quantized_copy(eq, r_eq, 8, False)(imgs), gt))


# figures

def plot_distributions(model, ranges, title, path):
    """Per layer: spread of per-channel max|w| and of per-channel max activation (conv input)."""
    convs = conv_layers(model)
    names = list(convs)
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, values, label in [
        (axes[0], [convs[n].weight.abs().amax(dim=[d for d in range(4) if d != out_axis(convs[n])]) for n in names],
         "per-channel max|w|"),
        (axes[1], [ranges[n]["ch_max"] for n in names], "per-channel max activation (conv input)"),
    ]:
        for i, v in enumerate(values):
            v = v[v > 0].detach().numpy()
            ax.vlines(i, v.min(), v.max(), color="#4a86e8", lw=6, alpha=0.5)
            ax.plot(i, np.median(v), "o", color="#1c4587")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_xticks(x, names, rotation=60, ha="right")
    axes[0].set_title(f"{title}: bar = min..max over channels, dot = median")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def plot_sensitivity(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (mname, rows) in zip(axes, results.items()):
        names = [r[0] for r in rows]
        x = np.arange(len(names))
        ax.bar(x - 0.27, [r[1] for r in rows], 0.27, label="activations INT8")
        ax.bar(x, [r[2] for r in rows], 0.27, label="weights INT8 per-tensor")
        ax.bar(x + 0.27, [r[3] for r in rows], 0.27, label="weights INT8 per-channel")
        ax.set_xticks(x, names, rotation=60, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("mean |diff| of output vs FP32")
        ax.set_title(f"{mname}: one layer quantized at a time")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    torch.manual_seed(0)
    FIG.mkdir(exist_ok=True)
    imgs, gt = load_images()
    net = load_net()
    models = {"original": net, "optimized": NetOpt(net).eval()}

    with torch.no_grad():
        refs = {k: m(imgs) for k, m in models.items()}
        calib_data = calibration_set("shapes")
        ranges = {k: calibrate(m, calib_data) for k, m in models.items()}

        exp_calibration(models, refs, imgs, gt)
        exp_main(models, refs, ranges, imgs, gt)
        for k, m in models.items():
            plot_distributions(m, ranges[k], k, FIG / f"distributions_{k}.png")
        exp_sensitivity(models, refs, ranges, imgs, gt)
        exp_fixes(models, refs, ranges, imgs, gt)

    print(f"\nfigures saved to {FIG}")
