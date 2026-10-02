import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).parent

model = torch.jit.load(str(HERE / "model.pt")).eval()
sd = model.state_dict()


def section(title):
    print(f"\n--- {title} ---")


def audit_all_weights():
    """Same statistics for every tensor: trained weights should not look 'clean'."""
    section("weight audit (all tensors)")
    print(f"{'tensor':24s} {'zeros':>14s} {'unique':>8s} {'dead ch':>8s} {'ch spread':>10s}")
    for name, t in sd.items():
        if t.dtype != torch.float32 or t.ndim == 0:
            continue
        zeros = f"{int((t == 0).sum())}/{t.numel()}"
        line = f"{name:24s} {zeros:>14s} {t.unique().numel():>8d}"
        if t.ndim == 4 and name.endswith("weight"):   # conv kernels only, not (1, C, 1, 1) buffers
            ch = t.abs().amax(dim=(1, 2, 3))
            alive = ch[ch > 0]
            spread = (alive.max() / alive.min()).item() if alive.numel() else float("nan")
            line += f" {int((ch == 0).sum()):>8d} {spread:>9.1f}x"
        print(line)


def check_dead_channels():
    """Zero filter + BN constant <= 0 -> channel is always 0 after ReLU."""
    section("dead channels (conv -> BN -> ReLU)")
    for key in sd:
        if not key.endswith(".conv.weight") or key.replace("conv.weight", "bn.weight") not in sd:
            continue
        prefix = key[: -len(".conv.weight")]
        zero = (sd[key].abs().amax(dim=(1, 2, 3)) == 0).nonzero().flatten()
        if zero.numel() == 0:
            print(f"{prefix:8s} zero filters: 0")
            continue
        g, b, mu, var = (sd[f"{prefix}.bn.{k}"] for k in ("weight", "bias", "running_mean", "running_var"))
        const = (b - g * mu / torch.sqrt(var + 1e-5))[zero]
        print(f"{prefix:8s} zero filters: {zero.tolist()}")
        print(f"{'':8s} BN constant: max = {const.max().item():.4f} -> "
              f"{'dead, removable' if (const <= 0).all() else 'constant, NOT removable'}")


def check_bn_gamma(tol=1e-3):
    section(f"BN channels with |gamma| < {tol}")
    for key in sd:
        if key.endswith("bn.weight"):
            print(f"{key:20s} {int((sd[key].abs() < tol).sum())}")


def check_input_normalization():
    section("input normalization (x - m) / s")
    print("m =", sd["m"].flatten().tolist())
    print("s =", sd["s"].flatten().tolist())
    print("foldable into b0; exact only if raw input is padded with m")


def check_permutation():
    section("pi (index_select before b10)")
    pi = sd["pi"]
    is_perm = torch.equal(pi.sort().values, torch.arange(pi.numel()))
    print("pi =", pi.tolist())
    print(f"unique: {pi.unique().numel()}/{pi.numel()}, permutation: {is_perm}")
    if is_perm:
        print("foldable into b10 input-channel order")


def check_residual_gate():
    section("b8: drop(x + g * f1(relu(f0(x))))")
    g = sd["b8.g"]
    print(f"g nonzero: {int(torch.count_nonzero(g))}/{g.numel()}")
    print(f"dropout: {model.b8.drop.original_name}, p = {model.b8.drop.p} (identity in eval)")
    if torch.count_nonzero(g) == 0:
        print("output equals input, block removable")


def check_alpha():
    section("b10.alpha (per-channel scale before BN)")
    a = sd["b10.alpha"].flatten()
    print(f"min = {a.min().item():.4f}, max = {a.max().item():.4f}, negative: {int((a < 0).sum())}/{a.numel()}")
    print("foldable into b10 conv weights")


def check_transposed_conv():
    section("b9: ConvTranspose2d k=2 s=2")
    w = sd["b9.weight"]                                  
    same_2x2 = torch.equal(w, w[..., :1, :1].expand_as(w))
    identity = torch.equal(w[:, :, 0, 0], torch.eye(w.shape[0]))
    print(f"unique values: {w.unique().tolist()}")
    print(f"constant over 2x2: {same_2x2}, identity across channels: {identity}")
    if same_2x2 and identity:
        print("equivalent to nearest upsampling x2")


def check_scale_compensation():
    section("b6 -> b7 per-channel scale")
    out_scale = sd["b6.weight"].abs().amax(dim=(1, 2, 3))      # b6 output channels
    in_scale = sd["b7.conv.weight"].abs().amax(dim=(0, 2, 3))  # b7 input channels
    corr = torch.corrcoef(torch.stack([out_scale.log(), in_scale.log()]))[0, 1].item()
    print(f"b6 out-channel spread: {(out_scale.max() / out_scale.min()).item():.1f}x")
    print(f"corr(log b6 out, log b7 in) = {corr:.3f}")
    print("same output in FP32, bad for per-tensor INT8 activations")


section("forward")
print(model.code)

section("weights")
for name, t in sd.items():
    print(name, tuple(t.shape))

section("submodule code")
for name in ["b0", "b1", "b2", "b3", "b8", "b9", "b10", "b16"]:
    print(f"----- {name}\n{getattr(model, name).code}")

section("_p (archive/code/__torch__/net.py)")
src = zipfile.ZipFile(HERE / "model.pt").read("archive/code/__torch__/net.py").decode()
start = src.index("def _p(")
end = src.find("\ndef ", start + 1)
print(src[start:end if end != -1 else None].rstrip())


audit_all_weights()
check_dead_channels()
check_bn_gamma()
check_input_normalization()
check_permutation()
check_residual_gate()
check_alpha()
check_transposed_conv()
check_scale_compensation()