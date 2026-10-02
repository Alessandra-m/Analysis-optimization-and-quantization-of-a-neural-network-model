from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from model import Net

HERE = Path(__file__).parent

#  data and metric

def load_images():
    """Images as float32 [N, 3, H, W] in [0, 1] and masks as int64 [N, H, W]."""
    imgs, masks = [], []
    for p in sorted((HERE / "images").glob("*.png")):
        img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(img).permute(2, 0, 1))
        masks.append(torch.from_numpy(np.array(Image.open(HERE / "masks" / p.name))).long())
    return torch.stack(imgs), torch.stack(masks)


def miou(pred, gt, num_classes=4):
    """Mean IoU over classes present in prediction or ground truth."""
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (gt == c)).sum().item()
        union = ((pred == c) | (gt == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / len(ious)


# layer table

def macs_of(mod, inp, out):
    """Conv2d: H_out * W_out * C_out * C_in * k * k; ConvTranspose2d: H_in * W_in * C_in * C_out * k * k."""
    if isinstance(mod, nn.Conv2d):
        cout, cin, kh, kw = mod.weight.shape
        return out[0].numel() * cin * kh * kw
    if isinstance(mod, nn.ConvTranspose2d):
        cin, cout, kh, kw = mod.weight.shape
        return inp[0].numel() * cout * kh * kw
    return 0


def layer_table(model, hw=128):
    """One forward pass with hooks. Rows: (name, type, input shape, output shape, params, MACs)."""
    rows = []

    def make_hook(name):
        def fn(mod, inputs, out):
            inp = inputs[0]
            params = sum(p.numel() for p in mod.parameters(recurse=False))
            rows.append((name, type(mod).__name__, tuple(inp.shape[1:]), tuple(out.shape[1:]),
                         params, macs_of(mod, inp, out)))
        return fn

    handles = [mod.register_forward_hook(make_hook(name))
               for name, mod in model.named_modules()
               if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d))]
    with torch.no_grad():
        model(torch.rand(1, 3, hw, hw))
    for h in handles:
        h.remove()
    return rows


def print_table(model, rows):
    print(f"\n{'layer':<12}{'type':<17}{'input':<16}{'output':<16}{'params':>10}{'MACs':>13}")
    for r in rows:
        print(f"{r[0]:<12}{r[1]:<17}{str(r[2]):<16}{str(r[3]):<16}{r[4]:>10,}{r[5]:>13,}")
    print(f"\ntotal params: {sum(p.numel() for p in model.parameters()):,}"
          f"   (in table: {sum(r[4] for r in rows):,})")
    print(f"total MACs: {sum(r[5] for r in rows):,}")


if __name__ == "__main__":
    torch.manual_seed(0)
    ref = torch.jit.load(str(HERE / "model.pt")).eval()
    net = Net().eval()
    net.load_state_dict(ref.state_dict())

    x, gt = load_images()
    with torch.no_grad():
        y_ref, y_net = ref(x), net(x)
    print("max|diff| on images:", (y_ref - y_net).abs().max().item())
    print("mIoU:", round(miou(y_net.argmax(1), gt), 4))

    print("\n--- Net (original) ---")
    print_table(net, layer_table(net))

    from optimize import NetOpt                        
    opt = NetOpt(net).eval()
    print("\n--- NetOpt (optimized) ---")
    print_table(opt, layer_table(opt))