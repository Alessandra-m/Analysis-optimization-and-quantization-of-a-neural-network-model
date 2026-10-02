import copy

import torch
import torch.nn.functional as F

from analyze import load_images, miou
from optimize import NetOpt, load_net

def logits_of(opt, x):
    """Output of NetOpt head before log_softmax."""
    feats = {}
    h = opt.head.register_forward_hook(lambda m, i, o: feats.setdefault("z", o))
    opt(x)
    h.remove()
    return feats["z"]


def fold_input_norm(opt):
    """Copy of NetOpt with (x - m) / s folded into c0.
    conv((x - m) / s) = conv'(x): W'[o, i] = W[o, i] / s[i], b' = b - sum W[o, i] * m[i] / s[i]."""
    folded = copy.deepcopy(opt)
    W, b = opt.c0.weight, opt.c0.bias
    m, s = opt.m.view(1, -1, 1, 1), opt.s.view(1, -1, 1, 1)
    folded.c0.weight.data = W / s
    folded.c0.bias.data = b - (W * m / s).sum(dim=(1, 2, 3))
    folded.m.zero_() # forward now computes (x - 0) / 1 = x
    folded.s.fill_(1.0)
    return folded


def border_mask(h, w, width):
    """Boolean mask of a frame of given width around the image."""
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[:width, :] = mask[-width:, :] = True
    mask[:, :width] = mask[:, -width:] = True
    return mask


def check_log_softmax(opt, imgs):
    print("1. log(softmax(z)) vs log_softmax(z)")
    z = logits_of(opt, imgs)
    a, b = torch.log(torch.softmax(z, 1)), F.log_softmax(z, 1)
    print(f"images: max|diff| = {(a - b).abs().max().item():.2e}, "
          f"min log-prob = {b.min().item():.2f}, -inf in log(softmax): {torch.isinf(a).sum().item()}")
    print("logits scaled by k (a more confident model):")
    for k in [1, 5, 10, 20, 50]:
        a, b = torch.log(torch.softmax(z * k, 1)), F.log_softmax(z * k, 1)
        finite = ~torch.isinf(a)
        print(f"  k = {k:>2}:  -inf in log(softmax): {int((~finite).sum()):>7} of {a.numel()}   "
              f"max|diff| on finite = {(a - b)[finite].abs().max().item():.2e}   "
              f"min log_softmax = {b.min().item():9.1f}")


def check_input_norm_fold(opt, imgs, gt):
    print("\n2. (x - m) / s folded into c0")
    folded = fold_input_norm(opt)
    border = border_mask(128, 128, 1)

    # first conv output: where exactly the difference appears
    c0_ref = opt.c0((imgs - opt.m) / opt.s)
    c0_fold = folded.c0(imgs)
    d = (c0_ref - c0_fold).abs().amax(dim=(0, 1))       # [H, W], max over images and channels
    print(f"c0 output: max|diff| on 1px border = {d[border].max().item():.2e}, "
          f"inside = {d[~border].max().item():.2e}")

    # network output: the border error spreads inward through the following convs
    y_ref, y_fold = opt(imgs), folded(imgs)
    d = (y_ref - y_fold).abs().amax(dim=(0, 1))
    for width in [1, 4, 8, 16]:
        bm = border_mask(128, 128, width)
        print(f"net output: max|diff| in {width:>2}px border = {d[bm].max().item():.2e}, "
              f"outside = {d[~bm].max().item():.2e}")
    agree = (y_ref.argmax(1) == y_fold.argmax(1)).float().mean().item()
    print(f"argmax match: {agree:.4%}, mIoU: original = {miou(y_ref.argmax(1), gt):.4f}, "
          f"folded = {miou(y_fold.argmax(1), gt):.4f}")

    # exact variant: pad the raw input with m (normalized zero) and convolve without padding
    x_pad = F.pad(imgs - opt.m, [1, 1, 1, 1]) + opt.m
    c0_exact = F.conv2d(x_pad, folded.c0.weight, folded.c0.bias)
    print(f"exact variant (pad with m): max|diff| of c0 output = {(c0_ref - c0_exact).abs().max().item():.2e}")


if __name__ == "__main__":
    torch.manual_seed(0)
    net = load_net()
    opt = NetOpt(net).eval()
    imgs, gt = load_images()
    with torch.no_grad():
        check_log_softmax(opt, imgs)
        check_input_norm_fold(opt, imgs, gt)