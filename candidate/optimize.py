from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from analyze import load_images, miou
from model import Net


HERE = Path(__file__).parent


def bn_affine(bn):
    """BatchNorm in eval mode is a per-channel affine map: y = k * x + t."""
    k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    t = bn.bias - k * bn.running_mean
    return k, t


def fuse_conv_bn(W, b, bn):
    """conv(W, b) -> bn  ==>  conv(W', b')."""
    k, t = bn_affine(bn)
    if b is None:
        b = torch.zeros(W.shape[0])
    return W * k.view(-1, 1, 1, 1), k * b + t


def make_conv(W, b=None):
    """nn.Conv2d with given weights; padding = k // 2 (3x3 -> 1, 1x1 -> 0)."""
    cout, cin, kh, _ = W.shape
    conv = nn.Conv2d(cin, cout, kh, padding=kh // 2, bias=b is not None)
    conv.weight.data.copy_(W)
    if b is not None:
        conv.bias.data.copy_(b)
    return conv


def fuse_convbn_block(blk):
    """ConvBN block (has .conv and .bn) -> single conv."""
    return make_conv(*fuse_conv_bn(blk.conv.weight, blk.conv.bias, blk.bn))


def fuse_b1(blk):
    """b1 (RepVGG-style): k3(x) + k1(x) + bn(x) -> single 3x3 conv."""
    W3, b3 = fuse_conv_bn(blk.k3.conv.weight, blk.k3.conv.bias, blk.k3.bn)
    W1, b1 = fuse_conv_bn(blk.k1.conv.weight, blk.k1.conv.bias, blk.k1.bn)
    W1 = F.pad(W1, [1, 1, 1, 1]) # 1x1 -> 3x3, weight stays in the center
    c = W3.shape[0]
    Wid = torch.zeros(c, c, 3, 3)
    Wid[torch.arange(c), torch.arange(c), 1, 1] = 1.0 # identity branch as a 3x3 conv
    k, t = bn_affine(blk.bn)
    return make_conv(W3 + W1 + Wid * k.view(-1, 1, 1, 1), b3 + b1 + t)


def fuse_b2(blk):
    """b2: bn(c1(c0(x))), no nonlinearity between c0 and c1 -> single 3x3 conv."""
    W0 = blk.c0.weight # [32, 16, 3, 3]
    C1 = blk.c1.weight[:, :, 0, 0] # 1x1 conv = matrix [32, 32]
    cout, cin, kh, kw = W0.shape
    W = (C1 @ W0.reshape(cout, -1)).reshape(cout, cin, kh, kw)
    return make_conv(*fuse_conv_bn(W, None, blk.bn))


def fuse_b3(blk):
    """b3: (a(x) + b(x)) / 2 -> single 3x3 conv (convolution is linear)."""
    Wa, ba = fuse_conv_bn(blk.a.conv.weight, blk.a.conv.bias, blk.a.bn)
    Wb, bb = fuse_conv_bn(blk.b.conv.weight, blk.b.conv.bias, blk.b.bn)
    return make_conv((Wa + Wb) / 2, (ba + bb) / 2)


def fuse_b10(blk, pi, n_up=128):
    """b10: bn(conv(cat[up, s3[pi]]) * alpha) -> conv(cat[up, s3]).
    n_up: number of channels before the skip part (output of b9)."""
    assert pi.unique().numel() == pi.numel(), "pi has duplicates, not a permutation"
    W = blk.conv.weight * blk.alpha.view(-1, 1, 1, 1) # alpha -> output filters
    W, b = fuse_conv_bn(W, blk.conv.bias, blk.bn)
    W_new = W.clone()
    W_new[:, n_up + pi] = W[:, n_up:] # permute skip input channels
    return make_conv(W_new, b)


def prune_b4_b5(b4, b5):
    """Remove dead output channels of b4 and the matching input channels of b5.
    Dead = zero filter and BN constant <= 0, so the channel is 0 after ReLU."""
    W4, bias4 = fuse_conv_bn(b4.conv.weight, b4.conv.bias, b4.bn)
    zero_filter = b4.conv.weight.flatten(1).abs().amax(dim=1) == 0
    dead = zero_filter & (bias4 <= 0) # for zero filters bias4 is the BN constant
    keep = (~dead).nonzero().flatten()
    W5, bias5 = fuse_conv_bn(b5.conv.weight, b5.conv.bias, b5.bn)
    return make_conv(W4[keep], bias4[keep]), make_conv(W5[:, keep], bias5), dead


def convert_b9(b9):
    """ConvTranspose2d(k=2, s=2) with a kernel constant over 2x2 = conv1x1 followed by nearest x2.
    If the 1x1 part is the identity matrix, only the upsampling remains."""
    w = b9.weight # [Cin, Cout, 2, 2]
    assert torch.equal(w, w[:, :, :1, :1].expand_as(w)), "b9 kernel differs across 2x2 positions"
    M = w[:, :, 0, 0].t() # [Cout, Cin]
    if torch.equal(M, torch.eye(M.shape[0])):
        return nn.Identity()
    return make_conv(M[:, :, None, None]) # b9 has no bias


def fuse_head(b16, b17):
    """b17(b16(x)), two 1x1 convs without ReLU -> single 1x1 conv."""
    A = b17.weight[:, :, 0, 0] # [4, 32]
    B = b16.weight[:, :, 0, 0] # [32, 16]
    return make_conv((A @ B)[:, :, None, None], A @ b16.bias + b17.bias)



# optimized model

class NetOpt(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.register_buffer("m", net.m.clone())
        self.register_buffer("s", net.s.clone())

        # encoder
        self.c0 = fuse_convbn_block(net.b0)
        self.c1 = fuse_b1(net.b1)
        self.c2 = fuse_b2(net.b2)
        self.c3 = fuse_b3(net.b3)
        self.c4, self.c5, _ = prune_b4_b5(net.b4, net.b5)

        # bottleneck (b8 removed: g == 0)
        assert (net.b8.g == 0).all(), "b8.g is not zero, b8 cannot be removed"
        self.c6 = make_conv(net.b6.weight, net.b6.bias)
        self.c7 = fuse_convbn_block(net.b7)

        # decoder
        self.up = convert_b9(net.b9)
        self.c10 = fuse_b10(net.b10, net.pi)
        self.c11 = fuse_convbn_block(net.b11)
        self.c12 = fuse_convbn_block(net.b12)
        self.c13 = fuse_convbn_block(net.b13)
        self.c14 = fuse_convbn_block(net.b14)
        self.c15 = fuse_convbn_block(net.b15)

        # head
        self.head = fuse_head(net.b16, net.b17)

    def forward(self, x):
        x = (x - self.m) / self.s

        # level 1, 128x128
        x = F.relu(self.c0(x))
        s1 = F.relu(self.c1(x))

        # level 2, 64x64
        x = F.max_pool2d(s1, 2)
        x = F.relu(self.c2(x))
        s2 = F.relu(self.c3(x))

        # level 3, 32x32
        x = F.max_pool2d(s2, 2)
        x = F.relu(self.c4(x))
        s3 = F.relu(self.c5(x))

        # bottleneck, 16x16
        x = F.max_pool2d(s3, 2)
        x = F.relu(self.c6(x))
        x = F.relu(self.c7(x))

        # decoder level 3, 32x32
        x = F.interpolate(self.up(x), scale_factor=2.0, mode="nearest")
        x = F.relu(self.c10(torch.cat([x, s3], dim=1)))
        x = F.relu(self.c11(x))

        # decoder level 2, 64x64
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = F.relu(self.c12(torch.cat([x, s2], dim=1)))
        x = F.relu(self.c13(x))

        # decoder level 1, 128x128
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = F.relu(self.c14(torch.cat([x, s1], dim=1)))
        x = F.relu(self.c15(x))

        # head
        return F.log_softmax(self.head(x), dim=1)


# checks

def load_net():
    ref = torch.jit.load(str(HERE / "model.pt")).eval()
    net = Net().eval()
    net.load_state_dict(ref.state_dict())
    return net


def report(name, a, b):
    print(f"{name:<10} max|diff| = {(a - b).abs().max().item():.2e}")


def check_blocks(net):
    print("--- per-block checks (random input) ---")
    for name in ["b0", "b4", "b5", "b7", "b11", "b12", "b13", "b14", "b15"]:
        blk = getattr(net, name)
        x = torch.randn(2, blk.conv.in_channels, 32, 32)
        report(name, blk(x), fuse_convbn_block(blk)(x))

    x = torch.randn(2, 16, 32, 32)
    report("b1", net.b1(x), fuse_b1(net.b1)(x))
    report("b2", net.b2(x), fuse_b2(net.b2)(x))
    x = torch.randn(2, 32, 32, 32)
    report("b3", net.b3(x), fuse_b3(net.b3)(x))

    up, s3 = torch.randn(2, 128, 16, 16), torch.randn(2, 72, 16, 16)
    report("b10", net.b10(torch.cat([up, torch.index_select(s3, 1, net.pi)], dim=1)),
           fuse_b10(net.b10, net.pi)(torch.cat([up, s3], dim=1)))

    conv4, conv5, dead = prune_b4_b5(net.b4, net.b5)
    x = torch.randn(2, 32, 32, 32)
    h = F.relu(net.b4(x))
    print(f"dead channels in b4: {dead.nonzero().flatten().tolist()}, max value after ReLU: {h[:, dead].abs().max().item()}")
    report("b4+b5", net.b5(h), conv5(F.relu(conv4(x))))

    x = torch.randn(2, 128, 16, 16)
    print(f"b9 replaced by: {type(convert_b9(net.b9)).__name__}")
    report("b9", net.b9(x), F.interpolate(convert_b9(net.b9)(x), scale_factor=2.0, mode="nearest"))

    x = torch.randn(2, 16, 32, 32)
    report("b16+b17", net.b17(net.b16(x)), fuse_head(net.b16, net.b17)(x))



def compare_models(net, opt):
    print("\n--- Net vs NetOpt ---")
    imgs, gt = load_images()
    for name, x in [("images", imgs), ("random", torch.rand(8, 3, 128, 128))]:
        a, b = net(x), opt(x)
        agree = (a.argmax(1) == b.argmax(1)).float().mean().item()
        print(f"{name:<10} max|diff| = {(a - b).abs().max().item():.2e}, argmax match = {agree:.4%}")
    print(f"mIoU: Net = {miou(net(imgs).argmax(1), gt):.4f}, NetOpt = {miou(opt(imgs).argmax(1), gt):.4f}")
    n_params = lambda model: sum(p.numel() for p in model.parameters())
    print(f"params: Net = {n_params(net):,}, NetOpt = {n_params(opt):,}")


if __name__ == "__main__":
    torch.manual_seed(0)
    net = load_net()
    with torch.no_grad():
        opt = NetOpt(net).eval()
        check_blocks(net)
        compare_models(net, opt)