from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent

class ConvBN(nn.Module):
    """Original class U: bn(conv(x))."""
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class RepBlock(nn.Module):
    """Original class V (b1): k3(x) + k1(x) + bn(x)."""
    def __init__(self, c):
        super().__init__()
        self.k3 = ConvBN(c, c, 3)
        self.k1 = ConvBN(c, c, 1)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        return self.k3(x) + self.k1(x) + self.bn(x)


class DoubleConvBN(nn.Module):
    """Original class W (b2): bn(c1(c0(x)))."""
    def __init__(self, cin, cout):
        super().__init__()
        self.c0 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.c1 = nn.Conv2d(cout, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.c1(self.c0(x)))


class AvgBranch(nn.Module):
    """Original class X (b3): (a(x) + b(x)) * 0.5."""
    def __init__(self, c):
        super().__init__()
        self.a = ConvBN(c, c)
        self.b = ConvBN(c, c)

    def forward(self, x):
        return (self.a(x) + self.b(x)) * 0.5


class GatedResidual(nn.Module):
    """Original class Y (b8): drop(x + g * f1(relu(f0(x))))."""
    def __init__(self, c):
        super().__init__()
        self.f0 = nn.Conv2d(c, c, 3, padding=1)
        self.f1 = nn.Conv2d(c, c, 3, padding=1)
        self.register_buffer("g", torch.zeros(1, c, 1, 1))
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        r = self.f1(F.relu(self.f0(x)))
        return self.drop(x + self.g * r)


class ScaledConvBN(nn.Module):
    """Original class Z (b10): bn(conv(x) * alpha)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.register_buffer("alpha", torch.ones(1, cout, 1, 1))
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x) * self.alpha)


def pool_p(x):
    """Original function _p: 2x2 max pooling via reshape + amax."""
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).amax(dim=(3, 5))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("m", torch.zeros(1, 3, 1, 1))
        self.register_buffer("s", torch.ones(1, 3, 1, 1))
        self.register_buffer("pi", torch.arange(72))

        # encoder
        self.b0 = ConvBN(3, 16)
        self.b1 = RepBlock(16)
        self.b2 = DoubleConvBN(16, 32)
        self.b3 = AvgBranch(32)
        self.b4 = ConvBN(32, 80)
        self.b5 = ConvBN(80, 72)

        # bottleneck
        self.b6 = nn.Conv2d(72, 128, 3, padding=1)
        self.b7 = ConvBN(128, 128)
        self.b8 = GatedResidual(128)

        # decoder
        self.b9 = nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False)
        self.b10 = ScaledConvBN(200, 64)
        self.b11 = ConvBN(64, 64)
        self.b12 = ConvBN(96, 32)
        self.b13 = ConvBN(32, 32)
        self.b14 = ConvBN(48, 16)
        self.b15 = ConvBN(16, 16)

        # head
        self.b16 = nn.Conv2d(16, 32, 1)
        self.b17 = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        x = (x - self.m) / self.s

        # level 1, 128x128
        x = self.b0(x)
        x = (x + x.abs()) * 0.5                                   # ReLU
        s1 = F.leaky_relu(self.b1(x), 0.0)                        # ReLU

        # level 2, 64x64
        x = pool_p(s1)
        x = self.b2(x)
        x = torch.where(x > 0, x, torch.zeros_like(x))            # ReLU
        x = self.b3(x)
        s2 = torch.maximum(x, torch.zeros_like(x))                # ReLU

        # level 3, 32x32
        x = F.max_pool2d(s2, 2)
        x = F.relu(self.b4(x))
        s3 = torch.clamp(self.b5(x), min=0.0)                     # ReLU

        # bottleneck, 16x16
        x = F.max_pool2d(s3, 2)
        x = F.relu(self.b6(x))
        x = F.relu(self.b7(x))
        x = self.b8(x)

        # decoder level 3, 32x32
        x = self.b9(x)
        x = torch.cat([x, torch.index_select(s3, 1, self.pi)], dim=1)
        x = F.relu(self.b10(x))
        x = F.relu(self.b11(x))

        # decoder level 2, 64x64
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = F.relu(self.b12(torch.cat([x, s2], dim=1)))
        x = F.relu(self.b13(x))

        # decoder level 1, 128x128
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = F.relu(self.b14(torch.cat([x, s1], dim=1)))
        x = F.relu(self.b15(x))

        # head
        x = self.b17(self.b16(x))
        return torch.log(torch.softmax(x, dim=1))


if __name__ == "__main__":
    torch.manual_seed(0)
    ref = torch.jit.load(str(HERE / "model.pt")).eval()
    net = Net().eval()
    net.load_state_dict(ref.state_dict())

    with torch.no_grad():
        x = torch.rand(4, 3, 128, 128)
        out_ref, out_net = ref(x), net(x)

    print("max |diff|:", (out_net - out_ref).abs().max().item())
    print("argmax match:", (out_net.argmax(1) == out_ref.argmax(1)).float().mean().item())
    print("params:", sum(p.numel() for p in net.parameters()))