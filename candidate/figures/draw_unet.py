from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

W, H, STEP = 1.9, 1.25, 2.4
LEVEL_Y = {128: 9.0, 64: 6.6, 32: 4.2, 16: 1.8}
COLORS = {"io": "#e6e6e6", "plain": "#cfe2f3", "merge": "#fce5cd", "remove": "#f4cccc"}

# (name, label, shape, level, x, kind, int8_risk)
BLOCKS = [
    ("in", "input\nRGB, (x-m)/s", "3×128×128", 128, 0.0, "io", False),
    ("b0", "b0\nConvBN 3→16", "16×128×128", 128, 1, "plain", False),
    ("b1", "b1  3 ветки\n3x3 + 1x1 + BN", "16×128×128", 128, 2, "merge", False),
    ("b2", "b2\n3x3 → 1x1 → BN", "32×64×64", 64, 2, "merge", False),
    ("b3", "b3  среднее\n2 свёрток", "32×64×64", 64, 3, "merge", False),
    ("b4", "b4 ConvBN 32→80\n16 мёртвых кан.", "80×32×32", 32, 3, "merge", False),
    ("b5", "b5\nConvBN 80→72", "72×32×32", 32, 4, "plain", False),
    ("b6", "b6\nConv 72→128", "128×16×16", 16, 4, "plain", True),
    ("b7", "b7\nConvBN 128→128", "128×16×16", 16, 5, "plain", True),
    ("b8", "b8 residual\ng = 0", "128×16×16", 16, 6, "remove", False),
    ("b9", "b9 ConvT 2x2\n= nearest ×2", "128×32×32", 16, 7, "remove", False),
    ("b10", "b10 conv·α+BN\n200→64", "64×32×32", 32, 7, "merge", True),
    ("b11", "b11\nConvBN 64→64", "64×32×32", 32, 8, "plain", False),
    ("b12", "b12\nConvBN 96→32", "32×64×64", 64, 8, "plain", False),
    ("b13", "b13\nConvBN 32→32", "32×64×64", 64, 9, "plain", False),
    ("b14", "b14\nConvBN 48→16", "16×128×128", 128, 9, "plain", False),
    ("b15", "b15\nConvBN 16→16", "16×128×128", 128, 10, "plain", False),
    ("head", "b16 → b17\n1x1 16→32→4", "4×128×128", 128, 11, "merge", False),
    ("out", "output\nlog_softmax", "4×128×128", 128, 12, "io", False),
]

fig, ax = plt.subplots(figsize=(22, 9.5))
pos = {}
for name, label, shape, level, x, kind, risk in BLOCKS:
    cx, cy = x * STEP, LEVEL_Y[level]
    pos[name] = (cx, cy)
    ax.add_patch(FancyBboxPatch((cx - W / 2, cy - H / 2), W, H, boxstyle="round,pad=0.03",
                                fc=COLORS[kind], ec="#7030a0" if risk else "#555555",
                                lw=3 if risk else 1.2))
    ax.text(cx, cy + 0.12, label, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(cx, cy - 0.45, shape, ha="center", va="center", fontsize=8.5, color="#333333")


def arrow(a, b, text="", color="#333333", style="-|>", ls="-", text_dx=0.0, text_dy=0.0):
    (x0, y0), (x1, y1) = pos[a], pos[b]
    if y0 == y1:                                          # same level: right edge -> left edge
        p0, p1 = (x0 + W / 2, y0), (x1 - W / 2, y1)
    elif y0 > y1:                                         # down: bottom -> top
        p0, p1 = (x0, y0 - H / 2), (x1, y1 + H / 2)
    else:                                                 # up: top -> bottom
        p0, p1 = (x0, y0 + H / 2), (x1, y1 - H / 2)
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle=style, color=color, lw=1.6, ls=ls))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + text_dx, (p0[1] + p1[1]) / 2 + text_dy, text,
                ha="center", va="center", fontsize=9, color=color,
                bbox=dict(fc="white", ec="none", pad=1))


# main path
for a, b in [("in", "b0"), ("b0", "b1"), ("b2", "b3"), ("b4", "b5"), ("b6", "b7"), ("b7", "b8"),
             ("b8", "b9"), ("b10", "b11"), ("b12", "b13"), ("b14", "b15"), ("b15", "head"), ("head", "out")]:
    arrow(a, b)
for a, b in [("b1", "b2"), ("b3", "b4"), ("b5", "b6")]:
    arrow(a, b, "max-pool 2x2", color="#c00000", text_dx=0.75)
arrow("b9", "b10", "×2", color="#38761d", text_dx=0.35)
for a, b in [("b11", "b12"), ("b13", "b14")]:
    arrow(a, b, "nearest ×2", color="#38761d", text_dx=0.65)

# skip connections
arrow("b1", "b14", "skip s1 (16 кан.)  →  cat: 32 + 16 = 48", color="#7f7f7f", ls="--", text_dy=0.25)
arrow("b3", "b12", "skip s2 (32 кан.)  →  cat: 64 + 32 = 96", color="#7f7f7f", ls="--", text_dy=0.25)
arrow("b5", "b10", "skip s3[:, pi] (72 кан.)\ncat: 128 + 72 = 200",
      color="#7f7f7f", ls="--", text_dy=0.45)

for level, y in LEVEL_Y.items():
    ax.text(-1.6, y, f"{level}×{level}", ha="right", va="center", fontsize=12, fontweight="bold", color="#555555")

ax.legend(handles=[
    Patch(fc=COLORS["plain"], ec="#555555", label="обычный блок (BN вшивается в свёртку)"),
    Patch(fc=COLORS["merge"], ec="#555555", label="упрощается: ветки/свёртки сливаются, каналы удаляются"),
    Patch(fc=COLORS["remove"], ec="#555555", label="удаляется или заменяется (b8: g = 0; b9: nearest ×2)"),
    Patch(fc="white", ec="#7030a0", lw=3, label="компенсация масштабов каналов: риск для INT8"),
], loc="lower right", fontsize=10, frameon=True)

ax.set_xlim(-2.6, 12 * STEP + 1.3)
ax.set_ylim(0.6, 10.0)
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
out = Path(__file__).parent / "unet.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
