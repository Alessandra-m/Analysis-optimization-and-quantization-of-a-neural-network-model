import os
import platform
import statistics
import time
from pathlib import Path

import torch

from analyze import layer_table, load_images
from optimize import NetOpt, load_net


HERE = Path(__file__).parent


def bench(model, x, warmup=10, runs=100):
    """Time of one forward pass in ms: median and 90th percentile."""
    with torch.no_grad():
        for _ in range(warmup):                         
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return statistics.median(times), times[int(0.9 * len(times))]


def cpu_name():
    """CPU model name (Linux), falls back to platform info."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    torch.manual_seed(0)
    ref = torch.jit.load(str(HERE / "model.pt")).eval()
    net = load_net()
    opt = NetOpt(net).eval()
    default_threads = torch.get_num_threads()

    print(f"CPU: {cpu_name()}, logical cores: {os.cpu_count()}, torch threads (default): {default_threads}")
    print(f"PyTorch {torch.__version__}, Python {platform.python_version()}\n")

    # ---- parameters, MACs, output difference ----
    imgs, _ = load_images()
    with torch.no_grad():
        out_ref = ref(imgs)
        diff_net = (net(imgs) - out_ref).abs().max().item()
        diff_opt = (opt(imgs) - out_ref).abs().max().item()

    print(f"{'model':<10}{'params':>12}{'MACs':>16}{'max|diff| vs model.pt':>24}")
    for name, m, diff in [("Net", net, diff_net), ("NetOpt", opt, diff_opt)]:
        params = sum(p.numel() for p in m.parameters())
        macs = sum(row[5] for row in layer_table(m))
        print(f"{name:<10}{params:>12,}{macs:>16,}{diff:>24.2e}")

    # ---- inference time ----
    for batch in [1, 8]:
        x = torch.rand(batch, 3, 128, 128)
        for threads in sorted({1, default_threads}):
            torch.set_num_threads(threads)
            print(f"\n--- batch {batch}, threads {threads} ---")
            base = None
            for name, m in [("model.pt", ref), ("Net", net), ("NetOpt", opt)]:
                med, p90 = bench(m, x)
                base = base or med
                print(f"{name:<10} median {med:7.2f} ms   p90 {p90:7.2f} ms   speedup x{base / med:.2f}")
    torch.set_num_threads(default_threads)