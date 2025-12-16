# filename: examples/add.py

from __future__ import annotations

import argparse
import os
import time

import eas
from eas import mk
from eas.runtime import get_runtime

try:
    import torch

    TORCH_AVAILABLE = True
    TORCH_MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_MPS_AVAILABLE = False


@eas.kernel
def add_kernel(a, b, c, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N

    a_val = mk.load(a, offs, mask)
    b_val = mk.load(b, offs, mask)
    mk.store(c, offs, a_val + b_val, mask)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="easy_as 元素级加法演示")
    parser.add_argument("--n", type=int, default=1024 + 7, help="元素数量")
    parser.add_argument(
        "--block", type=int, default=256, help="BLOCK 大小（线程组大小）"
    )
    parser.add_argument("--iters", type=int, default=1, help="计时的迭代次数")
    parser.add_argument("--print-msl", action="store_true", help="打印生成的 MSL")
    args = parser.parse_args(argv)

    if not TORCH_AVAILABLE:
        raise SystemExit(
            "torch is not installed; please install torch to run this example"
        )

    backend_env = os.environ.get("EAS_BACKEND", "auto")
    try:
        runtime = get_runtime()
    except Exception as e:
        raise SystemExit(
            f"failed to initialize runtime (EAS_BACKEND={backend_env!r}): {e}"
        )
    runtime_name = runtime.__class__.__name__
    print(f"backend={backend_env} -> {runtime_name}")

    n = int(args.n)
    block = int(args.block)
    iters = int(args.iters)
    if n <= 0 or block <= 0 or iters <= 0:
        raise SystemExit("--n/--block/--iters must be > 0")

    device = (
        torch.device("mps")
        if (runtime_name == "MetalRuntime" and TORCH_MPS_AVAILABLE)
        else torch.device("cpu")
    )
    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    c = torch.empty_like(a)

    # 正确性验证（包括首次编译）
    add_kernel(a, b, c, n, BLOCK=block)
    if device.type == "mps":
        torch.mps.synchronize()
    expected = a + b
    if hasattr(torch.testing, "assert_close"):
        torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-6)
    else:
        if not torch.allclose(c, expected, rtol=1e-5, atol=1e-6):
            raise AssertionError("torch.allclose failed")
    print(f"OK（正确性验证，设备={device}）")

    # 快速计时（尽力而为）
    if hasattr(runtime, "synchronize"):
        runtime.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    eas_ms: float
    eas_mode: str
    if runtime_name == "MetalRuntime" and callable(getattr(runtime, "benchmark", None)):
        ck = add_kernel.compile(a, b, c, n, BLOCK=block)
        warmup = min(10, iters)
        t = runtime.benchmark(
            ck,
            {"a": a, "b": b, "c": c, "N": n},
            {"BLOCK": block},
            repeat=iters,
            warmup=warmup,
            torch_mps_sync=False,
        )
        eas_ms = t * 1e3
        eas_mode = "single-cb repeat"
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            add_kernel(a, b, c, n, BLOCK=block, _sync=False)
        if hasattr(runtime, "synchronize"):
            runtime.synchronize()
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        eas_ms = (t1 - t0) * 1e3 / iters
        eas_mode = "loop"

    print(f"time: {eas_ms:.3f} ms/iter (iters={iters}, mode={eas_mode})")

    # torch 计时（CPU 或 MPS；尽力而为）
    warmup = min(20, iters)
    for _ in range(warmup):
        torch.add(a, b, out=c)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.add(a, b, out=c)
    if device.type == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    torch_ms = (t1 - t0) * 1e3 / iters
    ratio = eas_ms / torch_ms if torch_ms > 0 else float("inf")
    print(
        f"torch ({device.type}): {torch_ms:.3f} ms/iter (iters={iters}, warmup={warmup})"
    )
    print(f"ratio eas/torch: {ratio:.2f}x")

    if args.print_msl:
        print(add_kernel.to_msl(a, b, c, n, BLOCK=block))


if __name__ == "__main__":
    main()
