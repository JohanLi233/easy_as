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
def matmul_kernel(a, b, c, N, BLOCK: eas.constexpr, K: eas.constexpr):
    """
    Naive matmul: C[M, BLOCK] = A[M, K] @ B[K, BLOCK]

    MVP 限制：仅支持 1D grid + 1D threadgroup，因此这里把输出列数固定为 BLOCK，
    并把 grid sizing 的 `N` 设为 `M * BLOCK`（这样 grid.x == M）。
    """
    pid = mk.program_id(0)  # row index
    col = mk.arange(0, BLOCK)  # thread id within row
    out = pid * BLOCK + col
    mask = out < N

    acc = 0.0
    for k in range(K):
        a_off = pid * K + k
        b_off = k * BLOCK + col
        acc = acc + mk.load(a, a_off, mask) * mk.load(b, b_off, mask)
    mk.store(c, out, acc, mask)


def _torch_device_for_runtime(runtime_name: str) -> "torch.device":
    return (
        torch.device("mps")
        if (runtime_name == "MetalRuntime" and TORCH_MPS_AVAILABLE)
        else torch.device("cpu")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="easy_as 朴素矩阵乘演示（matmul）")
    parser.add_argument("--m", type=int, default=128, help="M（行数）")
    parser.add_argument("--k", type=int, default=64, help="K（constexpr，展开）")
    parser.add_argument(
        "--block",
        type=int,
        default=256,
        help="BLOCK（线程组大小；也是输出列数 N）",
    )
    parser.add_argument("--iters", type=int, default=1, help="计时的迭代次数")
    parser.add_argument("--print-msl", action="store_true", help="打印生成的 MSL")
    args = parser.parse_args(argv)

    if not TORCH_AVAILABLE:
        raise SystemExit(
            "torch is not installed; please install torch to run this example"
        )

    backend_env = os.environ.get("EAS_BACKEND", "auto")
    runtime = get_runtime()
    runtime_name = runtime.__class__.__name__
    print(f"backend={backend_env} -> {runtime_name}")

    m = int(args.m)
    k = int(args.k)
    block = int(args.block)
    iters = int(args.iters)
    if m <= 0 or k <= 0 or block <= 0 or iters <= 0:
        raise SystemExit("--m/--k/--block/--iters must be > 0")

    n_total = m * block

    device = _torch_device_for_runtime(runtime_name)
    a2 = torch.randn((m, k), dtype=torch.float32, device=device)
    b2 = torch.randn((k, block), dtype=torch.float32, device=device)
    c2 = torch.empty((m, block), dtype=torch.float32, device=device)

    a = a2.reshape(-1)
    b = b2.reshape(-1)
    c = c2.reshape(-1)

    matmul_kernel(a, b, c, n_total, BLOCK=block, K=k)
    if device.type == "mps":
        torch.mps.synchronize()
    expected = a2 @ b2
    if hasattr(torch.testing, "assert_close"):
        torch.testing.assert_close(c2, expected, rtol=1e-4, atol=1e-5)
    else:
        if not torch.allclose(c2, expected, rtol=1e-4, atol=1e-5):
            raise AssertionError("torch.allclose failed")
    print(f"OK（正确性验证，设备={device}，形状=({m},{k})x({k},{block}))")

    if hasattr(runtime, "synchronize"):
        runtime.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    eas_ms: float
    eas_mode: str
    if runtime_name == "MetalRuntime" and callable(getattr(runtime, "benchmark", None)):
        ck = matmul_kernel.compile(a, b, c, n_total, BLOCK=block, K=k)
        warmup = min(5, iters)
        t = runtime.benchmark(
            ck,
            {"a": a, "b": b, "c": c, "N": n_total},
            {"BLOCK": block, "K": k},
            repeat=iters,
            warmup=warmup,
            torch_mps_sync=False,
        )
        eas_ms = t * 1e3
        eas_mode = "single-cb repeat"
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            matmul_kernel(a, b, c, n_total, BLOCK=block, K=k, _sync=False)
        if hasattr(runtime, "synchronize"):
            runtime.synchronize()
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        eas_ms = (t1 - t0) * 1e3 / iters
        eas_mode = "loop"

    print(f"time: {eas_ms:.3f} ms/iter (iters={iters}, mode={eas_mode})")

    warmup = min(20, iters)
    for _ in range(warmup):
        _ = a2 @ b2
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a2 @ b2
    if device.type == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    torch_ms = (t1 - t0) * 1e3 / iters
    ratio = eas_ms / torch_ms if torch_ms > 0 else float("inf")
    print(f"torch ({device.type}): {torch_ms:.3f} ms/iter (iters={iters})")
    print(f"ratio eas/torch: {ratio:.2f}x")

    if args.print_msl:
        print(matmul_kernel.to_msl(a, b, c, n_total, BLOCK=block, K=k))


if __name__ == "__main__":
    main()
