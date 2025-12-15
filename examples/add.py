from __future__ import annotations

import argparse
import os
import time

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@eas.kernel
def add_kernel(a, b, c, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N

    a_val = mk.load(a, offs, mask)
    b_val = mk.load(b, offs, mask)
    mk.store(c, offs, a_val + b_val, mask)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="easy_as elementwise add demo")
    parser.add_argument("--n", type=int, default=1024 + 7, help="number of elements")
    parser.add_argument("--block", type=int, default=256, help="BLOCK size (threadgroup size)")
    parser.add_argument("--iters", type=int, default=1, help="number of timed iterations")
    parser.add_argument("--print-msl", action="store_true", help="print generated MSL")
    args = parser.parse_args(argv)

    backend_env = os.environ.get("EAS_BACKEND", "auto")
    runtime = get_runtime()
    runtime_name = runtime.__class__.__name__
    print(f"backend={backend_env} -> {runtime_name}")

    n = int(args.n)
    block = int(args.block)
    iters = int(args.iters)
    if n <= 0 or block <= 0 or iters <= 0:
        raise SystemExit("--n/--block/--iters must be > 0")

    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros_like(a)

    # correctness (includes first-time compile)
    add_kernel(a, b, c, n, BLOCK=block)
    np.testing.assert_allclose(c, a + b)
    print("OK (correctness)")

    # test with torch tensors if available
    if TORCH_AVAILABLE:
        a_torch = torch.randn(n, dtype=torch.float32)
        b_torch = torch.randn(n, dtype=torch.float32)
        c_torch = torch.empty_like(a_torch)
        a_np = a_torch.numpy()
        b_np = b_torch.numpy()
        c_np = np.zeros_like(a_np)

        add_kernel(a_np, b_np, c_np, n, BLOCK=block)
        np.testing.assert_allclose(c_np, a_np + b_np)
        torch.add(a_torch, b_torch, out=c_torch)
        np.testing.assert_allclose(c_torch.numpy(), a_np + b_np)
        print("OK (correctness with torch tensors)")

    # quick timing (best-effort; includes launch overhead)
    t0 = time.perf_counter()
    for _ in range(iters):
        add_kernel(a, b, c, n, BLOCK=block)
    t1 = time.perf_counter()
    eas_ms = (t1 - t0) * 1e3 / iters
    print(f"time: {eas_ms:.3f} ms/iter (iters={iters})")

    if TORCH_AVAILABLE:
        # torch timing (CPU; best-effort)
        warmup = min(20, iters)
        for _ in range(warmup):
            torch.add(a_torch, b_torch, out=c_torch)
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.add(a_torch, b_torch, out=c_torch)
        t1 = time.perf_counter()
        torch_ms = (t1 - t0) * 1e3 / iters
        ratio = eas_ms / torch_ms if torch_ms > 0 else float("inf")
        print(f"torch: {torch_ms:.3f} ms/iter (iters={iters}, warmup={warmup})")
        print(f"ratio eas/torch: {ratio:.2f}x")

    if args.print_msl:
        print(add_kernel.to_msl(a, b, c, n, BLOCK=block))


if __name__ == "__main__":
    main()
