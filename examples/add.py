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
    if runtime_name == "MetalRuntime":
        a_e = eas.tensor(a, device="metal")
        b_e = eas.tensor(b, device="metal")
        c_e = eas.empty_like(a_e)
        add_kernel(a_e, b_e, c_e, n, BLOCK=block)
        np.testing.assert_allclose(c_e.numpy(), a + b)
    else:
        add_kernel(a, b, c, n, BLOCK=block)
        np.testing.assert_allclose(c, a + b)
    print("OK (correctness)")

    # test with torch tensors if available
    if TORCH_AVAILABLE:
        # Determine device
        device = torch.device('mps') if TORCH_MPS_AVAILABLE else torch.device('cpu')
        a_torch = torch.randn(n, dtype=torch.float32, device=device)
        b_torch = torch.randn(n, dtype=torch.float32, device=device)
        c_torch = torch.empty_like(a_torch)
        # Move to CPU for numpy conversion
        a_np = a_torch.cpu().numpy()
        b_np = b_torch.cpu().numpy()
        c_np = np.zeros_like(a_np)

        if runtime_name == "MetalRuntime":
            a_e = eas.tensor(a_np, device="metal")
            b_e = eas.tensor(b_np, device="metal")
            c_e = eas.empty_like(a_e)
            add_kernel(a_e, b_e, c_e, n, BLOCK=block)
            np.testing.assert_allclose(c_e.numpy(), a_np + b_np)
        else:
            add_kernel(a_np, b_np, c_np, n, BLOCK=block)
            np.testing.assert_allclose(c_np, a_np + b_np)
        torch.add(a_torch, b_torch, out=c_torch)
        # Move result to CPU for comparison
        np.testing.assert_allclose(c_torch.cpu().numpy(), a_np + b_np)
        print(f"OK (correctness with torch tensors, device={device})")

    # quick timing (best-effort; includes launch overhead)
    if hasattr(runtime, "synchronize"):
        runtime.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        if runtime_name == "MetalRuntime":
            add_kernel(a_e, b_e, c_e, n, BLOCK=block, _sync=False)
        else:
            add_kernel(a, b, c, n, BLOCK=block, _sync=False)
    if hasattr(runtime, "synchronize"):
        runtime.synchronize()
    t1 = time.perf_counter()
    eas_ms = (t1 - t0) * 1e3 / iters
    print(f"time: {eas_ms:.3f} ms/iter (iters={iters})")

    if TORCH_AVAILABLE:
        # torch timing (CPU or MPS; best-effort)
        warmup = min(20, iters)
        for _ in range(warmup):
            torch.add(a_torch, b_torch, out=c_torch)
        if TORCH_MPS_AVAILABLE:
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.add(a_torch, b_torch, out=c_torch)
        if TORCH_MPS_AVAILABLE:
            torch.mps.synchronize()
        t1 = time.perf_counter()
        torch_ms = (t1 - t0) * 1e3 / iters
        ratio = eas_ms / torch_ms if torch_ms > 0 else float("inf")
        device_str = "mps" if TORCH_MPS_AVAILABLE else "cpu"
        print(f"torch ({device_str}): {torch_ms:.3f} ms/iter (iters={iters}, warmup={warmup})")
        print(f"ratio eas/torch: {ratio:.2f}x")

    if args.print_msl:
        print(add_kernel.to_msl(a, b, c, n, BLOCK=block))


if __name__ == "__main__":
    main()
