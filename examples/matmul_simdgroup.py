# filename: examples/matmul_simdgroup.py

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


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@eas.kernel
def matmul_simdgroup_mma_kernel(
    a,
    b,
    c,
    M,
    N,
    BM: eas.constexpr,
    BN: eas.constexpr,
    BK: eas.constexpr,
    K: eas.constexpr,
    NT: eas.constexpr,
):
    """
    SIMD-group MMA (8x8x8) matmul: C[M,N] = A[M,K] @ B[K,N]

    Constraints:
      - BM/BN/BK must be multiples of 8
      - NT must equal (BM/8)*(BN/8)*32
    """

    if BM % 8 != 0 or BN % 8 != 0 or BK % 8 != 0:
        raise ValueError("BM/BN/BK must be divisible by 8 for 8x8 MMA")
    if NT != (BM // 8) * (BN // 8) * 32:
        raise ValueError("NT must be (BM/8)*(BN/8)*32 for 8x8 MMA")

    pid_n = mk.program_id(0)
    pid_m = mk.program_id(1)
    tid = mk.tid(0, NT)
    sg = mk.sg_id()

    base_m = pid_m * BM
    base_n = pid_n * BN

    sgn = BN // 8
    sg_m = sg // sgn
    sg_n = sg % sgn

    lda = BK + 1
    ldb = BN + 1
    ldc = BN + 1
    As = mk.alloc_tg_f16(BM * lda)
    Bs = mk.alloc_tg_f16(BK * ldb)
    Cs = mk.alloc_tg(BM * ldc)

    acc = mk.mma_zero()

    for k0 in range(0, K, BK):
        # ---- load A tile (BM*BK) ----
        for i in range(0, BM * BK, NT):
            a_idx = tid + i
            a_r = a_idx // BK
            a_c = a_idx % BK
            as_off = a_r * lda + a_c
            a_row = base_m + a_r
            a_col = k0 + a_c
            in_tile = a_idx < BM * BK
            a_mask = mk.where(in_tile, mk.where(a_row < M, a_col < K, False), False)
            a_off = a_row * K + a_col
            mk.store(As, as_off, mk.to_f16(mk.load(a, a_off, a_mask)), in_tile)

        # ---- load B tile (BK*BN) ----
        for i in range(0, BK * BN, NT):
            b_idx = tid + i
            b_r = b_idx // BN
            b_c = b_idx % BN
            bs_off = b_r * ldb + b_c
            b_row = k0 + b_r
            b_col = base_n + b_c
            in_tile = b_idx < BK * BN
            b_mask = mk.where(in_tile, mk.where(b_row < K, b_col < N, False), False)
            b_off = b_row * N + b_col
            mk.store(Bs, bs_off, mk.to_f16(mk.load(b, b_off, b_mask)), in_tile)

        mk.barrier()

        # ---- MMA: each simdgroup computes one 8x8 C subtile ----
        for kk in range(0, BK, 8):
            a_base = (sg_m * 8) * lda + kk
            b_base = kk * ldb + (sg_n * 8)
            acc = mk.mma(As, a_base, lda, Bs, b_base, ldb, acc)

        mk.barrier()

    # ---- store MMA result to Cs (threadgroup) ----
    c_base = (sg_m * 8) * ldc + (sg_n * 8)
    mk.mma_store(Cs, c_base, ldc, acc)
    mk.barrier()

    # ---- write Cs -> C with bounds checks ----
    for i in range(0, BM * BN, NT):
        c_idx = tid + i
        r = c_idx // BN
        cc = c_idx % BN
        row = base_m + r
        col = base_n + cc
        mask = mk.where(row < M, col < N, False)
        val = mk.load(Cs, r * ldc + cc, True)
        mk.store(c, row * N + col, val, mask)


def _torch_device_for_runtime(runtime_name: str) -> "torch.device":
    return (
        torch.device("mps")
        if (runtime_name == "MetalRuntime" and TORCH_MPS_AVAILABLE)
        else torch.device("cpu")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="easy_as simdgroup MMA matmul 演示（8x8x8）"
    )
    parser.add_argument("--m", type=int, default=512, help="M（行数）")
    parser.add_argument("--n", type=int, default=512, help="N（列数）")
    parser.add_argument("--k", type=int, default=512, help="K（constexpr，展开）")
    parser.add_argument("--bm", type=int, default=16, help="BM（tile 行）")
    parser.add_argument("--bn", type=int, default=32, help="BN（tile 列）")
    parser.add_argument("--bk", type=int, default=8, help="BK（K 分块）")
    parser.add_argument(
        "--nt",
        type=int,
        default=256,
        help="NT（threads_per_threadgroup.x，需满足 (BM/8)*(BN/8)*32）",
    )
    parser.add_argument("--iters", type=int, default=50, help="计时迭代次数")
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

    m = int(args.m)
    n = int(args.n)
    k = int(args.k)
    bm = int(args.bm)
    bn = int(args.bn)
    bk = int(args.bk)
    nt = int(args.nt)
    iters = int(args.iters)

    if min(m, n, k, bm, bn, bk, nt, iters) <= 0:
        raise SystemExit("all sizes must be > 0")

    device = _torch_device_for_runtime(runtime_name)
    a = torch.randn((m, k), dtype=torch.float32, device=device)
    b = torch.randn((k, n), dtype=torch.float32, device=device)
    c = torch.empty((m, n), dtype=torch.float32, device=device)

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    c_flat = c.reshape(-1)

    grid = lambda meta, rargs: (  # noqa: E731
        _cdiv(int(rargs["N"]), int(meta["BN"])),
        _cdiv(int(rargs["M"]), int(meta["BM"])),
        1,
    )

    # correctness + first compile
    matmul_simdgroup_mma_kernel(
        a_flat,
        b_flat,
        c_flat,
        m,
        n,
        BM=bm,
        BN=bn,
        BK=bk,
        K=k,
        NT=nt,
        _grid=grid,
        _tptg=(nt, 1, 1),
    )
    if device.type == "mps":
        torch.mps.synchronize()

    # Reference: match the kernel's half-input MMA path on MPS.
    if device.type == "mps":
        ref = (a.to(torch.float16) @ b.to(torch.float16)).to(torch.float32)
    else:
        ref = a @ b
    if hasattr(torch.testing, "assert_close"):
        torch.testing.assert_close(c, ref, rtol=5e-3, atol=5e-3)
    else:
        if not torch.allclose(c, ref, rtol=5e-3, atol=5e-3):
            raise AssertionError("torch.allclose failed")
    print(f"OK（correctness, device={device}）")

    if hasattr(runtime, "synchronize"):
        runtime.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    eas_ms: float
    if runtime_name == "MetalRuntime" and callable(getattr(runtime, "benchmark", None)):
        ck = matmul_simdgroup_mma_kernel.compile(
            a_flat,
            b_flat,
            c_flat,
            m,
            n,
            BM=bm,
            BN=bn,
            BK=bk,
            K=k,
            NT=nt,
            _grid=grid,
            _tptg=(nt, 1, 1),
        )
        warmup = min(10, iters)
        t = runtime.benchmark(
            ck,
            {"a": a_flat, "b": b_flat, "c": c_flat, "M": m, "N": n},
            {"BM": bm, "BN": bn, "BK": bk, "K": k, "NT": nt},
            repeat=iters,
            warmup=warmup,
            torch_mps_sync=False,
            grid=grid(
                {"BM": bm, "BN": bn, "BK": bk, "K": k, "NT": nt}, {"M": m, "N": n}
            ),
            tptg=(nt, 1, 1),
        )
        eas_ms = t * 1e3
        print(f"eas mma: {eas_ms:.3f} ms/iter (iters={iters})")
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            matmul_simdgroup_mma_kernel(
                a_flat,
                b_flat,
                c_flat,
                m,
                n,
                BM=bm,
                BN=bn,
                BK=bk,
                K=k,
                NT=nt,
                _grid=grid,
                _tptg=(nt, 1, 1),
                _sync=False,
            )
        if hasattr(runtime, "synchronize"):
            runtime.synchronize()
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        eas_ms = (t1 - t0) * 1e3 / iters
        print(f"eas mma: {eas_ms:.3f} ms/iter (iters={iters}, mode=loop)")

    # torch timing (best-effort)
    warmup = min(20, iters)
    for _ in range(warmup):
        _ = a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    torch_ms = (t1 - t0) * 1e3 / iters
    ratio = eas_ms / torch_ms if torch_ms > 0 else float("inf")
    print(f"torch ({device.type}): {torch_ms:.3f} ms/iter")
    print(f"ratio eas/torch: {ratio:.2f}x")

    if args.print_msl:
        print(
            matmul_simdgroup_mma_kernel.to_msl(
                np.zeros((1,), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                m,
                n,
                BM=bm,
                BN=bn,
                BK=bk,
                K=k,
                NT=nt,
                _grid=grid,
                _tptg=(nt, 1, 1),
            )
        )


if __name__ == "__main__":
    main()
