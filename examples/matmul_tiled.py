# filename: examples/matmul_tiled.py

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


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@eas.kernel
def matmul_tiled_kernel(
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
    Threadgroup-tiled matmul: C[M,N] = A[M,K] @ B[K,N]

    Config (first version):
      - threads_per_threadgroup = NT (1D)
      - tile = (BM x BN)
      - K blocking = BK
      - each thread computes 4 cols (TN=4)
      - uses threadgroup memory: As[BM*BK], Bs[BK*BN]
    """

    pid_n = mk.program_id(0)  # tile along N
    pid_m = mk.program_id(1)  # tile along M
    tid = mk.arange(0, NT)  # 0..NT-1

    tn = 4
    if BN % tn != 0:
        raise ValueError("BN must be divisible by 4 for TN=4 mapping")
    ng = BN // tn
    if NT != BM * ng:
        raise ValueError("NT must equal BM*(BN/4) for this tiled mapping")
    lda = BK + 1
    ldb = BN + 1
    rm = tid // ng
    cg = tid % ng

    row = pid_m * BM + rm
    col0 = pid_n * BN + cg * tn

    As = mk.alloc_tg(BM * lda)
    Bs = mk.alloc_tg(BK * ldb)

    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0

    for k0 in range(0, K, BK):
        # ---- load As ----
        for i in range(0, BM * BK, NT):
            a_idx = tid + i
            in_tile = a_idx < (BM * BK)
            a_r = a_idx // BK
            a_c = a_idx % BK
            as_off = a_r * lda + a_c
            a_row = pid_m * BM + a_r
            a_col = k0 + a_c
            a_mask = mk.where(in_tile, mk.where(a_row < M, a_col < K, False), False)
            a_off = a_row * K + a_col
            mk.store(As, as_off, mk.load(a, a_off, a_mask), in_tile)

        # ---- load Bs ----
        for i in range(0, BK * BN, NT):
            b_idx = tid + i
            in_tile = b_idx < (BK * BN)
            b_r = b_idx // BN
            b_c = b_idx % BN
            bs_off = b_r * ldb + b_c
            b_row = k0 + b_r
            b_col = pid_n * BN + b_c
            b_mask = mk.where(in_tile, mk.where(b_row < K, b_col < N, False), False)
            b_off = b_row * N + b_col
            mk.store(Bs, bs_off, mk.load(b, b_off, b_mask), in_tile)

        mk.barrier()

        # ---- compute ----
        a_base = rm * lda
        b_base = cg * tn
        d0 = mk.dot(As, a_base, 1, Bs, b_base + 0, ldb, BK)
        d1 = mk.dot(As, a_base, 1, Bs, b_base + 1, ldb, BK)
        d2 = mk.dot(As, a_base, 1, Bs, b_base + 2, ldb, BK)
        d3 = mk.dot(As, a_base, 1, Bs, b_base + 3, ldb, BK)
        acc0 = acc0 + d0
        acc1 = acc1 + d1
        acc2 = acc2 + d2
        acc3 = acc3 + d3

        mk.barrier()

    # ---- store ----
    base = row * N + col0
    row_ok = row < M
    mk.store(c, base + 0, acc0, mk.where(row_ok, col0 + 0 < N, False))
    mk.store(c, base + 1, acc1, mk.where(row_ok, col0 + 1 < N, False))
    mk.store(c, base + 2, acc2, mk.where(row_ok, col0 + 2 < N, False))
    mk.store(c, base + 3, acc3, mk.where(row_ok, col0 + 3 < N, False))


@eas.kernel
def matmul_tiled_full_kernel(
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
    Fast path: valid only for *interior/full tiles*.

    This kernel omits all bounds checks on global loads/stores for performance,
    so it must only be launched with:
      - grid = (floor_div(N, BN), floor_div(M, BM), 1)
      - and K divisible by BK
    The example driver selects this kernel only when M/N/K are aligned; otherwise
    it falls back to the masked kernel.
    """

    pid_n = mk.program_id(0)
    pid_m = mk.program_id(1)
    tid = mk.arange(0, NT)

    tn = 4
    if BN % tn != 0:
        raise ValueError("BN must be divisible by 4 for TN=4 mapping")
    ng = BN // tn
    if NT != BM * ng:
        raise ValueError("NT must equal BM*(BN/4) for this tiled mapping")
    if (BM * BK) % NT != 0:
        raise ValueError("full kernel requires BM*BK divisible by NT")
    if (BK * BN) % NT != 0:
        raise ValueError("full kernel requires BK*BN divisible by NT")
    lda = BK + 1
    ldb = BN + 1
    rm = tid // ng
    cg = tid % ng

    row = pid_m * BM + rm
    col0 = pid_n * BN + cg * tn

    As = mk.alloc_tg(BM * lda)
    Bs = mk.alloc_tg(BK * ldb)

    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0

    for k0 in range(0, K, BK):
        for i in range(0, BM * BK, NT):
            a_idx = tid + i
            a_r = a_idx // BK
            a_c = a_idx % BK
            as_off = a_r * lda + a_c
            a_row = pid_m * BM + a_r
            a_col = k0 + a_c
            a_off = a_row * K + a_col
            mk.store(As, as_off, mk.load(a, a_off, True), True)

        for i in range(0, BK * BN, NT):
            b_idx = tid + i
            b_r = b_idx // BN
            b_c = b_idx % BN
            bs_off = b_r * ldb + b_c
            b_row = k0 + b_r
            b_col = pid_n * BN + b_c
            b_off = b_row * N + b_col
            mk.store(Bs, bs_off, mk.load(b, b_off, True), True)

        mk.barrier()

        a_base = rm * lda
        b_base = cg * tn
        d0 = mk.dot(As, a_base, 1, Bs, b_base + 0, ldb, BK)
        d1 = mk.dot(As, a_base, 1, Bs, b_base + 1, ldb, BK)
        d2 = mk.dot(As, a_base, 1, Bs, b_base + 2, ldb, BK)
        d3 = mk.dot(As, a_base, 1, Bs, b_base + 3, ldb, BK)
        acc0 = acc0 + d0
        acc1 = acc1 + d1
        acc2 = acc2 + d2
        acc3 = acc3 + d3

        mk.barrier()

    base = row * N + col0
    mk.store(c, base + 0, acc0, True)
    mk.store(c, base + 1, acc1, True)
    mk.store(c, base + 2, acc2, True)
    mk.store(c, base + 3, acc3, True)


def _torch_device_for_runtime(runtime_name: str) -> "torch.device":
    return (
        torch.device("mps")
        if (runtime_name == "MetalRuntime" and TORCH_MPS_AVAILABLE)
        else torch.device("cpu")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="easy_as tiled matmul 演示（threadgroup 缓存）"
    )
    parser.add_argument("--m", type=int, default=640, help="M（行数）")
    parser.add_argument("--n", type=int, default=640, help="N（列数）")
    parser.add_argument("--k", type=int, default=640, help="K（constexpr，展开）")
    parser.add_argument("--bm", type=int, default=16, help="BM（tile 行）")
    parser.add_argument("--bn", type=int, default=64, help="BN（tile 列）")
    parser.add_argument("--bk", type=int, default=16, help="BK（K 分块）")
    parser.add_argument(
        "--nt", type=int, default=256, help="NT（threads_per_threadgroup.x）"
    )
    parser.add_argument("--iters", type=int, default=100, help="计时迭代次数")
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
        raise SystemExit("all dims/params must be > 0")
    if bn % 4 != 0:
        raise SystemExit("--bn must be divisible by 4 for TN=4 mapping")
    if nt != bm * (bn // 4):
        raise SystemExit("--nt must equal bm*(bn/4) for this mapping")

    # Threadgroup memory for As[BM*BK] + Bs[BK*BN] (float32) must fit device limits.
    # 32 KiB is the common limit on Apple GPUs.
    tg_bytes = 4 * (bm * (bk + 1) + bk * (bn + 1))
    if tg_bytes > 32 * 1024:
        raise SystemExit(
            f"threadgroup memory too large: {tg_bytes} bytes (BM*(BK+1) + BK*(BN+1)). "
            "Try a smaller --bk/--bn/--bm."
        )

    device = _torch_device_for_runtime(runtime_name)
    a2 = torch.randn((m, k), dtype=torch.float32, device=device)
    b2 = torch.randn((k, n), dtype=torch.float32, device=device)
    c2 = torch.empty((m, n), dtype=torch.float32, device=device)
    a = a2.reshape(-1)
    b = b2.reshape(-1)
    c = c2.reshape(-1)

    grid = lambda meta, rargs: (  # noqa: E731
        _cdiv(int(rargs["N"]), int(meta["BN"])),
        _cdiv(int(rargs["M"]), int(meta["BM"])),
        1,
    )

    fast_path = (
        (m % bm == 0)
        and (n % bn == 0)
        and (k % bk == 0)
        and ((bm * bk) % nt == 0)
        and ((bk * bn) % nt == 0)
    )
    kernel = matmul_tiled_full_kernel if fast_path else matmul_tiled_kernel
    if fast_path:
        print("fast_path=1 (full tiles, unmasked loads/stores)")
    else:
        print("fast_path=0 (masked loads/stores)")

    # Correctness + compile
    kernel(
        a,
        b,
        c,
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
    expected = a2 @ b2
    # Tiled accumulation order differs from torch (and uses explicit fma), so allow a
    # slightly larger absolute tolerance than the naive kernel.
    if hasattr(torch.testing, "assert_close"):
        torch.testing.assert_close(c2, expected, rtol=1e-4, atol=1e-4)
    else:
        if not torch.allclose(c2, expected, rtol=1e-4, atol=1e-4):
            raise AssertionError("torch.allclose failed")
    print(f"OK（正确性验证，设备={device}，tile=({bm},{bn},{bk}), nt={nt})")

    if args.print_msl:
        print(
            kernel.to_msl(
                a,
                b,
                c,
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

    if hasattr(runtime, "synchronize"):
        runtime.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    # Timings (best-effort)
    t0 = time.perf_counter()
    for _ in range(iters):
        kernel(
            a,
            b,
            c,
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
    print(f"eas tiled: {eas_ms:.3f} ms/iter (iters={iters})")

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


if __name__ == "__main__":
    main()
