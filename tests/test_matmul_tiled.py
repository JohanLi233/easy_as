# filename: tests/test_matmul_tiled.py

from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@eas.kernel
def matmul_tiled_kernel_test(
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
    pid_n = mk.program_id(0)
    pid_m = mk.program_id(1)
    tid = mk.arange(0, NT)

    tn = 4
    if BN % tn != 0:
        raise ValueError("BN must be divisible by 4 for TN=4 mapping")
    ng = BN // tn
    if NT != BM * ng:
        raise ValueError("NT must equal BM*(BN/4) for this tiled mapping")
    rm = tid // ng
    cg = tid % ng

    row = pid_m * BM + rm
    col0 = pid_n * BN + cg * tn

    As = mk.alloc_tg(BM * BK)
    Bs = mk.alloc_tg(BK * BN)

    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0

    for k0 in range(0, K, BK):
        for i in range(0, BM * BK, NT):
            a_idx = tid + i
            in_tile = a_idx < (BM * BK)
            a_r = a_idx // BK
            a_c = a_idx % BK
            a_row = pid_m * BM + a_r
            a_col = k0 + a_c
            a_mask = mk.where(in_tile, mk.where(a_row < M, a_col < K, False), False)
            a_off = a_row * K + a_col
            mk.store(As, a_idx, mk.load(a, a_off, a_mask), in_tile)

        for i in range(0, BK * BN, NT):
            b_idx = tid + i
            in_tile = b_idx < (BK * BN)
            b_r = b_idx // BN
            b_c = b_idx % BN
            b_row = k0 + b_r
            b_col = pid_n * BN + b_c
            b_mask = mk.where(in_tile, mk.where(b_row < K, b_col < N, False), False)
            b_off = b_row * N + b_col
            mk.store(Bs, b_idx, mk.load(b, b_off, b_mask), in_tile)

        mk.barrier()

        a_base = rm * BK
        b_base = cg * tn
        d0 = mk.dot(As, a_base, 1, Bs, b_base + 0, BN, BK)
        d1 = mk.dot(As, a_base, 1, Bs, b_base + 1, BN, BK)
        d2 = mk.dot(As, a_base, 1, Bs, b_base + 2, BN, BK)
        d3 = mk.dot(As, a_base, 1, Bs, b_base + 3, BN, BK)
        acc0 = acc0 + d0
        acc1 = acc1 + d1
        acc2 = acc2 + d2
        acc3 = acc3 + d3

        mk.barrier()

    base = row * N + col0
    row_ok = row < M
    mk.store(c, base + 0, acc0, mk.where(row_ok, col0 + 0 < N, False))
    mk.store(c, base + 1, acc1, mk.where(row_ok, col0 + 1 < N, False))
    mk.store(c, base + 2, acc2, mk.where(row_ok, col0 + 2 < N, False))
    mk.store(c, base + 3, acc3, mk.where(row_ok, col0 + 3 < N, False))


class TestMatmulTiled(unittest.TestCase):
    def test_packed_float4_emitted_for_dot_group(self) -> None:
        m, n, k = 16, 64, 16
        bm, bn, bk, nt = 16, 64, 16, 256
        a2 = np.random.randn(m, k).astype(np.float32)
        b2 = np.random.randn(k, n).astype(np.float32)
        c2 = np.zeros((m, n), dtype=np.float32)

        a = a2.reshape(-1)
        b = b2.reshape(-1)
        c = c2.reshape(-1)

        ck = matmul_tiled_kernel_test.compile(
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
        )
        self.assertIn("packed_float4", ck.msl)

    def test_correctness_on_mps_if_available(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            cases = [
                (33, 65, 48),
                (64, 64, 32),
                (17, 130, 31),
            ]
            bm, bn, bk, nt = 16, 64, 16, 256
            for m, n, k in cases:
                if bn % 4 != 0:
                    raise AssertionError("bn must be divisible by 4")
                a2 = np.random.randn(m, k).astype(np.float32)
                b2 = np.random.randn(k, n).astype(np.float32)
                c2 = np.zeros((m, n), dtype=np.float32)

                a = a2.reshape(-1)
                b = b2.reshape(-1)
                c = c2.reshape(-1)

                grid = lambda meta, rargs: (  # noqa: E731
                    _cdiv(int(rargs["N"]), int(meta["BN"])),
                    _cdiv(int(rargs["M"]), int(meta["BM"])),
                    1,
                )
                matmul_tiled_kernel_test(
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
                expected = a2 @ b2
                np.testing.assert_allclose(c2, expected, rtol=2e-3, atol=2e-3)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
