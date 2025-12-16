# filename: tests/test_matmul.py

from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


@eas.kernel
def matmul_kernel(a, b, c, M, N, BLOCK: eas.constexpr, K: eas.constexpr):
    row = mk.program_id(0)
    tile = mk.program_id(1)
    col = tile * BLOCK + mk.arange(0, BLOCK)
    out = row * N + col
    mask = mk.where(row < M, col < N, False)

    acc = 0.0
    for k in range(K):
        a_off = row * K + k
        b_off = k * N + col
        acc = acc + mk.load(a, a_off, mask) * mk.load(b, b_off, mask)
    mk.store(c, out, acc, mask)


class TestMatmulKernel(unittest.TestCase):
    def test_matmul_mps_numpy(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            m = 32
            n = 100
            k = 16
            block = 64

            a2 = np.random.randn(m, k).astype(np.float32)
            b2 = np.random.randn(k, n).astype(np.float32)
            c2 = np.zeros((m, n), dtype=np.float32)

            a = a2.reshape(-1)
            b = b2.reshape(-1)
            c = c2.reshape(-1)

            tiles_n = (n + block - 1) // block
            matmul_kernel(a, b, c, m, n, BLOCK=block, K=k, _grid=(m, tiles_n))
            ref = (a2.astype(np.float64) @ b2.astype(np.float64)).astype(np.float32)
            np.testing.assert_allclose(c2, ref, rtol=1e-4, atol=1e-5)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

    def test_matmul_metal_tensor(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            m = 8
            n = 48
            k = 16
            block = 32

            a2 = np.random.randn(m, k).astype(np.float32)
            b2 = np.random.randn(k, n).astype(np.float32)
            tiles_n = (n + block - 1) // block
            m * tiles_n * block

            a = eas.tensor(a2.reshape(-1), device="mps")
            b = eas.tensor(b2.reshape(-1), device="mps")
            c = eas.empty(m * n, device="mps")

            matmul_kernel(a, b, c, m, n, BLOCK=block, K=k, _grid=(m, tiles_n))
            c2 = c.numpy().reshape(m, n)
            ref = (a2.astype(np.float64) @ b2.astype(np.float64)).astype(np.float32)
            np.testing.assert_allclose(c2, ref, rtol=1e-4, atol=1e-5)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

    def test_matmul_large_k_codegen_no_recursion_numpy(self) -> None:
        m = 1
        n = 3
        k = 1200
        block = 2

        a2 = np.random.randn(m, k).astype(np.float32)
        b2 = np.random.randn(k, n).astype(np.float32)
        c2 = np.zeros((m, n), dtype=np.float32)

        a = a2.reshape(-1)
        b = b2.reshape(-1)
        c = c2.reshape(-1)

        tiles_n = (n + block - 1) // block
        _ = matmul_kernel.compile(a, b, c, m, n, BLOCK=block, K=k, _grid=(m, tiles_n))


if __name__ == "__main__":
    unittest.main()
