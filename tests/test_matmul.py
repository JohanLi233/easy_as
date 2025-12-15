from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def matmul_kernel(a, b, c, N, BLOCK: eas.constexpr, K: eas.constexpr):
    pid = mk.program_id(0)
    col = mk.arange(0, BLOCK)
    out = pid * BLOCK + col
    mask = out < N

    acc = 0.0
    for k in range(K):
        a_off = pid * K + k
        b_off = k * BLOCK + col
        acc = acc + mk.load(a, a_off, mask) * mk.load(b, b_off, mask)
    mk.store(c, out, acc, mask)


class TestMatmulKernel(unittest.TestCase):
    def test_matmul(self) -> None:
        m = 32
        k = 16
        block = 64

        a2 = np.random.randn(m, k).astype(np.float32)
        b2 = np.random.randn(k, block).astype(np.float32)
        c2 = np.zeros((m, block), dtype=np.float32)

        a = a2.reshape(-1)
        b = b2.reshape(-1)
        c = c2.reshape(-1)

        n_total = m * block
        matmul_kernel(a, b, c, n_total, BLOCK=block, K=k)
        np.testing.assert_allclose(c2, a2 @ b2, rtol=1e-4, atol=1e-5)

    def test_matmul_large_k_codegen_no_recursion(self) -> None:
        m = 1
        k = 1200
        block = 1

        a2 = np.random.randn(m, k).astype(np.float32)
        b2 = np.random.randn(k, block).astype(np.float32)
        c2 = np.zeros((m, block), dtype=np.float32)

        a = a2.reshape(-1)
        b = b2.reshape(-1)
        c = c2.reshape(-1)

        n_total = m * block
        matmul_kernel(a, b, c, n_total, BLOCK=block, K=k)
        np.testing.assert_allclose(c2, a2 @ b2, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
