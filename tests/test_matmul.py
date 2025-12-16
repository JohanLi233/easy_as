from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


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
    def test_matmul_cpu_numpy(self) -> None:
        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "cpu"
        try:
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
            k = 16
            block = 32

            a2 = np.random.randn(m, k).astype(np.float32)
            b2 = np.random.randn(k, block).astype(np.float32)
            n_total = m * block

            a = eas.tensor(a2.reshape(-1), device="mps")
            b = eas.tensor(b2.reshape(-1), device="mps")
            c = eas.empty(n_total, device="mps")

            matmul_kernel(a, b, c, n_total, BLOCK=block, K=k)
            c2 = c.numpy().reshape(m, block)
            np.testing.assert_allclose(c2, a2 @ b2, rtol=1e-4, atol=1e-5)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

    def test_matmul_large_k_codegen_no_recursion_cpu_numpy(self) -> None:
        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "cpu"
        try:
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
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
