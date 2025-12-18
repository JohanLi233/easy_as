# filename: tests/test_dot_dynamic_k.py

from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


@eas.kernel
def gemv_dot_kernel(a, x, y, M, K, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.tid(0, BLOCK)
    row = offs
    mask = row < M
    base = row * K
    y_val = mk.dot(a, base, 1, x, 0, 1, K)
    mk.store(y, row, y_val, mask)


class TestDotDynamicK(unittest.TestCase):
    def test_dynamic_k_on_mps_if_available(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            m = 33
            ks = [7, 31, 64]
            block = 64
            for k in ks:
                a2 = np.random.randn(m, k).astype(np.float32)
                x = np.random.randn(k).astype(np.float32)
                y = np.zeros(m, dtype=np.float32)

                gemv_dot_kernel(a2.reshape(-1), x, y, m, k, m, BLOCK=block)
                expected = a2 @ x
                np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-4)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

    def test_msl_contains_for_loop(self) -> None:
        m, k = 8, 11
        block = 64
        a2 = np.random.randn(m, k).astype(np.float32)
        x = np.random.randn(k).astype(np.float32)
        y = np.zeros(m, dtype=np.float32)
        ck = gemv_dot_kernel.compile(a2.reshape(-1), x, y, m, k, m, BLOCK=block)
        self.assertIn("for (uint k", ck.msl)


if __name__ == "__main__":
    unittest.main()
