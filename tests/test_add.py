from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


@eas.kernel
def add_kernel(a, b, c, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N
    mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)


class TestAddKernel(unittest.TestCase):
    def test_add_mps_numpy(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            n = 4096 + 3
            a = np.random.randn(n).astype(np.float32)
            b = np.random.randn(n).astype(np.float32)
            c = np.zeros_like(a)

            add_kernel(a, b, c, n, BLOCK=256)
            np.testing.assert_allclose(c, a + b)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

    def test_add_metal_tensor(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            n = 4096 + 3
            a_np = np.random.randn(n).astype(np.float32)
            b_np = np.random.randn(n).astype(np.float32)
            a = eas.tensor(a_np, device="mps")
            b = eas.tensor(b_np, device="mps")
            c = eas.empty_like(a)

            add_kernel(a, b, c, n, BLOCK=256)
            np.testing.assert_allclose(c.numpy(), a_np + b_np)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
