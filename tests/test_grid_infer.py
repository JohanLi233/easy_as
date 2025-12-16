from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


@eas.kernel
def add2d_hw_kernel(a, b, c, H, W, BLOCK: eas.constexpr):
    tile = mk.program_id(0)
    row = mk.program_id(1)
    col = tile * BLOCK + mk.arange(0, BLOCK)
    offs = row * W + col
    mask = mk.where(row < H, col < W, False)
    mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)


@eas.kernel
def add2d_shape_kernel(a, b, c, M, N, BLOCK: eas.constexpr):
    tile = mk.program_id(0)
    row = mk.program_id(1)
    col = tile * BLOCK + mk.arange(0, BLOCK)
    offs = row * N + col
    mask = mk.where(row < M, col < N, False)
    mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)


class TestGridInfer(unittest.TestCase):
    def test_infer_grid_from_hw_cpu_numpy(self) -> None:
        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "cpu"
        try:
            h, w = 17, 103
            block = 32
            n = h * w
            a = np.random.randn(n).astype(np.float32)
            b = np.random.randn(n).astype(np.float32)
            c = np.zeros_like(a)

            add2d_hw_kernel(a, b, c, h, w, BLOCK=block)
            np.testing.assert_allclose(c, a + b)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

    def test_infer_grid_from_shape_option_cpu_numpy(self) -> None:
        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "cpu"
        try:
            m, n = 9, 65
            block = 16
            size = m * n
            a = np.random.randn(size).astype(np.float32)
            b = np.random.randn(size).astype(np.float32)
            c = np.zeros_like(a)

            add2d_shape_kernel(a, b, c, m, n, BLOCK=block, _shape=(m, n))
            np.testing.assert_allclose(c, a + b)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

    def test_infer_grid_from_hw_metal_tensor(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            h, w = 11, 77
            block = 32
            size = h * w
            a_np = np.random.randn(size).astype(np.float32)
            b_np = np.random.randn(size).astype(np.float32)
            a = eas.tensor(a_np, device="mps")
            b = eas.tensor(b_np, device="mps")
            c = eas.empty(size, device="mps")

            add2d_hw_kernel(a, b, c, h, w, BLOCK=block)
            np.testing.assert_allclose(c.numpy(), a_np + b_np)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
