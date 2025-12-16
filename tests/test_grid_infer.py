from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime
from eas.runtime.grid import infer_grid


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
    def test_infer_grid_from_hw_numpy(self) -> None:
        h, w = 17, 103
        block = 32
        n = h * w
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.zeros_like(a)

        ck = add2d_hw_kernel.compile(a, b, c, h, w, BLOCK=block)
        self.assertEqual(
            infer_grid(ck.ir, {"H": h, "W": w}, ck.threadgroup_size), (4, 17, 1)
        )

    def test_infer_grid_from_shape_option_numpy(self) -> None:
        m, n = 9, 65
        block = 16
        size = m * n
        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)
        c = np.zeros_like(a)

        ck = add2d_shape_kernel.compile(a, b, c, m, n, BLOCK=block)
        self.assertEqual(
            infer_grid(ck.ir, {}, ck.threadgroup_size, shape=(m, n)),
            (5, 9, 1),
        )

    def test_infer_grid_from_hw_metal_tensor(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

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


if __name__ == "__main__":
    unittest.main()
