from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def add_kernel(a, b, c, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N
    mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)


class TestAddKernel(unittest.TestCase):
    def test_add(self) -> None:
        n = 4096 + 3
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.zeros_like(a)

        add_kernel(a, b, c, n, BLOCK=256)
        np.testing.assert_allclose(c, a + b)


if __name__ == "__main__":
    unittest.main()
