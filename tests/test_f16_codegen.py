# filename: tests/test_f16_codegen.py

from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def add_f16_kernel(a, b, c, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N
    av = mk.load(a, offs, mask)
    bv = mk.load(b, offs, mask)
    mk.store(c, offs, av + bv, mask)


class TestF16Codegen(unittest.TestCase):
    def test_f16_buffers_emit_half_pointers(self) -> None:
        n = 256
        a = (np.random.randn(n).astype(np.float16)).reshape(-1)
        b = (np.random.randn(n).astype(np.float16)).reshape(-1)
        c = np.zeros_like(a)
        ck = add_f16_kernel.compile(a, b, c, n, BLOCK=256)
        self.assertIn("device const half* __restrict a", ck.msl)
        self.assertIn("device const half* __restrict b", ck.msl)
        self.assertIn("device half* __restrict c", ck.msl)


if __name__ == "__main__":
    unittest.main()
