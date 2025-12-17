# filename: tests/test_dot_f16_codegen.py

from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def dot_f16_scalar_kernel(a, b, out, K, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    base = offs * K
    v = mk.dot(a, base, 1, b, 0, 1, K)
    mk.store(out, offs, v, offs < (BLOCK * 2))


@eas.kernel
def dot_f16_tn4_kernel(a, b, out, K, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    _ = mk.arange(0, BLOCK)
    base = pid * 4
    d0 = mk.dot(a, 0, 1, b, base, 1, K)
    d1 = mk.dot(a, 0, 1, b, base + 1, 1, K)
    d2 = mk.dot(a, 0, 1, b, base + 2, 1, K)
    d3 = mk.dot(a, 0, 1, b, base + 3, 1, K)
    mk.store(out, base + 0, d0, True)
    mk.store(out, base + 1, d1, True)
    mk.store(out, base + 2, d2, True)
    mk.store(out, base + 3, d3, True)


class TestDotF16Codegen(unittest.TestCase):
    def test_dot_f16_scalar_loads_cast_to_float(self) -> None:
        a = np.zeros(256, dtype=np.float16)
        b = np.zeros(256, dtype=np.float16)
        out = np.zeros(256, dtype=np.float32)
        k = 8
        ck = dot_f16_scalar_kernel.compile(a, b, out, k, BLOCK=128)
        msl = ck.msl
        self.assertIn("device const half* __restrict a", msl)
        self.assertIn("device const half* __restrict b", msl)
        self.assertIn("device float* __restrict out", msl)
        self.assertIn("float(a[", msl)
        self.assertIn("float(b[", msl)

    def test_dot_f16_tn4_uses_packed_half4_when_aligned(self) -> None:
        a = np.zeros(256, dtype=np.float16)
        b = np.zeros(256, dtype=np.float16)
        out = np.zeros(256, dtype=np.float32)
        k = 8
        ck = dot_f16_tn4_kernel.compile(a, b, out, k, BLOCK=128)
        msl = ck.msl
        self.assertIn("packed_half4", msl)
        self.assertIn("& 3u", msl)


if __name__ == "__main__":
    unittest.main()
