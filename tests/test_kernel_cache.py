# filename: tests/test_kernel_cache.py

from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


@eas.kernel
def axpy_kernel(a, b, c, N, alpha, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N
    mk.store(
        c,
        offs,
        mk.load(a, offs, mask) * alpha + mk.load(b, offs, mask),
        mask,
    )


class TestKernelCache(unittest.TestCase):
    def test_compile_cache_ignores_scalar_values(self) -> None:
        n = 1024 + 3
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.zeros_like(a)

        ck1 = axpy_kernel.compile(a, b, c, n, alpha=2.0, BLOCK=256)
        ck2 = axpy_kernel.compile(a, b, c, n, alpha=3.0, BLOCK=256)
        self.assertIs(ck1, ck2)

        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")
        rt.run(
            ck1,
            {"a": a, "b": b, "c": c, "N": n, "alpha": 2.0},
            {"BLOCK": 256},
            sync=True,
        )
        np.testing.assert_allclose(c, a * 2.0 + b)
