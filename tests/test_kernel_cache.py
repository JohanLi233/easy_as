from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk


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
        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "cpu"
        try:
            n = 1024 + 3
            a = np.random.randn(n).astype(np.float32)
            b = np.random.randn(n).astype(np.float32)
            c = np.zeros_like(a)

            ck1 = axpy_kernel.compile(a, b, c, n, alpha=2.0, BLOCK=256)
            ck2 = axpy_kernel.compile(a, b, c, n, alpha=3.0, BLOCK=256)
            self.assertIs(ck1, ck2)

            axpy_kernel(a, b, c, n, alpha=2.0, BLOCK=256)
            np.testing.assert_allclose(c, a * 2.0 + b)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

