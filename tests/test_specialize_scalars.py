from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk


class TestSpecializeScalars(unittest.TestCase):
    def test_constexpr_removed_but_runtime_scalar_kept(self) -> None:
        @eas.kernel
        def add_kernel(a, b, c, N, BLOCK: eas.constexpr):
            pid = mk.program_id(0)
            offs = pid * BLOCK + mk.arange(0, BLOCK)
            mask = offs < N
            mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)

        n = 1024 + 3
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.empty_like(a)

        msl_default = add_kernel.to_msl(a, b, c, n, BLOCK=256)
        self.assertIn("constant uint& N", msl_default)
        self.assertNotIn("BLOCK", msl_default)

        # Scalar specialization was removed; this env var is now a no-op.
        old = os.environ.get("EAS_SPECIALIZE_SCALARS")
        os.environ["EAS_SPECIALIZE_SCALARS"] = "N"
        try:
            msl_spec = add_kernel.to_msl(a, b, c, n, BLOCK=256)
        finally:
            if old is None:
                os.environ.pop("EAS_SPECIALIZE_SCALARS", None)
            else:
                os.environ["EAS_SPECIALIZE_SCALARS"] = old

        self.assertIn("constant uint& N", msl_spec)
