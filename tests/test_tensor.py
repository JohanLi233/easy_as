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


class TestTensorCpu(unittest.TestCase):
    def test_cpu_tensor_kernel(self) -> None:
        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "cpu"
        try:
            n = 4096 + 3
            a = eas.tensor(np.random.randn(n).astype(np.float32), device="cpu")
            b = eas.tensor(np.random.randn(n).astype(np.float32), device="cpu")
            c = eas.empty_like(a)
            add_kernel(a, b, c, n, BLOCK=256)
            np.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old


class TestTensorMetal(unittest.TestCase):
    def test_metal_roundtrip_and_kernel(self) -> None:
        rt = get_runtime("metal")
        if not rt.is_available():
            self.skipTest("Metal is not available")

        n = 4096 + 3
        a_np = np.random.randn(n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        a = eas.tensor(a_np, device="metal")
        b = eas.tensor(b_np, device="metal")
        c = eas.empty_like(a)
        add_kernel(a, b, c, n, BLOCK=256)
        np.testing.assert_allclose(c.numpy(), a_np + b_np)
