from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk


def _torch_available() -> bool:
    try:
        import torch  # type: ignore

        _ = torch
        return True
    except Exception:
        return False


@eas.kernel
def add_kernel(a, b, c, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N
    mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)


@unittest.skipUnless(_torch_available(), "torch is not installed")
class TestTorchInterop(unittest.TestCase):
    def test_cpu_zero_copy_wrap(self) -> None:
        import torch  # type: ignore

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "cpu"
        try:
            n = 4096 + 3
            a_t = torch.randn(n, device="cpu", dtype=torch.float32)
            b_t = torch.randn(n, device="cpu", dtype=torch.float32)
            c_t = torch.zeros(n, device="cpu", dtype=torch.float32)

            a = eas.tensor(a_t, device="cpu")
            b = eas.tensor(b_t, device="cpu")
            c = eas.tensor(c_t, device="cpu")

            add_kernel(a, b, c, n, BLOCK=256)

            # c_t should reflect results without an explicit copy.
            np.testing.assert_allclose(c_t.numpy(), a_t.numpy() + b_t.numpy())
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

