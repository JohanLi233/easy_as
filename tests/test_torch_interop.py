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

        n = 1024 + 3
        a_t = torch.zeros(n, device="cpu", dtype=torch.float32)
        a = eas.tensor(a_t, device="cpu")

        a_t.add_(1.0)
        np.testing.assert_allclose(a.numpy(), a_t.numpy(), rtol=0, atol=0)

        a.numpy()[0] = 7.0
        self.assertEqual(float(a_t[0].item()), 7.0)

    def test_mps_zero_copy_via_dlpack(self) -> None:
        import torch  # type: ignore

        if not torch.backends.mps.is_available():
            self.skipTest("torch mps is not available")

        from eas.runtime import get_runtime

        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS runtime is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            n = 4096 + 3
            a = torch.randn(n, device="mps", dtype=torch.float32)
            b = torch.randn(n, device="mps", dtype=torch.float32)
            c = torch.empty_like(a)

            add_kernel(a, b, c, n, BLOCK=256)
            torch.mps.synchronize()

            np.testing.assert_allclose(
                c.cpu().numpy(), (a + b).cpu().numpy(), rtol=1e-5, atol=1e-6
            )
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old
