# filename: tests/test_dlpack.py

from __future__ import annotations

import unittest

import numpy as np

import eas


class TestDlpackInterop(unittest.TestCase):
    def test_numpy_roundtrip_cpu(self) -> None:
        a = np.random.randn(1024).astype(np.float32)
        t = eas.tensor(a, device="cpu")
        cap = eas.to_dlpack(t)
        self.assertEqual(type(cap).__name__, "PyCapsule")

        t2 = eas.from_dlpack(a)
        np.testing.assert_allclose(t2.numpy(), a)

        # NumPy consumes the DLPack protocol from objects, not raw capsules.
        b = np.from_dlpack(t)
        np.testing.assert_allclose(b, a)

        # A dlpack roundtrip should preserve values.
        t3 = eas.from_dlpack(np.from_dlpack(t.numpy()))
        np.testing.assert_allclose(t3.numpy(), a)

    def test_numpy_roundtrip_cpu_float16(self) -> None:
        a = np.random.randn(1024).astype(np.float16)
        t = eas.tensor(a, device="cpu")
        b = np.from_dlpack(t)
        self.assertEqual(b.dtype, np.float16)
        np.testing.assert_allclose(b, a)
