# filename: tests/test_metal_copy.py

from __future__ import annotations

import unittest

import numpy as np

from eas.runtime.metal_ext import load_metal_ext


class TestMetalCopy(unittest.TestCase):
    def setUp(self) -> None:
        mod = load_metal_ext(require=False)
        if mod is None or not callable(getattr(mod, "is_available", None)):
            self.skipTest("Metal extension is not available")
        if not mod.is_available():
            self.skipTest("Metal is not available")
        self._mod = mod

    def test_private_copy_roundtrip_numpy(self) -> None:
        mod = self._mod
        n = 4096 + 7
        src = np.random.randn(n).astype(np.float32)
        dst = np.empty_like(src)

        buf = mod.alloc_buffer(int(src.nbytes), "private")
        mod.copy_from_host(buf, src)
        mod.copy_to_host(buf, dst)
        np.testing.assert_allclose(dst, src)

    def test_private_copy_roundtrip_bytearray(self) -> None:
        mod = self._mod
        nbytes = 8192
        src = bytearray((i * 7 + 3) & 0xFF for i in range(nbytes))
        dst = bytearray(nbytes)

        buf = mod.alloc_buffer(int(nbytes), "private")
        mod.copy_from_host(buf, src)
        mod.copy_to_host(buf, dst)
        self.assertEqual(dst, src)
