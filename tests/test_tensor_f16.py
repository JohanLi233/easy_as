# filename: tests/test_tensor_f16.py

from __future__ import annotations

import unittest

import numpy as np

import eas


class TestTensorF16(unittest.TestCase):
    def test_cpu_tensor_preserves_float16(self) -> None:
        a = np.random.randn(128).astype(np.float16)
        t = eas.tensor(a, device="cpu")
        self.assertEqual(t.dtype, np.dtype(np.float16))
        np.testing.assert_array_equal(t.numpy(), a)

    def test_empty_float16(self) -> None:
        t = eas.empty((16, 8), device="cpu", dtype=np.float16)
        self.assertEqual(t.dtype, np.dtype(np.float16))
        self.assertEqual(t.numpy().dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
