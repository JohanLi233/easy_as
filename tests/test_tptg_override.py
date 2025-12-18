# filename: tests/test_tptg_override.py

from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


@eas.kernel
def tptg_independent_kernel(out, N, BLOCK: eas.constexpr):
    # Define threadgroup_size for compilation, but rely on the thread_id rewrite
    # so indexing is independent of threads_per_threadgroup.
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.tid(0, BLOCK)
    mk.store(out, offs, mk.to_f32(offs), offs < N)


class TestTptgOverride(unittest.TestCase):
    def test_can_override_tptg_on_mps_if_available(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old_backend = os.environ.get("EAS_BACKEND")
        old_rewrite = os.environ.get("EAS_DISABLE_THREAD_ID_REWRITE")
        os.environ["EAS_BACKEND"] = "mps"
        os.environ["EAS_DISABLE_THREAD_ID_REWRITE"] = "0"
        try:
            n = 1024 + 7
            out0 = np.zeros(n, dtype=np.float32)
            out1 = np.zeros(n, dtype=np.float32)

            with self.assertRaises(ValueError):
                tptg_independent_kernel(out0, n, BLOCK=256, _tptg=64)
            with self.assertRaises(ValueError):
                tptg_independent_kernel(out1, n, BLOCK=256, _tptg=(128, 1, 1))

            tptg_independent_kernel(out0, n, BLOCK=256, _tptg=256)
            tptg_independent_kernel(out1, n, BLOCK=256, _tptg=(256, 1, 1))

            expected = np.arange(n, dtype=np.float32)
            np.testing.assert_array_equal(out0, expected)
            np.testing.assert_array_equal(out1, expected)
        finally:
            if old_backend is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old_backend
            if old_rewrite is None:
                os.environ.pop("EAS_DISABLE_THREAD_ID_REWRITE", None)
            else:
                os.environ["EAS_DISABLE_THREAD_ID_REWRITE"] = old_rewrite


if __name__ == "__main__":
    unittest.main()
