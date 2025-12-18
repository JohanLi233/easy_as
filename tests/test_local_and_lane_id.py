# filename: tests/test_local_and_lane_id.py

from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


@eas.kernel
def local_id_kernel(out, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    _ = mk.tid(0, BLOCK)  # define threadgroup_size for MVP codegen/runtime
    lid = mk.local_id(0)
    offs = pid * BLOCK + lid
    mk.store(out, offs, mk.to_f32(lid), offs < N)


@eas.kernel
def lane_id_kernel(out, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    tid = pid * BLOCK + mk.tid(0, BLOCK)
    mk.store(out, tid, mk.to_f32(mk.lane_id()), tid < N)


class TestLocalAndLaneId(unittest.TestCase):
    def test_msl_declares_hidden_params(self) -> None:
        n = 256
        out = np.empty(n, dtype=np.float32)
        ck0 = local_id_kernel.compile(out, n, BLOCK=256)
        self.assertIn("[[thread_position_in_threadgroup]]", ck0.msl)

        ck1 = lane_id_kernel.compile(out, n, BLOCK=256)
        self.assertIn("[[thread_index_in_simdgroup]]", ck1.msl)

    def test_local_id_layout_on_mps_if_available(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            block = 128
            n = block * 3
            out = np.zeros(n, dtype=np.float32)
            local_id_kernel(out, n, BLOCK=block)
            expected = (np.arange(n, dtype=np.uint32) % block).astype(np.float32)
            np.testing.assert_array_equal(out, expected)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old

    def test_lane_id_cycles_0_31_on_mps_if_available(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            block = 256
            n = block
            out = np.zeros(n, dtype=np.float32)
            lane_id_kernel(out, n, BLOCK=block)
            expected = (np.arange(n, dtype=np.uint32) % 32).astype(np.float32)
            np.testing.assert_array_equal(out, expected)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
