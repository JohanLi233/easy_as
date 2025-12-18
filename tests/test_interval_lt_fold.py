# filename: tests/test_interval_lt_fold.py

from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def _in_tile_store_kernel(src, dst, LIMIT: eas.constexpr, NT: eas.constexpr):
    tid = mk.tid(0, NT)
    for i in range(0, LIMIT, NT):
        idx = tid + i
        in_tile = idx < LIMIT
        mk.store(dst, idx, mk.load(src, idx, True), in_tile)


class TestIntervalLtFold(unittest.TestCase):
    def test_in_tile_lt_folded_when_provably_true(self) -> None:
        limit, nt = 64, 16
        src = np.arange(limit, dtype=np.float32)
        dst = np.zeros_like(src)
        ck = _in_tile_store_kernel.compile(src, dst, LIMIT=limit, NT=nt)
        self.assertNotIn("lt", [inst.op for inst in ck.ir.insts])

    def test_in_tile_lt_kept_when_needed(self) -> None:
        limit, nt = 70, 16
        src = np.arange(limit, dtype=np.float32)
        dst = np.zeros_like(src)
        ck = _in_tile_store_kernel.compile(src, dst, LIMIT=limit, NT=nt)
        self.assertIn("lt", [inst.op for inst in ck.ir.insts])


if __name__ == "__main__":
    unittest.main()
