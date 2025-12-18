from __future__ import annotations

import re
import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def spmd_add_kernel(a, b, c, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N
    mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)


class TestSpmdCodegen(unittest.TestCase):
    def test_spmd_uses_tpig_and_unrolled_lane_loop(self) -> None:
        n = 1024 + 3
        block = 128
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.zeros_like(a)

        ck = spmd_add_kernel.compile(a, b, c, n, BLOCK=block)
        self.assertEqual(ck.launch_mode, "spmd")
        self.assertEqual(int(ck.block_size), block)

        msl = ck.msl
        self.assertIn("[[thread_position_in_grid]]", msl)
        self.assertNotIn("[[thread_position_in_threadgroup]]", msl)
        self.assertNotIn("[[threadgroup_position_in_grid]]", msl)

        self.assertRegex(msl, re.compile(rf"for \(uint i = 0; i < {block}u; \+\+i\)"))


if __name__ == "__main__":
    unittest.main()
