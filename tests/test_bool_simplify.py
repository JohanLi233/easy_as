# filename: tests/test_bool_simplify.py

from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def bool_simplify_kernel(out, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    tid = pid * BLOCK + mk.arange(0, BLOCK)
    c0 = tid < N
    c1 = tid < (N + 1)
    m = mk.where(c0, c1, False)
    mk.store(out, tid, mk.to_f32(m), tid < N)


class TestBoolSimplify(unittest.TestCase):
    def test_where_to_and_emits_logical_and(self) -> None:
        out = np.zeros(256, dtype=np.float32)
        ck = bool_simplify_kernel.compile(out, 128, BLOCK=256)
        self.assertIn("&&", ck.msl)


if __name__ == "__main__":
    unittest.main()
