# filename: tests/test_dot_codegen_unique_vars.py

from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def many_dot_kernel(out, K: eas.constexpr, BLOCK: eas.constexpr):
    _ = mk.arange(0, BLOCK)
    As = mk.alloc_tg(1 * K)
    Bs = mk.alloc_tg(K * 1)
    mk.barrier()

    acc = 0.0
    # Emit multiple dot ops so codegen groups them and repeats the pattern.
    for _i in range(20):
        acc = acc + mk.dot(As, 0, 1, Bs, 0, 1, K)
    mk.store(out, 0, acc, True)


class TestDotCodegenUniqueVars(unittest.TestCase):
    def test_dot_group_does_not_redeclare_a_idx(self) -> None:
        out = np.zeros(1, dtype=np.float32)
        ck = many_dot_kernel.compile(out, K=16, BLOCK=32)
        self.assertNotIn("uint a_idx =", ck.msl)


if __name__ == "__main__":
    unittest.main()
