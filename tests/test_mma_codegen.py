# filename: tests/test_mma_codegen.py

from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def mma_codegen_kernel(a, b, c, BLOCK: eas.constexpr):
    tid = mk.arange(0, BLOCK)

    As = mk.alloc_tg(8 * 8)
    Bs = mk.alloc_tg(8 * 8)
    Cs = mk.alloc_tg(8 * 8)

    acc = mk.mma_zero()
    acc = mk.mma(As, 0, 8, Bs, 0, 8, acc)
    mk.mma_store(Cs, 0, 8, acc)
    mk.barrier()
    mk.store(c, tid, mk.load(Cs, tid, True), tid < 64)


class TestMmaCodegen(unittest.TestCase):
    def test_mma_emits_simdgroup_matrix_msl(self) -> None:
        a = np.zeros(1, dtype=np.float32)
        b = np.zeros(1, dtype=np.float32)
        c = np.zeros(64, dtype=np.float32)
        ck = mma_codegen_kernel.compile(a, b, c, BLOCK=32)
        self.assertIn("#include <metal_simdgroup_matrix>", ck.msl)
        self.assertIn("[[thread_index_in_simdgroup]]", ck.msl)
        self.assertIn("make_filled_simdgroup_matrix", ck.msl)
        self.assertIn("simdgroup_load(", ck.msl)
        self.assertIn("simdgroup_multiply_accumulate", ck.msl)
        self.assertIn("simdgroup_store(", ck.msl)


if __name__ == "__main__":
    unittest.main()
