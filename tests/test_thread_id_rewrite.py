# filename: tests/test_thread_id_rewrite.py

from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.compiler import _rewrite_thread_id
from eas.ir import DType, IRModule, Inst, ValueRef


@eas.kernel
def _simple_add_kernel(a, b, c, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.tid(0, BLOCK)
    mask = offs < N
    mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)


class TestThreadIdRewrite(unittest.TestCase):
    def test_thread_id_rewrite_disabled_via_env(self) -> None:
        old = os.environ.get("EAS_DISABLE_THREAD_ID_REWRITE")
        os.environ["EAS_DISABLE_THREAD_ID_REWRITE"] = "1"
        try:
            n = 256
            a = np.random.randn(n).astype(np.float32)
            b = np.random.randn(n).astype(np.float32)
            c = np.empty_like(a)

            ck = _simple_add_kernel.compile(a, b, c, n, BLOCK=256)
            msl = ck.msl
            self.assertIn("[[threadgroup_position_in_grid]]", msl)
            self.assertIn("[[thread_position_in_threadgroup]]", msl)
            self.assertNotIn("[[thread_position_in_grid]]", msl)
        finally:
            if old is None:
                os.environ.pop("EAS_DISABLE_THREAD_ID_REWRITE", None)
            else:
                os.environ["EAS_DISABLE_THREAD_ID_REWRITE"] = old

    def test_thread_id_rewrite_requires_single_threadgroup_size(self) -> None:
        pid = ValueRef(1, DType.U32)
        block = ValueRef(2, DType.U32)
        mul = ValueRef(3, DType.U32)
        ar0 = ValueRef(4, DType.U32)
        ar1 = ValueRef(5, DType.U32)
        out = ValueRef(6, DType.U32)

        ir_multi = IRModule(
            name="k",
            args=(),
            insts=(
                Inst("program_id", pid, (0,)),
                Inst("const", block, (256,)),
                Inst("mul", mul, (pid, block)),
                Inst("tid", ar0, (0, 256)),
                Inst("tid", ar1, (0, 128)),
                Inst("add", out, (mul, ar0)),
            ),
        )
        rewritten_multi = _rewrite_thread_id(ir_multi)
        self.assertEqual(
            rewritten_multi.insts[-1].op,
            "add",
            "rewrite must be disabled when threadgroup size is ambiguous",
        )

        ir_single = IRModule(
            name="k",
            args=(),
            insts=(
                Inst("program_id", pid, (0,)),
                Inst("const", block, (256,)),
                Inst("mul", mul, (pid, block)),
                Inst("tid", ar0, (0, 256)),
                Inst("add", out, (mul, ar0)),
            ),
        )
        rewritten_single = _rewrite_thread_id(ir_single)
        self.assertEqual(rewritten_single.insts[-1].op, "thread_id")


if __name__ == "__main__":
    unittest.main()
