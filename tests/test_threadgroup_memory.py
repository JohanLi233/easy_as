# filename: tests/test_threadgroup_memory.py

from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


@eas.kernel
def tg_copy_kernel(a, out, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    tid = mk.tid(0, BLOCK)
    offs = pid * BLOCK + tid
    mask = offs < N

    scratch = mk.alloc_tg(BLOCK)
    x = mk.load(a, offs, mask)
    mk.store(scratch, tid, x)
    mk.barrier()
    y = mk.load(scratch, tid)
    mk.store(out, offs, y, mask)


@eas.kernel
def tg_copy_kernel_masked(a, out, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    tid = mk.tid(0, BLOCK)
    offs = pid * BLOCK + tid
    mask = offs < N

    scratch = mk.alloc_tg(BLOCK)
    # Fill scratch for all lanes (masked-out lanes write 0.0 via mk.load).
    mk.store(scratch, tid, mk.load(a, offs, mask))
    mk.barrier()
    mk.store(out, offs, mk.load(scratch, tid), mask)


class TestThreadgroupMemory(unittest.TestCase):
    def test_msl_contains_threadgroup_alloc_and_barrier(self) -> None:
        n = 256 + 3
        a = np.random.randn(n).astype(np.float32)
        out = np.empty_like(a)
        ck = tg_copy_kernel.compile(a, out, n, BLOCK=256)
        msl = ck.msl

        self.assertRegex(msl, r"\bthreadgroup float v\d+\[\d+\];")
        self.assertIn(
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            msl,
            "threadgroup barrier not found in generated MSL",
        )
        self.assertEqual(ck.writes, frozenset({"out"}))

    def test_barrier_not_in_mask_guard(self) -> None:
        n = 256 * 2 + 3
        a = np.random.randn(n).astype(np.float32)
        out = np.empty_like(a)
        ck = tg_copy_kernel_masked.compile(a, out, n, BLOCK=256)
        msl_lines = ck.msl.splitlines()

        # Barrier must not be lifted into an indented (mask) guard.
        self.assertTrue(
            any(line.startswith("  threadgroup_barrier(") for line in msl_lines),
            "expected a top-level threadgroup_barrier",
        )
        self.assertFalse(
            any(line.startswith("    threadgroup_barrier(") for line in msl_lines),
            "barrier must not be inside a mask-guarded block",
        )

    def test_correctness_on_mps_if_available(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            n = 4096 + 7
            a = np.random.randn(n).astype(np.float32)
            out = np.zeros_like(a)
            tg_copy_kernel(a, out, n, BLOCK=256)
            np.testing.assert_allclose(out, a, rtol=0, atol=0)
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
