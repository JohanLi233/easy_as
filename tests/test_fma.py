# filename: tests/test_fma.py

from __future__ import annotations

import unittest

import numpy as np

import eas
from eas import mk


@eas.kernel
def simple_fma_kernel(a, b, c, d, N, BLOCK: eas.constexpr):
    """a * b + c -> should become fma(a, b, c)"""
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.tid(0, BLOCK)
    mask = offs < N
    a_val = mk.load(a, offs, mask)
    b_val = mk.load(b, offs, mask)
    c_val = mk.load(c, offs, mask)
    d_val = a_val * b_val + c_val
    mk.store(d, offs, d_val, mask)


@eas.kernel
def matmul_with_fma(a, b, c, M, N, BLOCK: eas.constexpr, K: eas.constexpr):
    """Matmul kernel where each inner iteration should become fma"""
    row = mk.program_id(0)
    tile = mk.program_id(1)
    col = tile * BLOCK + mk.tid(0, BLOCK)
    out = row * N + col
    mask = mk.where(row < M, col < N, False)
    acc = 0.0
    for k in range(K):
        a_off = row * K + k
        b_off = k * N + col
        acc = acc + mk.load(a, a_off, mask) * mk.load(b, b_off, mask)
    mk.store(c, out, acc, mask)


class TestFmaOptimization(unittest.TestCase):
    def test_fma_instruction_generated(self) -> None:
        """Verify that fma instructions are generated for mul+add patterns."""
        n = 256
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.random.randn(n).astype(np.float32)
        d = np.empty_like(a)

        ck = simple_fma_kernel.compile(a, b, c, d, n, BLOCK=256)
        msl = ck.msl

        # Check that fma instruction appears
        self.assertIn("fma(", msl, "fma instruction not found in generated MSL")
        self.assertRegex(
            msl,
            r"\bif \(!v\d+\) return;",
            "expected early-return guard for single-store mask",
        )
        self.assertNotRegex(
            msl,
            r"\bif \(v\d+\) \{",
            "expected no if(mask){...} guard after early-return lowering",
        )

        # Check that the pattern a*b + c doesn't appear as separate mul and add
        # (though there might be other unrelated mul/add instructions)
        lines = msl.split("\n")
        fma_lines = [i for i, line in enumerate(lines) if "fma(" in line]
        self.assertGreater(len(fma_lines), 0, "No fma lines found")

        # Verify numerical correctness
        simple_fma_kernel(a, b, c, d, n, BLOCK=256)
        expected = a * b + c
        np.testing.assert_allclose(d, expected, rtol=1e-5, atol=1e-6)

    def test_matmul_fma_and_lift_optimization(self) -> None:
        """
        Test that matmul generates fma instructions AND lift optimization
        works (no per-load branches).
        """
        m, n, k = 32, 64, 8  # Small sizes
        block = 32
        a2 = np.random.randn(m, k).astype(np.float32)
        b2 = np.random.randn(k, n).astype(np.float32)
        c2 = np.empty((m, n), dtype=np.float32)
        a = a2.reshape(-1)
        b = b2.reshape(-1)
        c = c2.reshape(-1)

        ck = matmul_with_fma.compile(a, b, c, m, n, BLOCK=block, K=k)
        msl = ck.msl

        # 1. Check that fma instructions are generated
        self.assertIn("fma(", msl, "fma instruction not found in matmul MSL")
        self.assertRegex(
            msl,
            r"\bif \(!v\d+\) return;",
            "expected early-return guard for matmul mask",
        )
        self.assertNotRegex(
            msl,
            r"\bif \(v\d+\) \{",
            "expected no if(mask){...} guard after early-return lowering",
        )

        # 2. Check that lift optimization worked (no per-load branches)
        lines = msl.split("\n")
        per_load_branches = 0
        for i, line in enumerate(lines):
            if "float v" in line and "= 0.0f;" in line:
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                if "if (" in next_line and "{" in next_line and "[" in next_line:
                    per_load_branches += 1

        self.assertEqual(
            per_load_branches,
            0,
            f"Found {per_load_branches} per-load branches (lift optimization failed)",
        )

        # 3. Check numerical correctness
        tiles_n = (n + block - 1) // block
        matmul_with_fma(a, b, c, m, n, BLOCK=block, K=k, _grid=(m, tiles_n))
        expected = a2 @ b2
        np.testing.assert_allclose(c2, expected, rtol=1e-4, atol=1e-5)

    def test_fma_only_fuses_when_mul_has_single_user(self) -> None:
        """
        Test that fma fusion only happens when the mul result has only one user.
        This prevents duplicate computation.
        """

        @eas.kernel
        def multi_use_mul_kernel(a, b, c, d, e, N, BLOCK: eas.constexpr):
            pid = mk.program_id(0)
            offs = pid * BLOCK + mk.tid(0, BLOCK)
            mask = offs < N
            a_val = mk.load(a, offs, mask)
            b_val = mk.load(b, offs, mask)
            c_val = mk.load(c, offs, mask)
            mul_result = a_val * b_val  # This has two users
            d_val = mul_result + c_val  # User 1
            # Use where to create another user without requiring subtraction
            e_val = mk.where(mask, mul_result, c_val)  # User 2
            mk.store(d, offs, d_val, mask)
            mk.store(e, offs, e_val, mask)

        n = 128
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.random.randn(n).astype(np.float32)
        d = np.empty_like(a)
        e = np.empty_like(a)

        ck = multi_use_mul_kernel.compile(a, b, c, d, e, n, BLOCK=128)
        msl = ck.msl

        # Should NOT fuse to fma because mul has multiple users
        # Count mul and add instructions
        lines = msl.split("\n")
        mul_count = sum(1 for line in lines if "*" in line and "=" in line)
        add_count = sum(
            1 for line in lines if "+" in line and "=" in line and "fma" not in line
        )
        fma_count = sum(1 for line in lines if "fma(" in line)

        # There should be at least one mul and one add, and no fma
        self.assertGreaterEqual(mul_count, 1, "Expected mul instruction")
        self.assertGreaterEqual(add_count, 1, "Expected add instruction")
        self.assertEqual(
            fma_count, 0, "Should not have fma when mul has multiple users"
        )

        # Verify numerical correctness
        multi_use_mul_kernel(a, b, c, d, e, n, BLOCK=128)
        expected_mul = a * b
        expected_d = expected_mul + c
        expected_e = np.where(np.arange(n) < n, expected_mul, c)  # mask is always true
        np.testing.assert_allclose(d, expected_d, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(e, expected_e, rtol=1e-5, atol=1e-6)

    def test_explicit_fma_function(self) -> None:
        """Test that mk.fma() works correctly."""

        @eas.kernel
        def explicit_fma_kernel(a, b, c, d, N, BLOCK: eas.constexpr):
            pid = mk.program_id(0)
            offs = pid * BLOCK + mk.tid(0, BLOCK)
            mask = offs < N
            a_val = mk.load(a, offs, mask)
            b_val = mk.load(b, offs, mask)
            c_val = mk.load(c, offs, mask)
            d_val = mk.fma(a_val, b_val, c_val)
            mk.store(d, offs, d_val, mask)

        n = 256
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.random.randn(n).astype(np.float32)
        d = np.empty_like(a)

        ck = explicit_fma_kernel.compile(a, b, c, d, n, BLOCK=256)
        msl = ck.msl

        # Should contain fma instruction
        self.assertIn("fma(", msl, "fma instruction not found for explicit mk.fma()")

        # Verify numerical correctness
        explicit_fma_kernel(a, b, c, d, n, BLOCK=256)
        # Note: fma may have different rounding than a*b + c
        # We'll use a slightly larger tolerance
        expected = a * b + c
        np.testing.assert_allclose(d, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
