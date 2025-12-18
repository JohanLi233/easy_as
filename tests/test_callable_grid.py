# filename: tests/test_callable_grid.py

from __future__ import annotations

import os
import unittest

import numpy as np

import eas
from eas import mk
from eas.runtime import get_runtime


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@eas.kernel
def grid_callable_kernel(out, H, W, BLOCK: eas.constexpr):
    tile = mk.program_id(0)
    row = mk.program_id(1)
    col = tile * BLOCK + mk.tid(0, BLOCK)
    off = row * W + col
    mask = mk.where(row < H, col < W, False)
    mk.store(out, off, mk.to_f32(row), mask)


class TestCallableGrid(unittest.TestCase):
    def test_grid_callable_on_mps_if_available(self) -> None:
        rt = get_runtime("mps")
        if not rt.is_available():
            self.skipTest("MPS is not available")

        old = os.environ.get("EAS_BACKEND")
        os.environ["EAS_BACKEND"] = "mps"
        try:
            h, w, block = 5, 10, 4
            out = np.zeros(h * w, dtype=np.float32)
            grid_callable_kernel(
                out,
                h,
                w,
                BLOCK=block,
                _grid=lambda meta, args: (
                    _cdiv(int(args["W"]), int(meta["BLOCK"])),
                    int(args["H"]),
                    1,
                ),
            )
            expected = np.repeat(np.arange(h, dtype=np.float32), w)
            np.testing.assert_array_equal(out.reshape(h, w), expected.reshape(h, w))
        finally:
            if old is None:
                os.environ.pop("EAS_BACKEND", None)
            else:
                os.environ["EAS_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
