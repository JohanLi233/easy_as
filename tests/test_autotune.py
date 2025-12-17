# filename: tests/test_autotune.py

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import eas
from eas import mk


class _FakeRuntime:
    def __init__(self) -> None:
        self.benchmark_calls: list[dict[str, object]] = []
        self.run_calls: list[dict[str, object]] = []

    def benchmark(  # type: ignore[override]
        self,
        ck: object,
        runtime_args: object,
        meta: object,
        *,
        repeat: int,
        warmup: int = 0,
        **kwargs: object,
    ) -> float:
        _ = ck, runtime_args, repeat, warmup, kwargs
        meta_d = dict(meta)  # type: ignore[arg-type]
        self.benchmark_calls.append(meta_d)
        block = int(meta_d["BLOCK"])
        return 1.0 / float(block)

    def run(  # type: ignore[override]
        self, ck: object, runtime_args: object, meta: object, **kwargs: object
    ) -> None:
        _ = ck, runtime_args, kwargs
        self.run_calls.append(dict(meta))  # type: ignore[arg-type]


@eas.autotune(
    configs=[
        eas.Config(meta={"BLOCK": 64}),
        eas.Config(meta={"BLOCK": 128}),
        eas.Config(meta={"BLOCK": 256}),
    ],
    key=["N"],
    warmup=0,
    repeat=1,
)
@eas.kernel
def add_kernel(a, b, c, N, BLOCK: eas.constexpr = 64):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N
    mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)


class TestAutotune(unittest.TestCase):
    def test_autotune_caches_by_key(self) -> None:
        rt = _FakeRuntime()
        with mock.patch("eas.kernel.get_runtime", return_value=rt):
            n1 = 1024 + 3
            a1 = np.random.randn(n1).astype(np.float32)
            b1 = np.random.randn(n1).astype(np.float32)
            c1 = np.zeros_like(a1)

            add_kernel(a1, b1, c1, n1)
            self.assertEqual(len(rt.benchmark_calls), 3)
            self.assertEqual(rt.run_calls[-1]["BLOCK"], 256)

            add_kernel(a1, b1, c1, n1)
            self.assertEqual(len(rt.benchmark_calls), 3)
            self.assertEqual(rt.run_calls[-1]["BLOCK"], 256)

            n2 = 2048 + 7
            a2 = np.random.randn(n2).astype(np.float32)
            b2 = np.random.randn(n2).astype(np.float32)
            c2 = np.zeros_like(a2)

            add_kernel(a2, b2, c2, n2)
            self.assertEqual(len(rt.benchmark_calls), 6)
            self.assertEqual(rt.run_calls[-1]["BLOCK"], 256)


if __name__ == "__main__":
    unittest.main()
