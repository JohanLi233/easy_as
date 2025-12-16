from __future__ import annotations

import struct
import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

import eas
from eas import mk
from eas.runtime.metal import MetalRuntime


@dataclass
class _FakeBuffer:
    nbytes: int
    storage: str
    data: bytearray


class _FakeMetalExt:
    def __init__(self) -> None:
        self._buffers: list[_FakeBuffer] = []

    def is_available(self) -> bool:
        return True

    def alloc_buffer(self, nbytes: int, storage: str) -> _FakeBuffer:
        buf = _FakeBuffer(
            nbytes=int(nbytes), storage=str(storage), data=bytearray(int(nbytes))
        )
        self._buffers.append(buf)
        return buf

    def copy_from_host(self, buf: _FakeBuffer, src: Any) -> None:
        mv = memoryview(src).cast("B")
        if len(mv) > buf.nbytes:
            raise ValueError("source is larger than destination buffer")
        buf.data[: len(mv)] = mv.tobytes()

    def copy_to_host(self, buf: _FakeBuffer, dst: Any) -> None:
        mv = memoryview(dst).cast("B")
        if len(mv) < buf.nbytes:
            raise ValueError("destination is smaller than source buffer")
        mv[: buf.nbytes] = buf.data

    def compile(self, msl_src: str, name: str) -> object:
        _ = msl_src
        _ = name
        return object()

    def launch_tg(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def launch(
        self, pipeline: object, argv: list[object], writable: list[bool], *args: Any
    ) -> None:
        _ = pipeline
        _ = args
        if len(argv) != 4:
            raise AssertionError(f"expected 4 argv entries, got {len(argv)}")
        a, b, c, n_bytes = argv
        if (
            not isinstance(a, _FakeBuffer)
            or not isinstance(b, _FakeBuffer)
            or not isinstance(c, _FakeBuffer)
        ):
            raise AssertionError("buffer args must be Metal buffer capsules")
        if writable != [False, False, True, False]:
            raise AssertionError(f"unexpected writable flags: {writable!r}")
        if not isinstance(n_bytes, (bytes, bytearray)):
            raise AssertionError("N scalar must be bytes")
        (n,) = struct.unpack("<I", bytes(n_bytes)[:4])

        av = np.frombuffer(a.data, dtype=np.float32, count=n)
        bv = np.frombuffer(b.data, dtype=np.float32, count=n)
        cv = np.frombuffer(c.data, dtype=np.float32, count=n)
        np.copyto(cv, av + bv)


class TestMetalNumpyBridge(unittest.TestCase):
    def test_numpy_inputs_use_buffer_capsules(self) -> None:
        @eas.kernel
        def add_kernel(a, b, c, N, BLOCK: eas.constexpr):
            pid = mk.program_id(0)
            offs = pid * BLOCK + mk.arange(0, BLOCK)
            mask = offs < N
            mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)

        n = 1024 + 7
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.zeros_like(a)

        ck = add_kernel.compile(a, b, c, n, BLOCK=256)
        rt = MetalRuntime()
        rt._metal = _FakeMetalExt()
        rt._available = True

        rt.run(ck, {"a": a, "b": b, "c": c, "N": n}, {"BLOCK": 256}, sync=True)
        np.testing.assert_allclose(c, a + b, rtol=0, atol=0)
