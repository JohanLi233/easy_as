from __future__ import annotations

import struct
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..ir import DType, IRModule
from .metal_ext import load_metal_ext

def _pack_scalar(dtype: DType, value: Any) -> bytes:
    if dtype == DType.U32:
        return struct.pack("<I", int(value))
    if dtype == DType.F32:
        return struct.pack("<f", float(value))
    if dtype == DType.BOOL:
        return struct.pack("<?", bool(value))
    raise ValueError(f"unsupported scalar dtype: {dtype}")


@dataclass(slots=True)
class MetalRuntime:
    _metal: Any | None = None
    _pipeline_cache: dict[tuple[str, str], Any] | None = None
    _available: bool | None = None
    _pending: deque[Any] | None = None
    _max_in_flight: int | None = None

    def _get_max_in_flight(self) -> int:
        v = self._max_in_flight
        if v is not None:
            return v
        env = os.environ.get("EAS_MAX_IN_FLIGHT", "").strip()
        if env:
            try:
                v = int(env)
            except ValueError:
                v = 256
        else:
            v = 256
        if v < 0:
            v = 0
        self._max_in_flight = v
        return v

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        mod = load_metal_ext(require=False)
        if mod is None:
            self._available = False
            return False
        try:
            self._available = bool(mod.is_available())
        except Exception:
            self._available = False
        return self._available

    def _mod(self) -> Any:
        if self._metal is None:
            self._metal = load_metal_ext(require=True)
        return self._metal

    def _get_pipeline(self, ck: Any) -> Any:
        if self._pipeline_cache is None:
            self._pipeline_cache = {}
        key = (ck.msl, ck.ir.name)
        cached = self._pipeline_cache.get(key)
        if cached is not None:
            return cached
        mod = self._mod()
        pipeline = mod.compile(ck.msl, ck.ir.name)
        self._pipeline_cache[key] = pipeline
        return pipeline

    def run(
        self,
        ck: Any,
        runtime_args: Mapping[str, Any],
        meta: Mapping[str, Any],
        *,
        sync: bool = True,
    ) -> None:
        _ = meta
        ir: IRModule = ck.ir
        threadgroup_size: int = ck.threadgroup_size

        if "N" not in runtime_args:
            raise ValueError("MVP metal runtime requires a scalar argument named 'N'")
        n = int(runtime_args["N"])
        grid = (n + threadgroup_size - 1) // threadgroup_size
        writes = ck.writes

        argv: list[object] = []
        writable: list[bool] = []
        for arg in ir.args:
            v = runtime_args[arg.name]
            if arg.kind == "buffer":
                if not isinstance(v, np.ndarray):
                    raise TypeError(
                        f"expected numpy.ndarray for {arg.name!r}, got {type(v)!r}"
                    )
                if v.dtype != np.float32:
                    raise TypeError(f"{arg.name!r} must be float32, got {v.dtype}")
                if not v.flags.c_contiguous:
                    raise ValueError(f"{arg.name!r} must be C-contiguous for MVP")
                argv.append(v)
                writable.append(arg.name in writes)
            else:
                argv.append(_pack_scalar(arg.dtype, v))
                writable.append(False)

        pipeline = self._get_pipeline(ck)
        mod = self._mod()
        if sync or not callable(getattr(mod, "launch_async", None)):
            mod.launch(pipeline, argv, writable, int(grid), int(threadgroup_size))
            return

        pending = mod.launch_async(
            pipeline, argv, writable, int(grid), int(threadgroup_size)
        )
        if self._pending is None:
            self._pending = deque()
        self._pending.append(pending)
        max_in_flight = self._get_max_in_flight()
        if max_in_flight and len(self._pending) > max_in_flight:
            mod.synchronize(self._pending.popleft())

    def synchronize(self) -> None:
        pending = self._pending
        if not pending:
            return
        mod = self._mod()
        while pending:
            mod.synchronize(pending.popleft())
