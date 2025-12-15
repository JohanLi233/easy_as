from __future__ import annotations

import struct
import os
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..ir import DType, IRModule
from .metal_ext import load_metal_ext
from .grid import infer_1d_grid


def _is_tensor(v: Any) -> bool:
    return bool(getattr(v, "__eas_tensor__", False))


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
    _torch_mod: Any | None = None
    _torch_checked: bool = False
    _dlpack_buffer_cache: dict[int, tuple[weakref.ref[Any], Any]] | None = None

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

    def _get_torch(self) -> Any | None:
        if self._torch_checked:
            return self._torch_mod
        self._torch_checked = True
        try:
            import torch  # type: ignore

            self._torch_mod = torch
        except ImportError:
            self._torch_mod = None
        return self._torch_mod

    def _torch_mps_tensor_to_metal_buffer(self, t: Any) -> Any:
        cache = self._dlpack_buffer_cache
        if cache is None:
            cache = {}
            self._dlpack_buffer_cache = cache

        key = id(t)
        cached = cache.get(key)
        if cached is not None:
            ref, buf = cached
            if ref() is t:
                return buf
            cache.pop(key, None)

        cap = t.__dlpack__()
        buf, _shape = self._mod().dlpack_import(cap)
        try:
            weakref.finalize(t, cache.pop, key, None)
            cache[key] = (weakref.ref(t), buf)
        except TypeError:
            pass
        return buf

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

        grid = infer_1d_grid(runtime_args, threadgroup_size)
        writes = ck.writes

        argv: list[object] = []
        writable: list[bool] = []
        torch_mod = self._get_torch()
        torch_mps_sync_enabled = os.environ.get(
            "EAS_TORCH_MPS_SYNC", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        torch_mps_synced = False

        for arg in ir.args:
            v = runtime_args[arg.name]
            if arg.kind == "buffer":
                if torch_mod is not None and isinstance(v, torch_mod.Tensor):
                    t = v.detach() if getattr(v, "requires_grad", False) else v
                    if t.dtype != torch_mod.float32:
                        raise TypeError(f"{arg.name!r} must be float32, got {t.dtype}")
                    if t.device.type == "mps":
                        if torch_mps_sync_enabled and not torch_mps_synced:
                            torch_mod.mps.synchronize()
                            torch_mps_synced = True
                        if not t.is_contiguous():
                            t = t.contiguous()
                        argv.append(self._torch_mps_tensor_to_metal_buffer(t))
                        writable.append(arg.name in writes)
                        continue
                    if t.device.type == "cpu":
                        v = t.contiguous().numpy()
                    else:
                        raise TypeError(
                            f"unsupported torch device for {arg.name!r}: {t.device!s}"
                        )
                if _is_tensor(v):
                    if getattr(v, "device", None) == "metal":
                        argv.append(v._metal_buffer())  # type: ignore[attr-defined]
                        writable.append(arg.name in writes)
                        continue
                    v = v.numpy()  # type: ignore[assignment]
                if not isinstance(v, np.ndarray):
                    raise TypeError(
                        f"expected numpy.ndarray or eas.Tensor for {arg.name!r}, got {type(v)!r}"
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
