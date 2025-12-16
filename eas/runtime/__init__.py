from __future__ import annotations

import os
from typing import Literal, TYPE_CHECKING

from .cpu import CpuRuntime
from .base import Runtime

__all__ = ["get_runtime"]

_cpu = CpuRuntime()
_metal = None

if TYPE_CHECKING:  # pragma: no cover
    from .metal import MetalRuntime


def get_runtime(backend: Literal["auto", "cpu", "mps", "metal"] | None = None) -> Runtime:
    backend = (backend or os.environ.get("EAS_BACKEND", "auto")).lower()
    if backend == "metal":
        backend = "mps"
    if backend == "cpu":
        return _cpu
    if backend == "mps":
        return _get_metal()
    if backend == "auto":
        try:
            metal = _get_metal()
            if metal.is_available():
                return metal
        except Exception:
            pass
        return _cpu
    raise ValueError(
        f"unsupported backend: {backend!r} (expected 'auto'|'cpu'|'mps')"
    )


def _get_metal() -> "MetalRuntime":
    global _metal
    if _metal is None:
        from .metal import MetalRuntime

        _metal = MetalRuntime()
    return _metal
