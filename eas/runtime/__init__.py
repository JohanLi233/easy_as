from __future__ import annotations

import os
from typing import Literal

from .cpu import CpuRuntime

__all__ = ["get_runtime"]

_cpu = CpuRuntime()
_metal = None


def get_runtime(backend: Literal["auto", "cpu", "metal"] | None = None):
    backend = (backend or os.environ.get("EAS_BACKEND", "auto")).lower()
    if backend == "cpu":
        return _cpu
    if backend == "metal":
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
        f"unsupported backend: {backend!r} (expected 'auto'|'cpu'|'metal')"
    )


def _get_metal():
    global _metal
    if _metal is None:
        from .metal import MetalRuntime

        _metal = MetalRuntime()
    return _metal
