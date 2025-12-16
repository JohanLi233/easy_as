# filename: eas/runtime/__init__.py

from __future__ import annotations

import os
from typing import Literal, TYPE_CHECKING

from .base import Runtime

__all__ = ["get_runtime"]

_metal = None

if TYPE_CHECKING:  # pragma: no cover
    from .metal import MetalRuntime


def get_runtime(
    backend: Literal["auto", "mps", "metal"] | None = None,
) -> Runtime:
    backend = (backend or os.environ.get("EAS_BACKEND", "auto")).lower()
    if backend == "metal":
        backend = "mps"
    if backend == "mps":
        metal = _get_metal()
        if not metal.is_available():
            raise RuntimeError(
                "MPS runtime is not available. Build the Metal extension with "
                "`uv run python tools/build_metal_ext.py` (macOS + Xcode SDK required)."
            )
        return metal
    if backend == "auto":
        metal = _get_metal()
        if not metal.is_available():
            raise RuntimeError(
                "No available runtime (expected Metal/MPS). Build the Metal extension with "
                "`uv run python tools/build_metal_ext.py` (macOS + Xcode SDK required)."
            )
        return metal
    if backend == "cpu":
        raise ValueError(
            "unsupported backend: 'cpu' (CPU runtime was removed; use 'mps' or 'auto')"
        )
    raise ValueError(f"unsupported backend: {backend!r} (expected 'auto'|'mps')")


def _get_metal() -> "MetalRuntime":
    global _metal
    if _metal is None:
        from .metal import MetalRuntime

        _metal = MetalRuntime()
    return _metal
