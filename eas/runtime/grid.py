from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def infer_nthreads(
    runtime_args: Mapping[str, Any], *, nthreads: int | None = None
) -> int:
    """
    Infer total threads (1D) to launch.

    Backward compatible: if `nthreads` is not provided, read a required scalar
    argument named `N` from `runtime_args`.
    """
    if nthreads is None:
        if "N" not in runtime_args:
            raise ValueError(
                "MVP runtime requires either `_nthreads=...` or a scalar argument named 'N' for grid sizing"
            )
        nthreads = int(runtime_args["N"])
    n = int(nthreads)
    if n < 0:
        raise ValueError("nthreads must be >= 0")
    return n


def infer_1d_grid(
    runtime_args: Mapping[str, Any], threadgroup_size: int, *, nthreads: int | None = None
) -> int:
    """
    MVP grid sizing helper: infer grid.x (threadgroup count) for 1D launches.
    """
    n = infer_nthreads(runtime_args, nthreads=nthreads)
    if threadgroup_size <= 0:
        raise ValueError("threadgroup_size must be > 0")
    return (n + threadgroup_size - 1) // threadgroup_size
