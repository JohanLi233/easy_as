# filename: eas/runtime/grid.py

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..ir import IRModule


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
    runtime_args: Mapping[str, Any],
    threadgroup_size: int,
    *,
    nthreads: int | None = None,
) -> int:
    """
    MVP grid sizing helper: infer grid.x (threadgroup count) for 1D launches.
    """
    n = infer_nthreads(runtime_args, nthreads=nthreads)
    if threadgroup_size <= 0:
        raise ValueError("threadgroup_size must be > 0")
    return (n + threadgroup_size - 1) // threadgroup_size


def infer_grid(
    ir: IRModule,
    runtime_args: Mapping[str, Any],
    threadgroup_size: int,
    *,
    shape: tuple[int, ...] | None = None,
) -> tuple[int, int, int] | None:
    """
    Infer a 2D/3D threadgroup grid for tiled kernels.

    If the kernel uses `program_id(1)`/`thread_id(1)` (or axis 2), infer:
      - 2D: grid = (ceil_div(W, BLOCK), H, 1)
      - 3D: grid = (ceil_div(W, BLOCK), H, D)

    `BLOCK` is the inferred block size:
      - thread mode: from `mk.tid(0, BLOCK)`
      - spmd mode: from `mk.arange(0, BLOCK)`

    Dimension sources (in priority order):
      1) explicit `shape` runtime option (uses last 2/3 dims)
      2) scalar runtime args named `H`,`W` (and `D` for 3D)

    Returns `None` for 1D-only kernels (no axis >= 1 usage).
    """
    max_axis = -1
    for inst in ir.insts:
        if inst.op in {"program_id", "thread_id"}:
            axis = int(inst.args[0])
            if axis > max_axis:
                max_axis = axis

    required_ndim = max_axis + 1
    if required_ndim <= 1:
        return None
    if required_ndim > 3:
        raise ValueError("MVP runtime only supports 1D/2D/3D grids")
    if threadgroup_size <= 0:
        raise ValueError("threadgroup_size must be > 0")

    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    if shape is not None:
        if len(shape) < required_ndim:
            raise ValueError(
                f"_shape must have >= {required_ndim} dims for this kernel"
            )
        dims = tuple(int(d) for d in shape)
        if any(d < 0 for d in dims):
            raise ValueError("_shape dims must be >= 0")
        if required_ndim == 2:
            h, w = dims[-2], dims[-1]
            d = 1
        else:
            d, h, w = dims[-3], dims[-2], dims[-1]
    else:
        if required_ndim == 2:
            if "H" not in runtime_args or "W" not in runtime_args:
                raise ValueError(
                    "2D grid inference requires either `_shape=(H, W)` or scalar runtime args named 'H' and 'W' "
                    "(or pass `_grid=...` explicitly)"
                )
            h, w = int(runtime_args["H"]), int(runtime_args["W"])
            d = 1
        else:
            if (
                "D" not in runtime_args
                or "H" not in runtime_args
                or "W" not in runtime_args
            ):
                raise ValueError(
                    "3D grid inference requires either `_shape=(D, H, W)` or scalar runtime args named 'D', 'H', and 'W' "
                    "(or pass `_grid=...` explicitly)"
                )
            d, h, w = (
                int(runtime_args["D"]),
                int(runtime_args["H"]),
                int(runtime_args["W"]),
            )

    if h < 0 or w < 0 or d < 0:
        raise ValueError("shape dims must be >= 0")
    gx = _ceil_div(int(w), int(threadgroup_size)) if w else 0
    return (int(gx), int(h), int(d))
