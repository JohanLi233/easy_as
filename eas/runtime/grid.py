from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def infer_1d_grid(runtime_args: Mapping[str, Any], threadgroup_size: int) -> int:
    """
    MVP grid sizing helper: infer grid.x from a required scalar argument named `N`.
    """
    if "N" not in runtime_args:
        raise ValueError(
            "MVP runtime requires a scalar argument named 'N' for grid sizing"
        )
    n = int(runtime_args["N"])
    if threadgroup_size <= 0:
        raise ValueError("threadgroup_size must be > 0")
    return (n + threadgroup_size - 1) // threadgroup_size
