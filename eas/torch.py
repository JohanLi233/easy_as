from __future__ import annotations

from typing import Any

from .tensor import Tensor, tensor as _tensor


def is_available() -> bool:
    try:
        import torch  # type: ignore

        _ = torch
        return True
    except Exception:
        return False


def from_torch(x: Any, *, device: str | None = None) -> Tensor:
    return _tensor(x, device=device)


def to_torch(x: Tensor, *, device: str | None = None) -> Any:
    return x.to_torch(device=device)

