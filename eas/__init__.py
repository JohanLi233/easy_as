# filename: eas/__init__.py

from __future__ import annotations

from .kernel import Kernel, kernel
from .meta import constexpr
from .dlpack import from_dlpack, to_dlpack
from .tensor import Tensor, empty, empty_like, tensor
from .torch import from_torch, to_torch

# Public DSL module (similar to `triton.language as tl`)
from . import mk  # noqa: F401

__all__ = [
    "Kernel",
    "Tensor",
    "constexpr",
    "from_dlpack",
    "empty",
    "empty_like",
    "from_torch",
    "kernel",
    "mk",
    "tensor",
    "to_dlpack",
    "to_torch",
]
