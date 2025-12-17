# filename: eas/dlpack.py

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .tensor import Tensor, _MetalStorage, _normalize_device, tensor as _tensor
from .runtime.metal_ext import load_metal_ext


Device = Literal["cpu", "mps"]


def from_dlpack(x: Any, *, device: Device | str | None = None) -> Tensor:
    """
    Create an `eas.Tensor` from a DLPack-exporting object.

    Notes
    - CPU: zero-copy is possible via `numpy.from_dlpack(...)`.
    - MPS: supported when the producer exports a `kDLMetal` DLPack tensor (e.g. torch(mps)).
    """
    device_n = _normalize_device(device)

    if device_n == "mps":
        mod = load_metal_ext(require=True)
        capsule = x
        if type(capsule).__name__ != "PyCapsule":
            dlpack_fn = getattr(capsule, "__dlpack__", None)
            if not callable(dlpack_fn):
                raise TypeError(
                    "object does not implement the DLPack protocol (__dlpack__)"
                )
            capsule = dlpack_fn()
        buf, shape, dtype_bits = mod.dlpack_import(capsule)
        shape_t = tuple(int(d) for d in shape)
        bits_i = int(dtype_bits)
        if bits_i == 16:
            dtype_np = np.float16
        elif bits_i == 32:
            dtype_np = np.float32
        else:
            raise TypeError(f"unsupported dlpack dtype bits: {dtype_bits}")
        nbytes = int(np.prod(shape_t, dtype=np.int64)) * int(
            np.dtype(dtype_np).itemsize
        )
        return Tensor(
            shape=shape_t,
            dtype=dtype_np,
            device="mps",
            metal=_MetalStorage(buf=buf, nbytes=nbytes, storage="private"),
        )

    # CPU: prefer NumPy's native DLPack import.
    try:
        arr = np.from_dlpack(x)
    except Exception as e:
        raise TypeError(
            "object does not support DLPack import via numpy.from_dlpack"
        ) from e

    if arr.dtype not in (np.float16, np.float32):
        raise TypeError("only float16/float32 tensors are supported")
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return _tensor(arr, device=device_n)


def to_dlpack(x: Tensor) -> Any:
    """
    Export an `eas.Tensor` as a DLPack capsule.

    Notes
    - Consumers like `numpy.from_dlpack(...)` expect an object implementing the DLPack
      protocol (i.e. `__dlpack__`), so prefer passing the tensor object directly.
    - This function returns a raw `PyCapsule` (torch-style), useful for
      `torch.utils.dlpack.from_dlpack(...)`.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"to_dlpack expects eas.Tensor, got {type(x)!r}")
    return x.__dlpack__()
