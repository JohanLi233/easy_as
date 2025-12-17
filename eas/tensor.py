# filename: eas/tensor.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, overload

import os

import numpy as np

from .runtime.metal_ext import load_metal_ext

# Public API: use "mps" (Apple GPU) rather than the implementation detail "metal".
Device = Literal["cpu", "mps"]


def _torch() -> Any | None:
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _is_torch_tensor(x: Any) -> bool:
    torch = _torch()
    if torch is None:
        return False
    return isinstance(x, torch.Tensor)


def _normalize_device(device: str | None) -> Device:
    if device is None:
        return "cpu"
    device_l = str(device).lower()
    if device_l == "metal":
        device_l = "mps"
    if device_l not in ("cpu", "mps"):
        raise ValueError(f"unsupported device: {device!r} (expected 'cpu'|'mps')")
    return device_l  # type: ignore[return-value]


def _require_supported_dtype(dtype: np.dtype) -> np.dtype:
    dtype_n = np.dtype(dtype)
    if dtype_n not in (np.float16, np.float32):
        raise TypeError(f"only float16/float32 tensors are supported (got {dtype_n})")
    return dtype_n


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        if d < 0:
            raise ValueError("negative dimensions are not allowed")
        n *= int(d)
    return int(n)


@dataclass(frozen=True, slots=True)
class _MetalStorage:
    buf: object
    nbytes: int
    storage: Literal["private", "shared"]


class Tensor:
    __eas_tensor__ = True

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: np.dtype,
        device: Device,
        cpu: np.ndarray | None = None,
        metal: _MetalStorage | None = None,
    ) -> None:
        dtype = _require_supported_dtype(dtype)
        self._shape = tuple(int(d) for d in shape)
        self._dtype = dtype
        self._device: Device = device
        self._cpu = cpu
        self._metal = metal

        if self._device == "cpu":
            if self._cpu is None:
                raise ValueError("cpu tensor requires cpu storage")
            if self._metal is not None:
                raise ValueError("cpu tensor cannot have metal storage")
        else:
            if self._metal is None:
                raise ValueError("mps tensor requires metal storage")
            if self._cpu is not None:
                raise ValueError("mps tensor cannot have cpu storage")

    @property
    def device(self) -> Device:
        return self._device

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def nbytes(self) -> int:
        return _numel(self._shape) * int(self._dtype.itemsize)

    def __repr__(self) -> str:  # pragma: no cover
        return f"eas.Tensor(shape={self._shape}, dtype={self._dtype.name}, device={self._device!r})"

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        if self._device != "cpu":
            raise TypeError("cannot convert a mps tensor to numpy without .numpy()")
        arr = self._cpu
        assert arr is not None
        if dtype is None:
            return arr
        return arr.astype(dtype, copy=False)

    def __dlpack__(self, stream: Any | None = None) -> Any:
        _ = stream
        if self._device == "cpu":
            arr = self._cpu
            assert arr is not None
            return arr.__dlpack__()
        metal = self._metal
        assert metal is not None
        mod = load_metal_ext(require=True)
        if not callable(getattr(mod, "dlpack_export", None)):
            raise RuntimeError(
                "Metal extension `eas._metal` is missing DLPack export API; rebuild it with "
                "`uv run python tools/build_metal_ext.py`."
            )
        dtype_bits = int(self._dtype.itemsize) * 8
        return mod.dlpack_export(metal.buf, self._shape, int(dtype_bits))

    def __dlpack_device__(self) -> tuple[int, int]:
        if self._device == "cpu":
            # DLPack DLDeviceType.kDLCPU = 1
            return (1, 0)
        # DLPack DLDeviceType.kDLMetal = 8
        return (8, 0)

    def numpy(self) -> np.ndarray:
        if self._device == "cpu":
            arr = self._cpu
            assert arr is not None
            return arr
        metal = self._metal
        assert metal is not None
        mod = load_metal_ext(require=True)
        if not callable(getattr(mod, "copy_to_host", None)):
            raise RuntimeError(
                "Metal extension `eas._metal` is missing tensor copy API; rebuild it with "
                "`uv run python tools/build_metal_ext.py`."
            )
        out = np.empty(self._shape, dtype=self._dtype)
        mod.copy_to_host(metal.buf, out)
        return out

    def to_torch(self, device: str | None = None) -> Any:
        torch = _torch()
        if torch is None:
            raise RuntimeError(
                "torch is not available; install it to use Tensor.to_torch()"
            )

        if device is None:
            device = "cpu" if self._device == "cpu" else "mps"
        device_l = str(device).lower()
        if device_l not in ("cpu", "mps"):
            raise ValueError("device must be 'cpu' or 'mps' for torch interop")

        if device_l == "cpu":
            return torch.from_numpy(self.numpy())

        if self._device == "mps":
            from torch.utils import dlpack  # type: ignore

            return dlpack.from_dlpack(self)

        return torch.from_numpy(self.numpy()).to("mps")

    def to(self, device: Device | str) -> Tensor:
        device_n = _normalize_device(device)
        if device_n == self._device:
            return self
        if device_n == "cpu":
            return Tensor(
                shape=self._shape, dtype=self._dtype, device="cpu", cpu=self.numpy()
            )

        # cpu -> mps (Metal backend)
        mod = load_metal_ext(require=True)
        if not callable(getattr(mod, "alloc_buffer", None)) or not callable(
            getattr(mod, "copy_from_host", None)
        ):
            raise RuntimeError(
                "Metal extension `eas._metal` is missing tensor allocation/copy API; rebuild it with "
                "`uv run python tools/build_metal_ext.py`."
            )
        if not callable(getattr(mod, "is_available", None)) or not bool(
            mod.is_available()
        ):
            raise RuntimeError("MPS device is not available")

        buf = mod.alloc_buffer(int(self.nbytes), "private")
        arr = self._cpu
        assert arr is not None
        mod.copy_from_host(buf, arr)
        return Tensor(
            shape=self._shape,
            dtype=self._dtype,
            device="mps",
            metal=_MetalStorage(buf=buf, nbytes=self.nbytes, storage="private"),
        )

    def _metal_buffer(self) -> object:
        if self._device not in ("mps", "metal"):
            raise TypeError("expected an mps tensor")
        metal = self._metal
        assert metal is not None
        return metal.buf


@overload
def tensor(
    data: Tensor, *, device: Device | str | None = None, dtype: np.dtype | None = None
) -> Tensor: ...


@overload
def tensor(
    data: Any, *, device: Device | str | None = None, dtype: np.dtype | None = None
) -> Tensor: ...


def tensor(
    data: Any, *, device: Device | str | None = None, dtype: np.dtype | None = None
) -> Tensor:
    device_n = _normalize_device(device)
    if _is_torch_tensor(data):
        torch = _torch()
        assert torch is not None

        t = data.detach() if getattr(data, "requires_grad", False) else data
        if dtype is not None:
            dtype_n = _require_supported_dtype(np.dtype(dtype))
            torch_dtype = torch.float16 if dtype_n == np.float16 else torch.float32
            if t.dtype != torch_dtype:
                t = t.to(dtype=torch_dtype)
        else:
            if t.dtype not in (torch.float16, torch.float32):
                t = t.to(dtype=torch.float32)

        if device_n == "mps" and t.device.type == "mps":
            if os.environ.get("EAS_TORCH_MPS_SYNC", "1").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            ):
                torch.mps.synchronize()
            if not t.is_contiguous():
                t = t.contiguous()
            mod = load_metal_ext(require=True)
            if not callable(getattr(mod, "dlpack_import", None)):
                raise RuntimeError(
                    "Metal extension `eas._metal` is missing DLPack import API; rebuild it with "
                    "`uv run python tools/build_metal_ext.py`."
                )
            cap = t.__dlpack__()
            buf, shape, dtype_bits = mod.dlpack_import(cap)
            shape_t = tuple(int(d) for d in shape)
            bits_i = int(dtype_bits)
            if bits_i == 16:
                dtype_np = np.float16
            elif bits_i == 32:
                dtype_np = np.float32
            else:
                raise TypeError(f"unsupported dlpack dtype bits: {dtype_bits}")
            nbytes = _numel(shape_t) * int(np.dtype(dtype_np).itemsize)
            return Tensor(
                shape=shape_t,
                dtype=dtype_np,
                device="mps",
                metal=_MetalStorage(buf=buf, nbytes=nbytes, storage="private"),
            )

        if t.device.type != "cpu":
            t = t.to("cpu")
        cpu_arr = t.contiguous().numpy()
        cpu_t = Tensor(
            shape=tuple(cpu_arr.shape), dtype=cpu_arr.dtype, device="cpu", cpu=cpu_arr
        )
        return cpu_t.to(device_n)

    if isinstance(data, Tensor):
        out = data
        if dtype is not None and np.dtype(dtype) != out.dtype:
            if out.device != "cpu":
                out = out.to("cpu")
            out = Tensor(
                shape=out.shape,
                dtype=_require_supported_dtype(np.dtype(dtype)),
                device="cpu",
                cpu=out.numpy().astype(dtype, copy=True),
            )
        return out.to(device_n)

    if dtype is None:
        arr = np.asarray(data)
        if np.dtype(arr.dtype) not in (np.float16, np.float32):
            arr = arr.astype(np.float32)
    else:
        dtype_n = _require_supported_dtype(np.dtype(dtype))
        arr = np.asarray(data, dtype=dtype_n)
    _ = _require_supported_dtype(arr.dtype)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    cpu_t = Tensor(shape=arr.shape, dtype=arr.dtype, device="cpu", cpu=arr)
    return cpu_t.to(device_n)


def empty(
    shape: int | tuple[int, ...],
    *,
    device: Device | str | None = None,
    dtype: np.dtype | None = None,
) -> Tensor:
    device_n = _normalize_device(device)
    dtype_n = _require_supported_dtype(np.dtype(np.float32 if dtype is None else dtype))
    shape_t = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)

    if device_n == "cpu":
        arr = np.empty(shape_t, dtype=dtype_n)
        return Tensor(shape=shape_t, dtype=dtype_n, device="cpu", cpu=arr)

    mod = load_metal_ext(require=True)
    if not callable(getattr(mod, "alloc_buffer", None)):
        raise RuntimeError(
            "Metal extension `eas._metal` is missing tensor allocation API; rebuild it with "
            "`uv run python tools/build_metal_ext.py`."
        )
    if not callable(getattr(mod, "is_available", None)) or not bool(mod.is_available()):
        raise RuntimeError("MPS device is not available")
    nbytes = _numel(shape_t) * int(dtype_n.itemsize)
    buf = mod.alloc_buffer(int(nbytes), "private")
    return Tensor(
        shape=shape_t,
        dtype=dtype_n,
        device="mps",
        metal=_MetalStorage(buf=buf, nbytes=nbytes, storage="private"),
    )


def empty_like(x: Tensor, *, device: Device | str | None = None) -> Tensor:
    device_n = x.device if device is None else _normalize_device(device)
    return empty(x.shape, device=device_n, dtype=x.dtype)
