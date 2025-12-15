from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, overload

import os

import numpy as np

from .runtime.metal_ext import load_metal_ext

Device = Literal["cpu", "metal"]


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
    device_l = device.lower()
    if device_l not in ("cpu", "metal"):
        raise ValueError(f"unsupported device: {device!r} (expected 'cpu'|'metal')")
    return device_l  # type: ignore[return-value]


def _require_float32(dtype: np.dtype) -> None:
    if np.dtype(dtype) != np.float32:
        raise TypeError(f"only float32 tensors are supported in MVP (got {dtype})")


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
        dtype = np.dtype(dtype)
        _require_float32(dtype)
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
                raise ValueError("metal tensor requires metal storage")
            if self._cpu is not None:
                raise ValueError("metal tensor cannot have cpu storage")

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
            raise TypeError("cannot convert a metal tensor to numpy without .numpy()")
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
                "`python3 tools/build_metal_ext.py`."
            )
        return mod.dlpack_export(metal.buf, self._shape)

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
                "`python3 tools/build_metal_ext.py`."
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

        if self._device == "metal":
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

        # cpu -> metal
        mod = load_metal_ext(require=True)
        if not callable(getattr(mod, "alloc_buffer", None)) or not callable(
            getattr(mod, "copy_from_host", None)
        ):
            raise RuntimeError(
                "Metal extension `eas._metal` is missing tensor allocation/copy API; rebuild it with "
                "`python3 tools/build_metal_ext.py`."
            )
        if not callable(getattr(mod, "is_available", None)) or not bool(
            mod.is_available()
        ):
            raise RuntimeError("Metal device is not available")

        buf = mod.alloc_buffer(int(self.nbytes), "private")
        arr = self._cpu
        assert arr is not None
        mod.copy_from_host(buf, arr)
        return Tensor(
            shape=self._shape,
            dtype=self._dtype,
            device="metal",
            metal=_MetalStorage(buf=buf, nbytes=self.nbytes, storage="private"),
        )

    def _metal_buffer(self) -> object:
        if self._device != "metal":
            raise TypeError("expected a metal tensor")
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
        if dtype is not None and np.dtype(dtype) != np.float32:
            raise TypeError("only float32 tensors are supported in MVP")
        if t.dtype != torch.float32:
            t = t.to(dtype=torch.float32)

        if device_n == "metal" and t.device.type == "mps":
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
                    "`python3 tools/build_metal_ext.py`."
                )
            cap = t.__dlpack__()
            buf, shape = mod.dlpack_import(cap)
            shape_t = tuple(int(d) for d in shape)
            nbytes = _numel(shape_t) * 4
            return Tensor(
                shape=shape_t,
                dtype=np.float32,
                device="metal",
                metal=_MetalStorage(buf=buf, nbytes=nbytes, storage="private"),
            )

        if t.device.type != "cpu":
            t = t.to("cpu")
        cpu_arr = t.contiguous().numpy()
        cpu_t = Tensor(
            shape=tuple(cpu_arr.shape), dtype=np.float32, device="cpu", cpu=cpu_arr
        )
        return cpu_t.to(device_n)

    if isinstance(data, Tensor):
        out = data
        if dtype is not None and np.dtype(dtype) != out.dtype:
            if out.device != "cpu":
                out = out.to("cpu")
            out = Tensor(
                shape=out.shape,
                dtype=np.dtype(dtype),
                device="cpu",
                cpu=out.numpy().astype(dtype, copy=True),
            )
        return out.to(device_n)

    arr = np.asarray(data, dtype=np.float32 if dtype is None else dtype)
    _require_float32(arr.dtype)
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
    dtype_n = np.dtype(np.float32 if dtype is None else dtype)
    _require_float32(dtype_n)
    shape_t = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)

    if device_n == "cpu":
        arr = np.empty(shape_t, dtype=dtype_n)
        return Tensor(shape=shape_t, dtype=dtype_n, device="cpu", cpu=arr)

    mod = load_metal_ext(require=True)
    if not callable(getattr(mod, "alloc_buffer", None)):
        raise RuntimeError(
            "Metal extension `eas._metal` is missing tensor allocation API; rebuild it with "
            "`python3 tools/build_metal_ext.py`."
        )
    if not callable(getattr(mod, "is_available", None)) or not bool(mod.is_available()):
        raise RuntimeError("Metal device is not available")
    nbytes = _numel(shape_t) * int(dtype_n.itemsize)
    buf = mod.alloc_buffer(int(nbytes), "private")
    return Tensor(
        shape=shape_t,
        dtype=dtype_n,
        device="metal",
        metal=_MetalStorage(buf=buf, nbytes=nbytes, storage="private"),
    )


def empty_like(x: Tensor, *, device: Device | str | None = None) -> Tensor:
    device_n = x.device if device is None else _normalize_device(device)
    return empty(x.shape, device=device_n, dtype=x.dtype)
