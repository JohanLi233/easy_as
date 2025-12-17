# mk.py
from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .ir import DType, ValueRef


class _TraceTLS(threading.local):
    builder: Any | None = None


_tls = _TraceTLS()


@contextlib.contextmanager
def _trace(builder: Any) -> Iterator[None]:
    prev = _tls.builder
    _tls.builder = builder
    try:
        yield
    finally:
        _tls.builder = prev


def _b() -> Any:
    builder = _tls.builder
    if builder is None:
        raise RuntimeError("eas.mk is only valid inside an @eas.kernel trace/launch")
    return builder


@dataclass(frozen=True, slots=True)
class val:
    ref: ValueRef

    @property
    def dtype(self) -> DType:
        return self.ref.dtype

    def __add__(self, other: Any) -> "val":
        return _b().add(self, other)

    def __radd__(self, other: Any) -> "val":
        return _b().add(other, self)

    def __mul__(self, other: Any) -> "val":
        return _b().mul(self, other)

    def __rmul__(self, other: Any) -> "val":
        return _b().mul(other, self)

    def __floordiv__(self, other: Any) -> "val":
        return _b().floordiv(self, other)

    def __rfloordiv__(self, other: Any) -> "val":
        return _b().floordiv(other, self)

    def __mod__(self, other: Any) -> "val":
        return _b().mod(self, other)

    def __rmod__(self, other: Any) -> "val":
        return _b().mod(other, self)

    def __lt__(self, other: Any) -> "val":
        return _b().lt(self, other)

    def __and__(self, other: Any) -> "val":
        return _b().and_(self, other)

    def __rand__(self, other: Any) -> "val":
        return _b().and_(other, self)

    def __or__(self, other: Any) -> "val":
        return _b().or_(self, other)

    def __ror__(self, other: Any) -> "val":
        return _b().or_(other, self)

    def __invert__(self) -> "val":
        return _b().not_(self)


def program_id(axis: int) -> val:
    return _b().program_id(axis)


def local_id(axis: int) -> val:
    return _b().local_id(axis)


def lane_id() -> val:
    return _b().lane_id()


def sg_id() -> val:
    return _b().sg_id()


def arange(start: int, size: int) -> val:
    return _b().arange(start, size)


def alloc_tg(size: int) -> val:
    return _b().alloc_tg(size)


def alloc_tg_f16(size: int) -> val:
    return _b().alloc_tg(size, dtype=DType.F16)


def barrier() -> None:
    _b().barrier()


def load(buffer: Any, offset: Any, mask: Any | None = None) -> val:
    return _b().load(buffer, offset, mask)


def store(buffer: Any, offset: Any, value: Any, mask: Any | None = None) -> None:
    _b().store(buffer, offset, value, mask)


def where(cond: Any, a: Any, b: Any) -> val:
    return _b().where(cond, a, b)


def fma(a: Any, b: Any, c: Any) -> val:
    return _b().fma(a, b, c)


def to_f32(x: Any) -> val:
    return _b().cast(x, DType.F32)


def to_f16(x: Any) -> val:
    return _b().cast(x, DType.F16)


def to_u32(x: Any) -> val:
    return _b().cast(x, DType.U32)


def and_(a: Any, b: Any) -> val:
    return _b().and_(a, b)


def or_(a: Any, b: Any) -> val:
    return _b().or_(a, b)


def not_(x: Any) -> val:
    return _b().not_(x)


def dot(
    a_buffer: Any,
    a_base: Any,
    a_stride: Any,
    b_buffer: Any,
    b_base: Any,
    b_stride: Any,
    K: Any,
) -> val:
    return _b().dot(a_buffer, a_base, a_stride, b_buffer, b_base, b_stride, K)


def mma_zero() -> val:
    return _b().mma_zero()


def mma(
    a_buffer: Any,
    a_base: Any,
    a_stride: Any,
    b_buffer: Any,
    b_base: Any,
    b_stride: Any,
    acc: Any,
) -> val:
    return _b().mma(a_buffer, a_base, a_stride, b_buffer, b_base, b_stride, acc)


def mma_store(buffer: Any, base: Any, stride: Any, frag: Any) -> None:
    _b().mma_store(buffer, base, stride, frag)
