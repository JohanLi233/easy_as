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


def program_id(axis: int) -> val:
    return _b().program_id(axis)


def arange(start: int, size: int) -> val:
    return _b().arange(start, size)


def load(buffer: Any, offset: Any, mask: Any | None = None) -> val:
    return _b().load(buffer, offset, mask)


def store(buffer: Any, offset: Any, value: Any, mask: Any | None = None) -> None:
    _b().store(buffer, offset, value, mask)


def where(cond: Any, a: Any, b: Any) -> val:
    return _b().where(cond, a, b)
