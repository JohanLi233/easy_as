# filename: eas/autotune.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Config:
    meta: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AutotuneSpec:
    configs: tuple[Config, ...]
    key: tuple[str, ...]
    warmup: int
    repeat: int


def autotune(
    *,
    configs: Sequence[Config],
    key: Sequence[str],
    warmup: int = 5,
    repeat: int = 25,
) -> Callable[[Any], Any]:
    if not configs:
        raise ValueError("autotune(configs=...) must be non-empty")
    if not key:
        raise ValueError("autotune(key=...) must be non-empty")
    if warmup < 0:
        raise ValueError("autotune(warmup=...) must be >= 0")
    if repeat <= 0:
        raise ValueError("autotune(repeat=...) must be > 0")

    spec = AutotuneSpec(
        configs=tuple(configs),
        key=tuple(str(k) for k in key),
        warmup=int(warmup),
        repeat=int(repeat),
    )

    def decorate(obj: Any) -> Any:
        setter = getattr(obj, "_set_autotune_spec", None)
        if callable(setter):
            setter(spec)
            return obj
        if callable(obj):
            setattr(obj, "__eas_autotune_spec__", spec)
            return obj
        raise TypeError(
            f"@eas.autotune expects a callable or Kernel, got {type(obj)!r}"
        )

    return decorate
