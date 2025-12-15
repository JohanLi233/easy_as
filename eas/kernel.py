from __future__ import annotations

import inspect
from typing import Any, Callable

from .compiler import CompiledKernel, compile as _compile
from .meta import constexpr
from .runtime import get_runtime


class Kernel:
    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.name = fn.__name__
        self.sig = inspect.signature(fn)
        annotations = inspect.get_annotations(fn, eval_str=True)
        self.meta_params = set()
        for name, param in self.sig.parameters.items():
            ann = annotations.get(name, param.annotation)
            if ann is constexpr:
                self.meta_params.add(name)
        self._cache: dict[
            tuple[tuple[tuple[str, Any], ...], tuple[tuple[str, Any], ...]],
            CompiledKernel,
        ] = {}

    def compile(self, /, *args: Any, **kwargs: Any) -> CompiledKernel:
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()

        meta = {
            k: bound.arguments.pop(k)
            for k in list(bound.arguments)
            if k in self.meta_params
        }
        runtime_args = bound.arguments

        cache_key = (
            tuple(sorted(meta.items())),
            tuple(sorted((k, type(v)) for k, v in runtime_args.items())),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        ck = _compile(self.fn, runtime_args, meta)
        self._cache[cache_key] = ck
        return ck

    def __call__(self, /, *args: Any, **kwargs: Any) -> None:
        ck = self.compile(*args, **kwargs)
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()
        meta = {
            k: bound.arguments.pop(k)
            for k in list(bound.arguments)
            if k in self.meta_params
        }
        runtime_args = bound.arguments
        get_runtime().run(ck, runtime_args, meta)

    def to_msl(self, /, *args: Any, **kwargs: Any) -> str:
        return self.compile(*args, **kwargs).msl


def kernel(fn: Callable[..., Any]) -> Kernel:
    return Kernel(fn)
