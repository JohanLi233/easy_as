from __future__ import annotations

import inspect
from typing import Any, Callable

from .compiler import CompiledKernel, compile as _compile, runtime_arg_signature
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

    def _pop_runtime_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        sync = kwargs.pop("_sync", True)
        if not isinstance(sync, bool):
            raise TypeError(f"_sync must be bool, got {type(sync)!r}")
        return {"sync": sync}

    def _compile_from_split(
        self, meta: dict[str, Any], runtime_args: dict[str, Any]
    ) -> CompiledKernel:
        cache_key = (
            tuple(sorted(meta.items())),
            runtime_arg_signature(runtime_args),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        ck = _compile(self.fn, runtime_args, meta)
        self._cache[cache_key] = ck
        return ck

    def _bind_and_split(
        self, /, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()
        meta = {
            k: bound.arguments.pop(k)
            for k in list(bound.arguments)
            if k in self.meta_params
        }
        return meta, bound.arguments

    def compile(self, /, *args: Any, **kwargs: Any) -> CompiledKernel:
        _ = self._pop_runtime_options(kwargs)
        meta, runtime_args = self._bind_and_split(*args, **kwargs)
        return self._compile_from_split(meta, runtime_args)

    def __call__(self, /, *args: Any, **kwargs: Any) -> None:
        runtime_opts = self._pop_runtime_options(kwargs)
        meta, runtime_args = self._bind_and_split(*args, **kwargs)
        ck = self._compile_from_split(meta, runtime_args)
        get_runtime().run(ck, runtime_args, meta, **runtime_opts)

    def to_msl(self, /, *args: Any, **kwargs: Any) -> str:
        _ = self._pop_runtime_options(kwargs)
        return self.compile(*args, **kwargs).msl


def kernel(fn: Callable[..., Any]) -> Kernel:
    return Kernel(fn)
