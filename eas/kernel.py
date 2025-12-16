# kernel.py
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

        shape = kwargs.pop("_shape", None)
        shape_norm: tuple[int, ...] | None
        if shape is None:
            shape_norm = None
        else:
            if isinstance(shape, (int,)):
                dims = [int(shape)]
            elif isinstance(shape, (tuple, list)):
                dims = [int(x) for x in shape]
            else:
                raise TypeError(
                    f"_shape must be int or tuple/list of ints, got {type(shape)!r}"
                )
            if len(dims) < 1:
                raise ValueError("_shape must have >= 1 dim")
            if any(d < 0 for d in dims):
                raise ValueError("_shape dims must be >= 0")
            shape_norm = tuple(dims)

        grid = kwargs.pop("_grid", None)
        grid_norm: tuple[int, int, int] | None
        if grid is None:
            grid_norm = None
        else:
            if isinstance(grid, (int,)):
                dims = [int(grid)]
            elif isinstance(grid, (tuple, list)):
                dims = [int(x) for x in grid]
            else:
                raise TypeError(
                    f"_grid must be int or tuple/list of ints, got {type(grid)!r}"
                )
            if not (1 <= len(dims) <= 3):
                raise ValueError("_grid must have 1..3 dims")
            while len(dims) < 3:
                dims.append(1)
            gx, gy, gz = dims
            if gx < 0 or gy < 0 or gz < 0:
                raise ValueError("_grid dims must be >= 0")
            grid_norm = (gx, gy, gz)

        nthreads = kwargs.pop("_nthreads", None)
        if nthreads is not None:
            try:
                nthreads_i = int(nthreads)
            except Exception as e:  # pragma: no cover
                raise TypeError(
                    f"_nthreads must be an int-like value, got {type(nthreads)!r}"
                ) from e
            if nthreads_i < 0:
                raise ValueError("_nthreads must be >= 0")
        else:
            nthreads_i = None
        return {
            "sync": sync,
            "nthreads": nthreads_i,
            "grid": grid_norm,
            "shape": shape_norm,
        }

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
