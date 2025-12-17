# kernel.py
from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping

from .compiler import CompiledKernel, compile as _compile, runtime_arg_signature
from .autotune import AutotuneSpec, Config
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
        self._autotune_spec: AutotuneSpec | None = None
        self._tuned_cache: dict[tuple[Any, ...], Config] = {}

    def _set_autotune_spec(self, spec: AutotuneSpec) -> None:
        self._autotune_spec = spec

    def _autotune_key(
        self,
        spec: AutotuneSpec,
        meta: Mapping[str, Any],
        runtime_args: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        def norm(v: Any) -> Any:
            if isinstance(v, (bool, int, float, str, type(None))):
                return v
            if isinstance(v, (tuple, list)):
                return tuple(norm(x) for x in v)
            if isinstance(v, dict):
                return tuple(sorted((str(k), norm(val)) for k, val in v.items()))
            for cast in (int, float):
                try:
                    return cast(v)
                except Exception:
                    pass
            if hasattr(v, "shape"):
                try:
                    shape = tuple(int(x) for x in getattr(v, "shape"))
                except Exception:
                    shape = None
                if shape is not None:
                    return ("shape", shape)
            raise TypeError(
                f"autotune key value must be hashable/normalizable, got {type(v)!r}"
            )

        out: list[Any] = []
        for name in spec.key:
            if name in meta:
                out.append(norm(meta[name]))
            elif name in runtime_args:
                out.append(norm(runtime_args[name]))
            else:
                raise KeyError(f"autotune key {name!r} not found in args/meta")
        return tuple(out)

    def _autotune_select(
        self,
        spec: AutotuneSpec,
        meta: dict[str, Any],
        runtime_args: dict[str, Any],
        raw_opts: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        key = self._autotune_key(spec, meta, runtime_args)
        cached = self._tuned_cache.get(key)
        if cached is None:
            runtime = get_runtime()
            bench = getattr(runtime, "benchmark", None)
            if not callable(bench):
                raise RuntimeError(
                    "autotune requires a runtime with benchmark(); "
                    "use MetalRuntime (EAS_BACKEND=auto|mps)"
                )

            best_cfg: Config | None = None
            best_t: float | None = None
            for cfg in spec.configs:
                meta_cfg = dict(meta)
                meta_cfg.update(dict(cfg.meta))
                runtime_opts_cfg = self._normalize_runtime_options(
                    raw_opts, meta_cfg, runtime_args
                )
                ck = self._compile_from_split(meta_cfg, runtime_args)
                t = float(
                    bench(
                        ck,
                        runtime_args,
                        meta_cfg,
                        repeat=spec.repeat,
                        warmup=spec.warmup,
                        nthreads=runtime_opts_cfg["nthreads"],
                        grid=runtime_opts_cfg["grid"],
                        tptg=runtime_opts_cfg["tptg"],
                        shape=runtime_opts_cfg["shape"],
                    )
                )
                if best_t is None or t < best_t:
                    best_t = t
                    best_cfg = cfg

            if best_cfg is None:  # pragma: no cover
                raise RuntimeError("autotune failed to select a config")
            self._tuned_cache[key] = best_cfg
            cached = best_cfg

        meta_best = dict(meta)
        meta_best.update(dict(cached.meta))
        runtime_opts_best = self._normalize_runtime_options(
            raw_opts, meta_best, runtime_args
        )
        return meta_best, runtime_opts_best

    def _extract_runtime_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        sync = kwargs.pop("_sync", True)
        if not isinstance(sync, bool):
            raise TypeError(f"_sync must be bool, got {type(sync)!r}")

        shape = kwargs.pop("_shape", None)

        grid = kwargs.pop("_grid", None)

        tptg = kwargs.pop("_tptg", None)

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
            "grid": grid,
            "shape": shape,
            "tptg": tptg,
        }

    def _normalize_runtime_options(
        self,
        raw: dict[str, Any],
        meta: dict[str, Any],
        runtime_args: dict[str, Any],
    ) -> dict[str, Any]:
        sync = bool(raw["sync"])
        nthreads = raw["nthreads"]

        shape = raw.get("shape", None)
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

        grid = raw.get("grid", None)
        if callable(grid):
            grid = grid(meta, runtime_args)
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
                    f"_grid must be int/tuple/list or callable, got {type(grid)!r}"
                )
            if not (1 <= len(dims) <= 3):
                raise ValueError("_grid must have 1..3 dims")
            while len(dims) < 3:
                dims.append(1)
            gx, gy, gz = dims
            if gx < 0 or gy < 0 or gz < 0:
                raise ValueError("_grid dims must be >= 0")
            grid_norm = (gx, gy, gz)

        tptg = raw.get("tptg", None)
        tptg_norm: tuple[int, int, int] | None
        if tptg is None:
            tptg_norm = None
        else:
            if isinstance(tptg, (int,)):
                dims = [int(tptg)]
            elif isinstance(tptg, (tuple, list)):
                dims = [int(x) for x in tptg]
            else:
                raise TypeError(
                    f"_tptg must be int or tuple/list of ints, got {type(tptg)!r}"
                )
            if not (1 <= len(dims) <= 3):
                raise ValueError("_tptg must have 1..3 dims")
            while len(dims) < 3:
                dims.append(1)
            tx, ty, tz = dims
            if tx <= 0 or ty <= 0 or tz <= 0:
                raise ValueError("_tptg dims must be > 0")
            tptg_norm = (tx, ty, tz)

        return {
            "sync": sync,
            "nthreads": nthreads,
            "grid": grid_norm,
            "shape": shape_norm,
            "tptg": tptg_norm,
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
        raw_opts = self._extract_runtime_options(kwargs)
        meta, runtime_args = self._bind_and_split(*args, **kwargs)
        spec = self._autotune_spec
        if spec is not None:
            meta, _runtime_opts = self._autotune_select(
                spec, meta, runtime_args, raw_opts
            )
        ck = self._compile_from_split(meta, runtime_args)
        raw_tptg = raw_opts.get("tptg")
        if raw_tptg is not None:
            runtime_opts = self._normalize_runtime_options(raw_opts, meta, runtime_args)
            tptg = runtime_opts.get("tptg")
            if tptg is None:
                raise RuntimeError("internal error: normalized tptg must not be None")
            tx, ty, tz = (int(tptg[0]), int(tptg[1]), int(tptg[2]))
            if (tx, ty, tz) != (int(ck.threadgroup_size), 1, 1):
                raise ValueError(
                    f"_tptg must match threadgroup_size inferred from mk.arange(0, BLOCK): "
                    f"expected ({int(ck.threadgroup_size)}, 1, 1), got ({tx}, {ty}, {tz})"
                )
        return ck

    def __call__(self, /, *args: Any, **kwargs: Any) -> None:
        raw_opts = self._extract_runtime_options(kwargs)
        meta, runtime_args = self._bind_and_split(*args, **kwargs)
        spec = self._autotune_spec
        if spec is not None:
            meta, runtime_opts = self._autotune_select(
                spec, meta, runtime_args, raw_opts
            )
        else:
            runtime_opts = self._normalize_runtime_options(raw_opts, meta, runtime_args)
        ck = self._compile_from_split(meta, runtime_args)
        raw_tptg = raw_opts.get("tptg")
        if raw_tptg is not None:
            tptg = runtime_opts.get("tptg")
            if tptg is None:
                raise RuntimeError("internal error: normalized tptg must not be None")
            tx, ty, tz = (int(tptg[0]), int(tptg[1]), int(tptg[2]))
            if (tx, ty, tz) != (int(ck.threadgroup_size), 1, 1):
                raise ValueError(
                    f"_tptg must match threadgroup_size inferred from mk.arange(0, BLOCK): "
                    f"expected ({int(ck.threadgroup_size)}, 1, 1), got ({tx}, {ty}, {tz})"
                )
        get_runtime().run(ck, runtime_args, meta, **runtime_opts)

    def to_msl(self, /, *args: Any, **kwargs: Any) -> str:
        _ = self._extract_runtime_options(kwargs)
        return self.compile(*args, **kwargs).msl


def kernel(fn: Callable[..., Any]) -> Kernel:
    k = Kernel(fn)
    spec = getattr(fn, "__eas_autotune_spec__", None)
    if isinstance(spec, AutotuneSpec):
        k._set_autotune_spec(spec)
    return k
