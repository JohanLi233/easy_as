from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from . import mk
from .analysis import infer_writes
from .ir import Arg, DType, IRModule, Inst, ValueRef


def _scalar_dtype(v: Any) -> DType:
    if isinstance(v, (bool, np.bool_)):
        return DType.BOOL
    if isinstance(v, (int, np.integer)):
        return DType.U32
    raise TypeError(f"unsupported scalar type: {type(v)!r}")


def _buffer_dtype(a: np.ndarray) -> DType:
    if a.dtype == np.float32:
        return DType.F32
    raise TypeError(f"unsupported dtype: {a.dtype} (expected float32)")


def _is_tensor(v: Any) -> bool:
    return bool(getattr(v, "__eas_tensor__", False))


def _is_torch_tensor(v: Any) -> bool:
    try:
        import torch  # type: ignore

        return isinstance(v, torch.Tensor)
    except Exception:
        return False


def _torch_buffer_dtype(v: Any) -> DType:
    try:
        import torch  # type: ignore

        if v.dtype == torch.float32:
            return DType.F32
        raise TypeError(f"unsupported torch dtype: {v.dtype} (expected float32)")
    except ImportError:
        raise TypeError("torch tensor provided but torch is not importable")


def _tensor_dtype(v: Any) -> DType:
    dt = getattr(v, "dtype", None)
    if dt is None:
        raise TypeError("tensor argument is missing dtype")
    if np.dtype(dt) == np.float32:
        return DType.F32
    raise TypeError(f"unsupported tensor dtype: {dt} (expected float32)")


@dataclass(slots=True)
class _IRBuilder:
    name: str
    args: list[Arg]
    insts: list[Inst]
    _next_id: int = 0

    def _new(self, dtype: DType) -> mk.val:
        v = mk.val(ValueRef(self._next_id, dtype))
        self._next_id += 1
        return v

    def _coerce(self, x: Any) -> mk.val:
        if isinstance(x, mk.val):
            return x
        if isinstance(x, bool):
            v = self._new(DType.BOOL)
            self.insts.append(Inst("const", v.ref, (bool(x),)))
            return v
        if isinstance(x, int):
            v = self._new(DType.U32)
            self.insts.append(Inst("const", v.ref, (int(x),)))
            return v
        if isinstance(x, float):
            v = self._new(DType.F32)
            self.insts.append(Inst("const", v.ref, (float(x),)))
            return v
        raise TypeError(f"unsupported literal: {type(x)!r}")

    def program_id(self, axis: int) -> mk.val:
        v = self._new(DType.U32)
        self.insts.append(Inst("program_id", v.ref, (int(axis),)))
        return v

    def arange(self, start: int, size: int) -> mk.val:
        if not isinstance(start, int) or not isinstance(size, int):
            raise TypeError(
                "arange(start, size) requires Python ints (compile-time constants)"
            )
        if size <= 0:
            raise ValueError("arange(size) must be > 0")
        v = self._new(DType.U32)
        self.insts.append(Inst("arange", v.ref, (int(start), int(size))))
        return v

    def add(self, a: Any, b: Any) -> mk.val:
        av = self._coerce(a)
        bv = self._coerce(b)
        if av.dtype != bv.dtype:
            raise TypeError(f"add dtype mismatch: {av.dtype} vs {bv.dtype}")
        out = self._new(av.dtype)
        self.insts.append(Inst("add", out.ref, (av.ref, bv.ref)))
        return out

    def mul(self, a: Any, b: Any) -> mk.val:
        av = self._coerce(a)
        bv = self._coerce(b)
        if av.dtype != bv.dtype:
            raise TypeError(f"mul dtype mismatch: {av.dtype} vs {bv.dtype}")
        out = self._new(av.dtype)
        self.insts.append(Inst("mul", out.ref, (av.ref, bv.ref)))
        return out

    def lt(self, a: Any, b: Any) -> mk.val:
        av = self._coerce(a)
        bv = self._coerce(b)
        if av.dtype != bv.dtype:
            raise TypeError(f"lt dtype mismatch: {av.dtype} vs {bv.dtype}")
        out = self._new(DType.BOOL)
        self.insts.append(Inst("lt", out.ref, (av.ref, bv.ref)))
        return out

    def where(self, cond: Any, a: Any, b: Any) -> mk.val:
        cv = self._coerce(cond)
        av = self._coerce(a)
        bv = self._coerce(b)
        if cv.dtype != DType.BOOL:
            raise TypeError(f"where(cond, ...) expects bool condition, got {cv.dtype}")
        if av.dtype != bv.dtype:
            raise TypeError(f"where value dtype mismatch: {av.dtype} vs {bv.dtype}")
        out = self._new(av.dtype)
        self.insts.append(Inst("where", out.ref, (cv.ref, av.ref, bv.ref)))
        return out

    def load(self, buffer: Any, offset: Any, mask: Any | None) -> mk.val:
        if not isinstance(buffer, mk.val):
            raise TypeError("load(buffer, ...) expects a kernel argument (buffer)")
        offv = self._coerce(offset)
        maskv = self._coerce(True if mask is None else mask)
        out = self._new(DType.F32)
        self.insts.append(Inst("load", out.ref, (buffer.ref, offv.ref, maskv.ref)))
        return out

    def store(self, buffer: Any, offset: Any, value: Any, mask: Any | None) -> None:
        if not isinstance(buffer, mk.val):
            raise TypeError("store(buffer, ...) expects a kernel argument (buffer)")
        offv = self._coerce(offset)
        valv = self._coerce(value)
        maskv = self._coerce(True if mask is None else mask)
        self.insts.append(
            Inst("store", None, (buffer.ref, offv.ref, valv.ref, maskv.ref))
        )


def trace_to_ir(
    fn: Callable[..., Any], runtime_args: Mapping[str, Any], meta: Mapping[str, Any]
) -> IRModule:
    builder = _IRBuilder(name=fn.__name__, args=[], insts=[])

    trace_kwargs: dict[str, Any] = {}
    for name, value in runtime_args.items():
        if isinstance(value, np.ndarray) or _is_tensor(value) or _is_torch_tensor(value):
            if isinstance(value, np.ndarray):
                dtype = _buffer_dtype(value)
            elif _is_torch_tensor(value):
                dtype = _torch_buffer_dtype(value)
            else:
                dtype = _tensor_dtype(value)
            arg = Arg(name=name, dtype=dtype, kind="buffer")
            builder.args.append(arg)
            v = builder._new(DType.F32)
            builder.insts.append(Inst("arg", v.ref, (name, arg.kind, arg.dtype)))
            trace_kwargs[name] = v
        else:
            arg = Arg(name=name, dtype=_scalar_dtype(value), kind="scalar")
            builder.args.append(arg)
            v = builder._new(arg.dtype)
            builder.insts.append(Inst("arg", v.ref, (name, arg.kind, arg.dtype)))
            trace_kwargs[name] = v

    for k, v in meta.items():
        if isinstance(v, np.integer):
            v = int(v)
        if isinstance(v, np.floating):
            v = float(v)
        if not isinstance(v, (bool, int, float)):
            raise TypeError(
                f"meta parameter {k!r} must be a Python literal, got {type(v)!r}"
            )
        trace_kwargs[k] = v

    fn_for_trace = _with_globals(fn, meta)
    with mk._trace(builder):
        fn_for_trace(**trace_kwargs)

    return IRModule(
        name=fn.__name__, args=tuple(builder.args), insts=tuple(builder.insts)
    )


def _with_globals(
    fn: Callable[..., Any], extra_globals: Mapping[str, Any]
) -> Callable[..., Any]:
    if not extra_globals:
        return fn
    new_globals = dict(fn.__globals__)
    new_globals.update(extra_globals)
    return types.FunctionType(
        fn.__code__,
        new_globals,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )


@dataclass(frozen=True, slots=True)
class CompiledKernel:
    ir: IRModule
    msl: str
    threadgroup_size: int
    writes: frozenset[str]


def compile(
    fn: Callable[..., Any], runtime_args: Mapping[str, Any], meta: Mapping[str, Any]
) -> CompiledKernel:
    ir = trace_to_ir(fn, runtime_args, meta)
    from .codegen.msl import ir_to_msl

    msl_src, threadgroup_size = ir_to_msl(ir)
    writes = infer_writes(ir)
    return CompiledKernel(
        ir=ir, msl=msl_src, threadgroup_size=threadgroup_size, writes=writes
    )
