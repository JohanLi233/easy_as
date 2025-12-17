# filename: eas/compiler.py

from __future__ import annotations

import os
import types
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from . import mk
from .analysis import infer_writes
from .ir import Arg, ArgKind, DType, IRModule, Inst, ValueRef, validate_ir

_TORCH: Any | None = None
_TORCH_CHECKED = False


def _get_torch() -> Any | None:
    global _TORCH, _TORCH_CHECKED
    if _TORCH_CHECKED:
        return _TORCH
    _TORCH_CHECKED = True
    try:
        import torch  # type: ignore

        _TORCH = torch
    except Exception:
        _TORCH = None
    return _TORCH


@dataclass(frozen=True, slots=True)
class RuntimeArgSig:
    kind: ArgKind
    dtype: DType
    device: str | None = None


def runtime_arg_signature(
    runtime_args: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    """
    Return a stable signature for runtime args for kernel caching.

    The goal is to recompile when the traced types/layouts *could* change, while
    ignoring runtime values for scalars (they do not affect tracing).
    """
    sigs: list[tuple[str, Any]] = []
    for name, value in runtime_args.items():
        sig = _runtime_arg_sig(value)
        sigs.append((name, (sig.kind, sig.dtype, sig.device)))
    return tuple(sorted(sigs))


def _scalar_dtype(v: Any) -> DType:
    if isinstance(v, (bool, np.bool_)):
        return DType.BOOL
    if isinstance(v, (int, np.integer)):
        return DType.U32
    if isinstance(v, (np.float16,)):
        return DType.F16
    if isinstance(v, (float, np.floating)):
        return DType.F32
    raise TypeError(f"unsupported scalar type: {type(v)!r}")


def _buffer_dtype(a: np.ndarray) -> DType:
    if a.dtype == np.float32:
        return DType.F32
    if a.dtype == np.float16:
        return DType.F16
    raise TypeError(f"unsupported dtype: {a.dtype} (expected float32 or float16)")


def _is_tensor(v: Any) -> bool:
    return bool(getattr(v, "__eas_tensor__", False))


def _is_torch_tensor(v: Any) -> bool:
    torch = _get_torch()
    if torch is None:
        return False
    try:
        return isinstance(v, torch.Tensor)
    except Exception:
        return False


def _torch_buffer_dtype(v: Any) -> DType:
    torch = _get_torch()
    if torch is None:
        raise TypeError("torch tensor provided but torch is not importable")
    if v.dtype == torch.float32:
        return DType.F32
    if v.dtype == torch.float16:
        return DType.F16
    raise TypeError(f"unsupported torch dtype: {v.dtype} (expected float32 or float16)")


def _tensor_dtype(v: Any) -> DType:
    dt = getattr(v, "dtype", None)
    if dt is None:
        raise TypeError("tensor argument is missing dtype")
    if np.dtype(dt) == np.float32:
        return DType.F32
    if np.dtype(dt) == np.float16:
        return DType.F16
    raise TypeError(f"unsupported tensor dtype: {dt} (expected float32 or float16)")


def _runtime_arg_sig(v: Any) -> RuntimeArgSig:
    if isinstance(v, np.ndarray):
        return RuntimeArgSig(kind="buffer", dtype=_buffer_dtype(v), device="cpu")
    if _is_tensor(v):
        device = str(getattr(v, "device", None)) if getattr(v, "device", None) else None
        return RuntimeArgSig(kind="buffer", dtype=_tensor_dtype(v), device=device)
    if _is_torch_tensor(v):
        try:
            import torch  # type: ignore

            dt = _torch_buffer_dtype(v)
            dev = str(v.device.type) if isinstance(v, torch.Tensor) else None
        except Exception:
            dt = DType.F32
            dev = None
        return RuntimeArgSig(kind="buffer", dtype=dt, device=dev)
    return RuntimeArgSig(kind="scalar", dtype=_scalar_dtype(v), device=None)


@dataclass(slots=True)
class _IRBuilder:
    name: str
    args: list[Arg]
    insts: list[Inst]
    _buffer_ids: set[int]
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
        if isinstance(x, (int, np.integer)):
            v = self._new(DType.U32)
            self.insts.append(Inst("const", v.ref, (int(x),)))
            return v
        if isinstance(x, (float, np.floating)):
            v = self._new(DType.F32)
            self.insts.append(Inst("const", v.ref, (float(x),)))
            return v
        raise TypeError(f"unsupported literal: {type(x)!r}")

    def program_id(self, axis: int) -> mk.val:
        v = self._new(DType.U32)
        self.insts.append(Inst("program_id", v.ref, (int(axis),)))
        return v

    def local_id(self, axis: int) -> mk.val:
        if not isinstance(axis, int):
            raise TypeError("local_id(axis) requires an int axis")
        if axis not in (0, 1, 2):
            raise ValueError("local_id axis must be 0/1/2")
        v = self._new(DType.U32)
        self.insts.append(Inst("local_id", v.ref, (int(axis),)))
        return v

    def lane_id(self) -> mk.val:
        v = self._new(DType.U32)
        self.insts.append(Inst("lane_id", v.ref, ()))
        return v

    def sg_id(self) -> mk.val:
        v = self._new(DType.U32)
        self.insts.append(Inst("sg_id", v.ref, ()))
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

    def alloc_tg(self, size: int, *, dtype: DType = DType.F32) -> mk.val:
        if not isinstance(size, int):
            raise TypeError(
                "alloc_tg(size) requires a Python int (compile-time constant)"
            )
        if size <= 0:
            raise ValueError("alloc_tg(size) must be > 0")
        if dtype not in (DType.F16, DType.F32):
            raise TypeError(f"alloc_tg(dtype=...) must be f16/f32, got {dtype}")
        v = self._new(dtype)
        self.insts.append(Inst("alloc_tg", v.ref, (int(size),)))
        self._buffer_ids.add(v.ref.id)
        return v

    def barrier(self) -> None:
        self.insts.append(Inst("barrier", None, ()))

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

    def fma(self, a: Any, b: Any, c: Any) -> mk.val:
        av = self._coerce(a)
        bv = self._coerce(b)
        cv = self._coerce(c)
        if av.dtype != bv.dtype or av.dtype != cv.dtype:
            raise TypeError(
                f"fma dtype mismatch: {av.dtype} vs {bv.dtype} vs {cv.dtype}"
            )
        out = self._new(av.dtype)
        self.insts.append(Inst("fma", out.ref, (av.ref, bv.ref, cv.ref)))
        return out

    def floordiv(self, a: Any, b: Any) -> mk.val:
        av = self._coerce(a)
        bv = self._coerce(b)
        if av.dtype != bv.dtype:
            raise TypeError(f"floordiv dtype mismatch: {av.dtype} vs {bv.dtype}")
        if av.dtype != DType.U32:
            raise TypeError(f"floordiv expects u32, got {av.dtype}")
        out = self._new(DType.U32)
        self.insts.append(Inst("floordiv", out.ref, (av.ref, bv.ref)))
        return out

    def mod(self, a: Any, b: Any) -> mk.val:
        av = self._coerce(a)
        bv = self._coerce(b)
        if av.dtype != bv.dtype:
            raise TypeError(f"mod dtype mismatch: {av.dtype} vs {bv.dtype}")
        if av.dtype != DType.U32:
            raise TypeError(f"mod expects u32, got {av.dtype}")
        out = self._new(DType.U32)
        self.insts.append(Inst("mod", out.ref, (av.ref, bv.ref)))
        return out

    def lt(self, a: Any, b: Any) -> mk.val:
        av = self._coerce(a)
        bv = self._coerce(b)
        if av.dtype != bv.dtype:
            raise TypeError(f"lt dtype mismatch: {av.dtype} vs {bv.dtype}")
        out = self._new(DType.BOOL)
        self.insts.append(Inst("lt", out.ref, (av.ref, bv.ref)))
        return out

    def and_(self, a: Any, b: Any) -> mk.val:
        av = self._coerce(a)
        bv = self._coerce(b)
        if av.dtype != DType.BOOL or bv.dtype != DType.BOOL:
            raise TypeError("and/or/not expect bool operands")
        out = self._new(DType.BOOL)
        self.insts.append(Inst("and", out.ref, (av.ref, bv.ref)))
        return out

    def or_(self, a: Any, b: Any) -> mk.val:
        av = self._coerce(a)
        bv = self._coerce(b)
        if av.dtype != DType.BOOL or bv.dtype != DType.BOOL:
            raise TypeError("and/or/not expect bool operands")
        out = self._new(DType.BOOL)
        self.insts.append(Inst("or", out.ref, (av.ref, bv.ref)))
        return out

    def not_(self, x: Any) -> mk.val:
        xv = self._coerce(x)
        if xv.dtype != DType.BOOL:
            raise TypeError("and/or/not expect bool operands")
        out = self._new(DType.BOOL)
        self.insts.append(Inst("not", out.ref, (xv.ref,)))
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

    def cast(self, x: Any, dtype: DType) -> mk.val:
        xv = self._coerce(x)
        if not isinstance(dtype, DType):
            raise TypeError("cast(x, dtype) expects dtype to be a DType")
        if xv.dtype == dtype:
            return xv
        out = self._new(dtype)
        self.insts.append(Inst("cast", out.ref, (xv.ref, dtype)))
        return out

    def dot(
        self,
        a_buffer: Any,
        a_base: Any,
        a_stride: Any,
        b_buffer: Any,
        b_base: Any,
        b_stride: Any,
        K: Any,
    ) -> mk.val:
        if not isinstance(a_buffer, mk.val) or a_buffer.ref.id not in self._buffer_ids:
            raise TypeError("dot expects a_buffer to be a buffer (arg or alloc_tg)")
        if not isinstance(b_buffer, mk.val) or b_buffer.ref.id not in self._buffer_ids:
            raise TypeError("dot expects b_buffer to be a buffer (arg or alloc_tg)")
        if a_buffer.dtype != DType.F32 or b_buffer.dtype != DType.F32:
            raise TypeError("dot currently supports float32 buffers only")

        a_base_v = self._coerce(a_base)
        a_stride_v = self._coerce(a_stride)
        b_base_v = self._coerce(b_base)
        b_stride_v = self._coerce(b_stride)
        k_v = self._coerce(K)
        for name, v in {
            "a_base": a_base_v,
            "a_stride": a_stride_v,
            "b_base": b_base_v,
            "b_stride": b_stride_v,
            "K": k_v,
        }.items():
            if v.dtype != DType.U32:
                raise TypeError(f"dot {name} must be u32, got {v.dtype}")

        out = self._new(DType.F32)
        self.insts.append(
            Inst(
                "dot",
                out.ref,
                (
                    a_buffer.ref,
                    a_base_v.ref,
                    a_stride_v.ref,
                    b_buffer.ref,
                    b_base_v.ref,
                    b_stride_v.ref,
                    k_v.ref,
                ),
            )
        )
        return out

    def mma_zero(self) -> mk.val:
        out = self._new(DType.SG_F32_8X8)
        self.insts.append(Inst("mma_zero", out.ref, ()))
        return out

    def mma(
        self,
        a_buffer: Any,
        a_base: Any,
        a_stride: Any,
        b_buffer: Any,
        b_base: Any,
        b_stride: Any,
        acc: Any,
    ) -> mk.val:
        if not isinstance(a_buffer, mk.val) or a_buffer.ref.id not in self._buffer_ids:
            raise TypeError("mma expects a_buffer to be a buffer (arg or alloc_tg)")
        if not isinstance(b_buffer, mk.val) or b_buffer.ref.id not in self._buffer_ids:
            raise TypeError("mma expects b_buffer to be a buffer (arg or alloc_tg)")
        if a_buffer.dtype not in (DType.F16, DType.F32) or b_buffer.dtype not in (
            DType.F16,
            DType.F32,
        ):
            raise TypeError("mma currently supports f16/f32 buffers only")

        a_base_v = self._coerce(a_base)
        a_stride_v = self._coerce(a_stride)
        b_base_v = self._coerce(b_base)
        b_stride_v = self._coerce(b_stride)
        acc_v = self._coerce(acc)
        for name, v in {
            "a_base": a_base_v,
            "a_stride": a_stride_v,
            "b_base": b_base_v,
            "b_stride": b_stride_v,
        }.items():
            if v.dtype != DType.U32:
                raise TypeError(f"mma {name} must be u32, got {v.dtype}")
        if acc_v.dtype != DType.SG_F32_8X8:
            raise TypeError(f"mma acc must be sg_f32_8x8, got {acc_v.dtype}")

        out = self._new(DType.SG_F32_8X8)
        self.insts.append(
            Inst(
                "mma",
                out.ref,
                (
                    a_buffer.ref,
                    a_base_v.ref,
                    a_stride_v.ref,
                    b_buffer.ref,
                    b_base_v.ref,
                    b_stride_v.ref,
                    acc_v.ref,
                ),
            )
        )
        return out

    def mma_store(self, buffer: Any, base: Any, stride: Any, frag: Any) -> None:
        if not isinstance(buffer, mk.val) or buffer.ref.id not in self._buffer_ids:
            raise TypeError("mma_store(buffer, ...) expects a buffer (arg or alloc_tg)")
        if buffer.dtype != DType.F32:
            raise TypeError("mma_store currently supports float32 buffers only")

        base_v = self._coerce(base)
        stride_v = self._coerce(stride)
        frag_v = self._coerce(frag)
        if base_v.dtype != DType.U32:
            raise TypeError(f"mma_store base must be u32, got {base_v.dtype}")
        if stride_v.dtype != DType.U32:
            raise TypeError(f"mma_store stride must be u32, got {stride_v.dtype}")
        if frag_v.dtype != DType.SG_F32_8X8:
            raise TypeError(f"mma_store frag must be sg_f32_8x8, got {frag_v.dtype}")
        self.insts.append(
            Inst("mma_store", None, (buffer.ref, base_v.ref, stride_v.ref, frag_v.ref))
        )

    def load(self, buffer: Any, offset: Any, mask: Any | None) -> mk.val:
        if not isinstance(buffer, mk.val):
            raise TypeError("load(buffer, ...) expects a buffer (arg or alloc_tg)")
        if buffer.ref.id not in self._buffer_ids:
            raise TypeError("load(buffer, ...) expects a buffer (arg or alloc_tg)")
        offv = self._coerce(offset)
        maskv = self._coerce(True if mask is None else mask)
        if offv.dtype != DType.U32:
            raise TypeError(f"load offset must be u32, got {offv.dtype}")
        if maskv.dtype != DType.BOOL:
            raise TypeError(f"load mask must be bool, got {maskv.dtype}")
        out = self._new(buffer.dtype)
        self.insts.append(Inst("load", out.ref, (buffer.ref, offv.ref, maskv.ref)))
        return out

    def store(self, buffer: Any, offset: Any, value: Any, mask: Any | None) -> None:
        if not isinstance(buffer, mk.val):
            raise TypeError("store(buffer, ...) expects a buffer (arg or alloc_tg)")
        if buffer.ref.id not in self._buffer_ids:
            raise TypeError("store(buffer, ...) expects a buffer (arg or alloc_tg)")
        offv = self._coerce(offset)
        valv = self._coerce(value)
        maskv = self._coerce(True if mask is None else mask)
        if offv.dtype != DType.U32:
            raise TypeError(f"store offset must be u32, got {offv.dtype}")
        if maskv.dtype != DType.BOOL:
            raise TypeError(f"store mask must be bool, got {maskv.dtype}")
        if valv.dtype != buffer.dtype:
            raise TypeError(f"store dtype mismatch: {buffer.dtype} vs {valv.dtype}")
        self.insts.append(
            Inst("store", None, (buffer.ref, offv.ref, valv.ref, maskv.ref))
        )


def trace_to_ir(
    fn: Callable[..., Any], runtime_args: Mapping[str, Any], meta: Mapping[str, Any]
) -> IRModule:
    builder = _IRBuilder(name=fn.__name__, args=[], insts=[], _buffer_ids=set())

    trace_kwargs: dict[str, Any] = {}
    for name, value in runtime_args.items():
        if (
            isinstance(value, np.ndarray)
            or _is_tensor(value)
            or _is_torch_tensor(value)
        ):
            if isinstance(value, np.ndarray):
                dtype = _buffer_dtype(value)
            elif _is_torch_tensor(value):
                dtype = _torch_buffer_dtype(value)
            else:
                dtype = _tensor_dtype(value)
            arg = Arg(name=name, dtype=dtype, kind="buffer")
            builder.args.append(arg)
            v = builder._new(arg.dtype)
            builder.insts.append(Inst("arg", v.ref, (name, arg.kind, arg.dtype)))
            builder._buffer_ids.add(v.ref.id)
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
    ir = _optimize_ir(ir)
    validate_ir(ir)
    from .codegen.msl import ir_to_msl

    msl_src, threadgroup_size = ir_to_msl(ir)
    writes = infer_writes(ir)
    return CompiledKernel(
        ir=ir, msl=msl_src, threadgroup_size=threadgroup_size, writes=writes
    )


def _optimize_ir(ir: IRModule) -> IRModule:
    ir = _rewrite_thread_id(ir)
    ir = _fuse_fma(ir)
    ir = _simplify_bools(ir)
    ir = _cse_pure(ir)
    ir = _dce(ir)
    return ir


def _simplify_bools(ir: IRModule) -> IRModule:
    def_idx: dict[int, int] = {}
    for i, inst in enumerate(ir.insts):
        if inst.out is not None:
            def_idx[inst.out.id] = i

    def _const_bool(ref: ValueRef) -> bool | None:
        i = def_idx.get(ref.id)
        if i is None:
            return None
        inst = ir.insts[i]
        if inst.op != "const" or ref.dtype != DType.BOOL:
            return None
        if not isinstance(inst.args[0], bool):
            return None
        return bool(inst.args[0])

    id_to_ref: dict[int, ValueRef] = {}
    new_insts: list[Inst] = []

    def _remap(a: Any) -> Any:
        if isinstance(a, ValueRef):
            return id_to_ref.get(a.id, a)
        return a

    def _alias(out: ValueRef, ref: ValueRef) -> None:
        id_to_ref[out.id] = ref

    for inst in ir.insts:
        remapped_args = tuple(_remap(a) for a in inst.args)

        if inst.op == "arg":
            assert inst.out is not None
            id_to_ref[inst.out.id] = inst.out
            new_insts.append(Inst(inst.op, inst.out, remapped_args))
            continue

        if inst.op in {
            "store",
            "load",
            "dot",
            "mma",
            "mma_store",
            "alloc_tg",
            "barrier",
        }:
            if inst.out is not None:
                id_to_ref[inst.out.id] = inst.out
            new_insts.append(Inst(inst.op, inst.out, remapped_args))
            continue

        if inst.out is None:
            new_insts.append(Inst(inst.op, None, remapped_args))
            continue

        out = inst.out

        if inst.op == "where" and out.dtype == DType.BOOL:
            c_ref, a_ref, b_ref = remapped_args  # type: ignore[misc]
            assert isinstance(c_ref, ValueRef)
            assert isinstance(a_ref, ValueRef)
            assert isinstance(b_ref, ValueRef)
            cb = _const_bool(c_ref)
            if cb is not None:
                _alias(out, a_ref if cb else b_ref)
                continue
            ab = _const_bool(a_ref)
            bb = _const_bool(b_ref)
            if bb is False:
                # cond ? other : false  ->  cond && other
                if ab is True:
                    _alias(out, c_ref)
                    continue
                new_insts.append(Inst("and", out, (c_ref, a_ref)))
                id_to_ref[out.id] = out
                continue
            if ab is True and bb is not None:
                # cond ? true : b  -> cond || b
                if bb is False:
                    _alias(out, c_ref)
                    continue
                new_insts.append(Inst("or", out, (c_ref, b_ref)))
                id_to_ref[out.id] = out
                continue
            if ab is False and bb is not None:
                # cond ? false : b  -> (!cond) && b
                if bb is False:
                    # always false
                    new_insts.append(Inst("const", out, (False,)))
                    id_to_ref[out.id] = out
                    continue
                new_insts.append(Inst(inst.op, out, (c_ref, a_ref, b_ref)))
                id_to_ref[out.id] = out
                continue

        if inst.op in {"and", "or"}:
            a_ref, b_ref = remapped_args  # type: ignore[misc]
            assert isinstance(a_ref, ValueRef)
            assert isinstance(b_ref, ValueRef)
            av = _const_bool(a_ref)
            bv = _const_bool(b_ref)
            if inst.op == "and":
                if av is False or bv is False:
                    new_insts.append(Inst("const", out, (False,)))
                    id_to_ref[out.id] = out
                    continue
                if av is True:
                    _alias(out, b_ref)
                    continue
                if bv is True:
                    _alias(out, a_ref)
                    continue
            else:
                if av is True or bv is True:
                    new_insts.append(Inst("const", out, (True,)))
                    id_to_ref[out.id] = out
                    continue
                if av is False:
                    _alias(out, b_ref)
                    continue
                if bv is False:
                    _alias(out, a_ref)
                    continue

        if inst.op == "not":
            (x_ref,) = remapped_args  # type: ignore[misc]
            assert isinstance(x_ref, ValueRef)
            xv = _const_bool(x_ref)
            if xv is not None:
                new_insts.append(Inst("const", out, (not xv,)))
                id_to_ref[out.id] = out
                continue

        new_insts.append(Inst(inst.op, out, remapped_args))
        id_to_ref[out.id] = out

    return IRModule(name=ir.name, args=ir.args, insts=tuple(new_insts))


def _rewrite_thread_id(ir: IRModule) -> IRModule:
    disable = os.environ.get("EAS_DISABLE_THREAD_ID_REWRITE", "")
    if disable not in {"", "0", "false", "False"}:
        return ir

    inferred_threadgroup_size: int | None = None
    for inst in ir.insts:
        if inst.op == "arange":
            size = int(inst.args[1])
            if inferred_threadgroup_size is None:
                inferred_threadgroup_size = size
            elif inferred_threadgroup_size != size:
                # Multiple arange sizes: cannot safely rewrite to a single thread_id.
                return ir
    if inferred_threadgroup_size is None:
        return ir

    def_by_id: dict[int, Inst] = {}
    for inst in ir.insts:
        if inst.out is not None:
            def_by_id[inst.out.id] = inst

    new_insts: list[Inst] = []
    for inst in ir.insts:
        if inst.op != "add" or inst.out is None:
            new_insts.append(inst)
            continue

        a_ref, b_ref = inst.args  # type: ignore[misc]
        if not isinstance(a_ref, ValueRef) or not isinstance(b_ref, ValueRef):
            new_insts.append(inst)
            continue

        def _match(mul_ref: ValueRef, arange_ref: ValueRef) -> bool:
            mul_inst = def_by_id.get(mul_ref.id)
            arange_inst = def_by_id.get(arange_ref.id)
            if mul_inst is None or arange_inst is None:
                return False
            if mul_inst.op != "mul" or mul_inst.out is None:
                return False
            if arange_inst.op != "arange" or arange_inst.out is None:
                return False

            start, size = (int(arange_inst.args[0]), int(arange_inst.args[1]))
            if start != 0:
                return False
            if size != inferred_threadgroup_size:
                return False

            x_ref, y_ref = mul_inst.args  # type: ignore[misc]
            if not isinstance(x_ref, ValueRef) or not isinstance(y_ref, ValueRef):
                return False

            def _is_pid(ref: ValueRef) -> bool:
                pid_inst = def_by_id.get(ref.id)
                return bool(
                    pid_inst is not None
                    and pid_inst.op == "program_id"
                    and int(pid_inst.args[0]) == 0
                )

            def _is_block_const(ref: ValueRef) -> bool:
                const_inst = def_by_id.get(ref.id)
                return bool(
                    const_inst is not None
                    and const_inst.op == "const"
                    and isinstance(const_inst.args[0], int)
                    and int(const_inst.args[0]) == inferred_threadgroup_size
                )

            return (_is_pid(x_ref) and _is_block_const(y_ref)) or (
                _is_pid(y_ref) and _is_block_const(x_ref)
            )

        if _match(a_ref, b_ref) or _match(b_ref, a_ref):
            new_insts.append(Inst("thread_id", inst.out, (0,)))
        else:
            new_insts.append(inst)

    return IRModule(name=ir.name, args=ir.args, insts=tuple(new_insts))


def _fuse_fma(ir: IRModule) -> IRModule:
    # Build mapping from value id to its defining instruction
    def_by_id: dict[int, Inst] = {}
    for inst in ir.insts:
        if inst.out is not None:
            def_by_id[inst.out.id] = inst

    # Build use counts to check if mul has other users
    use_counts: dict[int, int] = {}
    for inst in ir.insts:
        for arg in inst.args:
            if isinstance(arg, ValueRef):
                use_counts[arg.id] = use_counts.get(arg.id, 0) + 1

    new_insts: list[Inst] = []
    replaced_mul_ids: set[int] = set()

    for inst in ir.insts:
        # Look for add(mul(x, y), z) or add(z, mul(x, y))
        if inst.op == "add" and inst.out is not None:
            a_ref, b_ref = inst.args  # type: ignore[misc]
            if not isinstance(a_ref, ValueRef) or not isinstance(b_ref, ValueRef):
                new_insts.append(inst)
                continue

            # Try to match mul as first or second argument
            mul_ref = None
            other_ref = None
            for mul_candidate, other_candidate in [(a_ref, b_ref), (b_ref, a_ref)]:
                mul_inst = def_by_id.get(mul_candidate.id)
                if (
                    mul_inst is not None
                    and mul_inst.op == "mul"
                    and mul_inst.out is not None
                ):
                    mul_ref = mul_candidate
                    other_ref = other_candidate
                    break

            if mul_ref is not None and other_ref is not None:
                mul_inst = def_by_id[mul_ref.id]
                x_ref, y_ref = mul_inst.args  # type: ignore[misc]
                if not isinstance(x_ref, ValueRef) or not isinstance(y_ref, ValueRef):
                    new_insts.append(inst)
                    continue

                # Check that all operands are float32 (fma only makes sense for float)
                # Also check that mul result is float32 (should be if inputs are float32)
                if inst.out.dtype == DType.F32 and mul_inst.out.dtype == DType.F32:
                    # Check if the mul result has any other users besides this add
                    # Only fuse if use count is 1 (only this add) to avoid duplicate computation
                    mul_out_id = mul_inst.out.id
                    if use_counts.get(mul_out_id, 0) == 1:
                        replaced_mul_ids.add(mul_out_id)
                        # Replace with fma(x, y, other)
                        new_insts.append(
                            Inst("fma", inst.out, (x_ref, y_ref, other_ref))
                        )
                        # Don't add the original add instruction
                        continue
                    # If mul has multiple users, keep the original add instruction
                    # (will be added below)

        new_insts.append(inst)

    # Filter out mul instructions that were replaced and have no other users
    filtered_insts: list[Inst] = []
    for inst in new_insts:
        if (
            inst.op == "mul"
            and inst.out is not None
            and inst.out.id in replaced_mul_ids
        ):
            # Skip this mul instruction
            continue
        filtered_insts.append(inst)

    return IRModule(name=ir.name, args=ir.args, insts=tuple(filtered_insts))


def _cse_pure(ir: IRModule) -> IRModule:
    id_to_ref: dict[int, ValueRef] = {}
    value_numbering: dict[tuple[object, ...], ValueRef] = {}
    new_insts: list[Inst] = []

    def _remap_arg(a: Any) -> Any:
        if isinstance(a, ValueRef):
            return id_to_ref.get(a.id, a)
        return a

    def _key_for(inst: Inst, out: ValueRef) -> tuple[object, ...] | None:
        if inst.op == "const":
            return ("const", out.dtype, inst.args[0])
        if inst.op in {"program_id", "thread_id", "local_id"}:
            return (inst.op, out.dtype, int(inst.args[0]))
        if inst.op in {"lane_id", "sg_id"}:
            return (inst.op, out.dtype)
        if inst.op == "arange":
            return ("arange", out.dtype, int(inst.args[0]), int(inst.args[1]))
        if inst.op in {"add", "mul"}:
            a_ref, b_ref = (_remap_arg(inst.args[0]), _remap_arg(inst.args[1]))
            assert isinstance(a_ref, ValueRef) and isinstance(b_ref, ValueRef)
            x, y = (a_ref, b_ref) if a_ref.id <= b_ref.id else (b_ref, a_ref)
            return (inst.op, out.dtype, x.id, y.id)
        if inst.op in {"floordiv", "mod"}:
            a_ref, b_ref = (_remap_arg(inst.args[0]), _remap_arg(inst.args[1]))
            assert isinstance(a_ref, ValueRef) and isinstance(b_ref, ValueRef)
            return (inst.op, out.dtype, a_ref.id, b_ref.id)
        if inst.op == "lt":
            a_ref, b_ref = (_remap_arg(inst.args[0]), _remap_arg(inst.args[1]))
            assert isinstance(a_ref, ValueRef) and isinstance(b_ref, ValueRef)
            return ("lt", out.dtype, a_ref.id, b_ref.id)
        if inst.op in {"and", "or"}:
            a_ref, b_ref = (_remap_arg(inst.args[0]), _remap_arg(inst.args[1]))
            assert isinstance(a_ref, ValueRef) and isinstance(b_ref, ValueRef)
            x, y = (a_ref, b_ref) if a_ref.id <= b_ref.id else (b_ref, a_ref)
            return (inst.op, out.dtype, x.id, y.id)
        if inst.op == "not":
            x_ref = _remap_arg(inst.args[0])
            assert isinstance(x_ref, ValueRef)
            return ("not", out.dtype, x_ref.id)
        if inst.op == "where":
            c_ref, a_ref, b_ref = map(_remap_arg, inst.args)
            assert isinstance(c_ref, ValueRef)
            assert isinstance(a_ref, ValueRef)
            assert isinstance(b_ref, ValueRef)
            return ("where", out.dtype, c_ref.id, a_ref.id, b_ref.id)
        if inst.op == "cast":
            x_ref = _remap_arg(inst.args[0])
            dst = inst.args[1]
            assert isinstance(x_ref, ValueRef)
            assert isinstance(dst, DType)
            return ("cast", out.dtype, x_ref.id, dst)
        if inst.op == "fma":
            x_ref, y_ref, z_ref = map(_remap_arg, inst.args)
            assert isinstance(x_ref, ValueRef)
            assert isinstance(y_ref, ValueRef)
            assert isinstance(z_ref, ValueRef)
            # x and y are commutative (multiplication), z is addition
            x, y = (x_ref, y_ref) if x_ref.id <= y_ref.id else (y_ref, x_ref)
            return ("fma", out.dtype, x.id, y.id, z_ref.id)
        return None

    for inst in ir.insts:
        remapped_args = tuple(_remap_arg(a) for a in inst.args)

        if inst.op == "arg":
            assert inst.out is not None
            id_to_ref[inst.out.id] = inst.out
            new_insts.append(Inst(inst.op, inst.out, remapped_args))
            continue

        if inst.op in {
            "store",
            "load",
            "dot",
            "mma",
            "mma_store",
            "alloc_tg",
            "barrier",
        }:
            if inst.out is not None:
                id_to_ref[inst.out.id] = inst.out
            new_insts.append(Inst(inst.op, inst.out, remapped_args))
            continue

        if inst.out is None:
            new_insts.append(Inst(inst.op, None, remapped_args))
            continue

        out = inst.out
        key = _key_for(Inst(inst.op, inst.out, remapped_args), out)
        if key is not None and key in value_numbering:
            id_to_ref[out.id] = value_numbering[key]
            continue

        if key is not None:
            value_numbering[key] = out
        id_to_ref[out.id] = out
        new_insts.append(Inst(inst.op, out, remapped_args))

    return IRModule(name=ir.name, args=ir.args, insts=tuple(new_insts))


def _dce(ir: IRModule) -> IRModule:
    def_idx: dict[int, int] = {}
    for i, inst in enumerate(ir.insts):
        if inst.out is not None:
            def_idx[inst.out.id] = i

    uses: dict[int, set[int]] = {}

    def _add_use(v: ValueRef, user_idx: int) -> None:
        uses.setdefault(v.id, set()).add(user_idx)

    for i, inst in enumerate(ir.insts):
        for a in inst.args:
            if isinstance(a, ValueRef):
                _add_use(a, i)

    live_ids: set[int] = set()

    def _mark(v: ValueRef) -> None:
        live_ids.add(v.id)

    # Roots: side effects and required metadata.
    for inst in ir.insts:
        if inst.op == "store":
            buf_ref, off_ref, val_ref, mask_ref = inst.args  # type: ignore[misc]
            _mark(buf_ref)
            _mark(off_ref)
            _mark(val_ref)
            _mark(mask_ref)
        elif inst.op == "mma_store":
            buf_ref, base_ref, stride_ref, frag_ref = inst.args  # type: ignore[misc]
            _mark(buf_ref)
            _mark(base_ref)
            _mark(stride_ref)
            _mark(frag_ref)
        elif inst.op == "barrier":
            pass
        elif inst.op == "arange":
            # Keep arange even if its SSA result is unused: it defines threadgroup_size.
            assert inst.out is not None
            live_ids.add(inst.out.id)
        elif inst.op == "arg":
            assert inst.out is not None
            live_ids.add(inst.out.id)

    changed = True
    while changed:
        changed = False
        for value_id in tuple(live_ids):
            idx = def_idx.get(value_id)
            if idx is None:
                continue
            inst = ir.insts[idx]
            for a in inst.args:
                if isinstance(a, ValueRef) and a.id not in live_ids:
                    live_ids.add(a.id)
                    changed = True

    new_insts: list[Inst] = []
    for inst in ir.insts:
        if inst.op in {"arg", "store", "arange", "barrier"}:
            new_insts.append(inst)
            continue
        if inst.out is None:
            new_insts.append(inst)
            continue
        if inst.out.id in live_ids:
            new_insts.append(inst)

    return IRModule(name=ir.name, args=ir.args, insts=tuple(new_insts))
