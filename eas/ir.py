# filename: eas/ir.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal, TypeAlias


class DType(str, Enum):
    BOOL = "bool"
    F16 = "f16"
    F32 = "f32"
    U32 = "u32"
    SG_F32_8X8 = "sg_f32_8x8"


ArgKind: TypeAlias = Literal["buffer", "scalar"]

Op: TypeAlias = Literal[
    "arg",
    "const",
    "program_id",
    "thread_id",
    "local_id",
    "lane_id",
    "sg_id",
    "arange",
    "alloc_tg",
    "add",
    "mul",
    "floordiv",
    "mod",
    "lt",
    "and",
    "or",
    "not",
    "where",
    "cast",
    "dot",
    "mma_zero",
    "mma",
    "mma_store",
    "load",
    "store",
    "fma",
    "barrier",
]


@dataclass(frozen=True, slots=True)
class ValueRef:
    id: int
    dtype: DType


@dataclass(frozen=True, slots=True)
class Arg:
    name: str
    dtype: DType
    kind: ArgKind


@dataclass(frozen=True, slots=True)
class Inst:
    op: Op
    out: ValueRef | None
    args: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class IRModule:
    name: str
    args: tuple[Arg, ...]
    insts: tuple[Inst, ...]

    def buffer_args(self) -> Iterable[Arg]:
        return (arg for arg in self.args if arg.kind == "buffer")

    def scalar_args(self) -> Iterable[Arg]:
        return (arg for arg in self.args if arg.kind == "scalar")


def validate_ir(ir: IRModule) -> None:
    args_by_name = {a.name: a for a in ir.args}
    seen_arg_names: set[str] = set()
    defined: dict[int, ValueRef] = {}
    buffer_value_ids: set[int] = set()

    def require(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"invalid IR: {msg}")

    def define(ref: ValueRef) -> None:
        require(ref.id not in defined, f"duplicate SSA value id: {ref.id}")
        require(isinstance(ref.dtype, DType), f"invalid dtype for v{ref.id}: {ref.dtype!r}")
        defined[ref.id] = ref

    def require_value_ref(x: Any, *, what: str) -> ValueRef:
        require(isinstance(x, ValueRef), f"{what} must be a ValueRef, got {type(x)!r}")
        return x

    def require_defined(ref: ValueRef, *, what: str) -> None:
        require(ref.id in defined, f"use of undefined value id {ref.id} for {what}")
        def_ref = defined[ref.id]
        require(
            def_ref.dtype == ref.dtype,
            f"{what} dtype mismatch for v{ref.id}: {def_ref.dtype} vs {ref.dtype}",
        )

    def require_numeric(dt: DType, *, what: str) -> None:
        require(dt in (DType.U32, DType.F16, DType.F32), f"{what} must be u32/f16/f32, got {dt}")

    def require_float(dt: DType, *, what: str) -> None:
        require(dt in (DType.F16, DType.F32), f"{what} must be f16/f32, got {dt}")

    for inst in ir.insts:
        require(inst.op in Op.__args__, f"unknown op: {inst.op!r}")

        if inst.op == "arg":
            require(inst.out is not None, "arg must produce a value")
            require(len(inst.args) == 3, f"arg expects 3 args, got {len(inst.args)}")
            name = str(inst.args[0])
            kind = inst.args[1]
            dtype = inst.args[2]
            require(name in args_by_name, f"arg references unknown name {name!r}")
            require(kind == args_by_name[name].kind, f"arg kind mismatch for {name!r}")
            require(dtype == args_by_name[name].dtype, f"arg dtype mismatch for {name!r}")
            require(name not in seen_arg_names, f"duplicate arg inst for {name!r}")
            seen_arg_names.add(name)
            define(inst.out)
            if args_by_name[name].kind == "buffer":
                buffer_value_ids.add(inst.out.id)
            continue

        if inst.op == "const":
            require(inst.out is not None, "const must produce a value")
            require(len(inst.args) == 1, f"const expects 1 arg, got {len(inst.args)}")
            v = inst.args[0]
            if inst.out.dtype == DType.BOOL:
                require(isinstance(v, bool), "bool const must be a Python bool")
            elif inst.out.dtype == DType.U32:
                require(
                    isinstance(v, int) and int(v) >= 0,
                    "u32 const must be a non-negative Python int",
                )
            elif inst.out.dtype == DType.F32:
                require(isinstance(v, (int, float)), "f32 const must be an int/float")
            else:
                require(False, f"unsupported const dtype: {inst.out.dtype}")
            define(inst.out)
            continue

        if inst.op == "alloc_tg":
            require(inst.out is not None, "alloc_tg must produce a value")
            require(len(inst.args) == 1, f"alloc_tg expects 1 arg, got {len(inst.args)}")
            require(
                isinstance(inst.args[0], int) and int(inst.args[0]) > 0,
                "alloc_tg(size) requires a positive Python int size",
            )
            require_float(inst.out.dtype, what="alloc_tg output")
            define(inst.out)
            buffer_value_ids.add(inst.out.id)
            continue

        if inst.op in {"program_id", "thread_id", "local_id"}:
            require(inst.out is not None, f"{inst.op} must produce a value")
            require(inst.out.dtype == DType.U32, f"{inst.op} output must be u32")
            require(len(inst.args) == 1, f"{inst.op} expects 1 arg, got {len(inst.args)}")
            require(
                isinstance(inst.args[0], int) and int(inst.args[0]) in (0, 1, 2),
                f"{inst.op} axis must be 0/1/2",
            )
            define(inst.out)
            continue

        if inst.op == "arange":
            require(inst.out is not None, "arange must produce a value")
            require(inst.out.dtype == DType.U32, "arange output must be u32")
            require(len(inst.args) == 2, f"arange expects 2 args, got {len(inst.args)}")
            require(
                isinstance(inst.args[0], int) and isinstance(inst.args[1], int),
                "arange(start, size) requires Python int literals",
            )
            require(int(inst.args[1]) > 0, "arange(size) must be > 0")
            define(inst.out)
            continue

        if inst.op == "lane_id":
            require(inst.out is not None, "lane_id must produce a value")
            require(len(inst.args) == 0, f"lane_id expects 0 args, got {len(inst.args)}")
            require(inst.out.dtype == DType.U32, "lane_id output must be u32")
            define(inst.out)
            continue

        if inst.op == "sg_id":
            require(inst.out is not None, "sg_id must produce a value")
            require(len(inst.args) == 0, f"sg_id expects 0 args, got {len(inst.args)}")
            require(inst.out.dtype == DType.U32, "sg_id output must be u32")
            define(inst.out)
            continue

        if inst.op in {"add", "mul", "floordiv", "mod"}:
            require(inst.out is not None, f"{inst.op} must produce a value")
            require(len(inst.args) == 2, f"{inst.op} expects 2 args, got {len(inst.args)}")
            a_ref = require_value_ref(inst.args[0], what=f"{inst.op} lhs")
            b_ref = require_value_ref(inst.args[1], what=f"{inst.op} rhs")
            require_defined(a_ref, what=f"{inst.op} lhs")
            require_defined(b_ref, what=f"{inst.op} rhs")
            require(a_ref.dtype == b_ref.dtype, f"{inst.op} dtype mismatch: {a_ref.dtype} vs {b_ref.dtype}")
            require_numeric(a_ref.dtype, what=f"{inst.op} operands")
            require(inst.out.dtype == a_ref.dtype, f"{inst.op} output dtype must match operands")
            if inst.op in {"floordiv", "mod"}:
                require(a_ref.dtype == DType.U32, f"{inst.op} operands must be u32")
            define(inst.out)
            continue

        if inst.op == "lt":
            require(inst.out is not None, "lt must produce a value")
            require(len(inst.args) == 2, f"lt expects 2 args, got {len(inst.args)}")
            a_ref = require_value_ref(inst.args[0], what="lt lhs")
            b_ref = require_value_ref(inst.args[1], what="lt rhs")
            require_defined(a_ref, what="lt lhs")
            require_defined(b_ref, what="lt rhs")
            require(a_ref.dtype == b_ref.dtype, f"lt dtype mismatch: {a_ref.dtype} vs {b_ref.dtype}")
            require_numeric(a_ref.dtype, what="lt operands")
            require(inst.out.dtype == DType.BOOL, "lt output must be bool")
            define(inst.out)
            continue

        if inst.op in {"and", "or"}:
            require(inst.out is not None, f"{inst.op} must produce a value")
            require(len(inst.args) == 2, f"{inst.op} expects 2 args, got {len(inst.args)}")
            a_ref = require_value_ref(inst.args[0], what=f"{inst.op} lhs")
            b_ref = require_value_ref(inst.args[1], what=f"{inst.op} rhs")
            require_defined(a_ref, what=f"{inst.op} lhs")
            require_defined(b_ref, what=f"{inst.op} rhs")
            require(inst.out.dtype == DType.BOOL, f"{inst.op} output must be bool")
            require(a_ref.dtype == DType.BOOL, f"{inst.op} lhs must be bool")
            require(b_ref.dtype == DType.BOOL, f"{inst.op} rhs must be bool")
            define(inst.out)
            continue

        if inst.op == "not":
            require(inst.out is not None, "not must produce a value")
            require(len(inst.args) == 1, f"not expects 1 arg, got {len(inst.args)}")
            x_ref = require_value_ref(inst.args[0], what="not input")
            require_defined(x_ref, what="not input")
            require(inst.out.dtype == DType.BOOL, "not output must be bool")
            require(x_ref.dtype == DType.BOOL, "not input must be bool")
            define(inst.out)
            continue

        if inst.op == "where":
            require(inst.out is not None, "where must produce a value")
            require(len(inst.args) == 3, f"where expects 3 args, got {len(inst.args)}")
            c_ref = require_value_ref(inst.args[0], what="where cond")
            a_ref = require_value_ref(inst.args[1], what="where a")
            b_ref = require_value_ref(inst.args[2], what="where b")
            require_defined(c_ref, what="where cond")
            require_defined(a_ref, what="where a")
            require_defined(b_ref, what="where b")
            require(c_ref.dtype == DType.BOOL, "where cond must be bool")
            require(a_ref.dtype == b_ref.dtype, f"where dtype mismatch: {a_ref.dtype} vs {b_ref.dtype}")
            require(inst.out.dtype == a_ref.dtype, "where output dtype must match values")
            define(inst.out)
            continue

        if inst.op == "cast":
            require(inst.out is not None, "cast must produce a value")
            require(len(inst.args) == 2, f"cast expects 2 args, got {len(inst.args)}")
            x_ref = require_value_ref(inst.args[0], what="cast input")
            dst = inst.args[1]
            require(isinstance(dst, DType), "cast target must be a DType")
            require_defined(x_ref, what="cast input")
            require(inst.out.dtype == dst, "cast output dtype must match target dtype")
            require(
                x_ref.dtype in (DType.BOOL, DType.U32, DType.F16, DType.F32)
                and dst in (DType.BOOL, DType.U32, DType.F16, DType.F32),
                "cast only supports bool/u32/f16/f32",
            )
            define(inst.out)
            continue

        if inst.op == "load":
            require(inst.out is not None, "load must produce a value")
            require(len(inst.args) == 3, f"load expects 3 args, got {len(inst.args)}")
            buf_ref = require_value_ref(inst.args[0], what="load buffer")
            off_ref = require_value_ref(inst.args[1], what="load offset")
            mask_ref = require_value_ref(inst.args[2], what="load mask")
            require_defined(buf_ref, what="load buffer")
            require_defined(off_ref, what="load offset")
            require_defined(mask_ref, what="load mask")
            require(buf_ref.id in buffer_value_ids, "load buffer must be a buffer (arg or alloc_tg)")
            require_float(buf_ref.dtype, what="load buffer dtype")
            require(off_ref.dtype == DType.U32, "load offset must be u32")
            require(mask_ref.dtype == DType.BOOL, "load mask must be bool")
            require(inst.out.dtype == buf_ref.dtype, "load output dtype must match buffer dtype")
            define(inst.out)
            continue

        if inst.op == "store":
            require(inst.out is None, "store must not produce a value")
            require(len(inst.args) == 4, f"store expects 4 args, got {len(inst.args)}")
            buf_ref = require_value_ref(inst.args[0], what="store buffer")
            off_ref = require_value_ref(inst.args[1], what="store offset")
            val_ref = require_value_ref(inst.args[2], what="store value")
            mask_ref = require_value_ref(inst.args[3], what="store mask")
            require_defined(buf_ref, what="store buffer")
            require_defined(off_ref, what="store offset")
            require_defined(val_ref, what="store value")
            require_defined(mask_ref, what="store mask")
            require(buf_ref.id in buffer_value_ids, "store buffer must be a buffer (arg or alloc_tg)")
            require_float(buf_ref.dtype, what="store buffer dtype")
            require(off_ref.dtype == DType.U32, "store offset must be u32")
            require(mask_ref.dtype == DType.BOOL, "store mask must be bool")
            require(val_ref.dtype == buf_ref.dtype, "store value dtype must match buffer dtype")
            continue

        if inst.op == "fma":
            require(inst.out is not None, "fma must produce a value")
            require(len(inst.args) == 3, f"fma expects 3 args, got {len(inst.args)}")
            x_ref = require_value_ref(inst.args[0], what="fma x")
            y_ref = require_value_ref(inst.args[1], what="fma y")
            z_ref = require_value_ref(inst.args[2], what="fma z")
            require_defined(x_ref, what="fma x")
            require_defined(y_ref, what="fma y")
            require_defined(z_ref, what="fma z")
            require(x_ref.dtype == y_ref.dtype == z_ref.dtype, "fma operands must have same dtype")
            require_float(x_ref.dtype, what="fma operands")
            require(inst.out.dtype == x_ref.dtype, "fma output dtype must match operands")
            define(inst.out)
            continue

        if inst.op == "dot":
            require(inst.out is not None, "dot must produce a value")
            require(len(inst.args) == 7, f"dot expects 7 args, got {len(inst.args)}")
            a_buf, a_base, a_stride, b_buf, b_base, b_stride, k_ref = inst.args
            a_buf = require_value_ref(a_buf, what="dot a_buffer")
            b_buf = require_value_ref(b_buf, what="dot b_buffer")
            a_base = require_value_ref(a_base, what="dot a_base")
            a_stride = require_value_ref(a_stride, what="dot a_stride")
            b_base = require_value_ref(b_base, what="dot b_base")
            b_stride = require_value_ref(b_stride, what="dot b_stride")
            k_ref = require_value_ref(k_ref, what="dot K")
            require_defined(a_buf, what="dot a_buffer")
            require_defined(b_buf, what="dot b_buffer")
            require_defined(a_base, what="dot a_base")
            require_defined(a_stride, what="dot a_stride")
            require_defined(b_base, what="dot b_base")
            require_defined(b_stride, what="dot b_stride")
            require_defined(k_ref, what="dot K")
            require(a_buf.id in buffer_value_ids, "dot a_buffer must be a buffer (arg or alloc_tg)")
            require(b_buf.id in buffer_value_ids, "dot b_buffer must be a buffer (arg or alloc_tg)")
            require(
                a_buf.dtype in (DType.F16, DType.F32) and b_buf.dtype in (DType.F16, DType.F32),
                "dot buffers must be f16/f32",
            )
            require(a_base.dtype == DType.U32, "dot a_base must be u32")
            require(a_stride.dtype == DType.U32, "dot a_stride must be u32")
            require(b_base.dtype == DType.U32, "dot b_base must be u32")
            require(b_stride.dtype == DType.U32, "dot b_stride must be u32")
            require(k_ref.dtype == DType.U32, "dot K must be u32")
            require(inst.out.dtype == DType.F32, "dot output must be f32")
            define(inst.out)
            continue

        if inst.op == "mma_zero":
            require(inst.out is not None, "mma_zero must produce a value")
            require(len(inst.args) == 0, f"mma_zero expects 0 args, got {len(inst.args)}")
            require(inst.out.dtype == DType.SG_F32_8X8, "mma_zero currently produces sg_f32_8x8 only")
            define(inst.out)
            continue

        if inst.op == "mma":
            require(inst.out is not None, "mma must produce a value")
            require(len(inst.args) == 7, f"mma expects 7 args, got {len(inst.args)}")
            a_buf, a_base, a_stride, b_buf, b_base, b_stride, acc = inst.args
            a_buf = require_value_ref(a_buf, what="mma a_buffer")
            b_buf = require_value_ref(b_buf, what="mma b_buffer")
            a_base = require_value_ref(a_base, what="mma a_base")
            a_stride = require_value_ref(a_stride, what="mma a_stride")
            b_base = require_value_ref(b_base, what="mma b_base")
            b_stride = require_value_ref(b_stride, what="mma b_stride")
            acc = require_value_ref(acc, what="mma acc")
            require_defined(a_buf, what="mma a_buffer")
            require_defined(b_buf, what="mma b_buffer")
            require_defined(a_base, what="mma a_base")
            require_defined(a_stride, what="mma a_stride")
            require_defined(b_base, what="mma b_base")
            require_defined(b_stride, what="mma b_stride")
            require_defined(acc, what="mma acc")
            require(a_buf.id in buffer_value_ids, "mma a_buffer must be a buffer (arg or alloc_tg)")
            require(b_buf.id in buffer_value_ids, "mma b_buffer must be a buffer (arg or alloc_tg)")
            require(inst.out.dtype == DType.SG_F32_8X8, "mma currently produces sg_f32_8x8 only")
            require(a_buf.dtype in (DType.F16, DType.F32), "mma a_buffer must be f16/f32")
            require(b_buf.dtype in (DType.F16, DType.F32), "mma b_buffer must be f16/f32")
            require(a_base.dtype == DType.U32, "mma a_base must be u32")
            require(a_stride.dtype == DType.U32, "mma a_stride must be u32")
            require(b_base.dtype == DType.U32, "mma b_base must be u32")
            require(b_stride.dtype == DType.U32, "mma b_stride must be u32")
            require(acc.dtype == DType.SG_F32_8X8, "mma acc must be sg_f32_8x8")
            define(inst.out)
            continue

        if inst.op == "mma_store":
            require(inst.out is None, "mma_store must not produce a value")
            require(len(inst.args) == 4, f"mma_store expects 4 args, got {len(inst.args)}")
            c_buf, c_base, c_stride, frag = inst.args
            c_buf = require_value_ref(c_buf, what="mma_store c_buffer")
            c_base = require_value_ref(c_base, what="mma_store c_base")
            c_stride = require_value_ref(c_stride, what="mma_store c_stride")
            frag = require_value_ref(frag, what="mma_store frag")
            require_defined(c_buf, what="mma_store c_buffer")
            require_defined(c_base, what="mma_store c_base")
            require_defined(c_stride, what="mma_store c_stride")
            require_defined(frag, what="mma_store frag")
            require(c_buf.id in buffer_value_ids, "mma_store c_buffer must be a buffer (arg or alloc_tg)")
            require(c_buf.dtype == DType.F32, "mma_store c_buffer must be f32")
            require(c_base.dtype == DType.U32, "mma_store c_base must be u32")
            require(c_stride.dtype == DType.U32, "mma_store c_stride must be u32")
            require(frag.dtype == DType.SG_F32_8X8, "mma_store frag must be sg_f32_8x8")
            continue

        if inst.op == "barrier":
            require(inst.out is None, "barrier must not produce a value")
            require(len(inst.args) == 0, f"barrier expects 0 args, got {len(inst.args)}")
            continue

        require(False, f"unhandled op in validate_ir: {inst.op!r}")

    require(
        seen_arg_names == set(args_by_name.keys()),
        "missing arg inst(s): " + ", ".join(sorted(set(args_by_name) - seen_arg_names)),
    )
