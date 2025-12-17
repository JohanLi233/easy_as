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
    arg_refs: dict[int, Arg] = {}
    seen_arg_names: set[str] = set()

    def require(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"invalid IR: {msg}")

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
            require(
                dtype == args_by_name[name].dtype, f"arg dtype mismatch for {name!r}"
            )
            require(name not in seen_arg_names, f"duplicate arg inst for {name!r}")
            seen_arg_names.add(name)
            arg_refs[inst.out.id] = args_by_name[name]
            continue

        if inst.op == "const":
            require(inst.out is not None, "const must produce a value")
            require(len(inst.args) == 1, f"const expects 1 arg, got {len(inst.args)}")
            continue

        if inst.op == "alloc_tg":
            require(inst.out is not None, "alloc_tg must produce a value")
            require(
                len(inst.args) == 1, f"alloc_tg expects 1 arg, got {len(inst.args)}"
            )
            require(
                isinstance(inst.args[0], int) and int(inst.args[0]) > 0,
                "alloc_tg(size) requires a positive Python int size",
            )
            continue

        if inst.op in {"program_id", "thread_id", "arange"}:
            require(inst.out is not None, f"{inst.op} must produce a value")
            continue

        if inst.op == "local_id":
            require(inst.out is not None, "local_id must produce a value")
            require(
                len(inst.args) == 1, f"local_id expects 1 arg, got {len(inst.args)}"
            )
            require(
                isinstance(inst.args[0], int) and int(inst.args[0]) in (0, 1, 2),
                "local_id axis must be 0/1/2",
            )
            continue

        if inst.op == "lane_id":
            require(inst.out is not None, "lane_id must produce a value")
            require(
                len(inst.args) == 0, f"lane_id expects 0 args, got {len(inst.args)}"
            )
            continue

        if inst.op == "sg_id":
            require(inst.out is not None, "sg_id must produce a value")
            require(len(inst.args) == 0, f"sg_id expects 0 args, got {len(inst.args)}")
            continue

        if inst.op in {"add", "mul", "floordiv", "mod", "lt", "where", "load", "fma"}:
            require(inst.out is not None, f"{inst.op} must produce a value")

        if inst.op == "and":
            require(inst.out is not None, "and must produce a value")
            require(len(inst.args) == 2, f"and expects 2 args, got {len(inst.args)}")
            a_ref, b_ref = inst.args
            require(isinstance(a_ref, ValueRef), "and expects ValueRef args")
            require(isinstance(b_ref, ValueRef), "and expects ValueRef args")
            require(inst.out.dtype == DType.BOOL, "and output must be bool")
            require(a_ref.dtype == DType.BOOL, "and lhs must be bool")
            require(b_ref.dtype == DType.BOOL, "and rhs must be bool")
            continue

        if inst.op == "or":
            require(inst.out is not None, "or must produce a value")
            require(len(inst.args) == 2, f"or expects 2 args, got {len(inst.args)}")
            a_ref, b_ref = inst.args
            require(isinstance(a_ref, ValueRef), "or expects ValueRef args")
            require(isinstance(b_ref, ValueRef), "or expects ValueRef args")
            require(inst.out.dtype == DType.BOOL, "or output must be bool")
            require(a_ref.dtype == DType.BOOL, "or lhs must be bool")
            require(b_ref.dtype == DType.BOOL, "or rhs must be bool")
            continue

        if inst.op == "not":
            require(inst.out is not None, "not must produce a value")
            require(len(inst.args) == 1, f"not expects 1 arg, got {len(inst.args)}")
            (x_ref,) = inst.args
            require(isinstance(x_ref, ValueRef), "not expects a ValueRef arg")
            require(inst.out.dtype == DType.BOOL, "not output must be bool")
            require(x_ref.dtype == DType.BOOL, "not input must be bool")
            continue

        if inst.op == "cast":
            require(inst.out is not None, "cast must produce a value")
            require(len(inst.args) == 2, f"cast expects 2 args, got {len(inst.args)}")
            require(isinstance(inst.args[0], ValueRef), "cast expects a ValueRef input")
            require(isinstance(inst.args[1], DType), "cast expects a DType target")
            continue

        if inst.op == "dot":
            require(inst.out is not None, "dot must produce a value")
            require(len(inst.args) == 7, f"dot expects 7 args, got {len(inst.args)}")
            a_buf, a_base, a_stride, b_buf, b_base, b_stride, k_ref = inst.args
            require(isinstance(a_buf, ValueRef), "dot expects a ValueRef a buffer")
            require(isinstance(b_buf, ValueRef), "dot expects a ValueRef b buffer")
            require(isinstance(a_base, ValueRef), "dot expects a ValueRef a_base")
            require(isinstance(a_stride, ValueRef), "dot expects a ValueRef a_stride")
            require(isinstance(b_base, ValueRef), "dot expects a ValueRef b_base")
            require(isinstance(b_stride, ValueRef), "dot expects a ValueRef b_stride")
            require(isinstance(k_ref, ValueRef), "dot expects a ValueRef K")
            continue

        if inst.op == "mma_zero":
            require(inst.out is not None, "mma_zero must produce a value")
            require(
                inst.out.dtype == DType.SG_F32_8X8,
                "mma_zero currently produces sg_f32_8x8 only",
            )
            require(
                len(inst.args) == 0, f"mma_zero expects 0 args, got {len(inst.args)}"
            )
            continue

        if inst.op == "mma":
            require(inst.out is not None, "mma must produce a value")
            require(len(inst.args) == 7, f"mma expects 7 args, got {len(inst.args)}")
            a_buf, a_base, a_stride, b_buf, b_base, b_stride, acc = inst.args
            require(isinstance(a_buf, ValueRef), "mma expects a ValueRef a buffer")
            require(isinstance(a_base, ValueRef), "mma expects a ValueRef a_base")
            require(isinstance(a_stride, ValueRef), "mma expects a ValueRef a_stride")
            require(isinstance(b_buf, ValueRef), "mma expects a ValueRef b buffer")
            require(isinstance(b_base, ValueRef), "mma expects a ValueRef b_base")
            require(isinstance(b_stride, ValueRef), "mma expects a ValueRef b_stride")
            require(isinstance(acc, ValueRef), "mma expects a ValueRef accumulator")
            require(
                inst.out.dtype == DType.SG_F32_8X8,
                "mma currently produces sg_f32_8x8 only",
            )
            require(
                a_buf.dtype in (DType.F16, DType.F32), "mma a buffer must be f16/f32"
            )
            require(
                b_buf.dtype in (DType.F16, DType.F32), "mma b buffer must be f16/f32"
            )
            require(a_base.dtype == DType.U32, "mma a_base must be u32")
            require(a_stride.dtype == DType.U32, "mma a_stride must be u32")
            require(b_base.dtype == DType.U32, "mma b_base must be u32")
            require(b_stride.dtype == DType.U32, "mma b_stride must be u32")
            require(acc.dtype == DType.SG_F32_8X8, "mma accumulator must be sg_f32_8x8")
            continue

        if inst.op == "mma_store":
            require(inst.out is None, "mma_store must not produce a value")
            require(
                len(inst.args) == 4, f"mma_store expects 4 args, got {len(inst.args)}"
            )
            c_buf, c_base, c_stride, frag = inst.args
            require(
                isinstance(c_buf, ValueRef), "mma_store expects a ValueRef c buffer"
            )
            require(isinstance(c_base, ValueRef), "mma_store expects a ValueRef c_base")
            require(
                isinstance(c_stride, ValueRef), "mma_store expects a ValueRef c_stride"
            )
            require(isinstance(frag, ValueRef), "mma_store expects a ValueRef fragment")
            require(c_buf.dtype == DType.F32, "mma_store c buffer must be f32")
            require(c_base.dtype == DType.U32, "mma_store c_base must be u32")
            require(c_stride.dtype == DType.U32, "mma_store c_stride must be u32")
            require(
                frag.dtype == DType.SG_F32_8X8, "mma_store fragment must be sg_f32_8x8"
            )
            continue

        if inst.op == "store":
            require(inst.out is None, "store must not produce a value")

        if inst.op == "barrier":
            require(inst.out is None, "barrier must not produce a value")
            require(
                len(inst.args) == 0, f"barrier expects 0 args, got {len(inst.args)}"
            )

    require(
        seen_arg_names == set(args_by_name),
        "missing arg inst(s): " + ", ".join(sorted(set(args_by_name) - seen_arg_names)),
    )
