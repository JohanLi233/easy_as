# filename: eas/ir.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal, TypeAlias


class DType(str, Enum):
    BOOL = "bool"
    F32 = "f32"
    U32 = "u32"


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
    "where",
    "cast",
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

        if inst.op == "cast":
            require(inst.out is not None, "cast must produce a value")
            require(len(inst.args) == 2, f"cast expects 2 args, got {len(inst.args)}")
            require(isinstance(inst.args[0], ValueRef), "cast expects a ValueRef input")
            require(isinstance(inst.args[1], DType), "cast expects a DType target")
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
