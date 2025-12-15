from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class DType(str, Enum):
    BOOL = "bool"
    F32 = "f32"
    U32 = "u32"


@dataclass(frozen=True, slots=True)
class ValueRef:
    id: int
    dtype: DType


@dataclass(frozen=True, slots=True)
class Arg:
    name: str
    dtype: DType
    kind: str  # "buffer" | "scalar"


@dataclass(frozen=True, slots=True)
class Inst:
    op: str
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
