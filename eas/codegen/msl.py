from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ir import DType, IRModule, ValueRef


def _msl_type(dt: DType) -> str:
    if dt == DType.F32:
        return "float"
    if dt == DType.U32:
        return "uint"
    if dt == DType.BOOL:
        return "bool"
    raise ValueError(f"unsupported dtype: {dt}")


def _const_literal(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        if v < 0:
            raise ValueError("negative constants not supported in MVP")
        return str(v)
    if isinstance(v, float):
        return f"{v}f"
    raise TypeError(f"unsupported const literal: {type(v)!r}")


@dataclass(frozen=True, slots=True)
class _Ctx:
    arg_names: dict[int, str]  # value_id -> arg name
    uses_store: set[str]


def _ref(ctx: _Ctx, v: ValueRef) -> str:
    return ctx.arg_names.get(v.id, f"v{v.id}")


def ir_to_msl(ir: IRModule) -> tuple[str, int]:
    arg_names: dict[int, str] = {}
    uses_store: set[str] = set()

    # Extract arg name mapping and store targets.
    for inst in ir.insts:
        if inst.op == "arg":
            assert inst.out is not None
            name = str(inst.args[0])
            arg_names[inst.out.id] = name
        elif inst.op == "store":
            buf_ref: ValueRef = inst.args[0]
            buf_name = arg_names.get(buf_ref.id)
            if buf_name is not None:
                uses_store.add(buf_name)

    ctx = _Ctx(arg_names=arg_names, uses_store=uses_store)

    threadgroup_size = _infer_threadgroup_size(ir)

    lines: list[str] = []
    lines.append("#include <metal_stdlib>")
    lines.append("using namespace metal;")
    lines.append("")
    lines.append(f"kernel void {ir.name}(")

    # kernel args (buffers/scalars) share buffer index space
    params: list[str] = []
    buf_index = 0
    for arg in ir.args:
        if arg.kind == "buffer":
            const_kw = "" if arg.name in uses_store else "const "
            params.append(f"device {const_kw}float* {arg.name} [[buffer({buf_index})]]")
        else:
            params.append(
                f"constant {_msl_type(arg.dtype)}& {arg.name} [[buffer({buf_index})]]"
            )
        buf_index += 1

    params.append("uint3 tgpig [[threadgroup_position_in_grid]]")
    params.append("uint3 tpitg [[thread_position_in_threadgroup]]")

    lines.append(",\n".join(f"    {p}" for p in params))
    lines.append(") {")

    # emit body
    for inst in ir.insts:
        if inst.op in {"arg"}:
            continue
        if inst.op == "const":
            out = inst.out
            assert out is not None
            lines.append(
                f"  {_msl_type(out.dtype)} v{out.id} = {_const_literal(inst.args[0])};"
            )
            continue
        if inst.op == "program_id":
            out = inst.out
            assert out is not None
            axis = int(inst.args[0])
            if axis != 0:
                raise ValueError("only program_id(0) is supported in MVP")
            lines.append(f"  uint v{out.id} = tgpig.x;")
            continue
        if inst.op == "arange":
            out = inst.out
            assert out is not None
            start, size = (int(inst.args[0]), int(inst.args[1]))
            _ = size
            if start != 0:
                lines.append(f"  uint v{out.id} = (uint)({start}) + tpitg.x;")
            else:
                lines.append(f"  uint v{out.id} = tpitg.x;")
            continue
        if inst.op in {"add", "mul", "lt"}:
            out = inst.out
            assert out is not None
            a: ValueRef
            b: ValueRef
            a, b = inst.args  # type: ignore[misc]
            op = {"add": "+", "mul": "*", "lt": "<"}[inst.op]
            lines.append(
                f"  {_msl_type(out.dtype)} v{out.id} = {_ref(ctx, a)} {op} {_ref(ctx, b)};"
            )
            continue
        if inst.op == "where":
            out = inst.out
            assert out is not None
            c, a, b = inst.args  # type: ignore[misc]
            lines.append(
                f"  {_msl_type(out.dtype)} v{out.id} = {_ref(ctx, c)} ? {_ref(ctx, a)} : {_ref(ctx, b)};"
            )
            continue
        if inst.op == "load":
            out = inst.out
            assert out is not None
            buf_ref, off_ref, mask_ref = inst.args  # type: ignore[misc]
            buf_name = ctx.arg_names[buf_ref.id]
            lines.append(
                f"  float v{out.id} = {_ref(ctx, mask_ref)} ? {buf_name}[{_ref(ctx, off_ref)}] : 0.0f;"
            )
            continue
        if inst.op == "store":
            buf_ref, off_ref, val_ref, mask_ref = inst.args  # type: ignore[misc]
            buf_name = ctx.arg_names[buf_ref.id]
            lines.append(
                f"  if ({_ref(ctx, mask_ref)}) {{ {buf_name}[{_ref(ctx, off_ref)}] = {_ref(ctx, val_ref)}; }}"
            )
            continue
        raise ValueError(f"unsupported op: {inst.op}")

    lines.append("}")
    return ("\n".join(lines) + "\n", threadgroup_size)


def _infer_threadgroup_size(ir: IRModule) -> int:
    size: int | None = None
    for inst in ir.insts:
        if inst.op == "arange":
            inst_size = int(inst.args[1])
            if size is None:
                size = inst_size
            elif size != inst_size:
                raise ValueError("multiple arange sizes not supported in MVP")
    if size is None:
        raise ValueError(
            "kernel must use mk.arange(0, BLOCK) to define threadgroup size"
        )
    return size
