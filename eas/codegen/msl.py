# filename: eas/codegen/msl.py

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
    def_by_id: dict[int, Any] = {}

    def _has_barrier_or_threadgroup_memory() -> bool:
        """
        Conservative check used for early-return lowering.

        Kernels that use threadgroup memory or barriers must not early-return
        based on a per-thread predicate, otherwise some threads may skip a
        barrier that other threads reach.
        """

        barrier_ops = {
            "barrier",
            "threadgroup_barrier",
            "tg_barrier",
            "simdgroup_barrier",
        }
        threadgroup_mem_ops = {
            "alloc_tg",
            "threadgroup_alloc",
            "alloc_threadgroup",
            "threadgroup_load",
            "threadgroup_store",
            "tgm_load",
            "tgm_store",
        }
        for inst in ir.insts:
            op = str(inst.op)
            if op in barrier_ops or op in threadgroup_mem_ops:
                return True
        return False

    # Extract arg name mapping and store targets.
    for inst in ir.insts:
        if inst.op == "arg":
            assert inst.out is not None
            name = str(inst.args[0])
            arg_names[inst.out.id] = name
            def_by_id[inst.out.id] = inst
        elif inst.op == "store":
            buf_ref: ValueRef = inst.args[0]
            buf_name = arg_names.get(buf_ref.id)
            if buf_name is not None:
                uses_store.add(buf_name)
        elif inst.out is not None:
            def_by_id[inst.out.id] = inst

    ctx = _Ctx(arg_names=arg_names, uses_store=uses_store)

    threadgroup_size = _infer_threadgroup_size(ir)

    use_counts: dict[int, int] = {}
    for inst in ir.insts:
        for a in inst.args:
            if isinstance(a, ValueRef):
                use_counts[a.id] = use_counts.get(a.id, 0) + 1

    def _is_const_bool(v: ValueRef, expected: bool) -> bool:
        inst = def_by_id.get(v.id)
        return bool(
            inst is not None
            and inst.op == "const"
            and isinstance(inst.args[0], bool)
            and bool(inst.args[0]) is expected
        )

    def _lifted_region_by_store_idx() -> dict[int, list[int]]:
        def_idx: dict[int, int] = {}
        for i, inst in enumerate(ir.insts):
            if inst.out is not None:
                def_idx[inst.out.id] = i

        uses: dict[int, set[int]] = {}

        def _add_use(v: ValueRef, idx: int) -> None:
            uses.setdefault(v.id, set()).add(idx)

        for i, inst in enumerate(ir.insts):
            for a in inst.args:
                if isinstance(a, ValueRef):
                    _add_use(a, i)

        def _deps(value_id: int, out: set[int]) -> None:
            """
            Collect transitive ValueRef dependencies for a given value id.

            Use an explicit stack (not recursion) so large unrolled kernels
            (e.g. big constexpr loops) don't hit Python recursion limits.
            """
            stack: list[int] = [value_id]
            while stack:
                vid = stack.pop()
                if vid in out:
                    continue
                out.add(vid)
                inst = def_by_id.get(vid)
                if inst is None or inst.op == "arg":
                    continue
                for a in inst.args:
                    if isinstance(a, ValueRef) and a.id not in out:
                        stack.append(a.id)

        moved: set[int] = set()
        lifted: dict[int, list[int]] = {}

        for store_idx, inst in enumerate(ir.insts):
            if inst.op != "store":
                continue
            buf_ref, off_ref, val_ref, mask_ref = inst.args  # type: ignore[misc]
            if _is_const_bool(mask_ref, True):
                continue

            mask_ids: set[int] = set()
            _deps(mask_ref.id, mask_ids)
            off_ids: set[int] = set()
            _deps(off_ref.id, off_ids)
            anchor_ids = mask_ids | off_ids

            val_ids: set[int] = set()
            _deps(val_ref.id, val_ids)

            candidate_ids = val_ids - anchor_ids
            if not candidate_ids:
                continue

            candidate_inst_idxs: set[int] = set()
            ok = True
            for value_id in candidate_ids:
                idx = def_idx.get(value_id)
                if idx is None:
                    continue
                if idx >= store_idx:
                    ok = False
                    break
                if idx in moved:
                    ok = False
                    break
                def_inst = ir.insts[idx]
                if def_inst.op in {"arg", "program_id", "thread_id", "arange"}:
                    continue
                if def_inst.op == "load":
                    _buf, _off, load_mask = def_inst.args  # type: ignore[misc]
                    if load_mask.id != mask_ref.id:
                        ok = False
                        break
                elif def_inst.op in {
                    "add",
                    "mul",
                    "fma",
                    "floordiv",
                    "mod",
                    "lt",
                    "where",
                    "const",
                }:
                    pass
                else:
                    ok = False
                    break
                candidate_inst_idxs.add(idx)

            if not ok or not candidate_inst_idxs:
                continue

            allowed_users = candidate_inst_idxs | {store_idx}
            moved_value_ids = {
                ir.insts[i].out.id
                for i in candidate_inst_idxs
                if ir.insts[i].out is not None
            }
            for value_id in moved_value_ids:
                for user_idx in uses.get(value_id, set()):
                    if user_idx not in allowed_users:
                        ok = False
                        break
                if not ok:
                    break

            if not ok:
                continue

            ordered = sorted(candidate_inst_idxs)
            lifted[store_idx] = ordered
            moved.update(candidate_inst_idxs)

        return lifted

    has_barrier_or_tgmem = _has_barrier_or_threadgroup_memory()
    # Lifting computations into a per-thread store guard is unsafe when barriers or
    # threadgroup memory are involved: it can move instructions across barriers or
    # (worse) put barriers behind divergent control flow.
    lifted_by_store = {} if has_barrier_or_tgmem else _lifted_region_by_store_idx()
    store_idxs = [i for i, inst in enumerate(ir.insts) if inst.op == "store"]
    allow_early_return = len(store_idxs) == 1 and not has_barrier_or_tgmem
    early_return_store_idx = store_idxs[0] if allow_early_return else None

    needs_tgpig = False
    needs_tpitg = False
    needs_tpig = False
    needs_lane = False
    needs_sg = False
    for inst in ir.insts:
        if inst.out is None:
            continue
        if use_counts.get(inst.out.id, 0) == 0:
            continue
        if inst.op == "program_id":
            needs_tgpig = True
        elif inst.op == "arange":
            needs_tpitg = True
        elif inst.op == "thread_id":
            needs_tpig = True
        elif inst.op == "local_id":
            needs_tpitg = True
        elif inst.op == "lane_id":
            needs_lane = True
        elif inst.op == "sg_id":
            needs_sg = True

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
            params.append(
                f"device {const_kw}float* __restrict {arg.name} [[buffer({buf_index})]]"
            )
        else:
            params.append(
                f"constant {_msl_type(arg.dtype)}& {arg.name} [[buffer({buf_index})]]"
            )
        buf_index += 1

    if needs_tpig:
        params.append("uint3 tpig [[thread_position_in_grid]]")
    if needs_tgpig:
        params.append("uint3 tgpig [[threadgroup_position_in_grid]]")
    if needs_tpitg:
        params.append("uint3 tpitg [[thread_position_in_threadgroup]]")
    if needs_lane:
        params.append("uint lane [[thread_index_in_simdgroup]]")
    if needs_sg:
        params.append("uint sg [[simdgroup_index_in_threadgroup]]")

    lines.append(",\n".join(f"    {p}" for p in params))
    lines.append(") {")

    # threadgroup allocations (declare at top of kernel body)
    alloc_tg_inst_idxs: set[int] = set()
    for idx, inst in enumerate(ir.insts):
        if inst.op != "alloc_tg":
            continue
        assert inst.out is not None
        (size,) = inst.args
        lines.append(
            f"  threadgroup {_msl_type(inst.out.dtype)} v{inst.out.id}[{int(size)}];"
        )
        alloc_tg_inst_idxs.add(idx)

    # emit body
    moved_inst_idxs = {idx for idxs in lifted_by_store.values() for idx in idxs}

    def _emit_inst(
        inst: Any, *, indent: str, in_guard_for_mask: ValueRef | None
    ) -> None:
        if inst.op in {"arg"}:
            return
        if inst.op == "alloc_tg":
            return
        if inst.op == "const":
            out = inst.out
            assert out is not None
            lines.append(
                f"{indent}{_msl_type(out.dtype)} v{out.id} = {_const_literal(inst.args[0])};"
            )
            return
        if inst.op == "program_id":
            out = inst.out
            assert out is not None
            axis = int(inst.args[0])
            comp = {0: "x", 1: "y", 2: "z"}.get(axis)
            if comp is None:
                raise ValueError(f"program_id axis must be 0/1/2, got {axis}")
            lines.append(f"{indent}uint v{out.id} = tgpig.{comp};")
            return
        if inst.op == "thread_id":
            out = inst.out
            assert out is not None
            axis = int(inst.args[0])
            comp = {0: "x", 1: "y", 2: "z"}.get(axis)
            if comp is None:
                raise ValueError(f"thread_id axis must be 0/1/2, got {axis}")
            lines.append(f"{indent}uint v{out.id} = tpig.{comp};")
            return
        if inst.op == "local_id":
            out = inst.out
            assert out is not None
            axis = int(inst.args[0])
            comp = {0: "x", 1: "y", 2: "z"}.get(axis)
            if comp is None:
                raise ValueError(f"local_id axis must be 0/1/2, got {axis}")
            lines.append(f"{indent}uint v{out.id} = tpitg.{comp};")
            return
        if inst.op == "lane_id":
            out = inst.out
            assert out is not None
            lines.append(f"{indent}uint v{out.id} = lane;")
            return
        if inst.op == "sg_id":
            out = inst.out
            assert out is not None
            lines.append(f"{indent}uint v{out.id} = sg;")
            return
        if inst.op == "arange":
            out = inst.out
            assert out is not None
            start, size = (int(inst.args[0]), int(inst.args[1]))
            _ = size
            if start != 0:
                lines.append(f"{indent}uint v{out.id} = (uint)({start}) + tpitg.x;")
            else:
                lines.append(f"{indent}uint v{out.id} = tpitg.x;")
            return
        if inst.op in {"add", "mul", "lt"}:
            out = inst.out
            assert out is not None
            a: ValueRef
            b: ValueRef
            a, b = inst.args  # type: ignore[misc]
            op = {"add": "+", "mul": "*", "lt": "<"}[inst.op]
            lines.append(
                f"{indent}{_msl_type(out.dtype)} v{out.id} = {_ref(ctx, a)} {op} {_ref(ctx, b)};"
            )
            return
        if inst.op in {"floordiv", "mod"}:
            out = inst.out
            assert out is not None
            a, b = inst.args  # type: ignore[misc]
            op = {"floordiv": "/", "mod": "%"}[inst.op]
            lines.append(
                f"{indent}{_msl_type(out.dtype)} v{out.id} = {_ref(ctx, a)} {op} {_ref(ctx, b)};"
            )
            return
        if inst.op == "where":
            out = inst.out
            assert out is not None
            c, a, b = inst.args  # type: ignore[misc]
            lines.append(
                f"{indent}{_msl_type(out.dtype)} v{out.id} = {_ref(ctx, c)} ? {_ref(ctx, a)} : {_ref(ctx, b)};"
            )
            return
        if inst.op == "cast":
            out = inst.out
            assert out is not None
            x_ref, dst = inst.args  # type: ignore[misc]
            if not isinstance(dst, DType):
                raise TypeError("cast expects DType target")
            lines.append(
                f"{indent}{_msl_type(out.dtype)} v{out.id} = ({_msl_type(dst)})({_ref(ctx, x_ref)});"
            )
            return
        if inst.op == "fma":
            out = inst.out
            assert out is not None
            x, y, z = inst.args  # type: ignore[misc]
            lines.append(
                f"{indent}{_msl_type(out.dtype)} v{out.id} = fma({_ref(ctx, x)}, {_ref(ctx, y)}, {_ref(ctx, z)});"
            )
            return
        if inst.op == "load":
            out = inst.out
            assert out is not None
            buf_ref, off_ref, mask_ref = inst.args  # type: ignore[misc]
            buf_name = _ref(ctx, buf_ref)

            if in_guard_for_mask is not None and mask_ref.id == in_guard_for_mask.id:
                lines.append(
                    f"{indent}float v{out.id} = {buf_name}[{_ref(ctx, off_ref)}];"
                )
            elif _is_const_bool(mask_ref, True):
                lines.append(
                    f"{indent}float v{out.id} = {buf_name}[{_ref(ctx, off_ref)}];"
                )
            else:
                lines.append(f"{indent}float v{out.id} = 0.0f;")
                lines.append(
                    f"{indent}if ({_ref(ctx, mask_ref)}) {{ v{out.id} = {buf_name}[{_ref(ctx, off_ref)}]; }}"
                )
            return
        if inst.op == "store":
            buf_ref, off_ref, val_ref, mask_ref = inst.args  # type: ignore[misc]
            buf_name = _ref(ctx, buf_ref)
            if in_guard_for_mask is not None and mask_ref.id == in_guard_for_mask.id:
                lines.append(
                    f"{indent}{buf_name}[{_ref(ctx, off_ref)}] = {_ref(ctx, val_ref)};"
                )
            else:
                lines.append(
                    f"{indent}if ({_ref(ctx, mask_ref)}) {{ {buf_name}[{_ref(ctx, off_ref)}] = {_ref(ctx, val_ref)}; }}"
                )
            return
        if inst.op == "barrier":
            lines.append(f"{indent}threadgroup_barrier(mem_flags::mem_threadgroup);")
            return
        raise ValueError(f"unsupported op: {inst.op}")

    for idx, inst in enumerate(ir.insts):
        if (
            inst.out is not None
            and use_counts.get(inst.out.id, 0) == 0
            and inst.op != "store"
        ):
            # Skip unused SSA defs (notably arange kept for threadgroup_size).
            continue
        if idx in moved_inst_idxs:
            continue
        if idx in alloc_tg_inst_idxs:
            continue

        if inst.op in {"arg"}:
            continue
        if inst.op == "store" and idx in lifted_by_store:
            _buf_ref, _off_ref, _val_ref, mask_ref = inst.args  # type: ignore[misc]
            if (
                early_return_store_idx == idx
                and isinstance(mask_ref, ValueRef)
                and not _is_const_bool(mask_ref, True)
            ):
                lines.append(f"  if (!{_ref(ctx, mask_ref)}) return;")
                for moved_idx in lifted_by_store[idx]:
                    _emit_inst(
                        ir.insts[moved_idx], indent="  ", in_guard_for_mask=mask_ref
                    )
                _emit_inst(inst, indent="  ", in_guard_for_mask=mask_ref)
            else:
                lines.append(f"  if ({_ref(ctx, mask_ref)}) {{")
                for moved_idx in lifted_by_store[idx]:
                    _emit_inst(
                        ir.insts[moved_idx], indent="    ", in_guard_for_mask=mask_ref
                    )
                _emit_inst(inst, indent="    ", in_guard_for_mask=mask_ref)
                lines.append("  }")
        elif inst.op == "store" and early_return_store_idx == idx:
            _buf_ref, _off_ref, _val_ref, mask_ref = inst.args  # type: ignore[misc]
            if (
                isinstance(mask_ref, ValueRef)
                and not _is_const_bool(mask_ref, True)
                and allow_early_return
            ):
                lines.append(f"  if (!{_ref(ctx, mask_ref)}) return;")
                _emit_inst(inst, indent="  ", in_guard_for_mask=mask_ref)
            else:
                _emit_inst(inst, indent="  ", in_guard_for_mask=None)
        else:
            _emit_inst(inst, indent="  ", in_guard_for_mask=None)

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
