# filename: eas/codegen/msl.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ir import DType, IRModule, ValueRef


def _msl_type(dt: DType) -> str:
    if dt == DType.F32:
        return "float"
    if dt == DType.F16:
        return "half"
    if dt == DType.U32:
        return "uint"
    if dt == DType.BOOL:
        return "bool"
    if dt == DType.SG_F32_8X8:
        return "simdgroup_float8x8"
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


def _zero_literal(dt: DType) -> str:
    if dt == DType.F32:
        return "0.0f"
    if dt == DType.F16:
        return "half(0.0f)"
    if dt == DType.U32:
        return "0u"
    if dt == DType.BOOL:
        return "false"
    raise ValueError(f"unsupported zero literal dtype: {dt}")


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
            "mma_zero",
            "mma",
            "mma_store",
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
        elif inst.op == "mma_store":
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

    def _const_u32(v: ValueRef) -> int | None:
        inst = def_by_id.get(v.id)
        if inst is None or inst.op != "const":
            return None
        if not isinstance(inst.args[0], int):
            return None
        if v.dtype != DType.U32:
            return None
        return int(inst.args[0])

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
    uses_simdgroup_matrix = False
    for inst in ir.insts:
        if inst.op in {"mma_zero", "mma", "mma_store"}:
            uses_simdgroup_matrix = True
            needs_lane = True
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
    if uses_simdgroup_matrix:
        lines.append("#include <metal_simdgroup_matrix>")
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
        if inst.op in {"and", "or"}:
            out = inst.out
            assert out is not None
            a, b = inst.args  # type: ignore[misc]
            op = "&&" if inst.op == "and" else "||"
            lines.append(
                f"{indent}bool v{out.id} = ({_ref(ctx, a)} {op} {_ref(ctx, b)});"
            )
            return
        if inst.op == "not":
            out = inst.out
            assert out is not None
            (x,) = inst.args  # type: ignore[misc]
            lines.append(f"{indent}bool v{out.id} = !{_ref(ctx, x)};")
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
            out_ty = _msl_type(out.dtype)
            zero = _zero_literal(out.dtype)

            if in_guard_for_mask is not None and mask_ref.id == in_guard_for_mask.id:
                lines.append(
                    f"{indent}{out_ty} v{out.id} = {buf_name}[{_ref(ctx, off_ref)}];"
                )
            elif _is_const_bool(mask_ref, True):
                lines.append(
                    f"{indent}{out_ty} v{out.id} = {buf_name}[{_ref(ctx, off_ref)}];"
                )
            else:
                lines.append(f"{indent}{out_ty} v{out.id} = {zero};")
                lines.append(
                    f"{indent}if ({_ref(ctx, mask_ref)}) {{ v{out.id} = {buf_name}[{_ref(ctx, off_ref)}]; }}"
                )
            return
        if inst.op == "dot":
            out = inst.out
            assert out is not None
            (
                a_buf,
                a_base,
                a_stride,
                b_buf,
                b_base,
                b_stride,
                k_ref,
            ) = inst.args  # type: ignore[misc]
            a_name = _ref(ctx, a_buf)
            b_name = _ref(ctx, b_buf)
            k_name = _ref(ctx, k_ref)
            k_var = f"k{out.id}"
            lines.append(f"{indent}float v{out.id} = 0.0f;")
            lines.append(f"{indent}uint a_idx{out.id} = {_ref(ctx, a_base)};")
            lines.append(f"{indent}uint b_idx{out.id} = {_ref(ctx, b_base)};")
            k_const = _const_u32(k_ref)
            if k_const is not None and 1 <= k_const <= 32:
                lines.append(f"{indent}#pragma clang loop unroll_count({k_const})")
            lines.append(
                f"{indent}for (uint {k_var} = 0; {k_var} < {k_name}; ++{k_var}) {{"
            )
            lines.append(
                f"{indent}  v{out.id} = fma({a_name}[a_idx{out.id}], {b_name}[b_idx{out.id}], v{out.id});"
            )
            lines.append(
                f"{indent}  a_idx{out.id} += {_ref(ctx, a_stride)}; b_idx{out.id} += {_ref(ctx, b_stride)};"
            )
            lines.append(f"{indent}}}")
            return
        if inst.op == "mma_zero":
            out = inst.out
            assert out is not None
            lines.append(
                f"{indent}{_msl_type(out.dtype)} v{out.id} = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
            )
            return
        if inst.op == "mma":
            out = inst.out
            assert out is not None
            (
                a_buf,
                a_base,
                a_stride,
                b_buf,
                b_base,
                b_stride,
                acc,
            ) = inst.args  # type: ignore[misc]
            a_mat = (
                "simdgroup_half8x8"
                if a_buf.dtype == DType.F16
                else "simdgroup_float8x8"
            )
            b_mat = (
                "simdgroup_half8x8"
                if b_buf.dtype == DType.F16
                else "simdgroup_float8x8"
            )
            a_tmp = f"v{out.id}_a"
            b_tmp = f"v{out.id}_b"
            lines.append(f"{indent}{a_mat} {a_tmp};")
            lines.append(f"{indent}{b_mat} {b_tmp};")
            lines.append(
                f"{indent}simdgroup_load({a_tmp}, {_ref(ctx, a_buf)} + {_ref(ctx, a_base)}, {_ref(ctx, a_stride)});"
            )
            lines.append(
                f"{indent}simdgroup_load({b_tmp}, {_ref(ctx, b_buf)} + {_ref(ctx, b_base)}, {_ref(ctx, b_stride)});"
            )
            lines.append(f"{indent}{_msl_type(out.dtype)} v{out.id};")
            lines.append(
                f"{indent}simdgroup_multiply_accumulate(v{out.id}, {a_tmp}, {b_tmp}, {_ref(ctx, acc)});"
            )
            return
        if inst.op == "mma_store":
            c_buf, c_base, c_stride, frag = inst.args  # type: ignore[misc]
            lines.append(
                f"{indent}simdgroup_store({_ref(ctx, frag)}, {_ref(ctx, c_buf)} + {_ref(ctx, c_base)}, {_ref(ctx, c_stride)});"
            )
            return
        if inst.op == "store":
            buf_ref, off_ref, val_ref, mask_ref = inst.args  # type: ignore[misc]
            buf_name = _ref(ctx, buf_ref)
            if _is_const_bool(mask_ref, True):
                lines.append(
                    f"{indent}{buf_name}[{_ref(ctx, off_ref)}] = {_ref(ctx, val_ref)};"
                )
            elif in_guard_for_mask is not None and mask_ref.id == in_guard_for_mask.id:
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

    def _emit_dot_group(start_idx: int, *, indent: str) -> int:
        inst0 = ir.insts[start_idx]
        assert inst0.op == "dot"
        assert inst0.out is not None
        (
            a_buf0,
            a_base0,
            a_stride0,
            b_buf0,
            _b_base0,
            b_stride0,
            k_ref0,
        ) = inst0.args  # type: ignore[misc]

        def _sig(inst: Any) -> tuple[int, int, int, int, int, int]:
            (
                a_buf,
                a_base,
                a_stride,
                b_buf,
                _b_base,
                b_stride,
                k_ref,
            ) = inst.args  # type: ignore[misc]
            return (
                a_buf.id,
                a_base.id,
                a_stride.id,
                b_buf.id,
                b_stride.id,
                k_ref.id,
            )

        sig0 = _sig(inst0)
        group: list[Any] = [inst0]
        idx = start_idx + 1
        while idx < len(ir.insts) and len(group) < 8:
            inst = ir.insts[idx]
            if inst.op != "dot" or inst.out is None:
                break
            if _sig(inst) != sig0:
                break
            group.append(inst)
            idx += 1

        a_name = _ref(ctx, a_buf0)
        b_name = _ref(ctx, b_buf0)
        k_name = _ref(ctx, k_ref0)
        k_var = f"k{inst0.out.id}"

        for inst in group:
            assert inst.out is not None
            lines.append(f"{indent}float v{inst.out.id} = 0.0f;")

        a_idx_name = f"a_idx{inst0.out.id}"
        lines.append(f"{indent}uint {a_idx_name} = {_ref(ctx, a_base0)};")
        b_idx_vars: list[str] = []
        for inst in group:
            assert inst.out is not None
            (
                _a_buf,
                _a_base,
                _a_stride,
                _b_buf,
                b_base,
                _b_stride,
                _k_ref,
            ) = inst.args  # type: ignore[misc]
            b_idx = f"b_idx{inst.out.id}"
            b_idx_vars.append(b_idx)
            lines.append(f"{indent}uint {b_idx} = {_ref(ctx, b_base)};")

        k_const = _const_u32(k_ref0)
        if k_const is not None and 1 <= k_const <= 32:
            lines.append(f"{indent}#pragma clang loop unroll_count({k_const})")
        lines.append(
            f"{indent}for (uint {k_var} = 0; {k_var} < {k_name}; ++{k_var}) {{"
        )
        lines.append(f"{indent}  float a_val = {a_name}[{a_idx_name}];")
        for inst, b_idx in zip(group, b_idx_vars, strict=True):
            assert inst.out is not None
            lines.append(
                f"{indent}  v{inst.out.id} = fma(a_val, {b_name}[{b_idx}], v{inst.out.id});"
            )
            lines.append(f"{indent}  {b_idx} += {_ref(ctx, b_stride0)};")
        lines.append(f"{indent}  {a_idx_name} += {_ref(ctx, a_stride0)};")
        lines.append(f"{indent}}}")
        return len(group)

    idx = 0
    while idx < len(ir.insts):
        inst = ir.insts[idx]
        if (
            inst.out is not None
            and use_counts.get(inst.out.id, 0) == 0
            and inst.op != "store"
        ):
            idx += 1
            continue
        if idx in moved_inst_idxs or idx in alloc_tg_inst_idxs:
            idx += 1
            continue
        if inst.op in {"arg"}:
            idx += 1
            continue

        if inst.op == "dot":
            idx += _emit_dot_group(idx, indent="  ")
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
            idx += 1
            continue

        if inst.op == "store" and early_return_store_idx == idx:
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
            idx += 1
            continue

        _emit_inst(inst, indent="  ", in_guard_for_mask=None)
        idx += 1

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
