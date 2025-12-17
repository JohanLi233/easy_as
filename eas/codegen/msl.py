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
    def_inst_idx_by_value_id: dict[int, int] = {}

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

    for idx, inst in enumerate(ir.insts):
        if inst.out is not None:
            def_inst_idx_by_value_id[inst.out.id] = idx

    ctx = _Ctx(arg_names=arg_names, uses_store=uses_store)

    threadgroup_size = _infer_threadgroup_size(ir)

    use_counts: dict[int, int] = {}
    for inst in ir.insts:
        for a in inst.args:
            if isinstance(a, ValueRef):
                use_counts[a.id] = use_counts.get(a.id, 0) + 1

    inline_bool_ids: set[int] = set()
    inline_bool_expr_by_id: dict[int, str] = {}

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

    def _const_int(v: ValueRef) -> int | None:
        inst = def_by_id.get(v.id)
        if inst is None or inst.op != "const":
            return None
        if not isinstance(inst.args[0], int):
            return None
        return int(inst.args[0])

    def _buf_addr_space(buf: ValueRef) -> str | None:
        inst = def_by_id.get(buf.id)
        if inst is None:
            return None
        if inst.op == "arg":
            # kind is arg.args[1], but both scalar/buffer use buffer(index) space;
            # only buffer args can appear as load/store/dot buffers.
            return "device"
        if inst.op == "alloc_tg":
            return "threadgroup"
        return None

    def _load_as_float(buf: ValueRef, idx_expr: str) -> str:
        name = _ref(ctx, buf)
        if buf.dtype == DType.F32:
            return f"{name}[{idx_expr}]"
        if buf.dtype == DType.F16:
            return f"float({name}[{idx_expr}])"
        raise ValueError(f"unsupported dot buffer dtype: {buf.dtype}")

    def _base_plus_const(ref: ValueRef) -> tuple[ValueRef, int] | None:
        """
        Match `ref = add(base, const)` (commuted by CSE) and return (base, const).
        """
        inst = def_by_id.get(ref.id)
        if inst is None or inst.op != "add":
            return None
        a, b = inst.args  # type: ignore[misc]
        if not isinstance(a, ValueRef) or not isinstance(b, ValueRef):
            return None
        a_c = _const_int(a)
        b_c = _const_int(b)
        if a_c is not None and b_c is None:
            return (b, a_c)
        if b_c is not None and a_c is None:
            return (a, b_c)
        return None

    def _try_inline_bool_expr(v: ValueRef) -> tuple[str, set[int]] | None:
        """
        Inline small boolean expression trees for masks/guards.

        This is used to turn nested `where(...)`-lowered boolean trees into a
        single `&&` / `||` / `!` expression in MSL, reducing temporary bool SSA
        values and register pressure.
        """
        if v.dtype != DType.BOOL:
            return None
        if use_counts.get(v.id, 0) != 1:
            return None
        inst = def_by_id.get(v.id)
        if inst is None:
            return None
        if inst.op == "const" and isinstance(inst.args[0], bool):
            return ("true" if bool(inst.args[0]) else "false", {v.id})
        if inst.op == "lt":
            a, b = inst.args  # type: ignore[misc]
            assert isinstance(a, ValueRef) and isinstance(b, ValueRef)
            return (f"({_ref(ctx, a)} < {_ref(ctx, b)})", {v.id})
        if inst.op in {"and", "or"}:
            a, b = inst.args  # type: ignore[misc]
            assert isinstance(a, ValueRef) and isinstance(b, ValueRef)
            a_expr = _try_inline_bool_expr(a)
            b_expr = _try_inline_bool_expr(b)
            op = "&&" if inst.op == "and" else "||"
            expr = f"({(a_expr[0] if a_expr else _ref(ctx, a))} {op} {(b_expr[0] if b_expr else _ref(ctx, b))})"
            ids: set[int] = {v.id}
            if a_expr is not None:
                ids |= a_expr[1]
            if b_expr is not None:
                ids |= b_expr[1]
            return (expr, ids)
        if inst.op == "not":
            (x,) = inst.args  # type: ignore[misc]
            assert isinstance(x, ValueRef)
            x_expr = _try_inline_bool_expr(x)
            expr = f"(!{(x_expr[0] if x_expr else _ref(ctx, x))})"
            ids = {v.id}
            if x_expr is not None:
                ids |= x_expr[1]
            return (expr, ids)
        return None

    def _mask_expr(v: ValueRef) -> str:
        return inline_bool_expr_by_id.get(v.id, _ref(ctx, v))

    # Precompute inlined expressions for store/load masks so we can also skip
    # emitting the intermediate boolean SSA values.
    for inst in ir.insts:
        if inst.op not in {"load", "store"}:
            continue
        mask_ref: ValueRef = inst.args[2] if inst.op == "load" else inst.args[3]  # type: ignore[assignment,misc]
        if _is_const_bool(mask_ref, True):
            continue
        inlined = _try_inline_bool_expr(mask_ref)
        if inlined is None:
            continue
        expr, ids = inlined
        inline_bool_expr_by_id[mask_ref.id] = expr
        inline_bool_ids.update(ids)

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
                f"device {const_kw}{_msl_type(arg.dtype)}* __restrict {arg.name} [[buffer({buf_index})]]"
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
            if out.dtype == DType.F16:
                lines.append(
                    f"{indent}half v{out.id} = half(fma((float){_ref(ctx, x)}, (float){_ref(ctx, y)}, (float){_ref(ctx, z)}));"
                )
            else:
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
                    f"{indent}if ({_mask_expr(mask_ref)}) {{ v{out.id} = {buf_name}[{_ref(ctx, off_ref)}]; }}"
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
            a_load = _load_as_float(a_buf, f"a_idx{out.id}")
            b_load = _load_as_float(b_buf, f"b_idx{out.id}")
            lines.append(f"{indent}  v{out.id} = fma({a_load}, {b_load}, v{out.id});")
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
                    f"{indent}if ({_mask_expr(mask_ref)}) {{ {buf_name}[{_ref(ctx, off_ref)}] = {_ref(ctx, val_ref)}; }}"
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
        b_bases: list[ValueRef] = []
        for inst in group:
            (
                _a_buf,
                _a_base,
                _a_stride,
                _b_buf,
                b_base,
                _b_stride,
                _k_ref,
            ) = inst.args  # type: ignore[misc]
            assert isinstance(b_base, ValueRef)
            b_bases.append(b_base)

        # Try to vectorize 4 contiguous B loads via packed_{float,half}4.
        vec4_idxs: dict[int, int] = {}  # group index -> lane 0..3
        vec4_base: ValueRef | None = None
        if b_buf0.dtype in (DType.F32, DType.F16) and len(group) >= 4:
            candidates: list[tuple[int, ValueRef, int]] = []
            for gi, b_base in enumerate(b_bases):
                if gi >= 4:
                    break
                m = _base_plus_const(b_base)
                if m is None:
                    # allow "base + 0" to be represented as base directly
                    candidates.append((gi, b_base, 0))
                    continue
                base_ref, off = m
                candidates.append((gi, base_ref, off))
            base0 = candidates[0][1]
            offsets = [off for (_gi, base, off) in candidates if base.id == base0.id]
            if len(offsets) == 4 and sorted(offsets) == [0, 1, 2, 3]:
                vec4_base = base0
                for gi, base, off in candidates:
                    if base.id == base0.id and 0 <= off <= 3:
                        vec4_idxs[gi] = off

        b_idx_vars: list[str] = []
        b_idx0_name: str | None = None
        for gi, inst in enumerate(group):
            assert inst.out is not None
            b_idx = f"b_idx{inst.out.id}"
            b_idx_vars.append(b_idx)
            if vec4_base is not None and gi in vec4_idxs:
                if b_idx0_name is None:
                    b_idx0_name = b_idx
                    lines.append(f"{indent}uint {b_idx0_name} = {_ref(ctx, vec4_base)};")
                else:
                    # re-use the same index var for vectorized lanes
                    lines.append(f"{indent}uint {b_idx} = {b_idx0_name};")
            else:
                lines.append(f"{indent}uint {b_idx} = {_ref(ctx, b_bases[gi])};")

        k_const = _const_u32(k_ref0)
        if k_const is not None and 1 <= k_const <= 32:
            lines.append(f"{indent}#pragma clang loop unroll_count({k_const})")
        lines.append(
            f"{indent}for (uint {k_var} = 0; {k_var} < {k_name}; ++{k_var}) {{"
        )
        lines.append(f"{indent}  float a_val = {_load_as_float(a_buf0, a_idx_name)};")
        if vec4_base is not None and b_idx0_name is not None and vec4_idxs:
            space = _buf_addr_space(b_buf0) or "device"
            lines.append(f"{indent}  bool b_aligned4 = (({b_idx0_name} & 3u) == 0u);")
            lines.append(f"{indent}  if (b_aligned4) {{")
            if b_buf0.dtype == DType.F32:
                lines.append(
                    f"{indent}    packed_float4 b_pack = *(({space} packed_float4*)({b_name} + {b_idx0_name}));"
                )
                lines.append(
                    f"{indent}    float4 b_vec = float4(b_pack.x, b_pack.y, b_pack.z, b_pack.w);"
                )
            else:
                lines.append(
                    f"{indent}    packed_half4 b_pack = *(({space} packed_half4*)({b_name} + {b_idx0_name}));"
                )
                lines.append(
                    f"{indent}    float4 b_vec = float4(half4(b_pack.x, b_pack.y, b_pack.z, b_pack.w));"
                )
            for gi, (inst, b_idx) in enumerate(zip(group, b_idx_vars, strict=True)):
                assert inst.out is not None
                lane = vec4_idxs.get(gi)
                if lane is None:
                    b_load = _load_as_float(b_buf0, b_idx)
                    lines.append(f"{indent}    v{inst.out.id} = fma(a_val, {b_load}, v{inst.out.id});")
                else:
                    comp = ["x", "y", "z", "w"][lane]
                    lines.append(
                        f"{indent}    v{inst.out.id} = fma(a_val, b_vec.{comp}, v{inst.out.id});"
                    )
            lines.append(f"{indent}  }} else {{")
            for gi, inst in enumerate(group):
                assert inst.out is not None
                lane = vec4_idxs.get(gi)
                if lane is None:
                    b_load = _load_as_float(b_buf0, b_idx_vars[gi])
                    lines.append(f"{indent}    v{inst.out.id} = fma(a_val, {b_load}, v{inst.out.id});")
                else:
                    b_load = _load_as_float(b_buf0, f"{b_idx0_name} + {lane}u")
                    lines.append(
                        f"{indent}    v{inst.out.id} = fma(a_val, {b_load}, v{inst.out.id});"
                    )
            lines.append(f"{indent}  }}")
            for gi, b_idx in enumerate(b_idx_vars):
                # Update vectorized lanes only once (they alias b_idx0_name).
                if vec4_base is not None and gi in vec4_idxs:
                    if b_idx == b_idx0_name:
                        lines.append(f"{indent}  {b_idx0_name} += {_ref(ctx, b_stride0)};")
                else:
                    lines.append(f"{indent}  {b_idx} += {_ref(ctx, b_stride0)};")
        else:
            for inst, b_idx in zip(group, b_idx_vars, strict=True):
                assert inst.out is not None
                b_load = _load_as_float(b_buf0, b_idx)
                lines.append(f"{indent}  v{inst.out.id} = fma(a_val, {b_load}, v{inst.out.id});")
                lines.append(f"{indent}  {b_idx} += {_ref(ctx, b_stride0)};")
        lines.append(f"{indent}  {a_idx_name} += {_ref(ctx, a_stride0)};")
        lines.append(f"{indent}}}")
        return len(group)

    skipped_inst_idxs: set[int] = set()

    def _try_emit_dot_tn4_group(start_idx: int, *, indent: str) -> bool:
        """
        Fuse 4 dots that share the same A/B pointers/strides/K and whose B bases
        are contiguous (base + 0/1/2/3), emitting one loop that loads B via a
        single packed_{float,half}4 per k.

        This targets the matmul TN=4 pattern even when the 4 `dot` insts are not
        contiguous in the IR (due to intervening add/const instructions).
        """
        if start_idx in skipped_inst_idxs:
            return False
        inst0 = ir.insts[start_idx]
        if inst0.op != "dot" or inst0.out is None:
            return False

        (
            a_buf0,
            a_base0,
            a_stride0,
            b_buf0,
            b_base0,
            b_stride0,
            k_ref0,
        ) = inst0.args  # type: ignore[misc]
        if not (
            isinstance(a_buf0, ValueRef)
            and isinstance(a_base0, ValueRef)
            and isinstance(a_stride0, ValueRef)
            and isinstance(b_buf0, ValueRef)
            and isinstance(b_base0, ValueRef)
            and isinstance(b_stride0, ValueRef)
            and isinstance(k_ref0, ValueRef)
        ):
            return False
        if b_buf0.dtype not in (DType.F32, DType.F16):
            return False

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
        m0 = _base_plus_const(b_base0)
        base0, off0 = (m0[0], m0[1]) if m0 is not None else (b_base0, 0)
        if off0 != 0:
            return False

        found: dict[int, int] = {0: start_idx}  # off -> inst idx
        scan_limit = min(len(ir.insts), start_idx + 64)
        for j in range(start_idx + 1, scan_limit):
            if j in skipped_inst_idxs:
                continue
            inst = ir.insts[j]
            if inst.op in {"store", "barrier", "mma", "mma_store"}:
                break
            if inst.op != "dot" or inst.out is None:
                continue
            if _sig(inst) != sig0:
                continue
            _a_buf, _a_base, _a_stride, _b_buf, b_base, _b_stride, _k_ref = inst.args  # type: ignore[misc]
            if not isinstance(b_base, ValueRef):
                continue
            m = _base_plus_const(b_base)
            base, off = (m[0], m[1]) if m is not None else (b_base, 0)
            if base.id != base0.id:
                continue
            if off in {0, 1, 2, 3} and off not in found:
                found[off] = j
                if len(found) == 4:
                    break

        if set(found.keys()) != {0, 1, 2, 3}:
            return False

        # Skip emitting the other 3 dot insts; also skip their `add(base, const)`
        # b_base defs when they are single-use.
        for off in (1, 2, 3):
            dot_idx = found[off]
            skipped_inst_idxs.add(dot_idx)
            inst = ir.insts[dot_idx]
            b_base: ValueRef = inst.args[4]  # type: ignore[assignment,misc]
            add_def_idx = def_inst_idx_by_value_id.get(b_base.id)
            if add_def_idx is not None and use_counts.get(b_base.id, 0) == 1:
                add_def = ir.insts[add_def_idx]
                if add_def.op == "add":
                    skipped_inst_idxs.add(add_def_idx)

        # Emit fused loop.
        insts_by_off = [ir.insts[found[i]] for i in (0, 1, 2, 3)]
        out_ids = [inst.out.id for inst in insts_by_off]  # type: ignore[union-attr]

        a_name = _ref(ctx, a_buf0)
        b_name = _ref(ctx, b_buf0)
        k_name = _ref(ctx, k_ref0)
        k_var = f"k{out_ids[0]}"
        a_idx_name = f"a_idx{out_ids[0]}"
        b_idx_name = f"b_idx{out_ids[0]}"

        for out_id in out_ids:
            lines.append(f"{indent}float v{out_id} = 0.0f;")

        lines.append(f"{indent}uint {a_idx_name} = {_ref(ctx, a_base0)};")
        lines.append(f"{indent}uint {b_idx_name} = {_ref(ctx, base0)};")

        k_const = _const_u32(k_ref0)
        if k_const is not None and 1 <= k_const <= 32:
            lines.append(f"{indent}#pragma clang loop unroll_count({k_const})")
        lines.append(
            f"{indent}for (uint {k_var} = 0; {k_var} < {k_name}; ++{k_var}) {{"
        )
        lines.append(f"{indent}  float a_val = {_load_as_float(a_buf0, a_idx_name)};")
        space = _buf_addr_space(b_buf0) or "device"
        lines.append(f"{indent}  bool b_aligned4 = (({b_idx_name} & 3u) == 0u);")
        lines.append(f"{indent}  if (b_aligned4) {{")
        if b_buf0.dtype == DType.F32:
            lines.append(
                f"{indent}    packed_float4 b_pack = *(({space} packed_float4*)({b_name} + {b_idx_name}));"
            )
            lines.append(
                f"{indent}    float4 b_vec = float4(b_pack.x, b_pack.y, b_pack.z, b_pack.w);"
            )
        else:
            lines.append(
                f"{indent}    packed_half4 b_pack = *(({space} packed_half4*)({b_name} + {b_idx_name}));"
            )
            lines.append(
                f"{indent}    float4 b_vec = float4(half4(b_pack.x, b_pack.y, b_pack.z, b_pack.w));"
            )
        comps = ["x", "y", "z", "w"]
        for out_id, comp in zip(out_ids, comps, strict=True):
            lines.append(
                f"{indent}    v{out_id} = fma(a_val, b_vec.{comp}, v{out_id});"
            )
        lines.append(f"{indent}  }} else {{")
        for out_id, off in zip(out_ids, range(4), strict=True):
            b_load = _load_as_float(b_buf0, f"{b_idx_name} + {off}u")
            lines.append(
                f"{indent}    v{out_id} = fma(a_val, {b_load}, v{out_id});"
            )
        lines.append(f"{indent}  }}")
        lines.append(f"{indent}  {b_idx_name} += {_ref(ctx, b_stride0)};")
        lines.append(f"{indent}  {a_idx_name} += {_ref(ctx, a_stride0)};")
        lines.append(f"{indent}}}")
        return True

    idx = 0
    while idx < len(ir.insts):
        inst = ir.insts[idx]
        if idx in skipped_inst_idxs:
            idx += 1
            continue
        if (
            inst.out is not None
            and use_counts.get(inst.out.id, 0) == 0
            and inst.op != "store"
        ):
            idx += 1
            continue
        if inst.out is not None and inst.out.id in inline_bool_ids:
            idx += 1
            continue
        if idx in moved_inst_idxs or idx in alloc_tg_inst_idxs:
            idx += 1
            continue
        if inst.op in {"arg"}:
            idx += 1
            continue

        if inst.op == "dot":
            if _try_emit_dot_tn4_group(idx, indent="  "):
                idx += 1
            else:
                idx += _emit_dot_group(idx, indent="  ")
            continue

        if inst.op == "store" and idx in lifted_by_store:
            _buf_ref, _off_ref, _val_ref, mask_ref = inst.args  # type: ignore[misc]
            if (
                early_return_store_idx == idx
                and isinstance(mask_ref, ValueRef)
                and not _is_const_bool(mask_ref, True)
            ):
                lines.append(f"  if (!{_mask_expr(mask_ref)}) return;")
                for moved_idx in lifted_by_store[idx]:
                    _emit_inst(
                        ir.insts[moved_idx], indent="  ", in_guard_for_mask=mask_ref
                    )
                _emit_inst(inst, indent="  ", in_guard_for_mask=mask_ref)
            else:
                lines.append(f"  if ({_mask_expr(mask_ref)}) {{")
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
                lines.append(f"  if (!{_mask_expr(mask_ref)}) return;")
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
