from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..ir import DType, IRModule, ValueRef


def _zero(dtype: DType) -> Any:
    if dtype == DType.F32:
        return np.float32(0.0)
    if dtype == DType.U32:
        return 0
    if dtype == DType.BOOL:
        return False
    raise ValueError(f"unsupported dtype: {dtype}")


@dataclass(slots=True)
class CpuRuntime:
    def run(
        self,
        ck: Any,
        runtime_args: Mapping[str, Any],
        meta: Mapping[str, Any],
        *,
        sync: bool = True,
    ) -> None:
        _ = meta
        _ = sync
        ir: IRModule = ck.ir
        threadgroup_size: int = ck.threadgroup_size

        # Heuristic default grid: ceil_div(N, threadgroup_size)
        n = None
        for k, v in runtime_args.items():
            if k == "N":
                n = int(v)
                break
        if n is None:
            raise ValueError(
                "MVP runtime requires a scalar argument named 'N' for grid sizing"
            )
        grid = (n + threadgroup_size - 1) // threadgroup_size

        # Prepare runtime binding for args (name -> value)
        arg_values: dict[str, Any] = {}
        for arg in ir.args:
            arg_values[arg.name] = runtime_args[arg.name]

        for pid in range(grid):
            for tid in range(threadgroup_size):
                self._run_thread(ir, arg_values, pid=pid, tid=tid)

    def _run_thread(
        self, ir: IRModule, arg_values: Mapping[str, Any], *, pid: int, tid: int
    ) -> None:
        env: dict[int, Any] = {}
        arg_ref_by_name: dict[str, int] = {}

        for inst in ir.insts:
            if inst.op == "arg":
                out = inst.out
                assert out is not None
                name = str(inst.args[0])
                arg_ref_by_name[name] = out.id
                env[out.id] = arg_values[name]
                continue
            if inst.op == "const":
                out = inst.out
                assert out is not None
                env[out.id] = inst.args[0]
                continue
            if inst.op == "program_id":
                out = inst.out
                assert out is not None
                axis = int(inst.args[0])
                if axis != 0:
                    raise ValueError("only program_id(0) is supported in MVP")
                env[out.id] = pid
                continue
            if inst.op == "arange":
                out = inst.out
                assert out is not None
                start, _size = (int(inst.args[0]), int(inst.args[1]))
                env[out.id] = start + tid
                continue
            if inst.op in {"add", "mul", "lt"}:
                out = inst.out
                assert out is not None
                a: ValueRef
                b: ValueRef
                a, b = inst.args  # type: ignore[misc]
                av = env[a.id]
                bv = env[b.id]
                if inst.op == "add":
                    env[out.id] = av + bv
                elif inst.op == "mul":
                    env[out.id] = av * bv
                else:
                    env[out.id] = av < bv
                continue
            if inst.op == "where":
                out = inst.out
                assert out is not None
                c, a, b = inst.args  # type: ignore[misc]
                env[out.id] = env[a.id] if env[c.id] else env[b.id]
                continue
            if inst.op == "load":
                out = inst.out
                assert out is not None
                buf_ref, off_ref, mask_ref = inst.args  # type: ignore[misc]
                buf = env[buf_ref.id]
                off = int(env[off_ref.id])
                m = bool(env[mask_ref.id])
                if not m:
                    env[out.id] = _zero(out.dtype)
                    continue
                arr = buf if isinstance(buf, np.ndarray) else np.asarray(buf)
                if off < 0 or off >= arr.size:
                    env[out.id] = _zero(out.dtype)
                else:
                    env[out.id] = np.float32(arr.flat[off])
                continue
            if inst.op == "store":
                buf_ref, off_ref, val_ref, mask_ref = inst.args  # type: ignore[misc]
                buf = env[buf_ref.id]
                off = int(env[off_ref.id])
                v = env[val_ref.id]
                m = bool(env[mask_ref.id])
                if not m:
                    continue
                arr = buf if isinstance(buf, np.ndarray) else np.asarray(buf)
                if off < 0 or off >= arr.size:
                    continue
                arr.flat[off] = np.float32(v)
                continue
            raise ValueError(f"unsupported op: {inst.op}")
