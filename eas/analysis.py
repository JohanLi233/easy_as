from __future__ import annotations

from .ir import IRModule, ValueRef


def infer_writes(ir: IRModule) -> frozenset[str]:
    id_to_name: dict[int, str] = {}
    for inst in ir.insts:
        if inst.op == "arg":
            assert inst.out is not None
            id_to_name[inst.out.id] = str(inst.args[0])

    writes: set[str] = set()
    for inst in ir.insts:
        if inst.op == "store":
            buf_ref: ValueRef = inst.args[0]
            name = id_to_name.get(buf_ref.id)
            if name is not None:
                writes.add(name)
    return frozenset(writes)
