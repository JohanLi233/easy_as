# filename: eas/runtime/metal_ext.py

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def load_metal_ext(*, require: bool) -> Any | None:
    spec = importlib.util.find_spec("eas._metal")
    if spec is None:
        if require:
            raise RuntimeError(
                "Metal backend is not built. Run `uv run python tools/build_metal_ext.py` "
                "to build `eas._metal`."
            )
        return None

    mod = importlib.import_module("eas._metal")
    missing: list[str] = []
    for name in (
        "is_available",
        "compile",
        "launch",
        "synchronize",
        "alloc_buffer",
        "copy_from_host",
        "copy_to_host",
        "dlpack_import",
        "dlpack_export",
        "queue_synchronize",
    ):
        if not callable(getattr(mod, name, None)):
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Metal extension `eas._metal` is missing required API: "
            + ", ".join(missing)
            + ". Rebuild it with `uv run python tools/build_metal_ext.py`."
        )
    return mod
