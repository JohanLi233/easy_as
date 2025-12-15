from __future__ import annotations


class constexpr:
    """Annotation marker for compile-time meta parameters.

    Parameters annotated with `constexpr` are compile-time constants: they are
    used during tracing/codegen, and are not treated as runtime arguments.
    """
