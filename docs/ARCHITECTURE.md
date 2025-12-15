## easy_as architecture

easy_as is a Python-first kernel framework with a small tracing DSL that lowers to a minimal IR, codegens Metal Shading Language (MSL), and executes via a runtime backend (Metal or CPU).

### Layering

- **Front-end (Python DSL)**: `eas/mk.py`
  - Tracing-only primitives (`mk.arange`, `mk.program_id`, `mk.load`, `mk.store`, arithmetic).
  - No eager execution; operations are recorded into an IR builder.

- **Kernel wrapper + caching**: `eas/kernel.py`
  - `@eas.kernel` wraps a Python function into a `Kernel`.
  - Separates **meta parameters** (`eas.constexpr`) from runtime parameters.
  - Caches `CompiledKernel` keyed by meta + runtime argument types.

- **IR**: `eas/ir.py`
  - Minimal, backend-agnostic IR (`IRModule`, `Arg`, `Inst`, `DType`).

- **Compiler**: `eas/compiler.py`
  - Traces the kernel with `mk` into IR.
  - Runs analysis (`eas/analysis.py`) and codegen (`eas/codegen/msl.py`).
  - Produces `CompiledKernel` (IR + MSL + threadgroup size + write set).

- **Runtime backends**: `eas/runtime/*`
  - `eas/runtime/metal.py`: executes `CompiledKernel` on Metal via `eas._metal` (ObjC++).
  - `eas/runtime/cpu.py`: CPU fallback for correctness and tests.
  - `eas/runtime/__init__.py`: runtime selection (`EAS_BACKEND=auto|cpu|metal`).

### Device tensors (torch-style)

`eas/tensor.py` introduces a torch-like `eas.Tensor` as the long-term data model:

- `eas.tensor(data, device="cpu"|"metal")`
- `eas.empty(shape, device=...)`, `eas.empty_like(t)`
- `t.to("cpu"|"metal")`, `t.numpy()`

Rationale:

- **Performance**: keep buffers resident on the device across iterations, instead of wrapping host arrays per launch.
- **Extensibility**: enables memory pooling, streams, graph replay, and fusion without changing kernel syntax.

Current constraints (intentional for now):

- Only `float32` buffers in the MVP codegen/runtime path.
- 1D grid sizing requires a scalar runtime argument named `N`.

### Torch interop

The architecture is designed to interoperate with PyTorch without coupling core compilation to torch internals:

- **CPU**: `torch.Tensor` ↔ `eas.Tensor(device="cpu")` is zero-copy via `tensor.numpy()` / `torch.from_numpy(...)` when possible.
- **GPU**: use explicit conversions for now; long-term, prefer a standard capsule-based interchange (e.g. DLPack) once the target device backend supports it.

### Execution model

1. User calls a `Kernel` with runtime args and constexpr meta.
2. `Kernel.compile(...)` traces to IR (if not cached) and codegens MSL.
3. Runtime binds arguments and dispatches:
   - CPU: interprets IR for validation.
   - Metal: compiles MSL to a pipeline state and dispatches the compute kernel.

### What “enterprise-grade” means here (design goals)

- Stable public API surface (`eas.kernel`, `eas.Tensor`, runtime selection).
- Clear backend contracts (Metal extension API, error handling, diagnostics).
- Deterministic caching and reproducible compilation.
- Testability: CPU baseline + Metal integration tests guarded by availability.
- Performance roadmap: device-private buffers by default, memory pooling, streams, and optional graph replay.
