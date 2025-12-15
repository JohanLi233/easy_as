## Roadmap

This project is transitioning from “single-op MVP” to a production-grade kernel framework. The kernel authoring experience (`@eas.kernel` + `eas.mk`) remains the center of the design.

### Near-term (make it solid)

- **API stability**: define and version public APIs (`eas.kernel`, `eas.Tensor`, runtime selection, environment variables).
- **Device tensors**: make `eas.Tensor(device="metal")` the default-performance path; keep `numpy` interop ergonomic.
- **Metal runtime correctness**: robust synchronization semantics; consistent error reporting; predictable resource lifetimes.
- **Build/distribution**: document macOS build requirements; add wheel build guidance; keep CPU fallback working everywhere.

### Mid-term (make it fast)

- **Memory pooling**: reuse Metal buffers (device-private + staging) to reduce allocations.
- **Streams**: explicit stream/queue API; avoid implicit sync unless required (e.g. `.numpy()`).
- **Argument binding**: reduce per-dispatch overhead (prepared arguments; consider argument buffers).
- **Kernel autotuning hooks**: expose meta-parameter sweeps (e.g. `BLOCK`) with benchmarking harness (run outside sandbox).

### Long-term (make it scalable)

- **Graph capture/replay**: amortize encoding/dispatch overhead for tiny kernels and repeated loops.
- **More ops + types**: broaden IR and codegen beyond 1D/float32; vectorized loads/stores; reductions.
- **Fusion**: multi-op lowering and scheduling at IR level.
- **Observability**: profiling hooks, structured logging, debug dumps of IR/MSL/pipelines.

