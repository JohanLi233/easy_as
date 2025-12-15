# easy_as 架构

## 概述

easy_as 是一个 Python 优先的内核框架，具有小型追踪 DSL，可降级为最小 IR，生成 Metal Shading Language (MSL) 代码，并通过运行时后端（Metal 或 CPU）执行。

### 分层

- **前端（Python DSL）**：`eas/mk.py`
  - 仅追踪原语（`mk.arange`、`mk.program_id`、`mk.load`、`mk.store`、算术运算）。
  - 无急切执行；操作记录到 IR 构建器中。

- **内核包装器 + 缓存**：`eas/kernel.py`
  - `@eas.kernel` 将 Python 函数包装为 `Kernel`。
  - 区分**元参数**（`eas.constexpr`）和运行时参数。
  - 缓存由元 + 运行时参数类型键控的 `CompiledKernel`。

- **IR**：`eas/ir.py`
  - 最小化、后端无关的 IR（`IRModule`、`Arg`、`Inst`、`DType`）。

- **编译器**：`eas/compiler.py`
  - 使用 `mk` 将内核追踪到 IR。
  - 运行分析（`eas/analysis.py`）和代码生成（`eas/codegen/msl.py`）。
  - 生成 `CompiledKernel`（IR + MSL + 线程组大小 + 写入集合）。

- **运行时后端**：`eas/runtime/*`
  - `eas/runtime/metal.py`：通过 `eas._metal`（ObjC++）在 Metal 上执行 `CompiledKernel`。
  - `eas/runtime/cpu.py`：用于正确性和测试的 CPU 回退。
  - `eas/runtime/__init__.py`：运行时选择（`EAS_BACKEND=auto|cpu|metal`）。

### 设备张量（torch风格）

`eas/tensor.py` 引入类似 torch 的 `eas.Tensor` 作为长期数据模型：

- `eas.tensor(data, device="cpu"|"metal")`
- `eas.empty(shape, device=...)`、`eas.empty_like(t)`
- `t.to("cpu"|"metal")`、`t.numpy()`

设计原理：

- **性能**：在迭代间保持缓冲区驻留在设备上，而不是每次启动时包装主机数组。
- **可扩展性**：启用内存池、流、图形重放和融合，而无需更改内核语法。

当前限制（目前是故意的）：

- 在 MVP 代码生成/运行时路径中仅支持 `float32` 缓冲区。
- 1D 网格大小调整需要一个名为 `N` 的标量运行时参数。

### Torch 互操作

架构设计为与 PyTorch 互操作，而不将核心编译耦合到 torch 内部：

- **CPU**：`torch.Tensor` ↔ `eas.Tensor(device="cpu")` 在可能时通过 `tensor.numpy()` / `torch.from_numpy(...)` 实现零拷贝。
- **GPU/Metal**：对于 contiguous float32，`torch(mps)` ↔ `eas(metal)` 可通过 DLPack(kDLMetal) 做到零拷贝互操作；将 DLPack 视为互操作边界（可能隐含同步）以保证企业级正确性。

#### DLPack

`eas.from_dlpack(...)` / `eas.to_dlpack(...)` 提供标准交换点：

- 现在对 CPU 张量工作良好（是企业互操作的首选路径）。
- 可以干净地扩展到支持 DLPack 设备类型（如 CUDA）的未来后端。
- 对于 Metal/MPS，当生产者支持输出 kDLMetal（例如 torch(mps)）时，同样可用于零拷贝互操作；否则回退到显式拷贝。

### 执行模型

1. 用户使用运行时参数和 constexpr 元调用 `Kernel`。
2. `Kernel.compile(...)` 追踪到 IR（如果未缓存）并生成 MSL。
3. 运行时绑定参数并分派：
   - CPU：解释 IR 进行验证。
   - Metal：将 MSL 编译为管道状态并分派计算内核。

### 这里的"企业级"含义（设计目标）

- 稳定的公共 API 表面（`eas.kernel`、`eas.Tensor`、运行时选择）。
- 清晰的后端契约（Metal 扩展 API、错误处理、诊断）。
- 确定性缓存和可重现编译。
- 可测试性：CPU 基线 + 可用性保护的 Metal 集成测试。
- 性能路线图：默认设备私有缓冲区、内存池、流和可选图形重放。
