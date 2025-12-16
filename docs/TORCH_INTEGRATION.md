# Torch 集成（企业指导）

## 概述

本文档定义了 easy_as 与 torch"无缝集成"的含义，以及哪些实现选择是*安全*的 vs *高性能*的。

### "无缝"的含义（明确选择）

1) **API 无缝**：接受 `torch.Tensor` 输入/输出，无需样板代码。
2) **内存无缝**：避免在 torch 和 EAS 之间拷贝（零拷贝）。
3) **调度无缝**：正确的流/队列语义（无意外竞争）。
4) **自动微分无缝**：在 torch 训练循环中前向/后向。

您可以快速获得 (1)，但在 Metal/MPS 上 (2)+(3) 在一起是困难的部分。

### 今天：保证什么

- **CPU**
  - `torch.Tensor(cpu)` ↔ `eas.Tensor(cpu)` 可以是零拷贝（通过 NumPy/DLPack 协议）。
  - 此路径稳定且推荐用于企业互操作。

- **Metal/MPS**
  - 对于 **contiguous 的 float32**，`torch.Tensor(mps)` ↔ `eas.Tensor(mps)` 可通过 **DLPack(kDLMetal)** 做到零拷贝互操作（例如 torch 2.9.1 已支持）。
  - 调度语义仍需谨慎：跨框架的队列/流排序并不自动等价于“同一个 stream”，因此将 DLPack 视为**互操作边界**是企业默认策略。
    - easy_as 默认在使用 torch(mps) 张量前做一次 `torch.mps.synchronize()`（可用 `EAS_TORCH_MPS_SYNC=0` 关闭），避免读取到 torch 侧尚未完成的写入。

### 为什么 MPS 零拷贝非同小可

即使您可以"指向"torch 的内存：

- 您仍然需要 torch 的 MPS 工作和 EAS 工作之间的正确**排序**。
- 独立的 `MTLCommandQueue` 可能会竞争，除非您在流/命令缓冲区级别集成。
- 仅共享指针但不共享调度的解决方案*不是企业安全的*。

### 最佳决策（推荐架构）

采用双层集成策略：

#### A 层（默认，稳定）：DLPack + 需要时显式拷贝

- CPU：使用 DLPack/NumPy 协议（零拷贝）。
- Metal/MPS：优先使用 DLPack（零拷贝），否则回退到显式拷贝：
  - `eas.tensor(torch_mps_tensor, device="mps")`（零拷贝，需 contiguous float32）
  - `eas_tensor.to_torch("mps")`（零拷贝，需 contiguous float32）
  - 若不满足条件（非 contiguous / dtype 不支持 / 上游不支持 DLPack），则采用显式拷贝路径。
- 保持核心运行时严格/可预测（避免在 `MetalRuntime.run` 中做不可见的大拷贝；必要时通过清晰的转换 API 完成边界操作）。

这为您提供了正确性和清晰的性能边界。

#### B 层（可选，高性能）：原生 torch-MPS 插件（零拷贝 + 正确调度）

如果您*必须*在 MPS 上零拷贝并且愿意固定 torch 版本：

- 构建一个可选的 PyTorch 扩展（例如 `eas._torch_mps`），它：
  - 从 MPS 张量中提取底层 `MTLBuffer`（torch 内部 API）。
  - 在 torch 的 MPS 流/队列上分派编译的 MSL 内核。
  - 将 EAS 内核包装为 `torch.library` 自定义算子和（可选）`autograd.Function`。

这是在 MPS 上现实地实现 (2)+(3) 的唯一途径，但它是一个更高的维护表面，必须进行版本门控。

### 探测脚本（在您的机器上运行）

为了有信心地选择 B 层，首先验证您的 torch 构建公开了什么：

```python
import torch

print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    x = torch.randn(16, device="mps", dtype=torch.float32)
    print("x.device:", x.device, "dtype:", x.dtype)
    print("has __dlpack__:", hasattr(x, "__dlpack__"))
    try:
        from torch.utils import dlpack
        cap = dlpack.to_dlpack(x)
        print("torch dlpack to_dlpack(mps) ok:", type(cap).__name__)
    except Exception as e:
        print("torch dlpack to_dlpack(mps) failed:", type(e).__name__, e)
```

如果你的 torch 构建不支持 MPS DLPack（或行为不稳定），A 层仍然是正确的默认设置：先用显式拷贝保证正确性，再考虑 B 层插件。
