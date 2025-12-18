# easy_as

Python优先的内核 DSL（追踪）→ IR → Metal Shading Language (MSL) 代码生成，配备 Metal(MPS) 运行时。

## 执行语义（重要）

- `mk.arange(0, BLOCK)`：Triton 风格 SPMD（向量）语义，返回一个长度为 `BLOCK` 的“程序内向量”；编译/运行时将该 kernel 视为 **SPMD**（`tptg=(1,1,1)`，grid 按 `ceil_div(N, BLOCK)` 推导）。
- `mk.tid(0, BLOCK)`：线程级语义（旧 `mk.arange`），返回 threadgroup 内的标量 thread id；用于 threadgroup memory、barrier、simdgroup MMA 等需要多线程协作的 kernel（编译/运行时将该 kernel 视为 **thread**，`tptg=(BLOCK,1,1)`）。

## 快速开始：元素级加法

```python
import numpy as np
import eas
from eas import mk


@eas.kernel
def add_kernel(a, b, c, N, BLOCK: eas.constexpr):
    pid = mk.program_id(0)
    offs = pid * BLOCK + mk.arange(0, BLOCK)
    mask = offs < N
    mk.store(c, offs, mk.load(a, offs, mask) + mk.load(b, offs, mask), mask)


n = 1024 + 7
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

add_kernel(a, b, c, n, BLOCK=256)
print(add_kernel.to_msl(a, b, c, n, BLOCK=256))
```

## 设备张量（torch风格）

为了在 GPU 上获得性能，请将缓冲区保留在设备上（推荐用 `eas.Tensor(device="mps")` 或 torch(MPS)）：

```python
import numpy as np
import eas

n = 1 << 20
a = eas.tensor(np.random.randn(n).astype(np.float32), device="mps")
b = eas.tensor(np.random.randn(n).astype(np.float32), device="mps")
c = eas.empty_like(a)

add_kernel(a, b, c, n, BLOCK=256)
out = c.numpy()
```

运行示例：

```bash
uv run python -m examples.add
```

运行测试：

```bash
uv run python -m unittest discover -s tests -p "test*.py" -v
```

## Metal 后端（GPU）

构建 `eas._metal` 扩展（ObjC++）：

```bash
uv run python tools/build_metal_ext.py
```

使用 Metal 运行：

```bash
EAS_BACKEND=mps uv run python -m examples.add
```

## Torch 互操作

`torch.Tensor` 可以转换为 `eas.Tensor`：

```python
import torch
import eas

a = torch.randn(1024, device="cpu", dtype=torch.float32)
a_e = eas.tensor(a, device="cpu")  # CPU 上的零拷贝
b_e = a_e.to("mps")               # 上传到 MPS（拷贝）

# MPS → eas(mps)：通过 DLPack(kDLMetal) 零拷贝（需 contiguous float32）
x = torch.randn(1024, device="mps", dtype=torch.float32)
x_e = eas.tensor(x, device="mps")
```

转换回去：

```python
t = x_e.to_torch("mps")            # eas(mps) → torch(mps)：DLPack 零拷贝
```

## DLPack 互操作

通过 DLPack 实现 CPU 零拷贝交换（当生产者支持时）：

```python
import eas
import torch

x = torch.randn(1024, device="cpu", dtype=torch.float32)
y = eas.from_dlpack(x)             # CPU 上的零拷贝
z = eas.to_dlpack(y)               # DLPack 胶囊（CPU；torch 风格）
```

## 配置

### 环境变量

- `EAS_BACKEND=auto|mps`：选择后端
- `EAS_MAX_IN_FLIGHT`：最大异步启动数量（在使用 `_sync=False` 调用内核时使用）

### 使用 uv

项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理和开发：

```bash
# 安装依赖（包含开发依赖）
uv sync --dev

# 运行示例
uv run python -m examples.add

# 运行测试
uv run python -m unittest discover -s tests -p "test*.py" -v

# 激活虚拟环境（在当前shell中）
source .venv/bin/activate
python -m examples.add
```
