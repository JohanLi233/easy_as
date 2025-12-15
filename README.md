# easy_as

Python-first kernel DSL (tracing) → IR → Metal Shading Language (MSL) codegen, with Metal + CPU runtimes.

## Quickstart: elementwise add

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

## Device tensors (torch-style)

For performance on Metal, keep buffers resident on the device:

```python
import numpy as np
import eas

n = 1 << 20
a = eas.tensor(np.random.randn(n).astype(np.float32), device="metal")
b = eas.tensor(np.random.randn(n).astype(np.float32), device="metal")
c = eas.empty_like(a)

add_kernel(a, b, c, n, BLOCK=256)
out = c.numpy()
```

Run the example:

```bash
python3 -m examples.add
```

Run tests:

```bash
python3 -m unittest discover -s tests -p "test*.py" -v
```

## Metal backend (GPU)

Build the `eas._metal` extension (ObjC++):

```bash
python3 tools/build_metal_ext.py
```

Run with Metal:

```bash
EAS_BACKEND=metal python3 -m examples.add
```

## Torch interop

`torch.Tensor` can be converted to `eas.Tensor`:

```python
import torch
import eas

a = torch.randn(1024, device="cpu", dtype=torch.float32)
a_e = eas.tensor(a, device="cpu")  # zero-copy on CPU
b_e = a_e.to("metal")              # upload to Metal (copy)
```

Convert back:

```python
t = b_e.to_torch("mps")            # best-effort; currently round-trips via host
```

## Configuration

- `EAS_BACKEND=auto|cpu|metal`
- `EAS_MAX_IN_FLIGHT`: max number of in-flight async launches (used when calling kernels with `_sync=False`).

## Docs

- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
