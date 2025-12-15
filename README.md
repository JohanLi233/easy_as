# easy_as

Python-first DSL (tracing) → IR → Metal Shading Language (MSL) codegen, with a CPU fallback runtime for MVP validation.

## MVP: elementwise add

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
