from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class Runtime(Protocol):
    def is_available(self) -> bool: ...

    def run(
        self,
        ck: Any,
        runtime_args: Mapping[str, Any],
        meta: Mapping[str, Any],
        *,
        sync: bool = True,
    ) -> None: ...

    def synchronize(self) -> None: ...
