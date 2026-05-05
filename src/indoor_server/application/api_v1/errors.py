"""V1 compatibility service errors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class V1ServiceError(Exception):
    status_code: int
    code: str
    message: str
    detail: dict[str, Any] | None = None
