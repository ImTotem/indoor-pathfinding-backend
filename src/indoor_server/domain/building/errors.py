"""Build Phase 도메인 예외."""
from __future__ import annotations


class BuildDomainError(Exception):
    code: str = "BUILD_INTERNAL"

    def __init__(self, message: str, detail: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, object] = detail or {}


class BuildAlreadyRunning(BuildDomainError):
    code = "BUILD_ALREADY_RUNNING"


class BuildNotFound(BuildDomainError):
    code = "BUILD_NOT_FOUND"


class GraphNotReady(BuildDomainError):
    code = "GRAPH_NOT_READY"


class ModelLoadFailed(BuildDomainError):
    code = "MODEL_LOAD_FAILED"
