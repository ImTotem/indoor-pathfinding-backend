"""도메인 예외 정의."""
from __future__ import annotations


class ScanDomainError(Exception):
    """스캔 도메인 베이스 예외."""

    code: str = "INTERNAL"

    def __init__(self, message: str, detail: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, object] = detail or {}


class ZipStructureInvalid(ScanDomainError):
    code = "ZIP_STRUCTURE_INVALID"


class MissingRequiredFile(ScanDomainError):
    code = "MISSING_REQUIRED_FILE"


class ScanIdMismatch(ScanDomainError):
    code = "SCAN_ID_MISMATCH"


class SidecarSchemaMismatch(ScanDomainError):
    code = "SIDECAR_SCHEMA_MISMATCH"


class ScanConflict(ScanDomainError):
    code = "SCAN_CONFLICT"


class PayloadTooLarge(ScanDomainError):
    code = "PAYLOAD_TOO_LARGE"


class ScanIdInvalidFormat(ScanDomainError):
    code = "INVALID_SCAN_ID"


class Unauthorized(ScanDomainError):
    code = "UNAUTHORIZED"
