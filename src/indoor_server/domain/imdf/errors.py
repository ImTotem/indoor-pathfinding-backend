"""IMDF-lite 도메인 예외."""
from __future__ import annotations


class ImdfDomainError(Exception):
    """IMDF export 관련 도메인 예외 기반 클래스."""

    code: str = "IMDF_INTERNAL"

    def __init__(self, message: str, detail: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, object] = detail or {}


class FootprintNotAvailableError(ImdfDomainError):
    """footprint_geojson이 build_job.counts에 없음.

    구버전 build job이거나 adaptive_buffer 경로를 사용하지 않은 경우.
    클라이언트에 재빌드를 유도한다.
    """

    code = "FOOTPRINT_NOT_AVAILABLE"

    def __init__(self, scan_id: str, build_job_id: str) -> None:
        super().__init__(
            "footprint이 저장되어 있지 않습니다. 재빌드 후 다시 시도하세요.",
            detail={"scan_id": scan_id, "build_job_id": build_job_id},
        )


class ImdfScanNotFoundError(ImdfDomainError):
    """scan_id에 해당하는 스캔이 없음."""

    code = "SCAN_NOT_FOUND"

    def __init__(self, scan_id: str) -> None:
        super().__init__(
            f"scan_id={scan_id}에 해당하는 스캔을 찾을 수 없습니다.",
            detail={"scan_id": scan_id},
        )


class ImdfGraphNotReadyError(ImdfDomainError):
    """build_job이 아직 succeeded 상태가 아님."""

    code = "GRAPH_NOT_READY"

    def __init__(self, scan_id: str, state: str) -> None:
        super().__init__(
            "빌드가 아직 완료되지 않았습니다.",
            detail={"scan_id": scan_id, "state": state},
        )
