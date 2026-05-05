"""ScanFileStore Protocol — 스토리지 추상화."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ScanFileStore(Protocol):
    async def commit(self, *, scan_id: str, source_dir: Path, replace: bool) -> str:
        """
        source_dir 를 영구 저장소로 원자적으로 이동.
        반환값: 저장소 내 상대 경로 (예: "scans/{scan_id}").
        replace=False 이고 이미 존재하면 no-op (멱등).
        """
        ...

    async def exists(self, *, scan_id: str) -> bool: ...

    async def readyz(self) -> bool:
        """저장소 쓰기 가능 여부 확인 (readyz 체크용)."""
        ...
