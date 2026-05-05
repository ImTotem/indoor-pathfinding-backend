"""LocalFileStorage — var/storage/scans/{scan_id}/ 기반 로컬 구현."""
from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalFileStorage:
    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root / "scans"
        self._root.mkdir(parents=True, exist_ok=True)

    async def commit(self, *, scan_id: str, source_dir: Path, replace: bool) -> str:
        target = self._root / scan_id
        if target.exists() and not replace:
            logger.debug("storage commit no-op (already exists): %s", scan_id)
            return f"scans/{scan_id}"
        await asyncio.to_thread(self._atomic_move, source_dir, target, replace)
        return f"scans/{scan_id}"

    async def exists(self, *, scan_id: str) -> bool:
        return (self._root / scan_id).exists()

    async def readyz(self) -> bool:
        probe = self._root / f".probe-{uuid.uuid4().hex}"
        try:
            probe.touch()
            probe.unlink()
            return True
        except OSError:
            return False

    # ── sync helpers ──────────────────────────────────────────────────────────

    def _atomic_move(self, source: Path, target: Path, replace: bool) -> None:
        if target.exists() and replace:
            shutil.rmtree(target)
        # 동일 파일시스템이면 rename이 원자적
        try:
            source.rename(target)
        except OSError:
            # 크로스 파일시스템 폴백
            shutil.copytree(source, target)
            shutil.rmtree(source, ignore_errors=True)
        logger.info("storage committed: %s", target)
