"""Streaming scan ingest orchestration.

Three operations:
- `start_scan`: create empty rtabmap.db + scan_ingest/scan_session/floor_scan
  rows. Marks the scan as `OPEN` and ready to receive frames.
- `append_frames`: append a batch of frames + links to the open rtabmap.db.
- `finalize_scan`: ingest the uploaded scan_metadata.db (sidecar) into the
  domain tables, recompute the rtabmap.db sha256, and flip the scan to
  `READY` so `/floors/{id}/build` can pick it up.

Reprocessing (rtabmap-reprocess + pose_backfill) happens inside the build
worker — finalize itself does not run reprocess.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import sqlalchemy as sa
from fastapi import UploadFile
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import BuildingFloorService
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.application.scan_streaming.rtabmap_db_writer import (
    FrameAppendError,
    FrameRecord,
    LinkRecord,
    append_batch,
    create_empty_db,
    last_node_id,
    node_count,
    sha256_of_db,
)
from indoor_server.application.sidecar_parser import SidecarParser
from indoor_server.config import settings
from indoor_server.infrastructure.db import tables as t
from indoor_server.infrastructure.db.repositories.scan_ingest_repo import (
    ScanIngestRepository,
)

logger = logging.getLogger(__name__)


# Placeholder sha256 used while the scan is still receiving frames. Replaced
# with the real hash at finalize time.
_OPEN_SCAN_SHA256 = "pending-streaming-scan"


@dataclass(frozen=True)
class ScanStartResult:
    scan_id: str
    floor_id: str
    storage_path: str
    rtabmap_db_path: str
    state: str  # OPEN


@dataclass(frozen=True)
class FrameBatchResult:
    scan_id: str
    frames_applied: int
    frames_skipped: int
    links_applied: int
    links_skipped: int
    last_node_id: int
    node_count: int


@dataclass(frozen=True)
class ScanFinalizeResult:
    scan_id: str
    floor_id: str
    state: str  # READY
    node_count: int
    keyframe_count: int
    poi_mark_count: int
    payload_sha256: str


class ScanStreamingService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start_scan(
        self,
        *,
        floor_id: UUID,
        scan_id: str | None,
        device_info: str | None,
    ) -> ScanStartResult:
        # Floor must exist.
        await BuildingFloorService(self._session).get_floor(floor_id)

        effective_scan_id = (scan_id or str(uuid4())).lower()
        storage_path = f"scans/{effective_scan_id}"
        scan_dir = settings.storage_root / storage_path
        rtabmap_db = scan_dir / "rtabmap.db"

        if rtabmap_db.exists():
            # Idempotent restart of an OPEN scan is fine — but if it's already
            # finalized (READY) we refuse to reopen, that would corrupt state.
            existing_state = await self._fetch_floor_scan_status(
                floor_id=floor_id, scan_id=effective_scan_id
            )
            if existing_state == "READY":
                raise V1ServiceError(
                    409,
                    "SCAN_ALREADY_FINALIZED",
                    "scan is already finalized; start a new scan_id",
                )
            # Resume — DB already exists, just return the current state.
            return ScanStartResult(
                scan_id=effective_scan_id,
                floor_id=str(floor_id),
                storage_path=storage_path,
                rtabmap_db_path=str(rtabmap_db),
                state="OPEN",
            )

        await asyncio.to_thread(create_empty_db, rtabmap_db)

        # scan_ingest first (FK target). Use a placeholder sha256 — we recompute
        # at finalize.
        import json as _json

        device_info_json = _json.loads(device_info) if device_info else None
        await self._session.execute(
            pg_insert(t.scan_ingest)
            .values(
                scan_id=effective_scan_id,
                payload_sha256=_OPEN_SCAN_SHA256,
                storage_path=storage_path,
                device_info=device_info_json,
            )
            .on_conflict_do_nothing(index_elements=["scan_id"])
        )
        # scan_session in 'in_progress' state — keyframe_count starts at 0 and
        # is filled at finalize from the sidecar.
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        await self._session.execute(
            pg_insert(t.scan_session)
            .values(
                scan_id=effective_scan_id,
                started_at=now_ms,
                ended_at=None,
                device_model=(device_info_json or {}).get("model", "unknown")
                if isinstance(device_info_json, dict)
                else "unknown",
                app_version=(device_info_json or {}).get("app_version", "0.0")
                if isinstance(device_info_json, dict)
                else "0.0",
                state="in_progress",
                keyframe_count=0,
                notes=None,
            )
            .on_conflict_do_nothing(index_elements=["scan_id"])
        )

        max_order = (
            await self._session.execute(
                sa.select(
                    sa.func.coalesce(sa.func.max(t.floor_scan.c.upload_order), 0)
                ).where(t.floor_scan.c.floor_id == str(floor_id))
            )
        ).scalar_one()
        # New scan becomes active; previous active scans on the floor stay as
        # historic rows (active=false).
        await self._session.execute(
            sa.update(t.floor_scan)
            .where(t.floor_scan.c.floor_id == str(floor_id))
            .values(active=False)
        )
        await self._session.execute(
            pg_insert(t.floor_scan)
            .values(
                floor_id=str(floor_id),
                scan_id=effective_scan_id,
                file_name=f"stream_{effective_scan_id}.db",
                file_size=0,
                status="OPEN",
                active=True,
                upload_order=int(max_order) + 1,
            )
            .on_conflict_do_update(
                constraint="uq_floor_scan_scan",
                set_={"status": "OPEN", "active": True},
            )
        )
        await self._session.commit()

        logger.info(
            "scan stream started scan_id=%s floor_id=%s storage=%s",
            effective_scan_id, floor_id, storage_path,
        )
        return ScanStartResult(
            scan_id=effective_scan_id,
            floor_id=str(floor_id),
            storage_path=storage_path,
            rtabmap_db_path=str(rtabmap_db),
            state="OPEN",
        )

    async def append_frames(
        self,
        *,
        scan_id: str,
        frames: list[FrameRecord],
        links: list[LinkRecord],
    ) -> FrameBatchResult:
        scan_id = scan_id.lower()
        status_value = await self._fetch_scan_status(scan_id=scan_id)
        if status_value is None:
            raise V1ServiceError(
                404, "SCAN_NOT_FOUND",
                "scan not found — call /scans/start first",
            )
        if status_value != "OPEN":
            raise V1ServiceError(
                409, "SCAN_NOT_OPEN",
                f"scan is in state {status_value!r}; cannot append frames",
            )

        rtabmap_db = settings.storage_root / "scans" / scan_id / "rtabmap.db"
        if not rtabmap_db.exists():
            raise V1ServiceError(
                500, "RTABMAP_DB_MISSING",
                "rtabmap.db is missing for an OPEN scan",
            )

        frames_in = len(frames)
        links_in = len(links)
        try:
            frames_applied, links_applied = await asyncio.to_thread(
                append_batch, rtabmap_db, frames, links
            )
        except FrameAppendError as e:
            raise V1ServiceError(
                400, "FRAME_APPEND_FAILED", str(e),
            ) from e

        last_id = await asyncio.to_thread(last_node_id, rtabmap_db)
        total_nodes = await asyncio.to_thread(node_count, rtabmap_db)

        logger.info(
            "frame batch appended scan_id=%s frames=%d/%d links=%d/%d "
            "last_node=%d total_nodes=%d",
            scan_id, frames_applied, frames_in, links_applied, links_in,
            last_id, total_nodes,
        )
        return FrameBatchResult(
            scan_id=scan_id,
            frames_applied=frames_applied,
            frames_skipped=frames_in - frames_applied,
            links_applied=links_applied,
            links_skipped=links_in - links_applied,
            last_node_id=last_id,
            node_count=total_nodes,
        )

    async def finalize_scan(
        self,
        *,
        scan_id: str,
        sidecar_upload: UploadFile,
        manifest_upload: UploadFile,
    ) -> ScanFinalizeResult:
        scan_id = scan_id.lower()
        status_value = await self._fetch_scan_status(scan_id=scan_id)
        if status_value is None:
            raise V1ServiceError(
                404, "SCAN_NOT_FOUND", "scan not found",
            )
        if status_value == "READY":
            raise V1ServiceError(
                409, "SCAN_ALREADY_FINALIZED", "scan is already finalized",
            )
        if status_value != "OPEN":
            raise V1ServiceError(
                409, "SCAN_NOT_OPEN",
                f"scan is in state {status_value!r}; cannot finalize",
            )

        scan_dir = settings.storage_root / "scans" / scan_id
        rtabmap_db = scan_dir / "rtabmap.db"
        sidecar_path = scan_dir / "scan_metadata.db"
        manifest_path = scan_dir / "manifest.json"
        if not rtabmap_db.exists():
            raise V1ServiceError(
                500, "RTABMAP_DB_MISSING",
                "rtabmap.db is missing for the open scan",
            )

        # Stream the sidecar upload to disk. Never trust client-provided size.
        total = 0
        tmp_path = scan_dir / "scan_metadata.db.partial"
        with open(tmp_path, "wb") as f:
            while chunk := await sidecar_upload.read(1 << 18):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    f.close()
                    tmp_path.unlink(missing_ok=True)
                    raise V1ServiceError(
                        413, "PAYLOAD_TOO_LARGE",
                        "scan_metadata.db too large",
                    )
                f.write(chunk)
        tmp_path.replace(sidecar_path)

        # Read + validate manifest, then write to disk. Client must provide it
        # — the build pipeline routes reprocess based on manifest.mode, so the
        # server refuses to fabricate one.
        manifest_bytes = await manifest_upload.read()
        if not manifest_bytes:
            raise V1ServiceError(
                400, "MANIFEST_REQUIRED",
                "manifest.json multipart part is required",
            )
        try:
            await asyncio.to_thread(
                _validate_and_write_manifest,
                manifest_bytes,
                manifest_path,
                scan_id,
            )
        except _ManifestInvalid as e:
            raise V1ServiceError(
                400, "MANIFEST_INVALID", str(e),
            ) from e

        # iOS clients generate their own scan_id internally and stamp it into
        # scan_metadata.db (and within keyframe_meta/poi_mark/... rows). If
        # the client didn't pass that scan_id back to /scans/start, we now
        # have a mismatch between the streaming scan_id and the sidecar's
        # internal scan_id, which SidecarParser would 400 on. Rewrite the
        # sidecar in-place so the streaming scan_id is canonical.
        try:
            rewritten_from = await asyncio.to_thread(
                _rewrite_sidecar_scan_id, sidecar_path, scan_id
            )
            if rewritten_from is not None and rewritten_from.lower() != scan_id:
                logger.info(
                    "sidecar scan_id rewritten on finalize: %s → %s",
                    rewritten_from, scan_id,
                )
            contents = await asyncio.to_thread(
                SidecarParser().parse, sidecar_path, scan_id
            )
        except Exception as e:
            raise V1ServiceError(
                400, "SIDECAR_PARSE_FAILED", str(e),
            ) from e

        # Recompute rtabmap.db sha256 so scan_ingest reflects the final payload.
        sha256 = await asyncio.to_thread(sha256_of_db, rtabmap_db)
        file_size = rtabmap_db.stat().st_size
        total_nodes = await asyncio.to_thread(node_count, rtabmap_db)

        # We replace any rows that the placeholder open-scan rows produced
        # (scan_session keyframe_count=0). Use the existing repo with
        # replace=True semantics — it CASCADE-deletes scan_session and re-inserts.
        repo = ScanIngestRepository(self._session)
        await repo.ingest(
            contents=contents,
            storage_path=f"scans/{scan_id}",
            payload_sha256=sha256,
            device_info=None,
            replace=True,
        )
        # Mark the scan ready.
        await self._session.execute(
            sa.update(t.floor_scan)
            .where(t.floor_scan.c.scan_id == scan_id)
            .values(status="READY", file_size=file_size)
        )
        await self._session.commit()

        logger.info(
            "scan stream finalized scan_id=%s nodes=%d keyframes=%d pois=%d "
            "sha256=%s",
            scan_id, total_nodes, len(contents.keyframes),
            len(contents.poi_marks), sha256,
        )

        floor_row = (
            await self._session.execute(
                sa.select(t.floor_scan.c.floor_id).where(
                    t.floor_scan.c.scan_id == scan_id
                )
            )
        ).first()
        return ScanFinalizeResult(
            scan_id=scan_id,
            floor_id=str(floor_row.floor_id) if floor_row else "",
            state="READY",
            node_count=total_nodes,
            keyframe_count=len(contents.keyframes),
            poi_mark_count=len(contents.poi_marks),
            payload_sha256=sha256,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _fetch_scan_status(self, *, scan_id: str) -> str | None:
        row = (
            await self._session.execute(
                sa.select(t.floor_scan.c.status).where(
                    t.floor_scan.c.scan_id == scan_id
                )
            )
        ).first()
        return None if row is None else str(row.status)

    async def _fetch_floor_scan_status(
        self, *, floor_id: UUID, scan_id: str
    ) -> str | None:
        row = (
            await self._session.execute(
                sa.select(t.floor_scan.c.status).where(
                    t.floor_scan.c.floor_id == str(floor_id),
                    t.floor_scan.c.scan_id == scan_id,
                )
            )
        ).first()
        return None if row is None else str(row.status)


# Tables in scan_metadata.db (v6+) that carry a scan_id column we may need to
# rewrite. scan_session uses the column name `id` instead.
_SIDECAR_SCAN_ID_TABLES = (
    "keyframe_meta",
    "poi_mark",
    "poi_photo",
    "branch_mark",
    "interfloor_mark",
    "branch_edge",
)


def _rewrite_sidecar_scan_id(sidecar_path: Path, new_scan_id: str) -> str | None:
    """Rewrite scan_session.id and every `*.scan_id` column to `new_scan_id`.

    Returns the previous scan_id (if found), or None if the sidecar already
    matched. Sync — wrap in `asyncio.to_thread`.
    """
    import sqlite3

    con = sqlite3.connect(str(sidecar_path))
    try:
        row = con.execute(
            "SELECT id FROM scan_session LIMIT 1"
        ).fetchone()
        old_id = row[0] if row else None
        if old_id is None or str(old_id).lower() == new_scan_id.lower():
            return old_id
        con.execute(
            "UPDATE scan_session SET id = ? WHERE id = ?",
            (new_scan_id, old_id),
        )
        for tbl in _SIDECAR_SCAN_ID_TABLES:
            try:
                con.execute(
                    f"UPDATE {tbl} SET scan_id = ? WHERE scan_id = ?",
                    (new_scan_id, old_id),
                )
            except sqlite3.OperationalError:
                # Table absent for this sidecar version — skip.
                pass
        con.commit()
        return old_id
    finally:
        con.close()


class _ManifestInvalid(Exception):
    """Client-provided manifest.json failed validation."""


# build_service._maybe_reprocess switches on manifest.mode to decide whether
# to run rtabmap-reprocess + pose_backfill. Only these are recognized.
_RECOGNIZED_MANIFEST_MODES = (
    "raw_arkit_recording",
    "raw_video_recording",
    "live_rtabmap",
)


def _validate_and_write_manifest(
    manifest_bytes: bytes,
    manifest_path: Path,
    scan_id: str,
) -> None:
    """Validate client manifest JSON and persist to disk.

    Enforced contract:
      - parseable JSON object
      - `metadata_version` int
      - `mode` ∈ recognized values (else build_service will silently skip
        reprocess — refuse upfront so the failure is obvious)
      - `scan_id` (if present) matches the streaming scan_id (the streaming
        endpoint is the canonical scan_id)

    The full manifest body is written verbatim — extra fields like intrinsics
    or video paths are preserved for downstream consumers.
    """
    import json

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise _ManifestInvalid(f"manifest.json is not valid UTF-8 JSON: {e}") from e
    if not isinstance(manifest, dict):
        raise _ManifestInvalid("manifest.json must be a JSON object")
    if not isinstance(manifest.get("metadata_version"), int):
        raise _ManifestInvalid(
            "manifest.json metadata_version must be an integer"
        )
    mode = manifest.get("mode")
    if mode not in _RECOGNIZED_MANIFEST_MODES:
        raise _ManifestInvalid(
            f"manifest.json mode {mode!r} not recognized; "
            f"expected one of {list(_RECOGNIZED_MANIFEST_MODES)}"
        )
    manifest_scan_id = manifest.get("scan_id")
    if (
        manifest_scan_id is not None
        and str(manifest_scan_id).lower() != scan_id.lower()
    ):
        raise _ManifestInvalid(
            f"manifest.json scan_id {manifest_scan_id!r} does not match "
            f"streaming scan_id {scan_id!r}"
        )
    # Persist normalized (pretty-printed) JSON for human inspection.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
