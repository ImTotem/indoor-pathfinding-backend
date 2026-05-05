"""scan_metadata.db v4 read-only 파서. stdlib sqlite3만 사용."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from indoor_server.domain.poi.enums import DetectionSource, POISource
from indoor_server.domain.scan.errors import ScanIdMismatch, SidecarSchemaMismatch
from indoor_server.domain.scan.models import (
    BranchMarkRow,
    InterfloorMarkRow,
    KeyframeRow,
    POIMarkRow,
    POIPhotoRow,
    ScanSessionRow,
    SidecarContents,
    YOLODetectionRow,
)

logger = logging.getLogger(__name__)

# Sprint 49 v5: poi_photo.image_blob 추가.
# Sprint 65 v6: yolo_detection drop, poi_mark.track_id drop, poi_photo.bbox_* drop,
#               interfloor_mark 테이블 신설.
EXPECTED_USER_VERSIONS = frozenset({4, 5, 6})
# legacy 지원 별칭 (외부 import)
EXPECTED_USER_VERSION = 4

# 테이블별 필수 컬럼 선언 (v4/v5 공통).
# v6 에서는 일부 컬럼/테이블이 사라지므로 _required_columns_for(version) 으로 분기.
_REQUIRED_COLUMNS_BASE: dict[str, list[str]] = {
    "scan_session": [
        "id", "started_at", "ended_at", "device_model",
        "app_version", "state", "keyframe_count", "notes",
    ],
    "keyframe_meta": [
        "scan_id", "seq", "captured_at", "image_path", "pose_matrix",
        "tx", "ty", "tz", "tracking_state", "rtabmap_node_id",
    ],
    "poi_mark": [
        "id", "scan_id", "keyframe_seq", "created_at", "pose_matrix",
        "tx", "ty", "tz", "track_id", "label", "source",
    ],
    "poi_photo": [
        "id", "poi_mark_id", "scan_id", "keyframe_seq", "captured_at",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "class_name", "confidence",
    ],
    "branch_mark": [
        "id", "scan_id", "keyframe_seq", "created_at", "pose_matrix",
        "tx", "ty", "tz",
    ],
    "yolo_detection": [
        "id", "scan_id", "keyframe_seq", "class_name", "confidence",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "mask_rle", "source", "track_id",
    ],
}


def _required_columns_for(version: int) -> dict[str, list[str]]:
    if version <= 5:
        return _REQUIRED_COLUMNS_BASE
    # Sprint 65 v6:
    #   - yolo_detection 테이블 자체 삭제.
    #   - poi_mark.track_id 삭제.
    #   - poi_photo.bbox_x/y/w/h 삭제.
    #   - interfloor_mark 테이블 신설.
    cols: dict[str, list[str]] = {}
    cols["scan_session"] = _REQUIRED_COLUMNS_BASE["scan_session"]
    cols["keyframe_meta"] = _REQUIRED_COLUMNS_BASE["keyframe_meta"]
    cols["poi_mark"] = [
        c for c in _REQUIRED_COLUMNS_BASE["poi_mark"] if c != "track_id"
    ]
    cols["poi_photo"] = [
        c for c in _REQUIRED_COLUMNS_BASE["poi_photo"]
        if c not in {"bbox_x", "bbox_y", "bbox_w", "bbox_h"}
    ]
    cols["branch_mark"] = _REQUIRED_COLUMNS_BASE["branch_mark"]
    cols["interfloor_mark"] = [
        "id", "scan_id", "keyframe_seq", "created_at",
        "connector_type", "prefix", "pose_matrix", "tx", "ty", "tz",
    ]
    return cols


# v5+ 추가 컬럼 (선택적: 존재해야만 함, 없으면 schema 위반)
V5_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "poi_photo": ["image_blob"],
}

# 외부 호환 alias (legacy code 가 import 함)
REQUIRED_COLUMNS = _REQUIRED_COLUMNS_BASE


class SidecarParser:
    def parse(self, sidecar_path: Path, expected_scan_id: str) -> SidecarContents:
        """
        PRAGMA user_version ∈ {4, 5} 확인 + 필수 테이블/컬럼 검증 + 모든 row 로드.
        read-only URI 모드로 오픈 — 서버가 사이드카 DB를 절대 수정하지 않음.

        Sprint 49: v5 schema 는 v4 + poi_photo.image_blob 컬럼.
        """
        uri = f"file:{sidecar_path}?mode=ro&immutable=1"
        try:
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.OperationalError as e:
            raise SidecarSchemaMismatch(f"scan_metadata.db 오픈 실패: {e}") from e

        conn.row_factory = sqlite3.Row
        try:
            return self._parse_with_conn(conn, expected_scan_id)
        finally:
            conn.close()

    def get_user_version(self, sidecar_path: Path) -> int:
        """Sprint 49: PRAGMA user_version 만 조회 (manifest 일관성 검증용)."""
        uri = f"file:{sidecar_path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        try:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])
        finally:
            conn.close()

    # ── internal ──────────────────────────────────────────────────────────────

    def _parse_with_conn(self, conn: sqlite3.Connection, expected_scan_id: str) -> SidecarContents:
        version = self._validate_version(conn)
        self._validate_schema(conn, version=version)

        session = self._load_session(conn, expected_scan_id)
        keyframes = self._load_keyframes(conn, expected_scan_id)
        poi_marks = self._load_poi_marks(conn, expected_scan_id, version=version)
        poi_photos = self._load_poi_photos(conn, expected_scan_id, version=version)
        branch_marks = self._load_branch_marks(conn, expected_scan_id)
        # Sprint 65 v6: yolo_detection 테이블 자체 없음. v5 이하는 기존 로딩 유지.
        yolo_detections = (
            self._load_yolo_detections(conn, expected_scan_id) if version <= 5 else []
        )
        # Sprint 65 v6: interfloor_mark 신설. v5 이하는 빈 리스트.
        interfloor_marks = (
            self._load_interfloor_marks(conn, expected_scan_id) if version >= 6 else []
        )

        return SidecarContents(
            scan_session=session,
            keyframes=keyframes,
            poi_marks=poi_marks,
            poi_photos=poi_photos,
            branch_marks=branch_marks,
            yolo_detections=yolo_detections,
            interfloor_marks=interfloor_marks,
        )

    def _validate_version(self, conn: sqlite3.Connection) -> int:
        version: int = conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in EXPECTED_USER_VERSIONS:
            raise SidecarSchemaMismatch(
                "scan_metadata.db user_version 불일치",
                detail={
                    "expected_user_versions": sorted(EXPECTED_USER_VERSIONS),
                    "got_user_version": version,
                },
            )
        return version

    def _validate_schema(self, conn: sqlite3.Connection, *, version: int) -> None:
        missing: list[str] = []
        for table, required_cols in _required_columns_for(version).items():
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            existing = {r["name"] for r in rows}
            for col in required_cols:
                if col not in existing:
                    missing.append(f"{table}.{col}")

        if version >= 5:
            for table, required_cols in V5_REQUIRED_COLUMNS.items():
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
                existing = {r["name"] for r in rows}
                for col in required_cols:
                    if col not in existing:
                        missing.append(f"{table}.{col}")

        if missing:
            raise SidecarSchemaMismatch(
                f"scan_metadata.db 스키마가 v{version} 요구사항과 다릅니다.",
                detail={"missing_columns": missing},
            )

    def _load_session(self, conn: sqlite3.Connection, expected_scan_id: str) -> ScanSessionRow:
        rows = conn.execute("SELECT * FROM scan_session").fetchall()
        if len(rows) != 1:
            raise SidecarSchemaMismatch(
                f"scan_session 행 수가 1이 아닙니다: {len(rows)}"
            )
        row = rows[0]
        if row["id"].lower() != expected_scan_id.lower():
            raise ScanIdMismatch(
                f"scan_id 불일치: form={expected_scan_id!r}, sidecar={row['id']!r}"
            )
        return ScanSessionRow(**{**dict(row), "id": expected_scan_id})

    def _load_keyframes(self, conn: sqlite3.Connection, scan_id: str) -> list[KeyframeRow]:
        rows = conn.execute(
            "SELECT * FROM keyframe_meta WHERE lower(scan_id) = lower(?) ORDER BY seq",
            (scan_id,),
        ).fetchall()
        return [KeyframeRow(**{**dict(r), "scan_id": scan_id}) for r in rows]

    def _load_poi_marks(
        self, conn: sqlite3.Connection, scan_id: str, *, version: int
    ) -> list[POIMarkRow]:
        rows = conn.execute(
            "SELECT * FROM poi_mark WHERE lower(scan_id) = lower(?) ORDER BY id",
            (scan_id,),
        ).fetchall()
        result: list[POIMarkRow] = []
        for r in rows:
            data = dict(r)
            data["scan_id"] = scan_id
            data["source"] = POISource(r["source"])
            # Sprint 65 v6: track_id 컬럼 없음 → None 으로 채워 downstream 호환 유지.
            if version >= 6:
                data["track_id"] = None
            result.append(POIMarkRow(**data))
        return result

    def _load_poi_photos(
        self, conn: sqlite3.Connection, scan_id: str, *, version: int
    ) -> list[POIPhotoRow]:
        rows = conn.execute(
            "SELECT * FROM poi_photo WHERE lower(scan_id) = lower(?) ORDER BY id",
            (scan_id,),
        ).fetchall()
        result: list[POIPhotoRow] = []
        for r in rows:
            data = dict(r)
            data["scan_id"] = scan_id
            # Sprint 65 v6: bbox_* 컬럼 없음 → None 으로 채움.
            if version >= 6:
                data["bbox_x"] = None
                data["bbox_y"] = None
                data["bbox_w"] = None
                data["bbox_h"] = None
            if version <= 4:
                data["image_blob"] = None
            result.append(POIPhotoRow(**data))
        return result

    def _load_branch_marks(self, conn: sqlite3.Connection, scan_id: str) -> list[BranchMarkRow]:
        rows = conn.execute(
            "SELECT * FROM branch_mark WHERE lower(scan_id) = lower(?) ORDER BY id",
            (scan_id,),
        ).fetchall()
        return [BranchMarkRow(**{**dict(r), "scan_id": scan_id}) for r in rows]

    def _load_yolo_detections(
        self, conn: sqlite3.Connection, scan_id: str
    ) -> list[YOLODetectionRow]:
        rows = conn.execute(
            "SELECT * FROM yolo_detection WHERE lower(scan_id) = lower(?) ORDER BY id",
            (scan_id,),
        ).fetchall()
        return [
            YOLODetectionRow(
                **{**dict(r), "scan_id": scan_id, "source": DetectionSource(r["source"])}
            )
            for r in rows
        ]

    def _load_interfloor_marks(
        self, conn: sqlite3.Connection, scan_id: str
    ) -> list[InterfloorMarkRow]:
        """Sprint 65 v6: interfloor_mark 테이블 read."""
        rows = conn.execute(
            "SELECT * FROM interfloor_mark WHERE lower(scan_id) = lower(?) ORDER BY id",
            (scan_id,),
        ).fetchall()
        return [
            InterfloorMarkRow(**{**dict(r), "scan_id": scan_id})
            for r in rows
        ]
