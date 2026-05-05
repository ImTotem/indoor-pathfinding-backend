"""Read-only reader for uploaded RTAB-Map SQLite databases."""
from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from indoor_server.domain.building.rtabmap_models import (
    Matrix4x4,
    RtabmapDataFrame,
    RtabmapDataSummary,
    RtabmapDiagnostics,
    RtabmapFeaturePoint,
    RtabmapLink,
    RtabmapNode,
)

_REQUIRED_TABLES = {
    "Node",
    "Data",
    "Link",
    "Feature",
    "Word",
    "Statistics",
    "GlobalDescriptor",
}

_MIN_NODE_COUNT = 3
_MIN_DATA_NODE_RATIO = 0.8
_MIN_FEATURES_PER_NODE = 100
_MIN_FEATURE_3D_RATIO = 0.8
# Sidecar keyframes can include frames rejected by RTAB-Map.  The map source is
# RTAB-Map Node/Data itself, so this gate should catch broken propagation without
# rejecting real scans that still have sufficient RTAB-Map evidence.
_MIN_KEYFRAME_NODE_COVERAGE = 0.4


class RtabmapReader:
    """Small SQLite reader around RTAB-Map's uploaded database artifact.

    The reader never mutates the file and deliberately returns diagnostics instead
    of silently falling back to legacy keyframe/raw-pose map building.
    """

    def inspect(
        self,
        db_path: Path,
        *,
        keyframe_node_ids: Iterable[int | None] | None = None,
    ) -> RtabmapDiagnostics:
        if not db_path.exists():
            return RtabmapDiagnostics.missing(db_path, "rtabmap_db_missing")
        if not db_path.is_file():
            return RtabmapDiagnostics.missing(db_path, "rtabmap_db_not_file")

        try:
            with closing(self._connect(db_path)) as conn:
                missing_tables = self._missing_tables(conn)
                if missing_tables:
                    return RtabmapDiagnostics(
                        db_path=str(db_path),
                        ready=False,
                        issues=[f"missing_tables:{','.join(sorted(missing_tables))}"],
                    )
                return self._inspect_with_conn(
                    conn,
                    db_path=db_path,
                    keyframe_node_ids=keyframe_node_ids,
                )
        except sqlite3.DatabaseError as e:
            return RtabmapDiagnostics.missing(db_path, f"sqlite_open_failed:{e}")

    def load_nodes(self, db_path: Path) -> list[RtabmapNode]:
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT id, map_id, stamp, pose, label FROM Node ORDER BY id"
            ).fetchall()
        return [
            RtabmapNode(
                node_id=int(row["id"]),
                map_id=int(row["map_id"]),
                stamp=float(row["stamp"] or 0.0),
                pose=decode_pose_3x4_blob(row["pose"]),
                label=row["label"],
            )
            for row in rows
        ]

    def load_links(self, db_path: Path) -> list[RtabmapLink]:
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT from_id, to_id, type, transform FROM Link ORDER BY from_id, to_id, type"
            ).fetchall()
        return [
            RtabmapLink(
                from_id=int(row["from_id"]),
                to_id=int(row["to_id"]),
                link_type=int(row["type"]),
                transform=(
                    decode_pose_3x4_blob(row["transform"])
                    if row["transform"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    def load_data_summaries(self, db_path: Path) -> list[RtabmapDataSummary]:
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute(
                """
                SELECT
                  id,
                  length(image) AS image_bytes,
                  length(depth) AS depth_bytes,
                  length(calibration) AS calibration_bytes,
                  length(scan) AS scan_bytes
                FROM Data
                ORDER BY id
                """
            ).fetchall()
        return [
            RtabmapDataSummary(
                node_id=int(row["id"]),
                image_bytes=_optional_int(row["image_bytes"]),
                depth_bytes=_optional_int(row["depth_bytes"]),
                calibration_bytes=_optional_int(row["calibration_bytes"]),
                scan_bytes=_optional_int(row["scan_bytes"]),
            )
            for row in rows
        ]

    def load_data_frames(
        self,
        db_path: Path,
        *,
        limit: int | None = None,
    ) -> list[RtabmapDataFrame]:
        sql = """
            SELECT id, image, depth, calibration
            FROM Data
            ORDER BY id
        """
        if limit is not None:
            sql += " LIMIT ?"
            params: tuple[int, ...] = (limit,)
        else:
            params = ()
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            RtabmapDataFrame(
                node_id=int(row["id"]),
                image_bytes=row["image"],
                depth_bytes=row["depth"],
                calibration_bytes=row["calibration"],
            )
            for row in rows
        ]

    def load_feature_points(
        self,
        db_path: Path,
        *,
        include_descriptors: bool = False,
        limit: int | None = None,
    ) -> list[RtabmapFeaturePoint]:
        descriptor_select = (
            "descriptor_size, descriptor"
            if include_descriptors
            else "descriptor_size, NULL AS descriptor"
        )
        sql = f"""
            SELECT
              node_id,
              word_id,
              pos_x,
              pos_y,
              depth_x,
              depth_y,
              depth_z,
              {descriptor_select}
            FROM Feature
            WHERE depth_x IS NOT NULL
              AND depth_y IS NOT NULL
              AND depth_z IS NOT NULL
            ORDER BY node_id, word_id
        """
        if limit is not None:
            sql += " LIMIT ?"
            params: tuple[int, ...] = (limit,)
        else:
            params = ()
        with closing(self._connect(db_path)) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            RtabmapFeaturePoint(
                node_id=int(row["node_id"]),
                word_id=int(row["word_id"]),
                pixel_x=float(row["pos_x"]),
                pixel_y=float(row["pos_y"]),
                local_xyz=(
                    float(row["depth_x"]),
                    float(row["depth_y"]),
                    float(row["depth_z"]),
                ),
                descriptor_size=_optional_int(row["descriptor_size"]),
                descriptor=row["descriptor"],
            )
            for row in rows
        ]

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        uri = f"file:{db_path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _missing_tables(self, conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        existing = {str(row["name"]) for row in rows}
        return _REQUIRED_TABLES - existing

    def _inspect_with_conn(
        self,
        conn: sqlite3.Connection,
        *,
        db_path: Path,
        keyframe_node_ids: Iterable[int | None] | None,
    ) -> RtabmapDiagnostics:
        node_stats = conn.execute(
            """
            SELECT
              COUNT(*) AS node_count,
              MIN(id) AS min_node_id,
              MAX(id) AS max_node_id,
              MIN(stamp) AS min_stamp,
              MAX(stamp) AS max_stamp
            FROM Node
            """
        ).fetchone()
        data_stats = conn.execute(
            """
            SELECT
              COUNT(*) AS data_count,
              SUM(CASE WHEN image IS NOT NULL THEN 1 ELSE 0 END) AS image_count,
              SUM(CASE WHEN depth IS NOT NULL THEN 1 ELSE 0 END) AS depth_count,
              SUM(CASE WHEN calibration IS NOT NULL THEN 1 ELSE 0 END) AS calibration_count
            FROM Data
            """
        ).fetchone()
        link_stats = conn.execute(
            """
            SELECT
              COUNT(*) AS link_count,
              SUM(CASE WHEN type = 0 THEN 1 ELSE 0 END) AS neighbor_link_count,
              SUM(CASE WHEN type IN (1,2,3,4,5,6) THEN 1 ELSE 0 END) AS loop_link_count
            FROM Link
            """
        ).fetchone()
        feature_stats = conn.execute(
            """
            SELECT
              COUNT(*) AS feature_count,
              SUM(
                CASE
                  WHEN depth_x IS NOT NULL AND depth_y IS NOT NULL AND depth_z IS NOT NULL
                  THEN 1 ELSE 0
                END
              ) AS feature_3d_count
            FROM Feature
            """
        ).fetchone()
        word_count = self._scalar_count(conn, "Word")
        descriptor_count = self._scalar_count(conn, "GlobalDescriptor")
        statistics_count = self._scalar_count(conn, "Statistics")

        node_count = int(node_stats["node_count"] or 0)
        data_count = int(data_stats["data_count"] or 0)
        feature_count = int(feature_stats["feature_count"] or 0)
        feature_3d_count = int(feature_stats["feature_3d_count"] or 0)
        image_count = int(data_stats["image_count"] or 0)
        depth_count = int(data_stats["depth_count"] or 0)
        calibration_count = int(data_stats["calibration_count"] or 0)
        keyframe_count, keyframe_node_count, coverage = _keyframe_coverage(
            keyframe_node_ids
        )

        min_stamp = _optional_float(node_stats["min_stamp"])
        max_stamp = _optional_float(node_stats["max_stamp"])
        issues = _readiness_issues(
            node_count=node_count,
            data_count=data_count,
            feature_count=feature_count,
            feature_3d_count=feature_3d_count,
            image_count=image_count,
            keyframe_count=keyframe_count,
            keyframe_node_coverage=coverage,
        )

        return RtabmapDiagnostics(
            db_path=str(db_path),
            ready=not issues,
            issues=issues,
            node_count=node_count,
            data_count=data_count,
            link_count=int(link_stats["link_count"] or 0),
            neighbor_link_count=int(link_stats["neighbor_link_count"] or 0),
            loop_closure_link_count=int(link_stats["loop_link_count"] or 0),
            feature_count=feature_count,
            feature_3d_count=feature_3d_count,
            word_count=word_count,
            global_descriptor_count=descriptor_count,
            statistics_count=statistics_count,
            data_image_count=image_count,
            data_depth_count=depth_count,
            data_calibration_count=calibration_count,
            keyframe_count=keyframe_count,
            keyframe_node_count=keyframe_node_count,
            keyframe_node_coverage=coverage,
            min_node_id=_optional_int(node_stats["min_node_id"]),
            max_node_id=_optional_int(node_stats["max_node_id"]),
            min_stamp=min_stamp,
            max_stamp=max_stamp,
            duration_s=(
                float(max_stamp - min_stamp)
                if min_stamp is not None and max_stamp is not None
                else None
            ),
        )

    def _scalar_count(self, conn: sqlite3.Connection, table: str) -> int:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"] or 0)


def decode_pose_3x4_blob(blob: bytes) -> Matrix4x4:
    """Decode RTAB-Map 3x4 float32 pose BLOB into a homogeneous 4x4 matrix."""
    if len(blob) != 48:
        raise ValueError(f"RTAB-Map pose blob must be 48 bytes, got {len(blob)}")
    values = struct.unpack("<12f", blob)
    return (
        (float(values[0]), float(values[1]), float(values[2]), float(values[3])),
        (float(values[4]), float(values[5]), float(values[6]), float(values[7])),
        (float(values[8]), float(values[9]), float(values[10]), float(values[11])),
        (0.0, 0.0, 0.0, 1.0),
    )


def _keyframe_coverage(
    keyframe_node_ids: Iterable[int | None] | None,
) -> tuple[int, int, float]:
    if keyframe_node_ids is None:
        return 0, 0, 0.0
    values = list(keyframe_node_ids)
    if not values:
        return 0, 0, 0.0
    set_count = sum(1 for value in values if value is not None)
    return len(values), set_count, set_count / len(values)


def _readiness_issues(
    *,
    node_count: int,
    data_count: int,
    feature_count: int,
    feature_3d_count: int,
    image_count: int,
    keyframe_count: int,
    keyframe_node_coverage: float,
) -> list[str]:
    issues: list[str] = []
    if node_count < _MIN_NODE_COUNT:
        issues.append(f"node_count_below_min:{node_count}<{_MIN_NODE_COUNT}")
    if node_count > 0 and data_count / node_count < _MIN_DATA_NODE_RATIO:
        issues.append(
            f"data_node_ratio_below_min:{data_count / node_count:.3f}"
            f"<{_MIN_DATA_NODE_RATIO:.3f}"
        )
    if image_count < data_count:
        issues.append(f"data_image_missing:{image_count}/{data_count}")
    if node_count > 0 and feature_count / node_count < _MIN_FEATURES_PER_NODE:
        issues.append(
            f"features_per_node_below_min:{feature_count / node_count:.1f}"
            f"<{_MIN_FEATURES_PER_NODE}"
        )
    if feature_count > 0 and feature_3d_count / feature_count < _MIN_FEATURE_3D_RATIO:
        issues.append(
            f"feature_3d_ratio_below_min:{feature_3d_count / feature_count:.3f}"
            f"<{_MIN_FEATURE_3D_RATIO:.3f}"
        )
    if keyframe_count > 0 and keyframe_node_coverage < _MIN_KEYFRAME_NODE_COVERAGE:
        issues.append(
            f"keyframe_node_coverage_below_min:{keyframe_node_coverage:.3f}"
            f"<{_MIN_KEYFRAME_NODE_COVERAGE:.3f}"
        )
    return issues


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
