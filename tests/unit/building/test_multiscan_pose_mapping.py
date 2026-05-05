from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from indoor_server.application.building.multiscan_pose_mapping import (
    SourceNodeRef,
    assign_frame_pose_source,
    build_node_pose_mapping,
    build_provenance_label,
    parse_provenance_label,
    resolve_frame_pose_in_merged,
)


def _pose_blob(tx: float) -> bytes:
    values = [
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    return struct.pack("<12f", *values)


def _make_db(
    path: Path,
    *,
    stamps: list[float],
    labels: list[str | None] | None = None,
    start_node_id: int = 1,
    map_ids: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE Node (
            id INTEGER NOT NULL PRIMARY KEY,
            map_id INTEGER NOT NULL,
            stamp FLOAT,
            pose BLOB,
            label TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE Link (
            from_id INTEGER NOT NULL,
            to_id INTEGER NOT NULL,
            type INTEGER NOT NULL,
            transform BLOB
        )
        """
    )
    for index, stamp in enumerate(stamps):
        node_id = start_node_id + index
        label = labels[index] if labels is not None else None
        map_id = map_ids[index] if map_ids is not None else 0
        conn.execute(
            "INSERT INTO Node (id, map_id, stamp, pose, label) VALUES (?, ?, ?, ?, ?)",
            (node_id, map_id, stamp, _pose_blob(float(node_id)), label),
        )
    conn.commit()
    conn.close()


def test_provenance_label_roundtrip() -> None:
    source = SourceNodeRef(
        scan_id="scan A",
        node_id=42,
        stamp=1234.5,
        original_label="door left",
    )

    parsed = parse_provenance_label(build_provenance_label(source))

    assert parsed == source


def test_build_node_pose_mapping_prefers_label_provenance(tmp_path: Path) -> None:
    left_db = tmp_path / "left.db"
    right_db = tmp_path / "right.db"
    _make_db(left_db, stamps=[10.0, 11.0])
    _make_db(right_db, stamps=[20.0], start_node_id=10)
    merged_labels = [
        build_provenance_label(SourceNodeRef("left", 1, 10.0)),
        build_provenance_label(SourceNodeRef("left", 2, 11.0)),
        build_provenance_label(SourceNodeRef("right", 10, 20.0)),
    ]
    merged_db = tmp_path / "merged.db"
    _make_db(
        merged_db,
        stamps=[100.0, 101.0, 102.0],
        labels=merged_labels,
        start_node_id=100,
    )

    result = build_node_pose_mapping(
        source_dbs={"left": left_db, "right": right_db},
        merged_db=merged_db,
    )

    assert result.metrics.label_count == 3
    assert result.metrics.usable_ratio == pytest.approx(1.0)
    assert result.metrics.per_scan_usable_ratio() == {"left": 1.0, "right": 1.0}
    assert {m.merged_node_id for m in result.mappings} == {100, 101, 102}


def test_timestamp_fallback_marks_ambiguous_when_two_scans_share_stamp(
    tmp_path: Path,
) -> None:
    left_db = tmp_path / "left.db"
    right_db = tmp_path / "right.db"
    _make_db(left_db, stamps=[10.0])
    _make_db(right_db, stamps=[10.0])
    merged_db = tmp_path / "merged.db"
    _make_db(merged_db, stamps=[10.0])

    result = build_node_pose_mapping(
        source_dbs={"left": left_db, "right": right_db},
        merged_db=merged_db,
        stamp_tolerance_s=0.05,
    )

    assert result.metrics.ambiguous_count == 2
    assert result.metrics.usable_count == 0
    assert all(mapping.mapping_source == "ambiguous" for mapping in result.mappings)


def test_timestamp_fallback_uses_unique_nearest_candidate(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    _make_db(source_db, stamps=[10.0, 10.030])
    merged_db = tmp_path / "merged.db"
    _make_db(merged_db, stamps=[10.0, 10.030], start_node_id=100)

    result = build_node_pose_mapping(
        source_dbs={"scan": source_db},
        merged_db=merged_db,
        stamp_tolerance_s=0.033,
    )

    assert result.metrics.stamp_count == 2
    assert result.metrics.ambiguous_count == 0
    assert result.metrics.per_scan_usable_ratio() == {"scan": 1.0}


def test_timestamp_fallback_uses_merged_map_id_when_available(
    tmp_path: Path,
) -> None:
    left_db = tmp_path / "left.db"
    right_db = tmp_path / "right.db"
    _make_db(left_db, stamps=[10.0])
    _make_db(right_db, stamps=[10.0])
    merged_db = tmp_path / "merged.db"
    _make_db(merged_db, stamps=[10.0, 10.0], start_node_id=100, map_ids=[0, 1])

    result = build_node_pose_mapping(
        source_dbs={"left": left_db, "right": right_db},
        merged_db=merged_db,
        stamp_tolerance_s=0.05,
    )

    assert result.metrics.stamp_count == 2
    assert result.metrics.ambiguous_count == 0
    assert result.metrics.per_scan_usable_ratio() == {"left": 1.0, "right": 1.0}


def test_assign_frame_pose_source_uses_interpolation_between_mapped_nodes(
    tmp_path: Path,
) -> None:
    source_db = tmp_path / "source.db"
    _make_db(source_db, stamps=[10.0, 12.0])
    merged_db = tmp_path / "merged.db"
    _make_db(
        merged_db,
        stamps=[10.0, 12.0],
        labels=[
            build_provenance_label(SourceNodeRef("scan", 1, 10.0)),
            build_provenance_label(SourceNodeRef("scan", 2, 12.0)),
        ],
    )
    mapping = build_node_pose_mapping(
        source_dbs={"scan": source_db},
        merged_db=merged_db,
    )

    assignment = assign_frame_pose_source(
        source_scan_id="scan",
        source_timestamp=11.0,
        mapping_result=mapping,
    )

    assert assignment.pose_source == "interpolated_optimized_pose"
    assert assignment.confidence == pytest.approx(0.85)
    assert assignment.before_node_id == 1
    assert assignment.after_node_id == 2


def test_resolve_frame_pose_in_merged_applies_source_to_merged_correction(
    tmp_path: Path,
) -> None:
    source_db = tmp_path / "source.db"
    _make_db(source_db, stamps=[0.0], start_node_id=1)
    merged_db = tmp_path / "merged.db"
    _make_db(
        merged_db,
        stamps=[0.0],
        labels=[build_provenance_label(SourceNodeRef("scan", 1, 0.0))],
        start_node_id=100,
    )
    # Override merged pose to be source pose shifted by +5m on x.
    conn = sqlite3.connect(str(merged_db))
    conn.execute("UPDATE Node SET pose = ? WHERE id = 100", (_pose_blob(6.0),))
    conn.commit()
    conn.close()
    mapping = build_node_pose_mapping(
        source_dbs={"scan": source_db},
        merged_db=merged_db,
    )

    raw_pose = np.eye(4, dtype=np.float64)
    raw_pose[0, 3] = 2.0
    resolved = resolve_frame_pose_in_merged(
        source_scan_id="scan",
        source_timestamp=0.0,
        raw_source_pose=raw_pose,
        mapping_result=mapping,
    )

    assert resolved.pose is not None
    # source node pose x=1, optimized pose x=6 -> correction +5.
    assert resolved.pose[0][3] == pytest.approx(7.0)
