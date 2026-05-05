from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np

from indoor_server.application.building.multiscan_pose_mapping import (
    SourceNodeRef,
    build_node_pose_mapping,
    build_provenance_label,
)
from indoor_server.application.building.steps.multiscan_rtabmap_floor_polygon import (
    merge_obstacle_heatmaps,
    optimized_nodes_for_scan,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
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
    scan_id: str,
    source_tx: float,
    merged_tx: float | None = None,
) -> None:
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
    label = None
    tx = source_tx
    node_id = 1
    if merged_tx is not None:
        label = build_provenance_label(SourceNodeRef(scan_id, 1, 0.0))
        tx = merged_tx
        node_id = 100
    conn.execute(
        "INSERT INTO Node (id, map_id, stamp, pose, label) VALUES (?, 0, 0.0, ?, ?)",
        (node_id, _pose_blob(tx), label),
    )
    conn.commit()
    conn.close()


def test_optimized_nodes_for_scan_uses_merged_pose(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    merged_db = tmp_path / "merged.db"
    _make_db(source_db, scan_id="scan", source_tx=1.0)
    _make_db(merged_db, scan_id="scan", source_tx=1.0, merged_tx=5.0)
    mapping = build_node_pose_mapping(
        source_dbs={"scan": source_db},
        merged_db=merged_db,
    )

    nodes = optimized_nodes_for_scan(mapping_result=mapping, scan_id="scan")

    assert len(nodes) == 1
    assert nodes[0].node_id == 1
    assert nodes[0].pose[0][3] == 5.0


def test_merge_obstacle_heatmaps_places_counts_in_common_grid() -> None:
    left = ObstacleHeatmap(
        counts=np.array([[1]], dtype=np.int32),
        origin_x=0.0,
        origin_y=0.0,
        cell_size_m=1.0,
        z0=0.0,
        height_min_m=0.3,
        height_max_m=2.5,
        metadata={"world_obstacle_point_count": 1},
    )
    right = ObstacleHeatmap(
        counts=np.array([[2]], dtype=np.int32),
        origin_x=2.0,
        origin_y=0.0,
        cell_size_m=1.0,
        z0=0.0,
        height_min_m=0.3,
        height_max_m=2.5,
        metadata={"world_obstacle_point_count": 2},
    )

    merged = merge_obstacle_heatmaps([left, right])

    assert merged is not None
    assert merged.counts.shape == (1, 3)
    assert merged.counts[0, 0] == 1
    assert merged.counts[0, 2] == 2
