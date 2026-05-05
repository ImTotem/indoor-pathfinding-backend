from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from indoor_server.application.building.multiscan_pose_mapping import (
    parse_provenance_label,
)
from indoor_server.application.building.multiscan_rtabmap_merge import (
    MultiScanReprocessParams,
    MultiScanRtabmapMergeError,
    SourceRtabmapScan,
    inject_provenance_labels,
    prepare_rtabmap_sources,
)


def _make_db(path: Path, *, node_count: int = 2) -> None:
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
    pose = struct.pack(
        "<12f",
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    )
    for node_id in range(1, node_count + 1):
        conn.execute(
            "INSERT INTO Node (id, map_id, stamp, pose, label) VALUES (?, 0, ?, ?, ?)",
            (node_id, 100.0 + node_id, pose, f"label-{node_id}"),
        )
    conn.commit()
    conn.close()


def test_inject_provenance_labels_updates_temp_db_only(tmp_path: Path) -> None:
    db = tmp_path / "copy.db"
    _make_db(db, node_count=2)

    count = inject_provenance_labels(db, scan_id="scan-a")

    assert count == 2
    conn = sqlite3.connect(str(db))
    labels = [row[0] for row in conn.execute("SELECT label FROM Node ORDER BY id")]
    conn.close()
    parsed = [parse_provenance_label(label) for label in labels]
    assert [item.scan_id for item in parsed if item is not None] == ["scan-a", "scan-a"]
    assert [item.original_label for item in parsed if item is not None] == [
        "label-1",
        "label-2",
    ]


def test_prepare_rtabmap_sources_rejects_duplicate_scan_id(tmp_path: Path) -> None:
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    _make_db(left)
    _make_db(right)

    with pytest.raises(MultiScanRtabmapMergeError):
        prepare_rtabmap_sources(
            sources=[
                SourceRtabmapScan("same", left),
                SourceRtabmapScan("same", right),
            ],
            work_dir=tmp_path / "work",
        )


def test_multiscan_reprocess_params_emit_append_preserve_args() -> None:
    args = MultiScanReprocessParams().to_args()

    assert "-a" in args
    assert "-skip" in args
    assert "0" in args
    assert "--Mem/RehearsalSimilarity=1.0" in args
    assert "--Mem/NotLinkedNodesKept=true" in args
    assert "--Mem/ReduceGraph=false" in args
    assert "--uwarn" in args

