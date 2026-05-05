from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import av
import numpy as np
import pytest

from indoor_server.application.building.multiscan_pose_mapping import (
    SourceNodeRef,
    build_node_pose_mapping,
    build_provenance_label,
)
from indoor_server.application.building.steps.back_projection import Intrinsics
from indoor_server.application.building.steps.floor_raster import (
    FloorRasterStep,
    FloorRasterStepParams,
)
from indoor_server.application.building.steps.multiscan_dense_video_polygon import (
    MultiScanDenseVideoPolygonParams,
    MultiScanDenseVideoPolygonStep,
    MultiScanDenseVideoSource,
)
from indoor_server.infrastructure.ml.segformer_onnx import SegmentationOutput

ADE20K_FLOOR_IDX = 3


class FakeSegmenter:
    async def segment(self, image: np.ndarray) -> SegmentationOutput:
        h, w = image.shape[:2]
        return SegmentationOutput(
            class_mask=np.full((h, w), ADE20K_FLOOR_IDX, dtype=np.int32)
        )


def _pose_blob(tx: float, *, z: float = 1.0) -> bytes:
    values = [
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, z,
    ]
    return struct.pack("<12f", *values)


def _make_db(path: Path, *, scan_id: str, tx: float, merged_id: int | None = None) -> None:
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
    node_id = merged_id or 1
    label = (
        build_provenance_label(SourceNodeRef(scan_id, 1, 0.0))
        if merged_id is not None
        else None
    )
    conn.execute(
        "INSERT INTO Node (id, map_id, stamp, pose, label) VALUES (?, 0, 0.0, ?, ?)",
        (node_id, _pose_blob(tx), label),
    )
    conn.commit()
    conn.close()


def _make_mp4(path: Path, *, num_frames: int = 2, fps: int = 30) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    for i in range(num_frames):
        arr = np.full((48, 64, 3), 100 + i * 10, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_poses_bin(path: Path, *, start_ns: int, num_records: int = 2, fps: int = 30) -> None:
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 1.0
    floats = list(pose.T.flatten())
    buf = bytearray()
    for i in range(num_records):
        pts_ns = start_ns + int(round(i / fps * 1_000_000_000))
        buf.extend(struct.pack("<q", pts_ns))
        buf.extend(struct.pack("<16f", *floats))
    path.write_bytes(bytes(buf))


@pytest.mark.asyncio
async def test_multiscan_dense_video_polygon_fuses_original_video_evidence(
    tmp_path: Path,
) -> None:
    left_db = tmp_path / "left.db"
    right_db = tmp_path / "right.db"
    merged_db = tmp_path / "merged.db"
    _make_db(left_db, scan_id="left", tx=0.0)
    _make_db(right_db, scan_id="right", tx=0.0)
    _make_db(merged_db, scan_id="left", tx=0.0, merged_id=100)
    conn = sqlite3.connect(str(merged_db))
    conn.execute(
        "INSERT INTO Node (id, map_id, stamp, pose, label) VALUES (?, 0, 0.0, ?, ?)",
        (
            101,
            _pose_blob(1.0),
            build_provenance_label(SourceNodeRef("right", 1, 0.0)),
        ),
    )
    conn.commit()
    conn.close()
    mapping = build_node_pose_mapping(
        source_dbs={"left": left_db, "right": right_db},
        merged_db=merged_db,
    )

    left_mp4 = tmp_path / "left.mp4"
    right_mp4 = tmp_path / "right.mp4"
    left_poses = tmp_path / "left_poses.bin"
    right_poses = tmp_path / "right_poses.bin"
    _make_mp4(left_mp4)
    _make_mp4(right_mp4)
    _write_poses_bin(left_poses, start_ns=0)
    # Starts at 10s to verify video matching can rebase pose timestamps.
    _write_poses_bin(right_poses, start_ns=10_000_000_000)

    intrinsics = Intrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0)
    step = MultiScanDenseVideoPolygonStep(
        segmenter=FakeSegmenter(),
        params=MultiScanDenseVideoPolygonParams(
            stride=1,
            pixel_stride=4,
            source_timestamp_mode="rebased_pose",
            exact_node_tolerance_s=0.2,
        ),
        raster_step=FloorRasterStep(
            FloorRasterStepParams(
                min_cell_hits=1,
                morph_open_radius_cells=0,
                morph_close_radius_cells=0,
                keep_largest_component=False,
            )
        ),
    )

    result = await step.run(
        sources=[
            MultiScanDenseVideoSource("left", left_mp4, left_poses, intrinsics),
            MultiScanDenseVideoSource("right", right_mp4, right_poses, intrinsics),
        ],
        mapping_result=mapping,
        z0=0.0,
        inter_session_loop_closure_count=1,
    )

    assert result.cloud.points_xy.shape[0] > 0
    assert result.raster.footprint_geojson is not None
    assert result.metadata["per_scan"]["left"]["point_count"] > 0
    assert result.metadata["per_scan"]["right"]["point_count"] > 0
    assert result.fusion_metrics.scan_support_ratio["left"] > 0.0
    assert result.fusion_metrics.scan_support_ratio["right"] > 0.0
