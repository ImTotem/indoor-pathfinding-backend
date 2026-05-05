"""FilesystemDebugSink / NoopDebugSink 단위 테스트."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from uuid import uuid4

import numpy as np

from indoor_server.application.building.debug.filesystem_sink import FilesystemDebugSink
from indoor_server.application.building.debug.sink import NoopDebugSink
from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks
from indoor_server.application.building.steps.skeletonize import SkeletonGraph
from indoor_server.domain.building.models import GridOrigin, WalkableGrid


def _fake_pose_bytes() -> bytes:
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return struct.pack("<16f", *identity)


def _make_grid(h: int = 20, w: int = 20) -> WalkableGrid:
    mask = np.zeros((h, w), dtype=bool)
    mask[5:15, 5:15] = True
    obs = mask.astype(np.uint16) * 3
    origin = GridOrigin(x0=0.0, y0=0.0, z0=0.0, cell_size=0.10, w=w, h=h)
    return WalkableGrid(origin=origin, mask=mask, observation_count=obs)


def _make_km() -> KeyframeMasks:
    scan_id = uuid4()
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    return KeyframeMasks(
        scan_id=scan_id,
        seq=1,
        floor_mask=mask,
        stair_mask=np.zeros((64, 64), dtype=bool),
        tx=0.0, ty=0.0, tz=0.0,
        pose_matrix=_fake_pose_bytes(),
    )


# ── NoopDebugSink ─────────────────────────────────────────────────────────────

def test_noop_sink_no_exception() -> None:
    """NoopDebugSink 모든 메서드 호출 시 예외 없음."""
    sink = NoopDebugSink()
    grid = _make_grid()
    km = _make_km()

    sink.on_floor_segmentation([km])
    sink.on_back_projection(np.zeros((10, 3), dtype=np.float32), z0=0.0)
    sink.on_walkable_grid(grid)
    sink.on_skeleton(grid.mask, SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=0))
    sink.on_node_placement([], [], grid.origin)
    sink.finalize()


# ── FilesystemDebugSink ───────────────────────────────────────────────────────

def test_filesystem_sink_files_created(tmp_path: Path) -> None:
    """FilesystemDebugSink가 npz/json 파일을 생성한다."""
    out_dir = tmp_path / "debug_out"
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True)

    sink = FilesystemDebugSink(out_dir=out_dir, storage_root=storage_root)
    grid = _make_grid()
    skel_mask = grid.mask.copy()
    skel = SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=0)

    sink.on_walkable_grid(grid)
    sink.on_skeleton(skel_mask, skel)
    sink.on_node_placement([], [], grid.origin)

    # 파일 존재 + 크기 > 0
    assert (out_dir / "walkable_grid.npz").exists()
    assert (out_dir / "walkable_grid.npz").stat().st_size > 0
    assert (out_dir / "skeleton.npz").exists()
    assert (out_dir / "skeleton.npz").stat().st_size > 0
    assert (out_dir / "nodes.json").exists()
    assert (out_dir / "edges.json").exists()


def test_filesystem_sink_npz_keys(tmp_path: Path) -> None:
    """walkable_grid.npz에 필수 키가 존재한다."""
    out_dir = tmp_path / "debug_keys"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    sink = FilesystemDebugSink(out_dir=out_dir, storage_root=storage_root)
    grid = _make_grid()
    sink.on_walkable_grid(grid)

    npz = np.load(str(out_dir / "walkable_grid.npz"))
    for key in ["mask", "observation_count", "origin_x0", "origin_y0", "origin_z0",
                "origin_cell_size", "origin_w", "origin_h"]:
        assert key in npz, f"missing key: {key}"


def test_filesystem_sink_skeleton_npz(tmp_path: Path) -> None:
    """skeleton.npz 'skeleton' 키 존재."""
    out_dir = tmp_path / "debug_skel"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    sink = FilesystemDebugSink(out_dir=out_dir, storage_root=storage_root)
    skel_mask = np.ones((10, 10), dtype=bool)
    sink.on_skeleton(skel_mask, SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=100))

    npz = np.load(str(out_dir / "skeleton.npz"))
    assert "skeleton" in npz
    assert npz["skeleton"].shape == (10, 10)


def test_filesystem_sink_nodes_edges_json(tmp_path: Path) -> None:
    """nodes.json / edges.json이 유효한 JSON 리스트."""
    out_dir = tmp_path / "debug_json"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    sink = FilesystemDebugSink(out_dir=out_dir, storage_root=storage_root)
    grid = _make_grid()
    sink.on_node_placement([], [], grid.origin)

    nodes_data = json.loads((out_dir / "nodes.json").read_text())
    edges_data = json.loads((out_dir / "edges.json").read_text())
    assert isinstance(nodes_data, list)
    assert isinstance(edges_data, list)


def test_filesystem_sink_finalize_creates_composite(tmp_path: Path) -> None:
    """finalize() 시 composite.png 생성 + 이미지로 열림."""
    import cv2

    out_dir = tmp_path / "debug_final"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    sink = FilesystemDebugSink(out_dir=out_dir, storage_root=storage_root)
    grid = _make_grid()
    skel_mask = grid.mask.copy()
    skel = SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=0)

    sink.on_walkable_grid(grid)
    sink.on_skeleton(skel_mask, skel)
    sink.on_node_placement([], [], grid.origin)
    sink.finalize()

    composite_path = out_dir / "composite.png"
    assert composite_path.exists(), "composite.png should be created"
    img = cv2.imread(str(composite_path))
    assert img is not None, "composite.png must be a valid image"
    h, w = img.shape[:2]
    assert h > 0 and w > 0
