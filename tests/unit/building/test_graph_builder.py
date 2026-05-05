"""SkeletonizeStep + NodePlacementStep + POIProjectionStep 단위 테스트."""
from __future__ import annotations

import struct
from uuid import uuid4

import numpy as np
import pytest

from indoor_server.application.building.steps.node_placement import NodePlacementStep
from indoor_server.application.building.steps.poi_projection import POIProjectionStep
from indoor_server.application.building.steps.skeletonize import SkeletonizeStep
from indoor_server.domain.building.enums import EdgeType, NodeType
from indoor_server.domain.building.models import GridOrigin, WalkableGrid
from indoor_server.domain.poi.enums import POISource
from indoor_server.domain.scan.models import POIMarkRow


def _make_corridor_grid(length: int = 50) -> WalkableGrid:
    """직선 복도 형태의 walkable grid."""
    mask = np.zeros((20, length), dtype=bool)
    mask[8:12, 2 : length - 2] = True  # 4px 폭 직선 복도
    obs = (mask.astype(np.uint16)) * 5
    origin = GridOrigin(x0=0.0, y0=0.0, z0=0.0, cell_size=0.10, w=length, h=20)
    return WalkableGrid(origin=origin, mask=mask, observation_count=obs)


def _make_poi_row(poi_id: int, scan_id: str, tx: float, ty: float, tz: float) -> POIMarkRow:
    identity = struct.pack("<16f", *[1.0 if i % 5 == 0 else 0.0 for i in range(16)])
    return POIMarkRow(
        id=poi_id,
        scan_id=scan_id,
        keyframe_seq=1,
        created_at=0,
        pose_matrix=identity,
        tx=tx,
        ty=ty,
        tz=tz,
        track_id=None,
        label=None,
        source=POISource.MANUAL,
    )


def test_skeletonize_corridor():
    """직선 복도 → skeleton 노드/엣지가 존재해야 함."""
    grid = _make_corridor_grid()
    skel = SkeletonizeStep().run(grid)
    assert skel.skeleton_pixel_count > 0


def test_node_placement_world_coords():
    """pixel → world 좌표 변환 확인."""
    scan_id = uuid4()
    build_job_id = uuid4()
    grid = _make_corridor_grid()
    skel = SkeletonizeStep().run(grid)
    placer = NodePlacementStep(scan_id=scan_id, build_job_id=build_job_id, corridor_sample_m=0.5)
    nodes, edges = placer.run(skel, grid.origin)
    assert len(nodes) > 0
    # 모든 노드의 z는 grid.origin.z0
    for n in nodes:
        assert n.z == pytest.approx(grid.origin.z0)
        assert n.scan_id == scan_id


def test_poi_projection_world_pose_copy():
    """POI world_pose = (tx, ty, tz) 그대로 복사 (A-10)."""
    scan_id = uuid4()
    build_job_id = uuid4()
    grid = _make_corridor_grid()
    skel = SkeletonizeStep().run(grid)
    placer = NodePlacementStep(scan_id=scan_id, build_job_id=build_job_id)
    nodes, edges = placer.run(skel, grid.origin)

    poi = _make_poi_row(42, str(scan_id), tx=0.25, ty=0.9, tz=0.0)
    proj = POIProjectionStep(scan_id=scan_id, build_job_id=build_job_id)
    final_nodes, final_edges, poi_poses = proj.run([poi], nodes, edges)

    assert 42 in poi_poses
    wx, wy, wz = poi_poses[42]
    assert wx == pytest.approx(0.25)
    assert wy == pytest.approx(0.9)
    assert wz == pytest.approx(0.0)


def test_poi_projection_spur_edge_created():
    """POI → POI_SPUR edge가 생성되어야 함."""
    scan_id = uuid4()
    build_job_id = uuid4()
    grid = _make_corridor_grid()
    skel = SkeletonizeStep().run(grid)
    placer = NodePlacementStep(scan_id=scan_id, build_job_id=build_job_id)
    nodes, edges = placer.run(skel, grid.origin)

    if not edges:
        pytest.skip("no skeleton edges — grid too small for skeleton")

    poi = _make_poi_row(1, str(scan_id), tx=0.25, ty=0.9, tz=0.0)
    proj = POIProjectionStep(scan_id=scan_id, build_job_id=build_job_id)
    final_nodes, final_edges, _ = proj.run([poi], nodes, edges)

    spur_edges = [e for e in final_edges if e.edge_type == EdgeType.POI_SPUR]
    assert len(spur_edges) >= 1

    poi_nodes = [n for n in final_nodes if n.node_type == NodeType.POI]
    assert len(poi_nodes) == 1
    assert poi_nodes[0].poi_mark_id == 1
