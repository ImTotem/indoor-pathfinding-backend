"""FloorPolygonSimplifyStep 단위 테스트."""
from __future__ import annotations

import struct

import numpy as np

from indoor_server.application.building.steps.floor_polygon import FloorPolygonSimplifyStep
from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks


def _pose_identity() -> bytes:
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return struct.pack("<16f", *identity)


def _make_km(mask: np.ndarray, seq: int = 0) -> KeyframeMasks:
    from uuid import uuid4
    h, w = mask.shape
    return KeyframeMasks(
        scan_id=uuid4(),
        seq=seq,
        floor_mask=mask,
        stair_mask=np.zeros((h, w), dtype=bool),
        tx=0.0, ty=0.0, tz=0.0,
        pose_matrix=_pose_identity(),
    )


def test_rect_mask_produces_polygon() -> None:
    """50x50 사각형 mask → polygon 1개, vertices >= 4."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=bool)
    mask[20:80, 20:80] = True
    km = _make_km(mask)

    step = FloorPolygonSimplifyStep(epsilon_ratio=0.01, min_area_px=50)
    polys = step.run([km])

    assert len(polys) == 1
    fp = polys[0]
    assert fp.seq == 0
    assert fp.polygon_index == 0
    assert len(fp.vertices_px) >= 4
    assert fp.area_px > 0


def test_l_shape_mask() -> None:
    """L자 mask → polygon 1개, vertices >= 6."""
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=bool)
    mask[20:80, 20:80] = True   # 가로 상단
    mask[20:160, 20:50] = True  # 세로
    km = _make_km(mask)

    step = FloorPolygonSimplifyStep(epsilon_ratio=0.01, min_area_px=50)
    polys = step.run([km])

    assert len(polys) >= 1
    # L자는 적어도 6 vertex
    max_vertices = max(len(fp.vertices_px) for fp in polys)
    assert max_vertices >= 6


def test_small_contour_filtered() -> None:
    """min_area_px 이하 작은 contour는 제거된다."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=bool)
    # 큰 사각형
    mask[10:90, 10:90] = True
    # 아주 작은 섬 (5x5=25px)
    small_mask = np.zeros((h, w), dtype=bool)
    small_mask[2:7, 2:7] = True
    km_big = _make_km(mask, seq=0)
    km_small = _make_km(small_mask, seq=1)

    step = FloorPolygonSimplifyStep(epsilon_ratio=0.01, min_area_px=100)
    polys = step.run([km_big, km_small])

    # 작은 mask는 필터됨
    seqs = [fp.seq for fp in polys]
    assert 1 not in seqs, "작은 contour(25px)는 min_area_px=100에 걸려 제거되어야 함"
    assert 0 in seqs, "큰 사각형은 통과해야 함"


def test_empty_mask_returns_empty() -> None:
    """빈 mask → 빈 list 반환."""
    mask = np.zeros((100, 100), dtype=bool)
    km = _make_km(mask)

    step = FloorPolygonSimplifyStep()
    polys = step.run([km])
    assert polys == []


def test_horizon_mask_excludes_top_rows() -> None:
    """horizon_top_fraction=0.30이면 상위 30% 행의 floor 픽셀은 contour에 포함되지 않는다."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=bool)
    # 상위 20% 구역에만 작은 사각형 (horizon 안쪽 — 마스킹 후 사라져야 함)
    mask[5:18, 30:70] = True
    # 하위 70% 구역에 큰 사각형 (통과해야 함)
    mask[40:90, 10:90] = True
    km = _make_km(mask)

    step = FloorPolygonSimplifyStep(
        epsilon_ratio=0.01,
        min_area_px=10,
        horizon_top_fraction=0.30,
    )
    polys = step.run([km])

    # 통과한 polygon의 모든 vertex y 좌표가 horizon line(30px) 이상이어야 함
    horizon_row = int(h * 0.30)
    for fp in polys:
        for _, vy in fp.vertices_px:
            assert vy >= horizon_row, (
                f"horizon 위 vertex({vy}) 가 polygon에 포함됨 (horizon_row={horizon_row})"
            )


def test_horizon_mask_zero_fraction_leaves_mask_intact() -> None:
    """horizon_top_fraction=0이면 마스킹 없음 — 전체 mask 그대로 contour 생성."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=bool)
    mask[5:20, 20:80] = True   # 상단 사각형
    mask[60:90, 10:90] = True  # 하단 사각형
    km = _make_km(mask)

    step_no_horizon = FloorPolygonSimplifyStep(
        epsilon_ratio=0.01, min_area_px=10, horizon_top_fraction=0.0
    )
    step_with_horizon = FloorPolygonSimplifyStep(
        epsilon_ratio=0.01, min_area_px=10, horizon_top_fraction=0.30
    )

    polys_no_horizon = step_no_horizon.run([km])
    polys_with_horizon = step_with_horizon.run([km])

    # horizon 0이면 더 많은 polygon (또는 동일 — 2개 분리 contour 가능)
    assert len(polys_no_horizon) >= len(polys_with_horizon), (
        "horizon_top_fraction=0이면 제거 없어 결과가 동일하거나 더 많아야 함"
    )


def test_epsilon_ratio_affects_vertex_count() -> None:
    """epsilon이 작을수록 더 많은 vertex가 나온다 (noisy circle)."""
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=bool)
    # 원형 mask 생성
    cy, cx, r = 100, 100, 60
    y_coords, x_coords = np.ogrid[:h, :w]
    mask[(y_coords - cy) ** 2 + (x_coords - cx) ** 2 <= r ** 2] = True
    km = _make_km(mask)

    step_fine = FloorPolygonSimplifyStep(epsilon_ratio=0.001, min_area_px=50)
    step_coarse = FloorPolygonSimplifyStep(epsilon_ratio=0.05, min_area_px=50)

    polys_fine = step_fine.run([km])
    polys_coarse = step_coarse.run([km])

    assert len(polys_fine) >= 1
    assert len(polys_coarse) >= 1
    v_fine = max(len(fp.vertices_px) for fp in polys_fine)
    v_coarse = max(len(fp.vertices_px) for fp in polys_coarse)
    assert v_fine > v_coarse, "epsilon 작을수록 vertex 더 많아야 함"
