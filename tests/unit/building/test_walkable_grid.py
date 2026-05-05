"""WalkableGridStep 단위 테스트."""
from __future__ import annotations

import numpy as np
import pytest

from indoor_server.application.building.steps.walkable_grid import WalkableGridStep


def test_z0_estimation():
    tz_values = [0.0, 0.1, 0.2, 1.0, 2.0, 3.0]
    step = WalkableGridStep()
    z0 = step.estimate_z0(tz_values)
    # 5%ile는 최소 근처
    assert z0 <= 0.2


def test_empty_points():
    step = WalkableGridStep()
    grid = step.run(iter([np.zeros((0, 3))]), trajectory_pts=[])
    assert grid.mask.sum() == 0


def test_accumulation_threshold():
    """같은 셀에 3회 이상 hit → walkable. 그리드가 충분히 커야 morph close가 셀을 지우지 않음."""
    step = WalkableGridStep(cell_size=0.10, obs_threshold=3)
    # 3x3 블록 영역에 4회씩 hit — 그리드가 크므로 morphological close 후에도 생존
    base = np.array([[x, y, 0.02] for x in np.arange(0.5, 0.9, 0.1)
                     for y in np.arange(0.5, 0.9, 0.1)], dtype=np.float64)
    pts = np.tile(base, (4, 1))  # 각 셀을 4번 hit
    traj = [(0.7, 0.7)]
    grid = step.run(iter([pts]), trajectory_pts=traj)
    # 최소 1개 walkable cell이 있어야 함
    assert grid.mask.sum() >= 1


def test_grid_origin_bounds():
    """grid origin이 point cloud를 포함해야 함."""
    step = WalkableGridStep(cell_size=0.10)
    pts = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
        [3.0, 4.0, 0.0],
        [3.0, 4.0, 0.0],
    ])
    grid = step.run(iter([pts]), trajectory_pts=[(2.0, 3.0)])
    assert grid.origin.x0 <= 1.0
    assert grid.origin.y0 <= 2.0


# ── W-7: GridOrigin.z0와 estimate_z0 정합성 회귀 테스트 ─────────────────────────

def test_z0_passed_to_grid_origin():
    """W-7: z0 파라미터로 전달한 값이 GridOrigin.z0에 그대로 반영돼야 한다."""
    step = WalkableGridStep(cell_size=0.10)
    expected_z0 = 0.042  # estimate_z0 결과라고 가정

    # z 값을 의도적으로 expected_z0와 다르게 설정
    pts = np.array([
        [0.5, 0.5, 0.50],
        [0.6, 0.5, 0.50],
        [0.5, 0.6, 0.50],
        [0.6, 0.6, 0.50],
    ] * 4)  # 각 셀 4번 히트 → walkable

    grid = step.run(iter([pts]), trajectory_pts=[(0.55, 0.55)], z0=expected_z0)

    # GridOrigin.z0는 점군의 z 평균(0.50)이 아닌 전달받은 z0(0.042)여야 한다
    assert grid.origin.z0 == pytest.approx(expected_z0, abs=1e-9), (
        f"GridOrigin.z0={grid.origin.z0!r} should equal passed z0={expected_z0}"
    )


def test_z0_not_overridden_by_point_cloud_mean():
    """W-7: z0=0.02를 전달하면 점군 z 평균(1.5)이 아닌 0.02가 origin.z0에 저장된다."""
    step = WalkableGridStep(cell_size=0.10)
    forced_z0 = 0.02

    # z 평균이 1.5인 점군
    pts = np.array([[x, y, 1.5] for x in np.arange(0.1, 0.5, 0.1)
                    for y in np.arange(0.1, 0.5, 0.1)] * 4)

    grid = step.run(iter([pts]), trajectory_pts=[(0.3, 0.3)], z0=forced_z0)

    assert grid.origin.z0 == pytest.approx(forced_z0, abs=1e-9)
    assert abs(grid.origin.z0 - 1.5) > 0.1, "z0가 점군 평균(1.5)으로 덮어써지면 안 됨"


def test_z0_none_fallback_uses_point_mean():
    """z0=None 이면 점군 z 평균을 사용하는 fallback 경로도 정상 동작해야 한다."""
    step = WalkableGridStep(cell_size=0.10)
    pts = np.array([[x, y, 0.8] for x in np.arange(0.1, 0.5, 0.1)
                    for y in np.arange(0.1, 0.5, 0.1)] * 4)

    grid = step.run(iter([pts]), trajectory_pts=[(0.3, 0.3)], z0=None)

    # fallback: 점군 z 평균 ≈ 0.8
    assert abs(grid.origin.z0 - 0.8) < 0.01


def test_empty_z0_with_explicit_z0():
    """빈 점군에서도 z0가 명시적으로 전달되면 origin.z0에 반영된다."""
    step = WalkableGridStep(cell_size=0.10)
    grid = step.run(iter([np.zeros((0, 3))]), trajectory_pts=[], z0=0.123)

    assert grid.origin.z0 == pytest.approx(0.123, abs=1e-9)
