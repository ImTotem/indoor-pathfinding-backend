"""TrajectoryBufferStep 단위 테스트."""
from __future__ import annotations

import numpy as np
import pytest

from indoor_server.application.building.steps.trajectory_buffer import TrajectoryBufferStep
from indoor_server.domain.building.models import GridOrigin, WalkableGrid


def _make_grid(
    h: int = 20,
    w: int = 20,
    cell_size: float = 0.1,
    x0: float = 0.0,
    y0: float = 0.0,
) -> WalkableGrid:
    """빈 mask의 작은 합성 grid."""
    origin = GridOrigin(
        x0=x0,
        y0=y0,
        z0=0.0,
        cell_size=cell_size,
        w=w,
        h=h,
    )
    mask = np.zeros((h, w), dtype=bool)
    obs = np.zeros((h, w), dtype=np.uint16)
    return WalkableGrid(origin=origin, mask=mask, observation_count=obs)


# ── 1. happy-path: T자 trajectory buffer가 line 주변 픽셀을 채우는지 ──────────

def test_buffer_rasterizes_along_line() -> None:
    """수평선 trajectory + buffer → 선 주변 픽셀이 True, 외부는 False."""
    grid = _make_grid(h=10, w=10, cell_size=0.1, x0=0.0, y0=0.0)
    step = TrajectoryBufferStep(buffer_m=0.2)

    # 수평선: x=0.2 ~ 0.8, y=0.5 (grid 중앙)
    trajectory = [(0.2, 0.5), (0.8, 0.5)]
    result = step.run(grid, trajectory)

    # buffer=0.2m, cell=0.1m → 선 위아래 약 2셀 범위에 True
    # y=0.5는 grid pixel row ≈ 5 (y0=0), 0.2m buffer → row 3~7 범위
    assert result.mask.any(), "buffer rasterization이 아무 셀도 채우지 않음"

    # 선 중앙(x≈0.5, y≈0.5) 픽셀은 반드시 True
    cx = int((0.5 - 0.0) / 0.1)  # 5
    cy = int((0.5 - 0.0) / 0.1)  # 5
    assert result.mask[cy, cx], f"선 중앙 픽셀 ({cy},{cx})이 False"

    # 코너(0,0) 셀은 선에서 멀어 False
    assert not result.mask[0, 0], "코너 픽셀이 잘못 True"


# ── 2. UNION 동작: 기존 True 셀 보존 + buffer 신규 추가 ───────────────────────

def test_buffer_unions_preserving_existing() -> None:
    """input mask의 기존 True가 보존되고, buffer로 새 cell이 추가되어야 한다."""
    grid = _make_grid(h=20, w=20, cell_size=0.1, x0=0.0, y0=0.0)

    # (row=3, col=3) 셀만 True로 설정 — 세계 좌표 x=0.3, y=0.3
    mask_pre = grid.mask.copy()
    mask_pre[3, 3] = True
    grid_with_existing = WalkableGrid(
        origin=grid.origin,
        mask=mask_pre,
        observation_count=grid.observation_count,
    )

    before_sum = int(mask_pre.sum())  # 1

    step = TrajectoryBufferStep(buffer_m=0.1)
    # 수평선: y≈0.7, buffer 0.1m → y 0.6~0.8 구간 커버
    trajectory = [(0.0, 0.7), (0.9, 0.7)]
    result = step.run(grid_with_existing, trajectory)

    after_sum = int(result.mask.sum())

    # 기존 True 셀 (3, 3) 보존
    assert result.mask[3, 3], "기존 True 셀이 사라짐"
    # 새 셀이 추가됨
    assert after_sum > before_sum, (
        f"buffer 후 셀 수({after_sum})가 이전({before_sum})보다 작거나 같음"
    )


# ── 3. edge case: trajectory 길이 0, 1 → no-op ────────────────────────────────

@pytest.mark.parametrize(
    "trajectory",
    [
        [],
        [(0.5, 0.5)],
    ],
    ids=["empty", "single_point"],
)
def test_buffer_noop_on_degenerate_trajectory(
    trajectory: list[tuple[float, float]],
) -> None:
    """len(trajectory_pts) < 2 → 입력 grid 그대로 반환 (no-op)."""
    grid = _make_grid(h=10, w=10)
    # 일부 셀 미리 True
    mask_pre = grid.mask.copy()
    mask_pre[2, 2] = True
    grid_in = WalkableGrid(
        origin=grid.origin,
        mask=mask_pre,
        observation_count=grid.observation_count,
    )

    step = TrajectoryBufferStep(buffer_m=0.5)
    result = step.run(grid_in, trajectory)

    # 같은 객체 반환 (no-op 경로)
    assert result is grid_in, "degenerate trajectory에서 다른 WalkableGrid 객체 반환"
    np.testing.assert_array_equal(result.mask, mask_pre)
