"""WalkableGridStep — 3D 포인트 누적 → 2D walkable grid."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from indoor_server.domain.building.models import GridOrigin, WalkableGrid

# ASSUMPTION A-1: 기본 셀 크기 10cm, 관측 횟수 임계 3
_DEFAULT_CELL_SIZE = 0.10
_DEFAULT_OBS_THRESHOLD = 3
_MORPH_KERNEL_SIZE = 3
_TRAJECTORY_RADIUS_M = 20.0


class WalkableGridStep:
    """world frame 3D point cloud → 2D walkable grid."""

    def __init__(
        self,
        cell_size: float = _DEFAULT_CELL_SIZE,
        obs_threshold: int = _DEFAULT_OBS_THRESHOLD,
    ) -> None:
        self._cell_size = cell_size
        self._obs_threshold = obs_threshold

    def estimate_z0(self, tz_values: list[float]) -> float:
        """trajectory tz 값의 5%ile → floor plane z0 (A-1)."""
        if not tz_values:
            return 0.0
        return float(np.percentile(tz_values, 5))

    def run(
        self,
        points_iter: Iterable[np.ndarray],
        trajectory_pts: list[tuple[float, float]],
        z0: float | None = None,
    ) -> WalkableGrid:
        """
        points_iter: Iterable[ndarray (N, 3)]
        trajectory_pts: [(x, y)] 궤적 xy 좌표 목록 (CC 필터용)
        z0: floor plane z 높이. None이면 점군 z 평균으로 fallback (비추천).
            파이프라인은 estimate_z0() 결과를 전달해야 한다 (W-7).
        """
        all_points = np.vstack(
            [p for p in points_iter if len(p) > 0]
            or [np.zeros((0, 3))]
        )

        if len(all_points) == 0:
            resolved_z0 = z0 if z0 is not None else 0.0
            origin = GridOrigin(x0=0.0, y0=0.0, z0=resolved_z0, cell_size=self._cell_size, w=1, h=1)
            mask = np.zeros((1, 1), dtype=bool)
            obs_count = np.zeros((1, 1), dtype=np.uint16)
            return WalkableGrid(origin=origin, mask=mask, observation_count=obs_count)

        xs = all_points[:, 0]
        ys = all_points[:, 1]
        # W-7: 외부에서 estimate_z0(5%ile) 값을 전달받아 사용. fallback은 mean (비추천).
        resolved_z0 = z0 if z0 is not None else float(np.mean(all_points[:, 2]))

        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        # Sprint 21: trajectory_pts도 bbox에 포함 — back-projected 점이 trajectory의 일부
        # 구간에서만 생기는 경우(triangulation 30% strip 등) trajectory buffer가 grid 밖으로
        # 잘려 빈 구간이 생기는 회귀 방지.
        if trajectory_pts:
            traj_arr = np.asarray(trajectory_pts, dtype=np.float64)
            x_min = min(x_min, float(traj_arr[:, 0].min()))
            x_max = max(x_max, float(traj_arr[:, 0].max()))
            y_min = min(y_min, float(traj_arr[:, 1].min()))
            y_max = max(y_max, float(traj_arr[:, 1].max()))
        cs = self._cell_size

        w = max(1, int(np.ceil((x_max - x_min) / cs)) + 2)
        h = max(1, int(np.ceil((y_max - y_min) / cs)) + 2)

        origin = GridOrigin(x0=x_min - cs, y0=y_min - cs, z0=resolved_z0, cell_size=cs, w=w, h=h)

        obs_count = np.zeros((h, w), dtype=np.uint16)
        gx = np.clip(((xs - origin.x0) / cs).astype(int), 0, w - 1)
        gy = np.clip(((ys - origin.y0) / cs).astype(int), 0, h - 1)
        np.add.at(obs_count, (gy, gx), 1)

        walkable = obs_count >= self._obs_threshold

        # morphological close
        walkable = self._morph_close(walkable)

        # connected component 필터 (궤적 반경 이내만)
        if trajectory_pts:
            walkable = self._filter_by_trajectory(walkable, origin, trajectory_pts)

        return WalkableGrid(origin=origin, mask=walkable, observation_count=obs_count)

    def _morph_close(self, mask: np.ndarray) -> np.ndarray:
        from scipy.ndimage import binary_closing

        kernel = np.ones((_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE), dtype=bool)
        closed: np.ndarray = binary_closing(mask, structure=kernel)
        return closed.astype(bool)

    def _filter_by_trajectory(
        self,
        mask: np.ndarray,
        origin: GridOrigin,
        trajectory_pts: list[tuple[float, float]],
    ) -> np.ndarray:
        """궤적 바운딩박스 반경 R=20m 이내 CC만 유지."""
        from scipy.ndimage import label

        if not trajectory_pts:
            return mask

        traj_arr = np.array(trajectory_pts)
        cx = float(traj_arr[:, 0].mean())
        cy = float(traj_arr[:, 1].mean())

        labeled, num_features = label(mask)
        result = np.zeros_like(mask)

        for comp_id in range(1, num_features + 1):
            comp_ys, comp_xs = np.where(labeled == comp_id)
            world_xs = origin.x0 + comp_xs * origin.cell_size
            world_ys = origin.y0 + comp_ys * origin.cell_size
            dist_sq = (world_xs - cx) ** 2 + (world_ys - cy) ** 2
            if float(dist_sq.min()) <= _TRAJECTORY_RADIUS_M ** 2:
                result[labeled == comp_id] = True

        return result
