"""TrajectoryBufferStep — ARKit trajectory → buffered polygon → walkable union.

triangulation walkable grid가 sparse한 feature-poor 구간을
ARKit trajectory 존재만으로 walkable로 가정하여 메꾼다.

입력 mask와 동일 shape / 동일 origin의 grid를 그대로 유지 (in-place 아님 —
새 WalkableGrid 반환). observation_count는 pass-through(buffer 영역은 0 유지).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from indoor_server.domain.building.models import WalkableGrid

if TYPE_CHECKING:
    from shapely.geometry import Polygon
    from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


class TrajectoryBufferStep:
    """trajectory line → buffered polygon → grid rasterize → walkable union.

    trajectory_pts 2개 미만이면 no-op으로 입력 그대로 반환.

    반환값 계약:
    - origin: 입력 grid.origin과 동일 (shape 변경 없음)
    - mask: 입력 mask OR trajectory buffer rasterized mask (dtype bool)
    - observation_count: 입력과 동일 참조 반환 (buffer 영역 obs=0)
    """

    def __init__(
        self,
        buffer_m: float = 0.8,
        clip_to_triangulation_hull: bool = False,  # Sprint 22 예정 slot
        hull_padding_m: float = 0.5,               # clip 활성일 때 hull 외곽 여유
    ) -> None:
        self._buffer_m = buffer_m
        self._clip_to_triangulation_hull = clip_to_triangulation_hull
        self._hull_padding_m = hull_padding_m

    def run(
        self,
        grid: WalkableGrid,
        trajectory_pts: list[tuple[float, float]],
        triangulated_xy: np.ndarray | None = None,  # clip 비활성 시 None 허용
    ) -> WalkableGrid:
        """walkable grid에 trajectory buffer를 union.

        Args:
            grid: 입력 WalkableGrid (triangulation 경로 결과).
            trajectory_pts: ARKit trajectory xy 좌표 목록 [(x, y), ...].
            triangulated_xy: (N, 2) 삼각측량 점 XY. clip_to_triangulation_hull=True일 때만 사용.

        Returns:
            fused WalkableGrid (input mask OR buffer mask).
        """
        if len(trajectory_pts) < 2:
            return grid

        from shapely.geometry import LineString, MultiPolygon, Polygon

        line = LineString(trajectory_pts)
        buffer_geom: BaseGeometry = line.buffer(
            self._buffer_m, cap_style="round", join_style="round"
        )

        polygons: list[Polygon]
        if isinstance(buffer_geom, Polygon):
            polygons = [buffer_geom]
        elif isinstance(buffer_geom, MultiPolygon):
            polygons = list(buffer_geom.geoms)
        else:
            return grid  # degenerate

        # (optional) clip by triangulated hull — Sprint 22 예정
        if (
            self._clip_to_triangulation_hull
            and triangulated_xy is not None
            and len(triangulated_xy) >= 3
        ):
            polygons = self._clip_by_hull(polygons, triangulated_xy)

        buffer_mask = self._rasterize(polygons, grid)
        fused_mask = grid.mask | buffer_mask

        rasterized_px = int(buffer_mask.sum())
        input_sum = int(grid.mask.sum())
        growth_pct = (
            (int(fused_mask.sum()) - input_sum) / max(input_sum, 1) * 100.0
        )
        logger.info(
            "trajectory_buffer: trajectory_pts=%d buffer_m=%.2f "
            "rasterized_pixels=%d union_growth=+%.1f%%",
            len(trajectory_pts),
            self._buffer_m,
            rasterized_px,
            growth_pct,
        )

        return WalkableGrid(
            origin=grid.origin,
            mask=fused_mask,
            observation_count=grid.observation_count,
        )

    def _clip_by_hull(
        self,
        polygons: list[Polygon],
        triangulated_xy: np.ndarray,
    ) -> list[Polygon]:
        """삼각측량 점의 convex hull + padding 내부로 polygon을 clip."""
        from shapely.geometry import MultiPoint, Polygon

        hull: BaseGeometry = MultiPoint(
            triangulated_xy.tolist()
        ).convex_hull.buffer(self._hull_padding_m)
        clipped: list[Polygon] = []
        for p in polygons:
            inter: BaseGeometry = p.intersection(hull)
            if inter.is_empty:
                continue
            if isinstance(inter, Polygon):
                clipped.append(inter)
            elif hasattr(inter, "geoms"):  # MultiPolygon
                clipped.extend(
                    g for g in inter.geoms if isinstance(g, Polygon)
                )
        return clipped

    def _rasterize(
        self,
        polygons: list[Polygon],
        grid: WalkableGrid,
    ) -> np.ndarray:
        """world 좌표 polygon list → grid pixel coords → cv2.fillPoly → bool mask."""
        import cv2

        origin = grid.origin
        cs = origin.cell_size
        h, w = grid.mask.shape
        buffer_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in polygons:
            coords = list(poly.exterior.coords)
            pixel_coords = np.array(
                [
                    [
                        int((x - origin.x0) / cs),
                        int((y - origin.y0) / cs),
                    ]
                    for (x, y) in coords
                ],
                dtype=np.int32,
            )
            # cv2.fillPoly가 grid 밖 좌표를 자동 clip
            cv2.fillPoly(buffer_mask, [pixel_coords], color=1)

        return buffer_mask.astype(bool)
