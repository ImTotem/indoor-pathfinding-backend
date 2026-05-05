"""FloorLayoutStep — per-frame polygon list → world MultiPolygon."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from indoor_server.application.building.steps.back_projection import (
    decode_pose_matrix,
    project_pixels_to_floor,
)
from indoor_server.domain.building.models import FloorLayout, FloorPolygon, WalkableGrid

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

_DEFAULT_SIMPLIFY_TOL_M = 0.05  # 5cm
_DEFAULT_MIN_AREA_M2 = 0.5
_DEFAULT_BUFFER_DISTANCE = 0.0   # buffer(0) for geometry cleanup
_DEFAULT_MAX_VERTEX_DISTANCE_M = 15.0  # 카메라로부터 수평 거리 상한 (관대하게)
_DEFAULT_TRAJECTORY_BUFFER_M = 5.0  # trajectory 주변 keep 반경


class FloorLayoutStep:
    """
    per-frame FloorPolygon 목록을 world frame에서 합집합하여 FloorLayout 반환.

    알고리즘:
        1. 각 polygon vertex를 project_pixels_to_floor()로 world (x, y) 변환
        2. Fix 2: 카메라로부터 수평 거리가 max_vertex_distance_m 초과하는 vertex가
           하나라도 있으면 polygon 통째 drop (spike 방지)
        3. shapely Polygon 생성 후 unary_union
        4. buffer(0) → simplify → area 필터
    """

    def __init__(
        self,
        simplify_tolerance_m: float = _DEFAULT_SIMPLIFY_TOL_M,
        min_area_m2: float = _DEFAULT_MIN_AREA_M2,
        max_vertex_distance_m: float = _DEFAULT_MAX_VERTEX_DISTANCE_M,
        trajectory_buffer_m: float = _DEFAULT_TRAJECTORY_BUFFER_M,
    ) -> None:
        self._simplify_tolerance_m = simplify_tolerance_m
        self._min_area_m2 = min_area_m2
        self._max_vertex_distance_m = max_vertex_distance_m
        self._trajectory_buffer_m = trajectory_buffer_m

    def run(
        self,
        polygons: list[FloorPolygon],
        z0: float,
        trajectory_xy: list[tuple[float, float]] | None = None,
    ) -> FloorLayout:
        """
        polygons: FloorPolygonSimplifyStep 출력 전량.
        z0: floor plane height (estimate_z0 결과).
        반환: FloorLayout (multipolygon이 MultiPolygon 인스턴스).
        """
        from shapely.geometry import MultiPolygon
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union

        if not polygons:
            empty = MultiPolygon()
            return FloorLayout(
                multipolygon=empty,
                z0=z0,
                sub_polygon_count=0,
                total_area_m2=0.0,
            )

        world_polys: list[ShapelyPolygon] = []
        dropped_by_distance = 0
        for fp in polygons:
            world_xy = self._back_project_vertices(fp, z0)
            if len(world_xy) < 3:
                continue
            # Fix 2: vertex distance filter — 카메라 origin과의 수평 거리 검사
            if self._has_far_vertex(world_xy, fp):
                dropped_by_distance += 1
                continue
            sp = ShapelyPolygon(world_xy)
            if not sp.is_valid:
                sp = sp.buffer(0)
            if sp.is_empty or sp.area <= 0:
                continue
            world_polys.append(sp)

        if dropped_by_distance > 0:
            logger.info(
                "FloorLayoutStep: %d polygon(s) dropped by distance filter (max=%.1fm)",
                dropped_by_distance,
                self._max_vertex_distance_m,
            )

        if not world_polys:
            empty = MultiPolygon()
            return FloorLayout(
                multipolygon=empty,
                z0=z0,
                sub_polygon_count=0,
                total_area_m2=0.0,
            )

        union = unary_union(world_polys).buffer(_DEFAULT_BUFFER_DISTANCE)
        union = union.simplify(self._simplify_tolerance_m, preserve_topology=True)
        union = self._filter_by_area(union)

        union_geom: BaseGeometry = union
        if hasattr(union_geom, "geoms"):
            count = len(list(union_geom.geoms))
        else:
            count = 0 if union_geom.is_empty else 1
        total_area = float(union_geom.area)

        return FloorLayout(
            multipolygon=union_geom,
            z0=z0,
            sub_polygon_count=count,
            total_area_m2=total_area,
        )

    def _back_project_vertices(
        self, fp: FloorPolygon, z0: float
    ) -> list[tuple[float, float]]:
        """FloorPolygon vertex → world (x, y) 목록. None은 제외."""
        world_pts = project_pixels_to_floor(
            pixel_coords=fp.vertices_px,
            pose_bytes=fp.pose_matrix,
            image_w=fp.image_w,
            image_h=fp.image_h,
            z0=z0,
        )
        return [pt for pt in world_pts if pt is not None]

    def _has_far_vertex(
        self,
        world_xy: list[tuple[float, float]],
        fp: FloorPolygon,
    ) -> bool:
        """카메라 origin에서 수평 거리가 max_vertex_distance_m 초과하는 vertex가 있으면 True."""
        pose_mat = decode_pose_matrix(fp.pose_matrix)
        cam_x = float(pose_mat[0, 3])
        cam_y = float(pose_mat[1, 3])
        max_d2 = self._max_vertex_distance_m ** 2
        return any(
            (vx - cam_x) ** 2 + (vy - cam_y) ** 2 > max_d2
            for vx, vy in world_xy
        )

    def _filter_by_area(self, geom: BaseGeometry) -> BaseGeometry:
        """min_area_m2 미만 sub-polygon 제거."""
        from shapely.geometry import MultiPolygon

        if hasattr(geom, "geoms"):
            filtered = [g for g in geom.geoms if g.area >= self._min_area_m2]
            return MultiPolygon(filtered)
        if geom.area < self._min_area_m2:
            return MultiPolygon()
        return geom


class FloorLayoutFromGridStep:
    """
    Fix 3: WalkableGrid mask → raster-fit FloorLayout.

    walkable_grid의 boolean mask를 contour로 변환하여 world frame MultiPolygon 생성.
    back-projection을 거치지 않으므로 horizon spike가 없고 깔끔한 boundary를 생성.
    """

    def __init__(
        self,
        contour_area_min_px2: float = 50.0,
        approx_epsilon_ratio: float = 0.005,
        min_area_m2: float = _DEFAULT_MIN_AREA_M2,
        simplify_tolerance_m: float = 0.05,
    ) -> None:
        self._contour_area_min_px2 = contour_area_min_px2
        self._approx_epsilon_ratio = approx_epsilon_ratio
        self._min_area_m2 = min_area_m2
        self._simplify_tolerance_m = simplify_tolerance_m

    def run(self, grid: WalkableGrid, z0: float) -> FloorLayout:
        """
        walkable_grid mask → cv2.findContours → approxPolyDP →
        world frame 변환 → unary_union → simplify → FloorLayout.
        """
        import cv2
        from shapely.geometry import MultiPolygon
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union

        mask_u8: np.ndarray = grid.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cs = grid.origin.cell_size
        x0 = grid.origin.x0
        y0 = grid.origin.y0

        polygons_world: list[ShapelyPolygon] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self._contour_area_min_px2:
                continue
            arc = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self._approx_epsilon_ratio * arc, True)
            if len(approx) < 3:
                continue
            world_pts = [
                (x0 + float(p[0][0]) * cs, y0 + float(p[0][1]) * cs)
                for p in approx
            ]
            sp = ShapelyPolygon(world_pts)
            if not sp.is_valid:
                sp = sp.buffer(0)
            if sp.is_valid and sp.area >= self._min_area_m2:
                polygons_world.append(sp)

        if not polygons_world:
            empty = MultiPolygon()
            return FloorLayout(
                multipolygon=empty,
                z0=z0,
                sub_polygon_count=0,
                total_area_m2=0.0,
            )

        union = unary_union(polygons_world).buffer(0)
        union = union.simplify(self._simplify_tolerance_m, preserve_topology=True)

        # area 필터 + MultiPolygon 정규화
        from shapely.geometry import Polygon as ShapelyPolygonCheck
        if isinstance(union, ShapelyPolygonCheck):
            if union.is_empty or union.area < self._min_area_m2:
                multi: object = MultiPolygon()
            else:
                multi = MultiPolygon([union])
        elif hasattr(union, "geoms"):
            filtered = [g for g in union.geoms if g.area >= self._min_area_m2]
            multi = MultiPolygon(filtered)
        else:
            multi = MultiPolygon()

        geom_obj: BaseGeometry = multi
        if hasattr(geom_obj, "geoms"):
            count = len(list(geom_obj.geoms))
        else:
            count = 0 if geom_obj.is_empty else 1

        return FloorLayout(
            multipolygon=multi,
            z0=z0,
            sub_polygon_count=count,
            total_area_m2=float(geom_obj.area),
        )

