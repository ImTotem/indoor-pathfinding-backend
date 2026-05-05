"""AdaptiveBufferStep — per-keyframe Segformer floor mask → 원형 buffer union → WalkableGrid.

SuperPoint / LightGlue / Depth NN 의존 없음. ARKit pose + Segformer floor mask + 평면 z=z0
가정만으로 walkable polygon을 직접 구성한다 (Sprint 22).

알고리즘 (per-keyframe):
  1. floor_mask 의 image bottom strip_bottom_fraction 부분만 keep
  2. strip 내 floor pixel을 ray-plane (z=z0) 교차로 world (x, y) 변환
  3. forward = -R[:, 2] (xy 성분만), perp = forward 90° CCW 회전
  4. horiz_width = perp 축 projection (max - min)   — 좌우 합 = diameter
  5. vert_depth  = forward 축 양의 projection의 max — 카메라 전방 편도
  6. r_i = clamp(min(horiz_width / 2, vert_depth), min_buffer_m, max_buffer_m)
  7. floor pixel 0 / 수직 카메라 / forward intersection 없음 → keyframe skip

모든 유효 r_i 에 Point(cam_xy_i).buffer(r_i) → unary_union → rasterize.
"""
from __future__ import annotations

import logging
from statistics import median

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from indoor_server.application.building.steps.back_projection import (
    Intrinsics,
    decode_pose_matrix,
)
from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks
from indoor_server.domain.building.models import GridOrigin, WalkableGrid

logger = logging.getLogger(__name__)

_CELL_SIZE_M = 0.10  # WalkableGridStep과 동일
_Z_TOLERANCE = 0.30  # back_projection.py 와 동일
# ray가 floor로 수렴하려면 최소 5% 아래 기울기 필요. near-horizon ray (dz≈0)는
# ray-plane intersection에서 λ가 폭발(km 단위 false point) — Sprint 19의 back_projection
# `rw_z < -0.05` 와 동일 안전장치.
_MIN_DOWNWARD_DZ = 0.05


class AdaptiveBufferStep:
    """per-keyframe Segformer floor mask → 발밑 strip 측정 → 원형 buffer union → WalkableGrid.

    LOC ≤ 300 (Sprint 22 제약 §10.3).
    """

    def __init__(
        self,
        strip_bottom_fraction: float = 0.5,
        min_buffer_m: float = 0.3,
        max_buffer_m: float = 5.0,
        cell_size_m: float = _CELL_SIZE_M,
        segmenter_pre_strips: bool = False,
    ) -> None:
        """
        Args:
            strip_bottom_fraction: sensor frame 발밑 비율. 이 비율만큼 아래쪽만 측정 대상.
            min_buffer_m: r_i 하한. strip이 거의 비었을 때 false-zero 방어.
            max_buffer_m: r_i 상한. horizon 픽셀 ray-plane 발산 방어.
            cell_size_m: rasterize 해상도 (WalkableGridStep과 동일 0.10m).
            segmenter_pre_strips: True이면 mask가 이미 strip 적용 상태 → re-strip skip.
        """
        self._strip_bottom_fraction = strip_bottom_fraction
        self._min_buffer_m = min_buffer_m
        self._max_buffer_m = max_buffer_m
        self._cell_size = cell_size_m
        self._segmenter_pre_strips = segmenter_pre_strips

    async def run(
        self,
        masks: list[KeyframeMasks],
        z0: float,
        intrinsics_by_frame: list[Intrinsics],
    ) -> WalkableGrid:
        """masks: FLOOR_SEG 결과. intrinsics_by_frame: 각 keyframe 카메라 내부 파라미터."""
        points: list[tuple[float, float]] = []
        radii: list[float] = []

        # 진단 카운터
        n_skip_no_floor = 0
        n_skip_vertical = 0
        n_skip_no_fwd = 0
        n_clamp_min = 0
        n_clamp_max = 0
        r_list: list[float] = []
        horiz_list: list[float] = []
        vert_list: list[float] = []

        for m, intrin in zip(masks, intrinsics_by_frame, strict=True):
            mask = m.floor_mask
            if not self._segmenter_pre_strips:
                mask = self._apply_strip(mask, self._strip_bottom_fraction)

            ys, xs = np.where(mask)
            if xs.size == 0:
                n_skip_no_floor += 1
                continue

            world_xy, pose = self._back_project_xy(xs, ys, m.pose_matrix, intrin, z0)
            if len(world_xy) == 0:
                n_skip_no_fwd += 1
                continue

            fwd3 = -pose[:3, 2]  # camera -z = forward in world
            fwd_norm = float(np.linalg.norm(fwd3[:2]))
            if fwd_norm < 1e-6:
                n_skip_vertical += 1
                continue

            fwd_xy = fwd3[:2] / fwd_norm
            perp_xy = np.array([-fwd_xy[1], fwd_xy[0]], dtype=np.float64)  # 90° CCW

            cam_xy = pose[:2, 3]
            delta = world_xy - cam_xy  # (N, 2)

            perp_proj = delta @ perp_xy   # (N,) signed 좌우
            fwd_proj = delta @ fwd_xy     # (N,) signed 전후

            horiz_width = float(perp_proj.max() - perp_proj.min())
            fwd_pos = fwd_proj[fwd_proj > 0]
            vert_depth = float(fwd_pos.max()) if fwd_pos.size > 0 else 0.0

            r_raw = min(horiz_width / 2.0, vert_depth)

            # clamp
            if r_raw < self._min_buffer_m:
                n_clamp_min += 1
            if r_raw > self._max_buffer_m:
                n_clamp_max += 1
            r = float(np.clip(r_raw, self._min_buffer_m, self._max_buffer_m))

            points.append((float(cam_xy[0]), float(cam_xy[1])))
            radii.append(r)
            r_list.append(r)
            horiz_list.append(horiz_width)
            vert_list.append(vert_depth)

        frame_total = len(masks)
        frame_used = len(points)
        frame_skipped = frame_total - frame_used

        # 진단 로그
        if r_list:
            logger.info(
                "adaptive_buffer: frame_total=%d frame_used=%d frame_skipped=%d "
                "r_min=%.2fm r_max=%.2fm r_median=%.2fm "
                "horiz_median=%.2fm vert_median=%.2fm",
                frame_total, frame_used, frame_skipped,
                min(r_list), max(r_list), median(r_list),
                median(horiz_list), median(vert_list),
            )
        else:
            logger.warning(
                "adaptive_buffer: all %d frames skipped "
                "(no_floor=%d vertical=%d no_fwd=%d) → empty WalkableGrid",
                frame_total, n_skip_no_floor, n_skip_vertical, n_skip_no_fwd,
            )

        logger.info(
            "adaptive_buffer: clamp_min=%d clamp_max=%d skip_no_floor=%d "
            "skip_vertical=%d skip_no_fwd=%d",
            n_clamp_min, n_clamp_max,
            n_skip_no_floor, n_skip_vertical, n_skip_no_fwd,
        )

        if not points:
            return self._empty_grid(z0)

        disks: list[Polygon] = [Point(p).buffer(r) for p, r in zip(points, radii, strict=True)]
        union = unary_union(disks)
        polygons: list[Polygon]
        if isinstance(union, Polygon):
            polygons = [union]
        elif isinstance(union, MultiPolygon):
            polygons = list(union.geoms)
        else:
            return self._empty_grid(z0)

        union_polygon_count = len(polygons)
        logger.info("adaptive_buffer: union_polygons=%d", union_polygon_count)

        grid = self._rasterize(polygons, points, radii, z0)

        logger.info(
            "adaptive_buffer: walkable_cells=%d origin=(%.2f,%.2f) grid=(%dx%d)",
            int(grid.mask.sum()),
            grid.origin.x0, grid.origin.y0,
            grid.origin.w, grid.origin.h,
        )
        return grid

    async def run_with_footprint(
        self,
        masks: list[KeyframeMasks],
        z0: float,
        intrinsics_by_frame: list[Intrinsics],
    ) -> tuple[WalkableGrid, dict[str, object] | None]:
        """run()과 동일하나 shapely union의 GeoJSON도 함께 반환.

        Returns:
            (WalkableGrid, footprint_geojson) — footprint_geojson은 MultiPolygon GeoJSON dict.
            포인트가 없어 빈 grid를 반환하는 경우 footprint_geojson은 None.
        """
        grid = await self.run(masks=masks, z0=z0, intrinsics_by_frame=intrinsics_by_frame)
        if int(grid.mask.sum()) == 0:
            return grid, None

        # grid origin/mask에서 shapely polygon 재구성: rasterize 역변환 대신
        # run() 로직을 별도 내부 메서드로 분리해 재호출 비용 없이 footprint 추출
        footprint_geojson: dict[str, object] | None = await self._extract_footprint_geojson(
            masks=masks, z0=z0, intrinsics_by_frame=intrinsics_by_frame
        )
        return grid, footprint_geojson

    async def _extract_footprint_geojson(
        self,
        masks: list[KeyframeMasks],
        z0: float,
        intrinsics_by_frame: list[Intrinsics],
    ) -> dict[str, object] | None:
        """run() 과 동일한 union polygon 계산 후 GeoJSON MultiPolygon으로 직렬화."""
        from shapely.geometry import mapping

        points: list[tuple[float, float]] = []
        radii: list[float] = []

        for m, intrin in zip(masks, intrinsics_by_frame, strict=True):
            mask = m.floor_mask
            if not self._segmenter_pre_strips:
                mask = self._apply_strip(mask, self._strip_bottom_fraction)
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            world_xy, pose = self._back_project_xy(xs, ys, m.pose_matrix, intrin, z0)
            if len(world_xy) == 0:
                continue
            fwd3 = -pose[:3, 2]
            fwd_norm = float(np.linalg.norm(fwd3[:2]))
            if fwd_norm < 1e-6:
                continue
            fwd_xy = fwd3[:2] / fwd_norm
            perp_xy = np.array([-fwd_xy[1], fwd_xy[0]], dtype=np.float64)
            cam_xy = pose[:2, 3]
            delta = world_xy - cam_xy
            perp_proj = delta @ perp_xy
            fwd_proj = delta @ fwd_xy
            horiz_width = float(perp_proj.max() - perp_proj.min())
            fwd_pos = fwd_proj[fwd_proj > 0]
            vert_depth = float(fwd_pos.max()) if fwd_pos.size > 0 else 0.0
            r_raw = min(horiz_width / 2.0, vert_depth)
            r = float(np.clip(r_raw, self._min_buffer_m, self._max_buffer_m))
            points.append((float(cam_xy[0]), float(cam_xy[1])))
            radii.append(r)

        if not points:
            return None

        disks: list[Polygon] = [Point(p).buffer(r) for p, r in zip(points, radii, strict=True)]
        union = unary_union(disks)

        # 항상 MultiPolygon 형태로 직렬화 (IMDF unit/footprint 스키마 일관성)
        if isinstance(union, Polygon):
            multi = MultiPolygon([union])
        elif isinstance(union, MultiPolygon):
            multi = union
        else:
            return None

        geojson: dict[str, object] = dict(mapping(multi))
        return geojson

    # ── private helpers ────────────────────────────────────────────────────────

    def _apply_strip(self, mask: np.ndarray, fraction: float) -> np.ndarray:
        """sensor frame 하단 fraction 비율만 남기고 위는 False."""
        h, _ = mask.shape
        keep_h = int(h * fraction)
        if keep_h == 0:
            return np.zeros_like(mask, dtype=bool)
        out = mask.copy()
        out[: h - keep_h, :] = False
        return out

    def _back_project_xy(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        pose_bytes: bytes,
        intrin: Intrinsics,
        z0: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """strip 내 floor pixel → (world (x, y) 배열, pose 행렬) 반환.

        world_xy: (N, 2) — 유효 ray만 포함.
        pose: (4, 4) ndarray — 호출 측에서 재사용 (decode 이중 호출 방지).
        """
        pose = decode_pose_matrix(pose_bytes)
        rotation = pose[:3, :3]
        t = pose[:3, 3]

        # ARKit 카메라 frame: x-right, y-up, -z forward. 이미지 좌표는 y-down.
        # Sprint 20에서 확정된 convention: D = diag(1, -1, -1) 를 ray_cam에 적용.
        rx = (xs - intrin.cx) / intrin.fx
        ry = -(ys - intrin.cy) / intrin.fy
        rz = -np.ones(len(xs), dtype=np.float64)
        rays_cam = np.stack([rx, ry, rz], axis=1)   # (N, 3)

        rays_world = rays_cam @ rotation.T            # (N, 3)
        dz = rays_world[:, 2]

        # 아래쪽 ray만 유효 (near-horizon/upward ray 제외). dz < -_MIN_DOWNWARD_DZ.
        downward = dz < -_MIN_DOWNWARD_DZ
        if downward.sum() == 0:
            return np.zeros((0, 2), dtype=np.float64), pose

        # camera 위치 t[2] > z0 이고 dz < 0 이면 lam > 0 자동 보장
        lam = (z0 - t[2]) / dz[downward]
        rays_f = rays_world[downward]
        points = t + lam[:, None] * rays_f            # (N, 3)

        # z-tolerance 필터 (back_projection.py 와 동일)
        z_ok = np.abs(points[:, 2] - z0) < _Z_TOLERANCE
        pts_ok: np.ndarray = points[z_ok, :2]
        return pts_ok, pose

    def _rasterize(
        self,
        polygons: list[Polygon],
        cam_points: list[tuple[float, float]],
        radii: list[float],
        z0: float,
    ) -> WalkableGrid:
        """world 좌표 polygon list → grid → bool mask."""
        import cv2

        cs = self._cell_size
        max_r = max(radii)
        all_xs = [p[0] for p in cam_points]
        all_ys = [p[1] for p in cam_points]

        x_min = min(all_xs) - max_r - cs
        x_max = max(all_xs) + max_r + cs
        y_min = min(all_ys) - max_r - cs
        y_max = max(all_ys) + max_r + cs

        w = max(1, int(np.ceil((x_max - x_min) / cs)) + 2)
        h = max(1, int(np.ceil((y_max - y_min) / cs)) + 2)

        origin = GridOrigin(x0=x_min, y0=y_min, z0=z0, cell_size=cs, w=w, h=h)

        buf = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            coords = list(poly.exterior.coords)
            pixel_coords = np.array(
                [
                    [int((x - x_min) / cs), int((y - y_min) / cs)]
                    for x, y in coords
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(buf, [pixel_coords], color=1)

        mask = buf.astype(bool)
        obs = np.zeros((h, w), dtype=np.uint16)

        return WalkableGrid(origin=origin, mask=mask, observation_count=obs)

    def _empty_grid(self, z0: float) -> WalkableGrid:
        origin = GridOrigin(x0=0.0, y0=0.0, z0=z0, cell_size=self._cell_size, w=1, h=1)
        return WalkableGrid(
            origin=origin,
            mask=np.zeros((1, 1), dtype=bool),
            observation_count=np.zeros((1, 1), dtype=np.uint16),
        )
