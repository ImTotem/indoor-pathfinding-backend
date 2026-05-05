"""Sprint 46 — FloorPointCloud → WalkableGrid + footprint GeoJSON.

10cm cell raster + morphology + largest-component filter + Shapely contour.
침대/가구 위 trajectory 영역이 polygon에 포함되지 않도록 floor mask 픽셀이
실측 depth로 back-project된 cloud에서만 walkable cell을 결정한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label
from shapely.geometry import MultiPolygon, mapping
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from indoor_server.application.building.steps.floor_point_cloud import FloorPointCloud
from indoor_server.domain.building.models import GridOrigin, WalkableGrid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FloorRasterStepParams:
    cell_size_m: float = 0.10
    min_cell_hits: int = 2
    morph_open_radius_cells: int = 1
    morph_close_radius_cells: int = 2
    keep_largest_component: bool = True
    # Sprint 49 hotfix: ㄴ자/T자 환경에서 corner 부근 floor pixel 부족으로
    # 두 corridor가 disconnect → largest 만 남기면 한쪽 hall 통째 drop.
    # min_component_area_m2 > 0 + keep_largest_component=False 면
    # 면적 임계 이상 모든 component 유지.
    min_component_area_m2: float = 0.0
    bbox_padding_m: float = 0.5
    contour_min_area_m2: float = 0.20
    contour_simplify_tolerance_m: float = 0.05

    def to_metadata(self) -> dict[str, object]:
        return {
            "cell_size_m": self.cell_size_m,
            "min_cell_hits": self.min_cell_hits,
            "morph_open_radius_cells": self.morph_open_radius_cells,
            "morph_close_radius_cells": self.morph_close_radius_cells,
            "keep_largest_component": self.keep_largest_component,
            "min_component_area_m2": self.min_component_area_m2,
            "bbox_padding_m": self.bbox_padding_m,
            "contour_min_area_m2": self.contour_min_area_m2,
            "contour_simplify_tolerance_m": self.contour_simplify_tolerance_m,
        }


@dataclass(frozen=True)
class FloorRasterResult:
    grid: WalkableGrid
    footprint_geojson: dict[str, object] | None
    metadata: dict[str, object] = field(default_factory=dict)


class FloorRasterStep:
    """FloorPointCloud → 10cm WalkableGrid + footprint MultiPolygon GeoJSON."""

    def __init__(self, params: FloorRasterStepParams | None = None) -> None:
        self._params = params if params is not None else FloorRasterStepParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.cell_size_m <= 0:
            raise ValueError(f"cell_size_m must be > 0, got {p.cell_size_m}")
        if p.min_cell_hits < 1:
            raise ValueError(f"min_cell_hits must be >= 1, got {p.min_cell_hits}")
        if p.morph_open_radius_cells < 0 or p.morph_close_radius_cells < 0:
            raise ValueError("morph radius must be >= 0")
        if p.bbox_padding_m < 0:
            raise ValueError("bbox_padding_m must be >= 0")

    def run(self, cloud: FloorPointCloud) -> FloorRasterResult:
        params = self._params
        if cloud.points_xy.shape[0] == 0:
            return self._empty(cloud=cloud, reason="no_points")

        grid_arrays = _build_hit_count_grid(
            points_xy=cloud.points_xy,
            cell_size_m=params.cell_size_m,
            bbox_padding_m=params.bbox_padding_m,
        )
        if grid_arrays is None:
            return self._empty(cloud=cloud, reason="degenerate_bbox")

        x0, y0, hit_count = grid_arrays
        height, width = hit_count.shape
        mask_raw = hit_count >= params.min_cell_hits
        after_threshold = int(mask_raw.sum())

        if after_threshold == 0:
            return self._empty(
                cloud=cloud,
                reason="below_threshold",
                origin_override=(x0, y0, width, height),
                hit_count_override=hit_count,
            )

        mask_open = _binary_open(mask_raw, params.morph_open_radius_cells)
        mask_closed = _binary_close(mask_open, params.morph_close_radius_cells)
        after_morphology = int(mask_closed.sum())

        cell_area_m2 = float(params.cell_size_m) ** 2
        components_before, mask_final = _select_components(
            mask_closed,
            keep_largest=params.keep_largest_component,
            min_component_area_cells=int(
                round(params.min_component_area_m2 / cell_area_m2)
            ),
        )
        after_largest = int(mask_final.sum())

        origin = GridOrigin(
            x0=float(x0),
            y0=float(y0),
            z0=float(cloud.z0),
            cell_size=float(params.cell_size_m),
            w=int(width),
            h=int(height),
        )
        observation_count = np.clip(hit_count, 0, np.iinfo(np.uint16).max).astype(
            np.uint16
        )
        grid = WalkableGrid(
            origin=origin,
            mask=mask_final.astype(bool, copy=False),
            observation_count=observation_count,
        )

        footprint_geojson, polygon_area_m2 = _mask_to_footprint_geojson(
            mask=mask_final,
            origin=origin,
            min_area_m2=params.contour_min_area_m2,
            simplify_tolerance_m=params.contour_simplify_tolerance_m,
        )

        metadata = {
            "raw_cells": after_threshold,
            "after_threshold_cells": after_threshold,
            "after_morphology_cells": after_morphology,
            "after_largest_component_cells": after_largest,
            "components_before_keep_largest": components_before,
            "polygon_area_m2": polygon_area_m2,
            "grid_w": width,
            "grid_h": height,
            "origin": {
                "x0": float(x0),
                "y0": float(y0),
                "z0": float(cloud.z0),
                "cell_size_m": float(params.cell_size_m),
            },
            "params": params.to_metadata(),
            # Sprint 48: cleanup 진입 직전 raw raster contour 보존 (Codex W-7
            # before/after evidence 비교용). cleanup이 OFF면 그대로 final.
            "footprint_geojson": footprint_geojson,
        }
        return FloorRasterResult(
            grid=grid,
            footprint_geojson=footprint_geojson,
            metadata=metadata,
        )

    def _empty(
        self,
        *,
        cloud: FloorPointCloud,
        reason: str,
        origin_override: tuple[float, float, int, int] | None = None,
        hit_count_override: np.ndarray | None = None,
    ) -> FloorRasterResult:
        params = self._params
        if origin_override is not None:
            x0, y0, w, h = origin_override
        else:
            x0, y0, w, h = 0.0, 0.0, 1, 1
        origin = GridOrigin(
            x0=float(x0),
            y0=float(y0),
            z0=float(cloud.z0),
            cell_size=float(params.cell_size_m),
            w=int(w),
            h=int(h),
        )
        if hit_count_override is not None:
            observation_count = np.clip(
                hit_count_override, 0, np.iinfo(np.uint16).max
            ).astype(np.uint16)
        else:
            observation_count = np.zeros((h, w), dtype=np.uint16)
        grid = WalkableGrid(
            origin=origin,
            mask=np.zeros((h, w), dtype=bool),
            observation_count=observation_count,
        )
        return FloorRasterResult(
            grid=grid,
            footprint_geojson=None,
            metadata={
                "raw_cells": 0,
                "after_threshold_cells": 0,
                "after_morphology_cells": 0,
                "after_largest_component_cells": 0,
                "components_before_keep_largest": 0,
                "polygon_area_m2": 0.0,
                "grid_w": int(w),
                "grid_h": int(h),
                "origin": {
                    "x0": float(x0),
                    "y0": float(y0),
                    "z0": float(cloud.z0),
                    "cell_size_m": float(params.cell_size_m),
                },
                "empty_reason": reason,
                "params": params.to_metadata(),
            },
        )


def _build_hit_count_grid(
    *,
    points_xy: np.ndarray,
    cell_size_m: float,
    bbox_padding_m: float,
) -> tuple[float, float, np.ndarray] | None:
    xs = points_xy[:, 0]
    ys = points_xy[:, 1]
    x_min = float(np.min(xs)) - bbox_padding_m
    x_max = float(np.max(xs)) + bbox_padding_m
    y_min = float(np.min(ys)) - bbox_padding_m
    y_max = float(np.max(ys)) + bbox_padding_m
    width = int(np.ceil((x_max - x_min) / cell_size_m))
    height = int(np.ceil((y_max - y_min) / cell_size_m))
    if width < 1 or height < 1:
        return None

    cols = np.floor((xs - x_min) / cell_size_m).astype(np.int64)
    rows = np.floor((ys - y_min) / cell_size_m).astype(np.int64)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cols = cols[inside]
    rows = rows[inside]

    hit_count = np.zeros((height, width), dtype=np.int32)
    if cols.size:
        np.add.at(hit_count, (rows, cols), 1)
    return x_min, y_min, hit_count


def _binary_open(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.copy()
    out = binary_opening(mask, iterations=iterations)
    return np.asarray(out, dtype=bool)


def _binary_close(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.copy()
    out = binary_closing(mask, iterations=iterations)
    return np.asarray(out, dtype=bool)


def _select_components(
    mask: np.ndarray,
    *,
    keep_largest: bool,
    min_component_area_cells: int = 0,
) -> tuple[int, np.ndarray]:
    """component 선택.

    - `keep_largest=True` → 단일 largest 만 남김 (Sprint 46 default).
    - `keep_largest=False` AND `min_component_area_cells>0` → 면적 임계 이상
      모든 component 유지 (Sprint 49: ㄴ자/T자 corner 끊김 대응).
    - `keep_largest=False` AND `min_component_area_cells<=0` → 모든 component 유지.
    """
    if not mask.any():
        return 0, mask.copy()
    labeled_arr, count_obj = label(mask)
    labeled = np.asarray(labeled_arr, dtype=np.int64)
    components_before = int(count_obj)
    if components_before <= 1:
        return components_before, mask.copy()

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background

    if keep_largest:
        largest = int(np.argmax(sizes))
        return components_before, np.asarray(labeled == largest, dtype=bool)

    if min_component_area_cells > 0:
        kept_labels = np.where(sizes >= min_component_area_cells)[0]
        kept_labels = kept_labels[kept_labels != 0]
        if kept_labels.size == 0:
            largest = int(np.argmax(sizes))
            kept_labels = np.array([largest], dtype=np.int64)
        keep_mask = np.isin(labeled, kept_labels)
        return components_before, np.asarray(keep_mask, dtype=bool)

    return components_before, mask.copy()


def _mask_to_footprint_geojson(
    *,
    mask: np.ndarray,
    origin: GridOrigin,
    min_area_m2: float,
    simplify_tolerance_m: float,
) -> tuple[dict[str, object] | None, float]:
    """boolean mask → world-frame MultiPolygon GeoJSON."""
    import cv2

    if not mask.any():
        return None, 0.0

    mask_u8: np.ndarray = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cs = origin.cell_size
    x0 = origin.x0
    y0 = origin.y0

    polygons_world: list[ShapelyPolygon] = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        pts = [
            (x0 + float(p[0][0]) * cs, y0 + float(p[0][1]) * cs)
            for p in cnt
        ]
        sp = ShapelyPolygon(pts)
        if not sp.is_valid:
            sp = sp.buffer(0)
        if isinstance(sp, ShapelyPolygon) and sp.is_valid and sp.area >= min_area_m2:
            polygons_world.append(sp)

    if not polygons_world:
        return None, 0.0

    union = unary_union(polygons_world)
    if simplify_tolerance_m > 0:
        union = union.simplify(simplify_tolerance_m, preserve_topology=True)

    if isinstance(union, ShapelyPolygon):
        if union.is_empty or union.area < min_area_m2:
            return None, 0.0
        multi = MultiPolygon([union])
    elif hasattr(union, "geoms"):
        filtered = [g for g in union.geoms if g.area >= min_area_m2]
        if not filtered:
            return None, 0.0
        multi = MultiPolygon(filtered)
    else:
        return None, 0.0

    geojson: dict[str, object] = dict(mapping(multi))
    return geojson, float(multi.area)
