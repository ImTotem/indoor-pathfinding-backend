"""Wall heatmap + trajectory seed -> walkable interior polygon.

This step treats wall segmentation evidence as *barriers*, not as the polygon
itself.  The interior is selected by flood-filling the support region from
RTABMap trajectory seeds while preventing leakage through high-confidence wall
cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from indoor_server.application.building.steps.manhattan_rectification import (
    ManhattanRectificationStep,
)
from indoor_server.application.building.steps.wall_polygon.heatmap_boundary import (
    _bridge_axis_aligned_gaps,
    _disk_kernel,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)
from indoor_server.domain.building.models import WalkableGrid


@dataclass(frozen=True)
class WallInteriorFillParams:
    """Tunable parameters for wall-constrained interior extraction."""

    wall_min_cell_hits: int = 4
    wall_bridge_gap_radius_cells: int = 2
    wall_close_radius_cells: int = 1
    wall_dilate_radius_cells: int = 2
    support_dilate_radius_cells: int = 8
    seed_radius_cells: int = 4
    interior_close_radius_cells: int = 3
    interior_open_radius_cells: int = 0
    min_component_area_m2: float = 1.0
    simplify_tolerance_m: float = 0.05
    rectilinear_enabled: bool = True
    rectilinear_area_change_limit: float = 0.65
    rectilinear_simplify_tolerance_m: float = 0.25
    wall_boundary_distance_m: float = 0.40
    seed_max_wall_distance_m: float = 3.0
    centerline_fallback_enabled: bool = True
    centerline_half_width_min_m: float = 0.45
    centerline_half_width_max_m: float = 1.50

    def to_metadata(self) -> dict[str, object]:
        return {
            "wall_min_cell_hits": self.wall_min_cell_hits,
            "wall_bridge_gap_radius_cells": self.wall_bridge_gap_radius_cells,
            "wall_close_radius_cells": self.wall_close_radius_cells,
            "wall_dilate_radius_cells": self.wall_dilate_radius_cells,
            "support_dilate_radius_cells": self.support_dilate_radius_cells,
            "seed_radius_cells": self.seed_radius_cells,
            "interior_close_radius_cells": self.interior_close_radius_cells,
            "interior_open_radius_cells": self.interior_open_radius_cells,
            "min_component_area_m2": self.min_component_area_m2,
            "simplify_tolerance_m": self.simplify_tolerance_m,
            "rectilinear_enabled": self.rectilinear_enabled,
            "rectilinear_area_change_limit": self.rectilinear_area_change_limit,
            "rectilinear_simplify_tolerance_m": self.rectilinear_simplify_tolerance_m,
            "wall_boundary_distance_m": self.wall_boundary_distance_m,
            "seed_max_wall_distance_m": self.seed_max_wall_distance_m,
            "centerline_fallback_enabled": self.centerline_fallback_enabled,
            "centerline_half_width_min_m": self.centerline_half_width_min_m,
            "centerline_half_width_max_m": self.centerline_half_width_max_m,
        }


@dataclass(frozen=True)
class WallInteriorFillResult:
    accepted: bool
    interior_geojson: dict[str, object] | None
    raw_interior_geojson: dict[str, object] | None
    interior_mask: np.ndarray
    wall_mask: np.ndarray
    barrier_mask: np.ndarray
    support_mask: np.ndarray
    seed_mask: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)
    fail_reason: str | None = None


class WallInteriorFillStep:
    """Generate a walkable interior polygon from wall heatmap barriers."""

    def __init__(self, params: WallInteriorFillParams | None = None) -> None:
        self._params = params if params is not None else WallInteriorFillParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.wall_min_cell_hits < 1:
            raise ValueError("wall_min_cell_hits must be >= 1")
        for key, value in (
            ("wall_bridge_gap_radius_cells", p.wall_bridge_gap_radius_cells),
            ("wall_close_radius_cells", p.wall_close_radius_cells),
            ("wall_dilate_radius_cells", p.wall_dilate_radius_cells),
            ("support_dilate_radius_cells", p.support_dilate_radius_cells),
            ("seed_radius_cells", p.seed_radius_cells),
            ("interior_close_radius_cells", p.interior_close_radius_cells),
            ("interior_open_radius_cells", p.interior_open_radius_cells),
        ):
            if value < 0:
                raise ValueError(f"{key} must be >= 0")
        if p.min_component_area_m2 < 0:
            raise ValueError("min_component_area_m2 must be >= 0")
        if p.simplify_tolerance_m < 0 or p.rectilinear_simplify_tolerance_m < 0:
            raise ValueError("simplify tolerances must be >= 0")
        if not (0.0 <= p.rectilinear_area_change_limit <= 1.0):
            raise ValueError("rectilinear_area_change_limit must be in [0, 1]")
        if p.wall_boundary_distance_m < 0:
            raise ValueError("wall_boundary_distance_m must be >= 0")
        if p.seed_max_wall_distance_m < 0:
            raise ValueError("seed_max_wall_distance_m must be >= 0")
        if p.centerline_half_width_min_m <= 0:
            raise ValueError("centerline_half_width_min_m must be > 0")
        if p.centerline_half_width_max_m < p.centerline_half_width_min_m:
            raise ValueError(
                "centerline_half_width_max_m must be >= centerline_half_width_min_m"
            )

    def run(
        self,
        *,
        heatmap: ObstacleHeatmap,
        seed_points_xy: list[tuple[float, float]],
        support_grid: WalkableGrid | None = None,
        dominant_angle_hint_deg: float | None = None,
    ) -> WallInteriorFillResult:
        params = self._params
        counts = np.asarray(heatmap.counts, dtype=np.int32)
        empty = np.zeros_like(counts, dtype=bool)
        if counts.size == 0 or int(np.count_nonzero(counts)) == 0:
            return self._failure(
                "no_wall_heatmap",
                heatmap=heatmap,
                wall_mask=empty,
                barrier_mask=empty,
                support_mask=empty,
                seed_mask=empty,
            )
        if len(seed_points_xy) < 2:
            return self._failure(
                "not_enough_seed_points",
                heatmap=heatmap,
                wall_mask=empty,
                barrier_mask=empty,
                support_mask=empty,
                seed_mask=empty,
            )

        wall_mask = counts >= params.wall_min_cell_hits
        threshold_wall_cells = int(wall_mask.sum())
        wall_mask = _bridge_axis_aligned_gaps(
            wall_mask,
            params.wall_bridge_gap_radius_cells,
        )
        if params.wall_close_radius_cells > 0:
            wall_mask = ndimage.binary_closing(
                wall_mask,
                structure=_disk_kernel(params.wall_close_radius_cells),
            )
        wall_mask = np.asarray(wall_mask, dtype=bool)
        barrier_mask = wall_mask
        if params.wall_dilate_radius_cells > 0:
            barrier_mask = ndimage.binary_dilation(
                wall_mask,
                structure=_disk_kernel(params.wall_dilate_radius_cells),
            )
        barrier_mask = np.asarray(barrier_mask, dtype=bool)

        support_mask = (
            _resample_support_mask(support_grid, heatmap)
            if support_grid is not None
            else np.ones_like(wall_mask, dtype=bool)
        )
        if params.support_dilate_radius_cells > 0:
            support_mask = ndimage.binary_dilation(
                support_mask,
                structure=_disk_kernel(params.support_dilate_radius_cells),
            )
        support_mask = np.asarray(support_mask, dtype=bool)

        filtered_seed_points_xy = _filter_seed_points(
            seed_points_xy,
            heatmap=heatmap,
            wall_mask=wall_mask,
            support_mask=support_mask,
            max_wall_distance_m=params.seed_max_wall_distance_m,
        )
        if len(filtered_seed_points_xy) < 2:
            return self._failure(
                "not_enough_contextual_seed_points",
                heatmap=heatmap,
                wall_mask=wall_mask,
                barrier_mask=barrier_mask,
                support_mask=support_mask,
                seed_mask=np.zeros_like(wall_mask, dtype=bool),
                extra={
                    "seed_point_count": len(seed_points_xy),
                    "filtered_seed_point_count": len(filtered_seed_points_xy),
                    "threshold_wall_cells": threshold_wall_cells,
                },
            )

        seed_mask = _rasterize_seed_polyline(
            filtered_seed_points_xy,
            heatmap=heatmap,
            radius_cells=params.seed_radius_cells,
        )
        allowed = (support_mask | seed_mask) & ~barrier_mask
        seed_in_allowed = seed_mask & allowed
        if not seed_in_allowed.any():
            return self._failure(
                "seed_outside_allowed_region",
                heatmap=heatmap,
                wall_mask=wall_mask,
                barrier_mask=barrier_mask,
                support_mask=support_mask,
                seed_mask=seed_mask,
                extra={
                    "threshold_wall_cells": threshold_wall_cells,
                    "wall_cells": int(wall_mask.sum()),
                    "support_cells": int(support_mask.sum()),
                    "seed_cells": int(seed_mask.sum()),
                },
            )

        interior_mask, component_count, seeded_component_count = _seeded_components(
            allowed=allowed,
            seed_mask=seed_in_allowed,
            min_area_cells=_area_to_cells(
                params.min_component_area_m2,
                heatmap.cell_size_m,
            ),
        )
        if params.interior_close_radius_cells > 0:
            interior_mask = ndimage.binary_closing(
                interior_mask,
                structure=_disk_kernel(params.interior_close_radius_cells),
            )
            interior_mask &= allowed
        if params.interior_open_radius_cells > 0:
            interior_mask = ndimage.binary_opening(
                interior_mask,
                structure=_disk_kernel(params.interior_open_radius_cells),
            )
        interior_mask = np.asarray(interior_mask, dtype=bool)
        if not interior_mask.any():
            return self._failure(
                "no_seeded_interior",
                heatmap=heatmap,
                wall_mask=wall_mask,
                barrier_mask=barrier_mask,
                support_mask=support_mask,
                seed_mask=seed_mask,
                extra={
                    "component_count": component_count,
                    "seeded_component_count": seeded_component_count,
                },
            )

        raw_geojson, raw_area_m2, raw_vertex_count = _mask_to_geojson(
            mask=interior_mask,
            heatmap=heatmap,
            min_area_m2=params.min_component_area_m2,
            simplify_tolerance_m=params.simplify_tolerance_m,
        )
        if raw_geojson is None:
            return self._failure(
                "raw_polygon_empty",
                heatmap=heatmap,
                wall_mask=wall_mask,
                barrier_mask=barrier_mask,
                support_mask=support_mask,
                seed_mask=seed_mask,
            )

        raw_geom = _shape_or_empty(raw_geojson)
        raw_metric_meta = _quality_metrics(
            final_geom=raw_geom,
            heatmap=heatmap,
            wall_mask=wall_mask,
            support_mask=support_mask,
            seed_mask=seed_mask,
            seed_points_xy=filtered_seed_points_xy,
            wall_boundary_distance_m=params.wall_boundary_distance_m,
        )
        raw_quality_pass = _candidate_quality_pass(
            geom=raw_geom,
            min_area_m2=params.min_component_area_m2,
            metric_meta=raw_metric_meta,
        )

        final_geojson = raw_geojson
        final_source = "raw_interior"
        rect_meta: dict[str, object] = {"enabled": False}
        if params.rectilinear_enabled:
            rectified = ManhattanRectificationStep().run(
                raw_geojson,
                area_change_limit_ratio=params.rectilinear_area_change_limit,
                manhattan_simplify_tolerance_m=params.rectilinear_simplify_tolerance_m,
                dominant_angle_hint_deg=dominant_angle_hint_deg,
                dominant_angle_hint_source=(
                    "rtabmap_link_segments"
                    if dominant_angle_hint_deg is not None
                    else None
                ),
            )
            rect_meta = rectified.metadata()
            if rectified.accepted:
                rect_geom = _shape_or_empty(rectified.rectified_geojson)
                rect_metric_meta = _quality_metrics(
                    final_geom=rect_geom,
                    heatmap=heatmap,
                    wall_mask=wall_mask,
                    support_mask=support_mask,
                    seed_mask=seed_mask,
                    seed_points_xy=filtered_seed_points_xy,
                    wall_boundary_distance_m=params.wall_boundary_distance_m,
                )
                rect_quality_pass = _candidate_quality_pass(
                    geom=rect_geom,
                    min_area_m2=params.min_component_area_m2,
                    metric_meta=rect_metric_meta,
                )
                rect_meta = {
                    **rect_meta,
                    "candidate_quality_pass": rect_quality_pass,
                    "candidate_seed_point_inside_ratio": rect_metric_meta[
                        "seed_point_inside_ratio"
                    ],
                    "candidate_seed_cell_inside_ratio": rect_metric_meta[
                        "seed_cell_inside_ratio"
                    ],
                    "candidate_wall_inside_ratio": rect_metric_meta[
                        "wall_inside_ratio"
                    ],
                }
                if rect_quality_pass:
                    final_geojson = rectified.rectified_geojson
                    final_source = "rectified_interior"
                else:
                    rect_meta = {
                        **rect_meta,
                        "fallback_to_raw_reason": "rectified_candidate_quality_failed",
                    }
        centerline_meta: dict[str, object] = {"enabled": False}
        if (
            params.centerline_fallback_enabled
            and final_source == "raw_interior"
            and not raw_quality_pass
        ):
            centerline_candidate = _centerline_buffer_candidate(
                filtered_seed_points_xy,
                heatmap=heatmap,
                wall_mask=wall_mask,
                min_half_width_m=params.centerline_half_width_min_m,
                max_half_width_m=params.centerline_half_width_max_m,
            )
            centerline_geojson, centerline_meta = centerline_candidate
            if centerline_geojson is not None:
                centerline_geom = _shape_or_empty(centerline_geojson)
                centerline_metric_meta = _quality_metrics(
                    final_geom=centerline_geom,
                    heatmap=heatmap,
                    wall_mask=wall_mask,
                    support_mask=support_mask,
                    seed_mask=seed_mask,
                    seed_points_xy=filtered_seed_points_xy,
                    wall_boundary_distance_m=params.wall_boundary_distance_m,
                )
                centerline_quality_pass = _candidate_quality_pass(
                    geom=centerline_geom,
                    min_area_m2=params.min_component_area_m2,
                    metric_meta=centerline_metric_meta,
                )
                centerline_meta = {
                    **centerline_meta,
                    "candidate_quality_pass": centerline_quality_pass,
                    "candidate_seed_point_inside_ratio": centerline_metric_meta[
                        "seed_point_inside_ratio"
                    ],
                    "candidate_seed_cell_inside_ratio": centerline_metric_meta[
                        "seed_cell_inside_ratio"
                    ],
                    "candidate_wall_inside_ratio": centerline_metric_meta[
                        "wall_inside_ratio"
                    ],
                }
                if centerline_quality_pass:
                    final_geojson = centerline_geojson
                    final_source = "wall_guided_centerline_buffer"

        final_geom = _shape_or_empty(final_geojson)
        final_area_m2 = float(final_geom.area) if not final_geom.is_empty else 0.0
        final_vertex_count = _count_vertices(final_geom)
        metric_meta = _quality_metrics(
            final_geom=final_geom,
            heatmap=heatmap,
            wall_mask=wall_mask,
            support_mask=support_mask,
            seed_mask=seed_mask,
            seed_points_xy=filtered_seed_points_xy,
            wall_boundary_distance_m=params.wall_boundary_distance_m,
        )
        accepted = _candidate_quality_pass(
            geom=final_geom,
            min_area_m2=params.min_component_area_m2,
            metric_meta=metric_meta,
        )
        fail_reason = None if accepted else "quality_gate_failed"

        metadata: dict[str, object] = {
            "params": params.to_metadata(),
            "threshold_wall_cells": threshold_wall_cells,
            "wall_cells": int(wall_mask.sum()),
            "barrier_cells": int(barrier_mask.sum()),
            "support_cells": int(support_mask.sum()),
            "seed_cells": int(seed_mask.sum()),
            "seed_point_count": len(seed_points_xy),
            "filtered_seed_point_count": len(filtered_seed_points_xy),
            "component_count": component_count,
            "seeded_component_count": seeded_component_count,
            "interior_cells": int(interior_mask.sum()),
            "raw_area_m2": raw_area_m2,
            "raw_vertex_count": raw_vertex_count,
            "raw_quality_pass": raw_quality_pass,
            "raw_seed_point_inside_ratio": raw_metric_meta[
                "seed_point_inside_ratio"
            ],
            "raw_seed_cell_inside_ratio": raw_metric_meta["seed_cell_inside_ratio"],
            "raw_wall_inside_ratio": raw_metric_meta["wall_inside_ratio"],
            "final_source": final_source,
            "final_area_m2": final_area_m2,
            "final_vertex_count": final_vertex_count,
            "rectification": rect_meta,
            "centerline_fallback": centerline_meta,
            **metric_meta,
        }
        return WallInteriorFillResult(
            accepted=accepted,
            interior_geojson=final_geojson,
            raw_interior_geojson=raw_geojson,
            interior_mask=interior_mask,
            wall_mask=wall_mask,
            barrier_mask=barrier_mask,
            support_mask=support_mask,
            seed_mask=seed_mask,
            metadata=metadata,
            fail_reason=fail_reason,
        )

    def _failure(
        self,
        fail_reason: str,
        *,
        heatmap: ObstacleHeatmap,
        wall_mask: np.ndarray,
        barrier_mask: np.ndarray,
        support_mask: np.ndarray,
        seed_mask: np.ndarray,
        extra: dict[str, object] | None = None,
    ) -> WallInteriorFillResult:
        metadata: dict[str, object] = {
            "params": self._params.to_metadata(),
            "grid_shape": list(heatmap.shape),
            **(extra or {}),
        }
        return WallInteriorFillResult(
            accepted=False,
            interior_geojson=None,
            raw_interior_geojson=None,
            interior_mask=np.zeros_like(wall_mask, dtype=bool),
            wall_mask=np.asarray(wall_mask, dtype=bool),
            barrier_mask=np.asarray(barrier_mask, dtype=bool),
            support_mask=np.asarray(support_mask, dtype=bool),
            seed_mask=np.asarray(seed_mask, dtype=bool),
            metadata=metadata,
            fail_reason=fail_reason,
        )


def _area_to_cells(area_m2: float, cell_size_m: float) -> int:
    if area_m2 <= 0:
        return 1
    return max(1, int(round(area_m2 / (cell_size_m * cell_size_m))))


def _resample_support_mask(
    support_grid: WalkableGrid,
    heatmap: ObstacleHeatmap,
) -> np.ndarray:
    rows, cols = np.indices(heatmap.shape, dtype=np.float64)
    world_x = heatmap.origin_x + (cols + 0.5) * heatmap.cell_size_m
    world_y = heatmap.origin_y + (rows + 0.5) * heatmap.cell_size_m
    src_cols = np.floor(
        (world_x - support_grid.origin.x0) / support_grid.origin.cell_size
    ).astype(np.int64)
    src_rows = np.floor(
        (world_y - support_grid.origin.y0) / support_grid.origin.cell_size
    ).astype(np.int64)
    inside = (
        (src_rows >= 0)
        & (src_rows < support_grid.origin.h)
        & (src_cols >= 0)
        & (src_cols < support_grid.origin.w)
    )
    out = np.zeros(heatmap.shape, dtype=bool)
    out[inside] = support_grid.mask[src_rows[inside], src_cols[inside]]
    return out


def _filter_seed_points(
    seed_points_xy: list[tuple[float, float]],
    *,
    heatmap: ObstacleHeatmap,
    wall_mask: np.ndarray,
    support_mask: np.ndarray,
    max_wall_distance_m: float,
) -> list[tuple[float, float]]:
    if len(seed_points_xy) < 2:
        return []
    wall_distance_m = ndimage.distance_transform_edt(~wall_mask) * heatmap.cell_size_m
    filtered: list[tuple[float, float]] = []
    last: tuple[float, float] | None = None
    for x_raw, y_raw in seed_points_xy:
        x = float(x_raw)
        y = float(y_raw)
        row, col = _world_to_cell(heatmap, x, y)
        if row is None or col is None:
            continue
        in_support = bool(support_mask[row, col])
        near_wall = float(wall_distance_m[row, col]) <= max_wall_distance_m
        if not in_support and not near_wall:
            continue
        point = (x, y)
        if last is None or ((last[0] - x) ** 2 + (last[1] - y) ** 2) ** 0.5 > 0.05:
            filtered.append(point)
            last = point
    return filtered


def _world_to_cell(
    heatmap: ObstacleHeatmap,
    x: float,
    y: float,
) -> tuple[int | None, int | None]:
    col = int(round((float(x) - heatmap.origin_x) / heatmap.cell_size_m - 0.5))
    row = int(round((float(y) - heatmap.origin_y) / heatmap.cell_size_m - 0.5))
    if row < 0 or col < 0 or row >= heatmap.shape[0] or col >= heatmap.shape[1]:
        return None, None
    return row, col


def _rasterize_seed_polyline(
    seed_points_xy: list[tuple[float, float]],
    *,
    heatmap: ObstacleHeatmap,
    radius_cells: int,
) -> np.ndarray:
    mask = np.zeros(heatmap.shape, dtype=np.uint8)
    points: list[tuple[int, int]] = []
    for x, y in seed_points_xy:
        col = int(round((float(x) - heatmap.origin_x) / heatmap.cell_size_m - 0.5))
        row = int(round((float(y) - heatmap.origin_y) / heatmap.cell_size_m - 0.5))
        points.append((col, row))
    for (c1, r1), (c2, r2) in zip(points[:-1], points[1:], strict=True):
        cv2.line(
            mask,
            (c1, r1),
            (c2, r2),
            color=1,
            thickness=max(1, 2 * radius_cells + 1),
            lineType=cv2.LINE_8,
        )
    for col, row in points:
        cv2.circle(mask, (col, row), max(1, radius_cells), color=1, thickness=-1)
    return mask.astype(bool)


def _centerline_buffer_candidate(
    seed_points_xy: list[tuple[float, float]],
    *,
    heatmap: ObstacleHeatmap,
    wall_mask: np.ndarray,
    min_half_width_m: float,
    max_half_width_m: float,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    meta: dict[str, object] = {
        "enabled": True,
        "seed_point_count": len(seed_points_xy),
    }
    if len(seed_points_xy) < 2:
        return None, {**meta, "fail_reason": "not_enough_seed_points"}

    half_width = _estimate_centerline_half_width(
        seed_points_xy,
        heatmap=heatmap,
        wall_mask=wall_mask,
        min_half_width_m=min_half_width_m,
        max_half_width_m=max_half_width_m,
    )
    line = LineString(seed_points_xy)
    if line.is_empty or line.length <= 0:
        return None, {**meta, "fail_reason": "empty_linestring"}
    geom = line.buffer(
        half_width,
        cap_style="square",
        join_style="mitre",
        mitre_limit=2.0,
    )
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty or geom.area <= 0:
        return None, {**meta, "fail_reason": "empty_buffer"}
    return dict(mapping(geom)), {
        **meta,
        "half_width_m": half_width,
        "area_m2": float(geom.area),
        "vertex_count": _count_vertices(geom),
    }


def _estimate_centerline_half_width(
    seed_points_xy: list[tuple[float, float]],
    *,
    heatmap: ObstacleHeatmap,
    wall_mask: np.ndarray,
    min_half_width_m: float,
    max_half_width_m: float,
) -> float:
    wall_distance_m = ndimage.distance_transform_edt(~wall_mask) * heatmap.cell_size_m
    distances: list[float] = []
    for x, y in seed_points_xy:
        row, col = _world_to_cell(heatmap, x, y)
        if row is None or col is None:
            continue
        d = float(wall_distance_m[row, col])
        if np.isfinite(d) and d > 0.05:
            distances.append(d)
    if not distances:
        return min_half_width_m
    # Slightly shrink the wall distance so the display polygon stays inside
    # observed walls rather than swallowing them.
    estimated = float(np.median(distances)) * 0.85
    return float(np.clip(estimated, min_half_width_m, max_half_width_m))


def _seeded_components(
    *,
    allowed: np.ndarray,
    seed_mask: np.ndarray,
    min_area_cells: int,
) -> tuple[np.ndarray, int, int]:
    labeled, count_obj = ndimage.label(allowed)
    component_count = int(count_obj)
    if component_count == 0:
        return np.zeros_like(allowed, dtype=bool), 0, 0
    labels = np.unique(labeled[seed_mask])
    labels = labels[labels != 0]
    if labels.size == 0:
        return np.zeros_like(allowed, dtype=bool), component_count, 0
    sizes = np.bincount(labeled.ravel())
    kept = [
        int(label)
        for label in labels
        if int(label) < len(sizes) and int(sizes[int(label)]) >= min_area_cells
    ]
    if not kept:
        return np.zeros_like(allowed, dtype=bool), component_count, 0
    mask = np.isin(labeled, np.asarray(kept, dtype=np.int64))
    return np.asarray(mask, dtype=bool), component_count, len(kept)


def _mask_to_geojson(
    *,
    mask: np.ndarray,
    heatmap: ObstacleHeatmap,
    min_area_m2: float,
    simplify_tolerance_m: float,
) -> tuple[dict[str, object] | None, float, int]:
    if not mask.any():
        return None, 0.0, 0

    polygons: list[Polygon] = []
    cell = heatmap.cell_size_m
    rows, cols = np.where(mask)
    for row, col in zip(rows, cols, strict=True):
        x_min = heatmap.origin_x + float(col) * cell
        y_min = heatmap.origin_y + float(row) * cell
        polygons.append(box(x_min, y_min, x_min + cell, y_min + cell))
    if not polygons:
        return None, 0.0, 0
    union = unary_union(polygons)
    if simplify_tolerance_m > 0:
        union = union.simplify(simplify_tolerance_m, preserve_topology=True)
    if isinstance(union, Polygon):
        if union.is_empty or union.area < min_area_m2:
            return None, 0.0, 0
        multi = MultiPolygon([union])
    elif hasattr(union, "geoms"):
        filtered = [
            geom
            for geom in union.geoms
            if isinstance(geom, Polygon) and geom.area >= min_area_m2
        ]
        if not filtered:
            return None, 0.0, 0
        multi = MultiPolygon(filtered)
    else:
        return None, 0.0, 0
    return dict(mapping(multi)), float(multi.area), _count_vertices(multi)


def _shape_or_empty(geojson: dict[str, object] | None) -> BaseGeometry:
    if geojson is None:
        return Polygon()
    geom = shape(geojson)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def _meta_float(meta: dict[str, object], key: str, default: float = 0.0) -> float:
    raw = meta.get(key, default)
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, int | float):
        return float(raw)
    return default


def _candidate_quality_pass(
    *,
    geom: BaseGeometry,
    min_area_m2: float,
    metric_meta: dict[str, object],
) -> bool:
    return (
        not geom.is_empty
        and float(geom.area) >= min_area_m2
        and (
            _meta_float(metric_meta, "seed_point_inside_ratio") >= 0.80
            or _meta_float(metric_meta, "seed_cell_inside_ratio") >= 0.80
        )
        and _meta_float(metric_meta, "wall_inside_ratio") <= 0.35
    )


def _count_vertices(geom: BaseGeometry) -> int:
    if isinstance(geom, Polygon):
        return max(0, len(geom.exterior.coords) - 1)
    if isinstance(geom, MultiPolygon):
        return sum(max(0, len(poly.exterior.coords) - 1) for poly in geom.geoms)
    if hasattr(geom, "geoms"):
        return sum(_count_vertices(g) for g in geom.geoms)
    return 0


def _quality_metrics(
    *,
    final_geom: BaseGeometry,
    heatmap: ObstacleHeatmap,
    wall_mask: np.ndarray,
    support_mask: np.ndarray,
    seed_mask: np.ndarray,
    seed_points_xy: list[tuple[float, float]],
    wall_boundary_distance_m: float,
) -> dict[str, object]:
    if final_geom.is_empty:
        return {
            "seed_point_inside_ratio": 0.0,
            "seed_cell_inside_ratio": 0.0,
            "support_inside_ratio": 0.0,
            "wall_inside_ratio": 1.0,
            "wall_near_boundary_ratio": 0.0,
        }
    seed_inside = [
        final_geom.covers(Point(float(x), float(y)))
        for x, y in seed_points_xy
    ]
    seed_cells = _mask_cell_centers(seed_mask, heatmap, max_samples=5000)
    support_points = _mask_cell_centers(support_mask, heatmap, max_samples=5000)
    wall_points = _mask_cell_centers(wall_mask, heatmap, max_samples=5000)
    seed_cell_inside = _point_inside_ratio(final_geom, seed_cells)
    support_inside = _point_inside_ratio(final_geom, support_points)
    wall_inside = _point_inside_ratio(final_geom, wall_points)
    wall_near_boundary = _wall_near_boundary_ratio(
        final_geom,
        wall_points,
        max_distance_m=wall_boundary_distance_m,
    )
    return {
        "seed_point_inside_ratio": (
            float(sum(seed_inside)) / float(max(1, len(seed_inside)))
        ),
        "seed_cell_inside_ratio": seed_cell_inside,
        "support_inside_ratio": support_inside,
        "wall_inside_ratio": wall_inside,
        "wall_near_boundary_ratio": wall_near_boundary,
    }


def _mask_cell_centers(
    mask: np.ndarray,
    heatmap: ObstacleHeatmap,
    *,
    max_samples: int,
) -> list[tuple[float, float]]:
    rows, cols = np.where(mask)
    count = int(rows.size)
    if count == 0:
        return []
    if count > max_samples:
        idx = np.linspace(0, count - 1, max_samples).astype(np.int64)
        rows = rows[idx]
        cols = cols[idx]
    return [
        heatmap.cell_to_world(float(row), float(col))
        for row, col in zip(rows, cols, strict=True)
    ]


def _point_inside_ratio(
    geom: BaseGeometry,
    points: list[tuple[float, float]],
) -> float:
    if not points:
        return 0.0
    inside = sum(1 for x, y in points if geom.covers(Point(x, y)))
    return float(inside) / float(len(points))


def _wall_near_boundary_ratio(
    geom: BaseGeometry,
    points: list[tuple[float, float]],
    *,
    max_distance_m: float,
) -> float:
    if not points:
        return 0.0
    boundary = geom.boundary
    near = sum(
        1
        for x, y in points
        if boundary.distance(Point(float(x), float(y))) <= max_distance_m
    )
    return float(near) / float(len(points))
