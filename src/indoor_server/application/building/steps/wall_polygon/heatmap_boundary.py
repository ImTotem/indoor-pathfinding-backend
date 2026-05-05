"""Obstacle heatmap boundary polygon extraction.

This step intentionally differs from the older wall-line assembly path.  The
wall-line path tries to fit CAD wall segments and validate them against a floor
polygon.  For scans where the RTABMap obstacle heatmap already has a clear
outline, the useful artifact is much simpler: connect small breaks in the
raster and trace the external contour directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.ops import unary_union

from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


@dataclass(frozen=True)
class HeatmapBoundaryParams:
    """Tunable parameters for direct heatmap-boundary extraction."""

    min_cell_hits: int = 8
    median_filter_size_cells: int = 1
    bridge_gap_radius_cells: int = 3
    close_radius_cells: int = 1
    min_component_area_m2: float = 0.20
    simplify_tolerance_m: float = 0.05
    keep_largest_components: int = 3

    def to_metadata(self) -> dict[str, object]:
        return {
            "min_cell_hits": self.min_cell_hits,
            "median_filter_size_cells": self.median_filter_size_cells,
            "bridge_gap_radius_cells": self.bridge_gap_radius_cells,
            "close_radius_cells": self.close_radius_cells,
            "min_component_area_m2": self.min_component_area_m2,
            "simplify_tolerance_m": self.simplify_tolerance_m,
            "keep_largest_components": self.keep_largest_components,
        }


@dataclass(frozen=True)
class HeatmapBoundaryResult:
    """Direct boundary extraction result."""

    accepted: bool
    boundary_geojson: dict[str, object] | None
    boundary_mask: np.ndarray
    threshold_mask: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)
    fail_reason: str | None = None


class HeatmapBoundaryStep:
    """Extract a polygon by tracing the obstacle heatmap outline."""

    def __init__(self, params: HeatmapBoundaryParams | None = None) -> None:
        self._params = params if params is not None else HeatmapBoundaryParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.min_cell_hits < 1:
            raise ValueError("min_cell_hits must be >= 1")
        if p.median_filter_size_cells < 1:
            raise ValueError("median_filter_size_cells must be >= 1")
        if p.bridge_gap_radius_cells < 0 or p.close_radius_cells < 0:
            raise ValueError("morphology radii must be >= 0")
        if p.min_component_area_m2 < 0:
            raise ValueError("min_component_area_m2 must be >= 0")
        if p.simplify_tolerance_m < 0:
            raise ValueError("simplify_tolerance_m must be >= 0")
        if p.keep_largest_components < 1:
            raise ValueError("keep_largest_components must be >= 1")

    def run(self, heatmap: ObstacleHeatmap) -> HeatmapBoundaryResult:
        params = self._params
        counts = np.asarray(heatmap.counts, dtype=np.int32)
        threshold_mask = counts >= params.min_cell_hits
        threshold_cells = int(threshold_mask.sum())
        if threshold_cells == 0:
            return HeatmapBoundaryResult(
                accepted=False,
                boundary_geojson=None,
                boundary_mask=np.zeros_like(threshold_mask, dtype=bool),
                threshold_mask=threshold_mask,
                fail_reason="no_threshold_cells",
                metadata={
                    "params": params.to_metadata(),
                    "threshold_cells": 0,
                },
            )

        mask = threshold_mask
        if params.median_filter_size_cells >= 2:
            mask = ndimage.median_filter(
                mask.astype(np.uint8),
                size=params.median_filter_size_cells,
            ).astype(bool)

        if params.bridge_gap_radius_cells > 0:
            mask = _bridge_axis_aligned_gaps(mask, params.bridge_gap_radius_cells)
        if params.close_radius_cells > 0:
            mask = ndimage.binary_closing(
                mask,
                structure=_disk_kernel(params.close_radius_cells),
            )

        components_before, mask = _keep_largest_components(
            np.asarray(mask, dtype=bool),
            keep_n=params.keep_largest_components,
            min_area_cells=_area_m2_to_cells(
                params.min_component_area_m2,
                heatmap.cell_size_m,
            ),
        )
        boundary_cells = int(mask.sum())
        threshold_retained_cells = int(np.logical_and(threshold_mask, mask).sum())
        if boundary_cells == 0:
            return HeatmapBoundaryResult(
                accepted=False,
                boundary_geojson=None,
                boundary_mask=mask,
                threshold_mask=threshold_mask,
                fail_reason="no_boundary_cells",
                metadata={
                    "params": params.to_metadata(),
                    "threshold_cells": threshold_cells,
                    "components_before_filter": components_before,
                    "boundary_cells": 0,
                },
            )

        boundary_geojson, polygon_area_m2, vertex_count = _mask_to_geojson(
            mask=mask,
            heatmap=heatmap,
            min_area_m2=params.min_component_area_m2,
            simplify_tolerance_m=params.simplify_tolerance_m,
        )
        if boundary_geojson is None:
            return HeatmapBoundaryResult(
                accepted=False,
                boundary_geojson=None,
                boundary_mask=mask,
                threshold_mask=threshold_mask,
                fail_reason="polygon_empty",
                metadata={
                    "params": params.to_metadata(),
                    "threshold_cells": threshold_cells,
                    "components_before_filter": components_before,
                    "boundary_cells": boundary_cells,
                },
            )

        metadata: dict[str, object] = {
            "params": params.to_metadata(),
            "threshold_cells": threshold_cells,
            "threshold_retained_cells": threshold_retained_cells,
            "threshold_recall": threshold_retained_cells / float(max(1, threshold_cells)),
            "boundary_cells": boundary_cells,
            "cells_added_by_gap_bridge": int(max(0, boundary_cells - threshold_cells)),
            "components_before_filter": components_before,
            "polygon_area_m2": polygon_area_m2,
            "polygon_vertex_count": vertex_count,
            "bridge_expansion_ratio": boundary_cells / float(max(1, threshold_cells)),
            "cell_size_m": heatmap.cell_size_m,
            "grid_shape": list(heatmap.shape),
        }
        return HeatmapBoundaryResult(
            accepted=True,
            boundary_geojson=boundary_geojson,
            boundary_mask=mask,
            threshold_mask=threshold_mask,
            metadata=metadata,
        )


def _disk_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    size = 2 * radius + 1
    yy, xx = np.ogrid[:size, :size]
    cy = cx = radius
    return np.asarray(((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2, dtype=bool)


def _bridge_axis_aligned_gaps(mask: np.ndarray, radius: int) -> np.ndarray:
    """Connect short horizontal/vertical gaps without heavy blob inflation."""
    if radius <= 0:
        return mask.copy()
    horizontal = np.ones((1, 2 * radius + 1), dtype=bool)
    vertical = np.ones((2 * radius + 1, 1), dtype=bool)
    bridged_h = ndimage.binary_closing(mask, structure=horizontal)
    bridged_v = ndimage.binary_closing(mask, structure=vertical)
    return np.asarray(mask | bridged_h | bridged_v, dtype=bool)


def _area_m2_to_cells(area_m2: float, cell_size_m: float) -> int:
    if area_m2 <= 0:
        return 1
    return max(1, int(round(area_m2 / (cell_size_m * cell_size_m))))


def _keep_largest_components(
    mask: np.ndarray,
    *,
    keep_n: int,
    min_area_cells: int,
) -> tuple[int, np.ndarray]:
    labeled, num = ndimage.label(mask)
    if num == 0:
        return 0, np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    candidates = [
        (label_id, int(size))
        for label_id, size in enumerate(sizes)
        if label_id != 0 and int(size) >= min_area_cells
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    keep = {label_id for label_id, _size in candidates[:keep_n]}
    return int(num), np.isin(labeled, list(keep))


def _mask_to_geojson(
    *,
    mask: np.ndarray,
    heatmap: ObstacleHeatmap,
    min_area_m2: float,
    simplify_tolerance_m: float,
) -> tuple[dict[str, object] | None, float, int]:
    if not mask.any():
        return None, 0.0, 0

    cell = heatmap.cell_size_m
    polygons: list[Polygon] = []
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

    vertex_count = 0
    for geom in multi.geoms:
        vertex_count += max(0, len(geom.exterior.coords) - 1)
    return dict(mapping(multi)), float(multi.area), vertex_count
