from __future__ import annotations

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, shape

from indoor_server.application.building.steps.rectilinear_cover import (
    RectilinearWalkableCoverStep,
)
from indoor_server.domain.building.models import GridOrigin, WalkableGrid


def _grid(mask: np.ndarray) -> WalkableGrid:
    origin = GridOrigin(
        x0=0.0,
        y0=0.0,
        z0=0.0,
        cell_size=1.0,
        w=mask.shape[1],
        h=mask.shape[0],
    )
    return WalkableGrid(
        origin=origin,
        mask=mask.astype(bool),
        observation_count=mask.astype(np.uint16),
    )


def test_rectilinear_cover_keeps_polygon_edges_axis_aligned() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[1:8, 2:5] = True
    mask[6:9, 2:10] = True

    result = RectilinearWalkableCoverStep(
        target_coverage=1.0,
        exact_tail=True,
    ).run(_grid(mask))

    assert result.footprint_geojson is not None
    assert result.metadata["coverage_ratio"] == pytest.approx(1.0)
    assert result.grid.mask.sum() >= mask.sum()
    assert result.metadata["all_edges_axis_aligned"] is True
    assert _all_edges_axis_aligned(shape(result.footprint_geojson))


def test_rectilinear_cover_can_abstract_sloped_evidence_with_rectangles() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    for idx in range(12):
        mask[idx + 2:idx + 4, idx + 1:idx + 5] = True

    result = RectilinearWalkableCoverStep(
        min_fill_ratio=0.22,
        overcoverage_penalty=0.8,
        target_coverage=0.90,
        exact_tail=False,
    ).run(_grid(mask))

    assert result.metadata["coverage_ratio"] >= 0.90
    assert result.metadata["rectangle_count"] < int(mask.sum())
    assert result.metadata["overcovered_cells"] > 0
    assert result.footprint_geojson is not None
    assert _all_edges_axis_aligned(shape(result.footprint_geojson))


def test_rectilinear_cover_uses_avoid_weights_when_scoring() -> None:
    mask = np.zeros((4, 6), dtype=bool)
    mask[:, 0:2] = True
    mask[:, 4:6] = True
    avoid = np.zeros(mask.shape, dtype=np.float32)
    avoid[:, 2:4] = 10.0

    result = RectilinearWalkableCoverStep(
        min_side_cells=2,
        min_fill_ratio=0.50,
        overcoverage_penalty=0.10,
        avoid_penalty=2.0,
        candidate_stride_cells=1,
        target_coverage=1.0,
    ).run(_grid(mask), avoid=avoid)

    assert result.metadata["avoid_used"] is True
    assert result.metadata["selected_avoid_sum"] == pytest.approx(0.0)
    assert result.metadata["coverage_ratio"] == pytest.approx(1.0)
    assert not result.grid.mask[:, 2:4].any()


def test_rectilinear_cover_rejects_candidates_with_too_much_avoid() -> None:
    mask = np.zeros((5, 9), dtype=bool)
    mask[:, 0:3] = True
    mask[:, 3:6] = True
    avoid = np.zeros(mask.shape, dtype=np.float32)
    avoid[:, 3:6] = 1.0

    result = RectilinearWalkableCoverStep(
        min_side_cells=2,
        min_fill_ratio=0.50,
        target_coverage=0.50,
        max_avoid_ratio=0.01,
        candidate_stride_cells=1,
    ).run(_grid(mask), avoid=avoid)

    assert result.metadata["hard_avoid_used"] is True
    assert result.grid.mask[:, 0:3].any()
    assert not result.grid.mask[:, 3:6].any()


def test_rectilinear_cover_rotated_grid_outputs_dominant_axis_edges() -> None:
    mask = _rotated_rectangle_mask(angle_deg=24.0)

    result = RectilinearWalkableCoverStep(
        min_fill_ratio=0.35,
        target_coverage=0.92,
        dominant_angle_deg=24.0,
    ).run(_grid(mask))

    assert result.footprint_geojson is not None
    assert result.metadata["rotated_grid_used"] is True
    assert result.metadata["dominant_angle_deg"] == pytest.approx(24.0)
    assert result.metadata["source_coverage_ratio"] >= 0.90
    assert result.metadata["all_edges_dominant_axis_aligned"] is True
    geom = shape(result.footprint_geojson)
    rotated_back = affinity.rotate(geom, -24.0, origin=(0.0, 0.0))
    assert _all_edges_axis_aligned(rotated_back)


def _rotated_rectangle_mask(*, angle_deg: float) -> np.ndarray:
    import cv2

    origin = GridOrigin(x0=-8.0, y0=-8.0, z0=0.0, cell_size=0.20, w=80, h=80)
    base = np.asarray(
        [
            [-4.0, -0.8],
            [4.0, -0.8],
            [4.0, 0.8],
            [-4.0, 0.8],
        ],
        dtype=np.float64,
    )
    theta = np.deg2rad(angle_deg)
    rot = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    world = base @ rot.T
    pts = np.asarray(
        [
            [
                int(round((x - origin.x0) / origin.cell_size)),
                int(round((y - origin.y0) / origin.cell_size)),
            ]
            for x, y in world
        ],
        dtype=np.int32,
    )
    mask = np.zeros((origin.h, origin.w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1)
    return mask.astype(bool)


def _all_edges_axis_aligned(geom: object) -> bool:
    polygons: list[Polygon]
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        return False

    for polygon in polygons:
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coords = list(ring.coords)
            for a, b in zip(coords, coords[1:], strict=False):
                if not (
                    a[0] == pytest.approx(b[0])
                    or a[1] == pytest.approx(b[1])
                ):
                    return False
    return True
