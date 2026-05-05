"""Rectilinear rectangle-cover abstraction for walkable grids."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import cos, radians, sin, sqrt

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

from indoor_server.domain.building.models import GridOrigin, WalkableGrid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RectilinearRectangle:
    row0: int
    col0: int
    row1: int
    col1: int
    target_cells: int
    total_cells: int
    fill_ratio: float
    confidence_sum: float = 0.0
    avoid_sum: float = 0.0

    @property
    def area_cells(self) -> int:
        return self.total_cells

    def metadata(self) -> dict[str, object]:
        return {
            "row0": self.row0,
            "col0": self.col0,
            "row1": self.row1,
            "col1": self.col1,
            "target_cells": self.target_cells,
            "total_cells": self.total_cells,
            "fill_ratio": self.fill_ratio,
            "confidence_sum": self.confidence_sum,
            "avoid_sum": self.avoid_sum,
        }


@dataclass(frozen=True)
class RectilinearCoverResult:
    grid: WalkableGrid
    footprint_geojson: dict[str, object] | None
    rectangles: list[RectilinearRectangle]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _RotatedGridEvidence:
    grid: WalkableGrid
    confidence: np.ndarray | None
    avoid: np.ndarray | None


class RectilinearWalkableCoverStep:
    """Approximate walkable evidence with a union of axis-aligned rectangles.

    The step treats the input mask as evidence, not as a contour to copy. It
    proposes multi-scale rectangles, scores them by newly-covered target cells
    minus overcoverage, then greedily selects a compact cover. The result is a
    rectilinear map polygon and a grid that downstream skeleton/route steps can
    consume.
    """

    def __init__(
        self,
        *,
        min_side_cells: int = 3,
        min_target_cells: int = 4,
        min_fill_ratio: float = 0.28,
        target_coverage: float = 0.95,
        overcoverage_penalty: float = 1.50,
        rectangle_penalty: float = 0.50,
        confidence_weight: float = 0.20,
        avoid_penalty: float = 1.00,
        max_avoid_ratio: float = 0.05,
        candidate_stride_cells: int = 2,
        max_candidates: int = 6000,
        max_iterations: int = 512,
        exact_tail: bool = False,
        dominant_angle_deg: float | None = None,
        rotation_min_abs_angle_deg: float = 1.0,
        rotated_padding_cells: int = 4,
    ) -> None:
        if min_side_cells <= 0:
            raise ValueError("min_side_cells must be > 0")
        if min_target_cells <= 0:
            raise ValueError("min_target_cells must be > 0")
        if not (0.0 < min_fill_ratio <= 1.0):
            raise ValueError("min_fill_ratio must be in (0, 1]")
        if not (0.0 < target_coverage <= 1.0):
            raise ValueError("target_coverage must be in (0, 1]")
        if candidate_stride_cells <= 0:
            raise ValueError("candidate_stride_cells must be > 0")
        if not (0.0 <= max_avoid_ratio <= 1.0):
            raise ValueError("max_avoid_ratio must be in [0, 1]")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be > 0")
        if rotation_min_abs_angle_deg < 0.0:
            raise ValueError("rotation_min_abs_angle_deg must be >= 0")
        if rotated_padding_cells < 0:
            raise ValueError("rotated_padding_cells must be >= 0")
        self._min_side_cells = min_side_cells
        self._min_target_cells = min_target_cells
        self._min_fill_ratio = min_fill_ratio
        self._target_coverage = target_coverage
        self._overcoverage_penalty = overcoverage_penalty
        self._rectangle_penalty = rectangle_penalty
        self._confidence_weight = confidence_weight
        self._avoid_penalty = avoid_penalty
        self._max_avoid_ratio = max_avoid_ratio
        self._candidate_stride_cells = candidate_stride_cells
        self._max_candidates = max_candidates
        self._max_iterations = max_iterations
        self._exact_tail = exact_tail
        self._dominant_angle_deg = dominant_angle_deg
        self._rotation_min_abs_angle_deg = rotation_min_abs_angle_deg
        self._rotated_padding_cells = rotated_padding_cells

    def run(
        self,
        grid: WalkableGrid,
        *,
        confidence: np.ndarray | None = None,
        avoid: np.ndarray | None = None,
    ) -> RectilinearCoverResult:
        target = grid.mask.astype(bool)
        confidence_weights = _prepare_weight_grid(confidence, target.shape, "confidence")
        avoid_weights = _prepare_weight_grid(avoid, target.shape, "avoid")
        target_cells = int(target.sum())
        if target_cells == 0:
            return RectilinearCoverResult(
                grid=grid,
                footprint_geojson=None,
                rectangles=[],
                metadata={
                    "source": "rectilinear_rectangle_cover",
                    "strategy": "multi_scale_weighted_greedy",
                    "issues": ["empty_target_mask"],
                    "target_cells": 0,
                    "confidence_used": confidence_weights is not None,
                    "avoid_used": avoid_weights is not None,
                    "max_avoid_ratio": self._max_avoid_ratio,
                    "hard_avoid_used": avoid_weights is not None,
                    "rotated_grid_used": False,
                },
            )

        dominant_angle = _normalize_axis_angle(self._dominant_angle_deg)
        if (
            dominant_angle is not None
            and abs(dominant_angle) >= self._rotation_min_abs_angle_deg
        ):
            return self._run_rotated(
                grid,
                target=target,
                confidence=confidence_weights,
                avoid=avoid_weights,
                dominant_angle_deg=dominant_angle,
            )

        candidates = self._generate_candidates(
            target,
            confidence=confidence_weights,
            avoid=avoid_weights,
        )
        selected = self._greedy_select(target, candidates)
        if self._exact_tail:
            selected.extend(self._cover_remaining_exact(target, selected))

        cover_mask = np.zeros_like(target, dtype=bool)
        for rect in selected:
            cover_mask[rect.row0:rect.row1, rect.col0:rect.col1] = True

        covered_target_cells = int(np.logical_and(cover_mask, target).sum())
        overcovered_cells = int(np.logical_and(cover_mask, ~target).sum())
        selected_confidence = float(sum(rect.confidence_sum for rect in selected))
        selected_avoid = float(sum(rect.avoid_sum for rect in selected))
        footprint_geojson = self._rectangles_to_geojson(selected, grid)
        obs = np.where(
            cover_mask,
            np.maximum(grid.observation_count, cover_mask.astype(np.uint16)),
            np.zeros_like(grid.observation_count, dtype=np.uint16),
        )
        result_grid = WalkableGrid(
            origin=grid.origin,
            mask=cover_mask,
            observation_count=obs.astype(np.uint16),
        )
        coverage_ratio = (
            covered_target_cells / float(target_cells)
            if target_cells > 0
            else 0.0
        )
        metadata = {
            "source": "rectilinear_rectangle_cover",
            "strategy": "multi_scale_weighted_greedy",
            "target_cells": target_cells,
            "covered_target_cells": covered_target_cells,
            "coverage_ratio": coverage_ratio,
            "overcovered_cells": overcovered_cells,
            "output_cells": int(cover_mask.sum()),
            "candidate_count": len(candidates),
            "rectangle_count": len(selected),
            "max_rectangle_area_cells": (
                max((rect.area_cells for rect in selected), default=0)
            ),
            "target_coverage": self._target_coverage,
            "min_fill_ratio": self._min_fill_ratio,
            "confidence_used": confidence_weights is not None,
            "avoid_used": avoid_weights is not None,
            "selected_confidence_sum": selected_confidence,
            "selected_avoid_sum": selected_avoid,
            "confidence_weight": self._confidence_weight,
            "avoid_penalty": self._avoid_penalty,
            "max_avoid_ratio": self._max_avoid_ratio,
            "hard_avoid_used": avoid_weights is not None,
            "all_edges_axis_aligned": True,
            "all_edges_dominant_axis_aligned": True,
            "world_edges_axis_aligned": True,
            "rotated_grid_used": False,
            "dominant_angle_deg": 0.0,
            "source_coverage_ratio": coverage_ratio,
            "rectangles": [rect.metadata() for rect in selected[:64]],
            "rectangles_truncated": max(0, len(selected) - 64),
        }
        logger.info(
            "rectilinear cover done target=%d covered=%d coverage=%.3f "
            "over=%d rects=%d candidates=%d",
            target_cells,
            covered_target_cells,
            coverage_ratio,
            overcovered_cells,
            len(selected),
            len(candidates),
        )
        return RectilinearCoverResult(
            grid=result_grid,
            footprint_geojson=footprint_geojson,
            rectangles=selected,
            metadata=metadata,
        )

    def _run_rotated(
        self,
        grid: WalkableGrid,
        *,
        target: np.ndarray,
        confidence: np.ndarray | None,
        avoid: np.ndarray | None,
        dominant_angle_deg: float,
    ) -> RectilinearCoverResult:
        rotated = _rotate_grid_evidence(
            grid,
            target=target,
            confidence=confidence,
            avoid=avoid,
            angle_deg=-dominant_angle_deg,
            padding_cells=self._rotated_padding_cells,
        )
        axis_cover = RectilinearWalkableCoverStep(
            min_side_cells=self._min_side_cells,
            min_target_cells=self._min_target_cells,
            min_fill_ratio=self._min_fill_ratio,
            target_coverage=self._target_coverage,
            overcoverage_penalty=self._overcoverage_penalty,
            rectangle_penalty=self._rectangle_penalty,
            confidence_weight=self._confidence_weight,
            avoid_penalty=self._avoid_penalty,
            max_avoid_ratio=self._max_avoid_ratio,
            candidate_stride_cells=self._candidate_stride_cells,
            max_candidates=self._max_candidates,
            max_iterations=self._max_iterations,
            exact_tail=self._exact_tail,
        ).run(
            rotated.grid,
            confidence=rotated.confidence,
            avoid=rotated.avoid,
        )
        if not axis_cover.rectangles:
            return axis_cover

        polygons = _rotated_rectangles_to_world_polygons(
            axis_cover.rectangles,
            rotated.grid,
            angle_deg=dominant_angle_deg,
        )
        footprint_geojson = _polygons_to_geojson(polygons)
        source_cover_mask = _rasterize_world_polygons_to_grid(polygons, grid)
        covered_target_cells = int(np.logical_and(source_cover_mask, target).sum())
        target_cells = int(target.sum())
        overcovered_cells = int(np.logical_and(source_cover_mask, ~target).sum())
        source_coverage_ratio = (
            covered_target_cells / float(target_cells) if target_cells else 0.0
        )
        obs = np.where(
            source_cover_mask,
            np.maximum(grid.observation_count, source_cover_mask.astype(np.uint16)),
            np.zeros_like(grid.observation_count, dtype=np.uint16),
        )
        result_grid = WalkableGrid(
            origin=grid.origin,
            mask=source_cover_mask,
            observation_count=obs.astype(np.uint16),
        )
        rotated_target_cells = axis_cover.metadata.get("target_cells")
        rotated_covered_target_cells = axis_cover.metadata.get("covered_target_cells")
        rotated_overcovered_cells = axis_cover.metadata.get("overcovered_cells")
        metadata = dict(axis_cover.metadata)
        metadata.update(
            {
                "strategy": "rotated_grid_multi_scale_weighted_greedy",
                "rotated_grid_used": True,
                "dominant_angle_deg": dominant_angle_deg,
                "rectangle_frame": "rotated_grid",
                "rotated_grid_origin": rotated.grid.origin.model_dump(),
                "rotated_grid_shape": {
                    "h": int(rotated.grid.mask.shape[0]),
                    "w": int(rotated.grid.mask.shape[1]),
                },
                "rotated_target_cells": rotated_target_cells,
                "rotated_covered_target_cells": rotated_covered_target_cells,
                "rotated_overcovered_cells": rotated_overcovered_cells,
                "rotated_coverage_ratio": axis_cover.metadata.get("coverage_ratio"),
                "target_cells": target_cells,
                "source_coverage_ratio": source_coverage_ratio,
                "coverage_ratio": source_coverage_ratio,
                "covered_target_cells": covered_target_cells,
                "overcovered_cells": overcovered_cells,
                "output_cells": int(source_cover_mask.sum()),
                "all_edges_axis_aligned": True,
                "all_edges_dominant_axis_aligned": True,
                "world_edges_axis_aligned": abs(dominant_angle_deg) < 1e-6,
            }
        )
        return RectilinearCoverResult(
            grid=result_grid,
            footprint_geojson=footprint_geojson,
            rectangles=axis_cover.rectangles,
            metadata=metadata,
        )

    def _generate_candidates(
        self,
        target: np.ndarray,
        *,
        confidence: np.ndarray | None,
        avoid: np.ndarray | None,
    ) -> list[RectilinearRectangle]:
        h, w = target.shape
        integral = _integral(target)
        confidence_integral = _integral_float(confidence) if confidence is not None else None
        avoid_integral = _integral_float(avoid) if avoid is not None else None
        avoid_binary_integral = _integral(avoid > 0) if avoid is not None else None
        row0, col0, row1, col1 = _target_bbox(target)
        heights = _size_ladder(row1 - row0, self._min_side_cells)
        widths = _size_ladder(col1 - col0, self._min_side_cells)
        seen: set[tuple[int, int, int, int]] = set()
        scored: list[tuple[float, RectilinearRectangle]] = []

        row_stop = max(row0 + 1, row1)
        col_stop = max(col0 + 1, col1)
        for r in range(row0, row_stop, self._candidate_stride_cells):
            for c in range(col0, col_stop, self._candidate_stride_cells):
                for rh in heights:
                    rr = r + rh
                    if rr > h or rr <= row0:
                        continue
                    for cw in widths:
                        cc = c + cw
                        if cc > w or cc <= col0:
                            continue
                        key = (r, c, rr, cc)
                        if key in seen:
                            continue
                        seen.add(key)
                        target_count = _rect_sum(integral, r, c, rr, cc)
                        area = (rr - r) * (cc - c)
                        if target_count < self._min_target_cells:
                            continue
                        fill_ratio = target_count / float(area)
                        if fill_ratio < self._min_fill_ratio:
                            continue
                        if avoid_binary_integral is not None:
                            avoid_cells = _rect_sum(avoid_binary_integral, r, c, rr, cc)
                            avoid_ratio = avoid_cells / float(area)
                            if avoid_ratio > self._max_avoid_ratio:
                                continue
                        confidence_sum = (
                            _rect_sum_float(confidence_integral, r, c, rr, cc)
                            if confidence_integral is not None
                            else 0.0
                        )
                        avoid_sum = (
                            _rect_sum_float(avoid_integral, r, c, rr, cc)
                            if avoid_integral is not None
                            else 0.0
                        )
                        rect = RectilinearRectangle(
                            row0=r,
                            col0=c,
                            row1=rr,
                            col1=cc,
                            target_cells=target_count,
                            total_cells=area,
                            fill_ratio=fill_ratio,
                            confidence_sum=confidence_sum,
                            avoid_sum=avoid_sum,
                        )
                        over = area - target_count
                        score = self._score(
                            new_cells=target_count,
                            over_cells=over,
                            area_cells=area,
                            confidence_sum=confidence_sum,
                            avoid_sum=avoid_sum,
                        )
                        scored.append((score, rect))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [rect for _, rect in scored[: self._max_candidates]]

    def _greedy_select(
        self,
        target: np.ndarray,
        candidates: list[RectilinearRectangle],
    ) -> list[RectilinearRectangle]:
        target_total = int(target.sum())
        remaining = target.copy()
        selected: list[RectilinearRectangle] = []
        covered = 0
        required = int(np.ceil(target_total * self._target_coverage))

        for _ in range(self._max_iterations):
            if covered >= required:
                break
            best_score = float("-inf")
            best_rect: RectilinearRectangle | None = None
            for rect in candidates:
                new_cells = int(
                    remaining[rect.row0:rect.row1, rect.col0:rect.col1].sum()
                )
                if new_cells <= 0:
                    continue
                over = rect.total_cells - rect.target_cells
                score = self._score(
                    new_cells=new_cells,
                    over_cells=over,
                    area_cells=rect.total_cells,
                    confidence_sum=rect.confidence_sum,
                    avoid_sum=rect.avoid_sum,
                )
                if score > best_score:
                    best_score = score
                    best_rect = rect
            if best_rect is None:
                break
            selected.append(best_rect)
            before = int(remaining.sum())
            remaining[best_rect.row0:best_rect.row1, best_rect.col0:best_rect.col1] = False
            covered += before - int(remaining.sum())

        return selected

    def _score(
        self,
        *,
        new_cells: int,
        over_cells: int,
        area_cells: int,
        confidence_sum: float,
        avoid_sum: float,
    ) -> float:
        return (
            new_cells
            - self._overcoverage_penalty * over_cells
            - self._rectangle_penalty
            + 0.05 * sqrt(area_cells)
            + self._confidence_weight * confidence_sum
            - self._avoid_penalty * avoid_sum
        )

    def _cover_remaining_exact(
        self,
        target: np.ndarray,
        selected: list[RectilinearRectangle],
    ) -> list[RectilinearRectangle]:
        remaining = target.copy()
        for rect in selected:
            remaining[rect.row0:rect.row1, rect.col0:rect.col1] = False

        tail: list[RectilinearRectangle] = []
        while remaining.any():
            exact_rect = _largest_exact_rectangle(remaining)
            if exact_rect is None:
                break
            tail.append(exact_rect)
            remaining[
                exact_rect.row0:exact_rect.row1,
                exact_rect.col0:exact_rect.col1,
            ] = False
        return tail

    def _rectangles_to_geojson(
        self,
        rectangles: list[RectilinearRectangle],
        grid: WalkableGrid,
    ) -> dict[str, object] | None:
        if not rectangles:
            return None
        cs = grid.origin.cell_size
        polygons: list[Polygon] = []
        for rect in rectangles:
            x0 = grid.origin.x0 + rect.col0 * cs
            x1 = grid.origin.x0 + rect.col1 * cs
            y0 = grid.origin.y0 + rect.row0 * cs
            y1 = grid.origin.y0 + rect.row1 * cs
            if x1 <= x0 or y1 <= y0:
                continue
            polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        if not polygons:
            return None
        union = unary_union(polygons)
        if isinstance(union, Polygon):
            union = MultiPolygon([union])
        elif not isinstance(union, MultiPolygon):
            union = MultiPolygon([geom for geom in union.geoms if isinstance(geom, Polygon)])
        return dict(mapping(union))


def _rotate_grid_evidence(
    grid: WalkableGrid,
    *,
    target: np.ndarray,
    confidence: np.ndarray | None,
    avoid: np.ndarray | None,
    angle_deg: float,
    padding_cells: int,
) -> _RotatedGridEvidence:
    import cv2

    rows, cols = np.where(target)
    if len(rows) == 0:
        return _RotatedGridEvidence(grid=grid, confidence=confidence, avoid=avoid)

    cs = grid.origin.cell_size
    row0 = int(rows.min())
    row1 = int(rows.max()) + 1
    col0 = int(cols.min())
    col1 = int(cols.max()) + 1
    bbox_corners = np.asarray(
        [
            [grid.origin.x0 + col0 * cs, grid.origin.y0 + row0 * cs],
            [grid.origin.x0 + col1 * cs, grid.origin.y0 + row0 * cs],
            [grid.origin.x0 + col1 * cs, grid.origin.y0 + row1 * cs],
            [grid.origin.x0 + col0 * cs, grid.origin.y0 + row1 * cs],
        ],
        dtype=np.float64,
    )
    rotated_bbox = _rotate_xy(bbox_corners, angle_deg)
    pad = padding_cells * cs
    rx0 = float(np.floor((rotated_bbox[:, 0].min() - pad) / cs) * cs)
    ry0 = float(np.floor((rotated_bbox[:, 1].min() - pad) / cs) * cs)
    rx1 = float(np.ceil((rotated_bbox[:, 0].max() + pad) / cs) * cs)
    ry1 = float(np.ceil((rotated_bbox[:, 1].max() + pad) / cs) * cs)
    w = max(1, int(np.ceil((rx1 - rx0) / cs)) + 1)
    h = max(1, int(np.ceil((ry1 - ry0) / cs)) + 1)

    rotated_mask = np.zeros((h, w), dtype=np.uint8)
    for r, c in zip(rows, cols, strict=False):
        x0 = grid.origin.x0 + c * cs
        x1 = x0 + cs
        y0 = grid.origin.y0 + r * cs
        y1 = y0 + cs
        corners = np.asarray(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            dtype=np.float64,
        )
        rotated = _rotate_xy(corners, angle_deg)
        px = np.rint((rotated[:, 0] - rx0) / cs).astype(np.int32)
        py = np.rint((rotated[:, 1] - ry0) / cs).astype(np.int32)
        pts = np.stack([px, py], axis=1)
        cv2.fillConvexPoly(rotated_mask, pts, color=1)

    rotated_confidence = _rotate_weight_grid(
        grid,
        confidence,
        angle_deg=angle_deg,
        x0=rx0,
        y0=ry0,
        shape=(h, w),
    )
    rotated_avoid = _rotate_weight_grid(
        grid,
        avoid,
        angle_deg=angle_deg,
        x0=rx0,
        y0=ry0,
        shape=(h, w),
    )
    origin = GridOrigin(x0=rx0, y0=ry0, z0=grid.origin.z0, cell_size=cs, w=w, h=h)
    return _RotatedGridEvidence(
        grid=WalkableGrid(
            origin=origin,
            mask=rotated_mask.astype(bool),
            observation_count=rotated_mask.astype(np.uint16),
        ),
        confidence=rotated_confidence,
        avoid=rotated_avoid,
    )


def _rotate_weight_grid(
    grid: WalkableGrid,
    values: np.ndarray | None,
    *,
    angle_deg: float,
    x0: float,
    y0: float,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if values is None:
        return None
    rows, cols = np.where(values > 0)
    out = np.zeros(shape, dtype=np.float32)
    if len(rows) == 0:
        return out
    cs = grid.origin.cell_size
    xy = np.stack(
        [
            grid.origin.x0 + (cols.astype(np.float64) + 0.5) * cs,
            grid.origin.y0 + (rows.astype(np.float64) + 0.5) * cs,
        ],
        axis=1,
    )
    rotated = _rotate_xy(xy, angle_deg)
    rr = np.floor((rotated[:, 1] - y0) / cs).astype(np.int64)
    cc = np.floor((rotated[:, 0] - x0) / cs).astype(np.int64)
    valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    if valid.any():
        np.maximum.at(out, (rr[valid], cc[valid]), values[rows[valid], cols[valid]])
    return out


def _rotated_rectangles_to_world_polygons(
    rectangles: list[RectilinearRectangle],
    grid: WalkableGrid,
    *,
    angle_deg: float,
) -> list[Polygon]:
    cs = grid.origin.cell_size
    polygons: list[Polygon] = []
    for rect in rectangles:
        x0 = grid.origin.x0 + rect.col0 * cs
        x1 = grid.origin.x0 + rect.col1 * cs
        y0 = grid.origin.y0 + rect.row0 * cs
        y1 = grid.origin.y0 + rect.row1 * cs
        if x1 <= x0 or y1 <= y0:
            continue
        corners = np.asarray(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            dtype=np.float64,
        )
        world = _rotate_xy(corners, angle_deg)
        polygons.append(Polygon([(float(x), float(y)) for x, y in world]))
    return polygons


def _rasterize_world_polygons_to_grid(
    polygons: list[Polygon],
    grid: WalkableGrid,
) -> np.ndarray:
    import cv2

    mask = np.zeros(grid.mask.shape, dtype=np.uint8)
    cs = grid.origin.cell_size
    for polygon in polygons:
        exterior = np.asarray(
            [
                [
                    int(round((x - grid.origin.x0) / cs)),
                    int(round((y - grid.origin.y0) / cs)),
                ]
                for x, y in polygon.exterior.coords
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [exterior], color=1)
        for interior in polygon.interiors:
            hole = np.asarray(
                [
                    [
                        int(round((x - grid.origin.x0) / cs)),
                        int(round((y - grid.origin.y0) / cs)),
                    ]
                    for x, y in interior.coords
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [hole], color=0)
    return mask.astype(bool)


def _polygons_to_geojson(polygons: list[Polygon]) -> dict[str, object] | None:
    if not polygons:
        return None
    union = unary_union(polygons)
    if isinstance(union, Polygon):
        union = MultiPolygon([union])
    elif not isinstance(union, MultiPolygon):
        union = MultiPolygon([geom for geom in union.geoms if isinstance(geom, Polygon)])
    return dict(mapping(union))


def _rotate_xy(points: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = radians(angle_deg)
    c = cos(theta)
    s = sin(theta)
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = points[:, 0] * c - points[:, 1] * s
    out[:, 1] = points[:, 0] * s + points[:, 1] * c
    return out


def _normalize_axis_angle(angle_deg: float | None) -> float | None:
    if angle_deg is None:
        return None
    angle = float(angle_deg) % 90.0
    if angle > 45.0:
        angle -= 90.0
    if angle <= -45.0:
        angle += 90.0
    return float(angle)


def _integral(mask: np.ndarray) -> np.ndarray:
    values = mask.astype(np.int32)
    return np.pad(values, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _integral_float(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64)
    return np.pad(arr, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(integral: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        integral[r1, c1]
        - integral[r0, c1]
        - integral[r1, c0]
        + integral[r0, c0]
    )


def _rect_sum_float(
    integral: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> float:
    return float(
        integral[r1, c1]
        - integral[r0, c1]
        - integral[r1, c0]
        + integral[r0, c0]
    )


def _prepare_weight_grid(
    values: np.ndarray | None,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    if values.shape != shape:
        raise ValueError(f"{name} shape must match grid mask shape")
    arr = values.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    return np.clip(arr, 0.0, None)


def _target_bbox(target: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(target)
    if len(rows) == 0:
        return 0, 0, 0, 0
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _size_ladder(max_size: int, min_size: int) -> list[int]:
    values: set[int] = {1, min_size}
    cur = min_size
    while cur < max_size:
        values.add(cur)
        values.add(max(min_size, int(round(cur * 1.5))))
        cur *= 2
    values.add(max_size)
    return sorted(v for v in values if 1 <= v <= max_size)


def _largest_exact_rectangle(mask: np.ndarray) -> RectilinearRectangle | None:
    h, w = mask.shape
    heights = [0] * w
    best: tuple[int, int, int, int, int] | None = None
    for r in range(h):
        for c in range(w):
            heights[c] = heights[c] + 1 if mask[r, c] else 0
        stack: list[int] = []
        for c in range(w + 1):
            cur_h = heights[c] if c < w else 0
            while stack and cur_h < heights[stack[-1]]:
                height = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                width = c - left
                area = height * width
                if area > 0 and (best is None or area > best[0]):
                    best = (area, r - height + 1, left, r + 1, c)
            stack.append(c)
    if best is None:
        return None
    area, r0, c0, r1, c1 = best
    return RectilinearRectangle(
        row0=r0,
        col0=c0,
        row1=r1,
        col1=c1,
        target_cells=area,
        total_cells=area,
        fill_ratio=1.0,
    )
