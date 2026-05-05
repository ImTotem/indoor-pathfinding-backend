"""RTAB-Map Node/Link trajectory -> walkable road polygon/grid."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import atan2, degrees, sqrt
from typing import Any

import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

from indoor_server.domain.building.models import GridOrigin, WalkableGrid
from indoor_server.domain.building.rtabmap_models import (
    RtabmapFeaturePoint,
    RtabmapLink,
    RtabmapNode,
)

logger = logging.getLogger(__name__)

_DEFAULT_CELL_SIZE_M = 0.10
_DEFAULT_HALF_WIDTH_M = 0.75
_MIN_SEGMENT_M = 0.05


@dataclass(frozen=True)
class RtabmapTrajectoryRoadResult:
    grid: WalkableGrid
    footprint_geojson: dict[str, object] | None
    z0: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class FeatureCloudRoadEvidence:
    transformed_count: int
    accepted_count: int
    estimated_half_width_m: float | None
    distance_percentiles_m: dict[str, float]

    def metadata(self) -> dict[str, object]:
        return {
            "transformed_count": self.transformed_count,
            "accepted_count": self.accepted_count,
            "estimated_half_width_m": self.estimated_half_width_m,
            "distance_percentiles_m": self.distance_percentiles_m,
        }


class RtabmapTrajectoryRoadStep:
    """Build a map-style corridor polygon from RTAB-Map graph trajectory.

    This is intentionally trajectory-first. It does not use keyframe jpg files
    or raw ARKit pose_matrix. The output is still a grid so the existing
    skeleton, route graph, POI spur, and IMDF pipeline can be reused.
    """

    def __init__(
        self,
        *,
        half_width_m: float = _DEFAULT_HALF_WIDTH_M,
        cell_size_m: float = _DEFAULT_CELL_SIZE_M,
        use_feature_evidence: bool = False,
        feature_width_percentile: float = 75.0,
        feature_min_count: int = 200,
        feature_max_distance_m: float = 3.0,
        feature_min_half_width_m: float = 0.6,
        feature_max_half_width_m: float = 1.4,
        rectify_centerline: bool = True,
        centerline_simplify_tolerance_m: float = 0.35,
        centerline_min_segment_m: float = 0.25,
    ) -> None:
        if half_width_m <= 0:
            raise ValueError(f"half_width_m must be > 0, got {half_width_m}")
        if cell_size_m <= 0:
            raise ValueError(f"cell_size_m must be > 0, got {cell_size_m}")
        if not (0.0 < feature_width_percentile < 100.0):
            raise ValueError("feature_width_percentile must be in (0, 100)")
        self._half_width_m = half_width_m
        self._cell_size_m = cell_size_m
        self._use_feature_evidence = use_feature_evidence
        self._feature_width_percentile = feature_width_percentile
        self._feature_min_count = feature_min_count
        self._feature_max_distance_m = feature_max_distance_m
        self._feature_min_half_width_m = feature_min_half_width_m
        self._feature_max_half_width_m = feature_max_half_width_m
        self._rectify_centerline = rectify_centerline
        self._centerline_simplify_tolerance_m = centerline_simplify_tolerance_m
        self._centerline_min_segment_m = centerline_min_segment_m

    def run(
        self,
        *,
        nodes: list[RtabmapNode],
        links: list[RtabmapLink],
        features: list[RtabmapFeaturePoint] | None = None,
    ) -> RtabmapTrajectoryRoadResult:
        if len(nodes) < 2:
            return self._empty_result(
                z0=0.0,
                metadata={
                    "source": "rtabmap_node_link",
                    "node_count": len(nodes),
                    "link_count": len(links),
                    "edge_count": 0,
                    "half_width_m": self._half_width_m,
                    "cell_size_m": self._cell_size_m,
                    "issues": ["node_count_below_min"],
                },
            )

        frame = _FloorFrame.from_nodes(nodes)
        node_xy = {
            node.node_id: frame.project_node_xy(node)
            for node in nodes
        }
        node_z = [frame.project_node_z(node) for node in nodes]
        z0 = float(np.median(node_z)) if node_z else 0.0

        segments = self._build_segments(nodes, links, node_xy)
        if not segments:
            return self._empty_result(
                z0=z0,
                metadata={
                    "source": "rtabmap_node_link",
                    "node_count": len(nodes),
                    "link_count": len(links),
                    "edge_count": 0,
                    "half_width_m": self._half_width_m,
                    "cell_size_m": self._cell_size_m,
                    "floor_frame": frame.metadata(),
                    "issues": ["no_valid_trajectory_segments"],
                },
            )

        dominant_raw, dominant_angle = _dominant_angle_from_segments(segments)
        raw_multiline = MultiLineString(segments)
        buffer_segments, centerline_metadata = self._rectified_centerline_segments(
            nodes=nodes,
            node_xy=node_xy,
            raw_segments=segments,
            dominant_angle_deg=dominant_angle,
        )
        multiline = MultiLineString(buffer_segments)
        feature_evidence = (
            self._estimate_feature_evidence(
                nodes=nodes,
                features=features or [],
                frame=frame,
                trajectory=raw_multiline,
            )
            if self._use_feature_evidence and features
            else None
        )
        resolved_half_width_m = self._resolve_half_width(feature_evidence)
        buffered = multiline.buffer(
            resolved_half_width_m,
            cap_style=2,   # flat caps: map/cad-like road ends
            join_style=2,  # mitre joins: angular corners, not round blobs
        )
        polygons = _as_polygons(buffered)
        if not polygons:
            return self._empty_result(
                z0=z0,
                metadata={
                    "source": "rtabmap_node_link",
                    "node_count": len(nodes),
                    "link_count": len(links),
                    "edge_count": len(buffer_segments),
                    "raw_edge_count": len(segments),
                    "half_width_m": self._half_width_m,
                    "cell_size_m": self._cell_size_m,
                    "floor_frame": frame.metadata(),
                    "issues": ["empty_buffer_polygon"],
                },
            )

        footprint = MultiPolygon(polygons)
        grid = self._rasterize(polygons, z0)
        footprint_geojson: dict[str, object] = dict(mapping(footprint))
        metadata = {
            "source": "rtabmap_node_link",
            "node_count": len(nodes),
            "link_count": len(links),
            "edge_count": len(buffer_segments),
            "raw_edge_count": len(segments),
            "half_width_m": resolved_half_width_m,
            "base_half_width_m": self._half_width_m,
            "resolved_half_width_m": resolved_half_width_m,
            "cell_size_m": self._cell_size_m,
            "floor_frame": frame.metadata(),
            "dominant_angle_raw_deg": dominant_raw,
            "dominant_angle_deg": dominant_angle,
            "dominant_angle_source": "rtabmap_link_segments",
            "feature_evidence_enabled": self._use_feature_evidence,
            "walkable_cells": int(grid.mask.sum()),
            "area_m2": float(footprint.area),
            "bounds": [float(v) for v in footprint.bounds],
        }
        if centerline_metadata is not None:
            metadata["centerline_rectification"] = centerline_metadata
        if feature_evidence is not None:
            metadata["feature_evidence"] = feature_evidence.metadata()
        logger.info(
            "rtabmap_trajectory road done nodes=%d links=%d segments=%d "
            "half_width=%.3f cells=%d area=%.3f",
            len(nodes),
            len(links),
            len(buffer_segments),
            resolved_half_width_m,
            int(grid.mask.sum()),
            float(footprint.area),
        )
        return RtabmapTrajectoryRoadResult(
            grid=grid,
            footprint_geojson=footprint_geojson,
            z0=z0,
            metadata=metadata,
        )

    def _rectified_centerline_segments(
        self,
        *,
        nodes: list[RtabmapNode],
        node_xy: dict[int, tuple[float, float]],
        raw_segments: list[tuple[tuple[float, float], tuple[float, float]]],
        dominant_angle_deg: float,
    ) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], dict[str, object] | None]:
        if not self._rectify_centerline:
            return raw_segments, None

        ordered = [node_xy[node.node_id] for node in sorted(nodes, key=lambda n: n.node_id)]
        rectified = _rectilinearize_centerline(
            ordered,
            dominant_angle_deg=dominant_angle_deg,
            simplify_tolerance_m=self._centerline_simplify_tolerance_m,
            min_segment_m=self._centerline_min_segment_m,
        )
        if len(rectified) < 2:
            return raw_segments, {
                "enabled": True,
                "accepted": False,
                "reason": "rectified_centerline_too_short",
                "raw_point_count": len(ordered),
            }

        segments = [
            (a, b)
            for a, b in zip(rectified, rectified[1:], strict=False)
            if _dist(a, b) >= _MIN_SEGMENT_M
        ]
        if not segments:
            return raw_segments, {
                "enabled": True,
                "accepted": False,
                "reason": "rectified_segments_empty",
                "raw_point_count": len(ordered),
                "rectified_point_count": len(rectified),
            }

        return segments, {
            "enabled": True,
            "accepted": True,
            "dominant_angle_deg": dominant_angle_deg,
            "raw_point_count": len(ordered),
            "rectified_point_count": len(rectified),
            "raw_segment_count": len(raw_segments),
            "rectified_segment_count": len(segments),
            "rectified_points": [[float(x), float(y)] for x, y in rectified],
            "simplify_tolerance_m": self._centerline_simplify_tolerance_m,
            "min_segment_m": self._centerline_min_segment_m,
        }

    def _estimate_feature_evidence(
        self,
        *,
        nodes: list[RtabmapNode],
        features: list[RtabmapFeaturePoint],
        frame: _FloorFrame,
        trajectory: MultiLineString,
    ) -> FeatureCloudRoadEvidence:
        from shapely.geometry import Point

        node_pose = {
            node.node_id: np.asarray(node.pose, dtype=np.float64)
            for node in nodes
        }
        distances: list[float] = []
        transformed_count = 0
        for feature in features:
            pose = node_pose.get(feature.node_id)
            if pose is None:
                continue
            local = np.asarray([
                feature.local_xyz[0],
                feature.local_xyz[1],
                feature.local_xyz[2],
                1.0,
            ], dtype=np.float64)
            world = pose @ local
            xy = frame.project_xyz_xy(world[:3])
            transformed_count += 1
            distance = float(trajectory.distance(Point(xy)))
            if distance <= self._feature_max_distance_m:
                distances.append(distance)

        if len(distances) < self._feature_min_count:
            return FeatureCloudRoadEvidence(
                transformed_count=transformed_count,
                accepted_count=len(distances),
                estimated_half_width_m=None,
                distance_percentiles_m={},
            )

        arr = np.asarray(distances, dtype=np.float64)
        percentiles = {
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }
        estimate = float(np.percentile(arr, self._feature_width_percentile))
        estimate = float(np.clip(
            estimate,
            self._feature_min_half_width_m,
            self._feature_max_half_width_m,
        ))
        return FeatureCloudRoadEvidence(
            transformed_count=transformed_count,
            accepted_count=len(distances),
            estimated_half_width_m=estimate,
            distance_percentiles_m=percentiles,
        )

    def _resolve_half_width(
        self,
        feature_evidence: FeatureCloudRoadEvidence | None,
    ) -> float:
        if feature_evidence is None or feature_evidence.estimated_half_width_m is None:
            return self._half_width_m
        return max(self._half_width_m, feature_evidence.estimated_half_width_m)

    def _build_segments(
        self,
        nodes: list[RtabmapNode],
        links: list[RtabmapLink],
        node_xy: dict[int, tuple[float, float]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        seen: set[tuple[int, int]] = set()
        for link in links:
            if link.link_type != 0:
                continue
            if link.from_id not in node_xy or link.to_id not in node_xy:
                continue
            key = (
                min(link.from_id, link.to_id),
                max(link.from_id, link.to_id),
            )
            if key in seen or key[0] == key[1]:
                continue
            seen.add(key)
            a = node_xy[link.from_id]
            b = node_xy[link.to_id]
            if _dist(a, b) >= _MIN_SEGMENT_M:
                segments.append((a, b))

        if segments:
            return segments

        ordered = sorted(nodes, key=lambda node: node.node_id)
        for a_node, b_node in zip(ordered, ordered[1:], strict=False):
            a = node_xy[a_node.node_id]
            b = node_xy[b_node.node_id]
            if _dist(a, b) >= _MIN_SEGMENT_M:
                segments.append((a, b))
        return segments

    def _rasterize(self, polygons: list[Polygon], z0: float) -> WalkableGrid:
        import cv2

        union = unary_union(polygons)
        min_x, min_y, max_x, max_y = [float(v) for v in union.bounds]
        cs = self._cell_size_m
        pad = cs * 2.0
        x0 = min_x - pad
        y0 = min_y - pad
        w = max(1, int(np.ceil((max_x - min_x + pad * 2.0) / cs)) + 1)
        h = max(1, int(np.ceil((max_y - min_y + pad * 2.0) / cs)) + 1)

        buf = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            exterior = np.array(
                [
                    [int(round((x - x0) / cs)), int(round((y - y0) / cs))]
                    for x, y in poly.exterior.coords
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(buf, [exterior], color=1)
            for interior in poly.interiors:
                hole = np.array(
                    [
                        [int(round((x - x0) / cs)), int(round((y - y0) / cs))]
                        for x, y in interior.coords
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(buf, [hole], color=0)

        origin = GridOrigin(x0=x0, y0=y0, z0=z0, cell_size=cs, w=w, h=h)
        return WalkableGrid(
            origin=origin,
            mask=buf.astype(bool),
            observation_count=buf.astype(np.uint16),
        )

    def _empty_result(
        self,
        *,
        z0: float,
        metadata: dict[str, object],
    ) -> RtabmapTrajectoryRoadResult:
        origin = GridOrigin(x0=0.0, y0=0.0, z0=z0, cell_size=self._cell_size_m, w=1, h=1)
        grid = WalkableGrid(
            origin=origin,
            mask=np.zeros((1, 1), dtype=bool),
            observation_count=np.zeros((1, 1), dtype=np.uint16),
        )
        return RtabmapTrajectoryRoadResult(
            grid=grid,
            footprint_geojson=None,
            z0=z0,
            metadata=metadata,
        )


@dataclass(frozen=True)
class _FloorFrame:
    horizontal_axes: tuple[int, int]
    vertical_axis: int

    @classmethod
    def from_nodes(cls, nodes: list[RtabmapNode]) -> _FloorFrame:
        translations = np.array([
            [node.pose[0][3], node.pose[1][3], node.pose[2][3]]
            for node in nodes
        ], dtype=np.float64)
        variances = np.var(translations, axis=0)
        vertical_axis = int(np.argmin(variances))
        horizontal_axes = tuple(
            int(axis)
            for axis in np.argsort(variances)[::-1]
            if int(axis) != vertical_axis
        )
        return cls(
            horizontal_axes=(horizontal_axes[0], horizontal_axes[1]),
            vertical_axis=vertical_axis,
        )

    def project_node_xy(self, node: RtabmapNode) -> tuple[float, float]:
        values = (node.pose[0][3], node.pose[1][3], node.pose[2][3])
        return (
            float(values[self.horizontal_axes[0]]),
            float(values[self.horizontal_axes[1]]),
        )

    def project_xyz_xy(self, values: np.ndarray) -> tuple[float, float]:
        return (
            float(values[self.horizontal_axes[0]]),
            float(values[self.horizontal_axes[1]]),
        )

    def project_node_z(self, node: RtabmapNode) -> float:
        values = (node.pose[0][3], node.pose[1][3], node.pose[2][3])
        return float(values[self.vertical_axis])

    def metadata(self) -> dict[str, Any]:
        names = ["x", "y", "z"]
        return {
            "horizontal_axes": [
                names[self.horizontal_axes[0]],
                names[self.horizontal_axes[1]],
            ],
            "vertical_axis": names[self.vertical_axis],
        }


def _as_polygons(geom: object) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _rectilinearize_centerline(
    points: list[tuple[float, float]],
    *,
    dominant_angle_deg: float,
    simplify_tolerance_m: float,
    min_segment_m: float,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    rotated = _rotate_points(points, -dominant_angle_deg)
    simplified = _rdp(rotated, simplify_tolerance_m)
    if len(simplified) < 2:
        simplified = rotated

    rectified: list[tuple[float, float]] = [simplified[0]]
    for target in simplified[1:]:
        current = rectified[-1]
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        if abs(dx) >= abs(dy):
            next_pt = (target[0], current[1])
        else:
            next_pt = (current[0], target[1])
        if _dist(current, next_pt) >= min_segment_m:
            rectified.append(next_pt)

    if len(rectified) < 2:
        return points
    return _rotate_points(rectified, dominant_angle_deg)


def _rdp(
    points: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    if tolerance <= 0.0 or len(points) <= 2:
        return list(points)

    start = points[0]
    end = points[-1]
    max_distance = -1.0
    split_index = 0
    for idx, point in enumerate(points[1:-1], start=1):
        distance = _point_line_distance(point, start, end)
        if distance > max_distance:
            max_distance = distance
            split_index = idx

    if max_distance <= tolerance:
        return [start, end]
    left = _rdp(points[: split_index + 1], tolerance)
    right = _rdp(points[split_index:], tolerance)
    return left[:-1] + right


def _point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denom = sqrt(dx * dx + dy * dy)
    if denom < 1e-9:
        return _dist(point, start)
    return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / denom


def _rotate_points(
    points: list[tuple[float, float]],
    angle_deg: float,
) -> list[tuple[float, float]]:
    theta = np.deg2rad(angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def _dominant_angle_from_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    bin_deg: float = 5.0,
) -> tuple[float, float]:
    """Estimate the building axis offset from RTAB-Map trajectory segments.

    Segment directions that differ by 90 degrees represent the same Manhattan
    frame, so each segment angle is folded into (-45, 45]. A length-weighted
    histogram keeps long corridor runs dominant over small jitter edges.
    """
    if not segments:
        return 0.0, 0.0

    n_bins = max(1, int(round(90.0 / bin_deg)))
    weights = np.zeros(n_bins, dtype=np.float64)
    angle_sum = np.zeros(n_bins, dtype=np.float64)
    for (ax, ay), (bx, by) in segments:
        dx = bx - ax
        dy = by - ay
        length = sqrt(dx * dx + dy * dy)
        if length < _MIN_SEGMENT_M:
            continue
        angle = _fold_axis_angle(degrees(atan2(dy, dx)))
        idx = int(np.floor((angle + 45.0) / bin_deg))
        idx = max(0, min(n_bins - 1, idx))
        weights[idx] += length
        angle_sum[idx] += angle * length

    if weights.max() < 1e-9:
        return 0.0, 0.0
    best_idx = int(np.argmax(weights))
    raw = float(angle_sum[best_idx] / weights[best_idx])
    return raw, _fold_axis_angle(raw)


def _fold_axis_angle(angle_deg: float) -> float:
    """Fold an arbitrary edge angle into the Manhattan-frame offset range."""
    angle = angle_deg % 90.0
    if angle > 45.0:
        angle -= 90.0
    if angle <= -45.0:
        angle += 90.0
    return float(angle)
