"""Sprint 51 — Step 6: polygon assembly.

4-tier fallback chain:
  Primary  — corner clustering + planar graph cycle finding (networkx.cycle_basis).
  Secondary — Manhattan pair fallback from close parallel wall-line pairs.
  Tertiary  — alpha shape (`shapely.concave_hull`) over wall_line endpoints.
  Final     — convex hull (raises corner_orthogonality_ratio reject signal).

corner_orthogonality_ratio = 90° 내각 / 전체 내각, tolerance 5°.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
from shapely import affinity
from shapely.geometry import (
    MultiPoint,
    Polygon,
    mapping,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from indoor_server.application.building.steps.wall_polygon.merge import (
    WallLine,
    WallLineSet,
)


@dataclass(frozen=True)
class PolygonAssemblyParams:
    intersection_tolerance_m: float = 0.30
    corner_max_distance_m: float = 0.50
    min_cycle_length: int = 4
    use_manhattan_pair_fallback: bool = True
    pair_min_width_m: float = 0.50
    pair_max_width_m: float = 3.00
    pair_min_overlap_m: float = 0.50
    pair_min_overlap_ratio: float = 0.25
    use_alpha_shape_fallback: bool = True
    alpha_shape_alpha_m: float = 1.5
    orthogonality_angle_tol_deg: float = 5.0

    def to_metadata(self) -> dict[str, object]:
        return {
            "intersection_tolerance_m": self.intersection_tolerance_m,
            "corner_max_distance_m": self.corner_max_distance_m,
            "min_cycle_length": self.min_cycle_length,
            "use_manhattan_pair_fallback": self.use_manhattan_pair_fallback,
            "pair_min_width_m": self.pair_min_width_m,
            "pair_max_width_m": self.pair_max_width_m,
            "pair_min_overlap_m": self.pair_min_overlap_m,
            "pair_min_overlap_ratio": self.pair_min_overlap_ratio,
            "use_alpha_shape_fallback": self.use_alpha_shape_fallback,
            "alpha_shape_alpha_m": self.alpha_shape_alpha_m,
            "orthogonality_angle_tol_deg": self.orthogonality_angle_tol_deg,
        }


@dataclass(frozen=True)
class AssembledPolygon:
    polygon_geojson: dict[str, object] | None
    vertex_count: int
    corner_orthogonality_ratio: float
    self_intersection_count: int
    fallback_used: str | None
    metadata: dict[str, object] = field(default_factory=dict)


def _line_intersection(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> tuple[float, float] | None:
    """Compute intersection of infinite lines (a1->a2) and (b1->b2). None if parallel."""
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_num / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return (px, py)


def _project_param(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Parametric position of point p on infinite line a→b. 0 = a, 1 = b."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom <= 0:
        return 0.0
    return ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom


def _on_segment(t: float, tol: float = 0.0) -> bool:
    return -tol <= t <= 1.0 + tol


def _cluster_corners(
    points: list[tuple[float, float]],
    tol_m: float,
) -> tuple[list[tuple[float, float]], list[int]]:
    """Greedy clustering of corner candidates. Returns (centroids, mapping)."""
    centroids: list[tuple[float, float]] = []
    mapping_idx: list[int] = []
    for p in points:
        best_idx = -1
        best_d = float("inf")
        for ci, c in enumerate(centroids):
            d = math.hypot(p[0] - c[0], p[1] - c[1])
            if d <= tol_m and d < best_d:
                best_d = d
                best_idx = ci
        if best_idx >= 0:
            mapping_idx.append(best_idx)
        else:
            centroids.append(p)
            mapping_idx.append(len(centroids) - 1)
    return centroids, mapping_idx


def _polygon_self_intersections(polygon_geojson: dict[str, object]) -> int:
    """Count self-intersections via shapely is_simple."""
    try:
        geom = shape(polygon_geojson)
        if geom.is_empty:
            return 0
        if geom.is_valid and geom.is_simple:
            return 0
        # if it's invalid, count via explain_validity-like heuristic.
        return 1
    except Exception:
        return 1


def _corner_orthogonality_ratio(
    polygon_geojson: dict[str, object],
    tol_deg: float,
) -> float:
    """ratio of vertices whose interior angle is within ±tol_deg of 90°."""
    try:
        geom = shape(polygon_geojson)
    except Exception:
        return 0.0
    polys: list[Polygon] = []
    if geom.geom_type == "Polygon":
        polys.append(geom)
    elif geom.geom_type == "MultiPolygon":
        polys.extend(list(geom.geoms))
    else:
        return 0.0

    total = 0
    ok = 0
    for poly in polys:
        coords = list(poly.exterior.coords)
        if len(coords) < 4:  # closed ring needs >= 4 coords
            continue
        # remove duplicate closing coord
        coords = coords[:-1]
        n = len(coords)
        for i in range(n):
            a = coords[(i - 1) % n]
            b = coords[i]
            c = coords[(i + 1) % n]
            v1 = (a[0] - b[0], a[1] - b[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            if n1 <= 1e-9 or n2 <= 1e-9:
                continue
            cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
            cos_a = max(-1.0, min(1.0, cos_a))
            angle_deg = math.degrees(math.acos(cos_a))
            total += 1
            if abs(angle_deg - 90.0) <= tol_deg:
                ok += 1
    if total == 0:
        return 0.0
    return float(ok) / float(total)


def _polygon_vertex_count(polygon_geojson: dict[str, object]) -> int:
    try:
        geom = shape(polygon_geojson)
    except Exception:
        return 0
    polys: list[Polygon] = []
    if geom.geom_type == "Polygon":
        polys.append(geom)
    elif geom.geom_type == "MultiPolygon":
        polys.extend(list(geom.geoms))
    total = 0
    for poly in polys:
        # exterior + interior rings, dedupe closing coord
        ext = list(poly.exterior.coords)
        if ext and ext[0] == ext[-1]:
            ext = ext[:-1]
        total += len(ext)
        for ring in poly.interiors:
            r = list(ring.coords)
            if r and r[0] == r[-1]:
                r = r[:-1]
            total += len(r)
    return total


def _try_primary_assembly(
    lines: list[WallLine],
    params: PolygonAssemblyParams,
) -> tuple[Polygon | None, dict[str, object]]:
    """Primary path: corner clustering + cycle basis."""
    info: dict[str, object] = {
        "candidate_count": 0,
        "corner_count": 0,
        "primary_cycle_count": 0,
    }
    if len(lines) < params.min_cycle_length:
        info["skip_reason"] = "too_few_lines"
        return None, info

    # Collect corner candidates: every endpoint + every line-line intersection.
    candidates: list[tuple[float, float]] = []
    for line in lines:
        candidates.append((line.world_x1, line.world_y1))
        candidates.append((line.world_x2, line.world_y2))

    for i, la in enumerate(lines):
        a1 = (la.world_x1, la.world_y1)
        a2 = (la.world_x2, la.world_y2)
        for j in range(i + 1, len(lines)):
            lb = lines[j]
            if la.axis_id == lb.axis_id:
                continue
            b1 = (lb.world_x1, lb.world_y1)
            b2 = (lb.world_x2, lb.world_y2)
            inter = _line_intersection(a1, a2, b1, b2)
            if inter is None:
                continue
            ta = _project_param(inter, a1, a2)
            tb = _project_param(inter, b1, b2)
            tol_a = params.intersection_tolerance_m / max(la.length_m, 1e-6)
            tol_b = params.intersection_tolerance_m / max(lb.length_m, 1e-6)
            if not _on_segment(ta, tol_a) or not _on_segment(tb, tol_b):
                continue
            candidates.append(inter)
    info["candidate_count"] = len(candidates)

    if not candidates:
        info["skip_reason"] = "no_corners"
        return None, info

    centroids, _mapping_idx = _cluster_corners(candidates, params.corner_max_distance_m)
    info["corner_count"] = len(centroids)
    if len(centroids) < params.min_cycle_length:
        info["skip_reason"] = "too_few_corners"
        return None, info

    # Build planar graph.
    # For each line, collect corners that lie on it (within corner_max_distance_m).
    # Then sort by parametric position and connect consecutive pairs.
    graph: nx.Graph = nx.Graph()
    for ci in range(len(centroids)):
        graph.add_node(ci)

    for line in lines:
        a = (line.world_x1, line.world_y1)
        b = (line.world_x2, line.world_y2)
        on_line: list[tuple[float, int]] = []
        for ci, c in enumerate(centroids):
            t = _project_param(c, a, b)
            # distance from line: project then residual
            px = a[0] + t * (b[0] - a[0])
            py = a[1] + t * (b[1] - a[1])
            dist = math.hypot(c[0] - px, c[1] - py)
            if dist <= params.corner_max_distance_m and _on_segment(
                t, params.corner_max_distance_m / max(line.length_m, 1e-6)
            ):
                on_line.append((t, ci))
        on_line.sort()
        for k in range(len(on_line) - 1):
            i_a = on_line[k][1]
            i_b = on_line[k + 1][1]
            if i_a != i_b:
                graph.add_edge(i_a, i_b)

    if graph.number_of_edges() == 0:
        info["skip_reason"] = "no_graph_edges"
        return None, info

    cycles = nx.cycle_basis(graph)
    info["primary_cycle_count"] = len(cycles)
    if not cycles:
        info["skip_reason"] = "no_cycles"
        return None, info

    # pick the cycle with largest area
    best_poly: Polygon | None = None
    best_area = 0.0
    for cycle in cycles:
        if len(cycle) < params.min_cycle_length:
            continue
        ring = [centroids[i] for i in cycle]
        ring.append(ring[0])
        try:
            poly = Polygon(ring)
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        area_m2 = float(poly.area)
        if area_m2 > best_area:
            best_area = area_m2
            best_poly = poly
    if best_poly is None or best_area <= 0:
        info["skip_reason"] = "empty_cycles"
        return None, info
    info["chosen_outer_area_m2"] = float(best_area)
    return best_poly, info


def _try_alpha_shape(
    lines: list[WallLine],
    params: PolygonAssemblyParams,
) -> Polygon | None:
    """Alpha shape (concave hull) fallback. shapely 2.0+ provides concave_hull."""
    pts: list[tuple[float, float]] = []
    for line in lines:
        pts.append((line.world_x1, line.world_y1))
        pts.append((line.world_x2, line.world_y2))
    if len(pts) < 4:
        return None
    mp = MultiPoint(pts)
    try:
        # shapely.concave_hull alpha is 0..1 ratio; lower = tighter concavity.
        ratio = float(min(0.95, max(0.05, 1.0 / max(params.alpha_shape_alpha_m, 0.5))))
        hull = mp.concave_hull(ratio=ratio)
    except Exception:
        try:
            hull = mp.convex_hull
        except Exception:
            return None
    if hull is None or hull.is_empty:
        return None
    if hull.geom_type != "Polygon":
        return None
    return hull


def _overlap_length(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _axis_rect_to_world(
    coords: list[tuple[float, float]],
    dominant_angle_deg: float,
) -> Polygon:
    """Build a world-frame polygon from dominant-axis `(u, v)` coordinates."""
    poly = Polygon(coords)
    if abs(dominant_angle_deg) < 1e-9:
        return poly
    rotated = affinity.rotate(
        poly,
        dominant_angle_deg,
        origin=(0.0, 0.0),
        use_radians=False,
    )
    if isinstance(rotated, Polygon):
        return rotated
    return poly


def _try_manhattan_pair_fallback(
    line_set: WallLineSet,
    params: PolygonAssemblyParams,
) -> tuple[BaseGeometry | None, dict[str, object]]:
    """Build corridor rectangles from close parallel wall-line pairs.

    This fallback is intentionally stricter than alpha-shape: it only accepts
    pairs of parallel snapped lines whose spacing looks like an indoor corridor
    width and whose along-axis spans substantially overlap. Each pair becomes a
    rectilinear rectangle in the dominant-angle frame; the final polygon is the
    union of those rectangles.
    """
    info: dict[str, object] = {
        "manhattan_pair_candidate_count": 0,
        "manhattan_pair_selected_count": 0,
    }
    rectangles: list[Polygon] = []
    candidate_count = 0
    selected_count = 0

    for axis_id in (0, 1):
        axis_lines = [line for line in line_set.lines if line.axis_id == axis_id]
        axis_lines.sort(key=lambda line: line.offset_m)
        for i, first in enumerate(axis_lines):
            for second in axis_lines[i + 1 :]:
                width = abs(second.offset_m - first.offset_m)
                if width < params.pair_min_width_m:
                    continue
                if width > params.pair_max_width_m:
                    break
                candidate_count += 1

                overlap = _overlap_length(
                    first.start_along_axis_m,
                    first.end_along_axis_m,
                    second.start_along_axis_m,
                    second.end_along_axis_m,
                )
                min_len = max(1e-6, min(first.length_m, second.length_m))
                overlap_ratio = overlap / min_len
                if overlap < params.pair_min_overlap_m:
                    continue
                if overlap_ratio < params.pair_min_overlap_ratio:
                    continue

                along_start = min(first.start_along_axis_m, second.start_along_axis_m)
                along_end = max(first.end_along_axis_m, second.end_along_axis_m)
                off_min = min(first.offset_m, second.offset_m)
                off_max = max(first.offset_m, second.offset_m)

                if axis_id == 0:
                    rect_coords = [
                        (along_start, off_min),
                        (along_end, off_min),
                        (along_end, off_max),
                        (along_start, off_max),
                    ]
                else:
                    rect_coords = [
                        (off_min, along_start),
                        (off_max, along_start),
                        (off_max, along_end),
                        (off_min, along_end),
                    ]
                rectangles.append(
                    _axis_rect_to_world(rect_coords, line_set.dominant_angle_deg)
                )
                selected_count += 1

    info["manhattan_pair_candidate_count"] = candidate_count
    info["manhattan_pair_selected_count"] = selected_count

    if not rectangles:
        info["manhattan_pair_skip_reason"] = "no_parallel_pairs"
        return None, info

    try:
        union = unary_union(rectangles)
    except Exception:
        info["manhattan_pair_skip_reason"] = "union_failed"
        return None, info
    if union.is_empty or union.area <= 0:
        info["manhattan_pair_skip_reason"] = "empty_union"
        return None, info
    if union.geom_type not in ("Polygon", "MultiPolygon"):
        info["manhattan_pair_skip_reason"] = "non_polygon_union"
        return None, info

    info["manhattan_pair_area_m2"] = float(union.area)
    return union, info


def _convex_hull_fallback(lines: list[WallLine]) -> Polygon | None:
    pts: list[tuple[float, float]] = []
    for line in lines:
        pts.append((line.world_x1, line.world_y1))
        pts.append((line.world_x2, line.world_y2))
    if len(pts) < 3:
        return None
    mp = MultiPoint(pts)
    hull = mp.convex_hull
    if hull is None or hull.is_empty or hull.geom_type != "Polygon":
        return None
    return hull


class PolygonAssemblyStep:
    """Step 6 — polygon assembly with primary cycle + 3 fallback tiers."""

    def __init__(self, params: PolygonAssemblyParams | None = None) -> None:
        self._params = params if params is not None else PolygonAssemblyParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.intersection_tolerance_m < 0:
            raise ValueError("intersection_tolerance_m must be >= 0")
        if p.corner_max_distance_m < 0:
            raise ValueError("corner_max_distance_m must be >= 0")
        if p.min_cycle_length < 3:
            raise ValueError("min_cycle_length must be >= 3")
        if p.pair_min_width_m < 0:
            raise ValueError("pair_min_width_m must be >= 0")
        if p.pair_max_width_m <= 0:
            raise ValueError("pair_max_width_m must be > 0")
        if p.pair_max_width_m < p.pair_min_width_m:
            raise ValueError("pair_max_width_m must be >= pair_min_width_m")
        if p.pair_min_overlap_m < 0:
            raise ValueError("pair_min_overlap_m must be >= 0")
        if not (0.0 <= p.pair_min_overlap_ratio <= 1.0):
            raise ValueError("pair_min_overlap_ratio must be in [0, 1]")

    def run(self, lines: WallLineSet) -> AssembledPolygon:
        params = self._params
        info: dict[str, object] = {"params": params.to_metadata()}

        if not lines.lines:
            return AssembledPolygon(
                polygon_geojson=None,
                vertex_count=0,
                corner_orthogonality_ratio=0.0,
                self_intersection_count=0,
                fallback_used=None,
                metadata={
                    **info,
                    "primary_cycle_count": 0,
                    "chosen_outer_area_m2": 0.0,
                    "vertex_count_after_assembly": 0,
                    "self_intersection_count": 0,
                    "skip_reason": "no_lines",
                },
            )

        polygon: BaseGeometry | None
        polygon, primary_info = _try_primary_assembly(lines.lines, params)
        info.update(primary_info)
        fallback_used: str | None = None

        if polygon is None and params.use_manhattan_pair_fallback:
            polygon, pair_info = _try_manhattan_pair_fallback(lines, params)
            info.update(pair_info)
            if polygon is not None:
                fallback_used = "manhattan_pair"

        if polygon is None and params.use_alpha_shape_fallback:
            polygon = _try_alpha_shape(lines.lines, params)
            if polygon is not None:
                fallback_used = "alpha_shape"

        if polygon is None:
            polygon = _convex_hull_fallback(lines.lines)
            if polygon is not None:
                fallback_used = "convex_hull"

        if polygon is None:
            return AssembledPolygon(
                polygon_geojson=None,
                vertex_count=0,
                corner_orthogonality_ratio=0.0,
                self_intersection_count=0,
                fallback_used=None,
                metadata={
                    **info,
                    "primary_cycle_count": int(info.get("primary_cycle_count", 0) or 0),  # type: ignore[call-overload]
                    "chosen_outer_area_m2": 0.0,
                    "vertex_count_after_assembly": 0,
                    "self_intersection_count": 0,
                    "fallback_used": None,
                },
            )

        polygon_geojson: dict[str, object] = mapping(polygon)
        vertex_count = _polygon_vertex_count(polygon_geojson)
        ortho = _corner_orthogonality_ratio(
            polygon_geojson, params.orthogonality_angle_tol_deg
        )
        self_int = _polygon_self_intersections(polygon_geojson)
        metadata: dict[str, object] = {
            **info,
            "primary_cycle_count": int(info.get("primary_cycle_count", 0) or 0),  # type: ignore[call-overload]
            "chosen_outer_area_m2": float(polygon.area),
            "vertex_count_after_assembly": int(vertex_count),
            "corner_orthogonality_ratio": float(ortho),
            "self_intersection_count": int(self_int),
            "fallback_used": fallback_used,
        }
        return AssembledPolygon(
            polygon_geojson=polygon_geojson,
            vertex_count=int(vertex_count),
            corner_orthogonality_ratio=float(ortho),
            self_intersection_count=int(self_int),
            fallback_used=fallback_used,
            metadata=metadata,
        )
