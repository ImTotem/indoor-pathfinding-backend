"""Sprint 47 — Post-rectification polygon CAD cleanup.

ManhattanRectificationStep 산출 polygon을 CAD-style로 정리한다.
floor_pointcloud 모드 전용. trajectory/adaptive_buffer 등 5경로는 호출하지 않는다.

알고리즘 (순차):
  1. 인접 vertex 거리 < near_vertex_merge_distance_m → midpoint 합치기
  2. 짧은 edge (< short_edge_min_length_m) 끝 vertex 제거 (시작 vertex 보존)
  3. collinear vertex 제거 — 인접 3 vertex 가 (90 - tol) <= 내각 <= (180 + tol) 내에서
     180°에 가까우면 (즉 sin(angle_offset_from_180) < sin(tol)) 가운데 vertex 제거
  4. corner orthogonality metric 계산 — 90° 내각 / 전체 내각 비율
  5. shape-preservation guard 메트릭 — IoU(raw vs cleaned), centroid shift, bbox change ratio.
     guard 자체는 step에서 reject 하지 않고 metric으로 노출 — 호출자가 정책 결정.

Sprint 47 critique W-1, W-3, W-5, W-7 반영:
  - W-1: shape-preservation guard 메트릭 (IoU/centroid/bbox) 출력
  - W-3: orthogonality는 90°만 카운트, 180°는 collinear_residual_count로 분리
  - W-7: component_count_after, polygon_area_m2 가드용 메타 출력
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import atan2, degrees, sqrt

from shapely.geometry import MultiPolygon, mapping, shape
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolygonCadCleanupStepParams:
    """CAD cleanup step 파라미터.

    floor_pointcloud 모드 전용. 다른 경로에서는 호출되지 않는다.
    """

    enabled: bool = True
    collinear_angle_tol_deg: float = 5.0
    short_edge_min_length_m: float = 0.20
    near_vertex_merge_distance_m: float = 0.15
    orthogonality_angle_tol_deg: float = 5.0

    def to_metadata(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "collinear_angle_tol_deg": self.collinear_angle_tol_deg,
            "short_edge_min_length_m": self.short_edge_min_length_m,
            "near_vertex_merge_distance_m": self.near_vertex_merge_distance_m,
            "orthogonality_angle_tol_deg": self.orthogonality_angle_tol_deg,
        }


@dataclass(frozen=True)
class PolygonCadCleanupResult:
    """cleanup 산출물.

    cleaned_geojson은 cleanup 적용 후 GeoJSON이며, cleanup 결과가 비유효하면
    raw geometry를 그대로 반환하고 metadata.skipped_reason에 사유를 적는다.
    """

    cleaned_geojson: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)


class PolygonCadCleanupStep:
    """post-rectification polygon CAD cleanup."""

    def __init__(self, params: PolygonCadCleanupStepParams | None = None) -> None:
        self._params = params if params is not None else PolygonCadCleanupStepParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if p.collinear_angle_tol_deg < 0:
            raise ValueError("collinear_angle_tol_deg must be >= 0")
        if p.short_edge_min_length_m < 0:
            raise ValueError("short_edge_min_length_m must be >= 0")
        if p.near_vertex_merge_distance_m < 0:
            raise ValueError("near_vertex_merge_distance_m must be >= 0")
        if p.orthogonality_angle_tol_deg < 0:
            raise ValueError("orthogonality_angle_tol_deg must be >= 0")

    def run(self, footprint_geojson: dict[str, object]) -> PolygonCadCleanupResult:
        params = self._params
        if not params.enabled:
            return PolygonCadCleanupResult(
                cleaned_geojson=footprint_geojson,
                metadata={
                    "enabled": False,
                    "skipped_reason": "disabled",
                    "params": params.to_metadata(),
                },
            )

        try:
            raw_geom = shape(footprint_geojson)
        except Exception as e:
            logger.warning("polygon_cad_cleanup: shape parse failed: %s", e)
            return PolygonCadCleanupResult(
                cleaned_geojson=footprint_geojson,
                metadata={
                    "enabled": True,
                    "skipped_reason": f"shape_parse_failed:{type(e).__name__}",
                    "params": params.to_metadata(),
                },
            )

        if raw_geom.is_empty or raw_geom.area <= 0:
            return PolygonCadCleanupResult(
                cleaned_geojson=footprint_geojson,
                metadata={
                    "enabled": True,
                    "skipped_reason": "empty_geometry",
                    "params": params.to_metadata(),
                },
            )

        raw_polygons = _iter_polygons(raw_geom)
        if not raw_polygons:
            return PolygonCadCleanupResult(
                cleaned_geojson=footprint_geojson,
                metadata={
                    "enabled": True,
                    "skipped_reason": "no_polygons",
                    "params": params.to_metadata(),
                },
            )

        vertex_count_before = sum(_count_vertices(p) for p in raw_polygons)

        cleaned_polys: list[ShapelyPolygon] = []
        total_near_merged = 0
        total_short_pruned = 0
        total_collinear_merged = 0
        for poly in raw_polygons:
            cleaned, near_merged, short_pruned, collinear_merged = _cleanup_polygon(
                poly,
                near_vertex_distance_m=params.near_vertex_merge_distance_m,
                short_edge_min_length_m=params.short_edge_min_length_m,
                collinear_angle_tol_deg=params.collinear_angle_tol_deg,
            )
            total_near_merged += near_merged
            total_short_pruned += short_pruned
            total_collinear_merged += collinear_merged
            if cleaned is not None and not cleaned.is_empty and cleaned.area > 0:
                cleaned_polys.append(cleaned)

        if not cleaned_polys:
            return PolygonCadCleanupResult(
                cleaned_geojson=footprint_geojson,
                metadata={
                    "enabled": True,
                    "skipped_reason": "all_polygons_degenerated",
                    "vertex_count_before": vertex_count_before,
                    "vertex_count_after": vertex_count_before,
                    "params": params.to_metadata(),
                },
            )

        cleaned_geom = unary_union(cleaned_polys)
        if cleaned_geom.is_empty or cleaned_geom.area <= 0:
            return PolygonCadCleanupResult(
                cleaned_geojson=footprint_geojson,
                metadata={
                    "enabled": True,
                    "skipped_reason": "union_degenerated",
                    "params": params.to_metadata(),
                },
            )

        cleaned_polys_final = _iter_polygons(cleaned_geom)
        vertex_count_after = sum(_count_vertices(p) for p in cleaned_polys_final)

        # W-3: 90° 직각 corner 비율 + 180° collinear residual 별도 카운트
        right_angle_count, collinear_residual, total_corner_count = _count_corners(
            cleaned_polys_final,
            orthogonality_tol_deg=params.orthogonality_angle_tol_deg,
            collinear_tol_deg=params.collinear_angle_tol_deg,
        )
        if total_corner_count > 0:
            corner_orthogonality_ratio = float(right_angle_count) / float(total_corner_count)
        else:
            corner_orthogonality_ratio = 0.0

        edge_lengths = _all_edge_lengths(cleaned_polys_final)
        edge_p50 = _percentile(edge_lengths, 0.50) if edge_lengths else 0.0
        edge_p95 = _percentile(edge_lengths, 0.95) if edge_lengths else 0.0

        # W-1: shape-preservation guard 메트릭
        iou = _compute_iou(raw_geom, cleaned_geom)
        centroid_shift = _compute_centroid_shift(raw_geom, cleaned_geom)
        bbox_w_ratio, bbox_h_ratio = _compute_bbox_change_ratio(raw_geom, cleaned_geom)

        # W-7: minimum guard용 메트릭
        component_count_after = len(cleaned_polys_final)
        polygon_area_m2 = float(cleaned_geom.area)

        if vertex_count_before > 0:
            reduction_ratio = 1.0 - float(vertex_count_after) / float(vertex_count_before)
        else:
            reduction_ratio = 0.0

        # Sprint 48: cleanup_changed = 어떤 cleanup operation 이라도 적용되었는지.
        # small polygon 보호 (W-4) 분기에서 사용된다.
        cleanup_changed = bool(
            total_near_merged > 0
            or total_short_pruned > 0
            or total_collinear_merged > 0
            or vertex_count_after != vertex_count_before
        )

        metadata: dict[str, object] = {
            "enabled": True,
            "vertex_count_before": vertex_count_before,
            "vertex_count_after": vertex_count_after,
            "vertex_count_reduction_ratio": reduction_ratio,
            "cleanup_changed": cleanup_changed,
            "corner_orthogonality_ratio": corner_orthogonality_ratio,
            "right_angle_corner_count": right_angle_count,
            "collinear_residual_count": collinear_residual,
            "total_corner_count": total_corner_count,
            "edge_length_p50_m": edge_p50,
            "edge_length_p95_m": edge_p95,
            "near_vertices_merged_count": total_near_merged,
            "short_edges_pruned_count": total_short_pruned,
            "collinear_merged_count": total_collinear_merged,
            # W-7
            "component_count_after": component_count_after,
            "polygon_area_m2": polygon_area_m2,
            # W-1 shape-preservation guard
            "iou_raw_rectified": iou,
            "centroid_shift_m": centroid_shift,
            "bbox_width_ratio": bbox_w_ratio,
            "bbox_height_ratio": bbox_h_ratio,
            "params": params.to_metadata(),
        }

        cleaned_geojson = _as_geojson(cleaned_geom)

        return PolygonCadCleanupResult(
            cleaned_geojson=cleaned_geojson,
            metadata=metadata,
        )


# ── private helpers ──────────────────────────────────────────────────────────


def _iter_polygons(geom: BaseGeometry) -> list[ShapelyPolygon]:
    if isinstance(geom, ShapelyPolygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    geoms = getattr(geom, "geoms", None)
    if geoms is not None:
        return [g for g in geoms if isinstance(g, ShapelyPolygon)]
    return []


def _count_vertices(poly: ShapelyPolygon) -> int:
    coords = list(poly.exterior.coords)
    return max(0, len(coords) - 1)


def _ring_to_open_pts(coords: list[tuple[float, ...]]) -> list[tuple[float, float]]:
    """닫힌 ring (마지막 = 첫번째)을 열린 vertex list로 정규화."""
    pts = [(float(x), float(y)) for x, y, *_ in coords]
    if len(pts) >= 2:
        if (
            abs(pts[0][0] - pts[-1][0]) < 1e-9
            and abs(pts[0][1] - pts[-1][1]) < 1e-9
        ):
            pts = pts[:-1]
    return pts


def _close_ring(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not pts:
        return pts
    if pts[0] != pts[-1]:
        return pts + [pts[0]]
    return pts


def _cleanup_polygon(
    poly: ShapelyPolygon,
    *,
    near_vertex_distance_m: float,
    short_edge_min_length_m: float,
    collinear_angle_tol_deg: float,
) -> tuple[ShapelyPolygon | None, int, int, int]:
    """단일 polygon에 cleanup 적용.

    Returns:
        (cleaned_polygon_or_None, near_merged_count, short_pruned_count, collinear_merged_count)
    """
    pts = _ring_to_open_pts(list(poly.exterior.coords))
    if len(pts) < 3:
        return poly, 0, 0, 0

    # 1. near vertex merge — 인접 거리 짧은 vertex 합치기
    pts_after_near, near_merged = _merge_near_vertices(pts, near_vertex_distance_m)
    if len(pts_after_near) < 3:
        return poly, 0, 0, 0

    # 2. short edge prune — 짧은 edge 끝 vertex 제거
    pts_after_short, short_pruned = _prune_short_edges(
        pts_after_near, short_edge_min_length_m
    )
    if len(pts_after_short) < 3:
        return poly, near_merged, 0, 0

    # 3. collinear merge — 가운데 vertex 가 직선상에 있으면 제거
    pts_after_collinear, collinear_merged = _merge_collinear_by_angle(
        pts_after_short, collinear_angle_tol_deg
    )
    if len(pts_after_collinear) < 3:
        return poly, near_merged, short_pruned, 0

    closed = _close_ring(pts_after_collinear)
    try:
        new_poly = ShapelyPolygon(closed)
        if not new_poly.is_valid:
            new_poly = new_poly.buffer(0)
        if isinstance(new_poly, ShapelyPolygon) and not new_poly.is_empty and new_poly.area > 0:
            return new_poly, near_merged, short_pruned, collinear_merged
    except Exception as e:
        logger.warning("polygon_cad_cleanup: ring rebuild failed: %s", e)

    return poly, 0, 0, 0


def _merge_near_vertices(
    pts: list[tuple[float, float]],
    distance_m: float,
) -> tuple[list[tuple[float, float]], int]:
    """인접 vertex 거리 < distance_m 면 midpoint 로 합친다.

    distance_m <= 0 이면 no-op.
    Returns:
        (merged_pts, merged_count)
    """
    if distance_m <= 0 or len(pts) < 2:
        return list(pts), 0

    out: list[tuple[float, float]] = [pts[0]]
    merged = 0
    for i in range(1, len(pts)):
        prev = out[-1]
        cur = pts[i]
        if _distance(prev, cur) < distance_m:
            mid = ((prev[0] + cur[0]) * 0.5, (prev[1] + cur[1]) * 0.5)
            out[-1] = mid
            merged += 1
        else:
            out.append(cur)

    # wrap-around: 마지막 vs 첫 vertex
    if len(out) >= 2 and _distance(out[0], out[-1]) < distance_m:
        mid = ((out[0][0] + out[-1][0]) * 0.5, (out[0][1] + out[-1][1]) * 0.5)
        out[0] = mid
        out.pop()
        merged += 1

    return out, merged


def _prune_short_edges(
    pts: list[tuple[float, float]],
    min_length_m: float,
) -> tuple[list[tuple[float, float]], int]:
    """edge 길이 < min_length_m 인 edge의 끝 vertex 제거.

    시작 vertex는 anchor로 보존. min_length_m <= 0 이면 no-op.
    """
    if min_length_m <= 0 or len(pts) < 2:
        return list(pts), 0

    out: list[tuple[float, float]] = [pts[0]]
    pruned = 0
    for i in range(1, len(pts)):
        prev = out[-1]
        cur = pts[i]
        if _distance(prev, cur) < min_length_m:
            pruned += 1
        else:
            out.append(cur)

    # wrap-around
    if len(out) >= 2 and _distance(out[0], out[-1]) < min_length_m:
        out.pop()
        pruned += 1

    return out, pruned


def _merge_collinear_by_angle(
    pts: list[tuple[float, float]],
    angle_tol_deg: float,
) -> tuple[list[tuple[float, float]], int]:
    """인접 3 vertex 의 가운데 vertex 가 직선상이면 제거.

    각도 판정: edge1=(p_prev→p), edge2=(p→p_next).
    두 edge 의 방향각 차이의 절댓값이 angle_tol_deg 미만이면 collinear 로 간주.
    angle_tol_deg <= 0 이면 no-op.
    """
    if angle_tol_deg <= 0 or len(pts) < 3:
        return list(pts), 0

    out = list(pts)
    removed_total = 0
    changed = True
    # 한 vertex 제거 후 인접 관계가 바뀌므로 변화 없을 때까지 반복.
    # 단 3개 미만으로 떨어지면 중단 (polygon ring 최소).
    while changed and len(out) > 3:
        changed = False
        new_out: list[tuple[float, float]] = []
        n = len(out)
        i = 0
        while i < n:
            prev_pt = new_out[-1] if new_out else out[(i - 1) % n]
            cur_pt = out[i]
            next_pt = out[(i + 1) % n]
            if _is_collinear(prev_pt, cur_pt, next_pt, angle_tol_deg):
                removed_total += 1
                changed = True
                # cur_pt skip — new_out에 추가 안 함
            else:
                new_out.append(cur_pt)
            i += 1
        if len(new_out) < 3:
            # 너무 많이 잘려서 폐기 — 직전 상태 유지
            break
        out = new_out

    return out, removed_total


def _is_collinear(
    prev_pt: tuple[float, float],
    cur_pt: tuple[float, float],
    next_pt: tuple[float, float],
    angle_tol_deg: float,
) -> bool:
    """edge1 = prev→cur, edge2 = cur→next. 두 edge가 같은 방향이면 collinear."""
    a1 = atan2(cur_pt[1] - prev_pt[1], cur_pt[0] - prev_pt[0])
    a2 = atan2(next_pt[1] - cur_pt[1], next_pt[0] - cur_pt[0])
    # 차이를 (-pi, pi] 로 정규화
    diff = degrees(a2 - a1)
    while diff > 180.0:
        diff -= 360.0
    while diff <= -180.0:
        diff += 360.0
    return abs(diff) < angle_tol_deg


def _count_corners(
    polys: list[ShapelyPolygon],
    *,
    orthogonality_tol_deg: float,
    collinear_tol_deg: float,
) -> tuple[int, int, int]:
    """polygon corner 분류.

    각 vertex의 내각이 (90 ± orthogonality_tol_deg) → right angle corner.
    180 ± collinear_tol_deg → collinear residual.
    Returns:
        (right_angle_count, collinear_residual_count, total_corner_count)
    """
    right_angle = 0
    collinear_residual = 0
    total = 0
    for poly in polys:
        pts = _ring_to_open_pts(list(poly.exterior.coords))
        n = len(pts)
        if n < 3:
            continue
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            cur_pt = pts[i]
            next_pt = pts[(i + 1) % n]
            interior_angle = _interior_angle_deg(prev_pt, cur_pt, next_pt)
            total += 1
            if abs(interior_angle - 90.0) <= orthogonality_tol_deg:
                right_angle += 1
            elif abs(interior_angle - 180.0) <= collinear_tol_deg:
                collinear_residual += 1
    return right_angle, collinear_residual, total


def _interior_angle_deg(
    prev_pt: tuple[float, float],
    cur_pt: tuple[float, float],
    next_pt: tuple[float, float],
) -> float:
    """vertex cur 의 내각 (0~180°) 반환.

    edge1 = cur→prev, edge2 = cur→next 사이 각도.
    """
    v1x = prev_pt[0] - cur_pt[0]
    v1y = prev_pt[1] - cur_pt[1]
    v2x = next_pt[0] - cur_pt[0]
    v2y = next_pt[1] - cur_pt[1]
    n1 = sqrt(v1x * v1x + v1y * v1y)
    n2 = sqrt(v2x * v2x + v2y * v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    from math import acos

    return float(degrees(acos(dot)))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


def _all_edge_lengths(polys: list[ShapelyPolygon]) -> list[float]:
    out: list[float] = []
    for poly in polys:
        pts = _ring_to_open_pts(list(poly.exterior.coords))
        n = len(pts)
        if n < 2:
            continue
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            out.append(_distance(a, b))
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return float(sorted_v[0])
    idx = q * (len(sorted_v) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return float(sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac)


def _compute_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    """두 polygon area의 IoU. 빈 geom 은 0.0."""
    if a.is_empty or b.is_empty:
        return 0.0
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        if union <= 0:
            return 0.0
        return float(inter) / float(union)
    except Exception as e:
        logger.warning("polygon_cad_cleanup: iou compute failed: %s", e)
        return 0.0


def _compute_centroid_shift(a: BaseGeometry, b: BaseGeometry) -> float:
    """두 polygon centroid 간 거리 (m)."""
    if a.is_empty or b.is_empty:
        return 0.0
    try:
        ca = a.centroid
        cb = b.centroid
        return float(_distance((ca.x, ca.y), (cb.x, cb.y)))
    except Exception as e:
        logger.warning("polygon_cad_cleanup: centroid compute failed: %s", e)
        return 0.0


def _compute_bbox_change_ratio(
    a: BaseGeometry, b: BaseGeometry
) -> tuple[float, float]:
    """bbox 가로/세로 비율 (cleaned/raw)."""
    if a.is_empty or b.is_empty:
        return 1.0, 1.0
    try:
        ax_min, ay_min, ax_max, ay_max = a.bounds
        bx_min, by_min, bx_max, by_max = b.bounds
        aw = ax_max - ax_min
        ah = ay_max - ay_min
        bw = bx_max - bx_min
        bh = by_max - by_min
        w_ratio = float(bw) / float(aw) if aw > 1e-9 else 1.0
        h_ratio = float(bh) / float(ah) if ah > 1e-9 else 1.0
        return w_ratio, h_ratio
    except Exception as e:
        logger.warning("polygon_cad_cleanup: bbox compute failed: %s", e)
        return 1.0, 1.0


def _as_geojson(geom: BaseGeometry) -> dict[str, object]:
    if isinstance(geom, ShapelyPolygon):
        geom = MultiPolygon([geom])
    geo = mapping(geom)
    return {"type": geo["type"], "coordinates": geo["coordinates"]}
