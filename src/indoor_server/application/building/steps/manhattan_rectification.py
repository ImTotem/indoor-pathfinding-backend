"""Manhattan rectification — noisy footprint polygon → CAD-style footprint.

iter4 알고리즘 (cycle_6):
  0. dominant axis 추정 (length-weighted histogram + 4-way axis snap)
  1. -dominant_angle_deg 회전 (rotated frame)
  2. simplify(0.3m) — rotated frame에서 수행
  3. simplify 후 zero-length edge 제거
  4. NEW: Forced Rectilinear Projection
       ring을 순회하면서 인접 (v1, v2) 비교:
       - |dx| >= |dy| → horizontal edge 강제 → v2.y = v1.y
       - |dy| > |dx|  → vertical edge 강제   → v2.x = v1.x
       결과: 모든 edge가 정확히 horizontal 또는 vertical
  5. zero-length / 짧은 edge (<manhattan_min_edge_length_m) 제거
  6. collinear merge (0.05m tolerance)
  7. +dominant_angle_deg 회전 (원래 frame 복귀)
  8. fallback check (area 변화 ±20% 초과 시 원본 반환)

iter3 (cycle_5) 대비 핵심 변경:
  - step 4 angle snap (threshold 판정)을 Forced Rectilinear Projection으로 교체
  - 사선 여부 무관하게 모든 edge를 H 또는 V로 강제 → rectified_ratio 항상 1.0
  - manhattan_max_area_change: 0.10 → 0.15 (강제 snap 손실 허용)
  - manhattan_min_rectified_ratio: 제거 (항상 1.0이므로 의미 없음)
  - 신규 메타: forced_rectilinear_used (bool), zero_edges_removed (int)
  - 기존 rectified_ratio: 항상 1.0으로 기록 (backward compat 유지)

Cycle 9 변경 (Sprint 34):
  - manhattan_max_area_change: 0.40 → 0.20 (면적 손실 16% 억제. fallback 가능성 소폭 증가)
  - dominant_angle_deg를 pipeline.py에서 NodePlacementStep에 직접 주입하여
    graph와 footprint가 동일한 회전축을 사용하도록 통일.

보수적 설계: fallback 조건 위반 시 raw polygon 반환.
환경변수 INDOOR_DOMINANT_ANGLE_FORCE_ZERO=true → dominant_angle을 0°로 강제 (디버그용).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from math import atan2, degrees, sqrt

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# 4-way axis snap: 0°, 22.5°, 45°, 67.5° 중 하나로 round
# 22.5° 단위 — 실내 건물이 대부분 0° 또는 약간 회전
_FOUR_WAY_SNAP_ANGLES = [0.0, 22.5, 45.0, 67.5]


@dataclass(frozen=True)
class RectifiedFootprint:
    raw_geojson: dict[str, object]
    rectified_geojson: dict[str, object]
    dominant_angle_deg: float
    area_raw_m2: float
    area_rectified_m2: float
    area_change_ratio: float
    accepted: bool
    # v3 신규 필드
    original_vertex_count: int
    rectified_vertex_count: int
    rectified_ratio: float  # iter4에서는 항상 1.0 (강제 snap)
    fallback_used: bool
    # cycle_3 신규 필드
    collinear_merged_count: int
    # cycle_4 신규 필드
    simplified_vertex_count: int
    # iter3 (cycle_5) 신규 필드
    dominant_angle_raw_deg: float   # length-weighted 추정 원본 (4-way snap 전)
    grid_snapped: bool              # vertex grid snap 적용 여부 (iter4에서 False — 사용 안 함)
    vertex_count_after_snap: int    # forced projection 직후, collinear merge 전
    # iter4 (cycle_6) 신규 필드
    forced_rectilinear_used: bool   # Forced Rectilinear Projection 적용 여부
    zero_edges_removed: int         # zero-length / 너무 짧은 edge 제거 수
    # Sprint 47 신규 (W-3, W-8): floor_pointcloud 모드 전용 추가 메타.
    # default 호출(trajectory 등)에서는 기본값 유지 — 회귀 0.
    snap_mode_used: str = "raw"
    area_change_threshold_used: float = 0.20
    dominant_angle_confidence: float = 0.0
    low_angle_confidence: bool = False
    # Sprint 48 신규: dominant_angle_hint 외부 주입 추적
    dominant_angle_hint_used: bool = False
    dominant_angle_hint_source: str | None = None
    dominant_angle_hint_deg: float | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "enabled": True,
            "accepted": self.accepted,
            "dominant_angle_raw_deg": self.dominant_angle_raw_deg,
            "dominant_angle_deg": self.dominant_angle_deg,
            "area_raw_m2": self.area_raw_m2,
            "area_rectified_m2": self.area_rectified_m2,
            "area_change_ratio": self.area_change_ratio,
            # v3 신규
            "original_vertex_count": self.original_vertex_count,
            "rectified_vertex_count": self.rectified_vertex_count,
            "rectified_ratio": self.rectified_ratio,
            "fallback_used": self.fallback_used,
            # cycle_3 신규
            "collinear_merged_count": self.collinear_merged_count,
            # cycle_4 신규
            "simplified_vertex_count": self.simplified_vertex_count,
            # iter3 (cycle_5) 신규
            "grid_snapped": self.grid_snapped,
            "vertex_count_after_snap": self.vertex_count_after_snap,
            # iter4 (cycle_6) 신규
            "forced_rectilinear_used": self.forced_rectilinear_used,
            "zero_edges_removed": self.zero_edges_removed,
            # Sprint 47 신규 (W-3, W-8)
            "snap_mode_used": self.snap_mode_used,
            "area_change_threshold_used": self.area_change_threshold_used,
            "dominant_angle_confidence": self.dominant_angle_confidence,
            "low_angle_confidence": self.low_angle_confidence,
            # Sprint 48 신규
            "dominant_angle_hint_used": self.dominant_angle_hint_used,
            "dominant_angle_hint_source": self.dominant_angle_hint_source,
            "dominant_angle_hint_deg": self.dominant_angle_hint_deg,
        }


class ManhattanRectificationStep:
    """Forced Rectilinear Projection 기반 dominant-axis 직각화 (iter4)."""

    def run(
        self,
        footprint_geojson: dict[str, object],
        *,
        manhattan_snap_threshold_deg: float = 25.0,  # iter4에서는 무시됨 (legacy compat)
        manhattan_dominant_bin_deg: float = 10.0,
        manhattan_max_area_change: float = 0.20,  # iter6: 0.40→0.20 (Cycle 9: area 손실 억제)
        manhattan_min_rectified_ratio: float = 0.60,  # iter4에서는 무시됨 (항상 1.0)
        # collinear merge 파라미터 (cycle_3)
        collinear_tolerance_m: float = 0.05,
        collinear_min_vertex_count: int = 4,
        # simplify 파라미터 — rotated frame에서 수행 (iter3)
        manhattan_simplify_tolerance_m: float = 0.3,
        # iter4 신규: zero-length / 짧은 edge 제거 임계
        manhattan_min_edge_length_m: float = 0.05,
        # vertex grid snap 파라미터 (iter3 — iter4에서는 사용 안 함, legacy compat)
        grid_snap_resolution_m: float = 0.0,
        # 하위 호환 — 구 파라미터명 허용 (pipeline.py가 기본값으로 호출)
        simplify_tolerance_m: float = 0.0,
        min_edge_length_m: float = 0.3,
        angle_snap_deg: float | None = None,
        area_change_limit_ratio: float | None = None,
        # Sprint 47 신규 (W-3, W-4, W-8): floor_pointcloud 모드 전용.
        # default 호출(trajectory 등 5경로)에서는 모두 기본값 → 회귀 0.
        manhattan_max_area_change_floor_pointcloud: float | None = None,
        area_change_dynamic: bool = False,
        dominant_angle_snap_mode: str = "raw",
        dominant_angle_confidence_threshold: float = 0.55,
        # Sprint 48 신규 (Codex F-1, F-5): 외부 hint 주입.
        # default None → 기존 호출자 회귀 0. hint 주입 시 internal contour 추정
        # 결과는 metadata 용으로 보존하되 dominant_angle 은 hint 로 강제.
        dominant_angle_hint_deg: float | None = None,
        dominant_angle_hint_source: str | None = None,
    ) -> RectifiedFootprint:
        # 구 파라미터 매핑
        area_limit = (
            area_change_limit_ratio
            if area_change_limit_ratio is not None
            else manhattan_max_area_change
        )
        # Sprint 47 W-3: floor_pointcloud 모드 area_change 임계 분리.
        # 명시 전달 시 area_limit 을 override (default None → 기존 0.20 유지).
        if manhattan_max_area_change_floor_pointcloud is not None:
            area_limit = float(manhattan_max_area_change_floor_pointcloud)

        geom = _clean_geometry(shape(footprint_geojson))
        if geom.is_empty or geom.area <= 0:
            return _rejected(footprint_geojson, 0.0, 0.0, 0.0, 0, 0)

        raw_area = float(geom.area)
        original_vertex_count = _count_vertices(geom)

        # Sprint 47 W-3 dynamic: 작은 polygon (< 30 m²) 은 floor_pc threshold 까지 완화.
        if area_change_dynamic and raw_area < 30.0:
            # floor_pc 임계가 명시되어 있으면 그쪽 선택, 없으면 0.55 fallback.
            dynamic_limit = (
                float(manhattan_max_area_change_floor_pointcloud)
                if manhattan_max_area_change_floor_pointcloud is not None
                else 0.55
            )
            area_limit = max(area_limit, dynamic_limit)

        # Step 0: dominant axis 추정 (원본 geometry에서, length-weighted + 4-way snap)
        # Sprint 47 W-8: confidence 함께 받음.
        dominant_raw, dominant_natural, dominant_confidence = (
            _dominant_angle_with_snap(
                geom,
                min_edge_length_m=min_edge_length_m,
                bin_deg=manhattan_dominant_bin_deg,
            )
        )

        # Sprint 47 W-4: snap mode 결정.
        # mode = "raw" → 환경변수 + _four_way_snap 동작 (기존 default).
        # mode = "four_way" → 강제 4-way snap (env 무시).
        # mode = "force_zero" → 강제 0°.
        # Sprint 48: dominant_angle_hint_deg != None 이면 internal confidence
        # 무시하고 hint 로 강제 (snap_mode_used="hint").
        low_angle_confidence = dominant_confidence < dominant_angle_confidence_threshold
        hint_used = False
        if dominant_angle_hint_deg is not None:
            dominant = _fold_to_manhattan_offset(float(dominant_angle_hint_deg))
            snap_mode_used = "hint"
            hint_used = True
        elif dominant_angle_snap_mode == "force_zero":
            dominant = 0.0
            snap_mode_used = "force_zero"
        elif dominant_angle_snap_mode == "four_way":
            # confidence 너무 낮으면 raw 보존 (사선 fallback)
            if low_angle_confidence:
                dominant = dominant_raw
                snap_mode_used = "fallback"
            else:
                dominant = _force_four_way_snap(dominant_raw)
                snap_mode_used = "four_way"
        else:
            # raw: 환경변수 기반 _four_way_snap 그대로 (legacy 동작 보존).
            dominant = dominant_natural
            snap_mode_used = "raw"

        # Step 1: rotate to axis-aligned frame
        rotated = affinity.rotate(geom, -dominant, origin=(0.0, 0.0), use_radians=False)

        # Step 2: simplify in rotated frame
        simplified_geom = _simplify_geometry(
            rotated,
            tolerance_m=manhattan_simplify_tolerance_m,
            raw_area=raw_area,
            area_change_limit=0.05,
        )
        simplified_vertex_count = _count_vertices(simplified_geom)

        # Step 3: simplify 후 zero-length edge 제거 (pre-projection cleanup)
        pre_proj_geom = _remove_short_edges(
            simplified_geom, min_length_m=manhattan_min_edge_length_m
        )

        # Step 4: Forced Rectilinear Projection — 모든 edge를 H 또는 V로 강제
        projected_geom, zero_edges_removed = _forced_rectilinear_projection_all(
            pre_proj_geom,
            min_edge_length_m=manhattan_min_edge_length_m,
        )

        if projected_geom.is_empty or projected_geom.area <= 0:
            return _rejected(
                footprint_geojson, raw_area, dominant_raw, dominant,
                original_vertex_count, simplified_vertex_count,
            )

        vertex_count_after_snap = _count_vertices(projected_geom)

        # Step 5: collinear merge — axis-aligned frame에서 작업
        merged_geom, collinear_merged_count = _merge_collinear_all_polygons(
            projected_geom,
            tolerance=collinear_tolerance_m,
            min_vertex_count=collinear_min_vertex_count,
        )

        if merged_geom.is_empty or merged_geom.area <= 0:
            merged_geom = projected_geom
            collinear_merged_count = 0

        # merge 후 area 변화 확인 (collinear 제거는 area를 보존해야 함)
        merged_area_snap = float(merged_geom.area)
        snap_area = float(projected_geom.area)
        if snap_area > 0 and abs(merged_area_snap - snap_area) / snap_area > 0.01:
            merged_geom = projected_geom
            collinear_merged_count = 0

        # Step 6: rotate back
        result_geom = affinity.rotate(merged_geom, dominant, origin=(0.0, 0.0), use_radians=False)
        result_geom = _clean_geometry(result_geom)

        if result_geom.is_empty or result_geom.area <= 0:
            return _rejected(
                footprint_geojson, raw_area, dominant_raw, dominant,
                original_vertex_count, simplified_vertex_count,
            )

        rectified_area = float(result_geom.area)
        area_delta = abs(rectified_area - raw_area) / raw_area

        # Step 7: fallback 조건 검사 (iter4: area ±15% 초과 시만 fallback)
        fallback_used = area_delta > area_limit
        accepted = not fallback_used
        output_geom = geom if fallback_used else result_geom

        # Sprint 48: hint 사용 후 area_change reject 시 라벨 분리.
        # caller (pipeline) 가 hint candidate retry 흐름에서 다음 후보로 진행할 때
        # 본 metadata 의 snap_mode_used 로 reject 사유 추적.
        final_snap_mode = snap_mode_used
        if hint_used and fallback_used:
            final_snap_mode = "hint_rejected"

        final_vertex_count = _count_vertices(output_geom)
        final_merged = collinear_merged_count if not fallback_used else 0
        final_zero_removed = zero_edges_removed if not fallback_used else 0

        return RectifiedFootprint(
            raw_geojson=_as_geojson(geom),
            rectified_geojson=_as_geojson(output_geom),
            dominant_angle_deg=dominant,
            area_raw_m2=raw_area,
            area_rectified_m2=float(output_geom.area),
            area_change_ratio=area_delta,
            accepted=accepted,
            original_vertex_count=original_vertex_count,
            rectified_vertex_count=final_vertex_count,
            rectified_ratio=1.0,  # iter4: 강제 snap이라 항상 1.0
            fallback_used=fallback_used,
            collinear_merged_count=final_merged,
            simplified_vertex_count=simplified_vertex_count,
            dominant_angle_raw_deg=dominant_raw,
            grid_snapped=False,  # iter4에서는 grid snap 사용 안 함
            vertex_count_after_snap=(
                vertex_count_after_snap if not fallback_used else final_vertex_count
            ),
            forced_rectilinear_used=not fallback_used,
            zero_edges_removed=final_zero_removed,
            # Sprint 47 신규
            snap_mode_used=final_snap_mode,
            area_change_threshold_used=float(area_limit),
            dominant_angle_confidence=float(dominant_confidence),
            low_angle_confidence=bool(low_angle_confidence),
            # Sprint 48 신규
            dominant_angle_hint_used=hint_used,
            dominant_angle_hint_source=(
                dominant_angle_hint_source if hint_used else None
            ),
            dominant_angle_hint_deg=(
                float(dominant_angle_hint_deg)
                if hint_used and dominant_angle_hint_deg is not None
                else None
            ),
        )


# ── private helpers ───────────────────────────────────────────────────────────


def _rejected(
    raw_geojson: dict[str, object],
    area_raw: float,
    dominant_raw: float,
    dominant: float,
    original_vertex_count: int,
    simplified_vertex_count: int = 0,
) -> RectifiedFootprint:
    return RectifiedFootprint(
        raw_geojson=raw_geojson,
        rectified_geojson=raw_geojson,
        dominant_angle_deg=dominant,
        area_raw_m2=area_raw,
        area_rectified_m2=area_raw,
        area_change_ratio=0.0,
        accepted=False,
        original_vertex_count=original_vertex_count,
        rectified_vertex_count=original_vertex_count,
        rectified_ratio=0.0,
        fallback_used=True,
        collinear_merged_count=0,
        simplified_vertex_count=simplified_vertex_count,
        dominant_angle_raw_deg=dominant_raw,
        grid_snapped=False,
        vertex_count_after_snap=original_vertex_count,
        forced_rectilinear_used=False,
        zero_edges_removed=0,
    )


def _four_way_snap(angle_deg: float) -> float:
    """dominant angle을 가장 가까운 22.5° 배수 중 하나로 round.

    입력: (-45, 45] 범위의 angle_deg (dominant axis offset).
    출력: 0.0 / 22.5 / -22.5 / 45.0 / -45.0 중 하나 — 동일 (-45, 45] 범위 유지.

    환경변수 INDOOR_DOMINANT_ANGLE_FORCE_ZERO=true 시 항상 0° 반환.
    """
    # iter7 final: 사용자 피드백 — 0° 강제는 L자→T자 변형(area 31%). 원본 모양 보존 우선.
    # raw angle 직접 반환을 hardcode (env 없이도 default raw).
    # 회귀 시 INDOOR_DOMINANT_ANGLE_FORCE_ZERO=true / INDOOR_DOMINANT_ANGLE_4WAY=true.
    if os.environ.get("INDOOR_DOMINANT_ANGLE_FORCE_ZERO", "").lower() in ("1", "true", "yes"):
        return 0.0
    if os.environ.get("INDOOR_DOMINANT_ANGLE_4WAY", "").lower() in ("1", "true", "yes"):
        candidates = [-45.0, -22.5, 0.0, 22.5, 45.0]
        best = min(candidates, key=lambda c: abs(angle_deg - c))
        return best if best != -45.0 else 45.0
    return angle_deg  # default: raw


def _dominant_angle_with_snap(
    geom: BaseGeometry,
    *,
    min_edge_length_m: float,
    bin_deg: float,
) -> tuple[float, float, float]:
    """length-weighted histogram → dominant angle → 4-way axis snap.

    Returns:
        (dominant_angle_raw_deg, dominant_angle_deg, confidence)
        raw = length-weighted 추정 결과 (4-way snap 전)
        snapped = 4-way snap 후 최종 dominant angle (env 기반 legacy 동작)
        confidence = best_bin_weight / total_weight (Sprint 47 W-8). 0.0~1.0.
    """
    dominant_raw, confidence = _dominant_angle_histogram_with_confidence(
        geom, min_edge_length_m=min_edge_length_m, bin_deg=bin_deg
    )
    dominant_snapped = _four_way_snap(dominant_raw)
    return dominant_raw, dominant_snapped, confidence


def _force_four_way_snap(angle_deg: float) -> float:
    """env 무시하고 4-way snap 강제 (Sprint 47 W-4 floor_pointcloud 모드)."""
    candidates = [-45.0, -22.5, 0.0, 22.5, 45.0]
    best = min(candidates, key=lambda c: abs(angle_deg - c))
    return best if best != -45.0 else 45.0


def _fold_to_manhattan_offset(angle_deg: float) -> float:
    """Sprint 48: arbitrary angle 을 Manhattan-frame offset (-45, 45]° 로 fold."""
    angle = angle_deg % 90.0
    if angle > 45.0:
        angle -= 90.0
    if angle <= -45.0:
        angle += 90.0
    return float(angle)


def _dominant_angle_histogram(
    geom: BaseGeometry,
    *,
    min_edge_length_m: float,
    bin_deg: float,
) -> float:
    """legacy API — angle 만 반환."""
    angle, _ = _dominant_angle_histogram_with_confidence(
        geom, min_edge_length_m=min_edge_length_m, bin_deg=bin_deg
    )
    return angle


def _dominant_angle_histogram_with_confidence(
    geom: BaseGeometry,
    *,
    min_edge_length_m: float,
    bin_deg: float,
) -> tuple[float, float]:
    """Step 1-2: length-weighted edge angle histogram → dominant angle + confidence.

    confidence = best_bin_weight / total_weight (Sprint 47 W-8).
    sparse polygon에서 dominant axis 추정의 신뢰도 — 0.5 미만이면 fallback 권고.

    iter3 변경: bin 가중치를 edge count에서 length로 변경.
    길이가 긴 edge가 dominant axis 결정에 더 큰 영향을 미침.
    noise short edge(min_edge_length_m 미만)는 필터로 제외.

    Returns:
        (angle_deg, confidence). angle 은 (-45, 45]°.
    """
    n_bins = int(180.0 / bin_deg)
    # bin별 length 가중 합 + length-weighted angle 합
    weights = np.zeros(n_bins, dtype=float)
    angle_sum = np.zeros(n_bins, dtype=float)

    for poly in _iter_polygons(geom):
        coords = list(poly.exterior.coords)
        for (ax, ay), (bx, by) in zip(coords, coords[1:], strict=False):
            dx = bx - ax
            dy = by - ay
            length = sqrt(dx * dx + dy * dy)
            if length < min_edge_length_m:
                continue
            angle = _edge_angle_deg(ax, ay, bx, by)
            if angle is None:
                continue
            bin_idx = int(angle / bin_deg) % n_bins
            # length-weighted: 긴 edge가 dominant 결정에 더 큰 영향
            weights[bin_idx] += length
            angle_sum[bin_idx] += angle * length

    total_weight = float(weights.sum())
    if weights.max() < 1e-9 or total_weight < 1e-9:
        return 0.0, 0.0

    best_bin = int(np.argmax(weights))
    # best bin 내의 length-weighted mean angle
    dominant_raw = angle_sum[best_bin] / weights[best_bin]
    confidence = float(weights[best_bin]) / total_weight

    # dominant_raw를 (-45, 45] 로 접기
    angle = dominant_raw % 90.0  # 0~90°
    if angle > 45.0:
        angle -= 90.0  # -45~0°
    return float(angle), confidence


def _remove_short_edges(geom: BaseGeometry, *, min_length_m: float) -> BaseGeometry:
    """ring에서 zero-length 또는 min_length_m 미만 edge의 중복 vertex를 제거.

    Args:
        geom: input geometry
        min_length_m: 이 길이 미만의 edge를 제거 (시작 vertex 보존, 끝 vertex 제거)

    Returns:
        정리된 geometry
    """
    if min_length_m <= 0.0:
        return geom

    cleaned_polys: list[Polygon] = []
    for poly in _iter_polygons(geom):
        coords = list(poly.exterior.coords)
        # 닫힌 ring에서 마지막 중복 제거
        pts = [(float(x), float(y)) for x, y, *_ in coords]
        if len(pts) >= 2 and (
            abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9
        ):
            pts = pts[:-1]

        if len(pts) < 3:
            cleaned_polys.append(poly)
            continue

        # 인접 중복 vertex 제거
        deduped: list[tuple[float, float]] = [pts[0]]
        for pt in pts[1:]:
            dx = pt[0] - deduped[-1][0]
            dy = pt[1] - deduped[-1][1]
            length = sqrt(dx * dx + dy * dy)
            if length >= min_length_m:
                deduped.append(pt)

        if len(deduped) < 3:
            cleaned_polys.append(poly)
            continue

        if deduped[0] != deduped[-1]:
            deduped.append(deduped[0])

        try:
            new_poly = Polygon(deduped).buffer(0)
            if isinstance(new_poly, Polygon) and not new_poly.is_empty and new_poly.area > 0:
                cleaned_polys.append(new_poly)
                continue
        except Exception:
            pass
        cleaned_polys.append(poly)

    if not cleaned_polys:
        return geom

    result = _clean_geometry(unary_union(cleaned_polys))
    if result.is_empty or result.area <= 0:
        return geom
    return result


def _simplify_geometry(
    geom: BaseGeometry,
    *,
    tolerance_m: float,
    raw_area: float,
    area_change_limit: float,
) -> BaseGeometry:
    """simplify — noise short edge 제거.

    iter3: rotated frame에서 호출되므로 직각 edge에 최적화.
    tolerance_m으로 simplify 시도. area 변화가 area_change_limit을 초과하면
    tolerance를 절반으로 줄여 재시도. 재시도도 실패하면 원본 반환.

    Args:
        geom: 원본 geometry (axis-aligned rotated frame)
        tolerance_m: simplify Douglas-Peucker epsilon (m)
        raw_area: 원본 area (비교 기준)
        area_change_limit: area 변화 허용 비율 (0.05 = 5%)
    """
    if tolerance_m <= 0.0:
        return geom

    for attempt_tolerance in (tolerance_m, tolerance_m / 2.0):
        simplified = geom.simplify(attempt_tolerance, preserve_topology=True)
        simplified = _clean_geometry(simplified)
        if simplified.is_empty or simplified.area <= 0:
            continue
        if raw_area > 0 and abs(simplified.area - raw_area) / raw_area <= area_change_limit:
            return simplified

    return geom


def _clean_geometry(geom: BaseGeometry) -> BaseGeometry:
    if geom.is_empty:
        return geom
    cleaned = geom.buffer(0)
    if isinstance(cleaned, (Polygon, MultiPolygon)):
        return cleaned
    polygons = [g for g in getattr(cleaned, "geoms", []) if isinstance(g, Polygon)]
    return unary_union(polygons) if polygons else cleaned


def _as_geojson(geom: BaseGeometry) -> dict[str, object]:
    geo = mapping(geom)
    return {"type": geo["type"], "coordinates": geo["coordinates"]}


def _iter_polygons(geom: BaseGeometry) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def _count_vertices(geom: BaseGeometry) -> int:
    total = 0
    for poly in _iter_polygons(geom):
        coords = list(poly.exterior.coords)
        total += max(0, len(coords) - 1)
    return total


def _edge_angle_deg(ax: float, ay: float, bx: float, by: float) -> float | None:
    """edge (a→b)의 각도 (°). 0~180° 범위로 정규화. 길이 0이면 None."""
    dx = bx - ax
    dy = by - ay
    length = sqrt(dx * dx + dy * dy)
    if length < 1e-9:
        return None
    angle = degrees(atan2(dy, dx))
    angle = angle % 180.0
    return angle


# ── iter4: Forced Rectilinear Projection ─────────────────────────────────────


def _forced_rectilinear_projection_all(
    geom: BaseGeometry,
    *,
    min_edge_length_m: float,
) -> tuple[BaseGeometry, int]:
    """모든 polygon에 Forced Rectilinear Projection 적용.

    Returns:
        (projected_geom, total_zero_edges_removed)
    """
    total_zero_removed = 0
    projected_polys: list[Polygon] = []

    for poly in _iter_polygons(geom):
        proj_poly, zero_removed = _forced_rectilinear_polygon(
            poly, min_edge_length_m=min_edge_length_m
        )
        total_zero_removed += zero_removed
        if not proj_poly.is_empty and proj_poly.area > 0:
            projected_polys.append(proj_poly)

    if not projected_polys:
        return geom, 0

    result = _clean_geometry(unary_union(projected_polys))
    if result.is_empty or result.area <= 0:
        return geom, 0
    return result, total_zero_removed


def _forced_rectilinear_polygon(
    poly: Polygon,
    *,
    min_edge_length_m: float,
) -> tuple[Polygon, int]:
    """단일 polygon에 Forced Rectilinear Projection 적용.

    axis-aligned frame에서 동작한다고 가정 (caller가 회전 후 호출).

    알고리즘:
      1. ring 순회. 각 vertex v[i]에서 다음 vertex v[i+1]로의 edge를 평가.
      2. dx = v[i+1].x - v[i].x, dy = v[i+1].y - v[i].y 계산.
      3. |dx| >= |dy|: horizontal edge → v[i+1].y = v[i].y (y 좌표를 v[i]에 맞춤)
         |dy| > |dx|:  vertical edge   → v[i+1].x = v[i].x (x 좌표를 v[i]에 맞춤)
      4. 위 결과로 zero-length / 짧은 edge 제거.

    Returns:
        (projected_polygon, zero_edges_removed_count)
    """
    exterior_coords = list(poly.exterior.coords)
    proj_coords, zero_removed = _force_rectilinear_ring(
        exterior_coords, min_edge_length_m=min_edge_length_m
    )

    if len(proj_coords) < 4:
        return poly, 0

    try:
        result = Polygon(proj_coords).buffer(0)
        if isinstance(result, Polygon) and not result.is_empty and result.area > 0:
            return result, zero_removed
    except Exception:
        pass

    return poly, 0


def _force_rectilinear_ring(
    coords: list[tuple[float, ...]],
    *,
    min_edge_length_m: float,
) -> tuple[list[tuple[float, float]], int]:
    """ring의 각 vertex를 Forced Rectilinear Projection으로 재계산.

    Step 4 핵심: 인접 vertex 쌍을 순회하면서 edge를 H 또는 V로 강제.
    이후 zero-length / 짧은 edge 제거.

    Args:
        coords: 닫힌 ring 좌표 (마지막 = 첫 번째)
        min_edge_length_m: 이 길이 미만 edge는 제거

    Returns:
        (projected_coords, zero_edges_removed_count)
    """
    n = len(coords) - 1  # 닫힌 ring → 마지막 점 = 첫 점
    if n < 3:
        return [(float(x), float(y)) for x, y, *_ in coords], 0

    pts = [(float(x), float(y)) for x, y, *_ in coords[:n]]

    # Forward pass: 각 vertex를 이전 vertex 기준으로 H/V 강제
    # pts[0]은 anchor (변경 없음), pts[1:]을 순차적으로 갱신
    projected = list(pts)
    for i in range(n):
        cur = projected[i]
        next_idx = (i + 1) % n
        nxt = projected[next_idx]

        dx = nxt[0] - cur[0]
        dy = nxt[1] - cur[1]

        if abs(dx) >= abs(dy):
            # horizontal edge 강제: next.y = cur.y
            projected[next_idx] = (nxt[0], cur[1])
        else:
            # vertical edge 강제: next.x = cur.x
            projected[next_idx] = (cur[0], nxt[1])

    # zero-length / 짧은 edge 제거
    zero_removed = 0
    deduped: list[tuple[float, float]] = []
    for pt in projected:
        if not deduped:
            deduped.append(pt)
            continue
        dx = pt[0] - deduped[-1][0]
        dy = pt[1] - deduped[-1][1]
        length = sqrt(dx * dx + dy * dy)
        if length < min_edge_length_m:
            zero_removed += 1
        else:
            deduped.append(pt)

    # wrap-around: 첫 번째와 마지막 vertex가 너무 가까우면 마지막 제거
    if len(deduped) >= 2:
        dx = deduped[0][0] - deduped[-1][0]
        dy = deduped[0][1] - deduped[-1][1]
        if sqrt(dx * dx + dy * dy) < min_edge_length_m:
            deduped.pop()
            zero_removed += 1

    if len(deduped) < 3:
        return [(float(x), float(y)) for x, y, *_ in coords], 0

    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])

    return deduped, zero_removed


# ── Step 5b: collinear merge ──────────────────────────────────────────────────


def _merge_collinear_ring(
    coords: list[tuple[float, float]],
    *,
    tolerance: float,
    min_vertex_count: int,
) -> tuple[list[tuple[float, float]], int]:
    """axis-aligned frame에서 ring의 collinear vertex 제거."""
    pts = list(coords)
    if len(pts) >= 2 and (
        abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9
    ):
        pts = pts[:-1]

    total_removed = 0
    changed = True
    while changed:
        changed = False
        n = len(pts)
        if n <= min_vertex_count:
            break

        new_pts: list[tuple[float, float]] = []
        i = 0
        while i < n:
            if len(new_pts) + (n - i) <= min_vertex_count:
                new_pts.extend(pts[i:])
                break

            prev_pt = pts[(i - 1) % n] if new_pts else pts[i - 1]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % n]

            if new_pts:
                prev_pt = new_pts[-1]

            same_h = (
                abs(prev_pt[1] - curr_pt[1]) <= tolerance
                and abs(curr_pt[1] - next_pt[1]) <= tolerance
            )
            same_v = (
                abs(prev_pt[0] - curr_pt[0]) <= tolerance
                and abs(curr_pt[0] - next_pt[0]) <= tolerance
            )

            if same_h or same_v:
                total_removed += 1
                changed = True
            else:
                new_pts.append(curr_pt)
            i += 1

        pts = new_pts

    if len(pts) < 3:
        return coords, 0

    result = list(pts)
    if result[0] != result[-1]:
        result.append(result[0])

    return result, total_removed


def _merge_collinear_all_polygons(
    geom: BaseGeometry,
    *,
    tolerance: float,
    min_vertex_count: int,
) -> tuple[BaseGeometry, int]:
    """모든 polygon에 collinear merge 적용."""
    total_removed = 0
    merged_polys: list[Polygon] = []

    for poly in _iter_polygons(geom):
        exterior_coords = list(poly.exterior.coords)
        merged_coords, removed = _merge_collinear_ring(
            [(float(x), float(y)) for x, y, *_ in exterior_coords],
            tolerance=tolerance,
            min_vertex_count=min_vertex_count,
        )
        total_removed += removed

        if len(merged_coords) >= 4:
            try:
                new_poly = Polygon(merged_coords).buffer(0)
                if isinstance(new_poly, Polygon) and not new_poly.is_empty and new_poly.area > 0:
                    merged_polys.append(new_poly)
                    continue
            except Exception:
                pass
        merged_polys.append(poly)

    if not merged_polys:
        return geom, 0

    result = _clean_geometry(unary_union(merged_polys))
    return result, total_removed


# ── grid snap — iter3 legacy, iter4에서 미사용이나 API 유지 ─────────────────


def _grid_snap_geometry(geom: BaseGeometry, *, resolution: float) -> BaseGeometry:
    """모든 polygon vertex를 resolution 격자에 맞게 round.

    iter3 신규. iter4에서는 기본적으로 호출 안 함 (grid_snap_resolution_m=0.0 기본값).
    단위 테스트 호환용 API 보존.
    """
    if resolution <= 0.0:
        return geom

    snapped_polys: list[Polygon] = []
    for poly in _iter_polygons(geom):
        coords = list(poly.exterior.coords)
        snapped_coords = [
            (round(x / resolution) * resolution, round(y / resolution) * resolution)
            for x, y, *_ in coords
        ]
        # 중복 vertex 제거 후 ring 유효성 확인
        deduped: list[tuple[float, float]] = []
        for pt in snapped_coords:
            if not deduped or (
                abs(pt[0] - deduped[-1][0]) > 1e-9 or abs(pt[1] - deduped[-1][1]) > 1e-9
            ):
                deduped.append(pt)

        if len(deduped) < 3:
            snapped_polys.append(poly)
            continue

        if deduped[0] != deduped[-1]:
            deduped.append(deduped[0])

        try:
            new_poly = Polygon(deduped).buffer(0)
            if isinstance(new_poly, Polygon) and not new_poly.is_empty and new_poly.area > 0:
                snapped_polys.append(new_poly)
                continue
        except Exception:
            pass
        snapped_polys.append(poly)

    if not snapped_polys:
        return geom

    result = _clean_geometry(unary_union(snapped_polys))
    if result.is_empty or result.area <= 0:
        return geom
    return result
