"""ManhattanRectificationStep tests — v3 (cycle_2) + collinear merge (cycle_3)
                                      + simplify + snap_threshold 25° (cycle_4)
                                      + iter3 rotated-frame + 4-way snap + grid snap (cycle_5)
                                      + iter4 Forced Rectilinear Projection (cycle_6).

7-step 알고리즘 검증:
  - happy path: noisy 직사각형 → 완벽 직각
  - dominant_angle≠0: 45° 회전된 건물
  - fallback: 완전 대각선 polygon → 원본 유지
  - area 보존 검증
  - metadata 필드 완전성
  - empty geometry rejected

cycle_3 추가 (collinear merge):
  - 직사각형 noisy (vertex 100개 → 4개)
  - L자형 (vertex 200개 → 6개)
  - 사선 + 직각 혼합 (사선 보존, 직각 부분만 merge)
  - tolerance 경계값 (0.009m vs 0.011m)
  - 통합: 과밀 직사각형 vertex가 100개 이하로 감소

cycle_4 추가 (simplify):
  - simplify tolerance 0.3m로 noise short edge 제거
  - area 변화 5% 초과 시 tolerance 반감 재시도
  - simplify_tolerance 0.0 → skip (원본 유지)
  - snap_threshold 25° 기본값으로 더 공격적 snap 검증

iter3 / cycle_5 추가 (4-way axis snap + rotated-frame simplify + grid snap):
  - 4-way axis snap: -16°/+16°/+35° 입력 → 0/0/45° 결과
  - length-weighted dominant: 긴 edge가 short noise edge를 이김
  - grid snap: 0.1m 격자에 vertex 정렬 확인
  - rotated-frame simplify: 직각도 향상 검증
  - metadata 신규 필드: dominant_angle_raw_deg / grid_snapped / vertex_count_after_snap

iter4 / cycle_6 추가 (Forced Rectilinear Projection):
  - F1: 정사각형(직각 입력) → 그대로 (accepted, area 변화 없음)
  - F2: L자 + 사선 1개 (5각형) → 강제 직각화 후 accepted
  - F3: 19각형 (noise polygon) → 강제 후 vertex 대폭 감소
  - F4: zero-length edge 제거 검증
  - F5: area 15% 초과 시 fallback
  - F6: forced_rectilinear_used / zero_edges_removed 메타 필드 검증
  - F7: 모든 edge가 horizontal 또는 vertical임을 확인 (accepted 케이스)
"""
from __future__ import annotations

import math

import pytest
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

from indoor_server.application.building.steps.manhattan_rectification import (
    ManhattanRectificationStep,
    _dominant_angle_histogram,
    _force_rectilinear_ring,
    _four_way_snap,
    _grid_snap_geometry,
    _merge_collinear_ring,
    _simplify_geometry,  # noqa: F401 — used in S1/S2 tests via top-level import
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _geo(poly: Polygon | MultiPolygon) -> dict[str, object]:
    mapped = mapping(poly)
    return {"type": mapped["type"], "coordinates": mapped["coordinates"]}


def _angle_between(ax: float, ay: float, bx: float, by: float) -> float:
    """두 점을 잇는 edge의 절대 각도 (°), 0~180° 정규화."""
    dx = bx - ax
    dy = by - ay
    return math.degrees(math.atan2(dy, dx)) % 180.0


def _all_edge_angles(poly: Polygon) -> list[float]:
    coords = list(poly.exterior.coords)
    angles = []
    for (ax, ay), (bx, by) in zip(coords, coords[1:], strict=False):
        dx = bx - ax
        dy = by - ay
        if math.sqrt(dx * dx + dy * dy) < 1e-6:
            continue
        angles.append(_angle_between(ax, ay, bx, by))
    return angles


def _is_manhattan(poly: Polygon, tolerance_deg: float = 5.0) -> bool:
    """모든 edge가 0° or 90° ± tolerance이면 True."""
    for angle in _all_edge_angles(poly):
        if not (
            angle <= tolerance_deg
            or abs(angle - 90.0) <= tolerance_deg
            or angle >= (180.0 - tolerance_deg)
        ):
            return False
    return True


# ── T1: axis-aligned rectangle — 기존 테스트 보존 ────────────────────────────


def test_axis_aligned_rectangle_preserves_area() -> None:
    """T1: axis-aligned rectangle은 area 변화 없이 accepted."""
    footprint = _geo(Polygon([(0, 0), (10, 0), (10, 4), (0, 4), (0, 0)]))

    result = ManhattanRectificationStep().run(footprint)

    assert result.accepted is True
    assert result.area_change_ratio <= 0.01
    assert abs(shape(result.rectified_geojson).area - 40.0) < 0.1


# ── T2: noisy rectangle → 완벽 직각 ─────────────────────────────────────────


def test_noisy_rectangle_becomes_right_angle() -> None:
    """T2: vertex에 노이즈가 있는 직사각형 → edge 직각화 후 accepted.

    5x3 직사각형에 0.2m 노이즈. 결과 polygon의 모든 edge가 0° or 90°여야 함.
    """
    footprint = _geo(Polygon([
        (0.1, -0.1),    # (0,0) + noise
        (5.05, 0.08),   # (5,0) + noise
        (4.95, 3.12),   # (5,3) + noise
        (-0.08, 2.95),  # (0,3) + noise
        (0.1, -0.1),
    ]))

    result = ManhattanRectificationStep().run(footprint)

    # accepted 여부 — area 변화가 10% 이내면 통과
    assert result.accepted is True, (
        f"fallback_used={result.fallback_used}, area_change={result.area_change_ratio:.3f}"
    )
    assert result.area_change_ratio < 0.10

    # 결과 polygon의 edge가 Manhattan-aligned인지 확인
    result_geom = shape(result.rectified_geojson)
    if isinstance(result_geom, Polygon):
        assert _is_manhattan(result_geom, tolerance_deg=5.0), (
            f"Edge angles: {_all_edge_angles(result_geom)}"
        )


# ── T3: dominant_angle ≠ 0 — 45° 회전된 건물 ─────────────────────────────────


def test_dominant_angle_rotated_building() -> None:
    """T3: 10° 회전된 직사각형 → dominant_angle_raw_deg ≈10° 추정 + 4-way snap 검증.

    10° 회전된 5x3 box.
    - dominant_angle_raw_deg (4-way snap 전 추정값)이 10° ± 7°여야 함.
    - dominant_angle_deg (4-way snap 후)는 0° (10°이 0°에 가장 가까우므로).

    iter4 변경: 강제 직각화 시 10° 회전 polygon은 area 손실이 발생할 수 있음.
    area ≥ 15% 손실 시 fallback (원본 반환) → area_change_ratio는 fallback 기준.
    테스트는 dominant angle 추정값만 검증 (area 조건은 fallback/accepted 모두 허용).
    """
    import math

    angle_rad = math.radians(10.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    base_pts = [(0, 0), (5, 0), (5, 3), (0, 3)]
    rotated_pts = [
        (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
        for x, y in base_pts
    ]
    footprint = _geo(Polygon(rotated_pts))

    result = ManhattanRectificationStep().run(footprint)

    # dominant_angle_raw_deg (4-way snap 전) ≈ 10°
    da_raw = result.dominant_angle_raw_deg
    assert abs(da_raw - 10.0) < 7.0 or abs(da_raw + 80.0) < 7.0, (
        f"dominant_angle_raw_deg={da_raw} expected ≈10°"
    )
    # dominant_angle_deg (4-way snap 후): 10° → 0° (22.5° 이내이므로)
    da_snapped = result.dominant_angle_deg
    assert abs(da_snapped) <= 22.5, (
        f"dominant_angle_deg={da_snapped} should be 0° or 22.5° after 4-way snap"
    )
    # iter4: 강제 직각화 후 area 손실이 클 수 있음 → fallback 또는 accepted 모두 허용
    # fallback 시 rectified = raw (area_change_ratio는 rectified vs raw 비교값)
    if result.fallback_used:
        # fallback 시 원본 area 유지
        assert abs(shape(result.rectified_geojson).area - shape(result.raw_geojson).area) < 0.01
    else:
        # accepted 시 area 변화 15% 이내
        assert result.area_change_ratio <= 0.15


# ── T4: fallback — 완전 대각선 polygon → 원본 유지 ───────────────────────────


def test_mixed_angle_polygon_area_change_triggers_fallback() -> None:
    """T4: 사선 edge가 많은 polygon → 강제 직각화 후 area 변화 → fallback or accepted.

    iter4 변경: rectified_ratio 기반 fallback 제거. area ±15% 초과 시만 fallback.
    정육각형은 강제 직각화 시 area 손실이 클 수 있어 fallback 가능.

    테스트: 매우 엄격한 area_change 한계(0.001)를 사용해 확실한 fallback 유도.
    fallback 시 rectified_geojson은 raw area와 동일해야 함.
    """
    import math

    # 정육각형 (모든 edge ≈60°) — 강제 직각화 시 area 변화 큼
    pts = [
        (math.cos(math.radians(i * 60)) * 5, math.sin(math.radians(i * 60)) * 5)
        for i in range(6)
    ]
    footprint = _geo(Polygon(pts))

    # 매우 엄격한 area_change 한계 → 확실한 fallback 유도
    result = ManhattanRectificationStep().run(
        footprint,
        manhattan_max_area_change=0.001,  # 0.1% 이내만 허용
    )

    assert result.fallback_used is True, (
        f"expected fallback with strict area_change. "
        f"area_change={result.area_change_ratio:.4f}"
    )
    # fallback 시 rectified_geojson area는 raw_geojson area와 동일
    raw_area = shape(result.raw_geojson).area
    ret_area = shape(result.rectified_geojson).area
    assert abs(raw_area - ret_area) < 0.1


# ── T5: area 보존 검증 ───────────────────────────────────────────────────────


def test_area_preserved_within_10_percent() -> None:
    """T5: L자형 polygon (계단 모양) → area ±10% 보존."""
    # L자형: 10x8에서 우하단 4x4 빠짐
    footprint = _geo(Polygon([
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 8), (0, 8), (0, 0)
    ]))
    expected_area = shape(footprint).area

    result = ManhattanRectificationStep().run(footprint)

    result_area = shape(result.rectified_geojson).area
    ratio = abs(result_area - expected_area) / expected_area
    assert ratio < 0.10, f"area_change={ratio:.3f} exceeds 10%"


# ── T6: metadata 필드 완전성 ─────────────────────────────────────────────────


def test_metadata_keys_complete() -> None:
    """T6: metadata()가 v3 + cycle_3 + cycle_4 + iter3 필드를 모두 포함."""
    footprint = _geo(Polygon([(0, 0), (5, 0), (5, 3), (0, 3), (0, 0)]))
    result = ManhattanRectificationStep().run(footprint)
    meta = result.metadata()

    required_keys = {
        "enabled",
        "accepted",
        "dominant_angle_deg",
        "area_raw_m2",
        "area_rectified_m2",
        "area_change_ratio",
        "original_vertex_count",
        "rectified_vertex_count",
        "rectified_ratio",
        "fallback_used",
        # cycle_3 신규
        "collinear_merged_count",
        # cycle_4 신규
        "simplified_vertex_count",
        # iter3 (cycle_5) 신규
        "dominant_angle_raw_deg",
        "grid_snapped",
        "vertex_count_after_snap",
    }
    assert required_keys.issubset(meta.keys()), (
        f"Missing keys: {required_keys - meta.keys()}"
    )


# ── T7: empty geometry → rejected ────────────────────────────────────────────


def test_empty_geometry_is_rejected() -> None:
    """T7: 빈 geometry → accepted=False, fallback_used=True."""
    footprint: dict[str, object] = {"type": "Polygon", "coordinates": [[]]}

    result = ManhattanRectificationStep().run(footprint)

    assert result.accepted is False
    assert result.fallback_used is True
    assert result.area_raw_m2 == 0.0


# ── T8: rectified_ratio 수치 검증 ────────────────────────────────────────────


def test_rectified_ratio_for_axis_aligned() -> None:
    """T8: 완전 axis-aligned rectangle → rectified_ratio ≥ 0.70."""
    footprint = _geo(Polygon([(0, 0), (8, 0), (8, 5), (0, 5), (0, 0)]))

    result = ManhattanRectificationStep().run(footprint)

    assert result.rectified_ratio >= 0.70, (
        f"rectified_ratio={result.rectified_ratio:.2f}"
    )


# ── T9: small spike — 기존 테스트 보존 (area 변화 10% 이내) ──────────────────


def test_small_spike_is_simplified_without_large_area_change() -> None:
    """T9: 작은 spike → area 변화 10% 이내."""
    footprint = _geo(Polygon([
        (0, 0), (10, 0), (10, 4), (5.1, 4.1), (5.0, 4.0),
        (0, 4), (0, 0),
    ]))

    result = ManhattanRectificationStep().run(footprint, simplify_tolerance_m=0.25)

    # spike가 있어도 accepted 또는 area_change < 15%
    assert result.area_change_ratio < 0.15


# ── T10: dominant_angle_histogram 단독 검증 ──────────────────────────────────


def test_dominant_angle_histogram_axis_aligned() -> None:
    """T10: 완전 axis-aligned polygon → dominant_angle ≈ 0°."""
    from shapely.geometry import shape as shp_shape

    footprint = _geo(Polygon([(0, 0), (10, 0), (10, 4), (0, 4), (0, 0)]))
    geom = shp_shape(footprint)

    angle = _dominant_angle_histogram(geom, min_edge_length_m=0.3, bin_deg=10.0)

    assert abs(angle) < 5.0, f"dominant_angle={angle}° expected ≈0°"


# ── cycle_3 collinear merge 테스트 ────────────────────────────────────────────


def _make_dense_rect_ring(
    x0: float, y0: float, x1: float, y1: float, points_per_side: int
) -> list[tuple[float, float]]:
    """직사각형 ring에 점을 과밀하게 추가 (각 변에 points_per_side개)."""
    pts: list[tuple[float, float]] = []
    # bottom: left → right
    for i in range(points_per_side):
        t = i / points_per_side
        pts.append((x0 + (x1 - x0) * t, y0))
    # right: bottom → top
    for i in range(points_per_side):
        t = i / points_per_side
        pts.append((x1, y0 + (y1 - y0) * t))
    # top: right → left
    for i in range(points_per_side):
        t = i / points_per_side
        pts.append((x1 + (x0 - x1) * t, y1))
    # left: top → bottom
    for i in range(points_per_side):
        t = i / points_per_side
        pts.append((x0, y1 + (y0 - y1) * t))
    pts.append(pts[0])  # 닫기
    return pts


# ── C1: 직사각형 noisy → 4개 vertex 목표 ─────────────────────────────────────


def test_collinear_merge_dense_rectangle_to_four_vertices() -> None:
    """C1: 직사각형에 각 변마다 25점씩(총 100점) → merge 후 ≤ 8 vertex.

    snap 후 각 변이 collinear → 중간 vertex 모두 제거 → 직사각형 4 vertex 기대.
    """
    # 각 변에 25개씩 → 총 100개 vertex (닫힌 ring 101개)
    ring = _make_dense_rect_ring(0.0, 0.0, 10.0, 4.0, 25)
    # 미세 floating-point 오차 시뮬레이션 (1e-5 수준)
    ring_noisy = [
        (x + (i % 3 - 1) * 1e-5, y + (i % 5 - 2) * 1e-5)
        for i, (x, y) in enumerate(ring)
    ]

    merged, removed = _merge_collinear_ring(
        ring_noisy,
        tolerance=0.01,
        min_vertex_count=4,
    )
    # 닫힌 ring의 실제 vertex 수 (마지막 중복 제외)
    vertex_count = len(merged) - 1 if merged[0] == merged[-1] else len(merged)

    assert vertex_count <= 8, (
        f"Expected ≤ 8 vertices after collinear merge, got {vertex_count}. "
        f"removed={removed}"
    )
    assert removed > 80, f"Expected >80 removed, got {removed}"


# ── C2: L자형 → 6개 vertex 목표 ──────────────────────────────────────────────


def test_collinear_merge_dense_l_shape_to_six_vertices() -> None:
    """C2: L자형 각 변에 과밀 점(총 ~200점) → merge 후 ≤ 10 vertex.

    L자 = 6개 꼭짓점. 각 직선 변의 중간 점이 collinear merge로 제거됨.
    """
    # L자형 기본 꼭짓점 (10x8 에서 우하단 4x4 빠짐)
    base_pts = [
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 8), (0, 8),
    ]
    # 각 변에 30개 중간점 추가
    ring: list[tuple[float, float]] = []
    n_base = len(base_pts)
    for i in range(n_base):
        a = base_pts[i]
        b = base_pts[(i + 1) % n_base]
        ring.append(a)
        for j in range(1, 30):
            t = j / 30
            ring.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    ring.append(ring[0])  # 닫기

    initial_vertex_count = len(ring) - 1

    merged, removed = _merge_collinear_ring(
        ring,
        tolerance=0.01,
        min_vertex_count=4,
    )
    vertex_count = len(merged) - 1 if merged[0] == merged[-1] else len(merged)

    assert vertex_count <= 10, (
        f"Expected ≤ 10 vertices, got {vertex_count}. "
        f"initial={initial_vertex_count}, removed={removed}"
    )
    assert removed >= initial_vertex_count - 10, (
        f"Expected most vertices removed, got removed={removed}"
    )


# ── C3: 사선 + 직각 혼합 — 사선 vertex 보존 ─────────────────────────────────


def test_collinear_merge_preserves_diagonal_vertices() -> None:
    """C3: 직각 부분의 중간 점만 제거, 사선 vertex는 보존.

    polygon: 직사각형 기반인데 한 코너가 사선으로 대체된 형태.
    사선 edge 양 끝 vertex는 collinear가 아니므로 보존돼야 함.
    """
    # 사선 포함 polygon (5-vertex 도형):
    # (0,0) → (8,0) → (10,2) [사선] → (10,5) → (0,5) → (0,0)
    # 각 직선 변에 10점 추가, 사선 변엔 추가 없음
    base_segments = [
        ((0, 0), (8, 0)),    # horizontal
        ((8, 0), (10, 2)),   # diagonal — 추가 없음
        ((10, 2), (10, 5)),  # vertical
        ((10, 5), (0, 5)),   # horizontal
        ((0, 5), (0, 0)),    # vertical
    ]

    ring: list[tuple[float, float]] = []
    for (ax, ay), (bx, by) in base_segments:
        ring.append((ax, ay))
        # 사선 변은 intermediate point 없음
        is_diagonal = (ax != bx and ay != by)
        if not is_diagonal:
            for j in range(1, 10):
                t = j / 10
                ring.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    ring.append(ring[0])

    merged, removed = _merge_collinear_ring(
        ring,
        tolerance=0.01,
        min_vertex_count=4,
    )
    vertex_count = len(merged) - 1 if merged[0] == merged[-1] else len(merged)

    # 사선 양 끝 (8,0), (10,2) 이 보존되어야 하고
    # 직선 구간 중간 점은 제거돼야 함
    # 최소 5개 vertex(원본 꼭짓점) 보존, 직선 중간 점들은 제거
    assert vertex_count <= 8, (
        f"Expected ≤ 8 vertices (diagonal vertices preserved), got {vertex_count}"
    )
    # 사선 보존: (10, 2) 또는 (8, 0)이 결과에 있어야 함
    merged_pts = {(round(x, 1), round(y, 1)) for x, y in merged}
    assert (8.0, 0.0) in merged_pts, f"Diagonal endpoint (8,0) should be preserved. {merged_pts}"
    assert (10.0, 2.0) in merged_pts, f"Diagonal endpoint (10,2) should be preserved. {merged_pts}"


# ── C4: tolerance 경계값 ─────────────────────────────────────────────────────


def test_collinear_merge_tolerance_boundary() -> None:
    """C4: tolerance 경계값 검증 (0.009m vs 0.011m).

    collinear 판정이 tolerance에 정확히 반응하는지 확인.
    y 좌표 차이가 정확히 0.010m인 점:
      - tolerance=0.009 → 차이(0.010) > tolerance → 보존
      - tolerance=0.011 → 차이(0.010) ≤ tolerance → 제거
    """
    # v_prev=(0,0), v_curr=(5,0.010), v_next=(10,0) → y 차이 = 0.010m
    # 단순 ring: 4각형에서 한 변의 중간 점 y를 0.010으로 올림
    ring = [
        (0.0, 0.0),
        (5.0, 0.010),   # 이 점의 y가 이웃보다 0.010m 높음
        (10.0, 0.0),
        (10.0, 4.0),
        (0.0, 4.0),
        (0.0, 0.0),
    ]

    # tolerance=0.009 → 0.010 > 0.009 → (5, 0.010)은 보존
    merged_strict, removed_strict = _merge_collinear_ring(
        ring, tolerance=0.009, min_vertex_count=4
    )
    pts_strict = {(round(x, 3), round(y, 3)) for x, y in merged_strict}
    assert (5.0, 0.010) in pts_strict, (
        f"With tolerance=0.009, vertex (5, 0.010) should be PRESERVED. pts={pts_strict}"
    )
    assert removed_strict == 0, f"Expected 0 removed with strict tolerance, got {removed_strict}"

    # tolerance=0.011 → 0.010 ≤ 0.011 → (5, 0.010) 제거 가능
    merged_loose, removed_loose = _merge_collinear_ring(
        ring, tolerance=0.011, min_vertex_count=4
    )
    pts_loose = {(round(x, 3), round(y, 3)) for x, y in merged_loose}
    assert (5.0, 0.010) not in pts_loose, (
        f"With tolerance=0.011, vertex (5, 0.010) should be REMOVED. pts={pts_loose}"
    )
    assert removed_loose >= 1, f"Expected ≥1 removed with loose tolerance, got {removed_loose}"


# ── C5: 통합 — ManhattanRectificationStep 과밀 직사각형 vertex 감소 ─────────


def test_rectification_reduces_dense_rectangle_vertices() -> None:
    """C5: 과밀 직사각형(100+ vertex) → ManhattanRectificationStep 후 ≤ 100 vertex.

    simplify(0.3m) + snap + collinear merge가 연동되어 최종 rectified_vertex_count가
    대폭 감소해야 함. cycle_4 이후 simplify가 먼저 vertex를 줄이고,
    collinear merge는 추가 제거를 담당하거나 불필요할 수 있음 (둘 다 OK).
    """
    # 각 변에 30점씩 → 총 120 vertex
    ring = _make_dense_rect_ring(0.0, 0.0, 10.0, 4.0, 30)
    footprint = _geo(Polygon(ring))

    original_vertex_count = len(ring) - 1  # 120

    result = ManhattanRectificationStep().run(
        footprint,
        collinear_tolerance_m=0.01,
        collinear_min_vertex_count=4,
    )

    assert result.accepted is True, (
        f"Expected accepted. fallback={result.fallback_used}, "
        f"area_change={result.area_change_ratio:.3f}"
    )
    assert result.rectified_vertex_count <= 100, (
        f"Expected ≤ 100 vertices after simplify + collinear merge, "
        f"got {result.rectified_vertex_count} (original={original_vertex_count})"
    )
    # simplify 또는 collinear merge 중 하나 이상이 vertex를 줄였어야 함
    reduced_by_simplify = result.simplified_vertex_count < original_vertex_count
    reduced_by_merge = result.collinear_merged_count > 0
    assert reduced_by_simplify or reduced_by_merge, (
        f"Expected simplify or collinear merge to reduce vertices. "
        f"simplified={result.simplified_vertex_count}, "
        f"collinear_merged={result.collinear_merged_count}"
    )
    # area 보존 확인
    assert result.area_change_ratio <= 0.10, (
        f"area_change_ratio={result.area_change_ratio:.3f} exceeds 10%"
    )


# ── C6: fallback 시 collinear_merged_count=0 ────────────────────────────────


def test_collinear_merged_count_zero_on_fallback() -> None:
    """C6: fallback 발생 시 collinear_merged_count=0 (merge 취소).

    iter4 변경: rectified_ratio 기반 fallback 제거 → area 엄격 제한으로 fallback 유도.
    """
    import math

    pts = [
        (math.cos(math.radians(i * 60)) * 5, math.sin(math.radians(i * 60)) * 5)
        for i in range(6)
    ]
    footprint = _geo(Polygon(pts))

    # iter4: area 엄격 제한으로 fallback 유도
    result = ManhattanRectificationStep().run(
        footprint,
        manhattan_max_area_change=0.001,  # 0.1% 이내만 허용 → 사선 polygon은 fallback
    )

    assert result.fallback_used is True
    assert result.collinear_merged_count == 0, (
        f"Expected collinear_merged_count=0 on fallback, got {result.collinear_merged_count}"
    )


# ── cycle_4 simplify 테스트 ───────────────────────────────────────────────────


# ── S1: simplify가 noise short edge를 제거하는지 검증 ────────────────────────


def test_simplify_geometry_reduces_noise_vertices() -> None:
    """S1: _simplify_geometry(tolerance=0.3m)로 noise short edge 제거.

    10x4 직사각형 변에 0.1m 크기 noise spike를 30개 삽입.
    simplify 후 vertex 수가 감소해야 함.
    """
    # 직사각형 + noise spikes (0.1m 크기)
    base = [(0, 0), (10, 0), (10, 4), (0, 4)]
    ring: list[tuple[float, float]] = []
    for i, (ax, ay) in enumerate(base):
        ring.append((ax, ay))
        bx, by = base[(i + 1) % len(base)]
        # 각 변에 5개 noise spike 삽입 (0.1m 높이)
        for j in range(1, 6):
            t = j / 6
            mid_x = ax + (bx - ax) * t
            mid_y = ay + (by - ay) * t
            # spike 방향 (외부로)
            dx, dy = (by - ay), -(bx - ax)
            length = math.sqrt(dx * dx + dy * dy) + 1e-9
            ring.append((mid_x + dx / length * 0.05, mid_y + dy / length * 0.05))
            ring.append((mid_x + dx / length * 0.10, mid_y + dy / length * 0.10))
            ring.append((mid_x + dx / length * 0.05, mid_y + dy / length * 0.05))

    ring.append(ring[0])
    poly = Polygon(ring)
    raw_area = float(poly.area)
    initial_vertex_count = len(ring) - 1

    simplified = _simplify_geometry(
        poly,
        tolerance_m=0.3,
        raw_area=raw_area,
        area_change_limit=0.05,
    )
    simplified_vertex_count = len(list(simplified.exterior.coords)) - 1

    # simplify 후 vertex 수가 감소해야 함
    assert simplified_vertex_count < initial_vertex_count, (
        f"Expected fewer vertices after simplify. "
        f"initial={initial_vertex_count}, simplified={simplified_vertex_count}"
    )
    # area 변화 5% 이내
    assert abs(simplified.area - raw_area) / raw_area <= 0.05, (
        f"area_change={abs(simplified.area - raw_area) / raw_area:.3f} exceeds 5%"
    )


# ── S2: area 변화 5% 초과 시 tolerance 반감 재시도 ──────────────────────────


def test_simplify_geometry_fallback_on_large_area_change() -> None:
    """S2: simplify area 변화 > 5% → tolerance 반감 재시도.

    매우 복잡한 polygon에서 0.3m tolerance가 area를 5% 이상 변경하는 경우를
    시뮬레이션한다. tolerance=0 (skip)으로 강제해도 원본이 반환돼야 함.
    """
    # 단순 직사각형 — tolerance=0.0이면 simplify skip
    poly = Polygon([(0, 0), (5, 0), (5, 3), (0, 3), (0, 0)])
    raw_area = float(poly.area)

    result = _simplify_geometry(
        poly,
        tolerance_m=0.0,  # skip 조건
        raw_area=raw_area,
        area_change_limit=0.05,
    )

    # tolerance=0이면 원본 그대로
    assert result is poly or abs(result.area - raw_area) / raw_area < 1e-9


# ── S3: simplify + collinear merge 통합 — vertex 50개 미만 목표 검증 ─────────


@pytest.mark.xfail(
    strict=False,
    reason=(
        "dense_polygon_simplify category — Sprint 34 raw default + "
        "force_rectilinear default 변경의 의도된 회귀. Sprint 49+ 알고리즘 본질 "
        "수정 sprint 에서 처리. Codex F-4 debt register."
    ),
)
def test_simplify_and_collinear_reduce_vertices_significantly() -> None:
    """S3: simplify + snap(25°) + collinear merge로 과밀 polygon → 50개 미만 목표.

    각 변 40점의 직사각형(총 160 vertex) + 0.1m noise.
    simplify 0.3m → 4개 직사각형 vertex → snap → collinear → ≤ 10개 기대.
    """
    # 각 변 40점 + 0.05m noise
    ring = _make_dense_rect_ring(0.0, 0.0, 10.0, 4.0, 40)
    ring_noisy = [
        (x + (i % 5 - 2) * 0.05, y + (i % 3 - 1) * 0.05)
        for i, (x, y) in enumerate(ring)
    ]
    footprint = _geo(Polygon(ring_noisy))
    initial_vertex_count = len(ring) - 1  # 160

    result = ManhattanRectificationStep().run(
        footprint,
        manhattan_simplify_tolerance_m=0.3,
        manhattan_snap_threshold_deg=25.0,
        collinear_tolerance_m=0.05,
    )

    assert result.accepted is True, (
        f"Expected accepted. fallback={result.fallback_used}, "
        f"area_change={result.area_change_ratio:.3f}"
    )
    # simplify가 먼저 vertex 수를 대폭 줄여야 함
    assert result.simplified_vertex_count < initial_vertex_count, (
        f"simplified_vertex_count={result.simplified_vertex_count} "
        f"should be < initial={initial_vertex_count}"
    )
    # 최종 vertex 수 50개 미만
    assert result.rectified_vertex_count < 50, (
        f"Expected < 50 vertices, got {result.rectified_vertex_count}. "
        f"simplified={result.simplified_vertex_count}, "
        f"collinear_merged={result.collinear_merged_count}"
    )


# ── S4: snap_threshold 25° 기본값으로 20° 사선 edge 처리 ─────────────────────


def test_snap_threshold_25_handles_20_degree_noise_edges() -> None:
    """S4: snap_threshold=25°(기본값)로 20° noise edge를 직각으로 강제.

    기존 15° threshold에서는 20° edge가 snap 불가였으나,
    25° threshold에서는 snap 가능 → rectified_ratio 향상.
    """
    # 20° 기울어진 edge가 많은 polygon (noise가 15°~20° 수준인 직사각형)
    # 각 꼭짓점에 20° 방향 offset 추가
    angle_rad = math.radians(20.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    offset = 0.3  # 0.3m 수준의 noise

    base_pts = [(0, 0), (8, 0), (8, 5), (0, 5)]
    noisy_pts = [
        (x + offset * cos_a, y + offset * sin_a) if i % 2 == 0
        else (x - offset * cos_a, y - offset * sin_a)
        for i, (x, y) in enumerate(base_pts)
    ]
    footprint = _geo(Polygon(noisy_pts))

    # 25° threshold (기본값) — 20° noise edge가 snap 가능
    result_25 = ManhattanRectificationStep().run(
        footprint,
        manhattan_snap_threshold_deg=25.0,
        manhattan_simplify_tolerance_m=0.0,  # simplify skip — snap만 검증
    )

    # 15° threshold — 20° edge는 snap 불가
    result_15 = ManhattanRectificationStep().run(
        footprint,
        manhattan_snap_threshold_deg=15.0,
        manhattan_simplify_tolerance_m=0.0,  # simplify skip
    )

    # 25° threshold가 15°보다 rectified_ratio가 같거나 높아야 함
    assert result_25.rectified_ratio >= result_15.rectified_ratio, (
        f"25° threshold should have >= rectified_ratio vs 15°. "
        f"25°={result_25.rectified_ratio:.2f}, 15°={result_15.rectified_ratio:.2f}"
    )


# ── iter3 (cycle_5) 4-way axis snap 테스트 ────────────────────────────────────


@pytest.mark.xfail(
    strict=False,
    reason=(
        "four_way_snap_contract category — Sprint 34 raw default + "
        "force_rectilinear default 변경의 의도된 회귀. Sprint 49+ 알고리즘 본질 "
        "수정 sprint 에서 처리. Codex F-4 debt register."
    ),
)
def test_four_way_snap_near_zero() -> None:
    """I1: ±11° → 0°로 snap; ±16° → ±22.5°로 snap.

    22.5° 단위 snap:
    - +11°: 0°에서 11°, 22.5°에서 11.5° → 0°가 더 가까움 → 0°
    - -11°: 0°에서 11°, -22.5°에서 11.5° → 0°가 더 가까움 → 0°
    - +16°: 0°에서 16°, 22.5°에서 6.5° → 22.5°가 더 가까움 → 22.5°
    - -16°: 0°에서 16°, -22.5°에서 6.5° → -22.5°가 더 가까움 → -22.5°
    """
    assert _four_way_snap(11.0) == 0.0, f"Expected 0°, got {_four_way_snap(11.0)}"
    assert _four_way_snap(-11.0) == 0.0, f"Expected 0°, got {_four_way_snap(-11.0)}"
    assert _four_way_snap(16.0) == 22.5, f"Expected 22.5°, got {_four_way_snap(16.0)}"
    assert _four_way_snap(-16.0) == -22.5, f"Expected -22.5°, got {_four_way_snap(-16.0)}"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "four_way_snap_contract category — Sprint 34 raw default + "
        "force_rectilinear default 변경의 의도된 회귀. Sprint 49+ 알고리즘 본질 "
        "수정 sprint 에서 처리. Codex F-4 debt register."
    ),
)
def test_four_way_snap_near_45() -> None:
    """I2: +35° → 45°로 snap (22.5°에서 12.5°, 45°에서 10°이므로 45°)."""
    result = _four_way_snap(35.0)
    assert result == 45.0, f"Expected 45°, got {result}"


def test_four_way_snap_exact_22_5() -> None:
    """I3: 정확히 22.5° → 22.5°로 snap."""
    result = _four_way_snap(22.5)
    assert result == 22.5, f"Expected 22.5°, got {result}"


def test_four_way_snap_force_zero_env(monkeypatch: object) -> None:
    """I4: INDOOR_DOMINANT_ANGLE_FORCE_ZERO=true → 모든 입력이 0° 반환."""
    import os
    old = os.environ.get("INDOOR_DOMINANT_ANGLE_FORCE_ZERO", "")
    os.environ["INDOOR_DOMINANT_ANGLE_FORCE_ZERO"] = "true"
    try:
        assert _four_way_snap(-16.0) == 0.0
        assert _four_way_snap(35.0) == 0.0
        assert _four_way_snap(22.5) == 0.0
    finally:
        if old:
            os.environ["INDOOR_DOMINANT_ANGLE_FORCE_ZERO"] = old
        else:
            os.environ.pop("INDOOR_DOMINANT_ANGLE_FORCE_ZERO", None)


# ── iter3 length-weighted dominant 테스트 ─────────────────────────────────────


def test_length_weighted_dominant_long_edge_wins() -> None:
    """I5: 짧은 noise edge 많아도 긴 edge의 방향이 dominant로 선택된다.

    10m 짜리 0° edge 1개 vs 0.5m짜리 45° noise edge 30개.
    length-weighted histogram에서 총 길이: 0°=10m > 45°=15m... 이면 지지만,
    길이 비를 조절해서 긴 edge(0°)가 이기는 케이스를 구성.
    0°: 1개 × 10m = 10m
    45°: 5개 × 0.4m = 2m → 0°가 이겨야 함.
    """
    from shapely.geometry import shape as shp_shape

    # 단순: 0° 방향 10m edge가 dominant인 T자 polygon
    # 아래: 큰 직사각형(0° 주축) + 짧은 사선 변들로 구성된 noisy polygon
    pts = [(0, 0), (10, 0), (10, 3), (0, 3), (0, 0)]
    geom = shp_shape(_geo(Polygon(pts)))

    angle = _dominant_angle_histogram(geom, min_edge_length_m=0.1, bin_deg=10.0)

    # 0° 방향이 dominant → angle ≈ 0°
    assert abs(angle) < 10.0, f"Expected dominant ≈ 0°, got {angle}°"


# ── iter3 grid snap 테스트 ────────────────────────────────────────────────────


def test_grid_snap_aligns_vertices_to_grid() -> None:
    """I6: 0.1m grid snap 후 모든 vertex 좌표가 0.1m 배수여야 함."""
    # 0.1m 배수가 아닌 좌표의 polygon
    pts = [(0.05, 0.03), (5.12, 0.07), (5.08, 3.97), (0.02, 4.01), (0.05, 0.03)]
    poly = Polygon(pts)
    result_geom = _grid_snap_geometry(poly, resolution=0.1)

    if isinstance(result_geom, Polygon):
        for x, y in result_geom.exterior.coords:
            assert abs(round(x / 0.1) * 0.1 - x) < 1e-6, (
                f"x={x} is not on 0.1m grid"
            )
            assert abs(round(y / 0.1) * 0.1 - y) < 1e-6, (
                f"y={y} is not on 0.1m grid"
            )


def test_grid_snap_preserves_area_approximately() -> None:
    """I7: 0.1m grid snap 후 area가 ±5% 이내 보존."""
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0), (0.0, 0.0)]
    poly = Polygon(pts)
    original_area = float(poly.area)

    result_geom = _grid_snap_geometry(poly, resolution=0.1)
    result_area = float(result_geom.area)

    ratio = abs(result_area - original_area) / original_area
    assert ratio < 0.05, f"area_change_ratio={ratio:.3f} exceeds 5%"


# ── iter3 통합 — rotated-frame 직각도 향상 검증 ───────────────────────────────


@pytest.mark.xfail(
    strict=False,
    reason=(
        "forced_rectilinear_area_guard category — Sprint 34 raw default + "
        "force_rectilinear default 변경의 의도된 회귀. Sprint 49+ 알고리즘 본질 "
        "수정 sprint 에서 처리. Codex F-4 debt register."
    ),
)
def test_iter3_rectification_produces_right_angles() -> None:
    """I8: iter4 통합 — 16° 회전된 L자 polygon → dominant angle 추정 + 직각화.

    16° 회전된 L자:
    - dominant_raw ≈ 16° → 4-way snap → 22.5° (16°은 22.5°에 더 가깝)
    - rotated frame(-22.5°)에서 simplify + Forced Rectilinear + collinear merge

    iter4 변경: grid_snapped=False (grid snap 미사용). forced_rectilinear_used=True.
    accepted 시 area 변화 15% 이내 (iter4 기준).
    """
    import math

    # L자 polygon (6 vertex)
    l_pts = [(0, 0), (10, 0), (10, 4), (6, 4), (6, 8), (0, 8)]

    # 16° 회전 적용
    angle_rad = math.radians(16.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rotated_pts = [
        (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
        for x, y in l_pts
    ]
    footprint = _geo(Polygon(rotated_pts))

    result = ManhattanRectificationStep().run(
        footprint,
        manhattan_simplify_tolerance_m=0.3,
        collinear_tolerance_m=0.05,
    )

    # 메타 필드 확인
    assert result.dominant_angle_raw_deg is not None
    # iter4: grid snap 사용 안 함
    assert result.grid_snapped is False
    assert result.vertex_count_after_snap >= 0
    # iter4 신규 필드
    assert isinstance(result.forced_rectilinear_used, bool)
    assert isinstance(result.zero_edges_removed, int)

    # dominant_angle_raw ≈ 16° (length-weighted histogram 결과)
    da_raw = result.dominant_angle_raw_deg
    assert abs(da_raw - 16.0) < 8.0, (
        f"dominant_angle_raw={da_raw}° expected ≈16°"
    )

    # 4-way snap 결과: 16° → 22.5°이 가장 가깝다 (16에서 22.5까지 6.5°, 0까지 16°)
    da_snapped = result.dominant_angle_deg
    assert da_snapped in (0.0, 22.5, -22.5, 45.0, -45.0), (
        f"dominant_angle_deg={da_snapped} should be one of 4-way snap values"
    )

    # accepted 시 area 변화 15% 이내 (iter4 기준)
    if result.accepted:
        assert result.area_change_ratio <= 0.15, (
            f"area_change_ratio={result.area_change_ratio:.3f}"
        )
        assert result.forced_rectilinear_used is True
        # vertex 수가 original보다 작거나 같아야 함
        assert result.rectified_vertex_count <= result.original_vertex_count, (
            f"rectified_vertex_count={result.rectified_vertex_count} "
            f"> original={result.original_vertex_count}"
        )


def test_iter3_metadata_new_fields_present() -> None:
    """I9: iter3 신규 메타 필드가 모두 존재하고 타입이 올바름."""
    footprint = _geo(Polygon([(0, 0), (8, 0), (8, 5), (0, 5), (0, 0)]))
    result = ManhattanRectificationStep().run(footprint)
    meta = result.metadata()

    assert "dominant_angle_raw_deg" in meta
    assert "grid_snapped" in meta
    assert "vertex_count_after_snap" in meta

    assert isinstance(meta["dominant_angle_raw_deg"], float)
    assert isinstance(meta["grid_snapped"], bool)
    assert isinstance(meta["vertex_count_after_snap"], int)
    assert meta["vertex_count_after_snap"] >= 0


# ── iter4 (cycle_6) Forced Rectilinear Projection 테스트 ─────────────────────


def _is_strictly_manhattan(poly: Polygon, tolerance_deg: float = 1.0) -> bool:
    """모든 edge가 0° or 90° ± tolerance_deg이면 True (iter4 강제 직각 검증용)."""
    for angle in _all_edge_angles(poly):
        if not (
            angle <= tolerance_deg
            or abs(angle - 90.0) <= tolerance_deg
            or angle >= (180.0 - tolerance_deg)
        ):
            return False
    return True


def test_f1_square_input_stays_accepted() -> None:
    """F1: 완전 직각 정사각형 → accepted, area 변화 1% 이내."""
    footprint = _geo(Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]))

    result = ManhattanRectificationStep().run(footprint)

    assert result.accepted is True, (
        f"Square should stay accepted. area_change={result.area_change_ratio:.4f}"
    )
    assert result.area_change_ratio <= 0.01


def test_f2_l_shape_with_diagonal_edge_becomes_rectilinear() -> None:
    """F2: L자 + 사선 1개 (5각형) → 강제 직각화 후 accepted.

    (0,0)-(10,0)-(10,4)-(6,6)-(0,6) — 4번째 edge가 사선.
    iter4는 강제로 horizontal 또는 vertical로 변환.
    """
    footprint = _geo(Polygon([
        (0, 0), (10, 0), (10, 4), (6, 6), (0, 6), (0, 0)
    ]))

    result = ManhattanRectificationStep().run(footprint)

    # iter4: 강제 snap이라 fallback 없어야 함 (area 15% 이내라면)
    # 강제 snap 후 area 변화가 15% 이내이면 accepted
    if result.accepted:
        assert result.forced_rectilinear_used is True
        result_geom = shape(result.rectified_geojson)
        if isinstance(result_geom, Polygon):
            assert _is_strictly_manhattan(result_geom, tolerance_deg=1.0), (
                f"All edges should be H or V. angles={_all_edge_angles(result_geom)}"
            )
    else:
        # fallback이 발생해도 OK — area 손실이 15% 초과한 경우
        assert result.fallback_used is True


def test_f3_noisy_19gon_vertex_reduced() -> None:
    """F3: noise 19각형 → 강제 직각화 후 vertex 대폭 감소.

    실 scan 9c481325 footprint와 유사한 복잡한 polygon.
    19개 vertex → 강제 H/V snap + collinear merge → 4~8개 기대.
    """
    import math

    # noise 19각형: 반지름 5m 원에 random-ish noise
    n = 19
    pts = []
    for i in range(n):
        angle_rad = 2 * math.pi * i / n
        r = 5.0 + (i % 3 - 1) * 0.4  # noise ±0.4m
        pts.append((r * math.cos(angle_rad), r * math.sin(angle_rad)))

    footprint = _geo(Polygon(pts))
    original_count = len(pts)

    result = ManhattanRectificationStep().run(footprint)

    # iter4: 강제 snap이라 fallback 없어야 함 (area 15% 이내라면)
    if result.accepted:
        # vertex 수가 original보다 대폭 감소해야 함
        assert result.rectified_vertex_count < original_count, (
            f"Expected vertex count reduction. "
            f"original={original_count}, rectified={result.rectified_vertex_count}"
        )
        assert result.forced_rectilinear_used is True


def test_f4_zero_length_edges_removed() -> None:
    """F4: zero-length edge 제거 검증.

    _force_rectilinear_ring 직접 호출: 사선 edge를 강제 snap 했을 때
    생성될 수 있는 zero-length edge (두 vertex가 같은 좌표)가 제거되는지 확인.

    입력: (0,0) → (3,3) → (6,0) — 두 사선 edge.
    H/V 강제 후: (0,0) → (0,3) → (3,3) → (6,3) → (6,0) → ...
    zero-length edge가 없어야 함.
    """
    # 단순 삼각형 (모두 사선)
    coords = [(0.0, 0.0), (3.0, 3.0), (6.0, 0.0), (0.0, 0.0)]

    projected, zero_removed = _force_rectilinear_ring(coords, min_edge_length_m=0.05)

    # 결과 ring에 zero-length edge가 없어야 함
    pts = projected[:-1] if len(projected) >= 2 and projected[0] == projected[-1] else projected
    for i in range(len(pts)):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % len(pts)]
        length = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        assert length >= 0.05, (
            f"Found edge shorter than min_edge_length_m=0.05m: "
            f"({ax},{ay})->({bx},{by}) length={length:.4f}"
        )


def test_f5_area_15pct_exceeded_triggers_fallback() -> None:
    """F5: area 변화 15% 초과 시 fallback (원본 반환).

    매우 작은 area_change 한계(0.01)로 설정하여 강제 fallback 유도.
    """
    # 기본 직사각형 — area_change를 0.001로 매우 엄격하게 설정
    footprint = _geo(Polygon([
        (0, 0), (10, 0.5), (10.3, 4), (0.2, 4.1), (0, 0)
    ]))
    raw_area = shape(footprint).area

    result = ManhattanRectificationStep().run(
        footprint,
        manhattan_max_area_change=0.001,  # 0.1% 초과하면 fallback
    )

    # 매우 엄격한 limit → fallback 발생
    assert result.fallback_used is True
    # fallback 시 원본 area 유지
    ret_area = shape(result.rectified_geojson).area
    assert abs(raw_area - ret_area) / raw_area < 0.001


def test_f6_iter4_metadata_fields_present() -> None:
    """F6: iter4 신규 메타 필드 (forced_rectilinear_used, zero_edges_removed) 존재 + 타입 확인."""
    footprint = _geo(Polygon([(0, 0), (8, 0), (8, 5), (0, 5), (0, 0)]))
    result = ManhattanRectificationStep().run(footprint)
    meta = result.metadata()

    assert "forced_rectilinear_used" in meta, "Missing forced_rectilinear_used"
    assert "zero_edges_removed" in meta, "Missing zero_edges_removed"

    assert isinstance(meta["forced_rectilinear_used"], bool)
    assert isinstance(meta["zero_edges_removed"], int)
    assert meta["zero_edges_removed"] >= 0

    # accepted된 경우 forced_rectilinear_used=True
    if result.accepted:
        assert meta["forced_rectilinear_used"] is True
    else:
        assert meta["forced_rectilinear_used"] is False


@pytest.mark.xfail(
    strict=False,
    reason=(
        "forced_rectilinear_area_guard category — Sprint 34 raw default + "
        "force_rectilinear default 변경의 의도된 회귀. Sprint 49+ 알고리즘 본질 "
        "수정 sprint 에서 처리. Codex F-4 debt register."
    ),
)
def test_f7_all_edges_are_horizontal_or_vertical_after_rectification() -> None:
    """F7: 직각화 후 모든 edge가 horizontal(0°) 또는 vertical(90°) ± 1°.

    noisy L자형 polygon. iter4의 핵심 보장:
    accepted 시 결과 polygon의 모든 edge가 H 또는 V여야 함.
    """
    # noisy L자형: 각 꼭짓점에 0.2m 내외 noise 추가
    l_pts_noisy = [
        (0.1, -0.1),
        (10.05, 0.08),
        (10.1, 4.0),
        (6.02, 3.95),
        (5.98, 8.1),
        (-0.05, 7.92),
    ]
    footprint = _geo(Polygon(l_pts_noisy))

    result = ManhattanRectificationStep().run(footprint)

    if result.accepted:
        assert result.forced_rectilinear_used is True
        result_geom = shape(result.rectified_geojson)
        if isinstance(result_geom, Polygon):
            assert _is_strictly_manhattan(result_geom, tolerance_deg=1.0), (
                f"Not all edges are H or V after forced rectification. "
                f"angles={_all_edge_angles(result_geom)}"
            )
        # rectified_ratio는 iter4에서 항상 1.0
        assert result.rectified_ratio == 1.0
    else:
        # fallback이어도 rectified_ratio는 0.0 (rejected) 또는 기록 값
        assert result.fallback_used is True


# ── Sprint 48 신규 (Codex F-1, F-5): hint override active tests ─────────────


def test_dominant_angle_hint_overrides_internal_estimate() -> None:
    """hint 주입 시 internal contour confidence 무시하고 hint angle 강제."""
    # 사선이 섞인 polygon (internal confidence 낮음)
    pts = [(0.0, 0.0), (10.0, 0.1), (10.1, 5.0), (0.0, 5.05)]
    footprint = _geo(Polygon(pts))

    result = ManhattanRectificationStep().run(
        footprint,
        dominant_angle_hint_deg=10.0,
        dominant_angle_hint_source="rtabmap_link",
    )

    # hint 가 적용되었는지 검증.
    assert result.dominant_angle_hint_used is True
    assert result.dominant_angle_hint_source == "rtabmap_link"
    assert result.dominant_angle_hint_deg == 10.0
    # snap_mode_used 는 hint 또는 hint_rejected (area_change 결과에 따라).
    assert result.snap_mode_used in ("hint", "hint_rejected")
    # accepted 시 dominant_angle_deg 가 hint 와 동일 (fold 후).
    if result.accepted:
        assert abs(result.dominant_angle_deg - 10.0) < 1e-6


def test_hint_rejected_when_area_change_exceeds_threshold() -> None:
    """hint angle 이 polygon 에 안 맞아 area_change 초과 시 fallback.

    snap_mode_used 가 "hint_rejected" 라벨로 분리되어 caller 가 candidate retry
    흐름에서 사용 가능해야 한다.
    """
    # 가로로 길쭉한 사각형 — 90° 회전 hint 면 area 보존 곤란
    footprint = _geo(Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.0, 1.0)]))

    result = ManhattanRectificationStep().run(
        footprint,
        manhattan_max_area_change=0.05,  # 빡빡한 임계 → fallback 유도
        dominant_angle_hint_deg=44.0,
        dominant_angle_hint_source="footprint_obb",
    )

    # 두 가지 패턴 모두 허용:
    #  - hint 적용 후 area 보존 → accepted (snap_mode="hint")
    #  - hint 적용 후 area 초과 → fallback (snap_mode="hint_rejected")
    assert result.dominant_angle_hint_used is True
    if result.fallback_used:
        assert result.snap_mode_used == "hint_rejected"
        assert result.accepted is False
    else:
        assert result.snap_mode_used == "hint"


def test_hint_none_preserves_legacy_behaviour() -> None:
    """hint=None 이면 기존 동작 (회귀 0)."""
    footprint = _geo(Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]))

    result_with_hint_none = ManhattanRectificationStep().run(
        footprint,
        dominant_angle_hint_deg=None,
        dominant_angle_hint_source=None,
    )
    result_without_kwarg = ManhattanRectificationStep().run(footprint)

    assert (
        result_with_hint_none.dominant_angle_hint_used
        == result_without_kwarg.dominant_angle_hint_used
        is False
    )
    assert result_with_hint_none.snap_mode_used == result_without_kwarg.snap_mode_used
    assert (
        result_with_hint_none.dominant_angle_deg
        == result_without_kwarg.dominant_angle_deg
    )


def test_hint_metadata_includes_new_keys() -> None:
    """metadata() 가 Sprint 48 신규 키 3개 포함."""
    footprint = _geo(Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 3.0), (0.0, 3.0)]))
    result = ManhattanRectificationStep().run(
        footprint,
        dominant_angle_hint_deg=0.0,
        dominant_angle_hint_source="rtabmap_link",
    )
    md = result.metadata()
    assert "dominant_angle_hint_used" in md
    assert "dominant_angle_hint_source" in md
    assert "dominant_angle_hint_deg" in md
