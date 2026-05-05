"""Sprint 48 — dominant_angle hint chain helper unit tests.

Codex W-2/W-3/W-9/F-5 반영:
- magic angle 하드코딩 금지 — synthetic input 으로 known angle 복원만.
- RTABMap link gate (segment_count, total_length, best_bin_ratio).
- footprint OBB gate (aspect_ratio).
- cross-check (rtabmap vs OBB).
"""
from __future__ import annotations

import math

from shapely.geometry import Polygon, mapping

from indoor_server.application.building.dominant_angle_hint import (
    HintGateThresholds,
    compute_dominant_angle_hints,
    cross_check_hints,
    extract_link_segments_xy,
)


def _rotate(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return (x * c - y * s, x * s + y * c)


def _make_corridor_segments(
    angle_deg: float,
    *,
    n_segments: int = 6,
    segment_length_m: float = 2.0,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """direction angle_deg 의 corridor 형 segment list 생성.

    각 segment 는 origin 에서 시작하지 않고 누적되어 직선 trajectory 를 형성.
    """
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    cum_x, cum_y = 0.0, 0.0
    dx, dy = _rotate(segment_length_m, 0.0, angle_deg)
    for _ in range(n_segments):
        a = (cum_x, cum_y)
        cum_x += dx
        cum_y += dy
        b = (cum_x, cum_y)
        segs.append((a, b))
    return segs


# ── RTABMap link tests ───────────────────────────────────────────────────────


def test_rtabmap_link_recovers_known_angle_30deg() -> None:
    """30° corridor segments → recovered angle ≈ 30° (folded)."""
    segs = _make_corridor_segments(30.0, n_segments=8, segment_length_m=2.0)
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=segs,
        footprint_polygon=None,
    )
    rtabmap_cand = next(c for c in candidates if c.source == "rtabmap_link")
    assert rtabmap_cand.accepted is True
    assert rtabmap_cand.angle_deg is not None
    # fold to (-45, 45]: 30° stays 30°
    assert abs(rtabmap_cand.angle_deg - 30.0) < 5.0


def test_rtabmap_link_recovers_known_angle_neg45deg() -> None:
    """-45° (= 45°) corridor segments → folded result close to -45/45."""
    segs = _make_corridor_segments(-45.0, n_segments=8, segment_length_m=2.0)
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=segs,
        footprint_polygon=None,
    )
    rtabmap_cand = next(c for c in candidates if c.source == "rtabmap_link")
    assert rtabmap_cand.accepted is True
    # Manhattan frame: -45° and 45° 동치. fold 은 둘 중 하나로 정착.
    assert rtabmap_cand.angle_deg is not None
    assert abs(abs(rtabmap_cand.angle_deg) - 45.0) < 5.0


def test_rtabmap_link_unavailable_returns_none() -> None:
    """segments empty → unavailable reject."""
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=[],
        footprint_polygon=None,
    )
    rtabmap_cand = next(c for c in candidates if c.source == "rtabmap_link")
    assert rtabmap_cand.accepted is False
    assert rtabmap_cand.reject_reason == "rtabmap_link_unavailable"


def test_rtabmap_link_gate_fails_on_short_segments() -> None:
    """segment_count 충분해도 total_length 짧으면 reject (W-2)."""
    # 5 segments × 0.05 m → total_length 0.25 m < default 3.0 m
    short_segs = _make_corridor_segments(0.0, n_segments=5, segment_length_m=0.05)
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=short_segs,
        footprint_polygon=None,
    )
    rtabmap_cand = next(c for c in candidates if c.source == "rtabmap_link")
    # length < min_total_length → reject. (segments < 0.05 m floor → 1.0 ratio)
    assert rtabmap_cand.accepted is False
    assert rtabmap_cand.reject_reason is not None
    # reject 사유에 length 또는 zero_weight 가 포함되어야 한다.
    assert (
        "total_length_m" in rtabmap_cand.reject_reason
        or "zero_weight" in rtabmap_cand.reject_reason
    )


def test_rtabmap_link_gate_fails_on_few_segments() -> None:
    """segment_count 미달 reject (W-2)."""
    segs = _make_corridor_segments(0.0, n_segments=2, segment_length_m=2.0)
    thresholds = HintGateThresholds(
        rtabmap_link_min_segments=4,
        rtabmap_link_min_total_length_m=3.0,
        rtabmap_link_min_best_bin_ratio=0.45,
    )
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=segs,
        footprint_polygon=None,
        gate_thresholds=thresholds,
    )
    rtabmap_cand = next(c for c in candidates if c.source == "rtabmap_link")
    assert rtabmap_cand.accepted is False
    assert "segment_count" in (rtabmap_cand.reject_reason or "")


# ── footprint OBB tests ──────────────────────────────────────────────────────


def test_footprint_obb_recovers_axis_aligned_rectangle() -> None:
    """축에 정렬된 직사각형 → OBB long edge angle ≈ 0°."""
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 3.0), (0.0, 3.0)])
    geojson = {"type": "Polygon", "coordinates": list(mapping(poly)["coordinates"])}
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=None,
        footprint_polygon=geojson,
    )
    obb_cand = next(c for c in candidates if c.source == "footprint_obb")
    assert obb_cand.accepted is True
    assert obb_cand.angle_deg is not None
    assert abs(obb_cand.angle_deg) < 5.0
    aspect = obb_cand.metadata.get("aspect_ratio")
    assert isinstance(aspect, float) and aspect >= 1.8


def test_footprint_obb_recovers_rotated_rectangle() -> None:
    """20° 회전된 직사각형 → folded angle ≈ 20°."""
    base = [(0.0, 0.0), (10.0, 0.0), (10.0, 3.0), (0.0, 3.0)]
    rotated = [_rotate(x, y, 20.0) for x, y in base]
    poly = Polygon(rotated)
    geojson = {"type": "Polygon", "coordinates": list(mapping(poly)["coordinates"])}
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=None,
        footprint_polygon=geojson,
    )
    obb_cand = next(c for c in candidates if c.source == "footprint_obb")
    assert obb_cand.accepted is True
    assert obb_cand.angle_deg is not None
    assert abs(obb_cand.angle_deg - 20.0) < 5.0


def test_footprint_obb_gate_fails_on_square() -> None:
    """정사각형 (aspect 1.0) → obb_not_elongated reject (W-3)."""
    poly = Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])
    geojson = {"type": "Polygon", "coordinates": list(mapping(poly)["coordinates"])}
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=None,
        footprint_polygon=geojson,
    )
    obb_cand = next(c for c in candidates if c.source == "footprint_obb")
    assert obb_cand.accepted is False
    assert obb_cand.reject_reason == "obb_not_elongated"


def test_footprint_obb_gate_threshold_override() -> None:
    """obb_min_aspect_ratio 설정 변경 → 같은 polygon 다른 결과."""
    poly = Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 3.0), (0.0, 3.0)])  # aspect ≈ 1.67
    geojson = {"type": "Polygon", "coordinates": list(mapping(poly)["coordinates"])}

    # default 1.8 → reject
    cands_strict = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=None,
        footprint_polygon=geojson,
        gate_thresholds=HintGateThresholds(obb_min_aspect_ratio=1.8),
    )
    obb_strict = next(c for c in cands_strict if c.source == "footprint_obb")
    assert obb_strict.accepted is False

    # 1.5 로 완화 → accept
    cands_loose = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=None,
        footprint_polygon=geojson,
        gate_thresholds=HintGateThresholds(obb_min_aspect_ratio=1.5),
    )
    obb_loose = next(c for c in cands_loose if c.source == "footprint_obb")
    assert obb_loose.accepted is True


def test_footprint_obb_unavailable_when_no_input() -> None:
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=None,
        footprint_polygon=None,
    )
    obb_cand = next(c for c in candidates if c.source == "footprint_obb")
    assert obb_cand.accepted is False
    assert obb_cand.reject_reason == "footprint_unavailable"


# ── graph centerline (Sprint 48 미호출) ─────────────────────────────────────


def test_graph_centerline_helper_unavailable_in_sprint_48() -> None:
    """graph centerline 후보는 floor_pc path 에서 호출되지 않음 — helper signature 만 검증."""
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=None,
        footprint_polygon=None,
        graph_edges=None,
    )
    graph_cand = next(c for c in candidates if c.source == "graph_centerline")
    assert graph_cand.accepted is False
    assert graph_cand.reject_reason == "graph_centerline_unavailable"


# ── cross-check ──────────────────────────────────────────────────────────────


def test_cross_check_no_conflict_when_aligned() -> None:
    """RTABMap 0° + OBB 0° → no conflict."""
    segs = _make_corridor_segments(0.0, n_segments=8, segment_length_m=2.0)
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 3.0), (0.0, 3.0)])
    geojson = {"type": "Polygon", "coordinates": list(mapping(poly)["coordinates"])}
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=segs,
        footprint_polygon=geojson,
    )
    cc = cross_check_hints(candidates)
    assert cc["hint_conflict"] is False
    diff = cc["rtabmap_obb_diff_deg"]
    assert isinstance(diff, float) and diff < 5.0


def test_cross_check_conflict_when_diverged() -> None:
    """RTABMap 30° + OBB 0° → 차이 30° → conflict (default threshold 15°)."""
    segs = _make_corridor_segments(30.0, n_segments=8, segment_length_m=2.0)
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 3.0), (0.0, 3.0)])
    geojson = {"type": "Polygon", "coordinates": list(mapping(poly)["coordinates"])}
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=segs,
        footprint_polygon=geojson,
    )
    cc = cross_check_hints(candidates)
    assert cc["hint_conflict"] is True
    diff = cc["rtabmap_obb_diff_deg"]
    assert isinstance(diff, float) and diff > 15.0


def test_cross_check_no_data_returns_no_conflict() -> None:
    """둘 다 reject 면 conflict 판단 불가 → False."""
    candidates = compute_dominant_angle_hints(
        rtabmap_links_with_node_xy=None,
        footprint_polygon=None,
    )
    cc = cross_check_hints(candidates)
    assert cc["hint_conflict"] is False
    assert cc["rtabmap_obb_diff_deg"] is None


# ── extract_link_segments_xy ─────────────────────────────────────────────────


class _FakeNode:
    def __init__(self, node_id: int, tx: float, ty: float, tz: float) -> None:
        self.node_id = node_id
        self.pose = [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]


class _FakeLink:
    def __init__(self, from_id: int, to_id: int, link_type: int = 0) -> None:
        self.from_id = from_id
        self.to_id = to_id
        self.link_type = link_type


def test_extract_link_segments_skips_non_type_zero() -> None:
    nodes = [_FakeNode(1, 0.0, 0.0, 0.0), _FakeNode(2, 1.0, 0.0, 0.0)]
    links = [_FakeLink(1, 2, link_type=1)]  # non type 0
    out = extract_link_segments_xy(nodes, links)
    assert out == []


def test_extract_link_segments_returns_xy_pairs() -> None:
    nodes = [
        _FakeNode(1, 0.0, 0.0, 0.0),
        _FakeNode(2, 1.0, 0.0, 0.0),
        _FakeNode(3, 1.0, 1.0, 0.0),
    ]
    links = [_FakeLink(1, 2, 0), _FakeLink(2, 3, 0), _FakeLink(1, 1, 0)]
    out = extract_link_segments_xy(nodes, links)
    # self-loop 제거 → 2 segments
    assert len(out) == 2
