"""Sprint 48 — dominant angle hint chain (RTABMap link → graph centerline → footprint OBB).

Floor pointcloud sparse polygon에서 내부 contour로만 dominant axis를 추정하면
confidence가 낮아 fallback이 발생한다 (Sprint 47 evidence: 0.237 / `four_way`
fallback / orthogonality 0.119). 본 모듈은 외부 신호(RTABMap trajectory link,
graph centerline, footprint OBB)를 후보 list 로 산출해 Manhattan rectification
호출자가 candidate retry 흐름을 제어하도록 한다.

Codex critique F-1/W-2/W-3/W-9/W-10 반영:
- 단일 angle 반환이 아니라 후보 list. 각 후보별 source/angle/confidence/gate metadata.
- RTABMap link 품질 gate: segment_count, total_length_m, best_bin_weight_ratio.
- footprint OBB gate: aspect_ratio.
- graph centerline은 helper signature only — Sprint 48 floor_pc path 미호출.
- hint cross-check: RTABMap vs OBB 각도 차이 15° 이상이면 conflict 표시.
- magic angle (-22.0206°) 하드코딩 금지 — 동적 계산만.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import atan2, degrees, sqrt
from pathlib import Path  # noqa: F401  (helper signature 호환)
from typing import Any

import numpy as np
from shapely.geometry import shape as _shape
from shapely.geometry.base import BaseGeometry

from indoor_server.application.building.steps.rtabmap_trajectory import (
    _dominant_angle_from_segments,
    _fold_axis_angle,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HintGateThresholds:
    """후보별 quality gate 임계값.

    모든 값은 cad_report.json.params에 dump되어 reject contract으로 노출된다.
    """

    rtabmap_link_min_segments: int = 4
    rtabmap_link_min_total_length_m: float = 3.0
    rtabmap_link_min_best_bin_ratio: float = 0.45
    obb_min_aspect_ratio: float = 1.8
    cross_check_max_diff_deg: float = 15.0

    def to_metadata(self) -> dict[str, object]:
        return {
            "rtabmap_link_min_segments": self.rtabmap_link_min_segments,
            "rtabmap_link_min_total_length_m": self.rtabmap_link_min_total_length_m,
            "rtabmap_link_min_best_bin_ratio": self.rtabmap_link_min_best_bin_ratio,
            "obb_min_aspect_ratio": self.obb_min_aspect_ratio,
            "cross_check_max_diff_deg": self.cross_check_max_diff_deg,
        }


@dataclass(frozen=True)
class HintCandidate:
    """단일 hint 후보.

    accepted 는 gate 통과 여부. caller 가 candidate retry 시 area_change reject
    이후 reject_reason 을 area_change_exceeded 로 update 한다.
    """

    source: str  # "rtabmap_link" | "graph_centerline" | "footprint_obb"
    angle_deg: float | None
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)
    accepted: bool = False
    reject_reason: str | None = None
    area_change_ratio: float | None = None  # caller 가 retry 후 채움

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "angle_deg": self.angle_deg,
            "confidence": float(self.confidence),
            "metadata": dict(self.metadata),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
            "area_change_ratio": self.area_change_ratio,
        }


def compute_dominant_angle_hints(
    *,
    rtabmap_links_with_node_xy: (
        list[tuple[tuple[float, float], tuple[float, float]]] | None
    ) = None,
    footprint_polygon: BaseGeometry | dict[str, object] | None = None,
    graph_edges: (
        list[tuple[tuple[float, float], tuple[float, float]]] | None
    ) = None,
    gate_thresholds: HintGateThresholds | None = None,
) -> list[HintCandidate]:
    """3-source hint chain 후보 list 산출.

    Args:
        rtabmap_links_with_node_xy: RTABMap `Link.type=0` segment list (xy pair).
            None / empty 면 candidate 미생성.
        footprint_polygon: 후보 OBB 산출용 polygon (shapely or geojson dict).
        graph_edges: graph centerline edge xy pair list. Sprint 48 미호출 — helper
            signature 만 정의.
        gate_thresholds: gate threshold override.

    Returns:
        list[HintCandidate]. 우선순위 순서 — RTABMap → graph_centerline →
        footprint_obb. 각 후보는 source / angle / confidence / metadata /
        accepted / reject_reason 을 채워 반환.
    """
    thresholds = gate_thresholds if gate_thresholds is not None else HintGateThresholds()
    candidates: list[HintCandidate] = []

    # 1) RTABMap link
    candidates.append(
        _compute_from_rtabmap_links(
            rtabmap_links_with_node_xy or [],
            thresholds=thresholds,
        )
    )

    # 2) graph centerline (Sprint 48 floor_pc path 에서는 호출되지 않지만 chain 일관성)
    candidates.append(
        _compute_from_graph_edges(
            graph_edges or [],
            thresholds=thresholds,
        )
    )

    # 3) footprint OBB
    candidates.append(
        _compute_from_footprint_obb(
            footprint_polygon,
            thresholds=thresholds,
        )
    )

    return candidates


def cross_check_hints(
    candidates: list[HintCandidate],
    *,
    gate_thresholds: HintGateThresholds | None = None,
) -> dict[str, object]:
    """RTABMap vs OBB hint 충돌 검사.

    Returns:
        {
          "hint_conflict": bool,
          "rtabmap_obb_diff_deg": float | None,
          "max_diff_deg_threshold": float,
        }
    """
    thresholds = gate_thresholds if gate_thresholds is not None else HintGateThresholds()
    rtabmap_angle: float | None = None
    obb_angle: float | None = None
    for c in candidates:
        if c.source == "rtabmap_link" and c.accepted and c.angle_deg is not None:
            rtabmap_angle = float(c.angle_deg)
        elif c.source == "footprint_obb" and c.accepted and c.angle_deg is not None:
            obb_angle = float(c.angle_deg)

    if rtabmap_angle is None or obb_angle is None:
        return {
            "hint_conflict": False,
            "rtabmap_obb_diff_deg": None,
            "max_diff_deg_threshold": float(thresholds.cross_check_max_diff_deg),
        }

    diff = abs(_fold_axis_angle(rtabmap_angle - obb_angle))
    return {
        "hint_conflict": diff > thresholds.cross_check_max_diff_deg,
        "rtabmap_obb_diff_deg": float(diff),
        "max_diff_deg_threshold": float(thresholds.cross_check_max_diff_deg),
    }


# ── source helpers ───────────────────────────────────────────────────────────


def _compute_from_rtabmap_links(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    thresholds: HintGateThresholds,
) -> HintCandidate:
    """RTABMap Link.type=0 segment list 에서 length-weighted dominant angle.

    gate (Codex W-2):
        segment_count >= rtabmap_link_min_segments
        total_length_m >= rtabmap_link_min_total_length_m
        best_bin_weight / total_weight >= rtabmap_link_min_best_bin_ratio
    """
    segment_count = len(segments)
    if segment_count == 0:
        return HintCandidate(
            source="rtabmap_link",
            angle_deg=None,
            confidence=0.0,
            metadata={"segment_count": 0},
            accepted=False,
            reject_reason="rtabmap_link_unavailable",
        )

    total_length_m = 0.0
    bin_deg = 5.0
    n_bins = max(1, int(round(90.0 / bin_deg)))
    weights = np.zeros(n_bins, dtype=np.float64)
    angle_sum = np.zeros(n_bins, dtype=np.float64)
    for (ax, ay), (bx, by) in segments:
        dx = bx - ax
        dy = by - ay
        length = sqrt(dx * dx + dy * dy)
        if length < 0.05:
            continue
        total_length_m += length
        angle = _fold_axis_angle(degrees(atan2(dy, dx)))
        idx = int(np.floor((angle + 45.0) / bin_deg))
        idx = max(0, min(n_bins - 1, idx))
        weights[idx] += length
        angle_sum[idx] += angle * length

    total_weight = float(weights.sum())
    if total_weight < 1e-9:
        return HintCandidate(
            source="rtabmap_link",
            angle_deg=None,
            confidence=0.0,
            metadata={
                "segment_count": segment_count,
                "total_length_m": total_length_m,
            },
            accepted=False,
            reject_reason="rtabmap_link_zero_weight",
        )

    best_idx = int(np.argmax(weights))
    raw = float(angle_sum[best_idx] / weights[best_idx])
    folded = _fold_axis_angle(raw)
    best_bin_weight_ratio = float(weights[best_idx]) / total_weight

    metadata: dict[str, object] = {
        "segment_count": segment_count,
        "total_length_m": float(total_length_m),
        "best_bin_weight_ratio": best_bin_weight_ratio,
        "raw_angle_deg": raw,
    }

    # gate 검증
    reject: list[str] = []
    if segment_count < thresholds.rtabmap_link_min_segments:
        reject.append(f"segment_count<{thresholds.rtabmap_link_min_segments}")
    if total_length_m < thresholds.rtabmap_link_min_total_length_m:
        reject.append(
            f"total_length_m<{thresholds.rtabmap_link_min_total_length_m:.2f}"
        )
    if best_bin_weight_ratio < thresholds.rtabmap_link_min_best_bin_ratio:
        reject.append(
            f"best_bin_weight_ratio<{thresholds.rtabmap_link_min_best_bin_ratio:.2f}"
        )

    if reject:
        return HintCandidate(
            source="rtabmap_link",
            angle_deg=folded,
            confidence=best_bin_weight_ratio,
            metadata=metadata,
            accepted=False,
            reject_reason="rtabmap_link_low_quality:" + "+".join(reject),
        )

    return HintCandidate(
        source="rtabmap_link",
        angle_deg=folded,
        confidence=best_bin_weight_ratio,
        metadata=metadata,
        accepted=True,
    )


def _compute_from_graph_edges(
    edges: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    thresholds: HintGateThresholds,  # noqa: ARG001 — Sprint 48 미호출
) -> HintCandidate:
    """graph centerline length-weighted dominant angle (Sprint 48 미호출).

    floor_pc path 에서 graph 가 아직 build 되지 않아 호출되지 않지만 chain
    일관성을 위해 helper 로 정의. unit test 에서 정확성 검증.
    """
    if not edges:
        return HintCandidate(
            source="graph_centerline",
            angle_deg=None,
            confidence=0.0,
            metadata={"edge_count": 0},
            accepted=False,
            reject_reason="graph_centerline_unavailable",
        )

    # rtabmap link 와 동일 알고리즘 재사용 (length-weighted histogram)
    raw, folded = _dominant_angle_from_segments(edges)
    return HintCandidate(
        source="graph_centerline",
        angle_deg=folded if folded != 0.0 or raw != 0.0 else None,
        confidence=0.0,  # Sprint 48 미사용 — confidence 는 호출 시 별도 계산
        metadata={
            "edge_count": len(edges),
            "raw_angle_deg": raw,
            "note": "graph_centerline_helper_not_exercised_in_floor_pc_path",
        },
        accepted=False,
        reject_reason="graph_centerline_not_exercised_in_sprint_48",
    )


def _compute_from_footprint_obb(
    footprint: BaseGeometry | dict[str, object] | None,
    *,
    thresholds: HintGateThresholds,
) -> HintCandidate:
    """footprint OBB (minimum_rotated_rectangle) 장축 각도.

    gate (Codex W-3):
        aspect_ratio = long_edge / short_edge >= obb_min_aspect_ratio.
    """
    if footprint is None:
        return HintCandidate(
            source="footprint_obb",
            angle_deg=None,
            confidence=0.0,
            metadata={"reason": "no_footprint"},
            accepted=False,
            reject_reason="footprint_unavailable",
        )

    geom: BaseGeometry
    if isinstance(footprint, BaseGeometry):
        geom = footprint
    else:
        try:
            geom = _shape(footprint)
        except Exception as e:
            logger.warning("dominant_angle_hint OBB shape parse failed: %s", e)
            return HintCandidate(
                source="footprint_obb",
                angle_deg=None,
                confidence=0.0,
                metadata={"reason": f"shape_parse_failed:{type(e).__name__}"},
                accepted=False,
                reject_reason="footprint_shape_parse_failed",
            )

    if geom.is_empty or geom.area <= 0:
        return HintCandidate(
            source="footprint_obb",
            angle_deg=None,
            confidence=0.0,
            metadata={"reason": "empty_geometry"},
            accepted=False,
            reject_reason="footprint_empty",
        )

    try:
        # shapely 2.x: oriented_envelope. 1.x fallback: minimum_rotated_rectangle
        if hasattr(geom, "oriented_envelope"):
            obb = geom.oriented_envelope
        else:
            obb = geom.minimum_rotated_rectangle
    except Exception as e:
        logger.warning("dominant_angle_hint OBB compute failed: %s", e)
        return HintCandidate(
            source="footprint_obb",
            angle_deg=None,
            confidence=0.0,
            metadata={"reason": f"obb_compute_failed:{type(e).__name__}"},
            accepted=False,
            reject_reason="footprint_obb_failed",
        )

    coords = list(obb.exterior.coords)
    if len(coords) < 5:  # 닫힌 ring (4 + 1 wrap) 미만
        return HintCandidate(
            source="footprint_obb",
            angle_deg=None,
            confidence=0.0,
            metadata={"reason": "obb_invalid"},
            accepted=False,
            reject_reason="footprint_obb_invalid",
        )

    # 4 edges (rectangle). 가장 긴 edge 의 angle.
    edges_lengths_angles: list[tuple[float, float]] = []
    for (ax, ay), (bx, by) in zip(coords, coords[1:], strict=False):
        dx = bx - ax
        dy = by - ay
        length = sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            continue
        ang = degrees(atan2(dy, dx))
        edges_lengths_angles.append((length, ang))

    if not edges_lengths_angles:
        return HintCandidate(
            source="footprint_obb",
            angle_deg=None,
            confidence=0.0,
            metadata={"reason": "obb_zero_edges"},
            accepted=False,
            reject_reason="footprint_obb_zero_edges",
        )

    edges_lengths_angles.sort(key=lambda t: t[0], reverse=True)
    long_length, long_angle = edges_lengths_angles[0]
    # 평행 edge 가 2개씩 있는 rectangle 이라 short edge는 마지막에서 골라도 안전
    short_length = edges_lengths_angles[-1][0]
    aspect_ratio = (
        long_length / short_length if short_length > 1e-9 else float("inf")
    )

    folded = _fold_axis_angle(long_angle)
    metadata: dict[str, object] = {
        "long_edge_length_m": float(long_length),
        "short_edge_length_m": float(short_length),
        "aspect_ratio": float(aspect_ratio),
        "raw_angle_deg": float(long_angle),
    }

    if aspect_ratio < thresholds.obb_min_aspect_ratio:
        return HintCandidate(
            source="footprint_obb",
            angle_deg=folded,
            confidence=0.0,
            metadata=metadata,
            accepted=False,
            reject_reason="obb_not_elongated",
        )

    return HintCandidate(
        source="footprint_obb",
        angle_deg=folded,
        # confidence proxy = 1 - 1/aspect_ratio (clipped)
        confidence=float(max(0.0, min(1.0, 1.0 - 1.0 / aspect_ratio))),
        metadata=metadata,
        accepted=True,
    )


# ── public utility for callers (pipeline) ────────────────────────────────────


def extract_link_segments_xy(
    nodes: list[Any],
    links: list[Any],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """RTABMap Node/Link → xy segment list.

    rtabmap_trajectory._FloorFrame 와 동일한 floor projection 을 사용해 vertical
    axis 를 자동 식별한다. Sprint 48 floor_pc 진입 시 hint chain 1순위 호출자가
    사용한다.
    """
    if not nodes or not links:
        return []

    from indoor_server.application.building.steps.rtabmap_trajectory import _FloorFrame

    frame = _FloorFrame.from_nodes(nodes)
    node_xy = {node.node_id: frame.project_node_xy(node) for node in nodes}
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    seen: set[tuple[int, int]] = set()
    for link in links:
        if getattr(link, "link_type", None) != 0:
            continue
        if link.from_id not in node_xy or link.to_id not in node_xy:
            continue
        key = (min(link.from_id, link.to_id), max(link.from_id, link.to_id))
        if key in seen or key[0] == key[1]:
            continue
        seen.add(key)
        out.append((node_xy[link.from_id], node_xy[link.to_id]))
    return out
