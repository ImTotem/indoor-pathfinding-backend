"""Quality gates for multi-scan heatmap fusion.

The actual floor/wall segmentation fusion will consume the pose mapping outputs
from Sprint 75. This module keeps the gate logic isolated and unit-testable so
future dense evidence code can fail closed before promoting a fused polygon.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MultiScanFusionMetrics:
    source_scan_count: int
    merged_node_count: int
    max_source_node_count: int
    per_scan_usable_frame_ratio: dict[str, float]
    inter_session_loop_closure_count: int
    mapping_ambiguous_ratio: float
    mapping_missing_ratio: float
    session_transform_residual_median: float
    session_transform_residual_p90: float
    polygon_area_inflation_ratio: float
    double_wall_line_score: float
    direction_bin_coverage: int
    scan_support_ratio: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_scan_count": self.source_scan_count,
            "merged_node_count": self.merged_node_count,
            "max_source_node_count": self.max_source_node_count,
            "per_scan_usable_frame_ratio": self.per_scan_usable_frame_ratio,
            "inter_session_loop_closure_count": self.inter_session_loop_closure_count,
            "mapping_ambiguous_ratio": self.mapping_ambiguous_ratio,
            "mapping_missing_ratio": self.mapping_missing_ratio,
            "session_transform_residual_median": self.session_transform_residual_median,
            "session_transform_residual_p90": self.session_transform_residual_p90,
            "polygon_area_inflation_ratio": self.polygon_area_inflation_ratio,
            "double_wall_line_score": self.double_wall_line_score,
            "direction_bin_coverage": self.direction_bin_coverage,
            "scan_support_ratio": self.scan_support_ratio,
        }


@dataclass(frozen=True)
class MultiScanFusionGateThresholds:
    min_source_scan_count: int = 2
    min_merged_node_ratio: float = 0.75
    min_per_scan_usable_frame_ratio: float = 0.65
    min_inter_session_loop_closure_count: int = 1
    max_mapping_ambiguous_ratio: float = 0.05
    max_mapping_missing_ratio: float = 0.25
    max_session_transform_residual_median: float = 0.30
    max_session_transform_residual_p90: float = 0.75
    max_polygon_area_inflation_ratio: float = 1.35
    max_double_wall_line_score: float = 0.20
    min_direction_bin_coverage: int = 2
    min_scan_support_ratio: float = 0.10


@dataclass(frozen=True)
class MultiScanFusionGateResult:
    accepted: bool
    failures: list[str] = field(default_factory=list)


def evaluate_multiscan_fusion_gates(
    metrics: MultiScanFusionMetrics,
    thresholds: MultiScanFusionGateThresholds | None = None,
) -> MultiScanFusionGateResult:
    t = thresholds or MultiScanFusionGateThresholds()
    failures: list[str] = []
    if metrics.source_scan_count < t.min_source_scan_count:
        failures.append(
            f"source_scan_count_below_min:{metrics.source_scan_count}<{t.min_source_scan_count}"
        )
    required_nodes = metrics.max_source_node_count * t.min_merged_node_ratio
    if metrics.merged_node_count < required_nodes:
        failures.append(
            f"merged_node_count_below_min:{metrics.merged_node_count}<{required_nodes:.1f}"
        )
    for scan_id, ratio in metrics.per_scan_usable_frame_ratio.items():
        if ratio < t.min_per_scan_usable_frame_ratio:
            failures.append(
                f"usable_frame_ratio_below_min:{scan_id}:{ratio:.3f}"
            )
    if metrics.inter_session_loop_closure_count < t.min_inter_session_loop_closure_count:
        failures.append(
            "inter_session_loop_closure_count_below_min:"
            f"{metrics.inter_session_loop_closure_count}"
        )
    if metrics.mapping_ambiguous_ratio > t.max_mapping_ambiguous_ratio:
        failures.append(
            f"mapping_ambiguous_ratio_above_max:{metrics.mapping_ambiguous_ratio:.3f}"
        )
    if metrics.mapping_missing_ratio > t.max_mapping_missing_ratio:
        failures.append(
            f"mapping_missing_ratio_above_max:{metrics.mapping_missing_ratio:.3f}"
        )
    if metrics.session_transform_residual_median > t.max_session_transform_residual_median:
        failures.append(
            "session_transform_residual_median_above_max:"
            f"{metrics.session_transform_residual_median:.3f}"
        )
    if metrics.session_transform_residual_p90 > t.max_session_transform_residual_p90:
        failures.append(
            f"session_transform_residual_p90_above_max:{metrics.session_transform_residual_p90:.3f}"
        )
    if metrics.polygon_area_inflation_ratio > t.max_polygon_area_inflation_ratio:
        failures.append(
            f"polygon_area_inflation_ratio_above_max:{metrics.polygon_area_inflation_ratio:.3f}"
        )
    if metrics.double_wall_line_score > t.max_double_wall_line_score:
        failures.append(
            f"double_wall_line_score_above_max:{metrics.double_wall_line_score:.3f}"
        )
    if metrics.direction_bin_coverage < t.min_direction_bin_coverage:
        failures.append(
            f"direction_bin_coverage_below_min:{metrics.direction_bin_coverage}"
        )
    for scan_id, ratio in metrics.scan_support_ratio.items():
        if ratio < t.min_scan_support_ratio:
            failures.append(f"scan_support_ratio_below_min:{scan_id}:{ratio:.3f}")
    return MultiScanFusionGateResult(accepted=not failures, failures=failures)
