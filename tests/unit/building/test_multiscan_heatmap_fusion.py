from __future__ import annotations

from indoor_server.application.building.steps.multiscan_heatmap_fusion import (
    MultiScanFusionMetrics,
    evaluate_multiscan_fusion_gates,
)


def _passing_metrics() -> MultiScanFusionMetrics:
    return MultiScanFusionMetrics(
        source_scan_count=2,
        merged_node_count=180,
        max_source_node_count=120,
        per_scan_usable_frame_ratio={"left": 0.8, "right": 0.75},
        inter_session_loop_closure_count=2,
        mapping_ambiguous_ratio=0.01,
        mapping_missing_ratio=0.10,
        session_transform_residual_median=0.15,
        session_transform_residual_p90=0.45,
        polygon_area_inflation_ratio=1.10,
        double_wall_line_score=0.05,
        direction_bin_coverage=2,
        scan_support_ratio={"left": 0.48, "right": 0.52},
    )


def test_multiscan_fusion_gate_accepts_balanced_opposite_direction_scan() -> None:
    result = evaluate_multiscan_fusion_gates(_passing_metrics())

    assert result.accepted
    assert result.failures == []


def test_multiscan_fusion_gate_accepts_short_secondary_scan_support() -> None:
    metrics = _passing_metrics()
    unbalanced = MultiScanFusionMetrics(
        **{
            **metrics.to_dict(),
            "scan_support_ratio": {"left": 0.19, "right": 0.81},
        }
    )

    result = evaluate_multiscan_fusion_gates(unbalanced)

    assert result.accepted
    assert result.failures == []


def test_multiscan_fusion_gate_fails_when_one_scan_does_not_contribute() -> None:
    metrics = _passing_metrics()
    failed = MultiScanFusionMetrics(
        **{
            **metrics.to_dict(),
            "scan_support_ratio": {"left": 0.99, "right": 0.01},
        }
    )

    result = evaluate_multiscan_fusion_gates(failed)

    assert not result.accepted
    assert any("scan_support_ratio_below_min:right" in item for item in result.failures)
