"""Sprint 47 — PolygonCadCleanupStep unit tests."""
from __future__ import annotations

import math

from indoor_server.application.building.steps.polygon_cad_cleanup import (
    PolygonCadCleanupStep,
    PolygonCadCleanupStepParams,
)


def _square_geojson(side: float = 4.0) -> dict[str, object]:
    """단순 직사각형 polygon GeoJSON (corner 4개, 모두 90°)."""
    s = side
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [0.0, 0.0],
                [s, 0.0],
                [s, s],
                [0.0, s],
                [0.0, 0.0],
            ]
        ],
    }


def _square_with_short_stub() -> dict[str, object]:
    """직사각형 위쪽 edge에 0.05m 짧은 stub 박힌 polygon."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [2.0, 4.0],
                [2.0, 4.05],   # 0.05m stub 시작
                [2.05, 4.05],  # 0.05m stub
                [2.05, 4.0],
                [0.0, 4.0],
                [0.0, 0.0],
            ]
        ],
    }


def _square_with_collinear_extra() -> dict[str, object]:
    """직사각형 바닥 edge에 collinear vertex 2개 추가."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [0.0, 0.0],
                [1.0, 0.0],   # collinear
                [2.0, 0.0],   # collinear
                [4.0, 0.0],
                [4.0, 4.0],
                [0.0, 4.0],
                [0.0, 0.0],
            ]
        ],
    }


def _square_with_near_pair() -> dict[str, object]:
    """직사각형 corner에 0.05m 거리 짝꿍 vertex (near pair) 박힘."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [0.0, 4.0],
                [0.0, 0.05],  # 0.05m 짝꿍 vertex
                [0.0, 0.0],
            ]
        ],
    }


def _l_shape() -> dict[str, object]:
    """L자 polygon — 모서리 6개, 모두 90°/270°."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 2.0],
                [2.0, 2.0],
                [2.0, 4.0],
                [0.0, 4.0],
                [0.0, 0.0],
            ]
        ],
    }


def test_disabled_returns_input_passthrough() -> None:
    step = PolygonCadCleanupStep(PolygonCadCleanupStepParams(enabled=False))
    inp = _square_geojson()
    res = step.run(inp)
    assert res.cleaned_geojson is inp
    assert res.metadata["enabled"] is False
    assert res.metadata["skipped_reason"] == "disabled"


def test_simple_square_orthogonality_is_one() -> None:
    step = PolygonCadCleanupStep()
    res = step.run(_square_geojson())
    meta = res.metadata
    # 4 corner 모두 90° → ratio == 1.0
    assert math.isclose(meta["corner_orthogonality_ratio"], 1.0, rel_tol=1e-6)
    assert meta["collinear_residual_count"] == 0
    assert meta["component_count_after"] == 1
    assert meta["polygon_area_m2"] >= 3.0
    # iou ~ 1, centroid_shift ~ 0
    assert meta["iou_raw_rectified"] >= 0.99
    assert meta["centroid_shift_m"] <= 0.01


def test_l_shape_orthogonality_is_one() -> None:
    step = PolygonCadCleanupStep()
    res = step.run(_l_shape())
    meta = res.metadata
    assert math.isclose(meta["corner_orthogonality_ratio"], 1.0, rel_tol=1e-6)
    assert meta["collinear_residual_count"] == 0
    assert meta["component_count_after"] == 1


def test_short_edge_pruning_removes_stub() -> None:
    step = PolygonCadCleanupStep(
        PolygonCadCleanupStepParams(short_edge_min_length_m=0.20)
    )
    res = step.run(_square_with_short_stub())
    meta = res.metadata
    # stub 가 만들었던 4 vertex 가 줄어듦 (8 → 4)
    assert meta["vertex_count_before"] == 8
    assert meta["vertex_count_after"] <= 6
    assert (
        meta["short_edges_pruned_count"]
        + meta["near_vertices_merged_count"]
        + meta["collinear_merged_count"]
    ) > 0


def test_collinear_merge_removes_inline_vertices() -> None:
    step = PolygonCadCleanupStep(
        PolygonCadCleanupStepParams(
            collinear_angle_tol_deg=5.0,
            short_edge_min_length_m=0.0,  # short edge prune 끔 → collinear만 검증
            near_vertex_merge_distance_m=0.0,
        )
    )
    res = step.run(_square_with_collinear_extra())
    meta = res.metadata
    # 6 vertex (4 corner + 2 collinear) → 4 vertex
    assert meta["vertex_count_before"] == 6
    assert meta["vertex_count_after"] == 4
    assert meta["collinear_merged_count"] >= 2


def test_near_vertex_merge_collapses_close_pair() -> None:
    step = PolygonCadCleanupStep(
        PolygonCadCleanupStepParams(
            near_vertex_merge_distance_m=0.15,
            short_edge_min_length_m=0.0,  # 다른 prune 끔
            collinear_angle_tol_deg=0.0,
        )
    )
    res = step.run(_square_with_near_pair())
    meta = res.metadata
    assert meta["near_vertices_merged_count"] >= 1
    assert meta["vertex_count_after"] < meta["vertex_count_before"]


def test_empty_geojson_returns_skipped() -> None:
    step = PolygonCadCleanupStep()
    empty = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 0.0]]]}
    res = step.run(empty)
    assert res.cleaned_geojson is empty
    assert res.metadata.get("skipped_reason") is not None


def test_invalid_geojson_returns_skipped() -> None:
    step = PolygonCadCleanupStep()
    invalid: dict[str, object] = {"type": "NotAGeometry"}
    res = step.run(invalid)
    assert res.cleaned_geojson is invalid
    assert "skipped_reason" in res.metadata


def test_shape_preservation_guard_metrics_present() -> None:
    """W-1: IoU/centroid/bbox shape-preservation guard 메트릭이 항상 기록된다."""
    step = PolygonCadCleanupStep()
    res = step.run(_square_with_collinear_extra())
    meta = res.metadata
    assert "iou_raw_rectified" in meta
    assert "centroid_shift_m" in meta
    assert "bbox_width_ratio" in meta
    assert "bbox_height_ratio" in meta
    # collinear merge는 shape를 보존해야 함 → IoU ~1
    assert meta["iou_raw_rectified"] >= 0.95


def test_minimum_guard_metrics_present() -> None:
    """W-7: component_count_after, polygon_area_m2 메트릭 노출."""
    step = PolygonCadCleanupStep()
    res = step.run(_square_geojson())
    meta = res.metadata
    assert meta["component_count_after"] == 1
    assert meta["polygon_area_m2"] > 0


def test_invalid_params_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        PolygonCadCleanupStep(
            PolygonCadCleanupStepParams(collinear_angle_tol_deg=-1.0)
        )
    with pytest.raises(ValueError):
        PolygonCadCleanupStep(
            PolygonCadCleanupStepParams(short_edge_min_length_m=-1.0)
        )
    with pytest.raises(ValueError):
        PolygonCadCleanupStep(
            PolygonCadCleanupStepParams(near_vertex_merge_distance_m=-1.0)
        )
    with pytest.raises(ValueError):
        PolygonCadCleanupStep(
            PolygonCadCleanupStepParams(orthogonality_angle_tol_deg=-1.0)
        )


# ── Sprint 48 (Codex F-2): cleanup_changed metadata ─────────────────────────


def test_cleanup_changed_true_when_short_edge_pruned() -> None:
    """짧은 stub edge 가 prune 되면 cleanup_changed=True."""
    from indoor_server.application.building.steps.polygon_cad_cleanup import (
        PolygonCadCleanupStep,
        PolygonCadCleanupStepParams,
    )

    # 짧은 stub edge 가 있는 직사각형 (8 vertex)
    poly_geojson = {
        "type": "Polygon",
        "coordinates": [
            [
                (0.0, 0.0),
                (5.0, 0.0),
                (5.05, 0.02),  # stub
                (5.0, 3.0),
                (0.0, 3.0),
                (0.0, 0.0),
            ]
        ],
    }
    step = PolygonCadCleanupStep(
        PolygonCadCleanupStepParams(
            short_edge_min_length_m=0.20,
            near_vertex_merge_distance_m=0.15,
        )
    )
    result = step.run(poly_geojson)
    assert result.metadata.get("cleanup_changed") is True


def test_cleanup_changed_false_for_clean_rectangle() -> None:
    """이미 깨끗한 직사각형 (4 vertex) 에 cleanup 적용 안 됨 → cleanup_changed=False."""
    from indoor_server.application.building.steps.polygon_cad_cleanup import (
        PolygonCadCleanupStep,
        PolygonCadCleanupStepParams,
    )

    poly_geojson = {
        "type": "Polygon",
        "coordinates": [
            [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0), (0.0, 0.0)]
        ],
    }
    step = PolygonCadCleanupStep(PolygonCadCleanupStepParams())
    result = step.run(poly_geojson)
    # 4 vertex 깨끗한 polygon 은 cleanup operation 없음
    md = result.metadata
    assert md.get("vertex_count_before") == md.get("vertex_count_after")
    assert md.get("cleanup_changed") is False
