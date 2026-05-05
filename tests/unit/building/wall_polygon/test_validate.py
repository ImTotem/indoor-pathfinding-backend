"""Sprint 51 — ValidateStep unit tests."""
from __future__ import annotations

from indoor_server.application.building.steps.wall_polygon.assembly import (
    AssembledPolygon,
)
from indoor_server.application.building.steps.wall_polygon.validate import (
    ValidateParams,
    ValidateStep,
)


def _square_geojson(x0: float, y0: float, side: float) -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [x0, y0],
                [x0 + side, y0],
                [x0 + side, y0 + side],
                [x0, y0 + side],
                [x0, y0],
            ]
        ],
    }


def _assembled(geojson: dict[str, object] | None) -> AssembledPolygon:
    return AssembledPolygon(
        polygon_geojson=geojson,
        vertex_count=4 if geojson else 0,
        corner_orthogonality_ratio=1.0 if geojson else 0.0,
        self_intersection_count=0,
        fallback_used=None,
        metadata={},
    )


def test_validate_passes_when_iou_high() -> None:
    floor = _square_geojson(0.0, 0.0, 4.0)
    wall = _square_geojson(0.0, 0.0, 4.0)  # identical
    result = ValidateStep().run(_assembled(wall), floor_polygon_geojson=floor)
    assert result.accepted is True
    assert result.iou_with_floor >= 0.99
    assert result.area_change_ratio < 0.01


def test_validate_rejects_disjoint() -> None:
    floor = _square_geojson(0.0, 0.0, 4.0)
    wall = _square_geojson(100.0, 100.0, 4.0)  # disjoint
    result = ValidateStep().run(_assembled(wall), floor_polygon_geojson=floor)
    assert result.accepted is False
    assert "iou_below_threshold" in result.reject_reasons
    assert result.iou_with_floor == 0.0


def test_validate_rejects_when_floor_unavailable() -> None:
    wall = _square_geojson(0.0, 0.0, 4.0)
    result = ValidateStep().run(_assembled(wall), floor_polygon_geojson=None)
    assert result.accepted is False
    assert "floor_unavailable" in result.reject_reasons


def test_validate_rejects_area_change() -> None:
    floor = _square_geojson(0.0, 0.0, 4.0)
    wall = _square_geojson(0.0, 0.0, 1.0)  # area change ~ 1 - 1/16 = 0.9375 > 0.40
    result = ValidateStep(ValidateParams(floor_iou_min=0.0)).run(
        _assembled(wall), floor_polygon_geojson=floor
    )
    assert result.accepted is False
    assert "area_change_exceeded" in result.reject_reasons
