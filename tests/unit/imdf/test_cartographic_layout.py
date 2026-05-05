"""Cycle 2 cartographic IMDF layout tests."""
from __future__ import annotations

from math import atan2, degrees
from uuid import UUID

import pytest
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union

from indoor_server.application.imdf.cartographic_layout import (
    AnalyzedPoi,
    CartographicLayoutBuilder,
)
from indoor_server.domain.routing.models import MapNodeRow
from indoor_server.domain.semantic.models import SemanticAnalysis

_SCAN_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def test_road_unit_uses_observed_footprint_not_centerline_buffer() -> None:
    footprint = _footprint(width=20.0, height=8.0)
    result = CartographicLayoutBuilder().build(
        scan_id=_SCAN_ID,
        footprint_geojson=footprint,
        analyzed_pois=[],
    )

    observed = shape(footprint)
    road = shape(result.road_geometry)

    assert 0.90 <= road.area / observed.area <= 1.10
    assert road.bounds == pytest.approx(observed.bounds)
    assert result.metadata["centerline_rendered_as_unit"] is False
    assert result.metadata["walkway_source"] == "rectified_observed_walkable_footprint"


def test_generated_blocks_live_outside_road_inside_envelope() -> None:
    result = CartographicLayoutBuilder().build(
        scan_id=_SCAN_ID,
        footprint_geojson=_footprint(width=20.0, height=8.0),
        analyzed_pois=[_analyzed_poi(1, 5.0, 1.0, "room")],
    )
    road = shape(result.road_geometry)
    envelope = shape(result.block_envelope_geometry)
    block_ring = envelope.difference(road.buffer(0.08, join_style=2)).buffer(0)
    block_geoms = [
        shape(unit.geometry) for unit in result.units if unit.id.startswith("unit-block-")
    ]
    block_geoms.extend(shape(place.geometry) for place in result.place_areas)

    assert block_geoms
    assert all(geom.intersection(road).area <= 0.02 for geom in block_geoms)
    assert all(envelope.covers(geom) for geom in block_geoms)
    assert unary_union(block_geoms).area / block_ring.area >= 0.75
    assert result.metadata["parcel_count"] == 0


def test_destination_blocks_are_subdivisions_not_tabs() -> None:
    result = CartographicLayoutBuilder().build(
        scan_id=_SCAN_ID,
        footprint_geojson=_footprint(width=20.0, height=8.0),
        analyzed_pois=[
            _analyzed_poi(1, 5.0, 1.0, "room"),
            _analyzed_poi(2, 10.0, 1.0, "lab"),
        ],
    )
    road = shape(result.road_geometry)

    assert len(result.place_areas) == 2
    for place in result.place_areas:
        geom = shape(place.geometry)
        minx, miny, maxx, maxy = geom.bounds
        assert maxy <= -0.08
        assert maxy - miny >= 2.4
        assert geom.area >= 1.0
        assert geom.intersection(road).area <= 0.05


def test_poi_display_points_are_inside_destination_blocks_outside_road() -> None:
    result = CartographicLayoutBuilder().build(
        scan_id=_SCAN_ID,
        footprint_geojson=_footprint(width=20.0, height=8.0),
        analyzed_pois=[_analyzed_poi(7, 6.0, 1.0, "stairs")],
    )
    road = shape(result.road_geometry)
    place_by_id = {place.id: shape(place.geometry) for place in result.place_areas}

    assert len(result.amenities) == 1
    amenity = result.amenities[0]
    assert amenity.display_area_id == "place-poi-7"
    display = Point(amenity.display_point[0], amenity.display_point[1])
    assert place_by_id["place-poi-7"].covers(display)
    assert road.distance(display) >= 0.20
    assert road.distance(Point(amenity.point[0], amenity.point[1])) <= 0.05


def test_all_pois_get_display_blocks_when_boundary_is_crowded() -> None:
    result = CartographicLayoutBuilder().build(
        scan_id=_SCAN_ID,
        footprint_geojson=_footprint(width=9.0, height=5.0),
        analyzed_pois=[
            _analyzed_poi(1, 2.00, 0.8, "room"),
            _analyzed_poi(2, 2.25, 0.8, "room"),
            _analyzed_poi(3, 2.50, 0.8, "lab"),
            _analyzed_poi(4, 2.75, 0.8, "elevator"),
            _analyzed_poi(5, 3.00, 0.8, "stairs"),
        ],
    )
    road = shape(result.road_geometry)
    place_by_id = {place.id: shape(place.geometry) for place in result.place_areas}

    assert len(result.place_areas) == 5
    assert result.metadata["semantic_place_count"] == 5
    assert result.metadata["block_split"]["destination_block_count"] == 5
    for amenity in result.amenities:
        assert amenity.display_area_id == f"place-poi-{amenity.poi_mark_id}"
        assert amenity.display_area_id in place_by_id
        display = Point(amenity.display_point[0], amenity.display_point[1])
        assert place_by_id[amenity.display_area_id].covers(display)
        assert road.distance(display) >= 0.20


def test_no_unit_parcels_and_road_geometry_consistent() -> None:
    footprint = _footprint(width=20.0, height=8.0)
    result = CartographicLayoutBuilder().build(
        scan_id=_SCAN_ID,
        footprint_geojson=footprint,
        analyzed_pois=[],
    )

    assert not any(unit.id.startswith("unit-parcel-") for unit in result.units)
    assert result.metadata["walkway_half_width_m"] is None
    assert not shape(result.road_geometry).is_empty


def test_rectilinearizer_outputs_orthogonal_edges() -> None:
    result = CartographicLayoutBuilder().build(
        scan_id=_SCAN_ID,
        footprint_geojson=_footprint(width=20.0, height=8.0),
        analyzed_pois=[],
    )
    road = shape(result.road_geometry)
    orientation = float(result.metadata["road_style"]["orientation_deg"])

    for x1, y1, x2, y2 in _exterior_edges(road):
        bearing = degrees(atan2(y2 - y1, x2 - x1)) % 180.0
        delta_axis = min(abs(bearing - orientation), abs(bearing - orientation - 180.0))
        delta_perp = abs(bearing - ((orientation + 90.0) % 180.0))
        assert min(delta_axis, delta_perp) <= 1.0


def test_road_outline_removes_grid_staircase_noise() -> None:
    result = CartographicLayoutBuilder().build(
        scan_id=_SCAN_ID,
        footprint_geojson=_stair_stepped_footprint(),
        analyzed_pois=[],
    )
    road = shape(result.road_geometry)
    if isinstance(road, Polygon):
        coords = list(road.exterior.coords)
    else:
        coords = list(road.geoms[0].exterior.coords)
    edge_lengths = [
        Point(first).distance(Point(second))
        for first, second in zip(coords, coords[1:], strict=False)
    ]

    assert len(coords) <= 32
    assert sum(length < 0.75 for length in edge_lengths) <= 2
    assert result.metadata["road_style"]["outline_simplified"] is True


def _footprint(*, width: float, height: float) -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [[
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
            [0.0, 0.0],
        ]],
    }


def _stair_stepped_footprint() -> dict[str, object]:
    left = []
    right = []
    for index in range(18):
        y = float(index)
        left.append([0.0 if index % 2 == 0 else 0.3, y])
        right.append([6.0 if index % 2 == 0 else 5.7, y])
    ring = left + list(reversed(right)) + [left[0]]
    return {"type": "Polygon", "coordinates": [ring]}


def _node(node_id: str, x: float, y: float, *, poi_mark_id: int | None = None) -> MapNodeRow:
    return MapNodeRow(
        node_id=UUID(node_id),
        x=x,
        y=y,
        z=0.0,
        node_type="poi" if poi_mark_id is not None else "corridor",
        label=None,
        poi_mark_id=poi_mark_id,
    )


def _analyzed_poi(
    poi_mark_id: int,
    x: float,
    y: float,
    category: str,
) -> AnalyzedPoi:
    return AnalyzedPoi(
        node=_node(
            f"00000000-0000-0000-0000-{poi_mark_id:012d}",
            x,
            y,
            poi_mark_id=poi_mark_id,
        ),
        analysis=SemanticAnalysis(
            category=category,  # type: ignore[arg-type]
            name=f"poi-{poi_mark_id}",
            confidence=0.9,
            analyzer="test",
        ),
    )


def _exterior_edges(geom) -> list[tuple[float, float, float, float]]:  # noqa: ANN001
    if isinstance(geom, Polygon):
        polygons = [geom]
    else:
        polygons = list(geom.geoms)
    edges: list[tuple[float, float, float, float]] = []
    for polygon in polygons:
        coords = list(polygon.exterior.coords)
        for start, end in zip(coords, coords[1:], strict=False):
            if Point(start).distance(Point(end)) > 0.1:
                edges.append((start[0], start[1], end[0], end[1]))
    return edges
