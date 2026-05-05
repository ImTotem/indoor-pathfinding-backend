from __future__ import annotations

from itertools import combinations
from uuid import UUID, uuid4

from shapely.geometry import Point, shape

from indoor_server.application.semantic.semantic_map_service import SemanticMapService
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow
from indoor_server.domain.semantic.models import PlaceAreaFeature


def _node(
    *,
    node_id: UUID,
    x: float,
    y: float,
    node_type: str = "corridor",
    poi_mark_id: int | None = None,
) -> MapNodeRow:
    return MapNodeRow(
        node_id=node_id,
        x=x,
        y=y,
        z=0.0,
        node_type=node_type,
        label=None,
        poi_mark_id=poi_mark_id,
    )


def test_semantic_map_service_creates_amenity_and_place_area() -> None:
    a = uuid4()
    b = uuid4()
    poi = uuid4()
    nodes = [
        _node(node_id=a, x=0.0, y=0.0),
        _node(node_id=b, x=10.0, y=0.0),
        _node(node_id=poi, x=5.0, y=0.0, node_type="poi", poi_mark_id=7),
    ]
    edges = [MapEdgeRow(edge_id=uuid4(), from_node_id=a, to_node_id=b, length_m=10.0)]

    amenities, places = SemanticMapService().build(
        nodes=nodes,
        edges=edges,
        poi_labels={7: "301호"},
    )

    assert len(amenities) == 1
    assert len(places) == 1
    assert amenities[0].poi_mark_id == 7
    assert amenities[0].route_node_id == poi
    assert amenities[0].display_area_id == "place-poi-7"
    assert places[0].geometry["type"] == "MultiPolygon"
    assert places[0].entrance_node_id == poi
    ring = places[0].geometry["coordinates"][0][0]
    xs = [pt[0] for pt in ring[:-1]]
    ys = [pt[1] for pt in ring[:-1]]
    assert max(xs) - min(xs) >= 1.9
    assert max(ys) - min(ys) >= 1.9


def test_semantic_place_area_chooses_wall_side_from_footprint_not_poi_parity() -> None:
    a = uuid4()
    b = uuid4()
    poi = uuid4()
    nodes = [
        _node(node_id=a, x=0.0, y=0.0),
        _node(node_id=b, x=10.0, y=0.0),
        _node(node_id=poi, x=5.0, y=0.0, node_type="poi", poi_mark_id=8),
    ]
    edges = [MapEdgeRow(edge_id=uuid4(), from_node_id=a, to_node_id=b, length_m=10.0)]
    footprint = {
        "type": "Polygon",
        "coordinates": [[
            [0.0, -1.2], [10.0, -1.2], [10.0, 4.0], [0.0, 4.0], [0.0, -1.2],
        ]],
    }
    walkway = {
        "type": "Polygon",
        "coordinates": [[
            [0.0, -0.8], [10.0, -0.8], [10.0, 0.8], [0.0, 0.8], [0.0, -0.8],
        ]],
    }

    amenities, places = SemanticMapService().build(
        nodes=nodes,
        edges=edges,
        poi_labels={8: "301호"},
        footprint_geojson=footprint,
        walkway_geojson=walkway,
    )

    place_geom = shape(places[0].geometry)
    walkway_geom = shape(walkway)
    assert place_geom.centroid.y > 0.8
    assert place_geom.intersection(walkway_geom).area / place_geom.area < 0.01
    assert place_geom.within(shape(footprint))
    assert place_geom.contains(Point(amenities[0].display_point[0], amenities[0].display_point[1]))
    assert amenities[0].point == (5.0, 0.0, 0.0)


def test_semantic_place_areas_pack_nearby_pois_without_overlap() -> None:
    a = uuid4()
    b = uuid4()
    poi_1 = uuid4()
    poi_2 = uuid4()
    poi_3 = uuid4()
    nodes = [
        _node(node_id=a, x=0.0, y=0.0),
        _node(node_id=b, x=12.0, y=0.0),
        _node(node_id=poi_1, x=5.00, y=0.0, node_type="poi", poi_mark_id=1),
        _node(node_id=poi_2, x=5.35, y=0.0, node_type="poi", poi_mark_id=2),
        _node(node_id=poi_3, x=5.70, y=0.0, node_type="poi", poi_mark_id=3),
    ]
    edges = [MapEdgeRow(edge_id=uuid4(), from_node_id=a, to_node_id=b, length_m=12.0)]
    footprint = {
        "type": "Polygon",
        "coordinates": [[
            [0.0, -1.3], [12.0, -1.3], [12.0, 4.0], [0.0, 4.0], [0.0, -1.3],
        ]],
    }
    walkway = {
        "type": "Polygon",
        "coordinates": [[
            [0.0, -1.25], [12.0, -1.25], [12.0, 1.25], [0.0, 1.25], [0.0, -1.25],
        ]],
    }

    service = SemanticMapService()
    amenities, places = service.build(
        nodes=nodes,
        edges=edges,
        poi_labels={1: "301호", 2: "302호", 3: "303호"},
        footprint_geojson=footprint,
        walkway_geojson=walkway,
    )
    amenities_again, places_again = service.build(
        nodes=nodes,
        edges=edges,
        poi_labels={1: "301호", 2: "302호", 3: "303호"},
        footprint_geojson=footprint,
        walkway_geojson=walkway,
    )

    assert [place.source_poi_mark_id for place in places] == [1, 2, 3]
    assert [amenity.poi_mark_id for amenity in amenities] == [1, 2, 3]
    assert len(places) == 3
    walkway_geom = shape(walkway)
    place_geoms = [shape(place.geometry) for place in places]
    for geom in place_geoms:
        assert geom.geom_type in {"Polygon", "MultiPolygon"}
        assert not geom.is_empty
        assert geom.centroid.y > 1.25
        assert geom.intersection(walkway_geom).area / geom.area < 0.01

    for first, second in combinations(place_geoms, 2):
        overlap_area = first.intersection(second).area
        overlap_ratio = overlap_area / min(first.area, second.area)
        assert overlap_area <= 0.05
        assert overlap_ratio <= 0.02

    place_by_id = {place.id: shape(place.geometry) for place in places}
    for amenity in amenities:
        display_point = Point(amenity.display_point[0], amenity.display_point[1])
        assert place_by_id[amenity.display_area_id].covers(display_point)

    assert _geometry_by_poi(places_again) == _geometry_by_poi(places)
    assert _geometry_by_poi(places_again) == _geometry_by_poi(
        service.build(
            nodes=nodes,
            edges=edges,
            poi_labels={1: "301호", 2: "302호", 3: "303호"},
            footprint_geojson=footprint,
            walkway_geojson=walkway,
        )[1]
    )
    assert [amenity.display_point for amenity in amenities_again] == [
        amenity.display_point for amenity in amenities
    ]


def _geometry_by_poi(places: list[PlaceAreaFeature]) -> dict[int, dict[str, object]]:
    return {
        place.source_poi_mark_id: place.geometry
        for place in places
    }
