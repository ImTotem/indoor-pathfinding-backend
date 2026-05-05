"""Semantic map assembly service."""
from __future__ import annotations

from shapely.geometry import shape

from indoor_server.application.imdf.cartographic_layout import AnalyzedPoi
from indoor_server.application.semantic.mock_analyzer import MockSemanticAnalyzer
from indoor_server.application.semantic.place_area import PlaceAreaBuilder
from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow
from indoor_server.domain.semantic.models import PlaceAreaFeature, SemanticAmenityFeature


class SemanticMapService:
    """POI graph node를 semantic amenity/place area로 승격한다."""

    def __init__(
        self,
        *,
        analyzer: MockSemanticAnalyzer | None = None,
        place_builder: PlaceAreaBuilder | None = None,
    ) -> None:
        self._analyzer = analyzer or MockSemanticAnalyzer()
        self._place_builder = place_builder or PlaceAreaBuilder()

    def build(
        self,
        *,
        nodes: list[MapNodeRow],
        edges: list[MapEdgeRow],
        poi_labels: dict[int, str | None],
        footprint_geojson: dict[str, object] | None = None,
        walkway_geojson: dict[str, object] | None = None,
    ) -> tuple[list[SemanticAmenityFeature], list[PlaceAreaFeature]]:
        analyzed_pois = self.analyze_pois(nodes=nodes, poi_labels=poi_labels)
        node_lookup = {str(node.node_id): node for node in nodes}
        amenities: list[SemanticAmenityFeature] = []
        analyzed_nodes = [(poi.node, poi.analysis) for poi in analyzed_pois]

        places = self._place_builder.build_many(
            items=analyzed_nodes,
            edges=edges,
            node_lookup=node_lookup,
            footprint_geojson=footprint_geojson,
            walkway_geojson=walkway_geojson,
        )
        place_by_poi_id = {place.source_poi_mark_id: place for place in places}

        for node, analysis in analyzed_nodes:
            assert node.poi_mark_id is not None
            place = place_by_poi_id[node.poi_mark_id]
            display_point = self._display_point(place)
            amenities.append(
                SemanticAmenityFeature(
                    id=f"amenity-{node.poi_mark_id}",
                    poi_mark_id=node.poi_mark_id,
                    route_node_id=node.node_id,
                    category=analysis.category,
                    name=analysis.name,
                    point=(node.x, node.y, node.z),
                    display_point=(display_point[0], display_point[1], node.z),
                    display_area_id=place.id,
                    analysis=analysis,
                )
            )
        return amenities, places

    def analyze_pois(
        self,
        *,
        nodes: list[MapNodeRow],
        poi_labels: dict[int, str | None],
    ) -> list[AnalyzedPoi]:
        poi_nodes = [
            node for node in nodes
            if node.node_type == "poi" and node.poi_mark_id is not None
        ]
        analyzed: list[AnalyzedPoi] = []
        for node in sorted(poi_nodes, key=lambda n: int(n.poi_mark_id or 0)):
            assert node.poi_mark_id is not None
            label = poi_labels.get(node.poi_mark_id) or node.label
            analysis = self._analyzer.analyze(label=label, class_name=None, source="poi_mark")
            analyzed.append(AnalyzedPoi(node=node, analysis=analysis))
        return analyzed

    def _display_point(self, place: PlaceAreaFeature) -> tuple[float, float]:
        geom = shape(place.geometry)
        if geom.is_empty:
            return (0.0, 0.0)
        point = geom.centroid
        if not geom.covers(point):
            point = geom.representative_point()
        return (float(point.x), float(point.y))
