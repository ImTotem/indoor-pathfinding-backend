"""Cartographic IMDF display layout.

The route graph is topology. This module owns the map display fabric:
observed walkable footprint as the broad road/open area, and synthetic blocks
around that road for context and destination labels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import atan2, ceil, degrees, hypot
from typing import Any

from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from indoor_server.application.imdf.unit_splitter import UnitFeature
from indoor_server.domain.routing.models import MapNodeRow
from indoor_server.domain.semantic.models import (
    PlaceAreaFeature,
    SemanticAmenityFeature,
    SemanticAnalysis,
)


@dataclass(frozen=True)
class AnalyzedPoi:
    node: MapNodeRow
    analysis: SemanticAnalysis


@dataclass(frozen=True)
class DestinationBlock:
    poi_mark_id: int
    geometry: dict[str, object]
    display_point: tuple[float, float, float]
    road_side: str
    boundary_segment_id: str
    boundary_station_m: float


@dataclass(frozen=True)
class CartographicLayoutResult:
    units: list[UnitFeature]
    place_areas: list[PlaceAreaFeature]
    amenities: list[SemanticAmenityFeature]
    road_geometry: dict[str, object]
    block_envelope_geometry: dict[str, object]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _BoundarySegment:
    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    tangent: tuple[float, float]
    normal: tuple[float, float]
    length_m: float


@dataclass(frozen=True)
class _PoiStation:
    poi: AnalyzedPoi
    segment: _BoundarySegment
    station_m: float
    width_m: float
    depth_m: float
    interval: tuple[float, float]


class CartographicLayoutBuilder:
    """Builds the Cycle 2 broad-road + generated-block IMDF layout."""

    def __init__(
        self,
        *,
        road_grid_m: float = 0.25,
        road_min_area_m2: float = 4.0,
        block_depth_m: float = 4.0,
        block_gap_m: float = 0.08,
        block_grid_m: float = 0.50,
        min_context_block_area_m2: float = 1.2,
        min_destination_block_area_m2: float = 1.0,
        max_context_block_length_m: float = 7.0,
        destination_gap_m: float = 0.20,
        display_point_inset_m: float = 0.45,
    ) -> None:
        self._road_grid_m = road_grid_m
        self._road_min_area_m2 = road_min_area_m2
        self._block_depth_m = block_depth_m
        self._block_gap_m = block_gap_m
        self._block_grid_m = block_grid_m
        self._min_context_block_area_m2 = min_context_block_area_m2
        self._min_destination_block_area_m2 = min_destination_block_area_m2
        self._max_context_block_length_m = max_context_block_length_m
        self._destination_gap_m = destination_gap_m
        self._display_point_inset_m = display_point_inset_m

    def build(
        self,
        *,
        scan_id: str,
        footprint_geojson: dict[str, object],
        analyzed_pois: list[AnalyzedPoi] | None = None,
        rectification: dict[str, Any] | None = None,
    ) -> CartographicLayoutResult:
        observed = _polygonal_part(shape(footprint_geojson).buffer(0))
        observed = _filter_polygonal(observed, min_area=self._road_min_area_m2)
        if observed.is_empty:
            return self._fallback(scan_id=scan_id, footprint_geojson=footprint_geojson)

        orientation_deg = _orientation_deg(observed, rectification=rectification)
        origin = (observed.centroid.x, observed.centroid.y)
        road = _rectilinearize(
            observed,
            grid_m=self._road_grid_m,
            orientation_deg=orientation_deg,
            origin=origin,
            min_area_m2=self._road_min_area_m2,
        )
        road = _straighten_road_outline(
            road,
            tolerance_m=max(0.5, self._road_grid_m * 2.0),
            min_area_m2=self._road_min_area_m2,
            max_area_change_ratio=0.12,
        )
        if road.is_empty:
            return self._fallback(scan_id=scan_id, footprint_geojson=footprint_geojson)

        road_rot = affinity.rotate(road, -orientation_deg, origin=origin)
        envelope_rot = _rectilinearize_axis_aligned(
            road_rot.buffer(self._block_depth_m, join_style=2, mitre_limit=3.0),
            grid_m=self._block_grid_m,
            min_area_m2=self._min_context_block_area_m2,
        )
        envelope = _polygonal_part(
            affinity.rotate(envelope_rot, orientation_deg, origin=origin).buffer(0)
        )
        block_ring_rot = _polygonal_part(
            envelope_rot.difference(
                road_rot.buffer(self._block_gap_m, join_style=2, mitre_limit=3.0)
            ).buffer(0)
        )

        boundary_segments = _road_boundary_segments(road_rot)
        stations = self._assign_and_pack_pois(
            analyzed_pois=sorted(analyzed_pois or [], key=_poi_sort_key),
            boundary_segments=boundary_segments,
            origin=origin,
            orientation_deg=orientation_deg,
        )
        destinations = self._build_destinations(
            stations=stations,
            road=road,
            road_rot=road_rot,
            envelope_rot=envelope_rot,
            block_ring_rot=block_ring_rot,
            origin=origin,
            orientation_deg=orientation_deg,
        )
        destination_by_poi_id = {item[0].poi.node.poi_mark_id: item for item in destinations}
        destination_union_rot = (
            unary_union([item[2] for item in destinations]).buffer(0)
            if destinations
            else Polygon()
        )
        context_units = self._build_context_units(
            scan_id=scan_id,
            block_ring_rot=block_ring_rot,
            destination_union_rot=destination_union_rot,
            boundary_segments=boundary_segments,
            stations=stations,
            origin=origin,
            orientation_deg=orientation_deg,
        )

        units = [
            UnitFeature(
                id="unit-walkway-0",
                geometry=_as_geojson(road),
                category="walkway",
                name={"ko": "통로"},
                properties={
                    "semantic": False,
                    "display_role": "walkable_surface",
                    "source": "rectified_observed_walkable_footprint",
                    "fill_intent": "road",
                    "route_graph_source": False,
                },
            )
        ]
        units.extend(context_units)

        place_areas: list[PlaceAreaFeature] = []
        amenities: list[SemanticAmenityFeature] = []
        for analyzed in sorted(analyzed_pois or [], key=_poi_sort_key):
            node = analyzed.node
            assert node.poi_mark_id is not None
            destination = destination_by_poi_id.get(node.poi_mark_id)
            if destination is None:
                display_area_id = None
                display_point = (node.x, node.y, node.z)
            else:
                station, block, _block_rot, display_point = destination
                display_area_id = f"place-poi-{node.poi_mark_id}"
                place_areas.append(
                    PlaceAreaFeature(
                        id=display_area_id,
                        category=analyzed.analysis.category,
                        name=analyzed.analysis.name,
                        geometry=_as_multipolygon_geojson(block),
                        entrance_node_id=node.node_id,
                        source_poi_mark_id=node.poi_mark_id,
                        source="generated_cartographic_block_subdivision",
                        road_side=_road_side_from_normal(station.segment.normal),
                        boundary_segment_id=station.segment.id,
                        boundary_station_m=round(float(station.station_m), 4),
                    )
                )
            amenities.append(
                SemanticAmenityFeature(
                    id=f"amenity-{node.poi_mark_id}",
                    poi_mark_id=node.poi_mark_id,
                    route_node_id=node.node_id,
                    category=analyzed.analysis.category,
                    name=analyzed.analysis.name,
                    point=(node.x, node.y, node.z),
                    display_point=display_point,
                    display_area_id=display_area_id,
                    analysis=analyzed.analysis,
                )
            )

        units_with_places_count = len(units) + len(place_areas)
        categories = Counter(str(unit.category) for unit in units)
        categories.update(str(place.category) for place in place_areas)
        block_ring = _polygonal_part(
            affinity.rotate(block_ring_rot, orientation_deg, origin=origin).buffer(0)
        )
        block_area = sum(float(shape(unit.geometry).area) for unit in context_units)
        block_area += sum(float(shape(area.geometry).area) for area in place_areas)
        block_ring_area = float(block_ring.area)
        metadata = {
            "enabled": True,
            "fallback": False,
            "display_mode": "observed_road_with_generated_blocks",
            "road_source": "observed_walkable_footprint",
            "walkway_source": "rectified_observed_walkable_footprint",
            "centerline_unit_count": 0,
            "centerline_rendered_as_unit": False,
            "walkway_half_width_m": None,
            "walkway_half_width_deprecated": True,
            "road_style": {
                "surface_role": "walkable_open_area",
                "rectilinearized": True,
                "orientation_deg": round(float(orientation_deg), 4),
                "grid_m": self._road_grid_m,
                "join_style": "orthogonal_grid_union",
                "cap_style": "not_applicable",
                "outline_simplified": True,
                "outline_simplify_tolerance_m": max(0.5, self._road_grid_m * 2.0),
            },
            "block_split": {
                "enabled": True,
                "source": "generated_cartographic_block_envelope",
                "context_block_count": len(context_units),
                "destination_block_count": len(place_areas),
                "block_depth_m": self._block_depth_m,
                "road_gap_m": self._block_gap_m,
                "max_context_block_length_m": self._max_context_block_length_m,
                "min_context_block_area_m2": self._min_context_block_area_m2,
                "min_destination_block_area_m2": self._min_destination_block_area_m2,
                "coverage_ratio": round(block_area / block_ring_area, 4)
                if block_ring_area > 0
                else 0.0,
            },
            "observed_footprint_area_m2": round(float(observed.area), 4),
            "road_area_m2": round(float(road.area), 4),
            "road_to_observed_area_ratio": round(float(road.area / observed.area), 4),
            "block_envelope_area_m2": round(float(envelope.area), 4),
            "block_ring_area_m2": round(block_ring_area, 4),
            "semantic_place_count": len(place_areas),
            "connector_policy": {
                "poi_spur_edges_in_route_path": False,
                "route_endpoint_for_poi": "poi_attach",
                "centerline_is_display_unit": False,
            },
            "unit_count": units_with_places_count,
            "parcel_count": 0,
            "categories": dict(sorted(categories.items())),
        }
        return CartographicLayoutResult(
            units=units,
            place_areas=place_areas,
            amenities=amenities,
            road_geometry=_as_geojson(road),
            block_envelope_geometry=_as_geojson(envelope),
            metadata=metadata,
        )

    def _assign_and_pack_pois(
        self,
        *,
        analyzed_pois: list[AnalyzedPoi],
        boundary_segments: list[_BoundarySegment],
        origin: tuple[float, float],
        orientation_deg: float,
    ) -> list[_PoiStation]:
        if not analyzed_pois or not boundary_segments:
            return []

        grouped: dict[str, list[_PoiStation]] = {}
        for analyzed in analyzed_pois:
            point_rot = affinity.rotate(
                Point(analyzed.node.x, analyzed.node.y),
                -orientation_deg,
                origin=origin,
            )
            width, depth = _size_for_category(analyzed.analysis.category)
            segment, station_m = _nearest_segment(point_rot, boundary_segments)
            grouped.setdefault(segment.id, []).append(
                _PoiStation(
                    poi=analyzed,
                    segment=segment,
                    station_m=station_m,
                    width_m=width,
                    depth_m=depth,
                    interval=(station_m - width / 2.0, station_m + width / 2.0),
                )
            )

        packed: list[_PoiStation] = []
        for segment_id in sorted(grouped):
            group = sorted(grouped[segment_id], key=lambda s: (s.station_m, _poi_sort_key(s.poi)))
            cursor = 0.0
            for packed_station in group:
                width = min(
                    packed_station.width_m,
                    max(0.7 * packed_station.width_m, packed_station.segment.length_m),
                )
                start = max(packed_station.station_m - width / 2.0, cursor)
                end = start + width
                if end > packed_station.segment.length_m:
                    end = packed_station.segment.length_m
                    start = max(0.0, end - width)
                cursor = end + self._destination_gap_m
                packed.append(
                    _PoiStation(
                        poi=packed_station.poi,
                        segment=packed_station.segment,
                        station_m=(start + end) / 2.0,
                        width_m=end - start,
                        depth_m=packed_station.depth_m,
                        interval=(start, end),
                    )
                )
        return sorted(packed, key=lambda s: _poi_sort_key(s.poi))

    def _build_destinations(
        self,
        *,
        stations: list[_PoiStation],
        road: BaseGeometry,
        road_rot: BaseGeometry,
        envelope_rot: BaseGeometry,
        block_ring_rot: BaseGeometry,
        origin: tuple[float, float],
        orientation_deg: float,
    ) -> list[tuple[_PoiStation, BaseGeometry, BaseGeometry, tuple[float, float, float]]]:
        accepted_rot: list[BaseGeometry] = []
        results: list[
            tuple[_PoiStation, BaseGeometry, BaseGeometry, tuple[float, float, float]]
        ] = []
        for station in stations:
            accepted_union = unary_union(accepted_rot).buffer(0.02) if accepted_rot else Polygon()
            placed_station, block_rot = self._pick_destination_block(
                station=station,
                road_rot=road_rot,
                envelope_rot=envelope_rot,
                block_ring_rot=block_ring_rot,
                accepted_union=accepted_union,
            )
            if block_rot.is_empty:
                continue
            block = _polygonal_part(
                affinity.rotate(block_rot, orientation_deg, origin=origin).buffer(0)
            )
            display_point = self._display_point(
                station=placed_station,
                block=block,
                block_rot=block_rot,
                road=road,
                road_rot=road_rot,
                origin=origin,
                orientation_deg=orientation_deg,
            )
            accepted_rot.append(block_rot)
            results.append((placed_station, block, block_rot, display_point))
        return results

    def _pick_destination_block(
        self,
        *,
        station: _PoiStation,
        road_rot: BaseGeometry,
        envelope_rot: BaseGeometry,
        block_ring_rot: BaseGeometry,
        accepted_union: BaseGeometry,
    ) -> tuple[_PoiStation, BaseGeometry]:
        """Place a POI marker block outside the road, even for crowded edges.

        The visual contract is marker first: POI capture nodes may live on the
        route graph, but map labels/icons must be displayed in the generated
        non-road fabric. Exact room subdivision is intentionally secondary.
        """

        min_area = self._min_destination_block_area_m2
        for candidate in self._destination_station_candidates(station):
            block_rot = self._clean_destination_candidate(
                self._destination_rect(candidate).intersection(block_ring_rot),
                road_rot=road_rot,
                accepted_union=accepted_union,
                clip_to=block_ring_rot,
                min_area_m2=0.6,
            )
            if block_rot.area >= min_area:
                return candidate, block_rot

        for candidate in self._destination_station_candidates(station, expanded=True):
            block_rot = self._clean_destination_candidate(
                self._destination_rect(candidate).intersection(envelope_rot),
                road_rot=road_rot,
                accepted_union=accepted_union,
                clip_to=envelope_rot,
                min_area_m2=0.35,
            )
            if block_rot.area >= min_area:
                return candidate, block_rot

        fallback = self._compact_marker_station(station)
        block_rot = self._clean_destination_candidate(
            self._destination_rect(fallback).intersection(envelope_rot),
            road_rot=road_rot,
            accepted_union=Polygon(),
            clip_to=envelope_rot,
            min_area_m2=0.2,
        )
        if block_rot.area >= 0.25:
            return fallback, block_rot
        return fallback, Polygon()

    def _destination_station_candidates(
        self,
        station: _PoiStation,
        *,
        expanded: bool = False,
    ) -> list[_PoiStation]:
        base_width = max(station.width_m, 1.2)
        widths = [base_width]
        if expanded:
            widths.extend([
                min(station.segment.length_m, base_width + 1.0),
                min(station.segment.length_m, base_width + 2.0),
                min(station.segment.length_m, max(base_width, 4.5)),
            ])
        offsets = [0.0, 0.6, -0.6, 1.2, -1.2, 2.0, -2.0, 3.0, -3.0]
        candidates: list[_PoiStation] = []
        seen: set[tuple[float, float]] = set()
        for width in widths:
            width = max(0.6, min(width, station.segment.length_m))
            for offset in offsets:
                candidate = _station_with_interval(
                    station,
                    center_m=station.station_m + offset,
                    width_m=width,
                )
                key = (round(candidate.interval[0], 3), round(candidate.interval[1], 3))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    def _compact_marker_station(self, station: _PoiStation) -> _PoiStation:
        return _station_with_interval(
            station,
            center_m=station.station_m,
            width_m=min(max(1.2, station.width_m), station.segment.length_m),
        )

    def _clean_destination_candidate(
        self,
        geom: BaseGeometry,
        *,
        road_rot: BaseGeometry,
        accepted_union: BaseGeometry,
        clip_to: BaseGeometry,
        min_area_m2: float,
    ) -> BaseGeometry:
        road_clearance = road_rot.buffer(self._block_gap_m, join_style=2, mitre_limit=3.0)
        cleaned = _polygonal_part(geom.difference(road_clearance).buffer(0))
        if not accepted_union.is_empty:
            cleaned = _polygonal_part(cleaned.difference(accepted_union).buffer(0))
        rectified = _rectilinearize_axis_aligned(
            cleaned,
            grid_m=self._block_grid_m,
            min_area_m2=min_area_m2,
        )
        rectified = _polygonal_part(rectified.intersection(clip_to).buffer(0))
        return _largest_polygonal(rectified)

    def _destination_rect(self, station: _PoiStation) -> Polygon:
        depth = min(station.depth_m, self._block_depth_m - self._block_gap_m)
        depth = max(2.4, depth)
        return _segment_rect_unclamped(
            segment=station.segment,
            start_m=station.interval[0],
            end_m=station.interval[1],
            inner_m=self._block_gap_m,
            outer_m=depth,
        )

    def _display_point(
        self,
        *,
        station: _PoiStation,
        block: BaseGeometry,
        block_rot: BaseGeometry,
        road: BaseGeometry,
        road_rot: BaseGeometry,
        origin: tuple[float, float],
        orientation_deg: float,
    ) -> tuple[float, float, float]:
        point = block.centroid
        if not block.covers(point):
            point = block.representative_point()
        if road.distance(point) >= self._display_point_inset_m:
            return (float(point.x), float(point.y), station.poi.node.z)

        sx, sy = station.segment.start
        tx, ty = station.segment.tangent
        nx, ny = station.segment.normal
        probe_start = (
            sx + tx * station.station_m + nx * self._display_point_inset_m,
            sy + ty * station.station_m + ny * self._display_point_inset_m,
        )
        probe = LineString([
            probe_start,
            (
                sx + tx * station.station_m + nx * (self._block_depth_m - 0.2),
                sy + ty * station.station_m + ny * (self._block_depth_m - 0.2),
            ),
        ])
        clipped = probe.intersection(block_rot)
        candidates = _line_points(clipped)
        if candidates:
            chosen = max(candidates, key=lambda p: road_rot.distance(p))
            point = affinity.rotate(chosen, orientation_deg, origin=origin)
        return (float(point.x), float(point.y), station.poi.node.z)

    def _build_context_units(
        self,
        *,
        scan_id: str,
        block_ring_rot: BaseGeometry,
        destination_union_rot: BaseGeometry,
        boundary_segments: list[_BoundarySegment],
        stations: list[_PoiStation],
        origin: tuple[float, float],
        orientation_deg: float,
    ) -> list[UnitFeature]:
        remaining = _polygonal_part(block_ring_rot.difference(destination_union_rot).buffer(0))
        occupied_by_segment: dict[str, list[tuple[float, float]]] = {}
        for station in stations:
            occupied_by_segment.setdefault(station.segment.id, []).append(station.interval)

        accepted = Polygon()
        context_rot: list[BaseGeometry] = []
        for segment in boundary_segments:
            cuts = [0.0, segment.length_m]
            chunk_count = max(1, ceil(segment.length_m / self._max_context_block_length_m))
            cuts.extend(segment.length_m * idx / chunk_count for idx in range(1, chunk_count))
            for start, end in occupied_by_segment.get(segment.id, []):
                cuts.extend([max(0.0, start), min(segment.length_m, end)])
            cuts = sorted({round(cut, 6) for cut in cuts if 0.0 <= cut <= segment.length_m})
            for start, end in zip(cuts, cuts[1:], strict=False):
                if end - start < 0.3:
                    continue
                if _interval_occupied(
                    start,
                    end,
                    occupied_by_segment.get(segment.id, []),
                    gap=self._destination_gap_m / 2.0,
                ):
                    continue
                rect = _segment_rect(
                    segment=segment,
                    start_m=start,
                    end_m=end,
                    inner_m=self._block_gap_m,
                    outer_m=self._block_depth_m,
                )
                candidate = _polygonal_part(
                    rect.intersection(remaining).difference(accepted).buffer(0)
                )
                if candidate.area < self._min_context_block_area_m2:
                    continue
                context_rot.append(candidate)
                accepted = unary_union([accepted, candidate]).buffer(0)

        leftover = _polygonal_part(remaining.difference(accepted).buffer(0))
        context_rot.extend(
            poly for poly in _iter_polygons(leftover)
            if poly.area >= self._min_context_block_area_m2
        )

        context_world = [
            _polygonal_part(affinity.rotate(geom, orientation_deg, origin=origin).buffer(0))
            for geom in context_rot
        ]
        context_world = sorted(
            [geom for geom in context_world if geom.area >= self._min_context_block_area_m2],
            key=lambda geom: (round(geom.centroid.x, 4), round(geom.centroid.y, 4)),
        )
        return [
            UnitFeature(
                id=f"unit-block-{idx}",
                geometry=_as_geojson(geom),
                category="block",
                name=None,
                properties={
                    "semantic": False,
                    "display_role": "context_block",
                    "source": "generated_cartographic_block_envelope",
                    "block_id": f"block-{idx}",
                    "route_excluded": True,
                    "generated_outside_observed_footprint": True,
                },
            )
            for idx, geom in enumerate(context_world)
        ]

    def _fallback(
        self,
        *,
        scan_id: str,
        footprint_geojson: dict[str, object],
    ) -> CartographicLayoutResult:
        unit = UnitFeature(
            id="unit-walkway-0",
            geometry=footprint_geojson,
            category="walkway",
            name={"ko": "통로"},
            properties={
                "semantic": False,
                "display_role": "walkable_surface",
                "source": "fallback_observed_walkable_footprint",
                "fill_intent": "road",
                "route_graph_source": False,
            },
        )
        metadata = {
            "enabled": True,
            "fallback": True,
            "display_mode": "observed_road_with_generated_blocks",
            "road_source": "observed_walkable_footprint",
            "walkway_source": "rectified_observed_walkable_footprint",
            "centerline_unit_count": 0,
            "centerline_rendered_as_unit": False,
            "walkway_half_width_m": None,
            "walkway_half_width_deprecated": True,
            "road_style": {
                "surface_role": "walkable_open_area",
                "rectilinearized": False,
                "orientation_deg": 0.0,
                "grid_m": self._road_grid_m,
                "join_style": "observed_geometry",
                "cap_style": "not_applicable",
            },
            "block_split": {
                "enabled": False,
                "source": "generated_cartographic_block_envelope",
                "context_block_count": 0,
                "destination_block_count": 0,
            },
            "semantic_place_count": 0,
            "connector_policy": {
                "poi_spur_edges_in_route_path": False,
                "route_endpoint_for_poi": "poi_attach",
                "centerline_is_display_unit": False,
            },
            "unit_count": 1,
            "parcel_count": 0,
            "categories": {"walkway": 1},
        }
        return CartographicLayoutResult(
            units=[unit],
            place_areas=[],
            amenities=[],
            road_geometry=footprint_geojson,
            block_envelope_geometry=footprint_geojson,
            metadata=metadata,
        )


def _poi_sort_key(poi: AnalyzedPoi) -> int:
    return int(poi.node.poi_mark_id or 0)


def _orientation_deg(
    observed: BaseGeometry,
    *,
    rectification: dict[str, Any] | None,
) -> float:
    if rectification:
        for key in ("orientation_deg", "primary_axis_deg", "angle_deg"):
            value = rectification.get(key)
            if isinstance(value, int | float):
                return float(value) % 90.0

    largest = max(_iter_polygons(observed), key=lambda poly: poly.area, default=None)
    if largest is None or largest.area <= 0:
        return 0.0
    coords = list(largest.exterior.coords)
    if len(coords) < 2:
        return 0.0
    edges = []
    for i in range(len(coords) - 1):
        length = hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
        edges.append((coords[i], coords[i + 1], length))
    start, end, length = max(edges, key=lambda item: item[2])
    if length <= 1e-6:
        return 0.0
    angle = degrees(atan2(end[1] - start[1], end[0] - start[0])) % 180.0
    return angle - 90.0 if angle >= 90.0 else angle


def _rectilinearize(
    geom: BaseGeometry,
    *,
    grid_m: float,
    orientation_deg: float,
    origin: tuple[float, float],
    min_area_m2: float,
) -> BaseGeometry:
    rotated = affinity.rotate(geom, -orientation_deg, origin=origin)
    rectified = _rectilinearize_axis_aligned(rotated, grid_m=grid_m, min_area_m2=min_area_m2)
    return _polygonal_part(affinity.rotate(rectified, orientation_deg, origin=origin).buffer(0))


def _straighten_road_outline(
    geom: BaseGeometry,
    *,
    tolerance_m: float,
    min_area_m2: float,
    max_area_change_ratio: float,
) -> BaseGeometry:
    geom = _polygonal_part(geom.buffer(0))
    if geom.is_empty or geom.area <= 0:
        return geom
    simplified = _polygonal_part(
        geom.simplify(tolerance_m, preserve_topology=True).buffer(0)
    )
    simplified = _filter_polygonal(simplified, min_area=min_area_m2)
    if simplified.is_empty or simplified.area <= 0:
        return geom
    area_change = abs(float(simplified.area) - float(geom.area)) / float(geom.area)
    if area_change > max_area_change_ratio:
        return geom
    return simplified


def _rectilinearize_axis_aligned(
    geom: BaseGeometry,
    *,
    grid_m: float,
    min_area_m2: float,
) -> BaseGeometry:
    geom = _polygonal_part(geom.buffer(0))
    if geom.is_empty:
        return Polygon()
    minx, miny, maxx, maxy = geom.bounds
    minx = grid_m * int((minx - grid_m) / grid_m)
    miny = grid_m * int((miny - grid_m) / grid_m)
    maxx = grid_m * ceil((maxx + grid_m) / grid_m)
    maxy = grid_m * ceil((maxy + grid_m) / grid_m)
    cells: list[Polygon] = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + grid_m, y + grid_m)
            center = Point(x + grid_m / 2.0, y + grid_m / 2.0)
            if geom.covers(center) or cell.intersection(geom).area / cell.area >= 0.35:
                cells.append(cell)
            y += grid_m
        x += grid_m
    if not cells:
        return Polygon()
    cleaned = _drop_small_holes(unary_union(cells).buffer(0))
    simplified = cleaned.simplify(grid_m / 2.0, preserve_topology=True).buffer(0)
    return _filter_polygonal(simplified, min_area=min_area_m2)


def _road_boundary_segments(road_rot: BaseGeometry) -> list[_BoundarySegment]:
    segments: list[_BoundarySegment] = []
    for ring_index, polygon in enumerate(_iter_polygons(road_rot)):
        coords = list(polygon.exterior.coords)
        for segment_index, (start, end) in enumerate(zip(coords, coords[1:], strict=False)):
            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])
            dx = x2 - x1
            dy = y2 - y1
            length = hypot(dx, dy)
            if length < 0.6:
                continue
            if abs(dx) >= abs(dy):
                dy = 0.0
                tangent = (1.0 if dx >= 0.0 else -1.0, 0.0)
            else:
                dx = 0.0
                tangent = (0.0, 1.0 if dy >= 0.0 else -1.0)
            candidates = [(-tangent[1], tangent[0]), (tangent[1], -tangent[0])]
            mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            normal = candidates[0]
            if road_rot.covers(Point(mid[0] + normal[0] * 0.2, mid[1] + normal[1] * 0.2)):
                normal = candidates[1]
            segments.append(
                _BoundarySegment(
                    id=f"road-boundary-{ring_index}-{segment_index}",
                    start=(x1, y1),
                    end=(x2, y2),
                    tangent=tangent,
                    normal=normal,
                    length_m=length,
                )
            )
    return segments


def _nearest_segment(
    point: Point,
    segments: list[_BoundarySegment],
) -> tuple[_BoundarySegment, float]:
    best_segment = segments[0]
    best_station = 0.0
    best_score = float("inf")
    for segment in segments:
        sx, sy = segment.start
        tx, ty = segment.tangent
        raw_station = (point.x - sx) * tx + (point.y - sy) * ty
        station = min(max(raw_station, 0.0), segment.length_m)
        px = sx + tx * station
        py = sy + ty * station
        endpoint_penalty = abs(raw_station - station)
        score = hypot(point.x - px, point.y - py) + endpoint_penalty * 0.35
        if score < best_score:
            best_segment = segment
            best_station = station
            best_score = score
    return best_segment, best_station


def _station_with_interval(
    station: _PoiStation,
    *,
    center_m: float,
    width_m: float,
) -> _PoiStation:
    width_m = max(0.0, min(width_m, station.segment.length_m))
    start = center_m - width_m / 2.0
    end = center_m + width_m / 2.0
    if start < 0.0:
        end -= start
        start = 0.0
    if end > station.segment.length_m:
        start -= end - station.segment.length_m
        end = station.segment.length_m
    start = max(0.0, start)
    end = max(start, min(station.segment.length_m, end))
    return _PoiStation(
        poi=station.poi,
        segment=station.segment,
        station_m=(start + end) / 2.0,
        width_m=end - start,
        depth_m=station.depth_m,
        interval=(start, end),
    )


def _segment_rect(
    *,
    segment: _BoundarySegment,
    start_m: float,
    end_m: float,
    inner_m: float,
    outer_m: float,
) -> Polygon:
    sx, sy = segment.start
    tx, ty = segment.tangent
    nx, ny = segment.normal
    start_m = max(0.0, min(segment.length_m, start_m))
    end_m = max(start_m, min(segment.length_m, end_m))
    return Polygon([
        (sx + tx * start_m + nx * inner_m, sy + ty * start_m + ny * inner_m),
        (sx + tx * end_m + nx * inner_m, sy + ty * end_m + ny * inner_m),
        (sx + tx * end_m + nx * outer_m, sy + ty * end_m + ny * outer_m),
        (sx + tx * start_m + nx * outer_m, sy + ty * start_m + ny * outer_m),
    ])


def _segment_rect_unclamped(
    *,
    segment: _BoundarySegment,
    start_m: float,
    end_m: float,
    inner_m: float,
    outer_m: float,
) -> Polygon:
    sx, sy = segment.start
    tx, ty = segment.tangent
    nx, ny = segment.normal
    if end_m < start_m:
        start_m, end_m = end_m, start_m
    return Polygon([
        (sx + tx * start_m + nx * inner_m, sy + ty * start_m + ny * inner_m),
        (sx + tx * end_m + nx * inner_m, sy + ty * end_m + ny * inner_m),
        (sx + tx * end_m + nx * outer_m, sy + ty * end_m + ny * outer_m),
        (sx + tx * start_m + nx * outer_m, sy + ty * start_m + ny * outer_m),
    ])


def _interval_occupied(
    start: float,
    end: float,
    intervals: list[tuple[float, float]],
    *,
    gap: float,
) -> bool:
    for occupied_start, occupied_end in intervals:
        if start < occupied_end + gap and end > occupied_start - gap:
            return True
    return False


def _line_points(geom: BaseGeometry) -> list[Point]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [Point(geom.x, geom.y)]
    if geom.geom_type in {"LineString", "LinearRing"}:
        coords = list(geom.coords)
        return [Point(x, y) for x, y in coords]
    if hasattr(geom, "geoms"):
        points: list[Point] = []
        for child in geom.geoms:
            points.extend(_line_points(child))
        return points
    return []


def _size_for_category(category: str) -> tuple[float, float]:
    sizes = {
        "room": (3.2, 3.2),
        "lab": (3.8, 3.6),
        "office": (3.0, 3.0),
        "restroom": (3.0, 3.0),
        "stairs": (2.8, 3.2),
        "elevator": (2.4, 2.8),
        "entrance": (3.0, 2.4),
        "destination": (3.0, 3.0),
    }
    return sizes.get(category, (3.0, 3.0))


def _road_side_from_normal(normal: tuple[float, float]) -> str:
    nx, ny = normal
    if abs(nx) >= abs(ny):
        return "right" if nx > 0 else "left"
    return "top" if ny > 0 else "bottom"


def _drop_small_holes(geom: BaseGeometry, *, min_hole_area_m2: float = 1.0) -> BaseGeometry:
    polygons: list[Polygon] = []
    for poly in _iter_polygons(geom):
        holes = [
            ring.coords
            for ring in poly.interiors
            if Polygon(ring).area >= min_hole_area_m2
        ]
        polygons.append(Polygon(poly.exterior.coords, holes))
    if not polygons:
        return Polygon()
    return unary_union(polygons).buffer(0)


def _filter_polygonal(geom: BaseGeometry, *, min_area: float) -> BaseGeometry:
    polygons = [poly for poly in _iter_polygons(geom) if poly.area >= min_area]
    if not polygons:
        return Polygon()
    return unary_union(polygons).buffer(0)


def _polygonal_part(geom: BaseGeometry) -> BaseGeometry:
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if not hasattr(geom, "geoms"):
        return Polygon()
    polygons = [
        child for child in geom.geoms
        if isinstance(child, (Polygon, MultiPolygon)) and not child.is_empty
    ]
    if not polygons:
        return Polygon()
    return unary_union(polygons).buffer(0)


def _largest_polygonal(geom: BaseGeometry) -> BaseGeometry:
    polygons = _iter_polygons(geom)
    if not polygons:
        return Polygon()
    return max(polygons, key=lambda poly: poly.area).buffer(0)


def _iter_polygons(geom: BaseGeometry) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if not poly.is_empty]
    if hasattr(geom, "geoms"):
        polygons: list[Polygon] = []
        for child in geom.geoms:
            polygons.extend(_iter_polygons(child))
        return polygons
    return []


def _as_geojson(geom: BaseGeometry) -> dict[str, object]:
    geo = mapping(_polygonal_part(geom.buffer(0)))
    return {"type": geo["type"], "coordinates": geo["coordinates"]}


def _as_multipolygon_geojson(geom: BaseGeometry) -> dict[str, Any]:
    geo = mapping(_polygonal_part(geom.buffer(0)))
    if geo["type"] == "Polygon":
        return {"type": "MultiPolygon", "coordinates": [geo["coordinates"]]}
    if geo["type"] == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": geo["coordinates"]}
    return {"type": "MultiPolygon", "coordinates": []}
