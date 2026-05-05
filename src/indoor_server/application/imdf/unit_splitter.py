"""Compatibility wrapper for IMDF display unit generation."""
from __future__ import annotations

from dataclasses import dataclass

from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow


@dataclass(frozen=True)
class UnitFeature:
    id: str
    geometry: dict[str, object]
    category: str
    name: dict[str, str] | None = None
    properties: dict[str, object] | None = None


@dataclass(frozen=True)
class UnitSplitResult:
    units: list[UnitFeature]
    metadata: dict[str, object]
    walkway_geometry: dict[str, object]


class UnitSplitter:
    """Legacy entry point now backed by the cartographic display layout.

    The visible walkway is the observed footprint road/open area. Route graph
    centerlines are accepted only for call-site compatibility and are never used
    to generate an IMDF unit. Production unit parcels are forbidden.
    """

    def __init__(
        self,
        *,
        corridor_half_width_m: float = 1.25,
        min_unit_area_m2: float = 4.0,
        open_area_threshold_m2: float = 20.0,
        parcel_length_m: float = 2.8,
        parcel_depth_m: float = 2.2,
        generate_side_parcels: bool = False,
        generate_context_blocks: bool = True,
    ) -> None:
        self._corridor_half_width_m = corridor_half_width_m
        self._min_unit_area_m2 = min_unit_area_m2
        self._open_area_threshold_m2 = open_area_threshold_m2
        self._parcel_length_m = parcel_length_m
        self._parcel_depth_m = parcel_depth_m
        self._generate_side_parcels = generate_side_parcels
        self._generate_context_blocks = generate_context_blocks

    def split(
        self,
        *,
        scan_id: str,
        footprint_geojson: dict[str, object],
        nodes: list[MapNodeRow],
        edges: list[MapEdgeRow],
    ) -> UnitSplitResult:
        from indoor_server.application.imdf.cartographic_layout import (
            CartographicLayoutBuilder,
        )

        _ = (
            self._corridor_half_width_m,
            self._min_unit_area_m2,
            self._open_area_threshold_m2,
            self._parcel_length_m,
            self._parcel_depth_m,
            self._generate_side_parcels,
            self._generate_context_blocks,
            nodes,
            edges,
        )
        layout = CartographicLayoutBuilder().build(
            scan_id=scan_id,
            footprint_geojson=footprint_geojson,
            analyzed_pois=[],
        )
        return UnitSplitResult(
            units=layout.units,
            metadata=layout.metadata,
            walkway_geometry=layout.road_geometry,
        )
