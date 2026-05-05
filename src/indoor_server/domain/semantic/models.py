"""Semantic map 도메인 VO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

PlaceCategory = Literal[
    "room",
    "lab",
    "office",
    "restroom",
    "stairs",
    "elevator",
    "entrance",
    "destination",
    "service",
    "unknown",
]


@dataclass(frozen=True)
class SemanticAnalysis:
    """Mock/LLM analyzer 결과."""

    category: PlaceCategory
    name: str | None
    confidence: float
    analyzer: str
    connector_type: Literal["stairs", "elevator"] | None = None
    connector_key: str | None = None
    raw_text: str | None = None
    needs_review: bool = False

    def as_metadata(self) -> dict[str, Any]:
        return {
            "analyzer": self.analyzer,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "raw_text": self.raw_text,
        }


@dataclass(frozen=True)
class PlaceAreaFeature:
    """지도 표시용 장소/시설 영역."""

    id: str
    category: PlaceCategory
    name: str | None
    geometry: dict[str, Any]
    entrance_node_id: UUID
    source_poi_mark_id: int
    source: str = "wall_boundary_subdivision"
    road_side: str = "unknown"
    boundary_segment_id: str | None = None
    boundary_station_m: float | None = None


@dataclass(frozen=True)
class SemanticAmenityFeature:
    """길찾기용 POI + 지도 표시 metadata."""

    id: str
    poi_mark_id: int
    route_node_id: UUID
    category: PlaceCategory
    name: str | None
    point: tuple[float, float, float]
    display_point: tuple[float, float, float]
    display_area_id: str | None
    analysis: SemanticAnalysis
