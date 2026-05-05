"""IMDF-lite 도메인 VO — 순수 dataclass/enum. 외부 의존 없음."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class ImdfFeatureClass(StrEnum):
    """IMDF-lite feature class 열거형."""

    LEVEL = "level"
    UNIT = "unit"
    FOOTPRINT = "footprint"
    AMENITY = "amenity"
    ANCHOR = "anchor"


@dataclass(frozen=True)
class ImdfFeatureSet:
    """5 FeatureCollection dict + manifest dict 묶음. 순수 VO."""

    manifest: dict[str, object]
    level_geojson: dict[str, object]
    unit_geojson: dict[str, object]
    footprint_geojson: dict[str, object]
    amenity_geojson: dict[str, object]
    anchor_geojson: dict[str, object]


@dataclass(frozen=True)
class ImdfArchive:
    """완성된 IMDF zip bytes + 메타."""

    scan_id: UUID
    build_job_id: UUID
    payload: bytes
    filename: str  # "{scan_id}.imdf.zip"
