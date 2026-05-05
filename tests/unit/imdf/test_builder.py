"""ImdfBuilder 단위 테스트 — U1~U7."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from indoor_server.application.imdf.builder import ImdfBuilder
from indoor_server.domain.routing.models import MapNodeRow
from indoor_server.domain.semantic.models import (
    SemanticAmenityFeature,
    SemanticAnalysis,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

_SCAN_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_BUILD_JOB_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_FLOOR_Z0 = 1.34
_NOW = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)

_FOOTPRINT_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0], [0.0, 0.0]]],
}

_FOOTPRINT_MULTIPOLYGON = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0], [0.0, 0.0]]]
    ],
}


def _make_poi_node(
    poi_mark_id: int,
    x: float = 5.0,
    y: float = 2.5,
    z: float = 1.34,
) -> MapNodeRow:
    return MapNodeRow(
        node_id=uuid4(),
        x=x,
        y=y,
        z=z,
        node_type="poi",
        label=None,
        poi_mark_id=poi_mark_id,
    )


def _build_feature_set(
    footprint: dict | None = None,
    poi_nodes: list[MapNodeRow] | None = None,
    poi_labels: dict[int, str | None] | None = None,
):
    builder = ImdfBuilder()
    return builder.build(
        scan_id=_SCAN_ID,
        build_job_id=_BUILD_JOB_ID,
        footprint_geojson=footprint or _FOOTPRINT_MULTIPOLYGON,
        poi_nodes=poi_nodes or [],
        poi_labels=poi_labels or {},
        floor_z0=_FLOOR_Z0,
        generated_at=_NOW,
    )


# ── U1: happy path — 5 FeatureCollection + manifest ─────────────────────────


def test_u1_happy_all_feature_collections() -> None:
    """U1: manifest.feature_files 길이 5, 모든 FeatureCollection 유효 GeoJSON."""
    features = _build_feature_set()

    assert features.manifest["format"] == "indoor-pathfinding-imdf-lite"
    assert len(features.manifest["feature_files"]) == 5

    for fc in [
        features.level_geojson,
        features.unit_geojson,
        features.footprint_geojson,
        features.amenity_geojson,
        features.anchor_geojson,
    ]:
        assert fc["type"] == "FeatureCollection"
        assert isinstance(fc["features"], list)


# ── U2: POI 0개 → amenity.features == [] ─────────────────────────────────────


def test_u2_no_poi_amenity_empty() -> None:
    """U2: poi_nodes=[] → amenity FeatureCollection.features == []."""
    features = _build_feature_set(poi_nodes=[], poi_labels={})
    assert features.amenity_geojson["features"] == []


# ── U3: Polygon → MultiPolygon 자동 wrap ─────────────────────────────────────


def test_u3_polygon_auto_wrapped_to_multipolygon() -> None:
    """U3: Polygon footprint 입력 시 footprint geometry.type == 'MultiPolygon'. unit은 빈 배열."""
    features = _build_feature_set(footprint=_FOOTPRINT_POLYGON)

    fp_geom = features.footprint_geojson["features"][0]["geometry"]

    assert fp_geom["type"] == "MultiPolygon"
    # unit polygon 완전 제거 — 빈 FeatureCollection
    assert features.unit_geojson["features"] == []


# ── U4: extents 계산 ─────────────────────────────────────────────────────────


def test_u4_extents_from_footprint() -> None:
    """U4: extents_m min/max 값이 footprint 좌표 bounds와 일치."""
    features = _build_feature_set(footprint=_FOOTPRINT_MULTIPOLYGON)
    extents = features.manifest["extents_m"]

    assert extents["min_x"] == pytest.approx(0.0)
    assert extents["max_x"] == pytest.approx(10.0)
    assert extents["min_y"] == pytest.approx(0.0)
    assert extents["max_y"] == pytest.approx(5.0)


# ── U5: feature_type 필드 확인 ───────────────────────────────────────────────


def test_u5_feature_types() -> None:
    """U5: 각 FeatureCollection의 첫 번째 feature.properties.feature_type 검증. unit은 빈 배열."""
    poi = _make_poi_node(poi_mark_id=42, x=3.0, y=1.0)
    features = _build_feature_set(
        poi_nodes=[poi],
        poi_labels={42: "강의실 A"},
    )

    assert features.level_geojson["features"][0]["properties"]["feature_type"] == "level"
    assert features.unit_geojson["features"] == []  # unit polygon 제거
    assert features.footprint_geojson["features"][0]["properties"]["feature_type"] == "footprint"
    assert features.amenity_geojson["features"][0]["properties"]["feature_type"] == "amenity"
    assert features.anchor_geojson["features"][0]["properties"]["feature_type"] == "anchor"


# ── U6: amenity label 반영 ───────────────────────────────────────────────────


def test_u6_amenity_label_in_name() -> None:
    """U6: poi_labels에 label 있으면 amenity name.ko에 반영."""
    poi = _make_poi_node(poi_mark_id=7)
    features = _build_feature_set(
        poi_nodes=[poi],
        poi_labels={7: "화장실"},
    )
    amenity_props = features.amenity_geojson["features"][0]["properties"]
    assert amenity_props["name"] == {"ko": "화장실"}
    assert amenity_props["poi_mark_id"] == 7


def test_u6b_amenity_no_label() -> None:
    """U6b: poi_labels에 label 없으면 amenity name은 None."""
    poi = _make_poi_node(poi_mark_id=99)
    features = _build_feature_set(
        poi_nodes=[poi],
        poi_labels={99: None},
    )
    amenity_props = features.amenity_geojson["features"][0]["properties"]
    assert amenity_props["name"] is None


# ── U7: manifest schema (Pydantic validate) ───────────────────────────────────


def test_u7_manifest_schema_valid() -> None:
    """U7: manifest dict이 ImdfManifest Pydantic 스키마 통과."""
    from indoor_server.interfaces.api.schemas import ImdfManifest

    features = _build_feature_set()
    manifest_validated = ImdfManifest.model_validate(features.manifest)

    assert manifest_validated.format == "indoor-pathfinding-imdf-lite"
    assert manifest_validated.version == "0.3"
    assert manifest_validated.coordinate_system == "local_metric"
    assert len(manifest_validated.feature_files) == 5
    assert manifest_validated.z_floor_m == pytest.approx(_FLOOR_Z0)


def test_u8_unit_polygon_removed_and_rectification_preserved() -> None:
    """U8: unit polygon 제거 — unit.geojson은 항상 빈 배열. rectification은 manifest에 유지."""
    features = ImdfBuilder().build(
        scan_id=_SCAN_ID,
        build_job_id=_BUILD_JOB_ID,
        footprint_geojson=_FOOTPRINT_POLYGON,
        poi_nodes=[],
        poi_labels={},
        floor_z0=_FLOOR_Z0,
        generated_at=_NOW,
        rectification={"enabled": True, "accepted": True},
    )

    assert features.unit_geojson["features"] == []
    assert "unit_split" not in features.manifest
    assert features.manifest["rectification"]["accepted"] is True


def test_u9_semantic_amenity_metadata() -> None:
    """U9: semantic amenity metadata는 amenity에 포함. unit은 항상 빈 배열."""
    node_id = uuid4()
    analysis = SemanticAnalysis(
        category="stairs",
        name="동쪽 계단",
        confidence=0.8,
        analyzer="mock",
        connector_type="stairs",
        connector_key="STAIR_A",
    )
    amenity = SemanticAmenityFeature(
        id="amenity-7",
        poi_mark_id=7,
        route_node_id=node_id,
        category="stairs",
        name="동쪽 계단",
        point=(1.0, 2.0, 0.0),
        display_point=(1.2, 2.3, 0.0),
        display_area_id="place-poi-7",
        analysis=analysis,
    )

    features = ImdfBuilder().build(
        scan_id=_SCAN_ID,
        build_job_id=_BUILD_JOB_ID,
        footprint_geojson=_FOOTPRINT_POLYGON,
        poi_nodes=[],
        poi_labels={},
        floor_z0=_FLOOR_Z0,
        generated_at=_NOW,
        semantic_amenities=[amenity],
    )

    # unit polygon 완전 제거
    assert features.unit_geojson["features"] == []

    amenity_props = features.amenity_geojson["features"][0]["properties"]
    assert amenity_props["category"] == "stairs"
    assert amenity_props["geometry_role"] == "route_point"
    assert amenity_props["display_point_role"] == "map_marker"
    assert amenity_props["route_excluded"] is True
    assert amenity_props["route_node_id"] == str(node_id)
    assert amenity_props["display_area_id"] == "place-poi-7"
    assert amenity_props["connector_key"] == "STAIR_A"
