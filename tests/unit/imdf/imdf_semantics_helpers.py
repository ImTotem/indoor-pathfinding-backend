"""Shared IMDF semantic geometry assertions for regression tests."""
from __future__ import annotations

from itertools import combinations
from typing import Any

import pytest
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry

_FORBIDDEN_BASE_CATEGORIES = {"unknown_area", "open_area", "pocket", "wall"}


def assert_naver_like_imdf_semantics(
    *,
    manifest: dict[str, object],
    unit_geojson: dict[str, object],
    footprint_geojson: dict[str, object],
    amenity_geojson: dict[str, object],
    walkway_ratio_range: tuple[float, float] = (0.85, 1.15),
    expected_semantic_count: int = 5,
) -> dict[str, float]:
    """Assert the measured Naver-like road + boundary block IMDF semantics."""
    unit_features = _features(unit_geojson)
    footprint_features = _features(footprint_geojson)
    amenity_features = _features(amenity_geojson)
    assert len(footprint_features) == 1, "footprint feature count"

    non_semantic_units = [
        feature for feature in unit_features
        if feature.get("properties", {}).get("semantic") is not True
    ]
    walkway_units = [
        feature for feature in non_semantic_units
        if feature.get("id") == "unit-walkway-0"
    ]
    context_units = [
        feature for feature in non_semantic_units
        if str(feature.get("id", "")).startswith("unit-block-")
    ]
    semantic_units = [
        feature for feature in unit_features
        if feature.get("properties", {}).get("semantic") is True
    ]
    unit_ids = [str(feature.get("id", "")) for feature in unit_features]
    assert not any(unit_id.startswith("unit-parcel-") for unit_id in unit_ids), "unit-parcel"
    assert len(walkway_units) == 1, "walkway unit count"
    assert walkway_units[0].get("properties", {}).get("category") == "walkway", (
        "walkway category"
    )
    assert walkway_units[0].get("properties", {}).get("display_role") == "walkable_surface"
    assert context_units, "context block count"
    for feature in context_units:
        props = feature.get("properties", {})
        assert props.get("category") == "block", "context block category"
        assert props.get("display_role") == "context_block", "context block display_role"
        assert props.get("route_excluded") is True, "context block route_excluded"

    unit_split = manifest.get("unit_split")
    assert isinstance(unit_split, dict), "manifest unit_split"
    assert unit_split.get("fallback") is False, "unit_split fallback"
    assert unit_split.get("walkway_source") == "rectified_observed_walkable_footprint", (
        "walkway_source"
    )
    assert unit_split.get("centerline_rendered_as_unit") is False, (
        "centerline_rendered_as_unit"
    )
    assert unit_split.get("walkway_half_width_m") is None, "walkway_half_width_m"
    assert unit_split.get("display_mode") == "observed_road_with_generated_blocks", (
        "display_mode"
    )
    road_style = unit_split.get("road_style")
    assert isinstance(road_style, dict), "road_style"
    assert road_style.get("surface_role") == "walkable_open_area", "road_style surface_role"
    assert road_style.get("join_style") == "orthogonal_grid_union", "road_style join_style"
    assert road_style.get("cap_style") == "not_applicable", "road_style cap_style"
    block_split = unit_split.get("block_split")
    assert isinstance(block_split, dict), "block_split"
    assert block_split.get("enabled") is True, "block_split enabled"
    assert block_split.get("source") == "generated_cartographic_block_envelope", (
        "block_split source"
    )
    assert block_split.get("context_block_count") == len(context_units), "context_block_count"
    assert block_split.get("destination_block_count") == expected_semantic_count, (
        "destination_block_count"
    )
    assert float(block_split.get("coverage_ratio", 0.0)) >= 0.75, "block ring coverage"

    footprint = _valid_shape(footprint_features[0]["geometry"])
    walkway = _valid_shape(walkway_units[0]["geometry"])
    footprint_area = float(footprint.area)
    walkway_area = float(walkway.area)
    walkway_ratio = walkway_area / footprint_area
    assert walkway_ratio_range[0] <= walkway_ratio <= walkway_ratio_range[1], (
        f"walkway_ratio {walkway_ratio:.3f} outside {walkway_ratio_range}"
    )
    assert unit_split.get("road_to_observed_area_ratio") == pytest.approx(
        walkway_ratio,
        abs=0.01,
    )
    assert float(unit_split.get("block_envelope_area_m2", 0.0)) > walkway_area, (
        "block envelope area"
    )
    assert float(unit_split.get("block_ring_area_m2", 0.0)) > 0.0, "block ring area"

    assert unit_split.get("parcel_count") == 0, "unit_split parcel_count"
    base_categories = {
        str(feature.get("properties", {}).get("category"))
        for feature in non_semantic_units
    }
    forbidden = base_categories & _FORBIDDEN_BASE_CATEGORIES
    assert not forbidden, f"forbidden base categories: {sorted(forbidden)}"

    max_context_walkway_overlap_area = 0.0
    for feature in context_units:
        geom = _valid_shape(feature["geometry"])
        overlap_area = float(geom.intersection(walkway).area)
        assert overlap_area <= 0.02, f"context-walkway overlap area {overlap_area:.3f}"
        max_context_walkway_overlap_area = max(max_context_walkway_overlap_area, overlap_area)

    assert len(semantic_units) == expected_semantic_count, "semantic unit count"
    category_counts: dict[str, int] = {}
    semantic_geoms: dict[str, BaseGeometry] = {}
    max_semantic_walkway_overlap_area = 0.0
    max_semantic_walkway_overlap_ratio = 0.0
    for feature in semantic_units:
        category = str(feature.get("properties", {}).get("category"))
        props = feature.get("properties", {})
        assert props.get("display_role") == "destination_block", "destination display_role"
        assert props.get("source") == "generated_cartographic_block_subdivision", (
            "destination source"
        )
        assert props.get("route_excluded") is True, "destination route_excluded"
        assert props.get("boundary_segment_id"), "destination boundary_segment_id"
        assert isinstance(props.get("boundary_station_m"), int | float), (
            "destination boundary_station_m"
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        geom = _valid_shape(feature["geometry"])
        assert geom.area >= 0.6, "semantic area"
        overlap_area = float(geom.intersection(walkway).area)
        overlap_ratio = overlap_area / float(geom.area)
        assert overlap_area <= 0.05, f"semantic-walkway overlap area {overlap_area:.3f}"
        assert overlap_ratio < 0.01, f"semantic-walkway overlap ratio {overlap_ratio:.3f}"
        semantic_geoms[str(feature["id"])] = geom
        max_semantic_walkway_overlap_area = max(
            max_semantic_walkway_overlap_area,
            overlap_area,
        )
        max_semantic_walkway_overlap_ratio = max(
            max_semantic_walkway_overlap_ratio,
            overlap_ratio,
        )

    assert category_counts.get("room", 0) >= 2, "room category count"
    assert category_counts.get("lab", 0) >= 1, "lab category count"
    assert category_counts.get("stairs", 0) >= 1, "stairs category count"
    assert category_counts.get("elevator", 0) >= 1, "elevator category count"

    max_place_place_overlap_area = 0.0
    max_place_place_overlap_ratio = 0.0
    for first, second in combinations(semantic_geoms.values(), 2):
        overlap_area = float(first.intersection(second).area)
        overlap_ratio = overlap_area / min(float(first.area), float(second.area))
        assert overlap_area <= 0.05, f"place-place overlap area {overlap_area:.3f}"
        assert overlap_ratio <= 0.02, f"place-place overlap ratio {overlap_ratio:.3f}"
        max_place_place_overlap_area = max(max_place_place_overlap_area, overlap_area)
        max_place_place_overlap_ratio = max(max_place_place_overlap_ratio, overlap_ratio)

    display_inside_all = 1.0
    for feature in amenity_features:
        props = feature.get("properties", {})
        assert isinstance(props, dict), "amenity properties"
        display_area_id = props.get("display_area_id")
        display_point = props.get("display_point")
        assert isinstance(display_area_id, str) and display_area_id, "display_area_id"
        assert isinstance(display_point, list) and len(display_point) >= 2, "display_point"
        assert props.get("geometry_role") == "route_point", "amenity geometry_role"
        assert props.get("display_point_role") == "map_marker", "display_point_role"
        assert props.get("route_excluded") is True, "amenity route_excluded"
        display_geom = semantic_geoms.get(display_area_id)
        assert display_geom is not None, f"display_area_id missing: {display_area_id}"
        display_xy = Point(float(display_point[0]), float(display_point[1]))
        assert display_geom.covers(display_xy), "display_point containment"
        assert walkway.distance(display_xy) >= 0.20, "display point walkway distance"
        route_point = shape(feature["geometry"])
        assert not route_point.is_empty, "route point empty"
        assert walkway.distance(route_point) <= 0.05, "route point walkway distance"

    assert unit_split.get("observed_footprint_area_m2") == pytest.approx(
        footprint_area,
        abs=0.01,
    )
    assert unit_split.get("road_area_m2") == pytest.approx(walkway_area, abs=0.01)
    assert unit_split.get("semantic_place_count") == len(semantic_units), (
        "semantic_place_count"
    )
    return {
        "fixture_footprint_area": footprint_area,
        "fixture_walkway_area": walkway_area,
        "fixture_walkway_ratio": walkway_ratio,
        "fixture_semantic_count": float(len(semantic_units)),
        "fixture_parcel_count": float(unit_split.get("parcel_count", -1)),
        "fixture_context_block_count": float(len(context_units)),
        "fixture_max_context_walkway_overlap_area": max_context_walkway_overlap_area,
        "fixture_max_semantic_walkway_overlap_area": max_semantic_walkway_overlap_area,
        "fixture_max_semantic_walkway_overlap_ratio": max_semantic_walkway_overlap_ratio,
        "fixture_max_place_place_overlap_area": max_place_place_overlap_area,
        "fixture_max_place_place_overlap_ratio": max_place_place_overlap_ratio,
        "fixture_display_inside_all": display_inside_all,
    }


def _features(feature_collection: dict[str, object]) -> list[dict[str, Any]]:
    features = feature_collection.get("features")
    assert isinstance(features, list), "FeatureCollection features"
    return features


def _valid_shape(geometry: object) -> BaseGeometry:
    geom = shape(geometry).buffer(0)
    assert not geom.is_empty, "geometry empty"
    assert geom.is_valid, "geometry valid"
    return geom
