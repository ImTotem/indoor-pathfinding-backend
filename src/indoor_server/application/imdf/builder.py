"""ImdfBuilder — 5 FeatureCollection + manifest dict 생성. 순수 함수, DB/IO 의존 없음."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from indoor_server.domain.imdf.models import ImdfFeatureSet
from indoor_server.domain.routing.models import MapNodeRow
from indoor_server.domain.semantic.models import SemanticAmenityFeature

_FORMAT = "indoor-pathfinding-imdf-lite"
_VERSION = "0.3"
_COORDINATE_SYSTEM = "local_metric"


def _feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _ensure_multipolygon(geojson: dict[str, Any]) -> dict[str, Any]:
    """Polygon 단일을 MultiPolygon으로 wrap. 이미 MultiPolygon이면 그대로."""
    if geojson.get("type") == "Polygon":
        return {"type": "MultiPolygon", "coordinates": [geojson["coordinates"]]}
    return geojson


def _extents_from_multipolygon(geojson: dict[str, Any]) -> dict[str, float]:
    """MultiPolygon GeoJSON에서 min_x/max_x/min_y/max_y 계산."""
    xs: list[float] = []
    ys: list[float] = []
    for polygon_coords in geojson.get("coordinates", []):
        for ring in polygon_coords:
            for pt in ring:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    if not xs:
        return {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0}
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
    }


class ImdfBuilder:
    """5 FeatureCollection + manifest를 메모리에서 조립.

    순수 — 단위 테스트 fixture만으로 검증 가능.
    """

    def build(
        self,
        *,
        scan_id: UUID,
        build_job_id: UUID,
        footprint_geojson: dict[str, Any],
        poi_nodes: list[MapNodeRow],
        poi_labels: dict[int, str | None],
        floor_z0: float,
        generated_at: datetime,
        language: str = "ko-KR",
        rectification: dict[str, Any] | None = None,
        semantic_amenities: list[SemanticAmenityFeature] | None = None,
        build_source: str | None = None,
        rtabmap: dict[str, Any] | None = None,
        rtabmap_trajectory: dict[str, Any] | None = None,
        # 하위 호환성 유지용 (무시됨)
        units: object = None,
        unit_split: object = None,
        place_areas: object = None,
    ) -> ImdfFeatureSet:
        """5 FeatureCollection + manifest dict 조립.

        Args:
            scan_id: 스캔 UUID.
            build_job_id: 빌드 job UUID.
            footprint_geojson: adaptive_buffer shapely union → GeoJSON MultiPolygon dict.
            poi_nodes: node_type='poi' map_node 목록.
            poi_labels: poi_mark_id → label (None 허용).
            floor_z0: floor plane z 좌표 (world 미터).
            generated_at: 생성 시각 (manifest.generated_at).
            language: 언어 코드 (기본 "ko-KR").
        """
        scan_id_str = str(scan_id)
        build_job_id_str = str(build_job_id)
        mp_geojson = _ensure_multipolygon(footprint_geojson)
        extents = _extents_from_multipolygon(mp_geojson)

        manifest = self._build_manifest(
            scan_id=scan_id_str,
            build_job_id=build_job_id_str,
            generated_at=generated_at,
            floor_z0=floor_z0,
            extents=extents,
            language=language,
            rectification=rectification,
            build_source=build_source,
            rtabmap=rtabmap,
            rtabmap_trajectory=rtabmap_trajectory,
        )
        level = self._build_level(scan_id_str)
        unit = _feature_collection([])  # unit polygon 완전 제거
        footprint = self._build_footprint(scan_id_str, mp_geojson)
        amenity = self._build_amenity(
            scan_id_str,
            poi_nodes,
            poi_labels,
            floor_z0,
            semantic_amenities,
        )
        anchor = self._build_anchor(scan_id_str, floor_z0)

        return ImdfFeatureSet(
            manifest=manifest,
            level_geojson=level,
            unit_geojson=unit,
            footprint_geojson=footprint,
            amenity_geojson=amenity,
            anchor_geojson=anchor,
        )

    # ── private builders ──────────────────────────────────────────────────────

    def _build_manifest(
        self,
        *,
        scan_id: str,
        build_job_id: str,
        generated_at: datetime,
        floor_z0: float,
        extents: dict[str, float],
        language: str,
        rectification: dict[str, Any] | None,
        build_source: str | None,
        rtabmap: dict[str, Any] | None,
        rtabmap_trajectory: dict[str, Any] | None,
    ) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "format": _FORMAT,
            "version": _VERSION,
            "generated_at": generated_at.isoformat(),
            "scan_id": scan_id,
            "build_job_id": build_job_id,
            "coordinate_system": _COORDINATE_SYSTEM,
            "language": language,
            "feature_files": [
                "level.geojson",
                "unit.geojson",
                "footprint.geojson",
                "amenity.geojson",
                "anchor.geojson",
            ],
            "z_floor_m": floor_z0,
            "extents_m": extents,
        }
        if rectification is not None:
            manifest["rectification"] = rectification
        if build_source is not None:
            manifest["build_source"] = build_source
        if rtabmap is not None:
            manifest["rtabmap"] = rtabmap
        if rtabmap_trajectory is not None:
            manifest["rtabmap_trajectory"] = rtabmap_trajectory
        return manifest

    def _build_level(self, scan_id: str) -> dict[str, Any]:
        return _feature_collection([
            {
                "type": "Feature",
                "id": "level-0",
                "geometry": None,
                "properties": {
                    "feature_type": "level",
                    "scan_id": scan_id,
                    "ordinal": 0,
                    "category": "indoor",
                    "name": {"ko": "1층"},
                    "outdoor": False,
                },
            }
        ])

    def _build_footprint(self, scan_id: str, mp_geojson: dict[str, Any]) -> dict[str, Any]:
        return _feature_collection([
            {
                "type": "Feature",
                "id": "footprint-0",
                "geometry": mp_geojson,
                "properties": {
                    "feature_type": "footprint",
                    "scan_id": scan_id,
                    "category": "ground",
                    "name": None,
                },
            }
        ])

    def _build_amenity(
        self,
        scan_id: str,
        poi_nodes: list[MapNodeRow],
        poi_labels: dict[int, str | None],
        floor_z0: float,
        semantic_amenities: list[SemanticAmenityFeature] | None,
    ) -> dict[str, Any]:
        """POI 노드 → amenity feature. POI 0개면 빈 FeatureCollection."""
        if semantic_amenities is not None:
            return self._build_semantic_amenity(scan_id, semantic_amenities)

        features: list[dict[str, Any]] = []
        for node in poi_nodes:
            poi_mark_id = node.poi_mark_id
            label = poi_labels.get(poi_mark_id) if poi_mark_id is not None else None
            name: dict[str, str | None] | None = {"ko": label} if label else None
            features.append({
                "type": "Feature",
                "id": f"amenity-{poi_mark_id}",
                "geometry": {
                    "type": "Point",
                    "coordinates": [node.x, node.y, node.z],
                },
                "properties": {
                    "feature_type": "amenity",
                    "scan_id": scan_id,
                    "level_id": "level-0",
                    "node_id": str(node.node_id),
                    "poi_mark_id": poi_mark_id,
                    "category": "unspecified",
                    "name": name,
                },
            })
        return _feature_collection(features)

    def _build_semantic_amenity(
        self,
        scan_id: str,
        semantic_amenities: list[SemanticAmenityFeature],
    ) -> dict[str, Any]:
        features: list[dict[str, Any]] = []
        for amenity in semantic_amenities:
            connector_type = amenity.analysis.connector_type
            connector_key = amenity.analysis.connector_key
            name = {"ko": amenity.name} if amenity.name else None
            features.append({
                "type": "Feature",
                "id": amenity.id,
                "geometry": {
                    "type": "Point",
                    "coordinates": list(amenity.point),
                },
                "properties": {
                    "feature_type": "amenity",
                    "scan_id": scan_id,
                    "level_id": "level-0",
                    "node_id": str(amenity.route_node_id),
                    "route_node_id": str(amenity.route_node_id),
                    "poi_mark_id": amenity.poi_mark_id,
                    "category": amenity.category,
                    "name": name,
                    "geometry_role": "route_point",
                    "display_point": list(amenity.display_point),
                    "display_point_role": "map_marker",
                    "display_area_id": amenity.display_area_id,
                    "route_excluded": True,
                    "connector_type": connector_type,
                    "connector_key": connector_key,
                    "analysis": amenity.analysis.as_metadata(),
                },
            })
        return _feature_collection(features)

    def _build_anchor(self, scan_id: str, floor_z0: float) -> dict[str, Any]:
        """scan local origin (0, 0, z0) placeholder."""
        return _feature_collection([
            {
                "type": "Feature",
                "id": "anchor-origin",
                "geometry": {
                    "type": "Point",
                    "coordinates": [0.0, 0.0, floor_z0],
                },
                "properties": {
                    "feature_type": "anchor",
                    "scan_id": scan_id,
                    "level_id": "level-0",
                    "purpose": "scan_origin",
                    "georef": None,
                },
            }
        ])
