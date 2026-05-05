"""POI route node 주변에 지도 표시용 semantic area를 만든다."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from typing import Any
from uuid import UUID

from shapely.geometry import LineString, MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from indoor_server.domain.routing.models import MapEdgeRow, MapNodeRow
from indoor_server.domain.semantic.models import PlaceAreaFeature, SemanticAnalysis

_PLACE_GAP_M = 0.20
_LOCAL_SHIFT_SEQUENCE_M = (
    0.0,
    0.5, -0.5,
    1.0, -1.0,
    1.5, -1.5,
    2.0, -2.0,
    2.5, -2.5,
    3.0, -3.0,
    4.0, -4.0,
)


@dataclass(frozen=True)
class _SideCandidate:
    normal: tuple[float, float]
    side_rank: int
    geometry: BaseGeometry
    score: float


@dataclass(frozen=True)
class _PlaceDraft:
    node: MapNodeRow
    analysis: SemanticAnalysis
    edge_id: str
    tangent: tuple[float, float]
    normal: tuple[float, float]
    side_rank: int
    width_m: float
    depth_m: float
    geometry: BaseGeometry
    score: float
    alternate_candidates: tuple[_SideCandidate, ...] = ()
    collision_adjusted: bool = False


class PlaceAreaBuilder:
    """복도 위 route target에서 옆 공간 polygon을 생성한다."""

    def __init__(
        self,
        *,
        offset_m: float = 1.8,
        width_m: float = 2.6,
        depth_m: float = 1.8,
    ) -> None:
        self._offset_m = offset_m
        self._width_m = width_m
        self._depth_m = depth_m

    def build(
        self,
        *,
        node: MapNodeRow,
        analysis: SemanticAnalysis,
        edges: list[MapEdgeRow],
        node_lookup: dict[str, MapNodeRow],
        footprint_geojson: dict[str, object] | None = None,
        walkway_geojson: dict[str, object] | None = None,
    ) -> PlaceAreaFeature:
        return self.build_many(
            items=[(node, analysis)],
            edges=edges,
            node_lookup=node_lookup,
            footprint_geojson=footprint_geojson,
            walkway_geojson=walkway_geojson,
        )[0]

    def build_many(
        self,
        *,
        items: list[tuple[MapNodeRow, SemanticAnalysis]],
        edges: list[MapEdgeRow],
        node_lookup: dict[str, MapNodeRow],
        footprint_geojson: dict[str, object] | None = None,
        walkway_geojson: dict[str, object] | None = None,
    ) -> list[PlaceAreaFeature]:
        if not items:
            return []

        footprint, walkway = self._semantic_bounds(
            footprint_geojson=footprint_geojson,
            walkway_geojson=walkway_geojson,
        )
        if footprint is None or walkway is None:
            return [
                self._feature_from_geometry(
                    node=node,
                    analysis=analysis,
                    geometry=self._fallback_rect(
                        node=node,
                        tangent=self._nearest_tangent(
                            node=node,
                            edges=edges,
                            node_lookup=node_lookup,
                        ),
                        width_m=self._size_for_category(analysis.category)[0],
                        depth_m=self._size_for_category(analysis.category)[1],
                    ),
                )
                for node, analysis in items
            ]

        drafts = [
            self._build_draft(
                node=node,
                analysis=analysis,
                edges=edges,
                node_lookup=node_lookup,
                footprint=footprint,
                walkway=walkway,
            )
            for node, analysis in items
        ]
        resolved = self._resolve_collisions(
            drafts=sorted(drafts, key=lambda d: int(d.node.poi_mark_id or 0)),
            footprint=footprint,
            walkway=walkway,
        )
        return [
            self._feature_from_geometry(
                node=draft.node,
                analysis=draft.analysis,
                geometry=draft.geometry,
            )
            for draft in sorted(resolved, key=lambda d: int(d.node.poi_mark_id or 0))
        ]

    def _feature_from_geometry(
        self,
        *,
        node: MapNodeRow,
        analysis: SemanticAnalysis,
        geometry: BaseGeometry,
    ) -> PlaceAreaFeature:
        return PlaceAreaFeature(
            id=f"place-poi-{node.poi_mark_id}",
            category=analysis.category,
            name=analysis.name,
            geometry=_as_multipolygon_geojson(geometry),
            entrance_node_id=node.node_id,
            source_poi_mark_id=int(node.poi_mark_id or 0),
        )

    def _build_draft(
        self,
        *,
        node: MapNodeRow,
        analysis: SemanticAnalysis,
        edges: list[MapEdgeRow],
        node_lookup: dict[str, MapNodeRow],
        footprint: BaseGeometry,
        walkway: BaseGeometry,
    ) -> _PlaceDraft:
        edge_id, tangent = self._nearest_edge_context(
            node=node,
            edges=edges,
            node_lookup=node_lookup,
        )
        width_m, depth_m = self._size_for_category(analysis.category)
        candidates = self._ranked_side_candidates(
            node=node,
            tangent=tangent,
            width_m=width_m,
            depth_m=depth_m,
            footprint=footprint,
            walkway=walkway,
        )
        if not candidates:
            geometry = self._fallback_rect(
                node=node,
                tangent=tangent,
                width_m=width_m,
                depth_m=depth_m,
            )
            return _PlaceDraft(
                node=node,
                analysis=analysis,
                edge_id=edge_id,
                tangent=tangent,
                normal=(-tangent[1], tangent[0]),
                side_rank=0,
                width_m=width_m,
                depth_m=depth_m,
                geometry=geometry,
                score=0.0,
            )
        best = candidates[0]
        return _PlaceDraft(
            node=node,
            analysis=analysis,
            edge_id=edge_id,
            tangent=tangent,
            normal=best.normal,
            side_rank=best.side_rank,
            width_m=width_m,
            depth_m=depth_m,
            geometry=best.geometry,
            score=best.score,
            alternate_candidates=tuple(candidates[1:]),
        )

    def _nearest_tangent(
        self,
        *,
        node: MapNodeRow,
        edges: list[MapEdgeRow],
        node_lookup: dict[str, MapNodeRow],
    ) -> tuple[float, float]:
        return self._nearest_edge_context(
            node=node,
            edges=edges,
            node_lookup=node_lookup,
        )[1]

    def _nearest_edge_context(
        self,
        *,
        node: MapNodeRow,
        edges: list[MapEdgeRow],
        node_lookup: dict[str, MapNodeRow],
    ) -> tuple[str, tuple[float, float]]:
        best: tuple[str, tuple[float, float]] | None = None
        best_d = float("inf")
        for edge in sorted(edges, key=lambda e: str(e.edge_id)):
            endpoints = [
                node_lookup.get(str(edge.from_node_id)),
                node_lookup.get(str(edge.to_node_id)),
            ]
            if endpoints[0] is None or endpoints[1] is None:
                continue
            a = endpoints[0]
            b = endpoints[1]
            assert a is not None and b is not None
            d = min(hypot(node.x - a.x, node.y - a.y), hypot(node.x - b.x, node.y - b.y))
            if d < best_d:
                dx = b.x - a.x
                dy = b.y - a.y
                norm = hypot(dx, dy)
                if norm > 1e-6:
                    tangent = _normalize_tangent(dx / norm, dy / norm)
                    best = (str(edge.edge_id), tangent)
                    best_d = d
        return best or ("", (1.0, 0.0))

    def _wall_side_geometry(
        self,
        *,
        node: MapNodeRow,
        tangent: tuple[float, float],
        width_m: float,
        depth_m: float,
        footprint_geojson: dict[str, object] | None,
        walkway_geojson: dict[str, object] | None,
    ) -> BaseGeometry:
        if footprint_geojson is None or walkway_geojson is None:
            return self._fallback_rect(node=node, tangent=tangent, width_m=width_m, depth_m=depth_m)

        footprint, walkway = self._semantic_bounds(
            footprint_geojson=footprint_geojson,
            walkway_geojson=walkway_geojson,
        )
        if footprint is None or walkway is None:
            return self._fallback_rect(node=node, tangent=tangent, width_m=width_m, depth_m=depth_m)

        candidates = self._ranked_side_candidates(
            node=node,
            tangent=tangent,
            width_m=width_m,
            depth_m=depth_m,
            footprint=footprint,
            walkway=walkway,
        )
        min_area = max(0.6, width_m * depth_m * 0.18)
        if (
            candidates
            and not candidates[0].geometry.is_empty
            and candidates[0].geometry.area >= min_area
        ):
            return candidates[0].geometry

        fallback = self._fallback_rect(
            node=node,
            tangent=tangent,
            width_m=width_m,
            depth_m=depth_m,
        )
        clipped_fallback = _polygonal_part(
            fallback.intersection(footprint).difference(walkway.buffer(0.03)).buffer(0)
        )
        return clipped_fallback if not clipped_fallback.is_empty else fallback

    def _semantic_bounds(
        self,
        *,
        footprint_geojson: dict[str, object] | None,
        walkway_geojson: dict[str, object] | None,
    ) -> tuple[BaseGeometry | None, BaseGeometry | None]:
        if footprint_geojson is None or walkway_geojson is None:
            return None, None
        footprint = shape(footprint_geojson).buffer(0)
        walkway = shape(walkway_geojson).buffer(0)
        if footprint.is_empty or walkway.is_empty:
            return None, None
        return footprint, walkway

    def _ranked_side_candidates(
        self,
        *,
        node: MapNodeRow,
        tangent: tuple[float, float],
        width_m: float,
        depth_m: float,
        footprint: BaseGeometry,
        walkway: BaseGeometry,
    ) -> list[_SideCandidate]:
        base_nx, base_ny = -tangent[1], tangent[0]
        scored: list[_SideCandidate] = []
        for side_rank, side in enumerate((1.0, -1.0)):
            normal = (base_nx * side, base_ny * side)
            rect = self._side_rect(
                node=node,
                tangent=tangent,
                normal=normal,
                width_m=width_m,
                depth_m=depth_m,
                walkway=walkway,
            )
            candidate, score = self._score_candidate(
                rect=rect,
                footprint=footprint,
                walkway=walkway,
            )
            scored.append(
                _SideCandidate(
                    normal=normal,
                    side_rank=side_rank,
                    geometry=candidate,
                    score=score,
                )
            )
        scored.sort(key=lambda c: (-c.score, c.side_rank))
        min_area = max(0.6, width_m * depth_m * 0.18)
        valid = [
            candidate for candidate in scored
            if not candidate.geometry.is_empty and candidate.geometry.area >= min_area
        ]
        return valid or scored

    def _score_candidate(
        self,
        *,
        rect: BaseGeometry,
        footprint: BaseGeometry,
        walkway: BaseGeometry,
    ) -> tuple[BaseGeometry, float]:
        clipped = rect.intersection(footprint)
        overlap_area = clipped.intersection(walkway).area
        candidate = _polygonal_part(clipped.difference(walkway.buffer(0.03)).buffer(0))
        if candidate.is_empty:
            return candidate, -1_000_000.0 - overlap_area
        exterior_distance = candidate.centroid.distance(footprint.boundary)
        boundary_contact = candidate.boundary.intersection(
            footprint.boundary.buffer(0.15)
        ).length
        score = (
            candidate.area * 5.0
            + boundary_contact * 2.0
            - exterior_distance * 0.25
            - overlap_area * 20.0
        )
        return candidate, score

    def _resolve_collisions(
        self,
        *,
        drafts: list[_PlaceDraft],
        footprint: BaseGeometry,
        walkway: BaseGeometry,
    ) -> list[_PlaceDraft]:
        grouped: dict[tuple[str, float, float], list[_PlaceDraft]] = {}
        for draft in drafts:
            grouped.setdefault(self._group_key(draft), []).append(draft)

        resolved_by_id: dict[UUID, _PlaceDraft] = {}
        accepted: list[BaseGeometry] = []
        for key in sorted(grouped):
            group = sorted(grouped[key], key=self._draft_sort_key)
            for cluster in self._cluster_group(group):
                if len(cluster) == 1:
                    draft = cluster[0]
                    resolved_by_id[draft.node.node_id] = draft
                    accepted.append(_polygonal_part(draft.geometry.buffer(0)))
                    continue
                for draft, target_s in self._packed_targets(cluster):
                    candidate = self._candidate_at_projection(
                        draft=draft,
                        target_s=target_s,
                        width_m=draft.width_m,
                        normal=draft.normal,
                        footprint=footprint,
                        walkway=walkway,
                    )
                    adjusted = abs(target_s - self._projection_s(draft)) > 1e-6
                    if not self._candidate_accepted(
                        candidate=candidate,
                        accepted=accepted,
                        walkway=walkway,
                        min_area=self._min_area(draft.width_m, draft.depth_m),
                    ):
                        candidate, adjusted = self._resolve_single(
                            draft=draft,
                            accepted=accepted,
                            footprint=footprint,
                            walkway=walkway,
                        )
                    resolved = replace(
                        draft,
                        geometry=_polygonal_part(candidate.buffer(0)),
                        collision_adjusted=adjusted,
                    )
                    resolved_by_id[resolved.node.node_id] = resolved
                    accepted.append(resolved.geometry)
        return [resolved_by_id[draft.node.node_id] for draft in drafts]

    def _resolve_single(
        self,
        *,
        draft: _PlaceDraft,
        accepted: list[BaseGeometry],
        footprint: BaseGeometry,
        walkway: BaseGeometry,
    ) -> tuple[BaseGeometry, bool]:
        candidate = self._search_offsets(
            draft=draft,
            accepted=accepted,
            footprint=footprint,
            walkway=walkway,
            normal=draft.normal,
            width_m=draft.width_m,
        )
        if candidate is not None:
            return candidate, True

        for side in draft.alternate_candidates:
            candidate = self._search_offsets(
                draft=draft,
                accepted=accepted,
                footprint=footprint,
                walkway=walkway,
                normal=side.normal,
                width_m=draft.width_m,
            )
            if candidate is not None:
                return candidate, True

        for width_scale in (0.85, 0.70):
            candidate = self._search_offsets(
                draft=draft,
                accepted=accepted,
                footprint=footprint,
                walkway=walkway,
                normal=draft.normal,
                width_m=draft.width_m * width_scale,
            )
            if candidate is not None:
                return candidate, True

        accepted_union = unary_union(accepted).buffer(0.03) if accepted else Polygon()
        candidate = _polygonal_part(draft.geometry.difference(accepted_union).buffer(0))
        if not candidate.is_empty and candidate.area >= 0.6:
            return candidate, True
        return draft.geometry, False

    def _search_offsets(
        self,
        *,
        draft: _PlaceDraft,
        accepted: list[BaseGeometry],
        footprint: BaseGeometry,
        walkway: BaseGeometry,
        normal: tuple[float, float],
        width_m: float,
    ) -> BaseGeometry | None:
        original_s = self._projection_s(draft)
        min_area = self._min_area(width_m, draft.depth_m)
        for shift_m in _LOCAL_SHIFT_SEQUENCE_M:
            candidate = self._candidate_at_projection(
                draft=draft,
                target_s=original_s + shift_m,
                width_m=width_m,
                normal=normal,
                footprint=footprint,
                walkway=walkway,
            )
            if self._candidate_accepted(
                candidate=candidate,
                accepted=accepted,
                walkway=walkway,
                min_area=min_area,
            ):
                return candidate
        return None

    def _candidate_at_projection(
        self,
        *,
        draft: _PlaceDraft,
        target_s: float,
        width_m: float,
        normal: tuple[float, float],
        footprint: BaseGeometry,
        walkway: BaseGeometry,
    ) -> BaseGeometry:
        original_s = self._projection_s(draft)
        tangent_delta = target_s - original_s
        tx, ty = draft.tangent
        origin = (
            draft.node.x + tx * tangent_delta,
            draft.node.y + ty * tangent_delta,
        )
        rect = self._side_rect_at(
            origin=origin,
            tangent=draft.tangent,
            normal=normal,
            width_m=width_m,
            depth_m=draft.depth_m,
            walkway=walkway,
        )
        return _polygonal_part(
            rect.intersection(footprint).difference(walkway.buffer(0.03)).buffer(0)
        )

    def _candidate_accepted(
        self,
        *,
        candidate: BaseGeometry,
        accepted: list[BaseGeometry],
        walkway: BaseGeometry,
        min_area: float,
    ) -> bool:
        if candidate.is_empty or candidate.area < min_area:
            return False
        walkway_overlap_ratio = candidate.intersection(walkway).area / candidate.area
        if walkway_overlap_ratio >= 0.01:
            return False
        for placed in accepted:
            overlap_area = candidate.intersection(placed).area
            overlap_ratio = overlap_area / min(candidate.area, placed.area)
            if overlap_area > 0.05 or overlap_ratio > 0.02:
                return False
        return True

    def _packed_targets(
        self,
        cluster: list[_PlaceDraft],
    ) -> list[tuple[_PlaceDraft, float]]:
        total_span = sum(draft.width_m for draft in cluster) + _PLACE_GAP_M * (len(cluster) - 1)
        cluster_center = sum(self._projection_s(draft) for draft in cluster) / len(cluster)
        cursor = cluster_center - total_span / 2.0
        targets: list[tuple[_PlaceDraft, float]] = []
        for draft in cluster:
            target_s = cursor + draft.width_m / 2.0
            targets.append((draft, target_s))
            cursor += draft.width_m + _PLACE_GAP_M
        return targets

    def _cluster_group(self, group: list[_PlaceDraft]) -> list[list[_PlaceDraft]]:
        clusters: list[list[_PlaceDraft]] = []
        current: list[_PlaceDraft] = []
        current_end = float("-inf")
        for draft in group:
            projection_s = self._projection_s(draft)
            start = projection_s - draft.width_m / 2.0
            end = projection_s + draft.width_m / 2.0
            if not current or start <= current_end + _PLACE_GAP_M:
                current.append(draft)
                current_end = max(current_end, end)
                continue
            clusters.append(current)
            current = [draft]
            current_end = end
        if current:
            clusters.append(current)
        return clusters

    def _group_key(self, draft: _PlaceDraft) -> tuple[str, float, float]:
        return (draft.edge_id, round(draft.normal[0], 3), round(draft.normal[1], 3))

    def _draft_sort_key(self, draft: _PlaceDraft) -> tuple[float, int, str]:
        return (
            self._projection_s(draft),
            int(draft.node.poi_mark_id or 0),
            str(draft.node.node_id),
        )

    def _projection_s(self, draft: _PlaceDraft) -> float:
        tx, ty = draft.tangent
        return draft.node.x * tx + draft.node.y * ty

    def _min_area(self, width_m: float, depth_m: float) -> float:
        return max(0.6, width_m * depth_m * 0.18)

    def _side_rect(
        self,
        *,
        node: MapNodeRow,
        tangent: tuple[float, float],
        normal: tuple[float, float],
        width_m: float,
        depth_m: float,
        walkway: BaseGeometry,
    ) -> Polygon:
        return self._side_rect_at(
            origin=(node.x, node.y),
            tangent=tangent,
            normal=normal,
            width_m=width_m,
            depth_m=depth_m,
            walkway=walkway,
        )

    def _side_rect_at(
        self,
        *,
        origin: tuple[float, float],
        tangent: tuple[float, float],
        normal: tuple[float, float],
        width_m: float,
        depth_m: float,
        walkway: BaseGeometry,
    ) -> Polygon:
        tx, ty = tangent
        nx, ny = normal
        half_w = width_m / 2.0
        inner_offset = self._walkway_exit_offset(
            origin=origin,
            normal=normal,
            walkway=walkway,
        ) + 0.05
        outer_offset = inner_offset + depth_m
        return Polygon([
            (
                origin[0] - tx * half_w + nx * inner_offset,
                origin[1] - ty * half_w + ny * inner_offset,
            ),
            (
                origin[0] + tx * half_w + nx * inner_offset,
                origin[1] + ty * half_w + ny * inner_offset,
            ),
            (
                origin[0] + tx * half_w + nx * outer_offset,
                origin[1] + ty * half_w + ny * outer_offset,
            ),
            (
                origin[0] - tx * half_w + nx * outer_offset,
                origin[1] - ty * half_w + ny * outer_offset,
            ),
        ])

    def _walkway_exit_offset(
        self,
        *,
        origin: tuple[float, float],
        normal: tuple[float, float],
        walkway: BaseGeometry,
    ) -> float:
        nx, ny = normal
        probe_len = max(20.0, self._offset_m + self._depth_m + 10.0)
        ray = LineString([origin, (origin[0] + nx * probe_len, origin[1] + ny * probe_len)])
        hit = ray.intersection(walkway.boundary)
        distances = [
            distance for distance in _projected_distances(hit, origin=origin, normal=normal)
            if distance >= -1e-6
        ]
        if distances:
            return max(0.0, min(distances))
        return max(0.05, self._offset_m - self._depth_m / 2.0)

    def _fallback_rect(
        self,
        *,
        node: MapNodeRow,
        tangent: tuple[float, float],
        width_m: float,
        depth_m: float,
    ) -> Polygon:
        tx, ty = tangent
        nx, ny = -tangent[1], tangent[0]
        center = (
            node.x + nx * self._offset_m,
            node.y + ny * self._offset_m,
        )
        half_w = width_m / 2.0
        half_d = depth_m / 2.0
        return Polygon([
            (center[0] - tx * half_w - nx * half_d, center[1] - ty * half_w - ny * half_d),
            (center[0] + tx * half_w - nx * half_d, center[1] + ty * half_w - ny * half_d),
            (center[0] + tx * half_w + nx * half_d, center[1] + ty * half_w + ny * half_d),
            (center[0] - tx * half_w + nx * half_d, center[1] - ty * half_w + ny * half_d),
        ])

    def _size_for_category(self, category: str) -> tuple[float, float]:
        sizes = {
            "room": (3.0, 2.0),
            "lab": (3.4, 2.2),
            "office": (2.6, 1.8),
            "restroom": (2.4, 1.8),
            "stairs": (2.2, 2.2),
            "elevator": (1.8, 1.8),
            "entrance": (2.4, 1.4),
            "destination": (2.6, 1.8),
        }
        return sizes.get(category, (self._width_m, self._depth_m))


def _normalize_tangent(tx: float, ty: float) -> tuple[float, float]:
    if tx < 0.0 or (abs(tx) < 1e-9 and ty < 0.0):
        return (-tx, -ty)
    return (tx, ty)


def _projected_distances(
    geom: BaseGeometry,
    *,
    origin: tuple[float, float],
    normal: tuple[float, float],
) -> list[float]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [_project(geom.x, geom.y, origin=origin, normal=normal)]
    if geom.geom_type in {"LineString", "LinearRing"}:
        return [
            _project(float(x), float(y), origin=origin, normal=normal)
            for x, y in geom.coords
        ]
    if hasattr(geom, "geoms"):
        distances: list[float] = []
        for child in geom.geoms:
            distances.extend(_projected_distances(child, origin=origin, normal=normal))
        return distances
    return []


def _project(
    x: float,
    y: float,
    *,
    origin: tuple[float, float],
    normal: tuple[float, float],
) -> float:
    return (x - origin[0]) * normal[0] + (y - origin[1]) * normal[1]


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


def _as_multipolygon_geojson(geom: BaseGeometry) -> dict[str, Any]:
    geom = _polygonal_part(geom.buffer(0))
    geo = mapping(geom)
    if geo["type"] == "Polygon":
        return {"type": "MultiPolygon", "coordinates": [geo["coordinates"]]}
    if geo["type"] == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": geo["coordinates"]}
    return {"type": "MultiPolygon", "coordinates": []}
