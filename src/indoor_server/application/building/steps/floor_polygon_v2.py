"""사용자 명시 노드+엣지 + corner cycle 로 floor polygon 생성 (sprint83 후속).

입력 데이터 모델 (클라가 명시적으로 보냄):
  nodes        : list of (node_id, kind: 'corridor'|'corner', x, y, width_m?, mark_session_id?)
  edges        : list of (edge_id, from_node_id, to_node_id, edge_kind: 'corridor'|'corner')

알고리즘:
  0. 정책: corridor 노드의 width_m=null = polygon 에서 제외 (route 그래프 전용 노드).
       양 끝 둘 중 하나라도 width null 이면 그 edge 도 polygon 빌드에서 제외.
  1. width_m 있는 corridor 만 추려 connected component 별 width 결정.
       (한 component = 같은 width 정책. 다른 width 섞이면 정책 위반 — log error
        후 dominant 값 사용)
  2. corner edges 로 cycle 검출 → 각 mark_session 의 polygon 생성 (room)
  3. corridor edges 를 buffer:
       - 완전히 room 안: skip
       - room 과 교차: outside segment 만 buffer
       - 완전히 밖: buffer 적용
     buffer style: cap_style=flat, join_style=mitre (직각 corner)
  4. 결과를 GeoJSON FeatureCollection 으로 출력
       - per-room Feature
       - per-corridor (buffered segment) Feature
       - 최종 union (floor_union) Feature
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
from shapely.geometry import LineString, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

DEFAULT_CORRIDOR_WIDTH_M = 1.5


@dataclass(frozen=True)
class Node:
    node_id: str
    kind: str  # 'corridor' | 'corner'
    x: float
    y: float
    width_m: float | None = None
    mark_session_id: str | None = None


@dataclass(frozen=True)
class Edge:
    edge_id: str
    from_node_id: str
    to_node_id: str
    kind: str  # 'corridor' | 'corner'


def _assign_component_width(nodes_in_comp: list[Node]) -> float:
    """Component 안의 모든 노드는 width_m not-null 보장됨 (filter 후 호출).
    같은 component 안에 다른 width 섞이면 정책 위반 — log error + dominant 값."""
    widths = {n.width_m for n in nodes_in_comp if n.width_m is not None and n.width_m > 0}
    if len(widths) == 1:
        return next(iter(widths))
    logger.error(
        "polygon_builder: corridor component has multiple widths %s — using max (정책 위반)",
        sorted(widths),
    )
    return max(widths)


def _extract_corner_cycle(
    corner_nodes: list[Node],
    corner_edges: list[Edge],
) -> list[Node]:
    """corner edge cycle 따라 vertex order 추출. cycle 없으면 input order 유지."""
    if len(corner_nodes) < 3:
        return corner_nodes

    g = nx.Graph()
    node_by_id = {n.node_id: n for n in corner_nodes}
    for e in corner_edges:
        if e.from_node_id in node_by_id and e.to_node_id in node_by_id:
            g.add_edge(e.from_node_id, e.to_node_id)

    if g.number_of_edges() == 0:
        return corner_nodes
    try:
        cycle = nx.find_cycle(g, source=corner_nodes[0].node_id)
        ordered_ids: list[str] = []
        for u, v in cycle:
            if not ordered_ids:
                ordered_ids.append(u)
            ordered_ids.append(v)
        # 마지막 원소가 첫 원소와 같으면 제거 (closed loop 표시)
        if ordered_ids and ordered_ids[-1] == ordered_ids[0]:
            ordered_ids.pop()
        return [node_by_id[nid] for nid in ordered_ids]
    except nx.NetworkXNoCycle:
        # 닫힌 cycle 못 찾으면 BFS 순서 (Hamiltonian path 가정)
        start = corner_nodes[0].node_id
        order = list(nx.dfs_preorder_nodes(g, source=start))
        return [node_by_id[nid] for nid in order]


def _geojson_geometry(geom: BaseGeometry) -> dict:
    return mapping(geom)


def build_floor_polygon(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    *,
    floor_id: str | None = None,
) -> dict:
    """사용자 명시 노드+엣지 + corner cycle 로 floor polygon FeatureCollection 생성.

    width_m=null corridor 노드는 polygon 에서 제외 (route 전용 노드).
    """
    nodes_list = list(nodes)
    edges_list = list(edges)
    node_by_id = {n.node_id: n for n in nodes_list}

    corridor_nodes = [n for n in nodes_list if n.kind == "corridor"]
    corner_nodes = [n for n in nodes_list if n.kind == "corner"]
    corridor_edges = [e for e in edges_list if e.kind == "corridor"]
    corner_edges = [e for e in edges_list if e.kind == "corner"]

    features: list[dict] = []

    # 1. Room polygons — mark_session 별 corner cycle
    room_polys: list[Polygon] = []
    sessions: dict[str, list[Node]] = {}
    for n in corner_nodes:
        sid = n.mark_session_id or "_unsessioned"
        sessions.setdefault(sid, []).append(n)
    for session_id, session_corners in sessions.items():
        if len(session_corners) < 3:
            logger.warning(
                "polygon_builder: corner session %s 는 vertex %d 개 — polygon 안 만듦",
                session_id, len(session_corners),
            )
            continue
        cycle_nodes = _extract_corner_cycle(session_corners, corner_edges)
        coords = [(n.x, n.y) for n in cycle_nodes]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)  # self-intersection 자동 수정
        except Exception as exc:
            logger.warning("polygon_builder: room session %s polygon 생성 실패: %s", session_id, exc)
            continue
        room_polys.append(poly)
        features.append({
            "type": "Feature",
            "geometry": _geojson_geometry(poly),
            "properties": {
                "kind": "room",
                "mark_session_id": session_id,
                "node_ids": [n.node_id for n in cycle_nodes],
                "vertex_count": len(cycle_nodes),
            },
        })

    rooms_union: BaseGeometry | None = unary_union(room_polys) if room_polys else None

    # 2. Corridor connected components → component-wide width
    # 정책: width_m=null 인 corridor 노드는 polygon 빌드에서 제외 (route 전용 노드).
    polygon_eligible_ids = {
        n.node_id for n in corridor_nodes
        if n.width_m is not None and n.width_m > 0
    }
    excluded_count = len(corridor_nodes) - len(polygon_eligible_ids)
    if excluded_count:
        logger.info(
            "polygon_builder: %d corridor 노드는 width_m=null 이라 polygon 제외 (route 전용)",
            excluded_count,
        )
    cg = nx.Graph()
    for e in corridor_edges:
        # 양 끝 둘 다 polygon-eligible 일 때만 edge 추가
        if e.from_node_id in polygon_eligible_ids and e.to_node_id in polygon_eligible_ids:
            cg.add_edge(e.from_node_id, e.to_node_id)
    components = list(nx.connected_components(cg))

    for comp_idx, comp_node_ids in enumerate(components):
        comp_nodes = [node_by_id[nid] for nid in comp_node_ids]
        comp_width = _assign_component_width(comp_nodes)
        comp_edges = [
            e for e in corridor_edges
            if e.from_node_id in comp_node_ids and e.to_node_id in comp_node_ids
        ]
        for e in comp_edges:
            a = node_by_id[e.from_node_id]
            b = node_by_id[e.to_node_id]
            line = LineString([(a.x, a.y), (b.x, b.y)])
            # rooms 와 관계 분류
            outside_part: BaseGeometry = line
            if rooms_union is not None:
                if rooms_union.contains(line):
                    continue  # 완전히 room 안 — room polygon 이 cover
                if rooms_union.intersects(line):
                    outside_part = line.difference(rooms_union)
                    if outside_part.is_empty:
                        continue
            # outside_part 가 LineString or MultiLineString
            segs = (
                list(outside_part.geoms)
                if hasattr(outside_part, "geoms")
                else [outside_part]
            )
            for seg in segs:
                if seg.is_empty or seg.length < 1e-6:
                    continue
                buffered = seg.buffer(
                    comp_width / 2,
                    cap_style="flat",
                    join_style="mitre",
                    mitre_limit=2.0,
                )
                features.append({
                    "type": "Feature",
                    "geometry": _geojson_geometry(buffered),
                    "properties": {
                        "kind": "corridor",
                        "edge_id": e.edge_id,
                        "from_node_id": e.from_node_id,
                        "to_node_id": e.to_node_id,
                        "width_m": comp_width,
                        "component_id": comp_idx,
                    },
                })

    # 3. Final floor_union (전체 polygon 합집합)
    individual_geoms = [
        # rooms
        *room_polys,
        # corridors (buffered geometries)
        *[
            f["geometry"]
            for f in features
            if f["properties"].get("kind") == "corridor"
        ],
    ]
    # geometry dict 일 수도 — shapely.shape 로 다시 변환
    from shapely.geometry import shape as _shape
    union_inputs: list[BaseGeometry] = []
    for g in individual_geoms:
        if isinstance(g, BaseGeometry):
            union_inputs.append(g)
        else:
            union_inputs.append(_shape(g))
    if union_inputs:
        floor_union = unary_union(union_inputs)
        features.append({
            "type": "Feature",
            "geometry": _geojson_geometry(floor_union),
            "properties": {
                "kind": "floor_union",
                "rooms_count": len(room_polys),
                "corridors_count": sum(
                    1 for f in features if f["properties"].get("kind") == "corridor"
                ),
                "is_union": True,
            },
        })

    fc = {
        "type": "FeatureCollection",
        "features": features,
    }
    if floor_id is not None:
        fc["floor_id"] = floor_id
    return fc
