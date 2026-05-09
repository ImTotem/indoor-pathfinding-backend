"""사용자 명시 corridor 노드+엣지 + POI/interfloor perpendicular drop attach 로
nav graph 빌드 (sprint82 격자 대체).

흐름:
  1. corridor 노드 + 엣지 (사용자 명시) 그대로 graph 시작점
  2. POI/interfloor/junction(branch_mark.corridor) 노드를 순회:
     - 각 노드를 가장 가까운 corridor edge 에 perpendicular drop
     - 발 위치 (foot) 가 edge 끝점 epsilon 이내면 끝점을 attach 로 사용 (split 안 함)
     - 그 외에는 edge 를 foot 위치로 split + spur edge 추가
  3. 결과 nav_nodes, nav_edges 반환

전제:
  - 폴리곤은 본 호출 전에 floor_polygon_v2.build_floor_polygon() 으로 미리 빌드됨
  - corridor 노드는 모두 같은 평면 (z 일정) 가정 (그래프 routing 은 2D)
  - corridor edge 는 양 끝 둘 다 corridor 노드 (사용자 명시)

미구현 (클라 데이터 schema 확정 후):
  - 명시적 POI↔corridor edge 가 클라에서 들어오면 자동 nearest 우회 (override)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable
from uuid import UUID, uuid4

from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)

# foot 이 edge 끝점 epsilon 이내면 split 안 하고 끝점 reuse
ATTACH_EPSILON_M = 0.05


@dataclass(frozen=True)
class V2Node:
    node_id: str           # UUID 문자열
    kind: str              # 'corridor' | 'poi' | 'junction' | 'connector' | 'attach'
    x: float
    y: float
    z: float = 0.0
    label: str | None = None
    source_ref: dict | None = None


@dataclass(frozen=True)
class V2Edge:
    edge_id: str           # UUID 문자열
    from_node_id: str
    to_node_id: str
    kind: str              # 'corridor' | 'poi_spur'
    length_m: float


@dataclass(frozen=True)
class AttachResult:
    foot_node: V2Node | None     # 새로 추가된 foot 노드 (None 이면 끝점 reuse)
    split_edges: list[V2Edge]    # foot 으로 분할된 두 edge (foot_node 없으면 빈 list)
    spur_edge: V2Edge            # target ↔ foot
    removed_edge_id: str | None  # split 으로 제거된 원본 edge_id (foot_node 있으면 not None)


def _segment_length(a: V2Node, b: V2Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def attach_to_nearest_edge(
    target: V2Node,
    corridor_nodes_by_id: dict[str, V2Node],
    corridor_edges: list[V2Edge],
    *,
    epsilon_m: float = ATTACH_EPSILON_M,
) -> AttachResult | None:
    """target 노드를 가장 가까운 corridor edge 에 perpendicular drop 으로 attach.

    Returns:
        AttachResult — foot 노드(또는 끝점 reuse), split edges, spur edge 정보.
        corridor_edges 가 비었으면 None.
    """
    if not corridor_edges:
        return None

    best_edge: V2Edge | None = None
    best_t: float = 0.0
    best_dist: float = float("inf")
    best_foot: tuple[float, float] | None = None
    pt = Point(target.x, target.y)

    for edge in corridor_edges:
        a = corridor_nodes_by_id.get(edge.from_node_id)
        b = corridor_nodes_by_id.get(edge.to_node_id)
        if a is None or b is None:
            continue
        line = LineString([(a.x, a.y), (b.x, b.y)])
        if line.length < 1e-9:
            continue
        t_norm = line.project(pt, normalized=True)
        t_norm = max(0.0, min(1.0, t_norm))
        foot_pt = line.interpolate(t_norm, normalized=True)
        dist = pt.distance(foot_pt)
        if dist < best_dist:
            best_dist = dist
            best_edge = edge
            best_t = t_norm
            best_foot = (foot_pt.x, foot_pt.y)

    if best_edge is None or best_foot is None:
        return None

    a = corridor_nodes_by_id[best_edge.from_node_id]
    b = corridor_nodes_by_id[best_edge.to_node_id]
    edge_len = _segment_length(a, b)
    epsilon_t = epsilon_m / edge_len if edge_len > 1e-9 else 1.0

    # 끝점 reuse 케이스 — split 안 함
    if best_t <= epsilon_t:
        attach_node_id = best_edge.from_node_id
        spur_len = _segment_length(target, a)
        spur = V2Edge(
            edge_id=str(uuid4()),
            from_node_id=target.node_id,
            to_node_id=attach_node_id,
            kind="poi_spur",
            length_m=spur_len,
        )
        return AttachResult(foot_node=None, split_edges=[], spur_edge=spur, removed_edge_id=None)
    if best_t >= 1.0 - epsilon_t:
        attach_node_id = best_edge.to_node_id
        spur_len = _segment_length(target, b)
        spur = V2Edge(
            edge_id=str(uuid4()),
            from_node_id=target.node_id,
            to_node_id=attach_node_id,
            kind="poi_spur",
            length_m=spur_len,
        )
        return AttachResult(foot_node=None, split_edges=[], spur_edge=spur, removed_edge_id=None)

    # split 케이스 — 새 foot 노드 + edge 분할
    fx, fy = best_foot
    foot_z = (a.z + b.z) / 2.0
    foot_node = V2Node(
        node_id=str(uuid4()),
        kind="attach",
        x=float(fx), y=float(fy), z=float(foot_z),
        source_ref={
            "role": "poi_attach_foot",
            "split_from_edge_id": best_edge.edge_id,
            "perpendicular_dist_m": float(best_dist),
        },
    )
    seg1_len = _segment_length(a, foot_node)
    seg2_len = _segment_length(foot_node, b)
    split_edges = [
        V2Edge(
            edge_id=str(uuid4()),
            from_node_id=best_edge.from_node_id,
            to_node_id=foot_node.node_id,
            kind=best_edge.kind,
            length_m=seg1_len,
        ),
        V2Edge(
            edge_id=str(uuid4()),
            from_node_id=foot_node.node_id,
            to_node_id=best_edge.to_node_id,
            kind=best_edge.kind,
            length_m=seg2_len,
        ),
    ]
    spur = V2Edge(
        edge_id=str(uuid4()),
        from_node_id=target.node_id,
        to_node_id=foot_node.node_id,
        kind="poi_spur",
        length_m=float(best_dist),
    )
    return AttachResult(
        foot_node=foot_node,
        split_edges=split_edges,
        spur_edge=spur,
        removed_edge_id=best_edge.edge_id,
    )


def build_nav_graph_v2(
    *,
    corridor_nodes: Iterable[V2Node],
    corridor_edges: Iterable[V2Edge],
    attach_targets: Iterable[V2Node],
) -> tuple[list[V2Node], list[V2Edge]]:
    """corridor backbone 에 attach_targets (POI/interfloor/junction 등) 를 perpendicular drop.

    Args:
        corridor_nodes: 사용자 명시 corridor 노드 (kind='corridor')
        corridor_edges: 사용자 명시 corridor↔corridor edges (kind='corridor')
        attach_targets: backbone 에 붙일 노드들 (POI / interfloor / junction 등)

    Returns:
        (nav_nodes, nav_edges) — graph 저장 직전 형태
    """
    nodes_by_id: dict[str, V2Node] = {n.node_id: n for n in corridor_nodes}
    edges_by_id: dict[str, V2Edge] = {e.edge_id: e for e in corridor_edges}

    targets = list(attach_targets)
    if not edges_by_id:
        # corridor edge 없음 — attach 불가, target 만 isolated 로 추가
        if targets:
            logger.warning(
                "nav_graph_v2: corridor edges 없음 — %d 개 attach target 모두 isolated",
                len(targets),
            )
        for t in targets:
            nodes_by_id[t.node_id] = t
        return list(nodes_by_id.values()), list(edges_by_id.values())

    for target in targets:
        # target 자체를 nodes 에 추가
        nodes_by_id[target.node_id] = target

        active_edges = list(edges_by_id.values())
        # corridor edge 만 후보 (다른 spur 으로 attach 안 함)
        corridor_only = [e for e in active_edges if e.kind == "corridor"]
        result = attach_to_nearest_edge(target, nodes_by_id, corridor_only)
        if result is None:
            logger.warning(
                "nav_graph_v2: target %s attach 실패 (가까운 edge 없음)", target.node_id
            )
            continue

        if result.foot_node is not None:
            nodes_by_id[result.foot_node.node_id] = result.foot_node
        if result.removed_edge_id is not None:
            edges_by_id.pop(result.removed_edge_id, None)
            for e in result.split_edges:
                edges_by_id[e.edge_id] = e
        edges_by_id[result.spur_edge.edge_id] = result.spur_edge

    return list(nodes_by_id.values()), list(edges_by_id.values())
