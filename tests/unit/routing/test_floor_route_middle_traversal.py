"""Sprint 78 A-8/S10: floor route middle traversal 금지 unit tests.

corridor-only Dijkstra 알고리즘이 POI/passage 노드를 중간 경유점으로 사용하지
않는지 검증한다. networkx 기반 직접 테스트 (DB 의존 없음).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import networkx as nx
import pytest


# ── 최소 stub ─────────────────────────────────────────────────────────────────


def _is_leaf(node_type: str | None, source_ref: dict | None) -> bool:
    """FloorRouteService._is_leaf_node_type 인라인 복사 (stub)."""
    if node_type and node_type.upper() in ("POI",):
        return True
    if isinstance(source_ref, dict) and source_ref.get("role") == "vertical_connector_stop":
        return True
    return False


def _run_route(
    G: nx.Graph,
    from_id: str,
    to_id: str,
) -> list[str]:
    """FloorRouteService.find_route 핵심 알고리즘 인라인 재현."""
    corridor_nodes = {
        nid for nid, data in G.nodes(data=True)
        if not _is_leaf(data.get("node_type"), data.get("source_ref"))
    }
    G_corridor = G.subgraph(corridor_nodes)

    def get_corridor_nbr(leaf: str) -> str | None:
        for nbr in G.neighbors(leaf):
            if nbr in corridor_nodes:
                return nbr
        return None

    from_data = G.nodes[from_id]
    to_data = G.nodes[to_id]
    from_is_leaf = _is_leaf(from_data.get("node_type"), from_data.get("source_ref"))
    to_is_leaf = _is_leaf(to_data.get("node_type"), to_data.get("source_ref"))

    start = get_corridor_nbr(from_id) if from_is_leaf else from_id
    end = get_corridor_nbr(to_id) if to_is_leaf else to_id

    assert start is not None, "leaf from has no corridor neighbor"
    assert end is not None, "leaf to has no corridor neighbor"

    if start == end:
        mid = [start]
    else:
        if not nx.has_path(G_corridor, start, end):
            raise ValueError("NO_PATH")
        mid = nx.shortest_path(G_corridor, start, end, weight="weight")

    result: list[str] = []
    if from_is_leaf:
        result.append(from_id)
    result.extend(mid)
    if to_is_leaf and to_id not in result:
        result.append(to_id)
    return result


# ── Case A: from=P1, to=P2 → 중간에 POI 없음 ─────────────────────────────────

def test_case_a_poi_to_poi_no_middle_poi() -> None:
    """P1(POI)-C1-C2-C3-P2(POI) 그래프에서 from=P1, to=P2 경로의 중간에 다른 poi 없음."""
    G: nx.Graph = nx.Graph()

    p1, c1, c2, c3, p2 = "P1", "C1", "C2", "C3", "P2"
    G.add_node(p1, node_type="POI", source_ref=None)
    G.add_node(c1, node_type="CORRIDOR", source_ref=None)
    G.add_node(c2, node_type="CORRIDOR", source_ref=None)
    G.add_node(c3, node_type="CORRIDOR", source_ref=None)
    G.add_node(p2, node_type="POI", source_ref=None)

    G.add_edge(p1, c1, weight=1.0)
    G.add_edge(c1, c2, weight=1.0)
    G.add_edge(c2, c3, weight=1.0)
    G.add_edge(c3, p2, weight=1.0)

    path = _run_route(G, p1, p2)

    assert path == [p1, c1, c2, c3, p2], f"unexpected path: {path}"
    # 중간 노드(첫/마지막 제외)에 POI 없음
    middle = path[1:-1]
    for nid in middle:
        assert G.nodes[nid].get("node_type") != "POI", f"POI {nid} found in middle!"


# ── Case B: from=C1, to=C3 → P_mid 미포함 ────────────────────────────────────

def test_case_b_corridor_to_corridor_no_poi_in_middle() -> None:
    """C1-C2-C3 직선 + C2에 P_mid spur. from=C1, to=C3 → P_mid 경로 미포함."""
    G: nx.Graph = nx.Graph()

    c1, c2, c3, p_mid = "C1", "C2", "C3", "P_MID"
    G.add_node(c1, node_type="CORRIDOR", source_ref=None)
    G.add_node(c2, node_type="CORRIDOR", source_ref=None)
    G.add_node(c3, node_type="CORRIDOR", source_ref=None)
    G.add_node(p_mid, node_type="POI", source_ref=None)

    G.add_edge(c1, c2, weight=1.0)
    G.add_edge(c2, c3, weight=1.0)
    G.add_edge(c2, p_mid, weight=0.5)  # spur

    path = _run_route(G, c1, c3)

    assert p_mid not in path, f"P_MID unexpectedly in path: {path}"
    assert path == [c1, c2, c3], f"unexpected path: {path}"


# ── Case C: disconnected → NO_PATH ────────────────────────────────────────────

def test_case_c_disconnected_no_path() -> None:
    """corridor subgraph가 disconnected → ValueError('NO_PATH')."""
    G: nx.Graph = nx.Graph()

    c1, c2, p_bridge = "C1", "C2", "P_BRIDGE"
    G.add_node(c1, node_type="CORRIDOR", source_ref=None)
    G.add_node(c2, node_type="CORRIDOR", source_ref=None)
    G.add_node(p_bridge, node_type="POI", source_ref=None)

    # C1—P_BRIDGE—C2: corridor끼리는 연결 안 됨 (POI만 다리 역할)
    G.add_edge(c1, p_bridge, weight=1.0)
    G.add_edge(p_bridge, c2, weight=1.0)

    with pytest.raises(ValueError, match="NO_PATH"):
        _run_route(G, c1, c2)
