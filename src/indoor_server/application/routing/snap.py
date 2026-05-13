"""snap.py — 좌표를 graph 에 attach. virtual node at P + spur edges to nearby backbone.

기존 단순 nearest-node snap 도 fallback 으로 유지 (`snap_coordinate_to_node`).
주 진입점은 `snap_coordinate_to_graph` — 측위 좌표 P 에 임시 노드 + 가까운 backbone
노드들과 spur edge 추가 후 임시 노드 반환. nearest edge perpendicular distance 는
threshold 검증 + log 용.
"""
from __future__ import annotations

import logging
import math
from uuid import UUID, uuid4

import networkx as nx

from indoor_server.domain.routing.errors import SnapDistanceExceededError
from indoor_server.domain.routing.models import MapNodeRow


logger = logging.getLogger(__name__)

VIRTUAL_NODE_TYPE = "virtual_snap"


def _euclidean_3d(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
) -> float:
    dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def snap_coordinate_to_node(
    coord: tuple[float, float, float],
    node_lookup: dict[UUID, MapNodeRow],
    threshold_m: float,
    side: str = "start",
) -> tuple[UUID, float]:
    """좌표를 가장 가까운 backbone 노드에 snap. perpendicular drop 불가능 시 fallback."""
    if not node_lookup:
        raise SnapDistanceExceededError(float("inf"), threshold_m, side)

    cx, cy, cz = coord
    candidates = {
        node_id: node
        for node_id, node in node_lookup.items()
        if node.node_type not in ("poi", "poi_attach")
    }
    if not candidates:
        candidates = node_lookup

    best_id: UUID | None = None
    best_dist = float("inf")
    for node_id, node in candidates.items():
        dist = _euclidean_3d(cx, cy, cz, node.x, node.y, node.z)
        if dist < best_dist:
            best_dist = dist
            best_id = node_id

    assert best_id is not None
    if best_dist > threshold_m:
        raise SnapDistanceExceededError(best_dist, threshold_m, side)
    return best_id, best_dist


def snap_coordinate_to_graph(
    coord: tuple[float, float, float],
    g: nx.Graph,
    node_lookup: dict[UUID, MapNodeRow],
    threshold_m: float,
    side: str = "start",
) -> tuple[UUID, float]:
    """측위 좌표 P 와 nearest backbone edge 위 foot 에 각각 가상 노드, 그 둘 사이 edge.

    그래프 모델:
      V_P     = P (사용자 측위 좌표) 위치의 가상 노드
      V_foot  = nearest backbone edge (u,v) 위 perpendicular foot 위치의 가상 노드
      edge: V_P    ↔ V_foot   (length = perp_d)         — P 에서 corridor 진입 segment
      edge: V_foot ↔ u        (length = foot_to_u)      — corridor 의 u 쪽 부분
      edge: V_foot ↔ v        (length = foot_to_v)      — corridor 의 v 쪽 부분

    기존 (u,v) edge 는 보존 (1회용 graph). dijkstra 가 V_foot 통해 corridor 따라 양쪽
    endpoint 로 갈 수 있고, V_P 의 path 시작점은 사용자 실제 위치 그대로.

    이전 (foot only) 동작: V_P 노드 없이 V_foot 만 — polyline 첫 점이 사용자 위치가
    아닌 corridor 위 → 화면에서 첫 step 방향이 측면으로 휘어 보임.

    `g` 와 `node_lookup` 은 mutate 됨 (1회용 graph, cleanup 불필요).
    """
    if not node_lookup:
        raise SnapDistanceExceededError(float("inf"), threshold_m, side)

    cx, cy, cz = coord

    backbone_types = {
        "corridor", "junction", "endpoint",
        "passage_stairs", "passage_elevator", "passage_escalator",
    }
    backbone_ids = {
        nid for nid, n in node_lookup.items() if n.node_type in backbone_types
    }

    best: tuple[float, UUID, UUID, float, float, float, tuple[float, float, float]] | None = None
    # (perp_d, u, v, t, foot_to_u, foot_to_v, foot_xyz)

    for u, v in g.edges():
        if u not in backbone_ids or v not in backbone_ids:
            continue
        nu = node_lookup[u]
        nv = node_lookup[v]
        dx, dy, dz = nv.x - nu.x, nv.y - nu.y, nv.z - nu.z
        seg_len2 = dx * dx + dy * dy + dz * dz
        if seg_len2 < 1e-9:
            continue
        t = ((cx - nu.x) * dx + (cy - nu.y) * dy + (cz - nu.z) * dz) / seg_len2
        t = max(0.0, min(1.0, t))
        fx = nu.x + t * dx
        fy = nu.y + t * dy
        fz = nu.z + t * dz
        d = _euclidean_3d(cx, cy, cz, fx, fy, fz)
        if best is None or d < best[0]:
            seg_len = math.sqrt(seg_len2)
            best = (d, u, v, t, t * seg_len, (1.0 - t) * seg_len, (fx, fy, fz))

    if best is None or best[0] > threshold_m:
        # backbone edge 가 없거나 너무 멀면 가까운 ANY 노드 (poi/poi_attach 포함) 에
        # V_P 를 가상 edge 로 직접 연결한다. 그래프가 sparse 한 (스캔 초기) 상황에서
        # SNAP_DISTANCE_EXCEEDED 로 실패하지 않고 path 가 POI spur 통해 corridor 로
        # 합류할 수 있도록 보장. fallback 도 실패하면 SnapDistanceExceededError.
        backbone_perp = best[0] if best is not None else float("inf")
        return _snap_via_nearest_node_fallback(
            coord=coord,
            g=g,
            node_lookup=node_lookup,
            threshold_m=threshold_m,
            side=side,
            backbone_perp_d=backbone_perp,
        )

    perp_d, u, v, t, foot_to_u, foot_to_v, foot_xyz = best

    nu_ref = node_lookup[u]

    # V_foot — corridor edge 위 perpendicular foot 의 가상 노드.
    foot_id = uuid4()
    foot_node = MapNodeRow(
        node_id=foot_id,
        x=foot_xyz[0],
        y=foot_xyz[1],
        z=foot_xyz[2],
        node_type=VIRTUAL_NODE_TYPE,
        label=None,
        poi_mark_id=None,
        source_ref={
            "role": "virtual_foot_on_edge",
            "side": side,
            "edge": [str(u), str(v)],
            "foot_t": t,
        },
        scan_id=nu_ref.scan_id,
        build_job_id=nu_ref.build_job_id,
        level_id=nu_ref.level_id,
    )
    node_lookup[foot_id] = foot_node
    g.add_node(
        foot_id,
        x=foot_node.x,
        y=foot_node.y,
        z=foot_node.z,
        node_type=foot_node.node_type,
        label=None,
        poi_mark_id=None,
        source_ref=foot_node.source_ref,
        scan_id=str(foot_node.scan_id) if foot_node.scan_id is not None else None,
        level_id=foot_node.level_id,
    )
    g.add_edge(foot_id, u, length_m=foot_to_u, edge_id=None,
               source_ref={"edge_kind": "virtual_foot_to_endpoint"})
    g.add_edge(foot_id, v, length_m=foot_to_v, edge_id=None,
               source_ref={"edge_kind": "virtual_foot_to_endpoint"})

    # V_P — 사용자 측위 좌표 위치의 가상 노드 (path 시작점).
    p_id = uuid4()
    p_node = MapNodeRow(
        node_id=p_id,
        x=cx,
        y=cy,
        z=cz,
        node_type=VIRTUAL_NODE_TYPE,
        label=None,
        poi_mark_id=None,
        source_ref={
            "role": "virtual_at_P",
            "side": side,
            "foot_node_id": str(foot_id),
            "perp_distance_m": perp_d,
        },
        scan_id=nu_ref.scan_id,
        build_job_id=nu_ref.build_job_id,
        level_id=nu_ref.level_id,
    )
    node_lookup[p_id] = p_node
    g.add_node(
        p_id,
        x=p_node.x,
        y=p_node.y,
        z=p_node.z,
        node_type=p_node.node_type,
        label=None,
        poi_mark_id=None,
        source_ref=p_node.source_ref,
        scan_id=str(p_node.scan_id) if p_node.scan_id is not None else None,
        level_id=p_node.level_id,
    )
    g.add_edge(p_id, foot_id, length_m=perp_d, edge_id=None,
               source_ref={"edge_kind": "virtual_p_to_foot"})

    ux, uy = node_lookup[u].x, node_lookup[u].y
    vx, vy = node_lookup[v].x, node_lookup[v].y
    cross_z = (vx - ux) * (cy - uy) - (vy - uy) * (cx - ux)
    if abs(cross_z) < 1e-9:
        edge_side = "on_edge"
    else:
        edge_side = "left" if cross_z > 0 else "right"
    xy_d = math.hypot(cx - foot_xyz[0], cy - foot_xyz[1])
    z_d = abs(cz - foot_xyz[2])

    logger.info(
        "snap side=%s P=(%.2f,%.2f,%.2f) foot=(%.2f,%.2f,%.2f) "
        "edge_side=%s cross_z=%.3f xy_d=%.3fm z_d=%.3fm perp_d=%.3fm "
        "foot_t=%.2f edge=(%s,%s) u=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f) "
        "foot_to_u=%.2f foot_to_v=%.2f",
        side, cx, cy, cz, foot_xyz[0], foot_xyz[1], foot_xyz[2],
        edge_side, cross_z, xy_d, z_d, perp_d,
        t, str(u)[:8], str(v)[:8],
        node_lookup[u].x, node_lookup[u].y, node_lookup[u].z,
        node_lookup[v].x, node_lookup[v].y, node_lookup[v].z,
        foot_to_u, foot_to_v,
    )

    return p_id, perp_d


def _snap_via_nearest_node_fallback(
    *,
    coord: tuple[float, float, float],
    g: nx.Graph,
    node_lookup: dict[UUID, MapNodeRow],
    threshold_m: float,
    side: str,
    backbone_perp_d: float,
) -> tuple[UUID, float]:
    """backbone edge snap 이 안 되면 가까운 ANY 노드 (poi/poi_attach 포함) 에 V_P 만들어 직결.

    SNAP_DISTANCE_EXCEEDED threshold 는 단일 backbone edge 기준 5m 였지만,
    fallback 은 그래프가 sparse 한 상황의 안전망이므로 약간 완화 (×2) 한 거리까지
    허용한다. 그래도 너무 멀면 SnapDistanceExceededError 그대로 던짐.
    """
    cx, cy, cz = coord
    fallback_threshold = threshold_m * 2.0

    nearest_id: UUID | None = None
    nearest_dist = float("inf")
    for nid, n in node_lookup.items():
        if n.node_type == VIRTUAL_NODE_TYPE:
            continue
        d = _euclidean_3d(cx, cy, cz, n.x, n.y, n.z)
        if d < nearest_dist:
            nearest_dist = d
            nearest_id = nid

    if nearest_id is None:
        raise SnapDistanceExceededError(backbone_perp_d, threshold_m, side)
    if nearest_dist > fallback_threshold:
        raise SnapDistanceExceededError(nearest_dist, fallback_threshold, side)

    anchor = node_lookup[nearest_id]
    p_id = uuid4()
    p_node = MapNodeRow(
        node_id=p_id,
        x=cx,
        y=cy,
        z=cz,
        node_type=VIRTUAL_NODE_TYPE,
        label=None,
        poi_mark_id=None,
        source_ref={
            "role": "virtual_at_P_fallback",
            "side": side,
            "anchor_node_id": str(nearest_id),
            "anchor_node_type": anchor.node_type,
            "backbone_perp_d_m": backbone_perp_d,
            "fallback_distance_m": nearest_dist,
        },
        scan_id=anchor.scan_id,
        build_job_id=anchor.build_job_id,
        level_id=anchor.level_id,
    )
    node_lookup[p_id] = p_node
    g.add_node(
        p_id,
        x=p_node.x,
        y=p_node.y,
        z=p_node.z,
        node_type=p_node.node_type,
        label=None,
        poi_mark_id=None,
        source_ref=p_node.source_ref,
        scan_id=str(p_node.scan_id) if p_node.scan_id is not None else None,
        level_id=p_node.level_id,
    )
    g.add_edge(
        p_id,
        nearest_id,
        length_m=nearest_dist,
        edge_id=None,
        source_ref={"edge_kind": "virtual_p_to_nearest_fallback"},
    )

    logger.info(
        "snap fallback side=%s P=(%.2f,%.2f,%.2f) anchor=%s type=%s dist=%.3fm "
        "(backbone_perp_d=%.3fm > %.2fm)",
        side, cx, cy, cz, str(nearest_id)[:8], anchor.node_type,
        nearest_dist, backbone_perp_d, threshold_m,
    )

    return p_id, nearest_dist
