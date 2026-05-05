"""snap.py — 좌표를 가장 가까운 map_node에 snap."""
from __future__ import annotations

import math
from uuid import UUID

from indoor_server.domain.routing.errors import SnapDistanceExceededError
from indoor_server.domain.routing.models import MapNodeRow


def _euclidean_3d(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
) -> float:
    """3D 유클리드 거리."""
    dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def snap_coordinate_to_node(
    coord: tuple[float, float, float],
    node_lookup: dict[UUID, MapNodeRow],
    threshold_m: float,
    side: str = "start",
) -> tuple[UUID, float]:
    """좌표를 가장 가까운 map_node에 snap.

    Args:
        coord: (x, y, z) world 좌표.
        node_lookup: node_id → MapNodeRow 매핑.
        threshold_m: snap 허용 최대 거리.
        side: 에러 메시지용 ("start" | "goal").

    Returns:
        (nearest_node_id, distance_m)

    Raises:
        SnapDistanceExceededError: 가장 가까운 노드도 threshold_m 초과 시.
    """
    if not node_lookup:
        raise SnapDistanceExceededError(float("inf"), threshold_m, side)

    cx, cy, cz = coord
    best_id: UUID | None = None
    best_dist = float("inf")

    candidates = {
        node_id: node
        for node_id, node in node_lookup.items()
        if node.node_type not in ("poi", "poi_attach")
    }
    if not candidates:
        candidates = node_lookup

    for node_id, node in candidates.items():
        dist = _euclidean_3d(cx, cy, cz, node.x, node.y, node.z)
        if dist < best_dist:
            best_dist = dist
            best_id = node_id

    assert best_id is not None
    if best_dist > threshold_m:
        raise SnapDistanceExceededError(best_dist, threshold_m, side)

    return best_id, best_dist
