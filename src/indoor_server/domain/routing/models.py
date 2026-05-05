"""routing 도메인 VO — 순수 dataclass. 외부 의존 없음."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class MapNodeRow:
    """DB에서 읽어온 map_node 행 — routing 용도."""

    node_id: UUID
    x: float
    y: float
    z: float
    node_type: str       # "junction" | "endpoint" | "corridor" | "poi" | "poi_attach"
    label: str | None
    poi_mark_id: int | None
    source_ref: dict[str, object] | None = None
    scan_id: UUID | None = None
    build_job_id: UUID | None = None
    level_id: str | None = None


@dataclass(frozen=True)
class MapEdgeRow:
    """DB에서 읽어온 map_edge 행 — routing 용도."""

    edge_id: UUID
    from_node_id: UUID
    to_node_id: UUID
    length_m: float
    scan_id: UUID | None = None
    build_job_id: UUID | None = None


@dataclass(frozen=True)
class SnapInfo:
    """좌표 입력에 대한 snap 결과. node_id/poi_mark_id 직접 입력 시 None."""

    start_snap_distance_m: float | None
    goal_snap_distance_m: float | None


@dataclass(frozen=True)
class RoutePath:
    """A* 경로 탐색 결과 VO."""

    nodes_in_order: list[MapNodeRow]
    polyline: list[tuple[float, float, float]]
    length_m: float
    snap: SnapInfo
    build_job_id: UUID
    scan_ids: list[UUID] | None = None
    build_job_ids: list[UUID] | None = None
    route_metadata: dict[str, object] | None = None
