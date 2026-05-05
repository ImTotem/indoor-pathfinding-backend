"""routing 도메인 예외."""
from __future__ import annotations


class ScanNotFoundError(Exception):
    """scan_id가 scan_ingest 테이블에 존재하지 않음."""

    def __init__(self, scan_id: str) -> None:
        self.scan_id = scan_id
        super().__init__(f"scan not found: scan_id={scan_id}")


class GraphNotReadyError(Exception):
    """build가 완료되지 않아 graph를 로드할 수 없음."""

    def __init__(self, scan_id: str, state: str, build_job_id: str | None = None) -> None:
        self.scan_id = scan_id
        self.state = state
        self.build_job_id = build_job_id
        super().__init__(f"graph not ready for scan_id={scan_id} state={state}")


class SnapDistanceExceededError(Exception):
    """좌표가 가장 가까운 map_node보다 threshold_m 이상 떨어져 있음."""

    def __init__(self, distance_m: float, threshold_m: float, side: str = "start") -> None:
        self.distance_m = distance_m
        self.threshold_m = threshold_m
        self.side = side
        super().__init__(
            f"{side} snap distance {distance_m:.2f}m exceeds threshold {threshold_m:.2f}m"
        )


class EndpointNodeNotFoundError(Exception):
    """node_id로 지정한 노드가 graph에 없음."""

    def __init__(self, node_id: str, side: str = "start") -> None:
        self.node_id = node_id
        self.side = side
        super().__init__(f"{side} node_id={node_id} not found in graph")


class POINodeNotFoundError(Exception):
    """poi_mark_id로 지정한 map_node가 없음 (POI 미투영 상태)."""

    def __init__(self, poi_mark_id: int, side: str = "start") -> None:
        self.poi_mark_id = poi_mark_id
        self.side = side
        super().__init__(f"{side} poi_mark_id={poi_mark_id} not found in map_node")


class PathNotFoundError(Exception):
    """start → goal 경로가 존재하지 않음 (graph disconnected 등)."""

    def __init__(self, start_node_id: str, goal_node_id: str) -> None:
        self.start_node_id = start_node_id
        self.goal_node_id = goal_node_id
        super().__init__(f"no path from {start_node_id} to {goal_node_id}")
