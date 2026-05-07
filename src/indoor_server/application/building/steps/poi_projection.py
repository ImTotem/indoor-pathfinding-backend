"""POI Projection step — world_pose 변환 + nearest edge spur.

A-10 (Sprint ~16): poi_mark.pose_matrix translation → world_pose 직접 복사.
Sprint 49 (Codex BLOCKER 1): ARKit↔RTABMap rigid SE(3) transform 주입.
- transform.confidence == "high" 인 경우만 ARKit raw → world frame 변환을 적용.
- low confidence 인 경우는 raw ARKit 좌표를 graph 위 nearest edge projection
  점 또는 nearest graph node 좌표로 대체 (graph_snap_fallback). 절대 raw
  ARKit 좌표를 그대로 두지 않는다.
"""
from __future__ import annotations

import math
from uuid import UUID, uuid4

from indoor_server.application.building.arkit_to_rtabmap_transform import (
    ArkitToRtabmapTransform,
)
from indoor_server.domain.building.enums import EdgeType, NodeType
from indoor_server.domain.building.models import MapEdgeVO, MapNodeVO
from indoor_server.domain.scan.models import POIMarkRow

_MIN_EDGE_SPLIT_DIST = 0.01  # 1cm 미만이면 split 생략


def _dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _project_point_to_segment_2d(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> tuple[float, float, float]:
    """점 P를 선분 AB에 사영. 반환: (proj_x, proj_y, t) t∈[0,1]."""
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-12:
        return ax, ay, 0.0
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    return ax + t * abx, ay + t * aby, t


class POIProjectionStep:
    def __init__(
        self,
        scan_id: UUID,
        build_job_id: UUID,
        *,
        arkit_to_rtabmap_transform: ArkitToRtabmapTransform | None = None,
    ) -> None:
        """
        Args:
            arkit_to_rtabmap_transform: Sprint 49 — confidence == "high" 일 때만
                apply 적용. low confidence 또는 None 일 때는 graph snap fallback.
        """
        self._scan_id = scan_id
        self._build_job_id = build_job_id
        self._transform = arkit_to_rtabmap_transform
        # Sprint 49: per-POI metadata
        self._poi_position_metadata: dict[int, dict[str, object]] = {}

    @property
    def poi_position_metadata(self) -> dict[int, dict[str, object]]:
        """Sprint 49: poi_id 별 position source 추적 (transform/graph_snap_fallback)."""
        return dict(self._poi_position_metadata)

    def run(
        self,
        pois: list[POIMarkRow],
        nodes: list[MapNodeVO],
        edges: list[MapEdgeVO],
    ) -> tuple[list[MapNodeVO], list[MapEdgeVO], dict[int, tuple[float, float, float]]]:
        """
        반환:
          - 확장된 nodes (POI + POI_ATTACH 추가)
          - 확장된 edges (split + spur 추가)
          - poi_mark_id → (x, y, z) 맵
        """
        poi_world_poses: dict[int, tuple[float, float, float]] = {}
        extra_nodes: list[MapNodeVO] = []
        extra_edges: list[MapEdgeVO] = []
        edges_to_remove: set[UUID] = set()
        edges_to_add: list[MapEdgeVO] = []

        transform_high = (
            self._transform is not None
            and self._transform.confidence == "high"
        )

        for poi in pois:
            arkit_xyz = (poi.tx, poi.ty, poi.tz)

            # Step 1: ARKit → world frame transform (high confidence 일 때만).
            if transform_high:
                assert self._transform is not None
                wx, wy, wz = self._transform.apply(arkit_xyz)
                primary_source = "transform"
            else:
                # low confidence / no transform: 일단 raw 를 후보로 두고
                # graph snap fallback 으로 덮어쓴다 (Codex BLOCKER 1).
                wx, wy, wz = arkit_xyz
                primary_source = "graph_snap_fallback"

            # Step 2: nearest edge 또는 nearest graph node 로 snap.
            # high confidence 의 경우는 spur 로 attach 만 (실제 좌표는 transform 결과 유지).
            # low confidence 의 경우는 POI 좌표 자체를 nearest projection 으로 대체.
            best_edge, best_proj, best_t, best_dist = self._find_nearest_edge(
                wx, wy, edges + edges_to_add
            )

            if not transform_high:
                # Codex BLOCKER 1: low confidence → POI 좌표를 graph snap point 로 대체.
                if best_edge is not None:
                    raw_to_snap_distance = best_dist
                    snap_x, snap_y = best_proj
                    snap_z = wz
                    wx, wy, wz = snap_x, snap_y, snap_z
                else:
                    # edge 가 없으면 nearest graph node 로 대체.
                    snap_node = self._find_nearest_node(wx, wy, nodes)
                    if snap_node is not None:
                        raw_to_snap_distance = _dist2d(
                            wx, wy, snap_node.x, snap_node.y
                        )
                        wx, wy, wz = snap_node.x, snap_node.y, snap_node.z
                    else:
                        # graph 자체가 비어있는 경우만 — POI 단독 노드 (raw 유지).
                        raw_to_snap_distance = 0.0
                        primary_source = "no_graph_isolated"

                # snap 적용 후 nearest edge 재계산 (spur 거리는 ~0 이 됨)
                best_edge, best_proj, best_t, best_dist = self._find_nearest_edge(
                    wx, wy, edges + edges_to_add
                )
            else:
                raw_to_snap_distance = 0.0  # transform path 에서는 사용 안 함

            poi_world_poses[poi.id] = (wx, wy, wz)
            self._poi_position_metadata[poi.id] = {
                "poi_position_source": primary_source,
                "raw_to_snap_distance_m": float(raw_to_snap_distance),
                "raw_arkit_xyz": [
                    float(arkit_xyz[0]),
                    float(arkit_xyz[1]),
                    float(arkit_xyz[2]),
                ],
                "final_world_xyz": [float(wx), float(wy), float(wz)],
                "transform_confidence": (
                    self._transform.confidence
                    if self._transform is not None
                    else "no_transform"
                ),
            }

            poi_node = MapNodeVO(
                node_id=uuid4(),
                scan_id=self._scan_id,
                build_job_id=self._build_job_id,
                node_type=NodeType.POI,
                x=wx, y=wy, z=wz,
                label=poi.label,
                poi_mark_id=poi.id,
            )
            extra_nodes.append(poi_node)

            if not edges or best_edge is None:
                continue

            bpx, bpy = best_proj
            bpz = wz  # floor plane 기준

            if best_dist < _MIN_EDGE_SPLIT_DIST:
                # POI가 edge 위에 거의 있음 → 별도 attach 노드 만들지 말고
                # best_edge 의 더 가까운 끝 노드를 attach 로 사용 (connectivity 보장)
                attach_node_id = (
                    best_edge.from_node_id if best_t < 0.5 else best_edge.to_node_id
                )
                attach_node = next(
                    (n for n in nodes if n.node_id == attach_node_id),
                    None,
                )
                if attach_node is None:
                    # 매우 드문 경우 — fallback: 별도 attach 만들고 spur 만
                    attach_node = MapNodeVO(
                        node_id=uuid4(),
                        scan_id=self._scan_id,
                        build_job_id=self._build_job_id,
                        node_type=NodeType.POI_ATTACH,
                        x=bpx, y=bpy, z=bpz,
                    )
                    extra_nodes.append(attach_node)
            else:
                # edge를 projection 지점에서 split
                attach_node = MapNodeVO(
                    node_id=uuid4(),
                    scan_id=self._scan_id,
                    build_job_id=self._build_job_id,
                    node_type=NodeType.POI_ATTACH,
                    x=bpx, y=bpy, z=bpz,
                )
                extra_nodes.append(attach_node)
                split_edges = self._split_edge(best_edge, attach_node, best_t)
                edges_to_remove.add(best_edge.edge_id)
                edges_to_add.extend(split_edges)

            # POI → POI_ATTACH spur edge
            spur_len = _dist2d(wx, wy, bpx, bpy)
            extra_edges.append(
                MapEdgeVO(
                    edge_id=uuid4(),
                    scan_id=self._scan_id,
                    build_job_id=self._build_job_id,
                    from_node_id=poi_node.node_id,
                    to_node_id=attach_node.node_id,
                    edge_type=EdgeType.POI_SPUR,
                    polyline=[(wx, wy, wz), (bpx, bpy, bpz)],
                    length_m=spur_len,
                )
            )

        # 원본 edges에서 제거 + 분할 edges 추가
        final_edges = (
            [e for e in edges if e.edge_id not in edges_to_remove]
            + edges_to_add
            + extra_edges
        )
        final_nodes = nodes + extra_nodes

        return final_nodes, final_edges, poi_world_poses

    def _find_nearest_edge(
        self,
        px: float,
        py: float,
        edges: list[MapEdgeVO],
    ) -> tuple[MapEdgeVO | None, tuple[float, float], float, float]:
        best_edge = None
        best_proj = (px, py)
        best_t = 0.0
        best_dist = float("inf")

        for edge in edges:
            if len(edge.polyline) < 2:
                continue
            for i in range(len(edge.polyline) - 1):
                ax, ay, _ = edge.polyline[i]
                bx, by, _ = edge.polyline[i + 1]
                proj_x, proj_y, t = _project_point_to_segment_2d(px, py, ax, ay, bx, by)
                d = _dist2d(px, py, proj_x, proj_y)
                if d < best_dist:
                    best_dist = d
                    best_proj = (proj_x, proj_y)
                    best_t = t
                    best_edge = edge

        return best_edge, best_proj, best_t, best_dist

    def _find_nearest_node(
        self,
        px: float,
        py: float,
        nodes: list[MapNodeVO],
    ) -> MapNodeVO | None:
        """Sprint 49: graph snap fallback 용 nearest graph node."""
        best_node: MapNodeVO | None = None
        best_dist = float("inf")
        for node in nodes:
            d = _dist2d(px, py, node.x, node.y)
            if d < best_dist:
                best_dist = d
                best_node = node
        return best_node

    def _split_edge(
        self,
        edge: MapEdgeVO,
        attach_node: MapNodeVO,
        t: float,
    ) -> list[MapEdgeVO]:
        """edge를 attach_node 위치에서 두 개로 분할."""
        from indoor_server.application.building.steps.node_placement import _polyline_length

        if len(edge.polyline) < 2:
            return []

        # polyline을 t 비율로 분할
        total = _polyline_length(edge.polyline)
        split_dist = t * total

        cum = 0.0
        split_idx = len(edge.polyline) - 1
        for i in range(1, len(edge.polyline)):
            seg_len = _dist2d(
                edge.polyline[i][0], edge.polyline[i][1],
                edge.polyline[i - 1][0], edge.polyline[i - 1][1],
            )
            if cum + seg_len >= split_dist:
                split_idx = i
                break
            cum += seg_len

        split_pt = (attach_node.x, attach_node.y, attach_node.z)

        poly1 = edge.polyline[:split_idx] + [split_pt]
        poly2 = [split_pt] + edge.polyline[split_idx:]

        len1 = _polyline_length(poly1)
        len2 = _polyline_length(poly2)

        return [
            MapEdgeVO(
                edge_id=uuid4(),
                scan_id=edge.scan_id,
                build_job_id=edge.build_job_id,
                from_node_id=edge.from_node_id,
                to_node_id=attach_node.node_id,
                edge_type=EdgeType.SKELETON,
                polyline=poly1,
                length_m=len1,
            ),
            MapEdgeVO(
                edge_id=uuid4(),
                scan_id=edge.scan_id,
                build_job_id=edge.build_job_id,
                from_node_id=attach_node.node_id,
                to_node_id=edge.to_node_id,
                edge_type=EdgeType.SKELETON,
                polyline=poly2,
                length_m=len2,
            ),
        ]
