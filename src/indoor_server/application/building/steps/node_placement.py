"""NodePlacementStep — skeleton graph → MapNodeVO/MapEdgeVO (world meters).

Sprint 34 Cycle 8: Forced Rectilinear Graph 후처리 추가.
  - skeleton polyline은 0.1m grid stair-step + skeleton noise로 zigzag 발생.
  - _forced_rectilinear_graph() 가 각 SKELETON edge polyline에
    Douglas-Peucker simplify (0.5m) → H/V 강제 → zero-length 제거 → collinear merge 적용.
  - dominant_angle: 외부 주입값(manhattan_rectification 결과) 우선.
    없으면 환경변수(INDOOR_DOMINANT_ANGLE_FORCE_ZERO, INDOOR_DOMINANT_ANGLE_DEG) 기반.
  - junction node 좌표는 소속 edge 끝점에서 재계산해 일치시킴.
  - densify (max_edge_length_m)는 rectilinear 적용 후에 수행.
  - Sprint 34 Cycle 9: graph point-in-polygon snap 추가.
    footprint polygon 외부로 나간 노드를 boundary 쪽으로 snap.
"""
from __future__ import annotations

import math
import os
from math import cos, radians, sin
from typing import Any
from uuid import UUID, uuid4

from indoor_server.application.building.steps.skeletonize import SkeletonEdge, SkeletonGraph
from indoor_server.domain.building.enums import EdgeType, NodeType
from indoor_server.domain.building.models import GridOrigin, MapEdgeVO, MapNodeVO

_DEFAULT_CORRIDOR_SAMPLE_M = 3.0
_DEFAULT_MAX_EDGE_LENGTH_M = 3.0
_RECTILINEAR_SIMPLIFY_M = 0.5   # Douglas-Peucker tolerance (m)
_RECTILINEAR_MIN_EDGE_M = 0.10  # zero-edge 제거 임계 (m)


def _pixel_to_world(px: int, py: int, origin: GridOrigin) -> tuple[float, float, float]:
    """grid 픽셀 → world frame xyz."""
    x = origin.x0 + px * origin.cell_size
    y = origin.y0 + py * origin.cell_size
    return x, y, origin.z0


def _polyline_length(pts: list[tuple[float, float, float]]) -> float:
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        dz = pts[i][2] - pts[i - 1][2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def _cumulative_distances(pts: list[tuple[float, float, float]]) -> list[float]:
    """polyline 각 점의 누적 거리 리스트 (len == len(pts))."""
    cum = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        dz = pts[i][2] - pts[i - 1][2]
        cum.append(cum[-1] + math.sqrt(dx * dx + dy * dy + dz * dz))
    return cum


# ── Forced Rectilinear Graph helpers ─────────────────────────────────────────


def _rdp_simplify(
    pts: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker 2D simplify (open polyline, 재귀)."""
    if len(pts) <= 2:
        return list(pts)
    ax, ay = pts[0]
    bx, by = pts[-1]
    abx, aby = bx - ax, by - ay
    ab_len = math.sqrt(abx * abx + aby * aby)

    max_dist = 0.0
    max_idx = 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        if ab_len < 1e-9:
            d = math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        else:
            # perpendicular distance from point to line AB
            d = abs(abx * (ay - py) - aby * (ax - px)) / ab_len
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > tolerance:
        left = _rdp_simplify(pts[:max_idx + 1], tolerance)
        right = _rdp_simplify(pts[max_idx:], tolerance)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def _rotate_pts(
    pts: list[tuple[float, float]],
    angle_deg: float,
) -> list[tuple[float, float]]:
    """2D 점들을 원점 기준으로 angle_deg만큼 회전."""
    if angle_deg == 0.0:
        return list(pts)
    rad = radians(angle_deg)
    c, s = cos(rad), sin(rad)
    return [(c * x - s * y, s * x + c * y) for x, y in pts]


def _force_rectilinear_polyline(
    pts: list[tuple[float, float]],
    min_edge_m: float,
) -> list[tuple[float, float]]:
    """open polyline의 각 edge를 H 또는 V로 강제.

    - pts[0]은 anchor (불변).
    - 각 pts[i]를 이전 pts[i-1] 기준으로 H/V 강제.
    - zero-length / min_edge_m 미만 edge 제거.
    - collinear merge (인접 3점이 같은 H 또는 V 축 공유).
    """
    if len(pts) < 2:
        return list(pts)

    projected: list[tuple[float, float]] = [pts[0]]
    for i in range(1, len(pts)):
        cur = projected[-1]
        nxt = pts[i]
        dx = nxt[0] - cur[0]
        dy = nxt[1] - cur[1]
        if abs(dx) >= abs(dy):
            projected.append((nxt[0], cur[1]))   # horizontal: y 고정
        else:
            projected.append((cur[0], nxt[1]))   # vertical: x 고정

    # zero-length / 너무 짧은 edge 제거 (연속 중복 vertex)
    deduped: list[tuple[float, float]] = [projected[0]]
    for pt in projected[1:]:
        dx = pt[0] - deduped[-1][0]
        dy = pt[1] - deduped[-1][1]
        if math.sqrt(dx * dx + dy * dy) >= min_edge_m:
            deduped.append(pt)
    if not deduped:
        return [projected[0], projected[-1]]

    # 마지막 점 보존 (endpoint 좌표 유지)
    if len(deduped) >= 1 and (
        abs(deduped[-1][0] - projected[-1][0]) > 1e-9
        or abs(deduped[-1][1] - projected[-1][1]) > 1e-9
    ):
        last_dx = projected[-1][0] - deduped[-1][0]
        last_dy = projected[-1][1] - deduped[-1][1]
        if math.sqrt(last_dx * last_dx + last_dy * last_dy) >= min_edge_m:
            deduped.append(projected[-1])

    # collinear merge: 인접 3점이 같은 H 또는 V 라인 → 중간 점 제거
    merged = _merge_collinear_open(deduped, tolerance=1e-6)
    return merged if len(merged) >= 2 else deduped


def _merge_collinear_open(
    pts: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    """open polyline의 collinear 중간 점 제거."""
    if len(pts) <= 2:
        return list(pts)
    result = [pts[0], pts[1]]
    for i in range(2, len(pts)):
        prev = result[-2]
        cur = result[-1]
        nxt = pts[i]
        same_h = (
            abs(prev[1] - cur[1]) <= tolerance
            and abs(cur[1] - nxt[1]) <= tolerance
        )
        same_v = (
            abs(prev[0] - cur[0]) <= tolerance
            and abs(cur[0] - nxt[0]) <= tolerance
        )
        if same_h or same_v:
            result[-1] = nxt  # 중간 점 대체 (cur 제거)
        else:
            result.append(nxt)
    return result


def _remove_zero_length_segments(
    polyline: list[tuple[float, float, float]],
    min_len: float,
) -> list[tuple[float, float, float]]:
    """연속 중복 점(길이 < min_len인 세그먼트) 제거. 첫 점·마지막 점은 보존."""
    if len(polyline) <= 2:
        return list(polyline)
    result = [polyline[0]]
    for pt in polyline[1:-1]:
        dx = pt[0] - result[-1][0]
        dy = pt[1] - result[-1][1]
        if math.sqrt(dx * dx + dy * dy) >= min_len:
            result.append(pt)
    result.append(polyline[-1])
    return result


def _apply_forced_rectilinear_to_polyline(
    polyline: list[tuple[float, float, float]],
    simplify_m: float = _RECTILINEAR_SIMPLIFY_M,
    min_edge_m: float = _RECTILINEAR_MIN_EDGE_M,
    dominant_angle_deg: float = 0.0,
) -> list[tuple[float, float, float]]:
    """world 3D polyline에 forced rectilinear 적용.

    z 좌표는 polyline[0]의 값을 모든 점에 유지 (floor-level).

    Steps:
      1. 2D (x,y) 추출
      2. -dominant_angle_deg 회전 (rotated frame)
      3. Douglas-Peucker simplify (simplify_m)
      4. forced rectilinear projection (H/V 강제)
      5. +dominant_angle_deg 역회전 (원래 frame)
    """
    if len(polyline) < 2:
        return list(polyline)

    z = polyline[0][2]
    xy = [(p[0], p[1]) for p in polyline]

    # Step 2: rotate to axis-aligned frame
    if dominant_angle_deg != 0.0:
        xy = _rotate_pts(xy, -dominant_angle_deg)

    # Step 3: simplify
    if simplify_m > 0.0 and len(xy) > 2:
        xy = _rdp_simplify(xy, simplify_m)

    # Step 4: forced rectilinear
    xy = _force_rectilinear_polyline(xy, min_edge_m)

    # Step 5: rotate back
    if dominant_angle_deg != 0.0:
        xy = _rotate_pts(xy, dominant_angle_deg)

    if len(xy) < 2:
        return list(polyline)

    return [(x, y, z) for x, y in xy]


def _get_dominant_angle(injected: float | None = None) -> float:
    """dominant_angle 결정.

    우선순위:
      1. injected 값 (manhattan_rectification에서 외부 주입) — None이 아니면 그대로 사용.
      2. INDOOR_DOMINANT_ANGLE_FORCE_ZERO=true → 0.0°
      3. INDOOR_DOMINANT_ANGLE_DEG env var.
    """
    if injected is not None:
        return injected
    force_zero = os.environ.get("INDOOR_DOMINANT_ANGLE_FORCE_ZERO", "true").lower()
    if force_zero in ("1", "true", "yes"):
        return 0.0
    # 환경변수로 사용자가 각도를 직접 지정할 수도 있음 (고급 옵션)
    raw = os.environ.get("INDOOR_DOMINANT_ANGLE_DEG", "0.0")
    try:
        return float(raw)
    except ValueError:
        return 0.0


class NodePlacementStep:
    def __init__(
        self,
        scan_id: UUID,
        build_job_id: UUID,
        corridor_sample_m: float = _DEFAULT_CORRIDOR_SAMPLE_M,
        max_edge_length_m: float = _DEFAULT_MAX_EDGE_LENGTH_M,
        force_rectilinear: bool = True,
        rectilinear_simplify_m: float = _RECTILINEAR_SIMPLIFY_M,
        rectilinear_min_edge_m: float = _RECTILINEAR_MIN_EDGE_M,
        dominant_angle_deg: float | None = None,
        footprint_polygon: Any | None = None,
        snap_to_footprint_threshold_m: float = 1.0,
    ) -> None:
        """
        Args:
            corridor_sample_m: corridor 노드 등간격 샘플 거리 (기본 3.0m).
            max_edge_length_m: 이 길이를 초과하는 edge는 polyline 점 수에 무관하게
                               강제 분할. 직선 2점 polyline도 mid-node 삽입.
            force_rectilinear: True(기본값)면 skeleton edge polyline에
                               forced rectilinear 적용 후 node 재배치.
            rectilinear_simplify_m: Douglas-Peucker tolerance (m).
            rectilinear_min_edge_m: zero-edge 제거 임계 (m).
            dominant_angle_deg: manhattan_rectification에서 사용한 dominant angle(deg).
                                None이면 환경변수(_get_dominant_angle) 기반.
            footprint_polygon: shapely Polygon/MultiPolygon — 노드 point-in-polygon snap용.
                               None이면 snap 비활성.
            snap_to_footprint_threshold_m: footprint 외부 노드를 boundary로 snap할 최대 거리.
                               이 거리 초과 노드는 제거.
        """
        self._scan_id = scan_id
        self._build_job_id = build_job_id
        self._corridor_sample_m = corridor_sample_m
        self._max_edge_length_m = max_edge_length_m
        self._force_rectilinear = force_rectilinear
        self._rectilinear_simplify_m = rectilinear_simplify_m
        self._rectilinear_min_edge_m = rectilinear_min_edge_m
        self._dominant_angle_deg = dominant_angle_deg
        self._footprint_polygon = footprint_polygon
        self._snap_to_footprint_threshold_m = snap_to_footprint_threshold_m

    def run(
        self,
        skel: SkeletonGraph,
        origin: GridOrigin,
    ) -> tuple[list[MapNodeVO], list[MapEdgeVO]]:
        """pixel 좌표 → world 좌표 변환. junction/endpoint 노드화. 3m 샘플링.

        Sprint 34 Cycle 8:
          force_rectilinear=True 시 skeleton edges를 axis-aligned 직선으로 변환 후
          junction/endpoint 좌표를 edge 끝점에서 재계산하여 일치시킴.
          densify (max_edge_length_m)는 rectilinear 적용 후 수행.
        Sprint 34 Cycle 9:
          footprint_polygon 주입 시 노드 point-in-polygon snap 후처리.
        """
        if self._force_rectilinear and skel.nodes and skel.edges:
            nodes, edges = self._run_with_rectilinear(skel, origin)
        else:
            nodes, edges = self._run_raw(skel, origin)

        if self._footprint_polygon is not None:
            nodes, edges = self._snap_nodes_to_footprint(nodes, edges)

        return nodes, edges

    def _run_raw(
        self,
        skel: SkeletonGraph,
        origin: GridOrigin,
    ) -> tuple[list[MapNodeVO], list[MapEdgeVO]]:
        """rectilinear 없이 기존 방식으로 실행."""
        pixel_to_node: dict[tuple[int, int], MapNodeVO] = {}

        for skel_node in skel.nodes:
            px, py = skel_node.pixel_x, skel_node.pixel_y
            ntype = self._classify_node(skel_node.degree)
            x, y, z = _pixel_to_world(px, py, origin)
            n = MapNodeVO(
                node_id=uuid4(),
                scan_id=self._scan_id,
                build_job_id=self._build_job_id,
                node_type=ntype,
                x=x, y=y, z=z,
                source_ref={"skeleton_pixel": [int(px), int(py)]},
            )
            pixel_to_node[(px, py)] = n

        nodes = list(pixel_to_node.values())
        edges: list[MapEdgeVO] = []

        for skel_edge in skel.edges:
            polyline_world = [
                _pixel_to_world(px, py, origin) for px, py in skel_edge.polyline_px
            ]
            total_len = _polyline_length(polyline_world)

            u_node = pixel_to_node.get(skel_edge.u)
            v_node = pixel_to_node.get(skel_edge.v)

            if u_node is None or v_node is None:
                continue

            if total_len > self._max_edge_length_m:
                corridor_nodes, split_edges = self._sample_corridor(
                    u_node, v_node, polyline_world, total_len
                )
                nodes.extend(corridor_nodes)
                edges.extend(split_edges)
            else:
                edges.append(
                    MapEdgeVO(
                        edge_id=uuid4(),
                        scan_id=self._scan_id,
                        build_job_id=self._build_job_id,
                        from_node_id=u_node.node_id,
                        to_node_id=v_node.node_id,
                        edge_type=EdgeType.SKELETON,
                        polyline=polyline_world,
                        length_m=total_len,
                    )
                )

        return nodes, edges

    def _run_with_rectilinear(
        self,
        skel: SkeletonGraph,
        origin: GridOrigin,
    ) -> tuple[list[MapNodeVO], list[MapEdgeVO]]:
        """Forced Rectilinear Graph 경로.

        1. pixel → world 변환
        2. 각 skeleton edge polyline에 rectilinear 적용 → 직선 axis-aligned polyline
        3. junction/endpoint node 좌표를 소속 edge 끝점에서 재계산
        4. densify (max_edge_length_m)
        """
        dominant_angle = _get_dominant_angle(self._dominant_angle_deg)

        # pixel → world: 원래 node 좌표 (rectilinear 후 갱신될 예정)
        pixel_to_node_type: dict[tuple[int, int], NodeType] = {}
        for skel_node in skel.nodes:
            px, py = skel_node.pixel_x, skel_node.pixel_y
            pixel_to_node_type[(px, py)] = self._classify_node(skel_node.degree)

        # 각 edge마다 world polyline → rectilinear polyline 생성
        # edge 끝점 좌표가 rectilinear 후 달라질 수 있으므로,
        # 각 skeleton node pixel에 대해 "갱신된 world 좌표"를 수집한다.
        pixel_to_world_updated: dict[tuple[int, int], tuple[float, float, float]] = {}
        # 같은 pixel에 여러 edge가 끝날 경우 평균냄 (junction)
        pixel_to_world_accumulator: dict[
            tuple[int, int], list[tuple[float, float, float]]
        ] = {}

        # 먼저 각 edge의 rectilinear polyline을 계산
        edge_rectilinear: list[tuple[SkeletonEdge, list[tuple[float, float, float]]]] = []
        for skel_edge in skel.edges:
            raw_poly = [
                _pixel_to_world(px, py, origin) for px, py in skel_edge.polyline_px
            ]
            if len(raw_poly) < 2:
                edge_rectilinear.append((skel_edge, raw_poly))
                continue

            rect_poly = _apply_forced_rectilinear_to_polyline(
                raw_poly,
                simplify_m=self._rectilinear_simplify_m,
                min_edge_m=self._rectilinear_min_edge_m,
                dominant_angle_deg=dominant_angle,
            )
            if len(rect_poly) < 2:
                rect_poly = [raw_poly[0], raw_poly[-1]]

            edge_rectilinear.append((skel_edge, rect_poly))

            # endpoint 좌표 수집 (u = 첫 점, v = 마지막 점)
            u_px = skel_edge.u
            v_px = skel_edge.v
            pixel_to_world_accumulator.setdefault(u_px, []).append(rect_poly[0])
            pixel_to_world_accumulator.setdefault(v_px, []).append(rect_poly[-1])

        # pixel 별 평균 좌표 계산 (junction에서 여러 edge 끝점이 약간 다를 수 있음)
        for px_key, coords_list in pixel_to_world_accumulator.items():
            if not coords_list:
                continue
            avg_x = sum(c[0] for c in coords_list) / len(coords_list)
            avg_y = sum(c[1] for c in coords_list) / len(coords_list)
            avg_z = coords_list[0][2]  # z는 floor level 고정
            pixel_to_world_updated[px_key] = (avg_x, avg_y, avg_z)

        # raw pixel → world 변환 (rectilinear 수집 안 된 isolated node용 fallback)
        pixel_to_node: dict[tuple[int, int], MapNodeVO] = {}
        for skel_node in skel.nodes:
            px, py = skel_node.pixel_x, skel_node.pixel_y
            px_key = (px, py)
            ntype = pixel_to_node_type.get(px_key, NodeType.CORRIDOR)
            if px_key in pixel_to_world_updated:
                x, y, z = pixel_to_world_updated[px_key]
            else:
                x, y, z = _pixel_to_world(px, py, origin)
            n = MapNodeVO(
                node_id=uuid4(),
                scan_id=self._scan_id,
                build_job_id=self._build_job_id,
                node_type=ntype,
                x=x, y=y, z=z,
                source_ref={"skeleton_pixel": [int(px), int(py)], "rectilinear": True},
            )
            pixel_to_node[px_key] = n

        nodes = list(pixel_to_node.values())
        edges: list[MapEdgeVO] = []

        for skel_edge, rect_poly in edge_rectilinear:
            u_node = pixel_to_node.get(skel_edge.u)
            v_node = pixel_to_node.get(skel_edge.v)
            if u_node is None or v_node is None:
                continue

            # edge의 실제 끝점 좌표를 node 좌표로 교체 (junction 좌표 통일)
            final_poly = list(rect_poly)
            final_poly[0] = (u_node.x, u_node.y, u_node.z)
            final_poly[-1] = (v_node.x, v_node.y, v_node.z)

            # zero-length segment 필터
            final_poly = _remove_zero_length_segments(final_poly, self._rectilinear_min_edge_m)
            if len(final_poly) < 2:
                final_poly = [(u_node.x, u_node.y, u_node.z), (v_node.x, v_node.y, v_node.z)]

            total_len = _polyline_length(final_poly)
            if total_len < 1e-9:
                continue

            # densify (max_edge_length_m 초과 시 강제 분할) — rectilinear 적용 후 수행
            if total_len > self._max_edge_length_m:
                corridor_nodes, split_edges = self._sample_corridor(
                    u_node, v_node, final_poly, total_len
                )
                nodes.extend(corridor_nodes)
                edges.extend(split_edges)
            else:
                edges.append(
                    MapEdgeVO(
                        edge_id=uuid4(),
                        scan_id=self._scan_id,
                        build_job_id=self._build_job_id,
                        from_node_id=u_node.node_id,
                        to_node_id=v_node.node_id,
                        edge_type=EdgeType.SKELETON,
                        polyline=final_poly,
                        length_m=total_len,
                    )
                )

        return nodes, edges

    def _snap_nodes_to_footprint(
        self,
        nodes: list[MapNodeVO],
        edges: list[MapEdgeVO],
    ) -> tuple[list[MapNodeVO], list[MapEdgeVO]]:
        """footprint polygon 기준 point-in-polygon snap.

        각 노드가 footprint polygon 내부인지 검사:
        - 내부 → 그대로 유지
        - 외부 → boundary까지 거리 계산
            * <= snap_to_footprint_threshold_m → boundary 가장 가까운 점으로 snap (좌표 교체)
            * >  snap_to_footprint_threshold_m → 노드 제거 (연결된 edge도 제거)

        제거된 노드와 연결된 edge도 일괄 제거.
        """
        import logging

        from shapely.geometry import Point

        log = logging.getLogger(__name__)
        # polygon은 run()에서 None 체크 후 호출되므로 None이 아님을 보장.
        # mypy가 Any|None 때문에 오류를 냄 → Any로 캐스팅해서 union-attr 억제.
        polygon: Any = self._footprint_polygon
        threshold = self._snap_to_footprint_threshold_m

        snapped_count = 0
        removed_count = 0
        removed_node_ids: set[UUID] = set()
        new_nodes: list[MapNodeVO] = []

        for node in nodes:
            pt = Point(node.x, node.y)
            if polygon.contains(pt):
                new_nodes.append(node)
                continue

            # MultiPolygon 대응: exterior가 없을 수 있으므로 polygon.distance로 fallback
            if hasattr(polygon, "exterior"):
                dist = float(polygon.exterior.distance(pt))
            else:
                dist = float(polygon.distance(pt))

            if dist <= threshold:
                # boundary에서 가장 가까운 점으로 snap
                if hasattr(polygon, "exterior"):
                    nearest = polygon.exterior.interpolate(polygon.exterior.project(pt))
                else:
                    nearest = polygon.interpolate(polygon.project(pt))
                snapped_node = MapNodeVO(
                    node_id=node.node_id,
                    scan_id=node.scan_id,
                    build_job_id=node.build_job_id,
                    node_type=node.node_type,
                    x=float(nearest.x),
                    y=float(nearest.y),
                    z=node.z,
                    source_ref={
                        **(node.source_ref or {}),
                        "footprint_snapped": True,
                        "snap_dist_m": round(dist, 4),
                    },
                )
                new_nodes.append(snapped_node)
                snapped_count += 1
            else:
                removed_node_ids.add(node.node_id)
                removed_count += 1

        if removed_node_ids:
            new_edges = [
                e for e in edges
                if e.from_node_id not in removed_node_ids
                and e.to_node_id not in removed_node_ids
            ]
        else:
            new_edges = edges

        if snapped_count or removed_count:
            log.info(
                "footprint snap: snapped=%d removed=%d threshold=%.2fm",
                snapped_count,
                removed_count,
                threshold,
            )

        return new_nodes, new_edges

    def _classify_node(self, degree: int) -> NodeType:
        if degree >= 3:
            return NodeType.JUNCTION
        if degree == 1:
            return NodeType.ENDPOINT
        return NodeType.CORRIDOR

    def _sample_corridor(
        self,
        u: MapNodeVO,
        v: MapNodeVO,
        polyline: list[tuple[float, float, float]],
        total_len: float,
    ) -> tuple[list[MapNodeVO], list[MapEdgeVO]]:
        """max_edge_length_m 등간격 corridor 노드 삽입 → edge 분할.

        polyline이 2점 직선이어도 mid-node를 삽입한다 (Sprint 23 densify 보강).
        """
        num_samples = max(0, int(total_len / self._max_edge_length_m) - 1)
        if num_samples == 0:
            return [], [self._make_edge(u.node_id, v.node_id, polyline, total_len)]

        corridor_nodes = self._densify_polyline(polyline, num_samples)
        split_edges = self._chain_to_edges([u] + corridor_nodes + [v])
        return corridor_nodes, split_edges

    def _densify_polyline(
        self,
        polyline: list[tuple[float, float, float]],
        num_samples: int,
    ) -> list[MapNodeVO]:
        """polyline 위 등간격 corridor 노드 num_samples 개 생성."""
        cum_dists = _cumulative_distances(polyline)
        sample_dist = self._max_edge_length_m
        corridor_nodes: list[MapNodeVO] = []
        for k in range(num_samples):
            pt = self._interpolate(polyline, cum_dists, (k + 1) * sample_dist)
            corridor_nodes.append(
                MapNodeVO(
                    node_id=uuid4(),
                    scan_id=self._scan_id,
                    build_job_id=self._build_job_id,
                    node_type=NodeType.CORRIDOR,
                    x=pt[0], y=pt[1], z=pt[2],
                )
            )
        return corridor_nodes

    def _chain_to_edges(self, chain: list[MapNodeVO]) -> list[MapEdgeVO]:
        """인접 노드 쌍을 순서대로 MapEdgeVO 로 연결."""
        edges: list[MapEdgeVO] = []
        for i in range(len(chain) - 1):
            seg = [(chain[i].x, chain[i].y, chain[i].z),
                   (chain[i + 1].x, chain[i + 1].y, chain[i + 1].z)]
            edges.append(self._make_edge(chain[i].node_id, chain[i + 1].node_id,
                                         seg, _polyline_length(seg)))
        return edges

    def _make_edge(
        self,
        from_id: UUID,
        to_id: UUID,
        polyline: list[tuple[float, float, float]],
        length_m: float,
    ) -> MapEdgeVO:
        """MapEdgeVO 생성 헬퍼."""
        return MapEdgeVO(
            edge_id=uuid4(),
            scan_id=self._scan_id,
            build_job_id=self._build_job_id,
            from_node_id=from_id,
            to_node_id=to_id,
            edge_type=EdgeType.SKELETON,
            polyline=polyline,
            length_m=length_m,
        )

    def _interpolate(
        self,
        polyline: list[tuple[float, float, float]],
        cum_dists: list[float],
        d: float,
    ) -> tuple[float, float, float]:
        """누적 거리 d에 해당하는 world 좌표 보간."""
        for i in range(1, len(cum_dists)):
            if cum_dists[i] >= d:
                t = (d - cum_dists[i - 1]) / max(1e-9, cum_dists[i] - cum_dists[i - 1])
                p0 = polyline[i - 1]
                p1 = polyline[i]
                return (
                    p0[0] + t * (p1[0] - p0[0]),
                    p0[1] + t * (p1[1] - p0[1]),
                    p0[2] + t * (p1[2] - p0[2]),
                )
        return polyline[-1]
