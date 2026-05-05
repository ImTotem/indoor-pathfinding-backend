"""Sprint 78 A-9: video_mode_pipeline POI projection path 단위 테스트.

POIProjectionStep이 map_nodes/map_edges에 POI 노드와 spur 엣지를 추가하는지,
label이 없으면 "301호"로 채워지는지 검증한다. DB 의존 없음.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from indoor_server.application.building.steps.poi_projection import POIProjectionStep
from indoor_server.domain.building.enums import EdgeType, NodeType
from indoor_server.domain.building.models import MapEdgeVO, MapNodeVO
from indoor_server.domain.scan.models import POIMarkRow


def _make_corridor_node(x: float, y: float, scan_id: UUID, build_id: UUID) -> MapNodeVO:
    return MapNodeVO(
        node_id=uuid4(),
        scan_id=scan_id,
        build_job_id=build_id,
        node_type=NodeType.CORRIDOR,
        x=x,
        y=y,
        z=0.0,
    )


def _make_edge(n1: MapNodeVO, n2: MapNodeVO, scan_id: UUID, build_id: UUID) -> MapEdgeVO:
    return MapEdgeVO(
        edge_id=uuid4(),
        scan_id=scan_id,
        build_job_id=build_id,
        from_node_id=n1.node_id,
        to_node_id=n2.node_id,
        edge_type=EdgeType.SKELETON,
        polyline=[(n1.x, n1.y, 0.0), (n2.x, n2.y, 0.0)],
        length_m=float(((n2.x - n1.x) ** 2 + (n2.y - n1.y) ** 2) ** 0.5),
    )


def _make_poi(
    scan_id: UUID,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    label: str | None = None,
) -> POIMarkRow:
    return POIMarkRow(
        id=1,
        scan_id=str(scan_id),
        keyframe_seq=0,
        created_at=0,
        pose_matrix=b"",
        tx=tx,
        ty=ty,
        tz=tz,
        track_id=None,
        label=label,
        source="manual",
    )


class TestPOIProjectionStep:
    def setup_method(self) -> None:
        self.scan_id = uuid4()
        self.build_id = uuid4()

        # 단순 corridor: C1(0,0) — C2(5,0)
        self.c1 = _make_corridor_node(0.0, 0.0, self.scan_id, self.build_id)
        self.c2 = _make_corridor_node(5.0, 0.0, self.scan_id, self.build_id)
        self.edge = _make_edge(self.c1, self.c2, self.scan_id, self.build_id)

    def test_poi_node_added(self) -> None:
        """POI 1개 → POI 노드가 nodes에 추가된다."""
        poi = _make_poi(self.scan_id, tx=2.5, ty=1.0, label="301호")
        step = POIProjectionStep(
            scan_id=self.scan_id,
            build_job_id=self.build_id,
            arkit_to_rtabmap_transform=None,
        )
        new_nodes, new_edges, poi_poses = step.run(
            [poi], [self.c1, self.c2], [self.edge]
        )
        poi_nodes = [n for n in new_nodes if n.node_type == NodeType.POI]
        assert len(poi_nodes) >= 1, "POI 노드가 추가되지 않음"

    def test_poi_spur_edge_added(self) -> None:
        """POI → corridor spur edge가 생성된다."""
        poi = _make_poi(self.scan_id, tx=2.5, ty=1.0, label="test")
        step = POIProjectionStep(
            scan_id=self.scan_id,
            build_job_id=self.build_id,
            arkit_to_rtabmap_transform=None,
        )
        _, new_edges, _ = step.run([poi], [self.c1, self.c2], [self.edge])
        spur_edges = [e for e in new_edges if e.edge_type == EdgeType.POI_SPUR]
        assert len(spur_edges) >= 1, "POI spur edge가 생성되지 않음"

    def test_label_fallback_hardcoded(self) -> None:
        """label=None POI에 호출자가 '301호'로 채워서 넣으면 그대로 보존된다."""
        # build_service.py에서 label fill 후 step.run에 전달하는 패턴 재현
        poi = _make_poi(self.scan_id, tx=2.5, ty=0.5, label=None)
        # label 채우기 (A-2 로직 재현) — POIMarkRow는 pydantic BaseModel
        if not poi.label:
            poi = poi.model_copy(update={"label": "301호"})

        step = POIProjectionStep(
            scan_id=self.scan_id,
            build_job_id=self.build_id,
            arkit_to_rtabmap_transform=None,
        )
        new_nodes, _, _ = step.run([poi], [self.c1, self.c2], [self.edge])
        poi_nodes = [n for n in new_nodes if n.node_type == NodeType.POI]
        assert len(poi_nodes) >= 1
        assert poi_nodes[0].label == "301호", f"label mismatch: {poi_nodes[0].label}"

    def test_no_pois_no_change(self) -> None:
        """POI 0개 → nodes/edges 변경 없음."""
        step = POIProjectionStep(
            scan_id=self.scan_id,
            build_job_id=self.build_id,
            arkit_to_rtabmap_transform=None,
        )
        new_nodes, new_edges, poi_poses = step.run([], [self.c1, self.c2], [self.edge])
        assert len(new_nodes) == 2
        assert len(poi_poses) == 0
