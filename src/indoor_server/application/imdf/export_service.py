"""ImdfExportService — IMDF-lite zip 생성 진입점."""
from __future__ import annotations

import logging
from uuid import UUID

from indoor_server.application.imdf.builder import ImdfBuilder
from indoor_server.application.imdf.zip_archiver import build_imdf_zip
from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.imdf.errors import (
    FootprintNotAvailableError,
    ImdfGraphNotReadyError,
    ImdfScanNotFoundError,
)
from indoor_server.domain.imdf.models import ImdfArchive
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository
from indoor_server.infrastructure.db.repositories.map_graph_repo import MapGraphRepository
from indoor_server.infrastructure.db.repositories.scan_ingest_repo import ScanIngestRepository

logger = logging.getLogger(__name__)


class ImdfExportService:
    """scan_id를 받아 IMDF-lite zip bytes를 반환한다.

    의존:
        ScanIngestRepository — scan 존재 여부 확인
        BuildJobRepository   — build_job 상태 + counts(footprint_geojson) 조회
        MapGraphRepository   — POI map_node 조회
    """

    def __init__(
        self,
        scan_repo: ScanIngestRepository,
        build_repo: BuildJobRepository,
        graph_repo: MapGraphRepository,
    ) -> None:
        self._scan_repo = scan_repo
        self._build_repo = build_repo
        self._graph_repo = graph_repo

    async def build_archive(self, *, scan_id: str) -> ImdfArchive:
        """경로 결정 → DB 조회 → zip bytes 생성.

        Raises:
            ImdfScanNotFoundError: scan_id 없음.
            ImdfGraphNotReadyError: build 미완료.
            FootprintNotAvailableError: footprint_geojson 없음 (구버전 build).
        """
        # 1. scan 존재 확인
        existing = await self._scan_repo.find_existing(scan_id)
        if existing is None:
            raise ImdfScanNotFoundError(scan_id)

        # 2. build_job 상태 확인
        latest_job = await self._build_repo.get_latest(scan_id)
        if latest_job is None or latest_job.state != BuildState.SUCCEEDED:
            state_str = latest_job.state.value if latest_job else "not_started"
            raise ImdfGraphNotReadyError(scan_id, state_str)

        # 3. footprint 확인
        counts = latest_job.counts
        footprint_raw = counts.footprint_geojson if counts is not None else None
        if footprint_raw is None:
            raise FootprintNotAvailableError(scan_id, str(latest_job.build_job_id))
        footprint_for_export = (
            counts.rectified_footprint_geojson
            if counts is not None and counts.rectified_footprint_geojson is not None
            else footprint_raw
        )

        # 4. POI 노드 조회 (node_type='poi')
        all_nodes, _edges, _job_id = await self._graph_repo.load_graph_for_routing(scan_id)
        poi_nodes = [n for n in all_nodes if n.node_type == "poi"]

        # poi_mark_id → label 매핑 (R-4 미구현 — poi_mark.label 직접 사용)
        poi_labels: dict[int, str | None] = {}
        for node in poi_nodes:
            if node.poi_mark_id is not None:
                poi_labels[node.poi_mark_id] = node.label
        # 5. footprint floor_z0
        floor_z0 = counts.floor_z0 if counts is not None and counts.floor_z0 is not None else 0.0

        # 6. ImdfFeatureSet 조립 (unit polygon 제거 — footprint + amenity + anchor만)
        builder = ImdfBuilder()
        # 결정성: 동일 build_job에 대해 두 번 export해도 sha256 동일.
        # build_job.started_at(workerが処理した時刻) → enqueued_at 순 fallback.
        generated_at = (
            latest_job.started_at
            or latest_job.enqueued_at
        )

        features = builder.build(
            scan_id=UUID(scan_id),
            build_job_id=latest_job.build_job_id,
            footprint_geojson=footprint_for_export,
            poi_nodes=poi_nodes,
            poi_labels=poi_labels,
            floor_z0=floor_z0,
            generated_at=generated_at,
            rectification=counts.rectification if counts is not None else None,
            build_source=counts.build_source if counts is not None else None,
            rtabmap=counts.rtabmap if counts is not None else None,
            rtabmap_trajectory=(
                counts.rtabmap_trajectory if counts is not None else None
            ),
        )

        # 7. zip 직렬화
        zip_bytes = build_imdf_zip(features)
        filename = f"{scan_id}.imdf.zip"

        logger.info(
            "imdf_export done scan_id=%s build_job_id=%s zip_bytes=%d",
            scan_id,
            latest_job.build_job_id,
            len(zip_bytes),
        )

        return ImdfArchive(
            scan_id=UUID(scan_id),
            build_job_id=latest_job.build_job_id,
            payload=zip_bytes,
            filename=filename,
        )
