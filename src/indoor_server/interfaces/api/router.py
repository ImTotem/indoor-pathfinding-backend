"""라우터 — /scan/upload, /scan/{scan_id}, /scan/{scan_id}/build,
/scan/{scan_id}/imdf, /mobile/v1/routes/coordinates, /healthz, /readyz."""
from __future__ import annotations

import io
import logging
import shutil
import uuid
from pathlib import Path

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.imdf.export_service import ImdfExportService
from indoor_server.application.routing.graph_loader import GraphLoader
from indoor_server.application.routing.route_service import RouteService
from indoor_server.application.scan_ingest_service import ScanIngestService
from indoor_server.application.sidecar_parser import SidecarParser
from indoor_server.application.zip_unpacker import ZipUnpacker
from indoor_server.config import settings
from indoor_server.domain.building.enums import BuildState
from indoor_server.domain.imdf.errors import (
    FootprintNotAvailableError,
    ImdfGraphNotReadyError,
    ImdfScanNotFoundError,
)
from indoor_server.domain.routing.errors import (
    EndpointNodeNotFoundError,
    GraphNotReadyError,
    PathNotFoundError,
    POINodeNotFoundError,
    ScanNotFoundError,
    SnapDistanceExceededError,
)
from indoor_server.domain.scan.errors import (
    PayloadTooLarge,
    ScanIdInvalidFormat,
)
from indoor_server.infrastructure.db import tables as t
from indoor_server.infrastructure.db.engine import get_session
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository
from indoor_server.infrastructure.db.repositories.map_graph_repo import MapGraphRepository
from indoor_server.infrastructure.db.repositories.scan_ingest_repo import (
    ScanIngestRepository,
)
from indoor_server.infrastructure.jobs.build_enqueuer import BuildEnqueuer
from indoor_server.infrastructure.storage.local_fs import LocalFileStorage
from indoor_server.interfaces.api.schemas import (
    ApiError,
    BuildCounts,
    BuildEnqueueResponse,
    BuildStatusResponse,
    CoordinateRouteRequest,
    GraphCollectionProperties,
    GraphFeatureCollection,
    LatestBuildInfo,
    RouteEndpoint,
    RouteNode,
    RouteRequest,
    RouteResponse,
    ScanStatusResponse,
    ScanUploadCounts,
    ScanUploadResponse,
    SnapInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_store = LocalFileStorage(settings.storage_root)


# ── /scan/upload ──────────────────────────────────────────────────────────────


@router.post(
    "/scan/upload",
    response_model=ScanUploadResponse,
    status_code=status.HTTP_200_OK,
    tags=["scan"],
)
async def upload_scan(
    payload: UploadFile,
    scan_id: str = Form(...),
    device_info: str | None = Form(None),
    force: bool = False,
    session: AsyncSession = Depends(get_session),
) -> ScanUploadResponse:
    """ZIP 업로드 + sidecar 파싱 + Postgres ingest."""
    _validate_scan_id_format(scan_id)

    tmp_dir = settings.tmp_root / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_dir / f"{scan_id}.zip"

    try:
        await _save_upload(payload, zip_path)
        # W-1: auto_enqueue 연동 — settings.build_auto_enqueue=True면 BuildEnqueuer 전달
        enqueuer = BuildEnqueuer(session) if settings.build_auto_enqueue else None
        result = await ScanIngestService(
            unpacker=ZipUnpacker(),
            parser=SidecarParser(),
            store=_store,
            session=session,
            build_enqueuer=enqueuer,
        ).ingest(
            zip_path=zip_path,
            expected_scan_id=scan_id,
            device_info=device_info,
            force=force,
            tmp_dir=tmp_dir,
        )
    finally:
        # W-1 수정: zip 파일만이 아닌 tmp_dir 전체를 cleanup
        # ScanDomainError 계열은 main.py domain_error_handler가 일관 포맷으로 처리 (W-7)
        _cleanup_dir(tmp_dir)

    return ScanUploadResponse(
        scan_id=result.scan_id,
        state=result.state,
        counts=ScanUploadCounts(
            keyframes=result.keyframe_count,
            poi_marks_track_lock=result.poi_marks_track_lock,
            poi_marks_manual=result.poi_marks_manual,
            poi_photos=result.poi_photo_count,
            branch_marks=result.branch_mark_count,
            yolo_detections=result.yolo_detection_count,
        ),
        build_job_id=result.build_job_id,  # W-1: auto_enqueue 결과 반영
        storage_path=result.storage_path,
        payload_sha256=result.payload_sha256,
    )


# ── /scan/{scan_id} ───────────────────────────────────────────────────────────


@router.get(
    "/scan/{scan_id}",
    response_model=ScanStatusResponse,
    status_code=status.HTTP_200_OK,
    tags=["scan"],
)
async def get_scan(
    scan_id: str,
    session: AsyncSession = Depends(get_session),
) -> ScanStatusResponse:
    """수신 기록 조회."""
    repo = ScanIngestRepository(session)
    existing = await repo.find_existing(scan_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="scan_id not found")

    row = (
        await session.execute(
            sa.select(t.scan_ingest).where(t.scan_ingest.c.scan_id == scan_id)
        )
    ).first()
    assert row is not None

    counts = await _count_rows(session, scan_id)

    # latest_build embed
    build_repo = BuildJobRepository(session)
    latest_job = await build_repo.get_latest(scan_id)
    latest_build = None
    if latest_job is not None:
        latest_build = LatestBuildInfo(
            build_job_id=str(latest_job.build_job_id),
            state=latest_job.state.value,
            finished_at=latest_job.finished_at,
        )

    return ScanStatusResponse(
        scan_id=scan_id,
        ingested_at=row.ingested_at.isoformat(),
        payload_sha256=row.payload_sha256,
        counts=counts,
        build_job_id=row.build_job_id,
        build_state=row.build_state,
        latest_build=latest_build,
    )


# ── /scan/{scan_id}/build ─────────────────────────────────────────────────────


@router.post(
    "/scan/{scan_id}/build",
    response_model=BuildEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["build"],
)
async def trigger_build(
    scan_id: str,
    force: bool = False,
    session: AsyncSession = Depends(get_session),
) -> BuildEnqueueResponse:
    """수동 build 트리거. force=true이면 running/pending job을 취소 후 재큐잉."""
    _validate_scan_id_format(scan_id)
    repo = ScanIngestRepository(session)
    existing = await repo.find_existing(scan_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SCAN_NOT_FOUND", "message": "scan_id not found"},
        )

    build_repo = BuildJobRepository(session)
    latest_job = await build_repo.get_latest(scan_id)

    if latest_job is not None and latest_job.state in (BuildState.PENDING, BuildState.RUNNING):
        if not force:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "BUILD_ALREADY_RUNNING",
                    "message": "another build is active. use ?force=true to cancel and requeue.",
                    "detail": {"running_job_id": str(latest_job.build_job_id)},
                },
            )
        # force=true: 기존 job 취소 (find_active_jobs가 이미 트랜잭션 시작)
        await build_repo.cancel_active_jobs(scan_id)
        await session.commit()

    enqueuer = BuildEnqueuer(session)
    job = await enqueuer.enqueue(scan_id)
    await session.commit()

    return BuildEnqueueResponse(
        scan_id=scan_id,
        build_job_id=str(job.build_job_id),
        state="pending",
        enqueued_at=job.enqueued_at,
    )


@router.get(
    "/scan/{scan_id}/build",
    response_model=BuildStatusResponse,
    status_code=status.HTTP_200_OK,
    tags=["build"],
)
async def get_build_status(
    scan_id: str,
    session: AsyncSession = Depends(get_session),
) -> BuildStatusResponse:
    """최신 build_job 상태 조회."""
    _validate_scan_id_format(scan_id)
    repo = ScanIngestRepository(session)
    existing = await repo.find_existing(scan_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SCAN_NOT_FOUND", "message": "scan_id not found"},
        )

    build_repo = BuildJobRepository(session)
    latest_job = await build_repo.get_latest(scan_id)

    if latest_job is None:
        return BuildStatusResponse(
            scan_id=scan_id,
            build_job_id=None,
            state=BuildState.NOT_STARTED.value,
        )

    counts = None
    if latest_job.counts is not None:
        counts = BuildCounts(**latest_job.counts.model_dump())

    return BuildStatusResponse(
        scan_id=scan_id,
        build_job_id=str(latest_job.build_job_id),
        state=latest_job.state.value,
        current_step=latest_job.current_step.value if latest_job.current_step else None,
        progress=latest_job.progress,
        started_at=latest_job.started_at,
        finished_at=latest_job.finished_at,
        failure_reason=latest_job.failure_reason.value if latest_job.failure_reason else None,
        counts=counts,
    )


@router.get(
    "/scan/{scan_id}/graph",
    response_model=GraphFeatureCollection,
    status_code=status.HTTP_200_OK,
    tags=["build"],
)
async def get_graph(
    scan_id: str,
    session: AsyncSession = Depends(get_session),
) -> GraphFeatureCollection:
    """완료된 map graph GeoJSON 반환. 미완료 시 422."""
    _validate_scan_id_format(scan_id)
    repo = ScanIngestRepository(session)
    existing = await repo.find_existing(scan_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SCAN_NOT_FOUND", "message": "scan_id not found"},
        )

    build_repo = BuildJobRepository(session)
    latest_job = await build_repo.get_latest(scan_id)

    if latest_job is None or latest_job.state != BuildState.SUCCEEDED:
        state_str = latest_job.state.value if latest_job else "not_started"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "GRAPH_NOT_READY",
                "message": "build has not succeeded yet.",
                "detail": {
                    "state": state_str,
                    "build_job_id": str(latest_job.build_job_id) if latest_job else None,
                },
            },
        )

    graph_repo = MapGraphRepository(session)
    features = await graph_repo.load_graph_geojson(scan_id)

    # W-6: BuildCounts.floor_z0에서 실제 값 조회
    floor_z0: float | None = None
    if latest_job.counts is not None:
        floor_z0 = latest_job.counts.floor_z0

    return GraphFeatureCollection(
        features=features,
        properties=GraphCollectionProperties(
            scan_id=scan_id,
            build_job_id=str(latest_job.build_job_id),
            srid=0,
            cell_size_m=settings.walkable_grid_cell_m,
            floor_z0=floor_z0,
        ),
    )


# ── POST /route ──────────────────────────────────────────────────────────────


@router.post(
    "/route",
    response_model=RouteResponse,
    status_code=status.HTTP_200_OK,
    tags=["route"],
)
async def post_route(
    request: RouteRequest,
    session: AsyncSession = Depends(get_session),
) -> RouteResponse:
    """A* 경로 탐색. start/goal 중 하나라도 좌표면 graph nearest-node snap."""
    service = RouteService(GraphLoader(session))
    try:
        route = await service.compute(
            scan_id=str(request.scan_id),
            scan_ids=[str(scan_id) for scan_id in request.scan_ids]
            if request.scan_ids is not None
            else None,
            merge_overlaps=request.merge_overlaps,
            start=request.start,
            goal=request.goal,
        )
    except ScanNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SCAN_NOT_FOUND",
                "message": "scan_id does not exist.",
                "detail": {"scan_id": e.scan_id},
            },
        ) from e
    except GraphNotReadyError as e:
        detail: dict[str, object] = {"state": e.state}
        if e.build_job_id is not None:
            detail["build_job_id"] = e.build_job_id
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "GRAPH_NOT_READY",
                "message": "build has not succeeded yet.",
                "detail": detail,
            },
        ) from e
    except SnapDistanceExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "SNAP_DISTANCE_EXCEEDED",
                "message": f"{e.side} coordinate too far from any map node.",
                "detail": {"distance_m": e.distance_m, "threshold_m": e.threshold_m},
            },
        ) from e
    except EndpointNodeNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "ENDPOINT_NODE_NOT_FOUND",
                "message": f"{e.side} node_id not found in graph.",
                "detail": {"node_id": e.node_id},
            },
        ) from e
    except POINodeNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "POI_NODE_NOT_FOUND",
                "message": f"{e.side} poi_mark_id has no projected map_node.",
                "detail": {"poi_mark_id": e.poi_mark_id},
            },
        ) from e
    except PathNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "PATH_NOT_FOUND",
                "message": "no path between start and goal.",
                "detail": {
                    "start_node_id": e.start_node_id,
                    "goal_node_id": e.goal_node_id,
                },
            },
        ) from e

    # RoutePath → RouteResponse 변환
    path_nodes = [
        RouteNode(
            node_id=n.node_id,
            x=n.x,
            y=n.y,
            z=n.z,
            node_type=n.node_type,
            label=n.label,
            poi_mark_id=n.poi_mark_id,
            scan_id=n.scan_id,
            level_id=n.level_id,
        )
        for n in route.nodes_in_order
    ]

    # GeoJSON LineString Z
    path_geometry: dict[str, object] = {
        "type": "LineString",
        "coordinates": [[x, y, z] for x, y, z in route.polyline],
    }

    snap_info = SnapInfo(
        start_snap_distance_m=route.snap.start_snap_distance_m,
        goal_snap_distance_m=route.snap.goal_snap_distance_m,
    )

    return RouteResponse(
        scan_id=request.scan_id,
        scan_ids=route.scan_ids,
        build_job_id=route.build_job_id,
        build_job_ids=route.build_job_ids,
        path_nodes=path_nodes,
        path_geometry=path_geometry,
        length_m=route.length_m,
        node_count=len(path_nodes),
        snap_info=snap_info,
        route_metadata=route.route_metadata,
    )


# ── POST /mobile/v1/routes/coordinates ──────────────────────────────────────


@router.post(
    "/mobile/v1/routes/coordinates",
    response_model=RouteResponse,
    status_code=status.HTTP_200_OK,
    tags=["mobile", "route"],
)
async def post_mobile_coordinate_route(
    request: CoordinateRouteRequest,
    session: AsyncSession = Depends(get_session),
) -> RouteResponse:
    """VPS/local map 좌표 start/goal → route.

    사용자앱은 node_id/poi_mark_id를 몰라도 되고, 서버가 기존 graph snap + A*를 수행한다.
    """
    route_request = RouteRequest(
        scan_id=request.scan_id,
        scan_ids=request.scan_ids,
        merge_overlaps=request.merge_overlaps,
        start=RouteEndpoint(coordinate=request.start_coordinate),
        goal=RouteEndpoint(coordinate=request.goal_coordinate),
    )
    return await post_route(request=route_request, session=session)


# ── /scan/{scan_id}/imdf ─────────────────────────────────────────────────────


@router.get(
    "/scan/{scan_id}/imdf",
    status_code=200,
    tags=["imdf"],
    responses={
        200: {"content": {"application/zip": {}}},
        404: {"model": ApiError},
        422: {"model": ApiError},
    },
)
async def get_imdf(
    scan_id: str,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """IMDF-lite zip export. build succeeded + footprint 있어야 200 OK."""
    _validate_scan_id_format(scan_id)

    service = ImdfExportService(
        scan_repo=ScanIngestRepository(session),
        build_repo=BuildJobRepository(session),
        graph_repo=MapGraphRepository(session),
    )

    try:
        archive = await service.build_archive(scan_id=scan_id)
    except ImdfScanNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message, "detail": e.detail},
        ) from e
    except ImdfGraphNotReadyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": e.code, "message": e.message, "detail": e.detail},
        ) from e
    except FootprintNotAvailableError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": e.code, "message": e.message, "detail": e.detail},
        ) from e

    return StreamingResponse(
        content=io.BytesIO(archive.payload),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{archive.filename}"',
            "ETag": f'"{archive.build_job_id}"',
            "X-IMDF-Format": "indoor-pathfinding-imdf-lite/0.1",
            "X-Build-Job-Id": str(archive.build_job_id),
        },
    )


# ── /healthz / /readyz ────────────────────────────────────────────────────────


@router.get("/healthz", tags=["health"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", tags=["health"])
async def readyz(session: AsyncSession = Depends(get_session)) -> dict[str, str]:
    db_ok = await _ping_db(session)
    storage_ok = await _store.readyz()
    if db_ok and storage_ok:
        return {"status": "ok"}
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"db": db_ok, "storage": storage_ok},
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def _validate_scan_id_format(scan_id: str) -> None:
    # W-7 수정: HTTPException 대신 ScanDomainError → main.py handler가 ApiError 포맷으로 반환
    try:
        uuid.UUID(scan_id)
    except ValueError as e:
        raise ScanIdInvalidFormat("scan_id가 UUID 형식이 아닙니다.") from e


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    """업로드 파일을 임시 경로에 저장. 크기 상한 체크 포함."""
    # W-7 수정: HTTPException 대신 PayloadTooLarge (ScanDomainError) → ApiError 포맷 일관성
    total = 0
    limit = settings.max_upload_bytes
    with open(dest, "wb") as f:
        while chunk := await upload.read(1 << 18):  # 256 KB 청크
            total += len(chunk)
            if total > limit:
                raise PayloadTooLarge(f"업로드 상한 {limit} bytes 초과")
            f.write(chunk)


def _cleanup_dir(path: Path) -> None:
    # W-1 수정: tmp_dir 전체 cleanup (zip 파일만이 아닌 디렉터리 통째)
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


async def _ping_db(session: AsyncSession) -> bool:
    try:
        await session.execute(sa.text("SELECT 1"))
        return True
    except Exception:
        return False


async def _count_rows(session: AsyncSession, scan_id: str) -> ScanUploadCounts:
    from indoor_server.domain.poi.enums import POISource

    kf = (
        await session.execute(
            sa.select(sa.func.count()).where(t.keyframe_meta.c.scan_id == scan_id)
        )
    ).scalar_one()
    tl = (
        await session.execute(
            sa.select(sa.func.count()).where(
                t.poi_mark.c.scan_id == scan_id,
                t.poi_mark.c.source == POISource.TRACK_LOCK.value,
            )
        )
    ).scalar_one()
    mn = (
        await session.execute(
            sa.select(sa.func.count()).where(
                t.poi_mark.c.scan_id == scan_id,
                t.poi_mark.c.source == POISource.MANUAL.value,
            )
        )
    ).scalar_one()
    pp = (
        await session.execute(
            sa.select(sa.func.count()).where(t.poi_photo.c.scan_id == scan_id)
        )
    ).scalar_one()
    bm = (
        await session.execute(
            sa.select(sa.func.count()).where(t.branch_mark.c.scan_id == scan_id)
        )
    ).scalar_one()
    yd = (
        await session.execute(
            sa.select(sa.func.count()).where(t.yolo_detection.c.scan_id == scan_id)
        )
    ).scalar_one()
    return ScanUploadCounts(
        keyframes=kf,
        poi_marks_track_lock=tl,
        poi_marks_manual=mn,
        poi_photos=pp,
        branch_marks=bm,
        yolo_detections=yd,
    )
