from __future__ import annotations

import asyncio
import base64
import functools
import io
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image, ImageOps

from config.settings import Settings
from models.slam_api import (
    HealthResponse,
    MapMetadata,
    MaskDebugImage,
    MaskDebugResponse,
    MatchDebugResponse,
    SLAMLocalizeRequest,
    SLAMLocalizeResponse,
    SLAMProcessRequest,
    SLAMProcessResponse,
)
from utils import logger

settings = Settings()

SLAM_TAG_PROCESS = "SLAM - 처리"
SLAM_TAG_LOCALIZE = "SLAM - 위치추정"
SLAM_TAG_DEBUG = "SLAM - 디버그"
USER_APP_TAG = "사용자 앱 API"

router = APIRouter(prefix="/api/slam")

postgres_adapter = None
job_queue = None

_sp_engine = None
_sp_engine_lock = threading.Lock()


def _get_sp_engine():
    global _sp_engine
    if _sp_engine is None:
        with _sp_engine_lock:
            if _sp_engine is None:
                from slam_engines.superpoint.engine import SuperPointEngine

                _sp_engine = SuperPointEngine()
    return _sp_engine


@router.post(
    "/process",
    response_model=SLAMProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[SLAM_TAG_PROCESS],
    summary="건물 SLAM 처리 요청",
    description=(
        "건물 ID에 연결된 활성 RTAB-Map DB를 처리 큐에 넣습니다. "
        "현재 통합 서버에서는 v2 스캔/빌드 테이블을 우선 조회하고, legacy "
        "`scan_sessions` 스키마가 있으면 fallback으로 사용합니다."
    ),
)
async def process_slam(request: SLAMProcessRequest) -> SLAMProcessResponse:
    sessions = await _get_sessions_or_raise(request.building_id)
    pairs = [
        (str(session["id"]), str(session["file_path"]))
        for session in sessions
        if session.get("id") and session.get("file_path")
    ]
    if not pairs:
        raise HTTPException(
            status_code=404,
            detail=f"No processable RTAB-Map DB found for building {request.building_id}",
        )

    if job_queue is not None:
        await job_queue.enqueue(request.building_id, pairs)
        return SLAMProcessResponse(
            map_id=request.building_id,
            status="PROCESSING",
            queue_position=_queue_length(),
        )

    return SLAMProcessResponse(
        map_id=request.building_id,
        status=_overall_status(sessions),
        queue_position=0,
    )


@router.get(
    "/status/{building_id}",
    tags=[SLAM_TAG_PROCESS],
    summary="건물 SLAM 처리 상태 조회",
    description=(
        "건물 ID 기준으로 활성 floor scan 또는 legacy scan session 상태를 조회합니다. "
        "`map_id`를 따로 받지 않고 건물 ID로 연결된 맵 후보를 찾는 통합 기준입니다."
    ),
)
async def get_slam_status(building_id: str) -> dict[str, Any]:
    sessions = await _get_sessions(building_id)
    if not sessions:
        fallback_db = settings.MAPS_DIR / f"{building_id}.db"
        if fallback_db.exists():
            return {
                "building_id": building_id,
                "status": "COMPLETED",
                "sessions": [],
                "source": "maps_dir",
            }
        return {
            "building_id": building_id,
            "status": "NOT_FOUND",
            "sessions": [],
            "source": "postgres" if postgres_adapter is not None else "unavailable",
        }

    return {
        "building_id": building_id,
        "status": _overall_status(sessions),
        "sessions": [_serialize_session(session) for session in sessions],
        "source": "postgres",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=[SLAM_TAG_PROCESS],
    summary="SLAM 모듈 헬스 체크",
    description="PostgreSQL 연결 상태와 SLAM 처리 큐 길이를 반환합니다.",
)
async def get_slam_health() -> HealthResponse:
    postgres_status = "disabled"
    if postgres_adapter is not None:
        postgres_status = await postgres_adapter.health_check()
    return HealthResponse(
        status="healthy" if postgres_status == "connected" else "degraded",
        postgres=postgres_status,
        queue_length=_queue_length(),
    )


@router.get(
    "/maps/{building_id}/metadata",
    response_model=MapMetadata,
    tags=[SLAM_TAG_PROCESS],
    summary="건물 맵 메타데이터 조회",
    description=(
        "건물 ID로 활성 floor map을 찾고, RTAB-Map DB가 있으면 keyframe 수를 직접 읽습니다. "
        "DB 연결이 없으면 `DATA_DIR/maps/{building_id}.db`를 fallback으로 확인합니다."
    ),
)
async def get_map_metadata(building_id: str) -> MapMetadata:
    sessions = await _get_sessions(building_id)
    floor_maps = await _get_floor_maps(building_id)
    candidate_paths = [Path(_resolve_floor_path(fm)["file_path"]) for fm in floor_maps]

    fallback_db = settings.MAPS_DIR / f"{building_id}.db"
    if fallback_db.exists():
        candidate_paths.append(fallback_db)

    keyframes = sum(await asyncio.gather(*[_count_keyframes(path) for path in candidate_paths]))
    if keyframes == 0 and sessions:
        keyframes = sum(int(session.get("total_nodes") or 0) for session in sessions)

    created_at = _first_created_at(sessions)
    if created_at == "" and fallback_db.exists():
        created_at = datetime.fromtimestamp(fallback_db.stat().st_mtime).isoformat()

    if not sessions and not candidate_paths:
        raise HTTPException(status_code=404, detail=f"No map metadata found for building {building_id}")

    return MapMetadata(
        map_id=building_id,
        building_id=building_id,
        num_keyframes=keyframes,
        created_at=created_at,
        status=_overall_status(sessions) if sessions else "COMPLETED",
    )


@router.post(
    "/localize",
    response_model=SLAMLocalizeResponse,
    status_code=status.HTTP_200_OK,
    tags=[SLAM_TAG_LOCALIZE],
    summary="기본 위치 추정",
    description=(
        "호환용 기본 위치 추정 API입니다. `multipart/form-data`로 이미지 파일을 직접 받고, "
        "`building_id` 또는 기존 호환 필드 `map_id`를 건물 ID로 해석해 활성 floor map을 찾습니다."
    ),
)
async def localize_in_map(
    images: list[UploadFile] = File(..., description="위치 추정에 사용할 이미지 파일 목록"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> SLAMLocalizeResponse:
    return await _localize_uploads(building_id=building_id, map_id=map_id, images=images)


@router.post(
    "/v2/localize",
    response_model=SLAMLocalizeResponse,
    status_code=status.HTTP_200_OK,
    tags=[SLAM_TAG_LOCALIZE],
    summary="v2 위치 추정",
    description=(
        "v2 호환 위치 추정 API입니다. 이미지 파일을 직접 업로드하며, 사람 마스킹 경로를 켠 상태로 "
        "현재 SuperPoint + LightGlue 위치 추정기를 호출합니다. 마스킹 모델이 없으면 fail-open으로 동작합니다."
    ),
)
async def localize_in_map_v2(
    images: list[UploadFile] = File(..., description="위치 추정에 사용할 이미지 파일 목록"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> SLAMLocalizeResponse:
    return await _localize_uploads(
        building_id=building_id,
        map_id=map_id,
        images=images,
        mask_persons=True,
    )


@router.post(
    "/v3/localize",
    response_model=SLAMLocalizeResponse,
    status_code=status.HTTP_200_OK,
    tags=[USER_APP_TAG, SLAM_TAG_LOCALIZE],
    summary="v3 위치 추정",
    description=(
        "v3 위치 추정 API입니다. 이미지 파일을 직접 업로드하고, 건물 ID로 찾은 활성 floor map 전체에 "
        "SuperPoint + LightGlue 매칭을 수행한 뒤 confidence가 가장 높은 층 결과를 반환합니다."
    ),
)
async def localize_in_map_v3(
    images: list[UploadFile] = File(..., description="위치 추정에 사용할 이미지 파일 목록"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> SLAMLocalizeResponse:
    return await _localize_uploads(building_id=building_id, map_id=map_id, images=images)


@router.post(
    "/v1/debug/matches",
    response_model=MatchDebugResponse,
    tags=[SLAM_TAG_DEBUG],
    summary="v1 매칭 디버그",
    description=(
        "v1 호환 매칭 디버그 API입니다. 이미지 파일 하나와 건물 ID를 받아 현재 통합된 매칭 "
        "시각화 경로로 결과 이미지를 반환합니다."
    ),
)
async def debug_matches_v1(
    image: UploadFile = File(..., description="디버그할 이미지 파일"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> MatchDebugResponse:
    return await _debug_matches_upload(building_id=building_id, map_id=map_id, image=image)


@router.post(
    "/v2/debug/mask",
    response_model=MaskDebugResponse,
    tags=[SLAM_TAG_DEBUG],
    summary="v2 사람 마스킹 디버그",
    description=(
        "업로드한 이미지에서 사람 감지 박스를 표시해 반환합니다. 현재 마스킹 구현은 모델이 없으면 "
        "감지 0개로 fail-open합니다."
    ),
)
async def debug_mask_v2(
    images: list[UploadFile] = File(..., description="마스킹 디버그에 사용할 이미지 파일 목록. 최대 5장"),
) -> MaskDebugResponse:
    image_bytes_list = await _read_upload_images(images, max_images=5)

    from slam_engines.rtabmap.person_masker import PersonMasker

    masker = PersonMasker()
    results: list[MaskDebugImage] = []
    loop = asyncio.get_running_loop()
    for index, image_bytes in enumerate(image_bytes_list):
        boxes = await loop.run_in_executor(None, functools.partial(masker.detect_boxes, image_bytes))
        annotated_b64 = _annotate_person_boxes(image_bytes, boxes)
        results.append(
            MaskDebugImage(
                index=index,
                original_b64=base64.b64encode(image_bytes).decode("ascii"),
                annotated_b64=annotated_b64,
                persons_detected=len(boxes),
            )
        )
    return MaskDebugResponse(total_images=len(results), results=results)


@router.post(
    "/v2/debug/matches",
    response_model=MatchDebugResponse,
    tags=[SLAM_TAG_DEBUG],
    summary="v2 매칭 디버그",
    description=(
        "v2 호환 매칭 디버그 API입니다. 이미지 파일 하나와 건물 ID를 받아 사람 마스킹 경로를 "
        "켠 상태로 매칭 시각화 결과를 반환합니다."
    ),
)
async def debug_matches_v2(
    image: UploadFile = File(..., description="디버그할 이미지 파일"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> MatchDebugResponse:
    return await _debug_matches_upload(
        building_id=building_id,
        map_id=map_id,
        image=image,
        mask_persons=True,
    )


@router.post(
    "/v3/debug/matches",
    response_model=MatchDebugResponse,
    tags=[SLAM_TAG_DEBUG],
    summary="v3 매칭 디버그",
    description=(
        "v3 SuperPoint 매칭 디버그 API입니다. 이미지 파일 하나와 건물 ID를 받아 활성 floor map "
        "후보 중 가장 좋은 매칭 시각화 결과를 반환합니다."
    ),
)
async def debug_matches_v3(
    image: UploadFile = File(..., description="디버그할 이미지 파일"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> MatchDebugResponse:
    return await _debug_matches_upload(building_id=building_id, map_id=map_id, image=image)


async def _localize_uploads(
    *,
    building_id: str | None,
    map_id: str | None,
    images: list[UploadFile],
    mask_persons: bool = False,
) -> SLAMLocalizeResponse:
    resolved_building_id = _coerce_building_id(building_id, map_id)
    image_bytes_list = await _read_upload_images(images)

    # Query image debug dump (사용자 앱 메인 endpoint 의 query 보존)
    try:
        import datetime as _dt
        from pathlib import Path as _Path
        import os as _os
        storage_root = _Path(_os.environ.get("INDOOR_STORAGE_ROOT", "/app/var/storage"))
        dump_dir = storage_root / "debug" / "localize" / str(resolved_building_id)
        dump_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
        for i, b in enumerate(image_bytes_list):
            (dump_dir / f"{ts}_{i:02d}.jpg").write_bytes(b)
        logger.info("v3 localize query dump: %s/ %d images", dump_dir, len(image_bytes_list))
    except Exception as _dump_exc:
        logger.warning("v3 localize query dump 실패: %s", _dump_exc)

    return await _localize_bytes(
        building_id=resolved_building_id,
        image_bytes_list=image_bytes_list,
        mask_persons=mask_persons,
    )


async def _localize_impl(
    request: SLAMLocalizeRequest,
    mask_persons: bool = False,
    engine=None,
) -> SLAMLocalizeResponse:
    """Compatibility core for internal callers that still pass base64 JSON."""

    image_bytes_list = []
    for index, img_b64 in enumerate(request.images):
        try:
            image_bytes_list.append(base64.b64decode(img_b64))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in image {index + 1}: {exc}")

    return await _localize_bytes(
        building_id=request.map_id,
        image_bytes_list=image_bytes_list,
        mask_persons=mask_persons,
        engine=engine,
    )


async def _localize_bytes(
    *,
    building_id: str,
    image_bytes_list: list[bytes],
    mask_persons: bool = False,
    engine=None,
) -> SLAMLocalizeResponse:
    logger.info(f"[SLAM-LOCALIZE] building_id: {building_id}, mask_persons: {mask_persons}")

    if not image_bytes_list:
        raise HTTPException(status_code=422, detail="At least one image file is required")

    floor_maps = await _get_floor_maps(building_id)
    if not floor_maps:
        single_db = settings.MAPS_DIR / f"{building_id}.db"
        if single_db.exists():
            floor_maps = [
                {
                    "floor_id": "",
                    "floor_name": "",
                    "level": 0,
                    "file_path": str(single_db),
                }
            ]
        else:
            raise HTTPException(status_code=404, detail=f"No maps found for building {building_id}")

    slam_engine = engine
    if slam_engine is None:
        slam_engine = await asyncio.get_running_loop().run_in_executor(None, _get_sp_engine)
    resolved_floors = [_resolve_floor_path(fm) for fm in floor_maps]

    intrinsics = None
    for fm in resolved_floors:
        try:
            intrinsics = slam_engine.extract_intrinsics_from_db(fm["file_path"])
            break
        except Exception:
            continue

    if intrinsics is None:
        raise HTTPException(status_code=500, detail="Failed to extract intrinsics from any floor DB")

    image_bytes_list = _resize_query_images(
        image_bytes_list,
        width=intrinsics["width"],
        height=intrinsics["height"],
    )

    async def _localize_floor(fm: dict) -> dict | None:
        try:
            result = await slam_engine.localize(
                fm["floor_id"] or building_id,
                image_bytes_list,
                intrinsics=intrinsics,
                db_path=fm["file_path"],
                mask_persons=mask_persons,
            )
            result["floor_id"] = fm["floor_id"]
            result["floor_name"] = fm["floor_name"]
            result["floor_level"] = fm["level"]
            return result
        except (FileNotFoundError, ValueError) as exc:
            logger.debug(f"[SLAM-LOCALIZE] Floor {fm['floor_name']}: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"[SLAM-LOCALIZE] Floor {fm['floor_name']} error: {exc}")
            return None

    results = await asyncio.gather(*[_localize_floor(fm) for fm in resolved_floors])
    valid = [r for r in results if r is not None]

    if not valid:
        raise HTTPException(status_code=503, detail="Localization failed on all floors")

    # 우선순위: inlier 절대값 > confidence (작은 graph 의 비율 기반 false positive 방지).
    # 작은 graph (예: 4 keyframe) 가 inliers 6 인데도 confidence 비율이 높아 큰 graph 보다
    # 잘못 선택되는 문제 — 큰 graph 의 inliers 많은 매칭이 더 신뢰할 만.
    best = max(valid, key=lambda r: (r.get("num_matches", 0), r["confidence"]))

    logger.info(
        f"[SLAM-LOCALIZE] Best: floor={best.get('floor_name')}, "
        f"confidence={best['confidence']:.2f}, matches={best.get('num_matches', 0)}"
    )

    return SLAMLocalizeResponse(
        pose=best["pose"],
        confidence=best["confidence"],
        mapId=building_id,
        numMatches=best.get("num_matches", 0),
        matchedImageIndex=best.get("matched_image_index", 0),
        floorId=best.get("floor_id", ""),
        floorLevel=best.get("floor_level", 0),
    )


async def _debug_matches_upload(
    *,
    building_id: str | None,
    map_id: str | None,
    image: UploadFile,
    mask_persons: bool = False,
) -> MatchDebugResponse:
    resolved_building_id = _coerce_building_id(building_id, map_id)
    image_bytes = (await _read_upload_images([image], max_images=1))[0]
    return await _debug_matches(
        building_id=resolved_building_id,
        image_bytes=image_bytes,
        mask_persons=mask_persons,
    )


async def _debug_matches(
    *,
    building_id: str,
    image_bytes: bytes,
    mask_persons: bool = False,
) -> MatchDebugResponse:
    floor_maps = await _get_floor_maps(building_id)
    single_db = settings.MAPS_DIR / f"{building_id}.db"
    if not floor_maps and single_db.exists():
        floor_maps = [
            {
                "floor_id": "",
                "floor_name": "",
                "level": 0,
                "file_path": str(single_db),
            }
        ]
    if not floor_maps:
        raise HTTPException(status_code=404, detail=f"No maps found for building {building_id}")

    slam_engine = await asyncio.get_running_loop().run_in_executor(None, _get_sp_engine)
    best_response: MatchDebugResponse | None = None
    best_score = -1

    for floor_map in [_resolve_floor_path(fm) for fm in floor_maps]:
        db_path = floor_map["file_path"]
        if not Path(db_path).exists():
            continue
        try:
            intrinsics = slam_engine.extract_intrinsics_from_db(db_path)
            resized = _resize_query_images(
                [image_bytes],
                width=intrinsics["width"],
                height=intrinsics["height"],
            )[0]
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    _visualize_superpoint_matches,
                    db_path,
                    floor_map["floor_id"] or building_id,
                    resized,
                    slam_engine,
                    mask_persons,
                ),
            )
        except Exception as exc:
            logger.warning(f"[SLAM-DEBUG] Floor {floor_map.get('floor_name')} error: {exc}")
            continue

        response = _match_debug_response(result, floor_map)
        score = response.num_node_matches or response.num_good_matches
        if score > best_score:
            best_response = response
            best_score = score

    if best_response is None:
        raise HTTPException(status_code=503, detail="Match debug failed on all floors")
    return best_response


def _visualize_superpoint_matches(
    db_path: str,
    map_id: str,
    image_bytes: bytes,
    slam_engine: object,
    mask_persons: bool,
) -> dict:
    _ = mask_persons
    from slam_engines.superpoint.match_debugger import visualize_matches_sp

    return visualize_matches_sp(db_path, map_id, image_bytes, slam_engine)


def _match_debug_response(result: dict, floor_map: dict) -> MatchDebugResponse:
    return MatchDebugResponse(
        query_b64=_bgr_to_jpeg_b64(result["query_bgr"]),
        matches_b64=_bgr_to_jpeg_b64(result["vis_bgr"]),
        db_frame_b64=_bgr_to_jpeg_b64(result["db_bgr"]) if result.get("db_bgr") is not None else None,
        best_node_id=int(result.get("best_node_id") or 0),
        num_good_matches=int(result.get("num_good_matches") or 0),
        num_node_matches=int(result.get("num_node_matches") or 0),
        floor_id=str(floor_map.get("floor_id") or ""),
        floor_name=str(floor_map.get("floor_name") or ""),
        has_db_image=bool(result.get("has_db_image")),
    )


async def _get_sessions_or_raise(building_id: str) -> list[dict]:
    sessions = await _get_sessions(building_id)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"No scan sessions found for building {building_id}")
    return sessions


async def _get_sessions(building_id: str) -> list[dict]:
    if postgres_adapter is None:
        return []
    try:
        return await postgres_adapter.get_sessions_by_building_id(building_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid building_id: {exc}")
    except Exception as exc:
        logger.warning(f"[SLAM] Failed to fetch sessions for building {building_id}: {exc}")
        raise HTTPException(status_code=503, detail=f"Failed to fetch sessions: {exc}")


async def _get_floor_maps(building_id: str) -> list[dict]:
    if postgres_adapter is None:
        return []
    try:
        return await postgres_adapter.get_floor_maps(building_id)
    except ValueError:
        return []
    except Exception as exc:
        logger.warning(f"[SLAM] Failed to fetch floor maps for building {building_id}: {exc}")
        return []


async def _read_upload_images(images: list[UploadFile], *, max_images: int | None = None) -> list[bytes]:
    if not images:
        raise HTTPException(status_code=422, detail="At least one image file is required")
    if max_images is not None and len(images) > max_images:
        raise HTTPException(status_code=422, detail=f"Maximum {max_images} image files allowed")

    image_bytes_list = []
    for index, upload in enumerate(images):
        content_type = upload.content_type or ""
        if content_type and not content_type.startswith("image/") and content_type != "application/octet-stream":
            raise HTTPException(
                status_code=422,
                detail=f"File {index + 1} must be an image, got {content_type}",
            )
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=422, detail=f"File {index + 1} is empty")
        image_bytes_list.append(data)
    return image_bytes_list


def _coerce_building_id(building_id: str | None, map_id: str | None) -> str:
    resolved = (building_id or map_id or "").strip()
    if not resolved:
        raise HTTPException(status_code=422, detail="building_id or map_id form field is required")
    return resolved


def _resolve_floor_path(floor_map: dict) -> dict:
    file_path = str(floor_map["file_path"])
    if file_path.startswith("./storage/uploads/") or file_path.startswith("storage/uploads/"):
        file_path = f"/app/storage/uploads/{file_path.split('/')[-1]}"
    return {**floor_map, "file_path": file_path}


def _resize_query_images(images: list[bytes], *, width: int, height: int) -> list[bytes]:
    resized = []
    for img_bytes in images:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)))
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            resized.append(buf.getvalue())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")
    return resized


async def _count_keyframes(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    from slam_engines.rtabmap.database_parser import DatabaseParser

    parsed = await DatabaseParser().parse_database(str(db_path), keyframe_limit=0)
    return int(parsed.get("num_keyframes") or 0)


def _overall_status(sessions: list[dict]) -> str:
    if not sessions:
        return "NOT_FOUND"
    statuses = {str(session.get("status") or "").upper() for session in sessions}
    if "FAILED" in statuses:
        return "FAILED"
    if statuses and statuses <= {"COMPLETED"}:
        return "COMPLETED"
    if statuses & {"PROCESSING", "EXTRACTING"}:
        return "PROCESSING"
    if statuses & {"UPLOADED", "PENDING"}:
        return "UPLOADED"
    return next(iter(statuses), "UNKNOWN")


def _serialize_session(session: dict) -> dict[str, Any]:
    return {
        "id": str(session.get("id") or ""),
        "building_id": str(session.get("building_id") or ""),
        "file_name": session.get("file_name"),
        "file_path": session.get("file_path"),
        "file_size": session.get("file_size"),
        "status": session.get("status"),
        "error_message": session.get("error_message"),
        "total_nodes": session.get("total_nodes"),
        "total_distance": session.get("total_distance"),
        "created_at": _iso(session.get("created_at")),
        "updated_at": _iso(session.get("updated_at")),
    }


def _first_created_at(sessions: list[dict]) -> str:
    values = [session.get("created_at") for session in sessions if session.get("created_at") is not None]
    if not values:
        return ""
    return _iso(min(values))


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _queue_length() -> int:
    if job_queue is None:
        return 0
    try:
        return int(job_queue.get_queue_length())
    except Exception:
        return 0


def _annotate_person_boxes(image_bytes: bytes, boxes: list[tuple[int, int, int, int]]) -> str:
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    annotated = bgr.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return _bgr_to_jpeg_b64(annotated)


def _bgr_to_jpeg_b64(bgr: Any) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode debug image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
