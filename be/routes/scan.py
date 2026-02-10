# routes/scan.py
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from datetime import datetime
import uuid
import asyncio
import json
from typing import List

from models.request_models import DeviceInfo, ChunkData, SessionFinish
from models.response_models import (
    SessionStartResponse,
    ChunkUploadResponse,
    SessionFinishResponse,
    SessionStatusResponse
)
from storage.storage_manager import StorageManager
from services.slam_service import process_slam_async
from utils import logger, log_error

router = APIRouter(prefix="/api/scan", tags=["scan"])

storage = StorageManager()

@router.post("/start", response_model=SessionStartResponse)
async def start_scan(device_info: DeviceInfo):
    """스캔 세션 시작"""
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"[START] 세션 시작 요청: {session_id}")
    logger.debug(f"  Device: {device_info.model} / {device_info.os} {device_info.os_version}")
    
    try:
        metadata = await storage.create_session(
            session_id=session_id,
            device_info=device_info.dict()
        )
        
        logger.info(f"[START] 세션 생성 완료: {session_id}")
        
        return SessionStartResponse(
            session_id=session_id,
            status="ready",
            created_at=metadata["created_at"],
            upload_url="/api/scan/chunk"
        )
        
    except Exception as e:
        log_error(e, f"START {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chunk", response_model=ChunkUploadResponse)
async def upload_chunk(chunk: ChunkData):
    """청크 단위 프레임 업로드"""
    
    logger.info(f"[CHUNK] 업로드: session={chunk.session_id}, chunk={chunk.chunk_index}, frames={len(chunk.frames)}")
    
    try:
        if not await storage.session_exists(chunk.session_id):
            logger.warning(f"[CHUNK] 세션 없음: {chunk.session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        frames_dict = [frame.dict() for frame in chunk.frames]
        
        saved_count = await storage.save_chunk(
            session_id=chunk.session_id,
            chunk_index=chunk.chunk_index,
            frames=frames_dict
        )
        
        logger.info(f"[CHUNK] 저장 완료: {saved_count}개 프레임")
        
        return ChunkUploadResponse(
            status="ok",
            session_id=chunk.session_id,
            chunk_index=chunk.chunk_index,
            received_frames=saved_count,
            message=f"Chunk {chunk.chunk_index} saved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, f"CHUNK {chunk.session_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chunk-binary", response_model=ChunkUploadResponse)
async def upload_chunk_binary(
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    num_frames: int = Form(...),
    images: List[UploadFile] = File(...),
    depths: List[UploadFile] = File(default=[]),
    metadata: str = Form(...)
):
    """바이너리 multipart 업로드 - JPEG 프레임 + 뎁스 맵 직접 업로드"""
    
    logger.info(f"[CHUNK-BINARY] 업로드: session={session_id}, chunk={chunk_index}, frames={len(images)}, depths={len(depths)}")
    
    try:
        if not await storage.session_exists(session_id):
            logger.warning(f"[CHUNK-BINARY] 세션 없음: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Parse metadata JSON
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError as e:
            logger.warning(f"[CHUNK-BINARY] 메타데이터 파싱 실패: {e}")
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        
        timestamps = metadata_dict.get("timestamps", [])
        positions = metadata_dict.get("positions", [])
        orientations = metadata_dict.get("orientations", [])
        imu_data_list = metadata_dict.get("imu_data", [])
        camera_intrinsics = metadata_dict.get("camera_intrinsics")
        depth_widths = metadata_dict.get("depth_widths", [])
        depth_heights = metadata_dict.get("depth_heights", [])
        
        # Read depth data into memory
        depth_data_map = {}
        for idx, depth_file in enumerate(depths):
            depth_bytes = await depth_file.read()
            depth_data_map[idx] = depth_bytes
        
        # Save each image with corresponding depth
        saved_count = 0
        for idx, image_file in enumerate(images):
            try:
                image_data = await image_file.read()  # Already binary JPEG
                
                # Prepare frame data
                frame_data = {
                    "image_data": image_data,
                    "timestamp": timestamps[idx] if idx < len(timestamps) else 0,
                    "position": positions[idx] if idx < len(positions) else [0, 0, 0],
                    "orientation": orientations[idx] if idx < len(orientations) else [0, 0, 0, 1],
                    "imu_data": imu_data_list[idx] if idx < len(imu_data_list) else None,
                    "camera_intrinsics": camera_intrinsics,
                    "depth_data": depth_data_map.get(idx),
                    "depth_width": depth_widths[idx] if idx < len(depth_widths) and depth_widths[idx] is not None else None,
                    "depth_height": depth_heights[idx] if idx < len(depth_heights) and depth_heights[idx] is not None else None,
                }
                
                # Save frame using storage manager
                await storage.save_frame_binary(
                    session_id=session_id,
                    chunk_index=chunk_index,
                    frame_index=idx,
                    frame_data=frame_data
                )
                saved_count += 1
                
            except Exception as e:
                logger.error(f"[CHUNK-BINARY] 프레임 저장 실패 {idx}: {e}")
                raise
        
        logger.info(f"[CHUNK-BINARY] 저장 완료: {saved_count}개 프레임")
        
        return ChunkUploadResponse(
            status="ok",
            session_id=session_id,
            chunk_index=chunk_index,
            received_frames=saved_count,
            message=f"Chunk {chunk_index} saved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, f"CHUNK-BINARY {session_id}/{chunk_index}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/finish", response_model=SessionFinishResponse)
async def finish_scan(data: SessionFinish):
    """스캔 완료 - SLAM 처리 시작"""
    
    logger.info(f"[FINISH] 완료 요청: {data.session_id}")
    
    try:
        if not await storage.session_exists(data.session_id):
            logger.warning(f"[FINISH] 세션 없음: {data.session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"[FINISH] 마지막 청크 완료 대기 중...")
        prev_frame_count = 0
        stable_count = 0
        max_wait_seconds = 10
        
        for _ in range(max_wait_seconds * 2):
            await asyncio.sleep(0.5)
            
            status = await storage.get_session_status(data.session_id)
            current_frame_count = status.get("total_frames", 0)
            
            if current_frame_count == prev_frame_count and current_frame_count > 0:
                stable_count += 1
                if stable_count >= 3:
                    logger.info(f"[FINISH] 프레임 수 안정화됨: {current_frame_count}개")
                    break
            else:
                stable_count = 0
            
            prev_frame_count = current_frame_count
        
        final_status = await storage.get_session_status(data.session_id)
        total_frames = final_status.get("total_frames", 0)
        logger.info(f"[FINISH] 최종 프레임 수: {total_frames}개")
        
        await storage.update_status(data.session_id, "queued")
        
        from slam_interface.factory import SLAMEngineFactory
        from config.settings import settings
        slam_engine = SLAMEngineFactory.create(settings.SLAM_ENGINE_TYPE)
        
        logger.info(f"[FINISH] SLAM 처리 시작: engine={settings.SLAM_ENGINE_TYPE}")
        
        asyncio.create_task(process_slam_async(data.session_id, slam_engine))
        
        return SessionFinishResponse(
            status="processing",
            session_id=data.session_id,
            message="SLAM processing started in background",
            status_url=f"/api/scan/status/{data.session_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, f"FINISH {data.session_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{session_id}", response_model=SessionStatusResponse)
async def get_scan_status(session_id: str):
    """스캔/처리 상태 확인"""
    
    try:
        status = await storage.get_session_status(session_id)
        
        return SessionStatusResponse(
            session_id=session_id,
            status=status["status"],
            progress=status.get("progress", 0),
            total_frames=status["total_frames"],
            total_chunks=status["total_chunks"],
            created_at=status["created_at"],
            updated_at=status.get("updated_at"),
            map_id=status.get("map_id"),
            error=status.get("error"),
        )
        
    except ValueError as e:
        logger.warning(f"[STATUS] 세션 없음: {session_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_error(e, f"STATUS {session_id}")
        raise HTTPException(status_code=500, detail=str(e))
