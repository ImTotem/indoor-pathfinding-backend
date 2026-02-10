# models/request_models.py
from pydantic import BaseModel
from typing import List, Optional

class DeviceInfo(BaseModel):
    """디바이스 정보"""
    model: str
    os: str
    os_version: Optional[str] = None

class FrameData(BaseModel):
    """단일 프레임 데이터"""
    image: str  # base64
    position: List[float]  # [x, y, z]
    orientation: List[float]  # [qx, qy, qz, qw]
    timestamp: int
    imu: Optional[dict] = None
    camera_intrinsics: Optional[dict] = None

class ChunkData(BaseModel):
    """청크 단위 프레임 묶음"""
    session_id: str
    chunk_index: int
    frames: List[FrameData]

class SessionFinish(BaseModel):
    """스캔 완료"""
    session_id: str

class LocalizationRequest(BaseModel):
    """위치 추정 요청"""
    map_id: str
    image: str  # base64
    initial_position: Optional[List[float]] = None
    initial_orientation: Optional[List[float]] = None

class PathRequest(BaseModel):
    """경로 계산 요청"""
    map_id: str
    start_position: List[float]
    destination_id: str

