# models/response_models.py
from pydantic import BaseModel
from typing import Optional, List

class SessionStartResponse(BaseModel):
    """스캔 시작 응답"""
    session_id: str
    status: str
    created_at: str
    upload_url: str

class ChunkUploadResponse(BaseModel):
    """청크 업로드 응답"""
    status: str
    session_id: str
    chunk_index: int
    received_frames: int
    message: str

class SessionFinishResponse(BaseModel):
    """스캔 완료 응답"""
    status: str
    session_id: str
    message: str
    status_url: str

class SessionStatusResponse(BaseModel):
    """세션 상태 응답"""
    session_id: str
    status: str  # scanning | queued | processing | completed | failed
    progress: float
    total_frames: int
    total_chunks: int
    created_at: str
    updated_at: Optional[str]
    map_id: Optional[str]
    error: Optional[str]

class LocalizationResponse(BaseModel):
    """위치 추정 응답"""
    success: bool
    position: List[float]
    orientation: List[float]
    confidence: float
    num_matches: int

class PathResponse(BaseModel):
    """경로 계산 응답"""
    map_id: str
    path: List[List[float]]
    distance: float
    instruction: str

