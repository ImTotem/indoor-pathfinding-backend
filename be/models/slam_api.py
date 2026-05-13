from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional


class SLAMProcessRequest(BaseModel):
    building_id: str = Field(
        ...,
        description="처리할 건물 ID. 이 값으로 활성 floor scan과 RTAB-Map DB를 찾습니다.",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )

    @field_validator('building_id')
    @classmethod
    def validate_building_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('building_id cannot be empty')
        return v.strip()


class SLAMProcessResponse(BaseModel):
    map_id: str = Field(..., description="처리 대상 건물 ID. legacy 호환을 위해 map_id 이름으로 반환합니다.")
    status: str = Field(..., description="현재 처리 상태")
    queue_position: int = Field(..., description="SLAM 처리 큐에서의 대기 위치")


class SLAMLocalizeRequest(BaseModel):
    map_id: str = Field(
        ...,
        description="기존 내부 호출 호환용 ID. 통합 서버에서는 building_id와 동일하게 해석합니다.",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    images: List[str] = Field(
        ...,
        description="내부 호환용 base64 이미지 목록. 외부 `/api/slam/*/localize`는 파일 업로드를 사용합니다.",
        json_schema_extra={"example": ["base64_img1"]},
    )
    camera_intrinsics: dict = Field(
        ...,
        description="내부 호환용 카메라 파라미터. 현재 외부 API는 RTAB-Map DB calibration을 우선 사용합니다.",
        json_schema_extra={"example": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}},
    )
    
    @field_validator('images')
    @classmethod
    def validate_images(cls, v: list) -> list:
        if len(v) < 1:
            raise ValueError('Must provide at least 1 image')
        return v


class SLAMLocalizeResponse(BaseModel):
    pose: dict = Field(..., description="추정된 로컬 좌표와 자세")
    confidence: float = Field(..., description="위치 추정 confidence")
    mapId: str = Field("", description="요청 건물 ID")
    numMatches: int = Field(0, description="매칭된 feature 수")
    matchedImageIndex: int = Field(0, description="가장 잘 매칭된 요청 이미지 인덱스")
    floorId: str = Field("", description="선택된 층 ID")
    floorLevel: int = Field(0, description="선택된 층 레벨")
    debug: dict = Field(default_factory=dict, description="엔진별 디버그 메타데이터")


class MapMetadata(BaseModel):
    map_id: str = Field(..., description="legacy 호환 map ID. 현재는 building_id와 동일합니다.")
    building_id: str = Field(..., description="건물 ID")
    num_keyframes: int = Field(..., description="RTAB-Map DB 또는 빌드 세션에서 확인한 keyframe 수")
    created_at: str = Field(..., description="맵 또는 첫 스캔 생성 시각")
    status: str = Field(..., description="맵 처리 상태")


class HealthResponse(BaseModel):
    status: str = Field(..., description="SLAM 모듈 상태")
    postgres: str = Field(..., description="PostgreSQL 연결 상태")
    queue_length: int = Field(..., description="SLAM 처리 큐 길이")


class MaskDebugRequest(BaseModel):
    images: List[str] = Field(
        ...,
        description="내부 호환용 base64 이미지 목록. 외부 `/api/slam/v2/debug/mask`는 파일 업로드를 사용합니다.",
        json_schema_extra={"example": ["base64_img1"]},
    )

    @field_validator('images')
    @classmethod
    def validate_images(cls, v: list) -> list:
        if len(v) < 1:
            raise ValueError('Must provide at least 1 image')
        if len(v) > 5:
            raise ValueError('Maximum 5 images allowed')
        return v


class MaskDebugImage(BaseModel):
    index: int = Field(..., description="요청 이미지 인덱스")
    original_b64: str = Field(..., description="원본 이미지 base64")
    annotated_b64: str = Field(..., description="사람 감지 박스를 표시한 이미지 base64")
    persons_detected: int = Field(..., description="감지된 사람 수")


class MaskDebugResponse(BaseModel):
    total_images: int = Field(..., description="처리한 이미지 수")
    results: List[MaskDebugImage] = Field(..., description="이미지별 마스킹 디버그 결과")


class MatchDebugResponse(BaseModel):
    query_b64: str = Field(..., description="쿼리 이미지 base64")
    matches_b64: str = Field(..., description="매칭 시각화 이미지 base64")
    db_frame_b64: Optional[str] = Field(None, description="매칭된 DB 프레임 이미지 base64")
    best_node_id: int = Field(..., description="가장 잘 매칭된 RTAB-Map node ID")
    num_good_matches: int = Field(..., description="good match 수")
    num_node_matches: int = Field(..., description="선택 node 기준 match 수")
    floor_id: str = Field("", description="선택된 층 ID")
    floor_name: str = Field("", description="선택된 층 이름")
    has_db_image: bool = Field(..., description="DB 프레임 이미지 포함 여부")
