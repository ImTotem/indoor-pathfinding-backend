# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings:
    """전역 설정"""
    
    # 프로젝트 루트
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # 데이터 디렉토리
    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    SESSIONS_DIR = DATA_DIR / "sessions"
    MAPS_DIR = DATA_DIR / "maps"
    
    # SLAM 엔진 설정
    SLAM_ENGINE_TYPE = os.getenv("SLAM_ENGINE", "rtabmap")
    
    # RTAB-Map 설정
    # - 로컬: /path/to/rtabmap
    # - Docker: docker://container_name (예: docker://rtabmap)
    RTABMAP_PATH = os.getenv("RTABMAP_PATH", "docker://rtabmap")
    
    # Fixed Map Mode
    USE_FIXED_MAP = os.getenv("USE_FIXED_MAP", "false").lower() == "true"
    FIXED_MAP_ID = os.getenv("FIXED_MAP_ID", "260202-202240")
    
    # API 설정
    API_TITLE = "Indoor Navigation SLAM Backend"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "실내 네비게이션을 위한 Visual SLAM 백엔드"
    
    # CORS
    CORS_ORIGINS = ["*"]
    
    # 로깅
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # 서버 포트 (중앙 관리)
    SERVER_PORT = int(os.getenv("SERVER_PORT", "5000"))
    
    @classmethod
    def validate(cls):
        """설정 검증"""
        if cls.SLAM_ENGINE_TYPE != "rtabmap":
            raise ValueError(f"Invalid SLAM_ENGINE '{cls.SLAM_ENGINE_TYPE}', must be 'rtabmap'")
        
        if cls.SLAM_ENGINE_TYPE == "rtabmap":
            # Docker 모드인지 확인
            if cls.RTABMAP_PATH.startswith("docker://"):
                container_name = cls.RTABMAP_PATH.replace("docker://", "")
                print(f"[Settings] RTAB-Map Docker 모드: 컨테이너 '{container_name}'")
            else:
                # 로컬 모드: 경로 존재 여부 확인 (선택사항)
                if not Path(cls.RTABMAP_PATH).exists():
                    print(f"Warning: RTAB-Map 경로를 찾을 수 없습니다: {cls.RTABMAP_PATH}")

settings = Settings()
settings.validate()
