import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """Legacy SLAM settings used by the restored `/api/slam/*` compatibility surface."""

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    MAPS_DIR = DATA_DIR / "maps"
    # Persistent SuperPoint feature cache (safetensors + sidecar JSON per floor).
    # Survives uvicorn --reload so the next localize warm-up reads from mmap'd
    # disk (~1s) instead of rebuilding from rtabmap.db (~26s per floor).
    SUPERPOINT_CACHE_DIR = Path(
        os.getenv(
            "SUPERPOINT_CACHE_DIR",
            BASE_DIR.parent / "var" / "cache" / "superpoint",
        )
    )

    SLAM_ENGINE_TYPE = os.getenv("SLAM_ENGINE", "rtabmap")
    RTABMAP_PATH = os.getenv(
        "RTABMAP_PATH",
        shutil.which("rtabmap-reprocess") or "/usr/bin/rtabmap-reprocess",
    )

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> None:
        if cls.SLAM_ENGINE_TYPE != "rtabmap":
            raise ValueError(f"Invalid SLAM_ENGINE '{cls.SLAM_ENGINE_TYPE}', must be 'rtabmap'")

        if not cls.RTABMAP_PATH.startswith("docker://") and not Path(cls.RTABMAP_PATH).exists():
            print(f"Warning: RTAB-Map 경로를 찾을 수 없습니다: {cls.RTABMAP_PATH}")


settings = Settings()
settings.validate()
