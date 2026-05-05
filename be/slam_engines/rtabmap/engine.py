"""Minimal RTAB-Map compatibility engine for legacy SLAM v3 localization.

The integrated backend keeps map building in `src/indoor_server`. The legacy
SuperPoint localizer still needs RTAB-Map camera intrinsics from uploaded
`rtabmap.db` files, so this module preserves that small surface.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any

from config.settings import settings
from slam_interface.base import SLAMEngineBase

_CAMERA_MODEL_HEADER_SIZE = 11
_MONO_CAMERA_TYPE = 0


class RTABMapEngine(SLAMEngineBase):
    def __init__(self) -> None:
        from slam_engines.rtabmap.database_parser import DatabaseParser

        self.database_parser = DatabaseParser()

    async def process(self, *args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError("RTAB-Map processing is owned by src/indoor_server")

    async def localize(self, *args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError("Use slam_engines.superpoint.engine.SuperPointEngine")

    def save_map(self, map_data: dict, map_id: str, base_dir: Path) -> Path:
        base_dir.mkdir(parents=True, exist_ok=True)
        output = base_dir / f"{map_id}.db"
        binary = map_data.get("binary")
        if not isinstance(binary, bytes):
            raise ValueError("map_data['binary'] must be bytes")
        output.write_bytes(binary)
        return output

    def load_map(self, map_id: str, base_dir: Path) -> bytes:
        return (base_dir / f"{map_id}.db").read_bytes()

    async def _load_map_file(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def extract_intrinsics_from_db(self, db_path: str) -> dict:
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        with sqlite3.connect(path) as conn:
            row = conn.execute(
                """
                SELECT calibration
                FROM Data
                WHERE calibration IS NOT NULL
                LIMIT 1
                """
            ).fetchone()

        if not row or row[0] is None:
            raise ValueError("RTABMap Data.calibration not found")

        calibration = _decode_calibration(bytes(row[0]))
        width, height = calibration["image_size"]
        k = calibration["k"]
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
            raise ValueError("invalid RTABMap camera intrinsics")
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": int(width),
            "height": int(height),
            "rtabmap_path": settings.RTABMAP_PATH,
        }

    def scale_intrinsics(self, original: dict, new_width: int, new_height: int) -> dict:
        old_width = float(original["width"])
        old_height = float(original["height"])
        scale_x = float(new_width) / old_width
        scale_y = float(new_height) / old_height
        return {
            "fx": float(original["fx"]) * scale_x,
            "fy": float(original["fy"]) * scale_y,
            "cx": float(original["cx"]) * scale_x,
            "cy": float(original["cy"]) * scale_y,
            "width": int(new_width),
            "height": int(new_height),
        }


def _decode_calibration(data: bytes) -> dict:
    header_bytes = _CAMERA_MODEL_HEADER_SIZE * 4
    if len(data) < header_bytes:
        raise ValueError("calibration header too short")

    header = struct.unpack("<11i", data[:header_bytes])
    if header[3] != _MONO_CAMERA_TYPE:
        raise ValueError(f"unsupported camera type: {header[3]}")

    k_count = header[6]
    d_count = header[7]
    r_count = header[8]
    p_count = header[9]
    local_count = header[10]
    if k_count != 9:
        raise ValueError(f"unexpected K count: {k_count}")

    required = header_bytes + 8 * (k_count + d_count + r_count + p_count) + 4 * local_count
    if len(data) < required:
        raise ValueError(f"calibration data too short: {len(data)} < {required}")

    offset = header_bytes
    k = struct.unpack("<9d", data[offset : offset + 72])
    return {
        "version": (int(header[0]), int(header[1]), int(header[2])),
        "image_size": (int(header[4]), int(header[5])),
        "k": k,
        "bytes_read": required,
    }
