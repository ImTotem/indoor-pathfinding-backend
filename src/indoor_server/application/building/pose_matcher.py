"""Sprint 67 — poses.bin 읽고 PTS 기준 pose 매칭.

binary 포맷 (iOS PoseFileWriter 와 정확히 동일):
    record = pts_ns Int64 little-endian (8B)
           + transform 16 × Float32 little-endian (column-major, 64B)
    = 72B / record. 36000 frame (10분 60Hz) = 2.6MB.

intrinsics 는 manifest 에 한 번만 (session 동안 고정) — record 에 포함 안 함.

알고리즘:
    1. mmap 또는 read_bytes 로 전체 적재 (수 MB).
    2. (pts_ns, pose_4x4) 배열로 unpack.
    3. video frame.pts_ns 와 binary search 로 가장 가까운 pose 매칭.
    4. abs(diff) > threshold_ns 면 매칭 실패 (caller 가 skip 결정).

threshold:
    60fps frame interval = 16.67 ms = 16.67e6 ns.
    반 프레임 = 8.3 ms = 8.3e6 ns. 이게 default.
    drop frame 가능성 + jitter 허용 → 16.67e6 ns 가 안전.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Sprint 67: iOS PoseFileWriter.recordSize 와 일치
RECORD_SIZE_BYTES = 72
DEFAULT_THRESHOLD_NS = 16_666_666  # 1 frame @ 60fps


@dataclass(frozen=True)
class PoseSample:
    """pts_ns + 4x4 column-major transform (ARKit 월드 좌표)."""

    pts_ns: int
    transform: np.ndarray  # shape (4, 4), float32, column-major

    @property
    def translation(self) -> np.ndarray:
        """transform[:, 3][:3] — column-major 의 4번째 column 의 xyz."""
        return self.transform[:3, 3]


class PoseMatcher:
    """poses.bin reader + PTS 기준 binary search matcher."""

    def __init__(self, poses_path: Path) -> None:
        if not poses_path.exists():
            raise FileNotFoundError(f"poses file not found: {poses_path}")
        self._poses_path = poses_path
        self._pts_array, self._transforms = self._load(poses_path)

    @staticmethod
    def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """poses.bin 전체 적재.

        Returns:
            (pts_ns_array shape=(N,) int64,
             transforms shape=(N, 4, 4) float32 column-major)
        """
        raw = path.read_bytes()
        if len(raw) % RECORD_SIZE_BYTES != 0:
            raise ValueError(
                f"poses.bin size {len(raw)} is not a multiple of "
                f"{RECORD_SIZE_BYTES} (corrupted)"
            )

        n = len(raw) // RECORD_SIZE_BYTES
        if n == 0:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 4, 4), dtype=np.float32)

        pts = np.empty(n, dtype=np.int64)
        transforms = np.empty((n, 4, 4), dtype=np.float32)

        # 일반 unpack — N 이 36000 정도라 numpy view 로도 충분.
        # 더 빠른 path: np.frombuffer 로 stride 잡기.
        for i in range(n):
            offset = i * RECORD_SIZE_BYTES
            (pts_ns,) = struct.unpack_from("<q", raw, offset)
            floats = struct.unpack_from("<16f", raw, offset + 8)
            # iOS PoseFileWriter writes column-major: cols.0.x .y .z .w, cols.1.x ...
            # 16 floats → 4 columns × 4 rows
            mat = np.array(floats, dtype=np.float32).reshape(4, 4)
            # numpy reshape 은 row-major 라 column-major 로 transpose
            mat = mat.T
            pts[i] = pts_ns
            transforms[i] = mat
        return pts, transforms

    def __len__(self) -> int:
        return int(self._pts_array.shape[0])

    @property
    def pts_array_ns(self) -> np.ndarray:
        return self._pts_array

    def find_pose(
        self,
        pts_ns: int,
        *,
        threshold_ns: int = DEFAULT_THRESHOLD_NS,
    ) -> PoseSample | None:
        """주어진 pts_ns 에 가장 가까운 PoseSample 반환.

        Returns:
            PoseSample, 또는 abs(diff) > threshold_ns 이면 None.
        """
        if self._pts_array.size == 0:
            return None
        idx = int(np.searchsorted(self._pts_array, pts_ns, side="left"))
        candidates: list[int] = []
        if idx < self._pts_array.size:
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        best_idx = min(candidates, key=lambda i: abs(int(self._pts_array[i]) - pts_ns))
        diff = abs(int(self._pts_array[best_idx]) - pts_ns)
        if diff > threshold_ns:
            return None
        return PoseSample(
            pts_ns=int(self._pts_array[best_idx]),
            transform=self._transforms[best_idx],
        )

    def all_samples(self) -> list[PoseSample]:
        """모든 record 를 PoseSample list 로 반환 (small N 가정)."""
        return [
            PoseSample(pts_ns=int(self._pts_array[i]), transform=self._transforms[i])
            for i in range(self._pts_array.size)
        ]
