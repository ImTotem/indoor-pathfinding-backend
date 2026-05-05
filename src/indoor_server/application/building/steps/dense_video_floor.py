"""Sprint 67 — DenseVideoFloorStep.

scan.mp4 (60Hz HEVC) + poses.bin (60Hz) + manifest intrinsics →
per-frame Segformer floor mask + ray-plane (z=z0) intersection →
dense world (x, y) point cloud → FloorPointCloud (기존 step 과 호환).

설계:
    - Sprint 22 AdaptiveBufferStep 의 ray-plane intersection 그대로 재사용 (depth 의존 없음).
    - Sprint 46 FloorPointCloud dataclass 그대로 출력 → downstream FloorRasterStep / WalkableGrid /
      polygon / graph 모든 step 변경 없이 동작.
    - keyframe(5Hz) 대신 video stride(default 12 = 5Hz, 권장 2 = 30Hz dense) 사용.
    - depth 사용 안 함 (segformer floor mask + ARKit pose + z=z0 평면 가정).

비교:
    기존 FloorPointCloudStep: rtabmap.db Data.depth × calibration × Node.pose, 5Hz keyframe.
    DenseVideoFloorStep    : video frame × manifest intrinsics × poses.bin pose, 30Hz dense.
    Floor mask 는 두 step 모두 ADE20K Segformer-B0 (`extract_floor_stair_masks`) 동일.

호출자: BuildPipeline.use_dense_video_floor=True 일 때 FloorPointCloudStep 대신 호출.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from indoor_server.application.building.pose_matcher import (
    DEFAULT_THRESHOLD_NS,
    PoseMatcher,
)
from indoor_server.application.building.steps.back_projection import Intrinsics
from indoor_server.application.building.steps.floor_point_cloud import FloorPointCloud
from indoor_server.application.building.video_frame_extractor import (
    VideoFrameExtractor,
)
from indoor_server.infrastructure.ml.segformer_onnx import (
    SegformerOnnxSegmenter,
    extract_floor_stair_masks,
)

logger = logging.getLogger(__name__)


# ARKit downward ray 최소 dz (Sprint 22 AdaptiveBuffer 와 동일).
# near-horizon ray 는 ray-plane intersection 에서 lambda 폭발 → false world point 방어.
_MIN_DOWNWARD_DZ = 0.05
_Z_TOLERANCE = 0.30


@dataclass(frozen=True)
class DenseVideoFloorParams:
    """DenseVideoFloorStep tunable parameters."""

    # video stride. 60fps 입력 기준 1=60Hz, 2=30Hz, 12=5Hz.
    # 30Hz 정도면 floor mask redundancy 충분 + segformer batch cost 감당 가능.
    stride: int = 2

    # segformer 입력은 segformer 자체 internal resize. 본 step 은 input 만 결정.
    pixel_stride: int = 4  # floor mask sampling stride (cost 절감)

    # ray-plane intersection 수직 허용오차.
    height_tolerance_m: float = _Z_TOLERANCE

    # PoseMatcher PTS 매칭 허용 시간 (ns). 1 frame @ 60fps 이내.
    pose_match_threshold_ns: int = DEFAULT_THRESHOLD_NS

    # pose 매칭 실패 비율이 이를 초과하면 build FAIL.
    max_pose_miss_ratio: float = 0.05

    def to_metadata(self) -> dict[str, object]:
        return {
            "stride": self.stride,
            "pixel_stride": self.pixel_stride,
            "height_tolerance_m": self.height_tolerance_m,
            "pose_match_threshold_ns": self.pose_match_threshold_ns,
            "max_pose_miss_ratio": self.max_pose_miss_ratio,
        }


class DenseVideoFloorError(RuntimeError):
    """DenseVideoFloorStep 실패 (pose 매칭 누락 비율 초과 등)."""


class DenseVideoFloorStep:
    """video + poses.bin → segformer → dense FloorPointCloud."""

    def __init__(
        self,
        segmenter: SegformerOnnxSegmenter,
        params: DenseVideoFloorParams | None = None,
    ) -> None:
        self._segmenter = segmenter
        self._params = params if params is not None else DenseVideoFloorParams()
        if self._params.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self._params.stride}")
        if not (0 <= self._params.max_pose_miss_ratio <= 1):
            raise ValueError(
                "max_pose_miss_ratio must be in [0, 1], got "
                f"{self._params.max_pose_miss_ratio}"
            )

    async def run(
        self,
        *,
        video_path: Path,
        poses_path: Path,
        intrinsics: Intrinsics,
        z0: float,
    ) -> FloorPointCloud:
        """video frame 디코드 → segformer → world (x,y) 누적.

        Args:
            video_path: scan.mp4
            poses_path: poses.bin (Sprint 67 binary 포맷)
            intrinsics: manifest 에서 읽은 fx/fy/cx/cy. 세션 동안 고정.
            z0: floor plane height (world z, 미터).
        """
        extractor = VideoFrameExtractor(video_path)
        info = extractor.probe()
        logger.info(
            "DenseVideoFloor: video codec=%s %dx%d fps=%s frames=%s",
            info.get("codec"), info.get("width"), info.get("height"),
            info.get("fps_avg"), info.get("frame_count_estimate"),
        )

        matcher = PoseMatcher(poses_path)
        if len(matcher) == 0:
            raise DenseVideoFloorError("poses.bin 이 비어 있습니다.")

        all_xy: list[np.ndarray] = []
        all_z: list[np.ndarray] = []
        frames_total = 0
        frames_pose_miss = 0
        frames_used = 0
        frames_no_floor = 0

        for vframe in extractor.iter_frames(stride=self._params.stride):
            frames_total += 1
            sample = matcher.find_pose(
                vframe.pts_ns,
                threshold_ns=self._params.pose_match_threshold_ns,
            )
            if sample is None:
                frames_pose_miss += 1
                continue

            seg = await self._segmenter.segment(vframe.image_rgb)
            floor_mask, _stair = extract_floor_stair_masks(seg.class_mask)

            if not floor_mask.any():
                frames_no_floor += 1
                continue

            ys_idx, xs_idx = np.where(floor_mask)
            stride = self._params.pixel_stride
            if stride > 1:
                ys_idx = ys_idx[::stride]
                xs_idx = xs_idx[::stride]

            xy, z_vals = self._back_project(
                xs=xs_idx.astype(np.float64),
                ys=ys_idx.astype(np.float64),
                pose=sample.transform,
                intrinsics=intrinsics,
                z0=z0,
            )
            if xy.shape[0] == 0:
                continue
            all_xy.append(xy)
            all_z.append(z_vals)
            frames_used += 1

        miss_ratio = frames_pose_miss / max(1, frames_total)
        if miss_ratio > self._params.max_pose_miss_ratio:
            raise DenseVideoFloorError(
                f"pose 매칭 누락 비율 {miss_ratio:.3f} > "
                f"{self._params.max_pose_miss_ratio} (frames={frames_total}, "
                f"miss={frames_pose_miss})"
            )

        if not all_xy:
            logger.warning(
                "DenseVideoFloor: world point 0 — frames_total=%d used=%d no_floor=%d miss=%d",
                frames_total, frames_used, frames_no_floor, frames_pose_miss,
            )
            return FloorPointCloud(
                points_xy=np.zeros((0, 2), dtype=np.float64),
                z_values=np.zeros((0,), dtype=np.float64),
                z0=z0,
                metadata={
                    "frames_total": frames_total,
                    "frames_used": frames_used,
                    "frames_pose_miss": frames_pose_miss,
                    "frames_no_floor": frames_no_floor,
                    "stride": self._params.stride,
                },
            )

        points_xy = np.concatenate(all_xy, axis=0)
        z_values = np.concatenate(all_z, axis=0)
        logger.info(
            "DenseVideoFloor: world_points=%d frames_total=%d used=%d no_floor=%d miss=%d",
            points_xy.shape[0], frames_total, frames_used, frames_no_floor, frames_pose_miss,
        )
        return FloorPointCloud(
            points_xy=points_xy,
            z_values=z_values,
            z0=z0,
            metadata={
                "frames_total": frames_total,
                "frames_used": frames_used,
                "frames_pose_miss": frames_pose_miss,
                "frames_no_floor": frames_no_floor,
                "stride": self._params.stride,
                "params": self._params.to_metadata(),
                "intrinsics": {
                    "fx": intrinsics.fx, "fy": intrinsics.fy,
                    "cx": intrinsics.cx, "cy": intrinsics.cy,
                },
            },
        )

    def run_sync(
        self,
        *,
        video_path: Path,
        poses_path: Path,
        intrinsics: Intrinsics,
        z0: float,
    ) -> FloorPointCloud:
        """sync wrapper for build_pipeline."""
        return asyncio.run(
            self.run(
                video_path=video_path,
                poses_path=poses_path,
                intrinsics=intrinsics,
                z0=z0,
            )
        )

    def _back_project(
        self,
        *,
        xs: np.ndarray,
        ys: np.ndarray,
        pose: np.ndarray,
        intrinsics: Intrinsics,
        z0: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """floor pixel → world (x, y) array + z. Sprint 22 adaptive_buffer 와 동일 convention.

        ARKit camera frame: x-right, y-up, -z forward. 이미지 좌표 y-down.
        D = diag(1, -1, -1) 적용으로 ray = ((x-cx)/fx, -(y-cy)/fy, -1).
        """
        rotation = pose[:3, :3]
        t = pose[:3, 3]

        rx = (xs - intrinsics.cx) / intrinsics.fx
        ry = -(ys - intrinsics.cy) / intrinsics.fy
        rz = -np.ones(len(xs), dtype=np.float64)
        rays_cam = np.stack([rx, ry, rz], axis=1)
        rays_world = rays_cam @ rotation.T

        dz = rays_world[:, 2]
        downward = dz < -_MIN_DOWNWARD_DZ
        if not downward.any():
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )

        lam = (z0 - t[2]) / dz[downward]
        rays_f = rays_world[downward]
        points = t + lam[:, None] * rays_f

        z_ok = np.abs(points[:, 2] - z0) < self._params.height_tolerance_m
        return points[z_ok, :2], points[z_ok, 2]
