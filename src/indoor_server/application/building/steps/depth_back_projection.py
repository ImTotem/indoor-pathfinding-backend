"""Depth-aware back-projection step — floor mask + NN depth → world 3D.

기존 BackProjectionStep과 동일한 출력 shape (N, 3) float64.
ScaleCalibrator가 relative depth → metric scale을 ARKit pose 기반으로 산출.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from indoor_server.application.building.steps.back_projection import (
    Intrinsics,
    decode_pose_matrix,
    default_intrinsics,
)

if TYPE_CHECKING:
    from indoor_server.infrastructure.ml.depth_anything import DepthAnythingV2Runner

logger = logging.getLogger(__name__)

_Z_TOLERANCE_DEFAULT = 0.30  # ±30cm floor plane tolerance
_MAX_DISTANCE_DEFAULT = 15.0  # 카메라 origin 기준 최대 거리 (m)
_SCALE_CV_WARN_THRESHOLD = 0.5  # σ/μ > 0.5 → warning


class ScaleCalibrator:
    """relative inverse depth → metric scale factor 산출.

    ASSUMPTION A-D-1: scale-only (b=0). metric_depth = scale / d_rel.
    floor plane z=z0 + ARKit pose의 카메라 높이로 scale 산출.
    """

    def __init__(self, use_transposed_rotation: bool = False) -> None:
        # Sprint 20: stored pose가 view matrix (cam_T_world) 성격이라 rotation을 R.T로
        # 적용해야 world ray가 된다. multiview scale 경로에서만 True로 주입.
        self._use_rt = use_transposed_rotation

    def calibrate(
        self,
        depth_relative: np.ndarray,
        floor_mask: np.ndarray,
        pose: np.ndarray,
        intrin: Intrinsics,
        z0: float,
        sample_strategy: str = "bottom_floor_median",
    ) -> float | None:
        """metric scale factor 반환. 산출 실패 시 None.

        depth_relative: (H, W) float32 relative inverse depth (큰 값 = 가까움)
        floor_mask: (H, W) bool
        pose: 4x4 world-from-camera float64
        intrin: 카메라 내부 파라미터
        z0: floor plane world z
        """
        return self._bottom_floor_median(depth_relative, floor_mask, pose, intrin, z0)

    def _bottom_floor_median(
        self,
        depth_relative: np.ndarray,
        floor_mask: np.ndarray,
        pose: np.ndarray,
        intrin: Intrinsics,
        z0: float,
    ) -> float | None:
        h, w = depth_relative.shape
        cam_tz = float(pose[2, 3])

        # floor mask True 픽셀 중 y >= 0.85*H (이미지 하단 15%)
        ys, xs = np.where(floor_mask)
        if len(ys) == 0:
            logger.warning("scale_calibrator: floor_mask is empty → skip")
            return None

        bottom_thresh = int(h * 0.85)
        keep = ys >= bottom_thresh
        ys = ys[keep]
        xs = xs[keep]

        if len(ys) < 50:
            logger.warning(
                "scale_calibrator: bottom floor candidates < 50 (%d) → skip", len(ys)
            )
            return None

        # camera ray 방향 (OpenCV; z=+1 forward)
        rx = (xs.astype(np.float64) - intrin.cx) / intrin.fx
        ry = (ys.astype(np.float64) - intrin.cy) / intrin.fy
        rz = np.ones(len(xs), dtype=np.float64)
        rays_cam = np.stack([rx, ry, rz], axis=1)  # (N, 3)

        # world frame ray
        rotation = pose[:3, :3]
        if self._use_rt:
            # stored pose가 cam_T_world → R.T가 world_from_camera
            rays_world = rays_cam @ rotation  # = (R.T @ rays_cam[i]).T per row
        else:
            rays_world = rays_cam @ rotation.T  # (N, 3), legacy Sprint 19 경로

        # r_world_z < -0.05 (충분히 아래 방향 ray만)
        rw_z = rays_world[:, 2]
        valid = rw_z < -0.05
        if valid.sum() == 0:
            logger.warning("scale_calibrator: no downward rays (camera looking up?) → skip")
            return None

        ys_v = ys[valid]
        xs_v = xs[valid]
        rw_z_v = rw_z[valid]
        rays_cam_v = rays_cam[valid]

        # ray-plane 교차로 기하적 metric distance
        lam = (z0 - cam_tz) / rw_z_v  # λ = (z0 - cam_z) / r_world_z
        # λ > 0 이어야 전방 교차
        fwd = lam > 0
        if fwd.sum() < 50:
            logger.warning(
                "scale_calibrator: forward intersection candidates < 50 → skip"
            )
            return None

        lam = lam[fwd]
        rays_cam_v = rays_cam_v[fwd]
        ys_v = ys_v[fwd]
        xs_v = xs_v[fwd]

        ray_len = np.linalg.norm(rays_cam_v, axis=1)  # ||r_cam||
        d_geom = lam * ray_len  # metric distance to floor

        # relative depth at those pixels
        d_rel = depth_relative[ys_v, xs_v].astype(np.float64)
        nonzero = d_rel > 1e-8
        if nonzero.sum() < 50:
            logger.warning("scale_calibrator: too few non-zero d_rel → skip")
            return None

        d_geom = d_geom[nonzero]
        d_rel = d_rel[nonzero]

        # candidate scale_i = d_geom_i * d_rel_i
        scale_candidates = d_geom * d_rel

        scale_med = float(np.median(scale_candidates))
        if not np.isfinite(scale_med) or scale_med <= 0:
            logger.warning("scale_calibrator: scale median is non-finite → skip")
            return None

        # 분포 경고
        scale_std = float(np.std(scale_candidates))
        cv = scale_std / (scale_med + 1e-8)
        if cv > _SCALE_CV_WARN_THRESHOLD:
            logger.warning(
                "scale_calibrator: high scale variance cv=%.2f (scale=%.3f ± %.3f) — "
                "calibration may be unreliable",
                cv,
                scale_med,
                scale_std,
            )

        logger.debug(
            "scale_calibrator: scale=%.4f candidates=%d cv=%.2f",
            scale_med,
            len(scale_candidates),
            cv,
        )
        return scale_med


class DepthAwareBackProjectionStep:
    """floor mask 픽셀 + depth NN → world frame 3D 포인트.

    기존 BackProjectionStep와 같은 출력 shape: (N, 3) float64.
    내부적으로 ScaleCalibrator로 metric scale 산출 → 픽셀별 3D 변환 → z-tolerance 필터.
    """

    def __init__(
        self,
        depth_runner: DepthAnythingV2Runner,
        scale_calibrator: ScaleCalibrator | None = None,
        z_tolerance_m: float = _Z_TOLERANCE_DEFAULT,
        max_distance_m: float = _MAX_DISTANCE_DEFAULT,
        use_transposed_rotation: bool = False,
    ) -> None:
        from indoor_server.infrastructure.ml.depth_anything import (
            DepthAnythingV2Runner as _Runner,
        )

        if not isinstance(depth_runner, _Runner):
            raise TypeError("depth_runner must be DepthAnythingV2Runner")
        self._runner = depth_runner
        self._calibrator = scale_calibrator or ScaleCalibrator()
        self._z_tolerance = z_tolerance_m
        self._max_distance = max_distance_m
        # Sprint 20: stored pose가 view matrix (cam_T_world)이라 rotation을 R.T로
        # 적용해야 world ray가 된다. multiview 경로에서만 True.
        self._use_rt = use_transposed_rotation

    async def run(
        self,
        image: np.ndarray,
        floor_mask: np.ndarray,
        pose_bytes: bytes,
        image_w: int,
        image_h: int,
        z0: float,
        intrinsics: Intrinsics | None = None,
        override_scale: float | None = None,
        precomputed_depth: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, float | None]:
        """
        Parameters:
            override_scale: None 이 아니면 ScaleCalibrator를 건너뛰고 이 값을 사용
                (Sprint 20 multi-view scale calibration이 주입한 최적값).
            precomputed_depth: 이미 계산된 relative depth. None이면 runner로 재계산.

        Returns:
            points: (N, 3) float64 world coords
            depth_map: (H, W) float32 relative depth (시각화용). 실패 시 None.
            used_scale: 실제 사용된 metric scale (override 또는 calibrator 결과). None이면 skip.
        """
        # 1. depth 추론 (precomputed 있으면 재사용)
        if precomputed_depth is not None:
            depth_relative = precomputed_depth
        else:
            try:
                depth_relative = await self._runner.estimate(image)
            except Exception as e:
                logger.error("depth_runner.estimate failed: %s", e)
                return np.zeros((0, 3), dtype=np.float64), None, None

        # 2. pose + intrinsics
        pose = decode_pose_matrix(pose_bytes)
        intrin = intrinsics or default_intrinsics(image_w, image_h)

        # 3. scale 결정: override 우선
        if override_scale is not None and override_scale > 0:
            scale: float | None = float(override_scale)
        else:
            scale = self._calibrator.calibrate(depth_relative, floor_mask, pose, intrin, z0)
        if scale is None:
            logger.warning("depth scale calibration failed; skip frame")
            return np.zeros((0, 3), dtype=np.float64), depth_relative, None

        # 4. floor 픽셀 추출
        ys, xs = np.where(floor_mask)
        if len(ys) == 0:
            return np.zeros((0, 3), dtype=np.float64), depth_relative, scale

        # 5. ray 방향 (OpenCV; z=+1 forward)
        rx = (xs.astype(np.float64) - intrin.cx) / intrin.fx
        ry = (ys.astype(np.float64) - intrin.cy) / intrin.fy
        rz = np.ones(len(xs), dtype=np.float64)
        rays_cam = np.stack([rx, ry, rz], axis=1)  # (N, 3)

        # 6. metric depth per pixel: scale / d_rel
        d_rel = depth_relative[ys, xs].astype(np.float64)
        nonzero = d_rel > 1e-8
        if nonzero.sum() == 0:
            return np.zeros((0, 3), dtype=np.float64), depth_relative, scale

        rays_cam = rays_cam[nonzero]
        d_rel = d_rel[nonzero]
        metric_depth = scale / d_rel  # (N,)

        # 7. 3D camera frame: rays_cam * (metric_depth / ray_len)[:, None]
        # z=1 기준 ray이므로 ||ray_cam|| 로 normalize한 후 metric_depth 곱.
        ray_len = np.linalg.norm(rays_cam, axis=1, keepdims=True)
        ray_len = np.where(ray_len < 1e-8, 1e-8, ray_len)
        points_cam = rays_cam * (metric_depth[:, None] / ray_len)

        # 8. world frame 변환
        rotation = pose[:3, :3]
        t = pose[:3, 3]
        if self._use_rt:
            points_world = (rotation.T @ points_cam.T).T + t
        else:
            points_world = (rotation @ points_cam.T).T + t  # (N, 3)

        # 9. z-tolerance 필터
        z_ok = np.abs(points_world[:, 2] - z0) < self._z_tolerance
        points_world = points_world[z_ok]

        # 10. distance cap
        if len(points_world) > 0:
            dist = np.linalg.norm(points_world - t, axis=1)
            points_world = points_world[dist < self._max_distance]

        return points_world.astype(np.float64), depth_relative, scale
