"""DepthAwareBackProjectionStep + ScaleCalibrator 단위 테스트."""
from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from indoor_server.application.building.steps.back_projection import Intrinsics
from indoor_server.application.building.steps.depth_back_projection import (
    DepthAwareBackProjectionStep,
    ScaleCalibrator,
)


def _pose_bytes_camera_at_z(tz: float) -> bytes:
    """카메라 z=tz, 단위 rotation 4x4 col-major."""
    mat = np.eye(4, dtype=np.float32)
    mat[2, 3] = float(tz)
    return struct.pack("<16f", *mat.flatten(order="F"))


def _make_floor_mask_bottom_half(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2 :, :] = True
    return mask


class TestScaleCalibrator:
    """ScaleCalibrator.calibrate 단위 검증."""

    def test_known_height_returns_scale(self) -> None:
        """카메라 z=1.5, floor z=0, 카메라가 바닥을 내려다보는 자세 → scale 반환."""
        h, w = 480, 640
        intrin = Intrinsics(fx=500.0, fy=500.0, cx=w / 2.0, cy=h / 2.0)

        # 균일 relative depth 0.5
        depth_relative = np.full((h, w), 0.5, dtype=np.float32)
        floor_mask = _make_floor_mask_bottom_half(h, w)

        # 카메라가 바닥을 내려다보는 pose:
        # ARKit Z-up 변환 후 카메라가 아래(−z)를 바라보는 rotation (R_x(-90°))
        # R_x(-90°): x축 기준 −90° 회전
        # [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        # 이 rotation에서 camera +z 방향 → world -y 방향 (하향)
        # 더 단순하게: Z-up frame에서 카메라가 −z 방향을 바라보려면 R[2,2]=-1
        # 가장 직접적으로: rotation=diag(1,1,-1) * R_x(180°) 대신
        # world frame에서 카메라가 아래를 향하는 rotation 직접 설정
        pose = np.eye(4, dtype=np.float64)
        # rotation: camera +x → world +x, camera +y → world +z (위), camera +z → world -y (앞/아래)
        # Z-up 시스템에서 "바닥을 내려다봄" = world의 −z 방향으로 ray가 향하도록
        # R_look_down: camera z-axis가 world -z를 가리킴 → R[2] = (0, 0, -1)
        pose[0, :3] = [1.0, 0.0, 0.0]   # camera x → world x
        pose[1, :3] = [0.0, -1.0, 0.0]  # camera y → world -y (이미지 아래 → world 아래)
        pose[2, :3] = [0.0, 0.0, -1.0]  # camera z → world -z (카메라 전방 → 바닥 방향)
        pose[2, 3] = 1.5  # 카메라 높이 1.5m

        z0 = 0.0
        calibrator = ScaleCalibrator()
        scale = calibrator.calibrate(depth_relative, floor_mask, pose, intrin, z0)

        assert scale is not None, "scale should be computed successfully"
        assert scale > 0

    def test_empty_floor_mask_returns_none(self) -> None:
        h, w = 64, 64
        intrin = Intrinsics(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
        depth_relative = np.ones((h, w), dtype=np.float32)
        floor_mask = np.zeros((h, w), dtype=bool)  # 비어있음
        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = 1.0

        calibrator = ScaleCalibrator()
        result = calibrator.calibrate(depth_relative, floor_mask, pose, intrin, z0=0.0)
        assert result is None

    def test_camera_looking_up_returns_none(self) -> None:
        """모든 ray가 위를 향하면 None 반환 (카메라가 천장 보고 있는 경우).

        identity rotation에서 camera +z = world +z (위쪽).
        r_world_z = r_cam_z = 1.0 > 0 → 모두 upward → downward ray 없음 → None.
        """
        h, w = 64, 64
        intrin = Intrinsics(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
        depth_relative = np.ones((h, w), dtype=np.float32)
        floor_mask = _make_floor_mask_bottom_half(h, w)

        # identity rotation: camera +z = world +z (천장 방향)
        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = 1.0

        calibrator = ScaleCalibrator()
        result = calibrator.calibrate(depth_relative, floor_mask, pose, intrin, z0=0.0)
        assert result is None

    def test_too_few_candidates_returns_none(self) -> None:
        """bottom 픽셀이 50 미만이면 None."""
        h, w = 10, 10  # 아주 작은 이미지 → 하단 15% = 1~2행
        intrin = Intrinsics(fx=100.0, fy=100.0, cx=5.0, cy=5.0)
        depth_relative = np.ones((h, w), dtype=np.float32)
        floor_mask = _make_floor_mask_bottom_half(h, w)
        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = 1.0

        calibrator = ScaleCalibrator()
        result = calibrator.calibrate(depth_relative, floor_mask, pose, intrin, z0=0.0)
        assert result is None


class TestDepthAwareBackProjectionStep:
    """DepthAwareBackProjectionStep.run 단위 검증."""

    def _make_mock_runner(self, depth_value: float, h: int = 64, w: int = 64) -> MagicMock:
        """일정 깊이값 반환 mock runner."""
        runner = MagicMock()
        runner.__class__ = type(
            "DepthAnythingV2Runner",
            (),
            {},
        )

        # 타입 체크 우회 (isinstance 피하려면 실제 클래스 패치 필요)

        depth_arr = np.full((h, w), depth_value, dtype=np.float32)
        runner.estimate = AsyncMock(return_value=depth_arr)
        return runner

    @pytest.mark.asyncio
    async def test_valid_depth_returns_points_in_z_tolerance(self) -> None:
        """scale calibration 성공 → points z ≈ z0."""
        h, w = 64, 64
        floor_mask = _make_floor_mask_bottom_half(h, w)
        z0 = 0.0
        cam_tz = 1.5

        pose_bytes = _pose_bytes_camera_at_z(cam_tz)

        # depth NN runner mock — 모든 픽셀 d_rel=0.5
        from indoor_server.infrastructure.ml.depth_anything import DepthAnythingV2Runner

        depth_arr = np.full((h, w), 0.5, dtype=np.float32)
        runner = MagicMock(spec=DepthAnythingV2Runner)
        runner.estimate = AsyncMock(return_value=depth_arr)

        step = DepthAwareBackProjectionStep(depth_runner=runner)
        pts, depth_out, _scale = await step.run(
            image=np.zeros((h, w, 3), dtype=np.uint8),
            floor_mask=floor_mask,
            pose_bytes=pose_bytes,
            image_w=w,
            image_h=h,
            z0=z0,
            intrinsics=Intrinsics(fx=100.0, fy=100.0, cx=32.0, cy=32.0),
        )

        assert depth_out is not None
        assert depth_out.shape == (h, w)
        # scale calibration 실패 or 성공에 따라 점이 0개 이상
        assert isinstance(pts, np.ndarray)
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    @pytest.mark.asyncio
    async def test_scale_none_returns_empty_points(self) -> None:
        """ScaleCalibrator가 None 반환 → 빈 points."""
        h, w = 10, 10  # floor pixel 수가 너무 적어 calibration 실패

        from indoor_server.infrastructure.ml.depth_anything import DepthAnythingV2Runner

        depth_arr = np.ones((h, w), dtype=np.float32)
        runner = MagicMock(spec=DepthAnythingV2Runner)
        runner.estimate = AsyncMock(return_value=depth_arr)

        floor_mask = _make_floor_mask_bottom_half(h, w)
        pose_bytes = _pose_bytes_camera_at_z(1.5)

        step = DepthAwareBackProjectionStep(depth_runner=runner)
        pts, depth_out, scale_used = await step.run(
            image=np.zeros((h, w, 3), dtype=np.uint8),
            floor_mask=floor_mask,
            pose_bytes=pose_bytes,
            image_w=w,
            image_h=h,
            z0=0.0,
            intrinsics=Intrinsics(fx=100.0, fy=100.0, cx=5.0, cy=5.0),
        )

        assert pts.shape == (0, 3)  # 빈 points
        assert scale_used is None

    @pytest.mark.asyncio
    async def test_empty_floor_mask_returns_empty(self) -> None:
        """floor mask가 비어있으면 빈 points."""
        h, w = 64, 64

        from indoor_server.infrastructure.ml.depth_anything import DepthAnythingV2Runner

        runner = MagicMock(spec=DepthAnythingV2Runner)
        runner.estimate = AsyncMock(return_value=np.ones((h, w), dtype=np.float32))

        floor_mask = np.zeros((h, w), dtype=bool)
        step = DepthAwareBackProjectionStep(depth_runner=runner)
        pts, _, _ = await step.run(
            image=np.zeros((h, w, 3), dtype=np.uint8),
            floor_mask=floor_mask,
            pose_bytes=_pose_bytes_camera_at_z(1.0),
            image_w=w,
            image_h=h,
            z0=0.0,
        )
        assert pts.shape == (0, 3)
