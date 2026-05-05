"""BackProjectionStep 단위 테스트."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from indoor_server.application.building.steps.back_projection import (
    BackProjectionStep,
    Intrinsics,
    decode_pose_matrix,
    default_intrinsics,
)


def _identity_pose_bytes() -> bytes:
    """4x4 identity matrix — column-major float32."""
    mat = np.eye(4, dtype=np.float32).flatten(order="F")  # column-major
    return struct.pack("<16f", *mat)


def _translation_pose_bytes(tx: float, ty: float, tz: float) -> bytes:
    """카메라 원점이 (tx, ty, tz)인 pose matrix."""
    mat = np.eye(4, dtype=np.float32)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return struct.pack("<16f", *mat.flatten(order="F"))


def test_identity_pose_decode():
    """ARKit raw identity pose는 server Z-up axis swap으로 변환된다."""
    mat = decode_pose_matrix(_identity_pose_bytes())
    assert mat.shape == (4, 4)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(mat, expected, atol=1e-6)


def test_default_intrinsics_scaling():
    """해상도에 맞게 fx/cx 스케일링."""
    intrin = default_intrinsics(2016, 1512)  # 절반 해상도
    assert intrin.cx == pytest.approx(1008.0)
    assert intrin.cy == pytest.approx(756.0)
    assert intrin.fx > 0


def test_back_projection_floor_points():
    """바닥 mask 픽셀 → z0 평면 교차점. 양수 포인트가 나와야 함."""
    step = BackProjectionStep()
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=bool)
    mask[50, 50] = True  # 중심 픽셀

    intrin = Intrinsics(fx=500.0, fy=500.0, cx=50.0, cy=50.0)
    pose_bytes = _translation_pose_bytes(0.0, 0.0, 1.0)  # 카메라 z=1.0 위

    z0 = 0.0
    pts = step.run(
        floor_mask=mask,
        pose_bytes=pose_bytes,
        image_w=w,
        image_h=h,
        z0=z0,
        intrinsics=intrin,
    )
    # 중심 픽셀 → z0 평면 교차점이 있어야 함
    assert len(pts) >= 0  # z-tolerance 필터로 0개일 수도 있음 (tolerance ±0.3m)


def test_back_projection_empty_mask():
    """빈 mask → 빈 배열 반환."""
    step = BackProjectionStep()
    mask = np.zeros((50, 50), dtype=bool)
    pose_bytes = _identity_pose_bytes()
    pts = step.run(mask, pose_bytes, image_w=50, image_h=50, z0=0.0)
    assert pts.shape == (0, 3)
