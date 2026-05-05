"""Back-projection step — floor mask 픽셀 → world frame 3D 좌표.

LiDAR 없음 (R-2-b). pose_matrix + 고정 intrinsics + floor plane z=z0 교차점.
"""
from __future__ import annotations

import struct

import numpy as np
from pydantic import BaseModel, ConfigDict

# ASSUMPTION A-2: iPhone 15 Pro 1x 근사 intrinsics (focal_length ~1500px @ 4032x3024)
# 이미지 해상도에 맞게 scaling됨
_IPHONE15_PRO_FX_BASE = 1500.0
_IPHONE15_PRO_BASE_W = 4032
_IPHONE15_PRO_BASE_H = 3024

# ASSUMPTION A-1: z-tolerance ±30cm 이내 floor mask만 유효
_Z_TOLERANCE = 0.30


class Intrinsics(BaseModel):
    """카메라 내부 파라미터 (픽셀 단위)."""

    model_config = ConfigDict(frozen=True)

    fx: float
    fy: float
    cx: float
    cy: float


def default_intrinsics(image_w: int, image_h: int) -> Intrinsics:
    """ASSUMPTION A-2: 이미지 해상도에 맞게 scaling된 iPhone 15 Pro 상수."""
    scale = image_w / _IPHONE15_PRO_BASE_W
    fx = _IPHONE15_PRO_FX_BASE * scale
    fy = _IPHONE15_PRO_FX_BASE * scale
    cx = image_w / 2.0
    cy = image_h / 2.0
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def decode_pose_matrix(pose_bytes: bytes) -> np.ndarray:
    """BLOB (16 float32 little-endian column-major) → 4x4 world-from-camera.

    ARKit world frame은 Y-up (X-right, Y-up, Z-back) 인데 서버 파이프라인은
    floor 평면을 (x, y), height를 z (Z-up) 로 가정한다. iOS sidecar는 ARKit raw
    pose를 그대로 저장하므로 여기서 axis swap을 적용:
        server_world_zup = S @ arkit_world_yup
    매핑: new_x = X, new_y = -Z (ARKit -Z = camera forward = 사용자가 걷는 방향),
          new_z = Y (height).
    """
    values = struct.unpack_from("<16f", pose_bytes)
    # iOS 저장 형식: column-major → reshape order='F'
    arkit_pose = np.array(values, dtype=np.float64).reshape(4, 4, order="F")
    # Y-up → Z-up axis swap (world frame change)
    s = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return s @ arkit_pose


_MAX_PROJ_DISTANCE_M = 15.0  # vertex 기준 거리 cap


def project_pixels_to_floor(
    pixel_coords: list[tuple[int, int]],
    pose_bytes: bytes,
    image_w: int,
    image_h: int,
    z0: float,
    intrinsics: Intrinsics | None = None,
    max_distance_m: float = _MAX_PROJ_DISTANCE_M,
) -> list[tuple[float, float] | None]:
    """
    pixel_coords 목록을 floor plane z=z0로 back-project → world (x, y) 목록.

    입력:
        pixel_coords: [(px, py)] image 좌표 (x, y)
        pose_bytes: 64 bytes BLOB (4x4 float32 col-major)
        image_w/h: 이미지 해상도
        z0: floor plane z height
        intrinsics: None이면 default_intrinsics 사용
        max_distance_m: 카메라 origin 기준 최대 거리 cap

    반환:
        list[tuple[float, float] | None] — 길이 len(pixel_coords).
        ray가 유효하지 않거나 거리 초과 시 None.
    """
    intrin = intrinsics or default_intrinsics(image_w, image_h)
    pose = decode_pose_matrix(pose_bytes)
    rotation = pose[:3, :3]
    t = pose[:3, 3]

    results: list[tuple[float, float] | None] = []
    for px, py in pixel_coords:
        rx = (px - intrin.cx) / intrin.fx
        ry = (py - intrin.cy) / intrin.fy
        rz = 1.0
        ray_cam = np.array([rx, ry, rz], dtype=np.float64)
        ray_world = rotation @ ray_cam
        dz = ray_world[2]
        if abs(dz) < 1e-6:
            results.append(None)
            continue
        lam = (z0 - t[2]) / dz
        if lam <= 0:
            results.append(None)
            continue
        world_pt = t + lam * ray_world
        dist = float(np.linalg.norm(world_pt - t))
        if dist > max_distance_m:
            results.append(None)
            continue
        results.append((float(world_pt[0]), float(world_pt[1])))
    return results


class BackProjectionStep:
    """floor mask 픽셀 → world frame 3D 포인트 (floor plane z=z0 교차)."""

    def run(
        self,
        floor_mask: np.ndarray,
        pose_bytes: bytes,
        image_w: int,
        image_h: int,
        z0: float,
        intrinsics: Intrinsics | None = None,
    ) -> np.ndarray:
        """
        floor_mask: (H, W) bool
        pose_bytes: 64 bytes BLOB (4x4 float32 col-major)
        반환: (N, 3) float64 world 좌표 (z ≈ z0 ± 0.3m 필터 후)
        """
        intrin = intrinsics or default_intrinsics(image_w, image_h)
        pose = decode_pose_matrix(pose_bytes)
        return self._back_project(floor_mask, pose, intrin, z0)

    def _back_project(
        self,
        mask: np.ndarray,
        pose: np.ndarray,
        intrin: Intrinsics,
        z0: float,
    ) -> np.ndarray:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.zeros((0, 3), dtype=np.float64)

        # 카메라 좌표계에서 ray 방향 (z=1 기준 normalized)
        rx = (xs - intrin.cx) / intrin.fx
        ry = (ys - intrin.cy) / intrin.fy
        rz = np.ones(len(xs), dtype=np.float64)

        rays_cam = np.stack([rx, ry, rz], axis=1)  # (N, 3)

        # world frame으로 변환
        rotation = pose[:3, :3]  # 3x3 rotation matrix
        t = pose[:3, 3]          # camera origin in world

        rays_world = rays_cam @ rotation.T  # (N, 3) world direction

        # floor plane z=z0 교차: t + lambda * d = [x, y, z0]
        # lambda = (z0 - t[2]) / d[2]
        dz = rays_world[:, 2]
        valid = np.abs(dz) > 1e-6  # 수평 ray 제외
        if valid.sum() == 0:
            return np.zeros((0, 3), dtype=np.float64)

        lam = (z0 - t[2]) / dz[valid]
        forward = lam > 0  # 카메라 전방만

        lam = lam[forward]
        rays_valid = rays_world[valid][forward]

        points = t + lam[:, None] * rays_valid  # (N, 3)

        # z-tolerance 필터 (A-1)
        z_ok = np.abs(points[:, 2] - z0) < _Z_TOLERANCE
        result: np.ndarray = points[z_ok]
        return result
