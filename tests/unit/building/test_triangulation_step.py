"""TriangulationStep 단위 테스트.

Case A: 합성 2-view happy-path — 알려진 3D 점을 정확히 복원하는지 확인.
Case B: outlier 필터 — reprojection error 필터 + baseline 필터 동작 확인.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import numpy as np
import pytest

from indoor_server.application.building.steps.back_projection import Intrinsics
from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks
from indoor_server.application.building.steps.triangulation import (
    TriangulationStep,
    _build_projection_matrix,
    _project_points,
)
from indoor_server.infrastructure.ml.superpoint_lightglue import MatchedPair

# ── 합성 helper ───────────────────────────────────────────────────────────────

_SERVER_FROM_ARKIT = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def _make_pose_bytes(rot: np.ndarray, t: np.ndarray) -> bytes:
    """(3,3) rotation + (3,) t → ARKit convention 4x4 col-major float32 bytes."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot.astype(np.float32)
    mat[:3, 3] = t.astype(np.float32)
    return struct.pack("<16f", *mat.flatten(order="F"))


def _make_keyframe_mask(
    scan_id: UUID,
    seq: int,
    pose_bytes: bytes,
    tx: float,
    ty: float,
    tz: float,
    h: int = 480,
    w: int = 640,
    floor_all: bool = True,
) -> KeyframeMasks:
    """합성 KeyframeMasks. floor_all=True면 전체 픽셀이 floor."""
    floor_mask = np.ones((h, w), dtype=bool) if floor_all else np.zeros((h, w), dtype=bool)
    stair_mask = np.zeros((h, w), dtype=bool)
    return KeyframeMasks(
        scan_id=scan_id,
        seq=seq,
        floor_mask=floor_mask,
        stair_mask=stair_mask,
        tx=tx,
        ty=ty,
        tz=tz,
        pose_matrix=pose_bytes,
    )


def _project_point_arkit(
    x_world: np.ndarray, pose: np.ndarray, intrin: Intrinsics
) -> tuple[float, float]:
    """ARKit world frame 3D 점 x_world → pixel (u, v).

    projection matrix는 _build_projection_matrix와 동일한 규칙 사용.
    """
    proj = _build_projection_matrix(pose, intrin)
    x_h = np.array([x_world[0], x_world[1], x_world[2], 1.0], dtype=np.float64)
    uvw = proj @ x_h
    u = float(uvw[0] / uvw[2])
    v = float(uvw[1] / uvw[2])
    return u, v


def _make_mock_runner(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    matches: np.ndarray,
    scores: np.ndarray,
    input_size: int = 384,
) -> Any:
    """SuperPointLightGlueRunner mock — match 호출 시 고정 MatchedPair 반환."""
    mp = MatchedPair(
        kpts_a=kpts_a,
        kpts_b=kpts_b,
        matches=matches,
        scores=scores,
        size_a=(input_size, input_size),
        size_b=(input_size, input_size),
    )
    runner = MagicMock()
    runner.match = AsyncMock(return_value=mp)
    runner.input_size = input_size
    return runner


# ── Case A: happy-path triangulation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_triangulation_recovers_known_3d_point(tmp_path: Path) -> None:
    """합성 2-view — ARKit raw point가 server Z-up 좌표로 복원되는지 확인."""
    scan_id = UUID("00000000-0000-0000-0000-000000000001")
    input_size = 384

    # cam0: world origin, cam1: baseline 0.5m 이동
    rot = np.eye(3, dtype=np.float64)
    t0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.array([0.5, 0.0, 0.0], dtype=np.float64)

    pose0 = np.eye(4, dtype=np.float64)
    pose0[:3, :3] = rot
    pose0[:3, 3] = t0
    pose1 = np.eye(4, dtype=np.float64)
    pose1[:3, :3] = rot
    pose1[:3, 3] = t1

    pose_bytes0 = _make_pose_bytes(rot, t0)
    pose_bytes1 = _make_pose_bytes(rot, t1)

    # 합성 3D 점
    x_world = np.array([0.1, 0.0, 2.0], dtype=np.float64)

    h, w = 480, 640
    intrin = Intrinsics(fx=500.0, fy=500.0, cx=w / 2.0, cy=h / 2.0)

    u0, v0 = _project_point_arkit(x_world, pose0, intrin)
    u1, v1 = _project_point_arkit(x_world, pose1, intrin)

    # model resolution → 원본 해상도 역변환 (kpt 좌표 = model_res 기준)
    kpt_u0_model = u0 * (input_size / w)
    kpt_v0_model = v0 * (input_size / h)
    kpt_u1_model = u1 * (input_size / w)
    kpt_v1_model = v1 * (input_size / h)

    kpts_a = np.array([[kpt_u0_model, kpt_v0_model]], dtype=np.int64)
    kpts_b = np.array([[kpt_u1_model, kpt_v1_model]], dtype=np.int64)
    matches = np.array([[0, 0]], dtype=np.int64)
    scores = np.array([0.99], dtype=np.float32)

    runner = _make_mock_runner(kpts_a, kpts_b, matches, scores, input_size=input_size)

    mask0 = _make_keyframe_mask(
        scan_id, 0, pose_bytes0, tx=float(t0[0]), ty=float(t0[1]), tz=float(t0[2]), h=h, w=w
    )
    mask1 = _make_keyframe_mask(
        scan_id, 1, pose_bytes1, tx=float(t1[0]), ty=float(t1[1]), tz=float(t1[2]), h=h, w=w
    )

    # 실제로 읽을 수 있는 JPG 생성 (8x8 흰색 이미지)
    scan_dir = tmp_path / "scans" / str(scan_id) / "keyframes"
    scan_dir.mkdir(parents=True)
    import cv2
    dummy_img = np.ones((h, w, 3), dtype=np.uint8) * 200
    cv2.imwrite(str(scan_dir / "000000.jpg"), dummy_img)
    cv2.imwrite(str(scan_dir / "000001.jpg"), dummy_img)

    step = TriangulationStep(
        sp_lg_runner=runner,
        window=1,
        min_match_score=0.80,
        max_pair_matches=64,
        min_baseline_m=0.05,
        max_reprojection_error_px=5.0,  # 합성이라 약간 여유
        max_distance_m=15.0,
        z_tolerance_m=3.0,  # 검증 환경 완화
        floor_mask_dilate_px=0,  # floor_mask 전체 True이므로 불필요
    )

    intrinsics_by_frame = [intrin, intrin]
    z0 = 0.0

    cloud = await step.run(
        masks=[mask0, mask1],
        storage_root=tmp_path,
        intrinsics_by_frame=intrinsics_by_frame,
        z0=z0,
    )

    # frame 0에 점이 귀속되어야 함
    pts0 = cloud.points_per_frame[0]
    assert pts0.shape[0] >= 1, (
        f"expected ≥1 point in frame 0, got {pts0.shape[0]}. total_kept={cloud.total_kept}"
    )

    expected_server = (_SERVER_FROM_ARKIT @ np.array([*x_world, 1.0]))[:3]
    dists = np.linalg.norm(pts0 - expected_server, axis=1)
    min_dist = float(np.min(dists))
    best_pt = pts0[int(np.argmin(dists))]
    assert min_dist < 0.01, (
        f"min_dist={min_dist:.4f}m (expected < 0.01m). recovered={best_pt}"
    )


# ── Case B: outlier 필터 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_triangulation_filters_outliers(tmp_path: Path) -> None:
    """outlier 필터 동작 검증.

    match 3종 주입:
        1. 정상 match (kept=1)
        2. reprojection error 큰 match (kp를 20px 오프셋) → stage 2 drop
        3. baseline 2cm pair (별도 카메라 pair) → baseline 필터로 pair 자체 skip

    기대: cloud.total_kept == 1
    """
    scan_id = UUID("00000000-0000-0000-0000-000000000002")
    input_size = 384
    h, w = 480, 640
    intrin = Intrinsics(fx=500.0, fy=500.0, cx=w / 2.0, cy=h / 2.0)

    # 세 카메라: cam0, cam1 (50cm baseline), cam2 (2cm baseline from cam1)
    rot = np.eye(3, dtype=np.float64)
    t0 = np.array([0.0, 0.0, 0.0])
    t1 = np.array([0.5, 0.0, 0.0])   # 50cm baseline
    t2 = np.array([0.52, 0.0, 0.0])  # 2cm from cam1

    pose0 = np.eye(4)
    pose0[:3, 3] = t0
    pose1 = np.eye(4)
    pose1[:3, 3] = t1
    pose2 = np.eye(4)
    pose2[:3, 3] = t2

    pose_bytes0 = _make_pose_bytes(rot, t0)
    pose_bytes1 = _make_pose_bytes(rot, t1)
    pose_bytes2 = _make_pose_bytes(rot, t2)

    mask0 = _make_keyframe_mask(
        scan_id, 0, pose_bytes0, tx=float(t0[0]), ty=float(t0[1]), tz=float(t0[2]), h=h, w=w
    )
    mask1 = _make_keyframe_mask(
        scan_id, 1, pose_bytes1, tx=float(t1[0]), ty=float(t1[1]), tz=float(t1[2]), h=h, w=w
    )
    mask2 = _make_keyframe_mask(
        scan_id, 2, pose_bytes2, tx=float(t2[0]), ty=float(t2[1]), tz=float(t2[2]), h=h, w=w
    )

    # 실제로 읽을 수 있는 JPG 생성
    scan_dir = tmp_path / "scans" / str(scan_id) / "keyframes"
    scan_dir.mkdir(parents=True)
    import cv2 as _cv2
    dummy_img = np.ones((h, w, 3), dtype=np.uint8) * 200
    for seq in range(3):
        _cv2.imwrite(str(scan_dir / f"{seq:06d}.jpg"), dummy_img)

    # 정상 3D 점 x_good
    x_good = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    u0_good, v0_good = _project_point_arkit(x_good, pose0, intrin)
    u1_good, v1_good = _project_point_arkit(x_good, pose1, intrin)

    # 잘못된 3D 점 (kp를 20px offset — pose 불일치)
    u0_bad = u0_good + 20.0
    v0_bad = v0_good
    u1_bad = u1_good
    v1_bad = v1_good + 20.0

    # match_call 카운터 관리
    call_count = 0

    async def _mock_match(img_a: Any, img_b: Any) -> MatchedPair:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # pair (0,1): 정상 match + 나쁜 match (reprojection error > 3px)
            kpts_a = np.array([
                [u0_good * (input_size / w), v0_good * (input_size / h)],
                [u0_bad * (input_size / w), v0_bad * (input_size / h)],
            ], dtype=np.int64)
            kpts_b = np.array([
                [u1_good * (input_size / w), v1_good * (input_size / h)],
                [u1_bad * (input_size / w), v1_bad * (input_size / h)],
            ], dtype=np.int64)
            matches = np.array([[0, 0], [1, 1]], dtype=np.int64)
            scores = np.array([0.99, 0.90], dtype=np.float32)
        else:
            kpts_a = np.empty((0, 2), dtype=np.int64)
            kpts_b = np.empty((0, 2), dtype=np.int64)
            matches = np.empty((0, 2), dtype=np.int64)
            scores = np.empty((0,), dtype=np.float32)

        return MatchedPair(
            kpts_a=kpts_a,
            kpts_b=kpts_b,
            matches=matches,
            scores=scores,
            size_a=(input_size, input_size),
            size_b=(input_size, input_size),
        )

    runner = MagicMock()
    runner.match = _mock_match
    runner.input_size = input_size

    step = TriangulationStep(
        sp_lg_runner=runner,
        window=1,
        min_match_score=0.80,
        max_pair_matches=64,
        min_baseline_m=0.05,        # 5cm — pair(1,2) 2cm는 skip
        max_reprojection_error_px=3.0,  # 엄격 — bad match drop
        max_distance_m=15.0,
        z_tolerance_m=3.0,          # 완화 (합성)
        floor_mask_dilate_px=0,
    )

    intrinsics_by_frame = [intrin, intrin, intrin]
    z0 = 0.0

    cloud = await step.run(
        masks=[mask0, mask1, mask2],
        storage_root=tmp_path,
        intrinsics_by_frame=intrinsics_by_frame,
        z0=z0,
    )

    # 정확히 1개만 kept (good match)
    assert cloud.total_kept == 1, f"expected total_kept=1, got {cloud.total_kept}"

    # pair (1,2)는 baseline < 5cm → pair_stats에 skip_reason 기록
    baseline_small_pairs = [
        s for s in cloud.pair_stats
        if s.get("skip_reason") == "baseline_too_small"
    ]
    assert len(baseline_small_pairs) >= 1, (
        f"expected ≥1 baseline_too_small pair, got stats={cloud.pair_stats}"
    )
    small_pair = baseline_small_pairs[0]
    assert float(small_pair["baseline_m"]) < 0.05


# ── 생성자 projection matrix 검증 ────────────────────────────────────────────


def test_build_projection_matrix_roundtrip() -> None:
    """_build_projection_matrix + _project_points가 알려진 3D 점을 올바른 픽셀로 투영."""
    intrin = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([0.0, 0.0, 0.0])

    x_pt = np.array([[0.1, 0.0, 2.0]], dtype=np.float64)  # (1, 3)
    proj = _build_projection_matrix(pose, intrin)

    px = _project_points(x_pt, proj)
    u, v = _project_point_arkit(x_pt[0], pose, intrin)

    assert abs(px[0, 0] - u) < 0.01, f"u mismatch: {px[0, 0]:.4f} vs {u:.4f}"
    assert abs(px[0, 1] - v) < 0.01, f"v mismatch: {px[0, 1]:.4f} vs {v:.4f}"
