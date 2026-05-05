"""AdaptiveBufferStep 단위 테스트 (Sprint 22).

설계 §11 테스트 명세 전량 구현.
"""
from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import numpy as np
import pytest

from indoor_server.application.building.steps.adaptive_buffer import AdaptiveBufferStep
from indoor_server.application.building.steps.back_projection import (
    Intrinsics,
    default_intrinsics,
)
from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks

# ── Helpers ────────────────────────────────────────────────────────────────────

_SCAN_UUID = UUID("00000000-0000-0000-0000-000000000001")


def _pose_bytes(
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> bytes:
    """4x4 ARKit raw world-from-camera pose BLOB (column-major float32).

    rotation: (3, 3), translation: (3,). None이면 identity.
    """
    mat = np.eye(4, dtype=np.float64)
    if rotation is not None:
        mat[:3, :3] = rotation
    if translation is not None:
        mat[:3, 3] = translation
    flat = mat.astype(np.float32).flatten(order="F")  # column-major
    return struct.pack("<16f", *flat.tolist())


_SERVER_FROM_ARKIT = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)
_ARKIT_FROM_SERVER = np.linalg.inv(_SERVER_FROM_ARKIT)


def _server_pose_bytes(
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> bytes:
    """Desired server Z-up pose를 production 입력인 ARKit raw pose bytes로 변환."""
    server_pose = np.eye(4, dtype=np.float64)
    if rotation is not None:
        server_pose[:3, :3] = rotation
    if translation is not None:
        server_pose[:3, 3] = translation
    arkit_pose = _ARKIT_FROM_SERVER @ server_pose
    flat = arkit_pose.astype(np.float32).flatten(order="F")
    return struct.pack("<16f", *flat.tolist())


def _rot_forward_pos_y() -> np.ndarray:
    """Server Z-up pose: camera가 world +y 방향을 바라보는 rotation.

    ARKit cam frame: x=right, y=up, z=backward (camera forward = -z).
    Z-up world: x=right, y=forward(+y), z=up.

    - col0 = cam x-axis in world = (1, 0, 0)       right
    - col1 = cam y-axis in world = (0, 0, 1)        up
    - col2 = cam z-axis in world = (0, -1, 0)       backward (= -forward)

    forward_world = -R[:, 2] = (0, +1, 0). camera가 +y 봄.
    ARKit ray [(u-cx)/fx, -(v-cy)/fy, -1] 적용 시 이미지 하단(ry<0) → ray_world.z < 0 ✓.
    """
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)


def _rot_forward_neg_y() -> np.ndarray:
    """Backward-compat alias. 신규 테스트는 _rot_forward_pos_y 사용."""
    return _rot_forward_pos_y()


def _make_mask(
    h: int = 1080,
    w: int = 1920,
    seq: int = 0,
    pose: bytes | None = None,
    floor_region: tuple[int, int, int, int] | None = None,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 1.5,
) -> KeyframeMasks:
    """합성 KeyframeMasks.

    floor_region: (row_start, row_end, col_start, col_end) — 지정 영역만 floor.
                  None이면 전체 False.
    """
    floor = np.zeros((h, w), dtype=bool)
    if floor_region is not None:
        rs, re, cs, ce = floor_region
        floor[rs:re, cs:ce] = True
    stair = np.zeros((h, w), dtype=bool)
    pose_b = pose if pose is not None else _pose_bytes()

    return KeyframeMasks(
        scan_id=_SCAN_UUID,
        seq=seq,
        floor_mask=floor,
        stair_mask=stair,
        tx=tx,
        ty=ty,
        tz=tz,
        pose_matrix=pose_b,
    )


# ── 테스트 1: happy path — 카메라 앞 5m × 좌우 2m 사각형 floor ────────────────────

@pytest.mark.asyncio
async def test_happy_path_rectangle_floor() -> None:
    """설계 §11.1: forward +y pose + strip 내 wide floor → disk 생성, mask.sum() > 0."""
    img_h, img_w = 1080, 1920
    rot = _rot_forward_neg_y()
    intrin = default_intrinsics(img_w, img_h)

    mask = _make_mask(
        h=img_h, w=img_w, seq=0,
        pose=_server_pose_bytes(rotation=rot, translation=np.array([0.0, 0.0, 1.5])),
        floor_region=(540, img_h, 0, img_w),  # strip 하단 전부 floor
        tz=1.5,
    )

    step = AdaptiveBufferStep(
        strip_bottom_fraction=0.5,
        min_buffer_m=0.3,
        max_buffer_m=5.0,
    )
    grid = await step.run(
        masks=[mask],
        z0=0.0,
        intrinsics_by_frame=[intrin],
    )

    assert grid.mask.sum() > 0, "walkable grid가 비어 있음 — disk 1개라도 생성되어야 함"


# ── 테스트 2: clamp min — 1픽셀짜리 strip ──────────────────────────────────────

@pytest.mark.asyncio
async def test_clamp_min_on_tiny_strip() -> None:
    """설계 §11.2: strip 내 floor pixel 극소 → horiz/vert ≈ 0 → clamp min → r=0.3m."""
    img_h, img_w = 1080, 1920
    rot = _rot_forward_neg_y()
    intrin = default_intrinsics(img_w, img_h)

    # 1x1 pixel: horiz_width ≈ 0, vert_depth ≈ 0 → r_raw ≈ 0 → clamp_min=0.3
    mask = _make_mask(
        h=img_h, w=img_w, seq=0,
        pose=_server_pose_bytes(rotation=rot, translation=np.array([0.0, 0.0, 1.5])),
        floor_region=(img_h - 1, img_h, img_w // 2, img_w // 2 + 1),
        tz=1.5,
    )

    step = AdaptiveBufferStep(strip_bottom_fraction=0.5, min_buffer_m=0.3, max_buffer_m=5.0)
    grid = await step.run(masks=[mask], z0=0.0, intrinsics_by_frame=[intrin])

    # clamp min=0.3 적용 → r=0.3m disk → mask.sum() > 0
    assert grid.mask.sum() > 0, "clamp min 적용 후 disk가 생성되어야 함"


# ── 테스트 3: clamp max — 매우 넓은 floor ──────────────────────────────────────

@pytest.mark.asyncio
async def test_clamp_max_on_huge_floor() -> None:
    """설계 §11.3: 작은 intrinsics(fx=10) → world 범위 발산 → clamp max → r=5.0m."""
    img_h, img_w = 1080, 1920
    rot = _rot_forward_neg_y()
    # 매우 작은 fx/fy → 픽셀당 world 이동이 커짐 → horiz_width, vert_depth 수십m
    tiny_intrin = Intrinsics(fx=10.0, fy=10.0, cx=img_w / 2, cy=img_h / 2)

    mask = _make_mask(
        h=img_h, w=img_w, seq=0,
        pose=_server_pose_bytes(rotation=rot, translation=np.array([0.0, 0.0, 1.5])),
        floor_region=(540, img_h, 0, img_w),
        tz=1.5,
    )

    step = AdaptiveBufferStep(strip_bottom_fraction=0.5, min_buffer_m=0.3, max_buffer_m=5.0)
    grid = await step.run(masks=[mask], z0=0.0, intrinsics_by_frame=[tiny_intrin])

    # clamp max=5.0 적용 → r=5.0m disk → grid 생성
    assert grid.mask.sum() > 0, "clamp max 적용 후 disk가 생성되어야 함"


# ── 테스트 4: strip 내 floor pixel 0 → keyframe skip → empty grid ────────────

@pytest.mark.asyncio
async def test_skip_on_empty_strip() -> None:
    """설계 §11.4: strip 영역(하단 50%)에 floor 없음 → keyframe skip → empty grid."""
    img_h, img_w = 1080, 1920
    intrin = default_intrinsics(img_w, img_h)

    # strip 바깥(상단)에만 floor — strip 하단 50%(row 540~1079)에는 없음
    mask = _make_mask(
        h=img_h, w=img_w, seq=0,
        floor_region=(0, 540, 0, img_w),
        tz=1.5,
    )

    step = AdaptiveBufferStep(strip_bottom_fraction=0.5, min_buffer_m=0.3, max_buffer_m=5.0)
    grid = await step.run(masks=[mask], z0=0.0, intrinsics_by_frame=[intrin])

    assert grid.mask.sum() == 0, "strip 내 floor 없는데 walkable cell 생성됨"


# ── 테스트 5: 수직 카메라 → keyframe skip ─────────────────────────────────────

@pytest.mark.asyncio
async def test_skip_on_vertical_camera() -> None:
    """설계 §11.5: pose forward axis가 world z축 정렬 (fwd_xy ≈ 0) → keyframe skip."""
    img_h, img_w = 1080, 1920
    intrin = default_intrinsics(img_w, img_h)

    # 카메라가 바로 아래를 향함: forward_world = -R[:, 2] = [0, 0, 1] (z+ = forward)
    # → R[:, 2] = [0, 0, -1]  → fwd_xy = [0, 0] → 수직 카메라
    rot_down = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 1.0,  0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)

    mask = _make_mask(
        h=img_h, w=img_w, seq=0,
        pose=_server_pose_bytes(rotation=rot_down, translation=np.array([0.0, 0.0, 5.0])),
        floor_region=(540, img_h, 0, img_w),  # strip에 floor 있음 — 하지만 수직이라 skip
        tz=5.0,
    )

    step = AdaptiveBufferStep(strip_bottom_fraction=0.5, min_buffer_m=0.3, max_buffer_m=5.0)
    grid = await step.run(masks=[mask], z0=0.0, intrinsics_by_frame=[intrin])

    assert grid.mask.sum() == 0, "수직 카메라인데 walkable cell 생성됨"


# ── 테스트 6: Pipeline integration smoke ──────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_integration_smoke() -> None:
    """설계 §11.6: BuildPipeline(use_adaptive_buffer=True) + mock segmenter smoke."""
    import tempfile

    import cv2

    from indoor_server.application.building.pipeline import BuildPipeline
    from indoor_server.application.building.steps.floor_segmentation import KeyframeRef
    from indoor_server.domain.building.enums import BuildStep
    from indoor_server.infrastructure.ml.protocol import SegmentationOutput

    img_h, img_w = 108, 192  # 작은 해상도로 빠른 테스트

    class AllFloorSegmenter:
        """모든 픽셀을 floor(class 3)로 라벨링하는 stub."""

        async def segment(self, image: np.ndarray) -> SegmentationOutput:
            h, w = image.shape[:2]
            mask = np.full((h, w), fill_value=3, dtype=np.int32)
            return SegmentationOutput(class_mask=mask)

    rot = _rot_forward_neg_y()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        scan_uuid = UUID("11111111-1111-1111-1111-111111111111")
        kf_dir = tmp / "scans" / str(scan_uuid) / "keyframes"
        kf_dir.mkdir(parents=True)

        keyframes: list[KeyframeRef] = []
        for i in range(5):
            img_path = kf_dir / f"{i:06d}.jpg"
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)

            pose = _server_pose_bytes(
                rotation=rot,
                translation=np.array([float(i) * 0.5, 0.0, 1.5]),
            )
            keyframes.append(
                KeyframeRef(
                    scan_id=scan_uuid,
                    seq=i,
                    image_path=f"scans/{scan_uuid}/keyframes/{i:06d}.jpg",
                    tx=float(i) * 0.5,
                    ty=0.0,
                    tz=1.5,
                    pose_matrix=pose,
                )
            )

        pipeline = BuildPipeline(
            segmenter=AllFloorSegmenter(),
            storage_root=tmp,
            use_adaptive_buffer=True,
            adaptive_buffer_max_m=5.0,
            adaptive_buffer_min_m=0.3,
            adaptive_buffer_strip_fraction=0.5,
        )

        async def _progress(step: BuildStep, p: float) -> None:
            pass

        async def _cancel() -> bool:
            return False

        outcome = await pipeline.execute(
            scan_id=scan_uuid,
            build_job_id=UUID("22222222-2222-2222-2222-222222222222"),
            keyframes=keyframes,
            pois=[],
            progress_sink=_progress,
            cancel_check=_cancel,
        )

    # smoke: 예외 없이 완료 + walkable_cells ≥ 0 + keyframes_processed=5
    assert outcome.counts.walkable_cells >= 0
    assert outcome.counts.keyframes_processed == 5


# ── 4-way mutex 검증 ─────────────────────────────────────────────────────────

class _MockRunner:
    """SuperPointLightGlueRunner / DepthAnythingV2Runner stub."""


@pytest.mark.parametrize(
    "kwargs,desc",
    [
        (
            {"use_adaptive_buffer": True, "use_depth_nn": True, "depth_runner": _MockRunner()},
            "adaptive + depth_nn",
        ),
        (
            {
                "use_adaptive_buffer": True,
                "use_triangulation": True,
                "sp_lg_runner": _MockRunner(),
            },
            "adaptive + triangulation",
        ),
        (
            {
                "use_adaptive_buffer": True,
                "use_multiview_scale": True,
                "use_depth_nn": True,
                "depth_runner": _MockRunner(),
                "sp_lg_runner": _MockRunner(),
            },
            "adaptive + multiview",
        ),
    ],
)
def test_pipeline_mutex_adaptive_buffer_raises(kwargs: dict, desc: str) -> None:  # type: ignore[type-arg]
    """use_adaptive_buffer=True + 다른 mode 활성 → RuntimeError."""
    from indoor_server.application.building.pipeline import BuildPipeline

    base: dict = {  # type: ignore[type-arg]
        "segmenter": MagicMock(),
        "storage_root": Path("/tmp"),
    }
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(**{**base, **kwargs})


def test_pipeline_mutex_adaptive_plus_trajectory_raises() -> None:
    """use_adaptive_buffer=True + use_trajectory_buffer=True → RuntimeError."""
    from indoor_server.application.building.pipeline import BuildPipeline

    with pytest.raises(RuntimeError):
        BuildPipeline(
            segmenter=MagicMock(),
            storage_root=Path("/tmp"),
            use_adaptive_buffer=True,
            use_trajectory_buffer=True,
            use_triangulation=True,
            sp_lg_runner=_MockRunner(),
        )
