"""Sprint 67 — DenseVideoFloorStep 통합 smoke test.

PyAV mp4 fixture + 합성 poses.bin + Fake segformer 로 ray-plane intersection 동작 검증.
실제 segformer 호출은 비싸기 때문에 stub.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from indoor_server.application.building.steps.back_projection import Intrinsics
from indoor_server.application.building.steps.dense_video_floor import (
    DenseVideoFloorParams,
    DenseVideoFloorStep,
)
from indoor_server.infrastructure.ml.segformer_onnx import SegmentationOutput


# ADE20K floor class index. extract_floor_stair_masks() 에서 floor=3, rug=28 가 floor 로 매핑.
ADE20K_FLOOR_IDX = 3


class FakeSegmenter:
    """모든 픽셀을 floor 로 분류하는 stub. SegformerOnnxSegmenter 대체."""

    async def segment(self, image: np.ndarray) -> SegmentationOutput:
        h, w = image.shape[:2]
        return SegmentationOutput(class_mask=np.full((h, w), ADE20K_FLOOR_IDX, dtype=np.int32))


def _make_mp4(path: Path, num_frames: int, fps: int) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    for i in range(num_frames):
        arr = np.full((48, 64, 3), 100 + i * 10, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _identity_pose_floats() -> list[float]:
    """4x4 identity (camera 1m 위에서 -z 방향 보는 자세). column-major flatten."""
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 1.0  # camera at z=1
    # column-major flatten: col0(xyzw), col1(xyzw), col2(xyzw), col3(xyzw)
    return list(pose.T.flatten())


def _write_poses_bin(path: Path, num_records: int, fps: int) -> None:
    buf = bytearray()
    floats = _identity_pose_floats()
    for i in range(num_records):
        # PTS 는 PyAV mp4 의 default time_base 와 일치해야 매칭됨.
        # PyAV 가 fps 의 inverse 를 time_base 로 두기 때문에 frame i 의 PTS = i / fps 초.
        pts_ns = int(round(i / fps * 1_000_000_000))
        buf.extend(struct.pack("<q", pts_ns))
        buf.extend(struct.pack("<16f", *floats))
    path.write_bytes(bytes(buf))


@pytest.mark.asyncio
async def test_dense_video_floor_returns_world_points(tmp_path: Path) -> None:
    fps = 30
    num_frames = 6
    mp4 = tmp_path / "scan.mp4"
    poses = tmp_path / "poses.bin"
    _make_mp4(mp4, num_frames=num_frames, fps=fps)
    _write_poses_bin(poses, num_records=num_frames, fps=fps)

    step = DenseVideoFloorStep(
        segmenter=FakeSegmenter(),  # type: ignore[arg-type]
        params=DenseVideoFloorParams(stride=1, pixel_stride=4),
    )
    intrinsics = Intrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0)
    result = await step.run(
        video_path=mp4,
        poses_path=poses,
        intrinsics=intrinsics,
        z0=0.0,
    )

    # Identity pose + camera 1m 위에서 floor mask 전체 → 충분한 world point.
    assert result.points_xy.shape[1] == 2
    assert result.points_xy.shape[0] > 0
    assert result.metadata["frames_total"] == num_frames
    assert result.metadata["frames_used"] == num_frames
    assert result.metadata["frames_pose_miss"] == 0


@pytest.mark.asyncio
async def test_dense_video_floor_stride_skips_frames(tmp_path: Path) -> None:
    fps = 30
    mp4 = tmp_path / "scan.mp4"
    poses = tmp_path / "poses.bin"
    _make_mp4(mp4, num_frames=12, fps=fps)
    _write_poses_bin(poses, num_records=12, fps=fps)

    step = DenseVideoFloorStep(
        segmenter=FakeSegmenter(),  # type: ignore[arg-type]
        params=DenseVideoFloorParams(stride=3, pixel_stride=4),
    )
    intrinsics = Intrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0)
    result = await step.run(
        video_path=mp4, poses_path=poses, intrinsics=intrinsics, z0=0.0,
    )

    # 12 frame / stride 3 = 4 frame 사용.
    assert result.metadata["frames_total"] == 4
    assert result.metadata["frames_used"] == 4


@pytest.mark.asyncio
async def test_dense_video_floor_invalid_stride_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DenseVideoFloorStep(
            segmenter=FakeSegmenter(),  # type: ignore[arg-type]
            params=DenseVideoFloorParams(stride=0),
        )


@pytest.mark.asyncio
async def test_dense_video_floor_pose_miss_threshold(tmp_path: Path) -> None:
    """poses.bin 이 video PTS 와 멀면 매칭 누락 비율 초과 → throw."""
    fps = 30
    mp4 = tmp_path / "scan.mp4"
    poses = tmp_path / "poses.bin"
    _make_mp4(mp4, num_frames=4, fps=fps)
    # poses 의 PTS 를 video PTS 와 1초 이상 어긋나게 작성
    buf = bytearray()
    floats = _identity_pose_floats()
    for i in range(4):
        buf.extend(struct.pack("<q", 10_000_000_000 + i * 33_333_333))  # 10초 후
        buf.extend(struct.pack("<16f", *floats))
    poses.write_bytes(bytes(buf))

    step = DenseVideoFloorStep(
        segmenter=FakeSegmenter(),  # type: ignore[arg-type]
        params=DenseVideoFloorParams(stride=1, max_pose_miss_ratio=0.0),
    )
    intrinsics = Intrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0)
    from indoor_server.application.building.steps.dense_video_floor import (
        DenseVideoFloorError,
    )
    with pytest.raises(DenseVideoFloorError, match="pose"):
        await step.run(
            video_path=mp4, poses_path=poses, intrinsics=intrinsics, z0=0.0,
        )
