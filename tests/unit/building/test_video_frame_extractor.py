"""Sprint 67 — VideoFrameExtractor smoke test.

PyAV 로 H.264 mp4 fixture 를 만들어 decode → PTS 추출 → ndarray 반환 확인.
실제 HEVC 는 hardware encoder 의존이라 H.264 로 fixture (decode 로직은 동일).
"""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from indoor_server.application.building.video_frame_extractor import (
    VideoFrameExtractor,
)


def _make_mp4_fixture(path: Path, num_frames: int = 10, fps: int = 30) -> None:
    """num_frames 개의 단색 frame 을 H.264 mp4 로 인코드."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        # 각 frame 마다 brightness 다르게 (decode 검증용)
        brightness = 30 + (i * 20) % 200
        arr = np.full((48, 64, 3), brightness, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_probe_returns_codec_and_dimensions(tmp_path: Path) -> None:
    mp4 = tmp_path / "test.mp4"
    _make_mp4_fixture(mp4, num_frames=10, fps=30)

    extractor = VideoFrameExtractor(mp4)
    info = extractor.probe()
    assert info["codec"] == "h264"
    assert info["width"] == 64
    assert info["height"] == 48
    assert info["fps_avg"] == pytest.approx(30.0)


def test_iter_frames_yields_all_at_stride_1(tmp_path: Path) -> None:
    mp4 = tmp_path / "test.mp4"
    _make_mp4_fixture(mp4, num_frames=10, fps=30)

    extractor = VideoFrameExtractor(mp4)
    frames = list(extractor.iter_frames(stride=1))

    assert len(frames) == 10
    # PTS 는 monotonic
    assert all(frames[i].pts_ns < frames[i + 1].pts_ns for i in range(len(frames) - 1))

    # 첫 frame PTS 는 0
    assert frames[0].pts_ns == 0

    # 30fps → 33.33ms = 33333333 ns
    expected_step = 1_000_000_000 // 30
    actual_step = frames[1].pts_ns - frames[0].pts_ns
    # PyAV 의 default time_base 가 fps 의 inverse 라 정확히 step 일치
    assert abs(actual_step - expected_step) < 100_000  # 0.1ms 오차 허용


def test_iter_frames_image_shape_rgb24(tmp_path: Path) -> None:
    mp4 = tmp_path / "test.mp4"
    _make_mp4_fixture(mp4, num_frames=3)

    extractor = VideoFrameExtractor(mp4)
    frames = list(extractor.iter_frames(stride=1))
    assert len(frames) == 3
    for f in frames:
        assert f.image_rgb.shape == (48, 64, 3)
        assert f.image_rgb.dtype == np.uint8


def test_iter_frames_with_stride(tmp_path: Path) -> None:
    """stride=3 → 30fps 영상에서 1/3 (10fps) 만 yield."""
    mp4 = tmp_path / "test.mp4"
    _make_mp4_fixture(mp4, num_frames=12, fps=30)

    extractor = VideoFrameExtractor(mp4)
    frames = list(extractor.iter_frames(stride=3))

    # 12 frame / stride 3 = 4 frame yield (index 0, 3, 6, 9)
    assert len(frames) == 4
    assert [f.frame_index for f in frames] == [0, 3, 6, 9]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VideoFrameExtractor(tmp_path / "missing.mp4")


def test_invalid_stride_raises(tmp_path: Path) -> None:
    mp4 = tmp_path / "test.mp4"
    _make_mp4_fixture(mp4, num_frames=2)
    extractor = VideoFrameExtractor(mp4)
    with pytest.raises(ValueError):
        list(extractor.iter_frames(stride=0))
