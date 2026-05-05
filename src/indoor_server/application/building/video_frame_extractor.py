"""Sprint 67 — HEVC scan.mp4 디코드 + PTS 추출 (PyAV 기반).

설계:
    - PyAV libavcodec wrapper. macOS 에서는 hevc_videotoolbox 자동 사용.
    - frame.pts × time_base → pts_ns (nanosecond Int).
    - multi-thread decode. JPG 안 만들고 numpy ndarray 직접 yield.
    - stride: 5Hz subset 만 디코드하려면 stride=12 (60Hz/5Hz).

밀도/비용:
    - M4 macOS + VideoToolbox: 36000 frame (10분 60fps 1080p) 약 1-2분 디코드.
    - stride=12 (5Hz subset, 3000 frame) 약 10-20초.

호출자:
    - dense_floor_segmentation: stride=2 또는 1 (전체 dense heatmap 용)
    - fake_rtabmap_db_builder: stride=12 (5Hz keyframe 만 RTAB-Map 입력)
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoFrame:
    """디코드된 frame + PTS."""

    pts_ns: int
    image_rgb: np.ndarray  # (H, W, 3) uint8
    frame_index: int


class VideoFrameExtractor:
    """scan.mp4 stream decoder.

    Usage:
        extractor = VideoFrameExtractor(video_path)
        for frame in extractor.iter_frames(stride=1):
            ...

    threading:
        PyAV `thread_type="AUTO"` 로 decode 병렬화.
        CPU bound 라 caller 가 별도 thread 에 둘 필요 없음.
    """

    def __init__(self, video_path: Path) -> None:
        if not video_path.exists():
            raise FileNotFoundError(f"video file not found: {video_path}")
        self._video_path = video_path

    def probe(self) -> dict[str, object]:
        """frame count / fps / duration_seconds / codec / width / height 메타 반환.

        nb_frames 가 없는 mp4 도 있어 stream.frames 사용. 정확하지 않으면 None.
        """
        with av.open(str(self._video_path)) as container:
            stream = container.streams.video[0]
            return {
                "codec": stream.codec_context.name,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "fps_avg": float(stream.average_rate) if stream.average_rate else None,
                "duration_seconds": (
                    float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None
                    else None
                ),
                "frame_count_estimate": stream.frames or None,
            }

    def iter_frames(
        self,
        *,
        stride: int = 1,
    ) -> Iterator[VideoFrame]:
        """frame 디코드 + PTS 동반 yield.

        Args:
            stride: 1 이면 모든 frame, N 이면 N 번째마다 1 개 (60fps → stride=12 ≒ 5Hz).
                    decode 자체는 reference frame 의존성 때문에 모든 frame 통과.
                    yield 만 stride 적용.

        Yields:
            VideoFrame(pts_ns, image_rgb (H,W,3), frame_index).
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        with av.open(str(self._video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            time_base = stream.time_base
            if time_base is None:
                raise RuntimeError("video stream has no time_base")

            frame_index = 0
            for frame in container.decode(video=0):
                if frame_index % stride != 0:
                    frame_index += 1
                    continue

                # PTS → nanosecond. frame.pts 는 stream.time_base 단위.
                if frame.pts is None:
                    # decode-only frame (no presentation). skip.
                    frame_index += 1
                    continue

                pts_seconds = float(frame.pts) * float(time_base)
                pts_ns = int(round(pts_seconds * 1_000_000_000))

                rgb = frame.to_ndarray(format="rgb24")
                yield VideoFrame(
                    pts_ns=pts_ns,
                    image_rgb=rgb,
                    frame_index=frame_index,
                )
                frame_index += 1
