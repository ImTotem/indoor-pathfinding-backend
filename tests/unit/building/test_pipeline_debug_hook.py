"""BuildPipeline debug_sink 주입 시 hook 호출 검증."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from indoor_server.application.building.pipeline import BuildPipeline
from indoor_server.application.building.steps.floor_segmentation import (
    KeyframeMasks,
    KeyframeRef,
)
from indoor_server.application.building.steps.skeletonize import SkeletonGraph
from indoor_server.domain.building.enums import BuildStep
from indoor_server.domain.building.models import GridOrigin, WalkableGrid


def _make_pose_bytes() -> bytes:
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return struct.pack("<16f", *identity)


def _make_keyframe(seq: int = 0) -> KeyframeRef:
    return KeyframeRef(
        scan_id=uuid4(),
        seq=seq,
        image_path=f"keyframes/{seq:06d}.jpg",
        tx=0.0, ty=0.0, tz=0.0,
        pose_matrix=_make_pose_bytes(),
    )


def _make_masks(n: int = 0) -> list[KeyframeMasks]:
    masks = []
    for i in range(n):
        scan_id = uuid4()
        masks.append(KeyframeMasks(
            scan_id=scan_id,
            seq=i,
            floor_mask=np.zeros((16, 16), dtype=bool),
            stair_mask=np.zeros((16, 16), dtype=bool),
            tx=0.0, ty=0.0, tz=0.0,
            pose_matrix=_make_pose_bytes(),
        ))
    return masks


def _make_grid() -> WalkableGrid:
    origin = GridOrigin(x0=0.0, y0=0.0, z0=0.0, cell_size=0.10, w=16, h=16)
    mask = np.zeros((16, 16), dtype=bool)
    obs = np.zeros((16, 16), dtype=np.uint16)
    return WalkableGrid(origin=origin, mask=mask, observation_count=obs)


class _CallRecorder:
    """hook 호출 횟수를 기록하는 spy sink."""

    def __init__(self) -> None:
        self.floor_seg_calls: int = 0
        self.walkable_grid_calls: int = 0
        self.skeleton_calls: int = 0
        self.node_placement_calls: int = 0
        self.finalize_calls: int = 0

    def on_floor_segmentation(self, samples: Any) -> None:
        self.floor_seg_calls += 1

    def on_back_projection(self, points: Any, z0: float) -> None:
        pass

    def on_walkable_grid(self, grid: Any) -> None:
        self.walkable_grid_calls += 1

    def on_skeleton(self, skeleton_mask: Any, skel: Any) -> None:
        self.skeleton_calls += 1

    def on_node_placement(self, nodes: Any, edges: Any, origin: Any) -> None:
        self.node_placement_calls += 1

    def finalize(self) -> None:
        self.finalize_calls += 1


class _FakeSegmenter:
    async def segment(self, image: np.ndarray) -> MagicMock:
        out = MagicMock()
        out.class_mask = np.zeros((16, 16), dtype=np.int64)
        return out


@pytest.mark.asyncio
async def test_debug_sink_hooks_called() -> None:
    """debug_sink 주입 시 4개 hook이 각 1회씩 호출된다."""
    pipeline = BuildPipeline(
        segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
    )

    recorder = _CallRecorder()
    masks = _make_masks(0)
    grid = _make_grid()
    skel_graph = SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=0)
    skel_mask = np.zeros((16, 16), dtype=bool)

    async def _progress_sink(step: BuildStep, pct: float) -> None:
        pass

    async def _never_cancel() -> bool:
        return False

    with (
        patch.object(pipeline, "_run_floor_seg", return_value=(masks, [])),
        patch.object(pipeline, "_run_back_project_and_grid", return_value=(0.0, grid, 0)),
        patch.object(
            pipeline, "_run_skeletonize", return_value=(skel_graph, skel_mask)
        ),
    ):
        await pipeline.execute(
            scan_id=uuid4(),
            build_job_id=uuid4(),
            keyframes=[],
            pois=[],
            progress_sink=_progress_sink,
            cancel_check=_never_cancel,
            debug_sink=recorder,
        )

    assert recorder.floor_seg_calls == 1, "on_floor_segmentation should be called once"
    assert recorder.walkable_grid_calls == 1, "on_walkable_grid should be called once"
    assert recorder.skeleton_calls == 1, "on_skeleton should be called once"
    assert recorder.node_placement_calls == 1, "on_node_placement should be called once"


@pytest.mark.asyncio
async def test_debug_sink_none_noop() -> None:
    """debug_sink=None (기본값) 이면 hook 미호출 + outcome 정상 반환."""
    pipeline = BuildPipeline(
        segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
    )

    masks = _make_masks(0)
    grid = _make_grid()
    skel_graph = SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=0)
    skel_mask = np.zeros((16, 16), dtype=bool)

    async def _progress_sink(step: BuildStep, pct: float) -> None:
        pass

    async def _never_cancel() -> bool:
        return False

    with (
        patch.object(pipeline, "_run_floor_seg", return_value=(masks, [])),
        patch.object(pipeline, "_run_back_project_and_grid", return_value=(0.0, grid, 0)),
        patch.object(
            pipeline, "_run_skeletonize", return_value=(skel_graph, skel_mask)
        ),
    ):
        outcome = await pipeline.execute(
            scan_id=uuid4(),
            build_job_id=uuid4(),
            keyframes=[],
            pois=[],
            progress_sink=_progress_sink,
            cancel_check=_never_cancel,
            # debug_sink 미지정 → 기본 None
        )

    # outcome 반환돼야 함
    assert outcome is not None


@pytest.mark.asyncio
async def test_debug_sink_exception_does_not_break_pipeline() -> None:
    """debug sink hook 예외가 발생해도 파이프라인이 실패하지 않는다."""
    pipeline = BuildPipeline(
        segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
    )

    class _FailingSink:
        def on_floor_segmentation(self, samples: Any) -> None:
            raise RuntimeError("intentional floor seg error")

        def on_back_projection(self, points: Any, z0: float) -> None:
            raise RuntimeError("intentional bp error")

        def on_walkable_grid(self, grid: Any) -> None:
            raise RuntimeError("intentional walkable error")

        def on_skeleton(self, skeleton_mask: Any, skel: Any) -> None:
            raise RuntimeError("intentional skeleton error")

        def on_node_placement(self, nodes: Any, edges: Any, origin: Any) -> None:
            raise RuntimeError("intentional node error")

        def finalize(self) -> None:
            raise RuntimeError("intentional finalize error")

    masks = _make_masks(0)
    grid = _make_grid()
    skel_graph = SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=0)
    skel_mask = np.zeros((16, 16), dtype=bool)

    async def _progress_sink(step: BuildStep, pct: float) -> None:
        pass

    async def _never_cancel() -> bool:
        return False

    with (
        patch.object(pipeline, "_run_floor_seg", return_value=(masks, [])),
        patch.object(pipeline, "_run_back_project_and_grid", return_value=(0.0, grid, 0)),
        patch.object(
            pipeline, "_run_skeletonize", return_value=(skel_graph, skel_mask)
        ),
    ):
        # 예외 전파 없이 정상 완료해야 함
        outcome = await pipeline.execute(
            scan_id=uuid4(),
            build_job_id=uuid4(),
            keyframes=[],
            pois=[],
            progress_sink=_progress_sink,
            cancel_check=_never_cancel,
            debug_sink=_FailingSink(),  # type: ignore[arg-type]
        )

    assert outcome is not None
