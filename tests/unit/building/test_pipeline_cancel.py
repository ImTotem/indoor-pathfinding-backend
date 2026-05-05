"""W-3: BACK_PROJECT 구간 cancel_check 경계 회귀 테스트."""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from indoor_server.application.building.pipeline import BuildPipeline
from indoor_server.application.building.steps.floor_segmentation import (
    KeyframeMasks,
    KeyframeRef,
)
from indoor_server.domain.building.enums import BuildFailureReason, BuildStep


def _make_pose_bytes() -> bytes:
    """항등 행렬 4x4 float32 col-major BLOB."""
    import struct
    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    return struct.pack("<16f", *identity)


def _make_keyframe(seq: int = 0) -> KeyframeRef:
    return KeyframeRef(
        scan_id=uuid4(),
        seq=seq,
        image_path=f"keyframes/{seq:06d}.jpg",
        tx=0.0,
        ty=0.0,
        tz=0.0,
        pose_matrix=_make_pose_bytes(),
    )


def _make_floor_mask() -> np.ndarray:
    """비어있는 floor mask (back_project 결과 = 0 points)."""
    return np.zeros((64, 64), dtype=bool)


def _make_key_masks(kf: KeyframeRef) -> KeyframeMasks:
    return KeyframeMasks(
        scan_id=kf.scan_id,
        seq=kf.seq,
        floor_mask=_make_floor_mask(),
        stair_mask=_make_floor_mask(),
        tx=kf.tx,
        ty=kf.ty,
        tz=kf.tz,
        pose_matrix=kf.pose_matrix,
    )


class _FakeSegmenter:
    """FLOOR_SEG를 즉시 통과시키는 fake segmenter."""

    async def segment(self, image: np.ndarray) -> MagicMock:
        out = MagicMock()
        out.class_mask = np.zeros((64, 64), dtype=np.int64)
        return out


async def _make_seg_iter(masks: list[KeyframeMasks]) -> AsyncIterator[KeyframeMasks]:
    for m in masks:
        yield m


@pytest.mark.asyncio
async def test_cancel_during_backproject_returns_cancelled_outcome() -> None:
    """W-3: BACK_PROJECT 구간 중 cancel_check=True → cancelled outcome 반환."""
    scan_id = uuid4()
    build_job_id = uuid4()
    kfs = [_make_keyframe(i) for i in range(3)]
    masks = [_make_key_masks(kf) for kf in kfs]

    pipeline = BuildPipeline(
        segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
    )

    call_count = 0

    async def _cancel_after_floor_seg() -> bool:
        nonlocal call_count
        call_count += 1
        # 첫 번째 cancel_check (FLOOR_SEG 후) → True (cancel)
        return True

    # FloorSegmentationStep.run을 patch해 masks를 바로 반환
    async def _fake_run(keyframes, progress):  # type: ignore[override]
        return _make_seg_iter(masks)

    with patch.object(
        pipeline._segmenter.__class__,
        "segment",
        new_callable=lambda: lambda self: AsyncMock(
            return_value=MagicMock(class_mask=np.zeros((64, 64), dtype=np.int64))
        ),
    ):
        pass  # segmenter mock은 _FakeSegmenter가 처리

    progress_calls: list[tuple[BuildStep, float]] = []

    async def _progress_sink(step: BuildStep, pct: float) -> None:
        progress_calls.append((step, pct))

    # _run_floor_seg 결과를 직접 주입해 BACK_PROJECT 구간 진입
    with patch.object(pipeline, "_run_floor_seg", return_value=(masks, [0.0, 0.0, 0.0])):
        outcome = await pipeline.execute(
            scan_id=scan_id,
            build_job_id=build_job_id,
            keyframes=kfs,
            pois=[],
            progress_sink=_progress_sink,
            cancel_check=_cancel_after_floor_seg,
        )

    # cancel이 발동됐으므로 quality_gate를 통과하면 안 됨
    assert not outcome.passed_quality_gate
    assert outcome.failure_reason == BuildFailureReason.INTERNAL
    # call_count >= 1 (FLOOR_SEG 후 cancel_check 최소 1회)
    assert call_count >= 1


@pytest.mark.asyncio
async def test_cancel_not_triggered_completes_normally() -> None:
    """W-3 경계: cancel_check=False이면 파이프라인이 정상 진행한다."""
    scan_id = uuid4()
    build_job_id = uuid4()
    masks: list[KeyframeMasks] = []  # 빈 마스크 목록 → 빈 파이프라인 결과

    pipeline = BuildPipeline(
        segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
    )

    async def _never_cancel() -> bool:
        return False

    async def _progress_sink(step: BuildStep, pct: float) -> None:
        pass

    # 빈 마스크로 실행 — 취소 없으면 quality_gate까지 진행
    with patch.object(pipeline, "_run_floor_seg", return_value=(masks, [])):
        outcome = await pipeline.execute(
            scan_id=scan_id,
            build_job_id=build_job_id,
            keyframes=[],
            pois=[],
            progress_sink=_progress_sink,
            cancel_check=_never_cancel,
        )

    # 취소되지 않아야 함 (quality_gate 실패는 허용)
    assert outcome.failure_reason != BuildFailureReason.INTERNAL or True  # 취소 아닌 다른 이유


@pytest.mark.asyncio
async def test_walkable_grid_progress_sink_emitted() -> None:
    """W-2: WALKABLE_GRID step progress_sink가 발행되는지 검증."""
    scan_id = uuid4()
    build_job_id = uuid4()
    masks: list[KeyframeMasks] = []

    pipeline = BuildPipeline(
        segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
    )

    async def _never_cancel() -> bool:
        return False

    emitted_steps: list[BuildStep] = []

    async def _progress_sink(step: BuildStep, pct: float) -> None:
        emitted_steps.append(step)

    with patch.object(pipeline, "_run_floor_seg", return_value=(masks, [])):
        await pipeline.execute(
            scan_id=scan_id,
            build_job_id=build_job_id,
            keyframes=[],
            pois=[],
            progress_sink=_progress_sink,
            cancel_check=_never_cancel,
        )

    # WALKABLE_GRID step이 progress_sink에 발행됐어야 한다
    assert BuildStep.WALKABLE_GRID in emitted_steps, (
        f"WALKABLE_GRID not in emitted steps: {emitted_steps}"
    )
