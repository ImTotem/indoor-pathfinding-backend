"""Sprint 20: BuildPipeline use_triangulation 분기 + 생성자 validation 테스트."""
from __future__ import annotations

import struct
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
from indoor_server.domain.building.enums import BuildStep
from indoor_server.infrastructure.ml.superpoint_lightglue import MatchedPair

# ── helpers ───────────────────────────────────────────────────────────────────


def _identity_pose_bytes() -> bytes:
    mat = np.eye(4, dtype=np.float32).flatten(order="F")
    return struct.pack("<16f", *mat)


def _make_keyframe(seq: int = 0) -> KeyframeRef:
    return KeyframeRef(
        scan_id=uuid4(),
        seq=seq,
        image_path=f"keyframes/{seq:06d}.jpg",
        tx=float(seq) * 0.3,
        ty=0.0,
        tz=0.0,
        pose_matrix=_identity_pose_bytes(),
    )


def _make_key_masks(kf: KeyframeRef, h: int = 64, w: int = 64) -> KeyframeMasks:
    return KeyframeMasks(
        scan_id=kf.scan_id,
        seq=kf.seq,
        floor_mask=np.ones((h, w), dtype=bool),
        stair_mask=np.zeros((h, w), dtype=bool),
        tx=kf.tx,
        ty=kf.ty,
        tz=kf.tz,
        pose_matrix=kf.pose_matrix,
    )


class _FakeSegmenter:
    async def segment(self, image: np.ndarray) -> MagicMock:
        out = MagicMock()
        out.class_mask = np.zeros((64, 64), dtype=np.int64)
        return out


def _make_mock_sp_lg_runner(n_matches: int = 1) -> MagicMock:
    """MatchedPair를 반환하는 mock runner."""
    kpts_a = np.zeros((max(n_matches, 1), 2), dtype=np.int64)
    kpts_b = np.zeros((max(n_matches, 1), 2), dtype=np.int64)
    if n_matches > 0:
        matches = np.array([[i, i] for i in range(n_matches)], dtype=np.int64)
    else:
        matches = np.empty((0, 2), dtype=np.int64)
    scores = np.ones(n_matches, dtype=np.float32) * 0.95

    mp = MatchedPair(
        kpts_a=kpts_a,
        kpts_b=kpts_b,
        matches=matches,
        scores=scores,
        size_a=(384, 384),
        size_b=(384, 384),
    )
    runner = MagicMock()
    runner.match = AsyncMock(return_value=mp)
    runner.input_size = 384
    return runner


# ── 생성자 validation ─────────────────────────────────────────────────────────


def test_constructor_use_triangulation_without_runner_raises() -> None:
    """use_triangulation=True + sp_lg_runner=None → RuntimeError."""
    with pytest.raises(RuntimeError, match="sp_lg_runner"):
        BuildPipeline(
            segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_triangulation=True,
            sp_lg_runner=None,
        )


def test_constructor_use_triangulation_and_depth_nn_raises() -> None:
    """use_triangulation=True + use_depth_nn=True → RuntimeError (mutually exclusive)."""
    mock_depth_runner = MagicMock()
    mock_sp_lg_runner = _make_mock_sp_lg_runner()
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_triangulation=True,
            use_depth_nn=True,
            depth_runner=mock_depth_runner,
            sp_lg_runner=mock_sp_lg_runner,
        )


def test_constructor_use_triangulation_and_multiview_raises() -> None:
    """use_triangulation=True + use_multiview_scale=True → RuntimeError."""
    mock_sp_lg_runner = _make_mock_sp_lg_runner()
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_triangulation=True,
            use_multiview_scale=True,
            sp_lg_runner=mock_sp_lg_runner,
        )


# ── triangulation 분기 integration ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_triangulation_branch_runs_without_depth_maps() -> None:
    """use_triangulation=True → BACK_PROJECT + WALKABLE_GRID 진행 확인."""
    scan_id = uuid4()
    build_job_id = uuid4()

    kfs = [_make_keyframe(i) for i in range(2)]
    masks = [_make_key_masks(kf) for kf in kfs]
    tz_values = [kf.tz for kf in kfs]

    mock_runner = _make_mock_sp_lg_runner(n_matches=0)  # 0 match — grid 빈 상태

    pipeline = BuildPipeline(
        segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
        sp_lg_runner=mock_runner,
        use_triangulation=True,
        triangulation_window=1,
    )

    async def _never_cancel() -> bool:
        return False

    progress_steps: list[BuildStep] = []

    async def _progress_sink(step: BuildStep, pct: float) -> None:
        progress_steps.append(step)

    with patch.object(pipeline, "_run_floor_seg", return_value=(masks, tz_values)):
        outcome = await pipeline.execute(
            scan_id=scan_id,
            build_job_id=build_job_id,
            keyframes=kfs,
            pois=[],
            progress_sink=_progress_sink,
            cancel_check=_never_cancel,
        )

    # WALKABLE_GRID progress가 발행되어야 함
    assert BuildStep.WALKABLE_GRID in progress_steps, (
        f"WALKABLE_GRID not in progress: {progress_steps}"
    )
    # BACK_PROJECT도 발행되어야 함
    assert BuildStep.BACK_PROJECT in progress_steps
    # outcome이 반환되어야 함
    assert outcome is not None


@pytest.mark.asyncio
async def test_triangulation_false_uses_baseline_path() -> None:
    """use_triangulation=False (default) → 기존 baseline 경로 진입 (회귀 가드)."""
    scan_id = uuid4()
    build_job_id = uuid4()
    masks: list[KeyframeMasks] = []

    pipeline = BuildPipeline(
        segmenter=_FakeSegmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
        use_triangulation=False,
    )

    async def _never_cancel() -> bool:
        return False

    progress_steps: list[BuildStep] = []

    async def _progress_sink(step: BuildStep, pct: float) -> None:
        progress_steps.append(step)

    with patch.object(pipeline, "_run_floor_seg", return_value=(masks, [])):
        outcome = await pipeline.execute(
            scan_id=scan_id,
            build_job_id=build_job_id,
            keyframes=[],
            pois=[],
            progress_sink=_progress_sink,
            cancel_check=_never_cancel,
        )

    # _use_triangulation=False → 정상적으로 파이프라인이 완료되어야 함.
    assert outcome is not None
