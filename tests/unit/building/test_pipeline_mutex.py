"""BuildPipeline 4-way mutex + trajectory_buffer 상호 배타 검증 (Sprint 23 W-2).

Sprint 23 W-2: adaptive=T, trajectory=T, triangulation=F 조합 →
  line 77 'traj requires triangulation' 이 먼저 발화함을 확인.
  (line 82-85 dead branch 제거 이후 회귀 방지)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from indoor_server.application.building.pipeline import BuildPipeline


class _FakeSegmenter:
    async def segment(self, image: object) -> object:  # pragma: no cover
        return MagicMock()


def _segmenter() -> _FakeSegmenter:
    return _FakeSegmenter()


# ── 4-way mutex ───────────────────────────────────────────────────────────────


def test_pipeline_mutex_depth_nn_and_triangulation_raises() -> None:
    """use_depth_nn=True + use_triangulation=True → 4-way mutex 에러."""
    mock_depth_runner = MagicMock()
    mock_sp_lg_runner = MagicMock()
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_depth_nn=True,
            depth_runner=mock_depth_runner,
            use_triangulation=True,
            sp_lg_runner=mock_sp_lg_runner,
        )


def test_pipeline_mutex_adaptive_and_triangulation_raises() -> None:
    """use_adaptive_buffer=True + use_triangulation=True → 4-way mutex 에러."""
    mock_sp_lg_runner = MagicMock()
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_adaptive_buffer=True,
            use_triangulation=True,
            sp_lg_runner=mock_sp_lg_runner,
        )


def test_pipeline_mutex_adaptive_and_depth_nn_raises() -> None:
    """use_adaptive_buffer=True + use_depth_nn=True → 4-way mutex 에러."""
    mock_depth_runner = MagicMock()
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_adaptive_buffer=True,
            use_depth_nn=True,
            depth_runner=mock_depth_runner,
        )


def test_pipeline_mutex_multiview_and_triangulation_raises() -> None:
    """use_multiview_scale=True + use_triangulation=True → 4-way mutex 에러."""
    mock_sp_lg_runner = MagicMock()
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_multiview_scale=True,
            use_triangulation=True,
            sp_lg_runner=mock_sp_lg_runner,
        )


# ── trajectory_buffer + adaptive 조합 (W-2 핵심) ─────────────────────────────


def test_pipeline_mutex_adaptive_plus_trajectory_raises() -> None:
    """adaptive=T, trajectory=T, triangulation=F 조합.

    Sprint 23 W-2: dead branch (lines 82-85) 제거 후 어떤 에러가 발화하는지 확인.
    - 4-way mutex (line 71): adaptive=1, triang=0, depth_nn=0, multiview=0 → mode_count=1 → OK
    - traj-requires-triangulation (line 77): trajectory=T, triangulation=F → 발화!

    즉, 이 조합은 'trajectory buffer is only meaningful for the triangulation path' 에러가 난다.
    """
    with pytest.raises(RuntimeError, match="use_triangulation"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_adaptive_buffer=True,
            use_trajectory_buffer=True,
            use_triangulation=False,
        )


def test_pipeline_trajectory_without_triangulation_raises() -> None:
    """use_trajectory_buffer=True + use_triangulation=False → traj-requires-tri 에러."""
    with pytest.raises(RuntimeError, match="use_triangulation"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_trajectory_buffer=True,
            use_triangulation=False,
        )


def test_pipeline_valid_baseline_no_error() -> None:
    """모든 mode flag 기본값 (baseline) 경로 → 에러 없음."""
    # 생성만 확인 (execute는 호출하지 않음)
    pipeline = BuildPipeline(
        segmenter=_segmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
    )
    assert pipeline is not None


def test_pipeline_valid_adaptive_no_error() -> None:
    """use_adaptive_buffer=True 단독 → 에러 없음."""
    pipeline = BuildPipeline(
        segmenter=_segmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
        use_adaptive_buffer=True,
    )
    assert pipeline is not None


# ── Sprint 46: 6-way mutex (floor_pointcloud 추가) ───────────────────────────


def test_pipeline_mutex_floor_pointcloud_and_rtabmap_trajectory_raises() -> None:
    """use_floor_pointcloud=True + use_rtabmap_trajectory=True → 6-way mutex 에러."""
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_floor_pointcloud=True,
            use_rtabmap_trajectory=True,
        )


def test_pipeline_mutex_floor_pointcloud_and_adaptive_raises() -> None:
    """use_floor_pointcloud=True + use_adaptive_buffer=True → 6-way mutex 에러."""
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_floor_pointcloud=True,
            use_adaptive_buffer=True,
        )


def test_pipeline_mutex_floor_pointcloud_and_triangulation_raises() -> None:
    """use_floor_pointcloud=True + use_triangulation=True → 6-way mutex 에러."""
    mock_sp_lg_runner = MagicMock()
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_floor_pointcloud=True,
            use_triangulation=True,
            sp_lg_runner=mock_sp_lg_runner,
        )


def test_pipeline_mutex_floor_pointcloud_and_depth_nn_raises() -> None:
    """use_floor_pointcloud=True + use_depth_nn=True → 6-way mutex 에러."""
    mock_depth_runner = MagicMock()
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_floor_pointcloud=True,
            use_depth_nn=True,
            depth_runner=mock_depth_runner,
        )


def test_pipeline_valid_floor_pointcloud_no_error() -> None:
    """use_floor_pointcloud=True 단독 → 에러 없음. image segmentation 자동 활성."""
    pipeline = BuildPipeline(
        segmenter=_segmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
        use_floor_pointcloud=True,
    )
    assert pipeline is not None
    # auto-enable 되어 있어야 한다 (private flag).
    assert pipeline._floor_pointcloud_auto_enabled_image_seg is True
    assert pipeline._rtabmap_image_segmentation_enabled is True


def test_pipeline_valid_rtabmap_trajectory_no_error() -> None:
    """use_rtabmap_trajectory=True 단독 → 회귀 보존 (5경로 smoke)."""
    pipeline = BuildPipeline(
        segmenter=_segmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
        use_rtabmap_trajectory=True,
    )
    assert pipeline is not None


def test_pipeline_valid_depth_nn_no_error() -> None:
    """use_depth_nn=True 단독 (depth_runner mock) → 회귀 보존."""
    mock_depth_runner = MagicMock()
    pipeline = BuildPipeline(
        segmenter=_segmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
        use_depth_nn=True,
        depth_runner=mock_depth_runner,
    )
    assert pipeline is not None


def test_pipeline_valid_triangulation_no_error() -> None:
    """use_triangulation=True 단독 (sp_lg_runner mock) → 회귀 보존."""
    mock_sp_lg_runner = MagicMock()
    pipeline = BuildPipeline(
        segmenter=_segmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
        use_triangulation=True,
        sp_lg_runner=mock_sp_lg_runner,
    )
    assert pipeline is not None
