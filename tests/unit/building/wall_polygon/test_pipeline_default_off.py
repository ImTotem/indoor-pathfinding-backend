"""Sprint 51 — pipeline integration: default OFF behavior + mutex check."""
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


def test_pipeline_default_off_does_not_set_wall_polygon_attribute_to_true() -> None:
    pipeline = BuildPipeline(
        segmenter=_segmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
    )
    assert pipeline._use_wall_polygon is False


def test_pipeline_wall_polygon_requires_floor_pointcloud() -> None:
    with pytest.raises(RuntimeError, match="requires use_floor_pointcloud"):
        BuildPipeline(
            segmenter=_segmenter(),  # type: ignore[arg-type]
            storage_root=Path("/tmp"),
            use_wall_polygon=True,
        )


def test_pipeline_wall_polygon_with_floor_pointcloud_ok() -> None:
    pipeline = BuildPipeline(
        segmenter=_segmenter(),  # type: ignore[arg-type]
        storage_root=Path("/tmp"),
        use_floor_pointcloud=True,
        use_adaptive_buffer=False,
        use_wall_polygon=True,
    )
    assert pipeline._use_wall_polygon is True
    assert pipeline._use_floor_pointcloud is True


def test_pipeline_build_counts_has_wall_polygon_field() -> None:
    """Sprint 51: BuildCounts schema must expose wall_polygon dict."""
    from indoor_server.domain.building.models import BuildCounts

    counts = BuildCounts()
    assert hasattr(counts, "wall_polygon")
    assert counts.wall_polygon is None


def test_wall_polygon_facade_import_path() -> None:
    """AC-1: package import surface stable."""
    from indoor_server.application.building.steps.wall_polygon import (
        WallPolygonFromObstacleStep,
        WallPolygonResult,
    )

    assert WallPolygonFromObstacleStep is not None
    assert WallPolygonResult is not None
