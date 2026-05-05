"""Sprint 51 — LineFitStep unit tests."""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.wall_polygon.components import (
    FragmentSet,
    WallFragment,
)
from indoor_server.application.building.steps.wall_polygon.line_fit import (
    LineFitParams,
    LineFitStep,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
)


def _heatmap(shape: tuple[int, int] = (50, 50)) -> ObstacleHeatmap:
    return ObstacleHeatmap(
        counts=np.zeros(shape, dtype=np.int32),
        origin_x=0.0,
        origin_y=0.0,
        cell_size_m=0.10,
        z0=0.0,
        height_min_m=0.30,
        height_max_m=2.50,
        metadata={},
    )


def _fragment_from_pixels(component_id: int, pixels: np.ndarray) -> WallFragment:
    return WallFragment(
        component_id=component_id,
        pixel_count=int(pixels.shape[0]),
        bbox_rows=(int(pixels[:, 0].min()), int(pixels[:, 0].max())),
        bbox_cols=(int(pixels[:, 1].min()), int(pixels[:, 1].max())),
        pixels_rc=pixels.astype(np.int32),
    )


def test_line_fit_accepts_long_line() -> None:
    # 1-row x 30-col strip → length 3.0 m, linearity ~ 1.0
    rows = np.full(30, 5)
    cols = np.arange(0, 30)
    pixels = np.column_stack([rows, cols])
    frag = _fragment_from_pixels(1, pixels)
    fragments = FragmentSet(
        fragments=[frag],
        label_map=np.zeros((50, 50), dtype=np.int32),
        metadata={},
    )
    lines = LineFitStep().run(fragments, heatmap=_heatmap())
    assert len(lines.segments) == 1
    seg = lines.segments[0]
    assert seg.length_m > 2.5
    assert seg.linearity > 0.9


def test_line_fit_rejects_blob() -> None:
    # 10x10 square blob → linearity ~ 0.5 → reject
    rows, cols = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    pixels = np.column_stack([rows.ravel(), cols.ravel()])
    frag = _fragment_from_pixels(1, pixels)
    fragments = FragmentSet(
        fragments=[frag],
        label_map=np.zeros((50, 50), dtype=np.int32),
        metadata={},
    )
    lines = LineFitStep().run(fragments, heatmap=_heatmap())
    assert lines.segments == []
    assert lines.rejected_blob_count == 1


def test_line_fit_rejects_short_line() -> None:
    # 1x4 strip → length 0.4 m → below default 0.5 m
    rows = np.full(4, 5)
    cols = np.arange(0, 4)
    pixels = np.column_stack([rows, cols])
    frag = _fragment_from_pixels(1, pixels)
    fragments = FragmentSet(
        fragments=[frag],
        label_map=np.zeros((50, 50), dtype=np.int32),
        metadata={},
    )
    lines = LineFitStep(LineFitParams(min_pixels=2, min_length_m=0.5)).run(
        fragments, heatmap=_heatmap()
    )
    assert lines.segments == []
    assert lines.rejected_short_count == 1
