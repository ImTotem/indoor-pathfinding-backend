"""Sprint 51 — ComponentSplitStep unit tests."""
from __future__ import annotations

import numpy as np

from indoor_server.application.building.steps.wall_polygon.components import (
    ComponentSplitParams,
    ComponentSplitStep,
)
from indoor_server.application.building.steps.wall_polygon.density import DensityRefined


def _refined(mask: np.ndarray) -> DensityRefined:
    return DensityRefined(occupancy_mask=mask.astype(bool), metadata={})


def test_components_two_separate_chains() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:12] = True  # horizontal chain
    mask[10:18, 15] = True  # vertical chain
    fragments = ComponentSplitStep(ComponentSplitParams(min_area_cells=4)).run(_refined(mask))
    assert len(fragments.fragments) == 2
    assert fragments.metadata["components_kept"] == 2


def test_components_drop_below_min_area() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = True
    mask[5:8, 5:7] = True  # 6 cells
    fragments = ComponentSplitStep(ComponentSplitParams(min_area_cells=4)).run(_refined(mask))
    assert len(fragments.fragments) == 1
    assert fragments.metadata["components_dropped_by_area"] >= 1


def test_components_empty_input() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    fragments = ComponentSplitStep().run(_refined(mask))
    assert fragments.fragments == []
    assert fragments.label_map.sum() == 0
