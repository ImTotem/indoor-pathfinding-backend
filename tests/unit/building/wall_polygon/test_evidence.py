"""Sprint 51 — Evidence renderer smoke tests.

matplotlib 가 설치되어 있지 않으면 skip. evidence renderer 는 dev-time tool
이므로 server runtime 에 강제 의존하지 않는다 (architect 8.x — dependency 추가 0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from indoor_server.application.building.steps.wall_polygon import (  # noqa: E402
    WallPolygonFromObstacleStep,
    WallPolygonStepParams,
)
from indoor_server.application.building.steps.wall_polygon.evidence import (  # noqa: E402
    render_wall_polygon_evidence,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (  # noqa: E402
    ObstacleHeatmap,
    ObstacleSourceStep,
)


def _heatmap_for_evidence() -> ObstacleHeatmap:
    h, w = 30, 30
    counts = np.zeros((h, w), dtype=np.int32)
    counts[5:7, 5:25] = 5
    counts[23:25, 5:25] = 5
    counts[5:25, 5:7] = 5
    counts[5:25, 23:25] = 5
    return ObstacleHeatmap(
        counts=counts,
        origin_x=0.0, origin_y=0.0, cell_size_m=0.10,
        z0=0.0, height_min_m=0.30, height_max_m=2.50,
        metadata={"world_obstacle_point_count": int(counts.sum()), "params": {}},
    )


def test_render_wall_polygon_evidence_writes_seven_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _heatmap_for_evidence()
    monkeypatch.setattr(ObstacleSourceStep, "run", lambda self, **kw: fixture)

    step = WallPolygonFromObstacleStep(WallPolygonStepParams())
    result = step.run(
        nodes=[], frames=[], floor_masks_by_node_id={}, z0=0.0,
        floor_polygon_geojson=None,
    )

    out = tmp_path / "wall_polygon"
    paths = render_wall_polygon_evidence(
        result,
        output_dir=out,
        scan_id="test_scan",
        floor_polygon_geojson={
            "type": "Polygon",
            "coordinates": [
                [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [0.0, 0.0]]
            ],
        },
    )
    expected = [
        "01_density", "02_components", "03_line_fits",
        "04_after_snap", "05_after_merge", "06_polygon",
        "07_iou_with_floor",
    ]
    for key in expected:
        assert key in paths
        assert paths[key].exists()
        # AC-2 sanity: file is non-trivially sized (matplotlib title etc.)
        assert paths[key].stat().st_size > 1000
    assert paths["report_json"].exists()
    json_text = paths["report_json"].read_text(encoding="utf-8")
    assert '"scan_id"' in json_text


def test_render_evidence_handles_failure_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If facade fails early, evidence still produces placeholder PNGs."""
    empty = ObstacleHeatmap(
        counts=np.zeros((1, 1), dtype=np.int32),
        origin_x=0.0, origin_y=0.0, cell_size_m=0.10,
        z0=0.0, height_min_m=0.0, height_max_m=2.0,
        metadata={"world_obstacle_point_count": 0, "params": {}},
    )
    monkeypatch.setattr(ObstacleSourceStep, "run", lambda self, **kw: empty)
    step = WallPolygonFromObstacleStep()
    result = step.run(
        nodes=[], frames=[], floor_masks_by_node_id={}, z0=0.0,
        floor_polygon_geojson=None,
    )
    out = tmp_path / "wp"
    paths = render_wall_polygon_evidence(result, output_dir=out, scan_id="empty")
    # All 7 PNGs created (some are placeholders).
    for key in ["01_density", "02_components", "03_line_fits",
                "04_after_snap", "05_after_merge", "06_polygon",
                "07_iou_with_floor"]:
        assert paths[key].exists()
