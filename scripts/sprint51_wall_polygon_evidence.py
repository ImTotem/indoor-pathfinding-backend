"""Sprint 51 (Sprint 52 algorithm switch) — Wall polygon evidence harness.

Runs `WallPolygonFromObstacleStep` on a synthetic ㄱ자 corridor heatmap and
dumps a 7-PNG evidence pack + JSON report into
`_workspace/sprint_<n>/evidence/wall_polygon/{scan_id}/`. Default workspace
is sprint_52 (current sprint); pass `--workspace` to dump elsewhere.

Sprint 52 W-3: monkey-patch removed. The facade exposes `run_from_heatmap`
which lets dev tools build a synthetic ObstacleHeatmap in process and run
Steps 1..7 directly without monkey-patching `ObstacleSourceStep`.

Sprint 52 C-3: removed unused `_l_corridor_floor_geojson` placeholder; only
`_l_corridor_floor_simple` is referenced.

Usage:
    python scripts/sprint51_wall_polygon_evidence.py --scan-id scan_a
    python scripts/sprint51_wall_polygon_evidence.py --scan-id scan_b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # server/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from indoor_server.application.building.steps.wall_polygon import (  # noqa: E402
    WallPolygonFromObstacleStep,
    WallPolygonStepParams,
)
from indoor_server.application.building.steps.wall_polygon.evidence import (  # noqa: E402
    render_wall_polygon_evidence,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (  # noqa: E402
    ObstacleHeatmap,
)


def _l_corridor_heatmap(scan_label: str) -> ObstacleHeatmap:
    """ㄱ자 corridor synthetic heatmap (production-realistic).

    실제 ARKit/RTAB-Map obstacle scan 은 벽 두께 0.2m + 가구/천장 누적으로
    obstacle cluster 가 4~6 cell wide blob 군집으로 들어온다.
    """
    h, w = 80, 80
    counts = np.zeros((h, w), dtype=np.int32)
    rng = np.random.default_rng(42 if scan_label == "scan_a" else 7)
    base = 8
    # 5-cell-wide bands (실제 벽 두께 0.5m + scatter blob).
    counts[5:10, 5:70] = base
    counts[20:25, 5:55] = base
    counts[5:25, 5:10] = base
    counts[20:75, 50:55] = base
    counts[20:75, 65:70] = base
    counts[70:75, 50:70] = base
    fill_noise = rng.poisson(2.0, size=(h, w)).astype(np.int32)
    counts[counts > 0] = counts[counts > 0] + fill_noise[counts > 0]
    scatter_mask = (rng.uniform(0, 1, (h, w)) < 0.01) & (counts == 0)
    counts[scatter_mask] = 2

    return ObstacleHeatmap(
        counts=counts,
        origin_x=0.0,
        origin_y=0.0,
        cell_size_m=0.10,
        z0=0.0,
        height_min_m=0.30,
        height_max_m=2.50,
        metadata={"world_obstacle_point_count": int(counts.sum()), "params": {}},
    )


def _l_corridor_floor_simple() -> dict[str, object]:
    """L floor aligned with `_l_corridor_heatmap`'s right-side vertical wing."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [0.6, 0.6],
                [7.0, 0.6],
                [7.0, 7.5],
                [5.0, 7.5],
                [5.0, 2.5],
                [0.6, 2.5],
                [0.6, 0.6],
            ]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-id", default="scan_a", choices=["scan_a", "scan_b"])
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "_workspace" / "sprint_52",
    )
    args = parser.parse_args()

    out_dir = args.workspace / "evidence" / "wall_polygon" / args.scan_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture = _l_corridor_heatmap(args.scan_id)

    step = WallPolygonFromObstacleStep(WallPolygonStepParams())
    result = step.run_from_heatmap(
        fixture,
        floor_polygon_geojson=_l_corridor_floor_simple(),
    )
    paths = render_wall_polygon_evidence(
        result,
        output_dir=out_dir,
        scan_id=args.scan_id,
        floor_polygon_geojson=_l_corridor_floor_simple(),
    )
    print(
        f"scan_id={args.scan_id} accepted={result.accepted} "
        f"fail_reason={result.fail_reason}"
    )
    print(f"  line_count={result.metadata.get('line_count')}")
    print(f"  vertex_count={result.metadata.get('vertex_count')}")
    print(f"  orthogonality={result.metadata.get('corner_orthogonality_ratio')}")
    print(f"  iou_with_floor={result.metadata.get('iou_with_floor')}")
    print(f"  area_change_ratio={result.metadata.get('area_change_ratio')}")
    for key, p in paths.items():
        print(f"  {key} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
