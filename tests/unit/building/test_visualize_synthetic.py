"""visualize_build.py synthetic 모드 검증."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# server/ 루트 경로 추가
_SERVER_ROOT = Path(__file__).resolve().parents[3]
_SRC = _SERVER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_build_synthetic_grid_t() -> None:
    """build_synthetic_grid('t') → walkable_cells > 1500."""
    from scripts.visualize_build import build_synthetic_grid  # type: ignore[import]

    mask, obs, origin = build_synthetic_grid("t")
    walkable_cells = int(mask.sum())
    assert walkable_cells > 1500, f"t shape walkable_cells={walkable_cells} should be > 1500"
    assert obs.shape == mask.shape
    assert origin.cell_size == pytest.approx(0.10)


def test_build_synthetic_grid_plus() -> None:
    """build_synthetic_grid('plus') → 가로+세로 합집합."""
    from scripts.visualize_build import build_synthetic_grid  # type: ignore[import]

    mask, _, _ = build_synthetic_grid("plus")
    assert mask.sum() > 2000


def test_build_synthetic_grid_l() -> None:
    """build_synthetic_grid('l') → L자 형태 > 500."""
    from scripts.visualize_build import build_synthetic_grid  # type: ignore[import]

    mask, _, _ = build_synthetic_grid("l")
    assert mask.sum() > 500


def test_run_synthetic_t(tmp_path: Path) -> None:
    """run_synthetic 't' → PNG 파일 생성."""
    from scripts.visualize_build import run_synthetic  # type: ignore[import]

    out = tmp_path / "composite.png"
    ret = run_synthetic("t", out)
    assert ret == 0, f"run_synthetic returned {ret}"
    assert out.exists(), "composite.png should be created"
    assert out.stat().st_size > 1024, "composite.png should be > 1KB"


def test_run_synthetic_plus(tmp_path: Path) -> None:
    """run_synthetic 'plus' → PNG 파일 생성."""
    from scripts.visualize_build import run_synthetic  # type: ignore[import]

    out = tmp_path / "plus_composite.png"
    ret = run_synthetic("plus", out)
    assert ret == 0


def test_synthetic_cli_subprocess(tmp_path: Path) -> None:
    """subprocess로 synthetic --shape t 실행 → exit 0 + PNG 존재."""
    _script = _SERVER_ROOT / "scripts" / "visualize_build.py"
    out_path = tmp_path / "cli_composite.png"

    result = subprocess.run(
        [sys.executable, str(_script), "synthetic", "--shape", "t", "--out", str(out_path)],
        capture_output=True,
        text=True,
        cwd=str(_SERVER_ROOT),
    )

    assert result.returncode == 0, (
        f"CLI exited {result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert out_path.exists(), "composite.png not created by CLI"
    assert out_path.stat().st_size > 1024


def test_dump_dir_mode(tmp_path: Path) -> None:
    """dump-dir 모드: FilesystemDebugSink로 만든 디렉터리에서 composite 재생성."""
    import cv2

    from indoor_server.application.building.debug.filesystem_sink import FilesystemDebugSink
    from indoor_server.application.building.steps.skeletonize import SkeletonGraph
    from indoor_server.domain.building.models import GridOrigin, WalkableGrid

    # FilesystemDebugSink로 디렉터리 생성
    out_dir = tmp_path / "dump"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    origin = GridOrigin(x0=0.0, y0=0.0, z0=0.0, cell_size=0.10, w=30, h=30)
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:20, 5:25] = True
    obs = mask.astype(np.uint16) * 4
    grid = WalkableGrid(origin=origin, mask=mask, observation_count=obs)
    skel_mask = mask.copy()
    skel = SkeletonGraph(nodes=[], edges=[], skeleton_pixel_count=int(skel_mask.sum()))

    sink = FilesystemDebugSink(out_dir=out_dir, storage_root=storage_root)
    sink.on_walkable_grid(grid)
    sink.on_skeleton(skel_mask, skel)
    sink.on_node_placement([], [], origin)
    sink.finalize()  # composite.png 생성

    # dump-dir 모드로 재렌더
    from scripts.visualize_build import run_dump_dir  # type: ignore[import]

    re_out = tmp_path / "re_composite.png"
    ret = run_dump_dir(out_dir, re_out, panel="all")
    assert ret == 0, f"run_dump_dir returned {ret}"
    assert re_out.exists()

    img = cv2.imread(str(re_out))
    assert img is not None
    assert img.shape[0] > 0 and img.shape[1] > 0
