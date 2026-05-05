"""visualize_build.py floor-polygons / floor-layout CLI 서브커맨드 테스트."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[4] / "server" / "scripts" / "visualize_build.py"


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT)] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_floor_polygons_synthetic_rect(tmp_path: Path) -> None:
    """floor-polygons --synthetic --shape rect → exit 0 + PNG 생성."""
    out = tmp_path / "fp.png"
    result = _run(["floor-polygons", "--synthetic", "--shape", "rect", "--out", str(out)])
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists(), "PNG 파일이 생성되지 않음"
    assert out.stat().st_size > 0


def test_floor_layout_synthetic(tmp_path: Path) -> None:
    """floor-layout --synthetic → exit 0 + PNG 생성."""
    out = tmp_path / "fl.png"
    result = _run(["floor-layout", "--synthetic", "--out", str(out)])
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists(), "PNG 파일이 생성되지 않음"
    assert out.stat().st_size > 0


def test_floor_polygons_dump_dir(tmp_path: Path) -> None:
    """floor-polygons <dump_dir> → floor_polygons.json 읽어 PNG 재렌더."""
    # 사전 조건: floor_polygons.json 생성
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()

    sample_data = [
        {
            "seq": 0,
            "polygon_index": 0,
            "vertices_px": [[10, 10], [50, 10], [50, 50], [10, 50]],
            "image_w": 100,
            "image_h": 100,
            "area_px": 1600.0,
        }
    ]
    (dump_dir / "floor_polygons.json").write_text(json.dumps(sample_data))

    out = tmp_path / "rerender.png"
    result = _run(["floor-polygons", str(dump_dir), "--out", str(out)])
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    assert out.stat().st_size > 0


def test_floor_layout_dump_dir(tmp_path: Path) -> None:
    """floor-layout <dump_dir> → floor_layout.geojson 읽어 PNG 재렌더."""
    dump_dir = tmp_path / "dump2"
    dump_dir.mkdir()

    geojson = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]],
            [[[10, 10], [15, 10], [15, 15], [10, 15], [10, 10]]],
        ],
    }
    (dump_dir / "floor_layout.geojson").write_text(json.dumps(geojson))

    out = tmp_path / "layout_rerender.png"
    result = _run(["floor-layout", str(dump_dir), "--out", str(out)])
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    assert out.stat().st_size > 0
