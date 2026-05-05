"""FilesystemDebugSink.on_floor_polygons + on_floor_layout 테스트."""
from __future__ import annotations

import json
import struct
from pathlib import Path

from indoor_server.application.building.debug.filesystem_sink import FilesystemDebugSink
from indoor_server.application.building.debug.sink import NoopDebugSink
from indoor_server.domain.building.models import FloorLayout, FloorPolygon


def _pose_identity() -> bytes:
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return struct.pack("<16f", *identity)


def _make_fp(seq: int = 0, pi: int = 0) -> FloorPolygon:
    return FloorPolygon(
        seq=seq,
        polygon_index=pi,
        vertices_px=[(10, 10), (50, 10), (50, 50), (10, 50)],
        image_w=100,
        image_h=100,
        area_px=1600.0,
        pose_matrix=_pose_identity(),
        tx=0.0, ty=0.0, tz=0.0,
    )


def _make_layout() -> FloorLayout:
    from shapely.geometry import MultiPolygon
    from shapely.geometry import Polygon as ShapelyPolygon

    p1 = ShapelyPolygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p2 = ShapelyPolygon([(20, 0), (30, 0), (30, 10), (20, 10)])
    mp = MultiPolygon([p1, p2])
    return FloorLayout(
        multipolygon=mp,
        z0=0.0,
        sub_polygon_count=2,
        total_area_m2=float(mp.area),
    )


def test_on_floor_polygons_creates_json(tmp_path: Path) -> None:
    """on_floor_polygons → floor_polygons.json 생성."""
    sink = FilesystemDebugSink(
        out_dir=tmp_path / "out",
        storage_root=tmp_path / "storage",
        sample_count=2,
    )
    polys = [_make_fp(seq=i) for i in range(5)]
    sink.on_floor_polygons(polys)

    json_path = tmp_path / "out" / "floor_polygons.json"
    assert json_path.exists()
    assert json_path.stat().st_size > 0

    data = json.loads(json_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 5


def test_on_floor_polygons_creates_pngs(tmp_path: Path) -> None:
    """on_floor_polygons → floor_polygons/ 디렉터리에 PNG 생성."""
    import cv2

    sink = FilesystemDebugSink(
        out_dir=tmp_path / "out",
        storage_root=tmp_path / "storage",
        sample_count=3,
    )
    polys = [_make_fp(seq=i) for i in range(6)]
    sink.on_floor_polygons(polys)

    poly_dir = tmp_path / "out" / "floor_polygons"
    assert poly_dir.exists()
    pngs = list(poly_dir.glob("*.png"))
    assert len(pngs) >= 1, "floor_polygons/ 안에 PNG가 없음"

    for png in pngs:
        img = cv2.imread(str(png))
        assert img is not None, f"{png} cv2.imread 실패"
        h, w = img.shape[:2]
        assert h > 0 and w > 0


def test_on_floor_layout_creates_png_and_geojson(tmp_path: Path) -> None:
    """on_floor_layout → floor_layout.png + floor_layout.geojson 생성."""
    import cv2

    sink = FilesystemDebugSink(
        out_dir=tmp_path / "out",
        storage_root=tmp_path / "storage",
    )
    layout = _make_layout()
    trajectory = [(float(i), 0.0) for i in range(5)]
    sink.on_floor_layout(layout, trajectory)

    png_path = tmp_path / "out" / "floor_layout.png"
    geojson_path = tmp_path / "out" / "floor_layout.geojson"

    assert png_path.exists() and png_path.stat().st_size > 0
    assert geojson_path.exists() and geojson_path.stat().st_size > 0

    img = cv2.imread(str(png_path))
    assert img is not None
    h, w = img.shape[:2]
    assert h > 100 and w > 100

    geojson_data = json.loads(geojson_path.read_text())
    assert "type" in geojson_data
    assert geojson_data["type"] == "MultiPolygon"


def test_noop_sink_new_hooks_no_exception() -> None:
    """NoopDebugSink의 신규 hook은 예외 없이 동작한다."""
    sink = NoopDebugSink()
    polys = [_make_fp()]
    layout = _make_layout()

    sink.on_floor_polygons(polys)
    sink.on_floor_layout(layout, [(0.0, 0.0)])
