"""floor_layout_render — per-frame polygon overlay + top-down layout PNG."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable

import numpy as np

from indoor_server.domain.building.models import FloorLayout, FloorPolygon

logger = logging.getLogger(__name__)

_PIXEL_PER_METER = 40  # 4cm/px (composite와 동일)
_MAX_IMG_LONG_SIDE = 1600  # 자동 다운스케일 임계
_MARGIN_M = 0.5

WorldToPxFn = Callable[[float, float], tuple[int, int]]


# ── per-frame overlay ────────────────────────────────────────────────────────


def render_floor_polygon_overlay(
    km_jpg: np.ndarray | None,
    floor_mask: np.ndarray,
    polygons: list[FloorPolygon],
    seq: int,
) -> np.ndarray:
    """
    원본 jpg(BGR) + cyan mask + 빨강 polygon overlay PNG 반환.

    km_jpg: (H, W, 3) BGR or None (없으면 검정 배경).
    floor_mask: (H, W) bool — mask 해상도 기준.
    polygons: 해당 seq의 FloorPolygon 목록.
    """
    import cv2

    h, w = floor_mask.shape

    # 배경 준비
    if km_jpg is not None:
        bgr: np.ndarray = km_jpg.copy()
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h))
    else:
        bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # cyan mask overlay (alpha 0.25)
    if floor_mask.any():
        cyan_layer = np.zeros((h, w, 3), dtype=np.uint8)
        cyan_layer[floor_mask] = (255, 255, 0)  # BGR cyan
        bgr = cv2.addWeighted(bgr, 1.0, cyan_layer, 0.25, 0)

    # polygon fill alpha 0.30 빨강
    if polygons:
        fill_layer = bgr.copy()
        for fp in polygons:
            pts = np.array(fp.vertices_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(fill_layer, [pts], color=(0, 0, 180))  # 어두운 빨강
        bgr = cv2.addWeighted(bgr, 0.70, fill_layer, 0.30, 0)

        # polygon outline 빨강 2px
        for fp in polygons:
            pts = np.array(fp.vertices_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(bgr, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # 헤더 텍스트
    n_poly = len(polygons)
    v_counts = "+".join(str(len(fp.vertices_px)) for fp in polygons) if polygons else "0"
    header_text = f"seq={seq:06d}  poly={n_poly}  vertices={v_counts}"
    overlay_h = 24
    header_blank = np.zeros((overlay_h, w, 3), dtype=np.uint8)
    bgr[0:overlay_h] = cv2.addWeighted(bgr[0:overlay_h], 0.4, header_blank, 0.6, 0)
    cv2.putText(bgr, header_text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return bgr


# ── top-down layout PNG ──────────────────────────────────────────────────────


def render_floor_layout_png(
    layout: FloorLayout,
    trajectory_xy: list[tuple[float, float]],
    scan_id: str = "",
    job_id: str = "",
) -> np.ndarray:
    """
    top-down world floor layout PNG 반환.

    layout.multipolygon: shapely MultiPolygon / Polygon
    trajectory_xy: [(x, y)] world coords
    """
    import cv2
    from shapely.geometry import mapping

    multi = layout.multipolygon

    # bbox 계산
    _bounds = getattr(multi, "bounds", None)
    _is_empty = getattr(multi, "is_empty", True)
    if _bounds is not None and not _is_empty:
        minx, miny, maxx, maxy = _bounds[0], _bounds[1], _bounds[2], _bounds[3]
    else:
        minx = miny = maxx = maxy = 0.0

    if trajectory_xy:
        xs = [p[0] for p in trajectory_xy]
        ys = [p[1] for p in trajectory_xy]
        minx = min(minx, min(xs))
        miny = min(miny, min(ys))
        maxx = max(maxx, max(xs))
        maxy = max(maxy, max(ys))

    # 최소 bbox 보장
    if maxx - minx < 1.0:
        maxx = minx + 1.0
    if maxy - miny < 1.0:
        maxy = miny + 1.0

    minx -= _MARGIN_M
    miny -= _MARGIN_M
    maxx += _MARGIN_M
    maxy += _MARGIN_M

    scale: float = float(_PIXEL_PER_METER)
    img_w = int((maxx - minx) * scale)
    img_h = int((maxy - miny) * scale)

    # 자동 다운스케일
    long_side = max(img_w, img_h)
    if long_side > _MAX_IMG_LONG_SIDE:
        down = _MAX_IMG_LONG_SIDE / long_side
        scale *= down
        img_w = int((maxx - minx) * scale)
        img_h = int((maxy - miny) * scale)

    img_w = max(img_w, 100)
    img_h = max(img_h, 100)

    def world_to_px(x: float, y: float) -> tuple[int, int]:
        """world → 이미지 픽셀 (y-up → y-down)."""
        px = int((x - minx) * scale)
        py = img_h - int((y - miny) * scale)
        return px, py

    # 배경
    img: np.ndarray = np.full((img_h, img_w, 3), (24, 24, 24), dtype=np.uint8)

    # gridline 5m 간격
    grid_layer: np.ndarray = img.copy()
    x_grid = minx
    while x_grid <= maxx:
        gpx, _ = world_to_px(x_grid, miny)
        cv2.line(grid_layer, (gpx, 0), (gpx, img_h), (120, 120, 120), 1)
        x_grid += 5.0
    y_grid = miny
    while y_grid <= maxy:
        _, gpy = world_to_px(minx, y_grid)
        cv2.line(grid_layer, (0, gpy), (img_w, gpy), (120, 120, 120), 1)
        y_grid += 5.0
    img = cv2.addWeighted(img, 0.7, grid_layer, 0.3, 0)

    # MultiPolygon 그리기
    geom_data: dict[str, object] = dict(mapping(multi))
    _draw_geojson_geom(img, geom_data, world_to_px)

    # trajectory
    if len(trajectory_xy) > 1:
        for i in range(len(trajectory_xy) - 1):
            p1 = world_to_px(*trajectory_xy[i])
            p2 = world_to_px(*trajectory_xy[i + 1])
            _draw_dashed_line(img, p1, p2, (0, 0, 220), 2)

    for pt in trajectory_xy:
        px, py = world_to_px(*pt)
        if 0 <= px < img_w and 0 <= py < img_h:
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

    # scale bar (좌하단)
    _draw_scale_bar(img, scale, img_h)

    # footer
    n_poly = layout.sub_polygon_count
    area = layout.total_area_m2
    footer_text = (
        f"polygons={n_poly}  area={area:.1f} m2  z0={layout.z0:.2f} m"
        f"  scan={scan_id[:8]}  job={job_id[:8]}"
    )
    footer_h = 18
    footer_row: np.ndarray = np.full((footer_h, img_w, 3), (40, 40, 40), dtype=np.uint8)
    cv2.putText(
        footer_row, footer_text, (4, 13),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1,
    )
    img = np.vstack([img, footer_row])

    return img


def _draw_geojson_geom(
    img: np.ndarray,
    geom_data: dict[str, object],
    world_to_px: WorldToPxFn,
) -> None:
    """GeoJSON geometry dict를 img에 그린다."""
    gtype = str(geom_data.get("type", ""))
    raw_coords = geom_data.get("coordinates", [])

    if gtype == "Polygon" and isinstance(raw_coords, list):
        _draw_polygon_coords(img, raw_coords, world_to_px)
    elif gtype == "MultiPolygon" and isinstance(raw_coords, list):
        for poly_coords in raw_coords:
            _draw_polygon_coords(img, poly_coords, world_to_px)
    elif gtype == "GeometryCollection":
        raw_geoms = geom_data.get("geometries", [])
        if isinstance(raw_geoms, list):
            for geom in raw_geoms:
                if isinstance(geom, dict):
                    _draw_geojson_geom(img, geom, world_to_px)


def _draw_polygon_coords(
    img: np.ndarray,
    coords: list[list[list[float]]],
    world_to_px: WorldToPxFn,
) -> None:
    """Polygon coords (exterior + holes) → fill + outline."""
    import cv2

    if not coords:
        return
    exterior = coords[0]
    pts = np.array([world_to_px(x, y) for x, y in exterior], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [pts], color=(80, 80, 80))  # 회색 fill
    cv2.polylines(img, [pts], isClosed=True, color=(240, 240, 240), thickness=2)  # 흰 outline


def _draw_dashed_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """5px on / 5px off 점선."""
    import cv2

    x1, y1 = p1
    x2, y2 = p2
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if dist < 1:
        return
    dash_len = 5
    gap_len = 5
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    pos = 0.0
    draw = True
    while pos < dist:
        seg = dash_len if draw else gap_len
        next_pos = min(pos + seg, dist)
        if draw:
            x_s = int(x1 + dx * pos)
            y_s = int(y1 + dy * pos)
            x_e = int(x1 + dx * next_pos)
            y_e = int(y1 + dy * next_pos)
            cv2.line(img, (x_s, y_s), (x_e, y_e), color, thickness)
        pos = next_pos
        draw = not draw


def _draw_scale_bar(img: np.ndarray, scale: float, img_h: int) -> None:
    """좌하단 1m scale bar."""
    import cv2

    bar_px = int(scale)  # 1m = scale px
    x0 = 10
    y0 = img_h - 28
    x1 = x0 + bar_px
    cv2.line(img, (x0, y0), (x1, y0), (255, 255, 255), 2)
    cv2.line(img, (x0, y0 - 4), (x0, y0 + 4), (255, 255, 255), 2)
    cv2.line(img, (x1, y0 - 4), (x1, y0 + 4), (255, 255, 255), 2)
    cv2.putText(img, "1 m", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)


# ── JSON serialization ───────────────────────────────────────────────────────


def polygons_to_json(polygons: list[FloorPolygon]) -> str:
    """FloorPolygon list → JSON string."""
    data = [
        {
            "seq": fp.seq,
            "polygon_index": fp.polygon_index,
            "vertices_px": fp.vertices_px,
            "image_w": fp.image_w,
            "image_h": fp.image_h,
            "area_px": fp.area_px,
        }
        for fp in polygons
    ]
    return json.dumps(data, ensure_ascii=False)


def layout_to_geojson(layout: FloorLayout) -> str:
    """FloorLayout.multipolygon → GeoJSON string."""
    from shapely.geometry import mapping

    geom_dict = mapping(layout.multipolygon)
    return json.dumps(geom_dict, ensure_ascii=False)
