"""V1 floor map image renderer — floor_polygon.geojson → PNG.

raster 미리보기/썸네일용. 좌표계는 server world meter 그대로 사용하지만,
이미지는 픽셀 단위라 응답 헤더로 (minX, minY, maxX, maxY, scalePxPerM) 를
알려준다. 클라가 헤더를 보고 측위 좌표를 픽셀로 매핑 가능.
"""
from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from PIL import Image, ImageDraw
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import BuildingFloorService
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.config import settings
from indoor_server.domain.building.enums import BuildState
from indoor_server.infrastructure.db.repositories.build_job_repo import BuildJobRepository

logger = logging.getLogger(__name__)

_MAX_WIDTH_PX = 4096
_MIN_WIDTH_PX = 64
_HEX_RE = re.compile(r"^#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


@dataclass(frozen=True)
class RenderResult:
    png_bytes: bytes
    width_px: int
    height_px: int
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    scale_px_per_m: float
    build_job_id: str
    feature_count: int


class FloorMapImageService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def render(
        self,
        floor_id: UUID,
        *,
        width_px: int = 1024,
        padding_px: int = 16,
        fill: str = "#3399FF80",
        stroke: str = "#1A66CC",
        stroke_width: int = 2,
        background: str = "transparent",
    ) -> RenderResult:
        # 입력 검증
        if not (_MIN_WIDTH_PX <= width_px <= _MAX_WIDTH_PX):
            raise V1ServiceError(
                422,
                "INVALID_WIDTH",
                f"widthPx must be in [{_MIN_WIDTH_PX}, {_MAX_WIDTH_PX}].",
                {"widthPx": width_px},
            )
        if padding_px < 0 or padding_px > 256:
            raise V1ServiceError(
                422,
                "INVALID_PADDING",
                "paddingPx must be in [0, 256].",
                {"paddingPx": padding_px},
            )
        fill_rgba = _parse_hex_rgba(fill, "fill")
        stroke_rgba = _parse_hex_rgba(stroke, "stroke") if stroke else None

        # active scan + build_job
        svc = BuildingFloorService(self._session)
        active = await svc.get_active_scan_for_floor(floor_id)
        if active is None:
            raise V1ServiceError(
                404,
                "ACTIVE_SCAN_NOT_FOUND",
                "floor has no active scan.",
                {"floorId": str(floor_id)},
            )
        build_job = await BuildJobRepository(self._session).get_latest(active.scan_id)
        if build_job is None or build_job.state != BuildState.SUCCEEDED:
            raise V1ServiceError(
                422,
                "GRAPH_NOT_READY",
                "build has not succeeded yet.",
                {"state": build_job.state.value if build_job else "not_started"},
            )

        # polygon load
        polygon_path = (
            settings.storage_root
            / "builds"
            / str(build_job.build_job_id)
            / "polygon_v2"
            / "floor_polygon.geojson"
        )
        if not polygon_path.exists():
            raise V1ServiceError(
                422,
                "POLYGON_NOT_AVAILABLE",
                "floor_polygon.geojson not generated for this build.",
                {"buildJobId": str(build_job.build_job_id)},
            )
        geoms = _load_floor_geometries(polygon_path)
        if not geoms:
            raise V1ServiceError(
                422,
                "POLYGON_EMPTY",
                "floor_polygon.geojson has no Polygon features.",
            )

        # bounds (geometry 만 기준)
        min_x, min_y, max_x, max_y = _union_bounds(geoms)
        width_m = max_x - min_x
        height_m = max_y - min_y
        if width_m <= 0 or height_m <= 0:
            raise V1ServiceError(
                422,
                "POLYGON_DEGENERATE",
                "polygon has zero extent.",
                {"widthM": width_m, "heightM": height_m},
            )

        # px 사이즈 결정 (longest side = width_px - 2*padding)
        usable = width_px - 2 * padding_px
        if usable <= 0:
            raise V1ServiceError(
                422, "PADDING_TOO_LARGE", "paddingPx leaves no room to render."
            )
        scale = usable / max(width_m, height_m)
        out_w = int(round(width_m * scale)) + 2 * padding_px
        out_h = int(round(height_m * scale)) + 2 * padding_px

        # render
        if background == "transparent":
            bg = (0, 0, 0, 0)
        elif background == "white":
            bg = (255, 255, 255, 255)
        else:
            bg = _parse_hex_rgba(background, "background")

        img = Image.new("RGBA", (out_w, out_h), bg)
        draw = ImageDraw.Draw(img, "RGBA")

        def world_to_pixel(x: float, y: float) -> tuple[float, float]:
            px = (x - min_x) * scale + padding_px
            py = (max_y - y) * scale + padding_px  # y 반전
            return px, py

        for geom in geoms:
            for ring_outer, holes in _iter_polygon_rings(geom):
                pts_outer = [world_to_pixel(x, y) for x, y in ring_outer]
                if len(pts_outer) >= 3:
                    draw.polygon(
                        pts_outer,
                        fill=fill_rgba,
                        outline=stroke_rgba,
                        width=stroke_width,
                    )
                for hole in holes:
                    pts_hole = [world_to_pixel(x, y) for x, y in hole]
                    if len(pts_hole) >= 3:
                        draw.polygon(pts_hole, fill=(0, 0, 0, 0))

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return RenderResult(
            png_bytes=buf.getvalue(),
            width_px=out_w,
            height_px=out_h,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            scale_px_per_m=scale,
            build_job_id=str(build_job.build_job_id),
            feature_count=len(geoms),
        )


# ── helpers ──────────────────────────────────────────────────────────────────


def _parse_hex_rgba(value: str, name: str) -> tuple[int, int, int, int]:
    if not _HEX_RE.match(value):
        raise V1ServiceError(
            422,
            "INVALID_COLOR",
            f"{name} must be #RRGGBB or #RRGGBBAA.",
            {name: value},
        )
    h = value[1:]
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    a = int(h[6:8], 16) if len(h) == 8 else 255
    return r, g, b, a


def _load_floor_geometries(path: Path) -> list[BaseGeometry]:
    """floor_polygon.geojson 에서 floor_union (있으면) 만, 없으면 모든 Polygon 의 union.

    Build pipeline 이 'floor_union' kind 를 만들어 두므로 그것만 그리면 충분.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", []) if isinstance(data, dict) else []
    union_feats: list[BaseGeometry] = []
    other_polys: list[BaseGeometry] = []
    for feat in feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        gtype = geom.get("type")
        if gtype not in ("Polygon", "MultiPolygon"):
            continue
        try:
            g = shape(geom)
        except Exception as exc:
            logger.warning("invalid geometry skipped: %s", exc)
            continue
        if not g.is_valid or g.is_empty:
            continue
        kind = (feat.get("properties") or {}).get("kind")
        if kind == "floor_union":
            union_feats.append(g)
        else:
            other_polys.append(g)
    return union_feats or other_polys


def _union_bounds(
    geoms: list[BaseGeometry],
) -> tuple[float, float, float, float]:
    min_x = min(g.bounds[0] for g in geoms)
    min_y = min(g.bounds[1] for g in geoms)
    max_x = max(g.bounds[2] for g in geoms)
    max_y = max(g.bounds[3] for g in geoms)
    return min_x, min_y, max_x, max_y


def _iter_polygon_rings(geom: BaseGeometry):
    """Polygon / MultiPolygon → (exterior_coords, [hole_coords...]) 시퀀스."""
    if geom.geom_type == "Polygon":
        yield list(geom.exterior.coords), [list(r.coords) for r in geom.interiors]
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            yield list(poly.exterior.coords), [list(r.coords) for r in poly.interiors]
