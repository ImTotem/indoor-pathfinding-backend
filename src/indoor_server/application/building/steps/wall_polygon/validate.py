"""Sprint 51 — Step 7: validate + IoU vs floor pointcloud."""
from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import shape
from shapely.ops import unary_union

from indoor_server.application.building.steps.wall_polygon.assembly import (
    AssembledPolygon,
)


@dataclass(frozen=True)
class ValidateParams:
    floor_iou_min: float = 0.50
    area_change_max_ratio: float = 0.40
    self_intersection_max: int = 0

    def to_metadata(self) -> dict[str, object]:
        return {
            "floor_iou_min": self.floor_iou_min,
            "area_change_max_ratio": self.area_change_max_ratio,
            "self_intersection_max": self.self_intersection_max,
        }


@dataclass(frozen=True)
class WallPolygonValidation:
    accepted: bool
    iou_with_floor: float
    area_change_ratio: float
    self_intersection_count: int
    reject_reasons: list[str]
    final_polygon_geojson: dict[str, object] | None
    metadata: dict[str, object] = field(default_factory=dict)


def _safe_polygon(geojson: dict[str, object] | None):  # type: ignore[no-untyped-def]
    if geojson is None:
        return None
    try:
        geom = shape(geojson)
    except Exception:
        return None
    if geom.is_empty:
        return None
    if not geom.is_valid:
        try:
            geom = geom.buffer(0)
        except Exception:
            return None
    return geom


class ValidateStep:
    """Step 7 — validate."""

    def __init__(self, params: ValidateParams | None = None) -> None:
        self._params = params if params is not None else ValidateParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if not (0.0 <= p.floor_iou_min <= 1.0):
            raise ValueError("floor_iou_min must be in [0, 1]")
        if p.area_change_max_ratio < 0:
            raise ValueError("area_change_max_ratio must be >= 0")
        if p.self_intersection_max < 0:
            raise ValueError("self_intersection_max must be >= 0")

    def run(
        self,
        assembled: AssembledPolygon,
        *,
        floor_polygon_geojson: dict[str, object] | None,
    ) -> WallPolygonValidation:
        params = self._params
        reasons: list[str] = []

        wall_geom = _safe_polygon(assembled.polygon_geojson)
        if wall_geom is None:
            reasons.append("no_wall_polygon")
            return WallPolygonValidation(
                accepted=False,
                iou_with_floor=0.0,
                area_change_ratio=0.0,
                self_intersection_count=int(assembled.self_intersection_count),
                reject_reasons=reasons,
                final_polygon_geojson=None,
                metadata={
                    "iou_with_floor": 0.0,
                    "area_change_ratio": 0.0,
                    "self_intersection_count": int(
                        assembled.self_intersection_count
                    ),
                    "reject_reasons": list(reasons),
                    "params": params.to_metadata(),
                },
            )

        if assembled.self_intersection_count > params.self_intersection_max:
            reasons.append("self_intersection")

        floor_geom = _safe_polygon(floor_polygon_geojson)
        iou = 0.0
        area_change = 0.0
        if floor_geom is None:
            reasons.append("floor_unavailable")
        else:
            try:
                inter = wall_geom.intersection(floor_geom)
                union = unary_union([wall_geom, floor_geom])
                if union.area > 0:
                    iou = float(inter.area / union.area)
                else:
                    iou = 0.0
            except Exception:
                iou = 0.0
            floor_area = float(floor_geom.area)
            wall_area = float(wall_geom.area)
            if floor_area > 0:
                area_change = abs(wall_area - floor_area) / floor_area
            if iou < params.floor_iou_min:
                reasons.append("iou_below_threshold")
            if area_change > params.area_change_max_ratio:
                reasons.append("area_change_exceeded")

        accepted = len(reasons) == 0
        metadata: dict[str, object] = {
            "iou_with_floor": float(iou),
            "area_change_ratio": float(area_change),
            "self_intersection_count": int(assembled.self_intersection_count),
            "reject_reasons": list(reasons),
            "params": params.to_metadata(),
        }
        return WallPolygonValidation(
            accepted=accepted,
            iou_with_floor=float(iou),
            area_change_ratio=float(area_change),
            self_intersection_count=int(assembled.self_intersection_count),
            reject_reasons=reasons,
            final_polygon_geojson=(
                assembled.polygon_geojson if accepted else None
            ),
            metadata=metadata,
        )
