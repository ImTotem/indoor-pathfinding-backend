"""Sprint 51 (algorithm switched in Sprint 52) — Wall polygon facade.

Single top-level orchestrator `WallPolygonFromObstacleStep` that runs Steps
0..7 sequentially and returns a `WallPolygonResult` describing the final
polygon (if accepted), full per-stage metadata, and the line set hint
fallback used when assembly/validation fails.

Sprint 52 algorithm change: Step 2~3 dispatch now uses skeleton split
(primary) + Hough fallback (secondary). The legacy ComponentSplit + LineFit
PCA path is preserved as dead code (see `components.py`, `line_fit.py`) and
is not invoked from the facade. `run_from_heatmap` is a Step 0 bypass for
test/dev callers (replaces evidence-script monkey-patch).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from indoor_server.application.building.steps.wall_polygon.assembly import (
    AssembledPolygon,
    PolygonAssemblyParams,
    PolygonAssemblyStep,
)
from indoor_server.application.building.steps.wall_polygon.components import (
    ComponentSplitParams,
)
from indoor_server.application.building.steps.wall_polygon.density import (
    DensityRefineParams,
    DensityRefineStep,
)
from indoor_server.application.building.steps.wall_polygon.hough_fallback import (
    HoughFallbackParams,
    HoughFallbackStep,
)
from indoor_server.application.building.steps.wall_polygon.line_fit import (
    LineFitParams,
    LineSegment,
    LineSegmentSet,
)
from indoor_server.application.building.steps.wall_polygon.merge import (
    CollinearMergeStep,
    MergeParams,
    WallLineSet,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
    ObstacleSourceStep,
    ObstacleSourceStepParams,
)
from indoor_server.application.building.steps.wall_polygon.skeleton_split import (
    SkeletonSplitParams,
    SkeletonSplitStep,
)
from indoor_server.application.building.steps.wall_polygon.snap import (
    AngleSnapParams,
    AngleSnapStep,
)
from indoor_server.application.building.steps.wall_polygon.validate import (
    ValidateParams,
    ValidateStep,
    WallPolygonValidation,
)
from indoor_server.domain.building.rtabmap_models import RtabmapDataFrame, RtabmapNode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WallPolygonStepParams:
    obstacle_source: ObstacleSourceStepParams = field(
        default_factory=ObstacleSourceStepParams
    )
    density: DensityRefineParams = field(default_factory=DensityRefineParams)
    # Sprint 52: Step 2~3 algorithm pair
    skeleton_split: SkeletonSplitParams = field(default_factory=SkeletonSplitParams)
    hough_fallback: HoughFallbackParams = field(default_factory=HoughFallbackParams)
    # Sprint 51 dead path — preserved for rollback / Sprint 53+ cleanup
    components: ComponentSplitParams = field(default_factory=ComponentSplitParams)
    line_fit: LineFitParams = field(default_factory=LineFitParams)
    snap: AngleSnapParams = field(default_factory=AngleSnapParams)
    merge: MergeParams = field(default_factory=MergeParams)
    assembly: PolygonAssemblyParams = field(default_factory=PolygonAssemblyParams)
    validate: ValidateParams = field(default_factory=ValidateParams)
    min_wall_lines: int = 4
    max_wall_lines: int = 20
    min_snap_pass_ratio: float = 0.60
    # Sprint 52: algorithm switch + dedup tunables
    use_skeleton_split: bool = True
    use_hough_fallback: bool = True
    dedup_distance_m: float = 0.20
    dedup_angle_deg: float = 5.0

    def to_metadata(self) -> dict[str, object]:
        return {
            "obstacle_source": self.obstacle_source.to_metadata(),
            "density": self.density.to_metadata(),
            "skeleton_split": self.skeleton_split.to_metadata(),
            "hough_fallback": self.hough_fallback.to_metadata(),
            "components": self.components.to_metadata(),
            "line_fit": self.line_fit.to_metadata(),
            "snap": self.snap.to_metadata(),
            "merge": self.merge.to_metadata(),
            "assembly": self.assembly.to_metadata(),
            "validate": self.validate.to_metadata(),
            "min_wall_lines": self.min_wall_lines,
            "max_wall_lines": self.max_wall_lines,
            "min_snap_pass_ratio": self.min_snap_pass_ratio,
            "use_skeleton_split": self.use_skeleton_split,
            "use_hough_fallback": self.use_hough_fallback,
            "dedup_distance_m": self.dedup_distance_m,
            "dedup_angle_deg": self.dedup_angle_deg,
        }


@dataclass(frozen=True)
class WallPolygonResult:
    accepted: bool
    final_polygon_geojson: dict[str, object] | None
    stage_outputs: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    fail_reason: str | None = None
    line_set_for_hint: WallLineSet | None = None


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two undirected angles (deg)."""
    d = (float(a) - float(b)) % 180.0
    if d > 90.0:
        d = 180.0 - d
    return float(d)


def _segment_midpoint(seg: LineSegment) -> tuple[float, float]:
    return ((seg.x1 + seg.x2) / 2.0, (seg.y1 + seg.y2) / 2.0)


def _meta_int(meta: dict[str, object], key: str, default: int = 0) -> int:
    """Coerce a metadata field to int safely (mypy-friendly)."""
    raw = meta.get(key, default)
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int | float):
        return int(raw)
    return default


def _dedup_line_segments(
    segments: list[LineSegment],
    *,
    distance_m: float,
    angle_deg: float,
) -> tuple[list[LineSegment], int]:
    """Drop near-duplicate segments. Keeps the higher-pixel-count member.

    Two segments collide when midpoint distance < `distance_m` AND undirected
    angle diff < `angle_deg`.
    """
    kept: list[LineSegment] = []
    dropped = 0
    for seg in segments:
        absorbed = False
        seg_mid = _segment_midpoint(seg)
        for idx, existing in enumerate(kept):
            mid = _segment_midpoint(existing)
            dx = seg_mid[0] - mid[0]
            dy = seg_mid[1] - mid[1]
            if math.hypot(dx, dy) < distance_m and _angle_diff_deg(
                seg.angle_deg, existing.angle_deg
            ) < angle_deg:
                if seg.pixel_count > existing.pixel_count:
                    kept[idx] = seg
                absorbed = True
                dropped += 1
                break
        if not absorbed:
            kept.append(seg)
    return kept, dropped


class WallPolygonFromObstacleStep:
    """Top-level facade — runs Steps 0..7 sequentially."""

    def __init__(self, params: WallPolygonStepParams | None = None) -> None:
        self._params = params if params is not None else WallPolygonStepParams()

    @property
    def params(self) -> WallPolygonStepParams:
        return self._params

    def run(
        self,
        *,
        nodes: list[RtabmapNode],
        frames: list[RtabmapDataFrame],
        floor_masks_by_node_id: dict[int, np.ndarray],
        obstacle_masks_by_node_id: dict[int, np.ndarray] | None = None,
        z0: float,
        floor_polygon_geojson: dict[str, object] | None,
    ) -> WallPolygonResult:
        p = self._params
        # Step 0
        heatmap = ObstacleSourceStep(p.obstacle_source).run(
            nodes=nodes,
            frames=frames,
            floor_masks_by_node_id=floor_masks_by_node_id,
            obstacle_masks_by_node_id=obstacle_masks_by_node_id,
            z0=z0,
        )
        return self.run_from_heatmap(
            heatmap, floor_polygon_geojson=floor_polygon_geojson
        )

    def run_from_heatmap(
        self,
        heatmap: ObstacleHeatmap,
        *,
        floor_polygon_geojson: dict[str, object] | None,
    ) -> WallPolygonResult:
        """Step 0 bypass entrypoint — used by tests and evidence scripts.

        Production worker still calls `run`. The split lets dev tools build a
        synthetic ObstacleHeatmap in process and invoke Steps 1..7 without
        monkey-patching ObstacleSourceStep.
        """
        p = self._params
        obstacle_meta = dict(heatmap.metadata)

        if heatmap.metadata.get("world_obstacle_point_count", 0) == 0:
            return self._build_failure(
                fail_reason="no_obstacle_data",
                stage_outputs={"obstacle_source": heatmap},
                stages_meta={"obstacle_source": obstacle_meta},
                line_set=None,
            )

        # Step 1
        refined = DensityRefineStep(p.density).run(heatmap)
        density_meta = dict(refined.metadata)
        if refined.metadata.get("cells_after_morph", 0) == 0:
            return self._build_failure(
                fail_reason="noise_blob_only",
                stage_outputs={
                    "obstacle_source": heatmap,
                    "density": refined,
                },
                stages_meta={
                    "obstacle_source": obstacle_meta,
                    "density": density_meta,
                },
                line_set=None,
            )

        # Sprint 52: Step 2~3 = skeleton split + optional Hough fallback
        segments, primary_meta, fallback_meta, source = self._build_line_set(
            refined,
            heatmap,
        )
        primary_count = (
            _meta_int(primary_meta, "segment_count")
            if "segment_count" in primary_meta
            else _meta_int(primary_meta, "accept_count")
        )
        fallback_count = (
            _meta_int(fallback_meta, "accept_count") if fallback_meta else 0
        )
        rejected_short = _meta_int(primary_meta, "rejected_short_count")
        if fallback_meta:
            rejected_short += _meta_int(fallback_meta, "rejected_short_count")
        line_fit_meta: dict[str, object] = {
            "segment_id_origin": "skeleton_subsegment_or_hough",
            "primary_count": primary_count,
            "fallback_count": fallback_count,
            "dedup_dropped": _meta_int(primary_meta, "dedup_dropped"),
            "source": source,
            "skeleton_split": dict(primary_meta),
            "hough_fallback": dict(fallback_meta),
            "accept_count": len(segments),
        }
        lines = LineSegmentSet(
            segments=segments,
            rejected_blob_count=0,
            rejected_short_count=rejected_short,
            metadata=line_fit_meta,
        )
        if not lines.segments:
            return self._build_failure(
                fail_reason="noise_blob_only",
                stage_outputs={
                    "obstacle_source": heatmap,
                    "density": refined,
                    "line_fit": lines,
                },
                stages_meta={
                    "obstacle_source": obstacle_meta,
                    "density": density_meta,
                    "line_fit": line_fit_meta,
                },
                line_set=None,
            )

        # Step 4
        snapped = AngleSnapStep(p.snap).run(lines)
        snap_meta = dict(snapped.metadata)
        if snapped.snap_pass_ratio < p.min_snap_pass_ratio:
            return self._build_failure(
                fail_reason="snap_low_acceptance",
                stage_outputs={
                    "obstacle_source": heatmap,
                    "density": refined,
                    "line_fit": lines,
                    "snap": snapped,
                },
                stages_meta={
                    "obstacle_source": obstacle_meta,
                    "density": density_meta,
                    "line_fit": line_fit_meta,
                    "snap": snap_meta,
                },
                line_set=None,
            )

        # Step 5
        merged = CollinearMergeStep(p.merge).run(snapped)
        merge_meta = dict(merged.metadata)
        line_count = len(merged.lines)
        if line_count < p.min_wall_lines or line_count > p.max_wall_lines:
            return self._build_failure(
                fail_reason="line_count_out_of_range",
                stage_outputs={
                    "obstacle_source": heatmap,
                    "density": refined,
                    "line_fit": lines,
                    "snap": snapped,
                    "merge": merged,
                },
                stages_meta={
                    "obstacle_source": obstacle_meta,
                    "density": density_meta,
                    "line_fit": line_fit_meta,
                    "snap": snap_meta,
                    "merge": merge_meta,
                },
                line_set=merged,
            )

        # Step 6
        assembled = PolygonAssemblyStep(p.assembly).run(merged)
        assembly_meta = dict(assembled.metadata)
        if assembled.polygon_geojson is None:
            return self._build_failure(
                fail_reason="polygon_assembly_failed",
                stage_outputs={
                    "obstacle_source": heatmap,
                    "density": refined,
                    "line_fit": lines,
                    "snap": snapped,
                    "merge": merged,
                    "assembly": assembled,
                },
                stages_meta={
                    "obstacle_source": obstacle_meta,
                    "density": density_meta,
                    "line_fit": line_fit_meta,
                    "snap": snap_meta,
                    "merge": merge_meta,
                    "assembly": assembly_meta,
                },
                line_set=merged,
            )

        # Step 7
        validation = ValidateStep(p.validate).run(
            assembled,
            floor_polygon_geojson=floor_polygon_geojson,
        )
        validate_meta = dict(validation.metadata)

        fail_reason: str | None = None
        if not validation.accepted:
            if "iou_below_threshold" in validation.reject_reasons:
                fail_reason = "iou_below_threshold"
            elif "self_intersection" in validation.reject_reasons:
                fail_reason = "self_intersection"
            elif "area_change_exceeded" in validation.reject_reasons:
                fail_reason = "area_change_exceeded"
            elif "floor_unavailable" in validation.reject_reasons:
                fail_reason = "floor_unavailable"
            else:
                fail_reason = "validation_rejected"

        stage_outputs = {
            "obstacle_source": heatmap,
            "density": refined,
            "line_fit": lines,
            "snap": snapped,
            "merge": merged,
            "assembly": assembled,
            "validate": validation,
        }
        stages_meta: dict[str, object] = {
            "obstacle_source": obstacle_meta,
            "density": density_meta,
            "line_fit": line_fit_meta,
            "snap": snap_meta,
            "merge": merge_meta,
            "assembly": assembly_meta,
            "validate": validate_meta,
        }

        metadata = self._build_metadata(
            stages=stages_meta,
            accepted=validation.accepted,
            fail_reason=fail_reason,
            assembled=assembled,
            validation=validation,
            line_count=line_count,
        )

        return WallPolygonResult(
            accepted=validation.accepted,
            final_polygon_geojson=validation.final_polygon_geojson,
            stage_outputs=stage_outputs,
            metadata=metadata,
            fail_reason=fail_reason,
            line_set_for_hint=merged,
        )

    def _build_line_set(
        self,
        refined: object,
        heatmap: ObstacleHeatmap,
    ) -> tuple[
        list[LineSegment],
        dict[str, object],
        dict[str, object],
        str,
    ]:
        """Run Step 2~3 dispatch.

        Returns (segments, primary_metadata, fallback_metadata or {}, source).
        source is one of "skeleton_only" | "hough_only" | "skeleton+hough".
        """
        p = self._params
        primary_segments: list[LineSegment] = []
        primary_meta: dict[str, object] = {}
        fallback_meta: dict[str, object] = {}

        if p.use_skeleton_split:
            primary = SkeletonSplitStep(p.skeleton_split).run(
                refined,  # type: ignore[arg-type]
                heatmap=heatmap,
            )
            primary_segments = list(primary.segments)
            primary_meta = dict(primary.metadata)

        need_fallback = (
            p.use_hough_fallback
            and len(primary_segments) < p.min_wall_lines
        )
        if need_fallback:
            fallback = HoughFallbackStep(p.hough_fallback).run(
                refined,  # type: ignore[arg-type]
                heatmap=heatmap,
            )
            fallback_segments = list(fallback.segments)
            fallback_meta = dict(fallback.metadata)
        else:
            fallback_segments = []

        combined = primary_segments + fallback_segments
        if combined:
            deduped, dropped = _dedup_line_segments(
                combined,
                distance_m=p.dedup_distance_m,
                angle_deg=p.dedup_angle_deg,
            )
        else:
            deduped, dropped = [], 0
        primary_meta["dedup_dropped"] = int(dropped)

        if primary_segments and fallback_segments:
            source = "skeleton+hough"
        elif primary_segments:
            source = "skeleton_only"
        elif fallback_segments:
            source = "hough_only"
        else:
            source = "empty"
        return deduped, primary_meta, fallback_meta, source

    def _build_failure(
        self,
        *,
        fail_reason: str,
        stage_outputs: dict[str, object],
        stages_meta: Mapping[str, object],
        line_set: WallLineSet | None,
    ) -> WallPolygonResult:
        stages_object: dict[str, object] = dict(stages_meta)
        metadata = self._build_metadata(
            stages=stages_object,
            accepted=False,
            fail_reason=fail_reason,
            assembled=None,
            validation=None,
            line_count=len(line_set.lines) if line_set else 0,
        )
        return WallPolygonResult(
            accepted=False,
            final_polygon_geojson=None,
            stage_outputs=stage_outputs,
            metadata=metadata,
            fail_reason=fail_reason,
            line_set_for_hint=line_set,
        )

    def _build_metadata(
        self,
        *,
        stages: Mapping[str, object],
        accepted: bool,
        fail_reason: str | None,
        assembled: AssembledPolygon | None,
        validation: WallPolygonValidation | None,
        line_count: int,
    ) -> dict[str, object]:
        return {
            "enabled": True,
            "accepted": bool(accepted),
            "fail_reason": fail_reason,
            "stages": dict(stages),
            "line_count": int(line_count),
            "vertex_count": (
                int(assembled.vertex_count) if assembled is not None else None
            ),
            "corner_orthogonality_ratio": (
                float(assembled.corner_orthogonality_ratio)
                if assembled is not None
                else None
            ),
            "iou_with_floor": (
                float(validation.iou_with_floor) if validation is not None else None
            ),
            "area_change_ratio": (
                float(validation.area_change_ratio)
                if validation is not None
                else None
            ),
            "self_intersection_count": (
                int(validation.self_intersection_count)
                if validation is not None
                else None
            ),
            "fallback_used": (
                assembled.fallback_used if assembled is not None else None
            ),
            "params": self._params.to_metadata(),
        }
