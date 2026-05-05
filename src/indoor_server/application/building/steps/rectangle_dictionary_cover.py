"""Sprint 50 — Rectangle Dictionary Cover.

obs heatmap (WalkableGrid.observation_count) 를 source-of-truth 로 두고,
회전 grid + integral image 기반으로 angle/thickness/length dictionary 후보를
greedy 선택해 자연 직각 polygon union 을 생산한다.

설계 근거: Sprint 50 architect_design + Codex evaluator-design FAIL 5 BLOCKER.
사용자 절충 C 적용.

핵심 안전장치 (Codex BLOCKER 1~5 + WARN 6):
- candidate stride = 3 cells (start position 폭발 차단)
- length 는 fixed ladder ([1,2,3,5,8,15,30] m) — 단조 가정 회피
- max_candidates_per_dimension cap 200 — 메모리 폭주 방지
- precision threshold 는 candidate 채택 시 static gate. greedy iteration 에서는
  marginal hits (uncovered hits) 만 재계산 (W-3)
- recall / over-cover gate 미달 시 fallback metadata (`accepted=False`,
  `fallback_source="sprint49_hint_chain"`) 만 반환. 호출자 (pipeline) 가
  hint chain path 로 대체 (BLOCKER 5)
- rotation 보존: cv2.warpAffine + INTER_NEAREST 사용 + sum 보존 측정값을
  metadata 에 기록 (W-2)
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

from indoor_server.domain.building.models import GridOrigin, WalkableGrid

logger = logging.getLogger(__name__)


DEFAULT_ANGLES_DEG: tuple[float, ...] = tuple(float(a) for a in range(0, 90, 5))
DEFAULT_THICKNESSES_M: tuple[float, ...] = (
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
    10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
)
DEFAULT_LENGTH_LADDER_M: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0)


@dataclass(frozen=True)
class RectangleCandidate:
    """단일 후보 직사각형. rotated grid frame 좌표 + world polygon 둘 다 보유."""

    angle_deg: float
    thickness_m: float
    length_m: float
    # rotated grid frame
    row0: int
    col0: int
    row1: int
    col1: int
    rotated_origin_x: float
    rotated_origin_y: float
    cell_size_m: float
    # static metric (candidate 채택 시 한 번만 계산)
    hits: int          # rotated grid 안 obs heatmap weight 총합
    area_cells: int
    precision: float
    # world polygon (4 corner)
    world_polygon: Polygon

    def metadata(self) -> dict[str, object]:
        return {
            "angle_deg": float(self.angle_deg),
            "thickness_m": float(self.thickness_m),
            "length_m": float(self.length_m),
            "row0": int(self.row0),
            "col0": int(self.col0),
            "row1": int(self.row1),
            "col1": int(self.col1),
            "hits": int(self.hits),
            "area_cells": int(self.area_cells),
            "precision": float(self.precision),
        }


@dataclass(frozen=True)
class RectangleDictionaryCoverParams:
    """Codex BLOCKER 1~3 + WARN 1 반영된 dictionary scope.

    angles/thicknesses 는 사용자 절충 C 그대로 18×18.
    length 는 fixed ladder, candidate stride 3 cells, max_candidates 200/dim 으로
    후보 폭발 차단.

    Sprint 50 v2:
        - axes_override: None 이면 angles_deg full 18 sweep (v1). list 주입 시
          그 angle 만 사용. 일반적으로 (primary, primary+90°) axis pair 두 개.
        - precision_threshold_dynamic: True 면 thickness 비례 동적 threshold
          (thick 큰 환경에서 완화). False 면 precision_threshold fixed (v1).
        - precision_threshold_min: dynamic 시 최소 floor (default 0.65).
    """

    angles_deg: tuple[float, ...] = DEFAULT_ANGLES_DEG
    thicknesses_m: tuple[float, ...] = DEFAULT_THICKNESSES_M
    length_candidates_m: tuple[float, ...] = DEFAULT_LENGTH_LADDER_M
    candidate_stride_cells: int = 3
    max_candidates_per_dimension: int = 200
    precision_threshold: float = 0.85
    min_hits: int = 5
    min_area_m2: float = 0.5
    recall_min: float = 0.70
    over_cover_max: float = 0.20
    min_incremental_hits: int = 5
    max_iterations: int = 256
    time_budget_sec: float = 30.0
    rotated_padding_cells: int = 4
    # Sprint 50 v2 — axis pair injection + dynamic precision
    axes_override: tuple[float, ...] | None = None
    precision_threshold_dynamic: bool = False
    precision_threshold_min: float = 0.65

    def to_metadata(self) -> dict[str, object]:
        return {
            "angles_deg": list(self.angles_deg),
            "thicknesses_m": list(self.thicknesses_m),
            "length_candidates_m": list(self.length_candidates_m),
            "candidate_stride_cells": int(self.candidate_stride_cells),
            "max_candidates_per_dimension": int(self.max_candidates_per_dimension),
            "precision_threshold": float(self.precision_threshold),
            "min_hits": int(self.min_hits),
            "min_area_m2": float(self.min_area_m2),
            "recall_min": float(self.recall_min),
            "over_cover_max": float(self.over_cover_max),
            "min_incremental_hits": int(self.min_incremental_hits),
            "max_iterations": int(self.max_iterations),
            "time_budget_sec": float(self.time_budget_sec),
            "axes_override": (
                list(self.axes_override)
                if self.axes_override is not None
                else None
            ),
            "precision_threshold_dynamic": bool(self.precision_threshold_dynamic),
            "precision_threshold_min": float(self.precision_threshold_min),
        }


def compute_dynamic_precision_threshold(
    *,
    thickness_m: float,
    base: float,
    floor_min: float,
) -> float:
    """thickness 큰 corridor 에서 precision threshold 완화.

    base 0.85, floor_min 0.65 기준:
        thickness ≤ 3m: base 그대로 (0.85)
        thickness 4m: 0.80
        thickness 5m: 0.75
        thickness 6m: 0.70
        thickness ≥ 9m: floor_min (0.65)
    선형: drop = max(0, thickness/3 - 1) * 0.05.
    """
    drop = max(0.0, thickness_m / 3.0 - 1.0) * 0.05
    return float(max(floor_min, base - drop))


@dataclass(frozen=True)
class RectangleDictionaryCoverResult:
    """결과 + Codex BLOCKER 5 fallback 계약 metadata."""

    rectangles: list[RectangleCandidate]
    union_polygon: MultiPolygon | None
    footprint_geojson: dict[str, object] | None
    accepted: bool
    fallback_used: bool
    fallback_source: str | None
    metadata: dict[str, object] = field(default_factory=dict)


class RectangleDictionaryCoverStep:
    """obs heatmap 을 회전 grid × integral image × greedy cover.

    호출자는 fallback_used / accepted 만 보고 dispatch 한다.
    """

    def __init__(self, params: RectangleDictionaryCoverParams | None = None) -> None:
        self._params = params if params is not None else RectangleDictionaryCoverParams()
        self._validate()

    def _validate(self) -> None:
        p = self._params
        if not p.angles_deg:
            raise ValueError("angles_deg must not be empty")
        if not p.thicknesses_m:
            raise ValueError("thicknesses_m must not be empty")
        if not p.length_candidates_m:
            raise ValueError("length_candidates_m must not be empty")
        if p.candidate_stride_cells < 1:
            raise ValueError("candidate_stride_cells must be >= 1")
        if p.max_candidates_per_dimension < 1:
            raise ValueError("max_candidates_per_dimension must be >= 1")
        if not (0.0 < p.precision_threshold <= 1.0):
            raise ValueError("precision_threshold must be in (0, 1]")
        if p.min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if p.min_area_m2 < 0.0:
            raise ValueError("min_area_m2 must be >= 0")
        if not (0.0 < p.recall_min <= 1.0):
            raise ValueError("recall_min must be in (0, 1]")
        if not (0.0 <= p.over_cover_max <= 5.0):
            raise ValueError("over_cover_max must be in [0, 5]")
        if p.time_budget_sec <= 0.0:
            raise ValueError("time_budget_sec must be > 0")
        if p.axes_override is not None and len(p.axes_override) == 0:
            raise ValueError("axes_override must be None or non-empty tuple")
        if not (0.0 < p.precision_threshold_min <= 1.0):
            raise ValueError("precision_threshold_min must be in (0, 1]")
        if p.precision_threshold_min > p.precision_threshold:
            raise ValueError(
                "precision_threshold_min must be <= precision_threshold"
            )

    def run(
        self,
        grid: WalkableGrid,
    ) -> RectangleDictionaryCoverResult:
        params = self._params
        cell_size = float(grid.origin.cell_size)
        heatmap = grid.observation_count.astype(np.int64, copy=False)
        if heatmap.ndim != 2:
            raise ValueError("observation_count must be 2D")
        total_hits = int(heatmap.sum())
        if total_hits == 0:
            return self._fallback_result(
                reason="empty_heatmap",
                metadata={
                    "total_hits": 0,
                    "rectangle_count": 0,
                    "params": params.to_metadata(),
                },
            )

        start_time = time.monotonic()
        wall_deadline = start_time + params.time_budget_sec

        # cell 단위로 변환된 dictionary
        thickness_cells_list = [
            max(1, int(round(t_m / cell_size))) for t_m in params.thicknesses_m
        ]
        length_cells_list = [
            max(1, int(round(l_m / cell_size))) for l_m in params.length_candidates_m
        ]
        min_area_cells = int(round(params.min_area_m2 / (cell_size * cell_size)))

        all_candidates: list[RectangleCandidate] = []
        rotated_grid_count = 0
        rotation_sum_check: list[dict[str, object]] = []
        timed_out = False

        # v2: axes_override 면 (보통 axis pair) 그 angle 만 sweep. v1 호환은
        # axes_override=None → angles_deg full 18 sweep.
        axes_to_sweep: tuple[float, ...] = (
            params.axes_override
            if params.axes_override is not None
            else params.angles_deg
        )
        # v2: dynamic precision threshold 사용 시 thickness 별 임계 산출.
        threshold_by_thickness: dict[int, float] = {}
        for t_idx, t_m in enumerate(params.thicknesses_m):
            if params.precision_threshold_dynamic:
                threshold_by_thickness[t_idx] = compute_dynamic_precision_threshold(
                    thickness_m=float(t_m),
                    base=params.precision_threshold,
                    floor_min=params.precision_threshold_min,
                )
            else:
                threshold_by_thickness[t_idx] = params.precision_threshold

        for angle_deg in axes_to_sweep:
            if time.monotonic() > wall_deadline:
                timed_out = True
                break
            rotated, rot_meta = self._rotate_heatmap(
                heatmap=heatmap,
                origin=grid.origin,
                angle_deg=float(angle_deg),
            )
            rotated_grid_count += 1
            rotation_sum_check.append(rot_meta)
            integral = _integral_image(rotated)
            for t_idx, t_cells in enumerate(thickness_cells_list):
                if time.monotonic() > wall_deadline:
                    timed_out = True
                    break
                t_m = float(params.thicknesses_m[t_idx])
                tau_p_eff = threshold_by_thickness[t_idx]
                per_dim_top: list[RectangleCandidate] = []
                for l_idx, l_cells in enumerate(length_cells_list):
                    l_m = float(params.length_candidates_m[l_idx])
                    if l_cells < 1 or t_cells < 1:
                        continue
                    cands = self._sweep_axis_aligned(
                        rotated_shape=rotated.shape,
                        integral=integral,
                        rotated_origin_xy=rot_meta["rx0_ry0"],  # type: ignore[arg-type]
                        cell_size=cell_size,
                        angle_deg=float(angle_deg),
                        thickness_m=t_m,
                        thickness_cells=t_cells,
                        length_m=l_m,
                        length_cells=l_cells,
                        min_area_cells=min_area_cells,
                        source_heatmap=heatmap,
                        source_origin=grid.origin,
                        precision_threshold_eff=tau_p_eff,
                    )
                    per_dim_top.extend(cands)

                # max_candidates_per_dimension cap (BLOCKER 1)
                per_dim_top.sort(key=lambda r: r.hits, reverse=True)
                if len(per_dim_top) > params.max_candidates_per_dimension:
                    per_dim_top = per_dim_top[: params.max_candidates_per_dimension]
                all_candidates.extend(per_dim_top)

        # greedy
        selected, greedy_meta = self._greedy_cover(
            heatmap=heatmap,
            origin=grid.origin,
            candidates=all_candidates,
            total_hits=total_hits,
            wall_deadline=wall_deadline,
        )
        if greedy_meta.get("timed_out"):
            timed_out = True

        wall_time_sec = time.monotonic() - start_time

        # recall/over-cover 평가
        if not selected:
            metadata = {
                "rectangle_count": 0,
                "candidate_count": len(all_candidates),
                "rotated_grid_count": rotated_grid_count,
                "total_hits": total_hits,
                "covered_hits": 0,
                "uncovered_hits": total_hits,
                "recall": 0.0,
                "over_cover_ratio": 0.0,
                "wall_time_sec": float(wall_time_sec),
                "timed_out": bool(timed_out),
                "params": params.to_metadata(),
                "rotation_sum_preservation": rotation_sum_check[:4],
                # v2
                "axes_used": (
                    list(params.axes_override)
                    if params.axes_override is not None
                    else []
                ),
                "all_axes_in_pair": False,
                "axes_override_active": params.axes_override is not None,
                "precision_threshold_dynamic_active": bool(
                    params.precision_threshold_dynamic
                ),
            }
            return self._fallback_result(
                reason="no_candidate_passed_gate",
                metadata=metadata,
            )

        union, polygons = _union_world_polygons(
            [r.world_polygon for r in selected]
        )
        cover_mask, hits_inside_union = _world_polygons_to_grid_hit_count(
            polygons=polygons,
            heatmap=heatmap,
            origin=grid.origin,
        )
        covered_hits = int(hits_inside_union)
        uncovered_hits = total_hits - covered_hits
        union_area_cells = int(cover_mask.sum())
        target_cells_in_heatmap = int((heatmap > 0).sum())
        # over_cover: union 안 빈 cell / union 면적 (얼마나 빈 공간을 덮었나)
        over_cover_cells = max(0, union_area_cells - target_cells_in_heatmap)
        over_cover_ratio = (
            over_cover_cells / float(union_area_cells)
            if union_area_cells > 0
            else 0.0
        )
        recall = covered_hits / float(total_hits) if total_hits > 0 else 0.0

        accepted = (
            recall >= params.recall_min
            and over_cover_ratio <= params.over_cover_max
            and not timed_out
        )

        footprint_geojson = _polygons_to_geojson(union)

        selected_angles = sorted({float(round(r.angle_deg, 4)) for r in selected})
        selected_thicknesses = sorted({float(round(r.thickness_m, 4)) for r in selected})
        edges_axis_aligned = _all_edges_aligned_to_dictionary(
            polygons=polygons,
            allowed_angles_deg=selected_angles,
            tol_deg=2.0,
        )

        # v2: axis pair 보장 검증. axes_override 가 주입되어 있으면 selected
        # rectangles 의 angle 이 모두 axes_override set 안에 있는지 확인
        # (직각 보장의 핵심 메트릭).
        axes_used: list[float]
        all_axes_in_pair: bool
        if params.axes_override is not None:
            axes_set = {float(round(a % 180.0, 4)) for a in params.axes_override}
            axes_used = sorted(axes_set)
            all_axes_in_pair = all(
                float(round(r.angle_deg % 180.0, 4)) in axes_set
                for r in selected
            )
        else:
            axes_used = list(selected_angles)
            # v1 backward-compat: axes_override 없을 때는 selected angle 자체가
            # axes_used. 자기참조이므로 trivially True.
            all_axes_in_pair = True

        metadata = {
            "rectangle_count": len(selected),
            "candidate_count": len(all_candidates),
            "rotated_grid_count": rotated_grid_count,
            "total_hits": total_hits,
            "covered_hits": covered_hits,
            "uncovered_hits": uncovered_hits,
            "recall": float(recall),
            "over_cover_ratio": float(over_cover_ratio),
            "over_cover_cells": int(over_cover_cells),
            "union_area_cells": union_area_cells,
            "target_cells_in_heatmap": target_cells_in_heatmap,
            "selected_angles_unique": selected_angles,
            "selected_thicknesses_unique": selected_thicknesses,
            "all_edges_axis_aligned_to_dictionary": bool(edges_axis_aligned),
            # v2 신규
            "axes_used": axes_used,
            "all_axes_in_pair": bool(all_axes_in_pair),
            "axes_override_active": params.axes_override is not None,
            "precision_threshold_dynamic_active": bool(
                params.precision_threshold_dynamic
            ),
            "precision_threshold_by_thickness": {
                float(params.thicknesses_m[i]): float(threshold_by_thickness[i])
                for i in range(len(params.thicknesses_m))
            },
            "wall_time_sec": float(wall_time_sec),
            "timed_out": bool(timed_out),
            "rotation_sum_preservation": rotation_sum_check[:4],
            "params": params.to_metadata(),
            "rectangles": [r.metadata() for r in selected[:64]],
            "rectangles_truncated": max(0, len(selected) - 64),
            "iteration_log": greedy_meta.get("iteration_log", []),
        }

        if not accepted:
            return self._fallback_result(
                reason=(
                    "recall_or_over_cover_gate_failed"
                    if not timed_out
                    else "time_budget_exceeded"
                ),
                metadata=metadata,
                rectangles=selected,
                union_polygon=union,
                footprint_geojson=footprint_geojson,
            )

        logger.info(
            "rectangle_dictionary_cover accepted rects=%d candidates=%d "
            "recall=%.3f over=%.3f wall=%.2fs",
            len(selected),
            len(all_candidates),
            recall,
            over_cover_ratio,
            wall_time_sec,
        )
        return RectangleDictionaryCoverResult(
            rectangles=selected,
            union_polygon=union,
            footprint_geojson=footprint_geojson,
            accepted=True,
            fallback_used=False,
            fallback_source=None,
            metadata=metadata,
        )

    # ── 내부: 회전 + sweep ──────────────────────────────────────────────────

    def _rotate_heatmap(
        self,
        *,
        heatmap: np.ndarray,
        origin: GridOrigin,
        angle_deg: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """heatmap 을 -angle 만큼 회전한 axis-aligned grid 반환.

        cv2.warpAffine + INTER_NEAREST. angle=0 면 no-op fast path.
        Codex W-2: sum 보존 측정 metadata 에 기록.
        """
        if abs(angle_deg) < 1e-9:
            return heatmap.astype(np.int32, copy=False), {
                "angle_deg": 0.0,
                "source_sum": int(heatmap.sum()),
                "rotated_sum": int(heatmap.sum()),
                "preservation_ratio": 1.0,
                "rx0_ry0": (origin.x0, origin.y0),
                "rotated_shape": [int(heatmap.shape[0]), int(heatmap.shape[1])],
            }

        import cv2

        h, w = heatmap.shape
        cs = float(origin.cell_size)
        # heatmap world bbox: [origin.x0, origin.x0+w*cs] x [origin.y0, origin.y0+h*cs]
        # source grid index (col, row) → world (x, y) = origin + (col, row) * cs
        # rotation 은 world frame 에서 angle 만큼 (origin 0,0 기준) 회전.
        theta = math.radians(-angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        corners_world = np.array(
            [
                [origin.x0, origin.y0],
                [origin.x0 + w * cs, origin.y0],
                [origin.x0 + w * cs, origin.y0 + h * cs],
                [origin.x0, origin.y0 + h * cs],
            ],
            dtype=np.float64,
        )
        rotated_corners = np.empty_like(corners_world)
        rotated_corners[:, 0] = corners_world[:, 0] * cos_t - corners_world[:, 1] * sin_t
        rotated_corners[:, 1] = corners_world[:, 0] * sin_t + corners_world[:, 1] * cos_t
        pad = self._params.rotated_padding_cells * cs
        rx0 = float(np.floor((rotated_corners[:, 0].min() - pad) / cs) * cs)
        ry0 = float(np.floor((rotated_corners[:, 1].min() - pad) / cs) * cs)
        rx1 = float(np.ceil((rotated_corners[:, 0].max() + pad) / cs) * cs)
        ry1 = float(np.ceil((rotated_corners[:, 1].max() + pad) / cs) * cs)
        new_w = max(1, int(np.ceil((rx1 - rx0) / cs)))
        new_h = max(1, int(np.ceil((ry1 - ry0) / cs)))

        # Affine matrix (cv2 expects 2x3):
        # rotated_pixel = M @ source_pixel
        # source_pixel (col, row) → world (origin.x + col*cs, origin.y + row*cs)
        # → rotate(theta) → rotated_pixel ((rx-rx0)/cs, (ry-ry0)/cs)
        # = ((cos*x - sin*y - rx0)/cs, (sin*x + cos*y - ry0)/cs)
        # x = origin.x0 + col*cs, y = origin.y0 + row*cs.
        # So per-pixel: rotated_col = cos*(origin.x0+col*cs)-sin*(origin.y0+row*cs)-rx0,
        #               then divide by cs.
        # Simplify: M[0] = cos, M[1] = -sin, M[2] = (cos*origin.x0 - sin*origin.y0 - rx0)/cs
        # but cv2 uses col-major (x,y) where input pixel = (col, row) = (x_src, y_src).
        # warpAffine output[y_dst, x_dst] = input[y_src, x_src] where
        # (x_src, y_src) = M^-1 @ (x_dst, y_dst, 1).
        # We provide M such that (x_dst, y_dst) = M @ (x_src, y_src, 1).
        m00 = cos_t
        m01 = -sin_t
        m02 = (cos_t * origin.x0 - sin_t * origin.y0 - rx0) / cs
        m10 = sin_t
        m11 = cos_t
        m12 = (sin_t * origin.x0 + cos_t * origin.y0 - ry0) / cs
        affine = np.array([[m00, m01, m02], [m10, m11, m12]], dtype=np.float64)
        # heatmap shape (h, w). cv2 dsize = (W, H).
        rotated = cv2.warpAffine(
            heatmap.astype(np.int32, copy=False),
            affine,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rotated = rotated.astype(np.int32, copy=False)
        src_sum = int(heatmap.sum())
        rot_sum = int(rotated.sum())
        preservation = rot_sum / float(src_sum) if src_sum > 0 else 1.0
        return rotated, {
            "angle_deg": float(angle_deg),
            "source_sum": src_sum,
            "rotated_sum": rot_sum,
            "preservation_ratio": float(preservation),
            "rx0_ry0": (rx0, ry0),
            "rotated_shape": [int(new_h), int(new_w)],
        }

    def _sweep_axis_aligned(
        self,
        *,
        rotated_shape: tuple[int, int],
        integral: np.ndarray,
        rotated_origin_xy: tuple[float, float],
        cell_size: float,
        angle_deg: float,
        thickness_m: float,
        thickness_cells: int,
        length_m: float,
        length_cells: int,
        min_area_cells: int,
        source_heatmap: np.ndarray,
        source_origin: GridOrigin,
        precision_threshold_eff: float | None = None,
    ) -> list[RectangleCandidate]:
        """rotated grid 에서 axis-aligned (length_cells × thickness_cells) 후보 sweep.

        Codex W-2: rotation 보존 한계 (NEAREST 로 sum 1±5% 변동) 때문에
        rotated grid 의 hits 는 1차 prune 용으로만 쓰고, 후보 채택 시 hits/
        precision 은 source grid 에 polygon rasterize 한 정확값으로 재계산한다.

        candidate_stride_cells (default 3) 로 row/col 모두 stride. integral image
        로 1차 hits sum O(1). source-frame precision >= τ_p 인 후보만 통과.

        Sprint 50 v2: precision_threshold_eff 가 None 이면 params.precision_threshold
        그대로 (v1). dynamic mode 에서 호출자가 thickness 별 임계 주입.
        """
        params = self._params
        h, w = rotated_shape
        if length_cells > w or thickness_cells > h:
            return []
        area_cells = thickness_cells * length_cells
        if area_cells < max(1, min_area_cells):
            return []
        stride = params.candidate_stride_cells
        precision_threshold = (
            params.precision_threshold
            if precision_threshold_eff is None
            else float(precision_threshold_eff)
        )
        min_hits = params.min_hits
        rx0, ry0 = rotated_origin_xy

        # vectorized sweep:
        # rows = np.arange(0, h - thickness_cells + 1, stride)
        # cols = np.arange(0, w - length_cells + 1, stride)
        rows = np.arange(0, h - thickness_cells + 1, stride, dtype=np.int64)
        cols = np.arange(0, w - length_cells + 1, stride, dtype=np.int64)
        if rows.size == 0 or cols.size == 0:
            return []

        # integral has shape (h+1, w+1)
        # hits[r, c] = integral[r+t, c+l] - integral[r, c+l]
        #             - integral[r+t, c] + integral[r, c]
        rr = rows
        cc = cols
        # broadcast index to (len(rr), len(cc))
        r0 = rr[:, None]
        c0 = cc[None, :]
        r1 = r0 + thickness_cells
        c1 = c0 + length_cells
        hits_block = (
            integral[r1, c1]
            - integral[r0, c1]
            - integral[r1, c0]
            + integral[r0, c0]
        )
        # 1차 prune: rotated frame precision (rotation 으로 sum 변동 ±5%
        # 가능하므로 약간 느슨하게) + min_hits.
        rotated_precision_loose = max(0.0, precision_threshold - 0.10)
        precision_block_rot = hits_block / float(area_cells)
        mask_pass = (
            (hits_block >= min_hits)
            & (precision_block_rot >= rotated_precision_loose)
        )
        if not mask_pass.any():
            return []

        idx_r, idx_c = np.where(mask_pass)
        if idx_r.size == 0:
            return []

        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        candidates: list[RectangleCandidate] = []
        # python loop 이지만 후보 수 적음 (stride 3 + cap 200/dim).
        for i, j in zip(idx_r.tolist(), idx_c.tolist(), strict=False):
            r0_int = int(rows[i])
            c0_int = int(cols[j])
            r1_int = r0_int + thickness_cells
            c1_int = c0_int + length_cells
            # 4 corner in rotated_world frame
            rx_world_corners = np.array(
                [
                    [rx0 + c0_int * cell_size, ry0 + r0_int * cell_size],
                    [rx0 + c1_int * cell_size, ry0 + r0_int * cell_size],
                    [rx0 + c1_int * cell_size, ry0 + r1_int * cell_size],
                    [rx0 + c0_int * cell_size, ry0 + r1_int * cell_size],
                ],
                dtype=np.float64,
            )
            # rotated_world → world by +angle_deg rotation
            world = np.empty_like(rx_world_corners)
            world[:, 0] = rx_world_corners[:, 0] * cos_t - rx_world_corners[:, 1] * sin_t
            world[:, 1] = rx_world_corners[:, 0] * sin_t + rx_world_corners[:, 1] * cos_t
            poly = Polygon([(float(p[0]), float(p[1])) for p in world])
            if not poly.is_valid or poly.is_empty:
                continue
            # ── source frame 정확 재계산 (W-2 rotation sum 변동 회피) ──
            # precision 은 cell occupancy = hit_cells / mask_area.
            # hits (greedy marginal 비교용) 는 weight 합.
            source_mask, source_hit_cells, source_hits_weighted = (
                _world_polygons_to_hit_cells_count(
                    polygons=[poly],
                    heatmap=source_heatmap,
                    origin=source_origin,
                )
            )
            source_area = int(source_mask.sum())
            if source_area < max(1, min_area_cells):
                continue
            if source_hits_weighted < min_hits:
                continue
            source_precision = source_hit_cells / float(source_area)
            if source_precision < precision_threshold:
                continue
            candidates.append(
                RectangleCandidate(
                    angle_deg=float(angle_deg),
                    thickness_m=float(thickness_m),
                    length_m=float(length_m),
                    row0=r0_int,
                    col0=c0_int,
                    row1=r1_int,
                    col1=c1_int,
                    rotated_origin_x=float(rx0),
                    rotated_origin_y=float(ry0),
                    cell_size_m=float(cell_size),
                    hits=int(source_hits_weighted),
                    area_cells=int(source_area),
                    precision=float(source_precision),
                    world_polygon=poly,
                )
            )
        return candidates

    # ── 내부: greedy ────────────────────────────────────────────────────────

    def _greedy_cover(
        self,
        *,
        heatmap: np.ndarray,
        origin: GridOrigin,
        candidates: Sequence[RectangleCandidate],
        total_hits: int,
        wall_deadline: float,
    ) -> tuple[list[RectangleCandidate], dict[str, object]]:
        """W-3: marginal hits (uncovered) 만 재계산. precision 은 채택 시 static.

        각 후보의 source grid mask 를 lazy cache. 후보 polygon → source grid
        rasterize 1 회만 발생. covered_hits_mask 와 AND 만 매 iteration 반복.
        """
        params = self._params
        if not candidates:
            return [], {"iteration_log": [], "timed_out": False}

        covered_hits_mask = np.zeros(heatmap.shape, dtype=bool)
        mask_cache: list[np.ndarray | None] = [None] * len(candidates)

        selected: list[RectangleCandidate] = []
        iteration_log: list[dict[str, object]] = []
        timed_out = False
        # initial sort by static hits (greedy 첫 iteration 빠르게 best 찾기 도움)
        cand_indexed: list[tuple[int, RectangleCandidate]] = list(
            enumerate(candidates)
        )
        cand_indexed.sort(key=lambda kv: kv[1].hits, reverse=True)

        for iteration in range(params.max_iterations):
            if time.monotonic() > wall_deadline:
                timed_out = True
                break
            best_idx = -1
            best_marginal = 0
            best_cand: RectangleCandidate | None = None
            best_mask: np.ndarray | None = None
            for cand_idx, cand in cand_indexed:
                cached = mask_cache[cand_idx]
                if cached is None:
                    cached = _polygon_to_source_mask(
                        cand.world_polygon,
                        shape=heatmap.shape,
                        origin=origin,
                    )
                    mask_cache[cand_idx] = cached
                # marginal hits = sum of heatmap where (cached & ~covered)
                uncovered_in_cand = cached & (~covered_hits_mask)
                if not uncovered_in_cand.any():
                    continue
                marginal = int(heatmap[uncovered_in_cand].sum())
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_idx = cand_idx
                    best_cand = cand
                    best_mask = cached
            if best_cand is None or best_marginal < params.min_incremental_hits:
                break
            selected.append(best_cand)
            assert best_mask is not None
            covered_hits_mask |= best_mask
            iteration_log.append(
                {
                    "iteration": int(iteration),
                    "selected_candidate_idx": int(best_idx),
                    "marginal_hits": int(best_marginal),
                    "angle_deg": float(best_cand.angle_deg),
                    "thickness_m": float(best_cand.thickness_m),
                    "length_m": float(best_cand.length_m),
                }
            )
            covered_now = int(heatmap[covered_hits_mask].sum())
            if total_hits > 0 and covered_now / total_hits >= 0.98:
                break

        return selected, {
            "iteration_log": iteration_log,
            "timed_out": bool(timed_out),
        }

    # ── 내부: fallback 결과 ─────────────────────────────────────────────────

    def _fallback_result(
        self,
        *,
        reason: str,
        metadata: dict[str, object],
        rectangles: list[RectangleCandidate] | None = None,
        union_polygon: MultiPolygon | None = None,
        footprint_geojson: dict[str, object] | None = None,
    ) -> RectangleDictionaryCoverResult:
        meta = dict(metadata)
        meta["accepted"] = False
        meta["fallback_used"] = True
        meta["fallback_source"] = "sprint49_hint_chain"
        meta["fallback_reason"] = reason
        return RectangleDictionaryCoverResult(
            rectangles=rectangles or [],
            union_polygon=union_polygon,
            footprint_geojson=footprint_geojson,
            accepted=False,
            fallback_used=True,
            fallback_source="sprint49_hint_chain",
            metadata=meta,
        )


def split_heatmap_by_components(
    heatmap: np.ndarray,
    *,
    origin: GridOrigin,
    min_component_cells: int = 50,
) -> list[tuple[WalkableGrid, tuple[int, int, int, int]]]:
    """obs heatmap 을 connected component 별 sub-grid 로 분리.

    Sprint 50 v2 — 각 component (가로 hall / 세로 corridor 등) 마다 별도 dominant
    axis 추정 후 cover 를 수행하기 위한 헬퍼.

    Args:
        heatmap: 2D obs count.
        origin: heatmap 의 GridOrigin (cell_size, x0, y0).
        min_component_cells: 이 cell 수 미만 component 는 noise 로 간주, drop.

    Returns:
        list[(sub_grid, (row0, col0, row1, col1))]. row/col 은 inclusive-exclusive
        bbox of the component in source frame. sub_grid.origin 은 component 의
        bbox-aware 로 재계산된 GridOrigin.
    """
    from scipy.ndimage import label as _label

    if heatmap.ndim != 2 or heatmap.size == 0:
        return []
    binary = (heatmap > 0).astype(np.int32)
    if binary.sum() == 0:
        return []
    labeled, n_components = _label(binary)
    out: list[tuple[WalkableGrid, tuple[int, int, int, int]]] = []
    cs = float(origin.cell_size)
    for label_id in range(1, n_components + 1):
        comp_mask = labeled == label_id
        if int(comp_mask.sum()) < min_component_cells:
            continue
        rows = np.where(comp_mask.any(axis=1))[0]
        cols = np.where(comp_mask.any(axis=0))[0]
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        sub_heatmap = np.where(
            comp_mask[r0:r1, c0:c1],
            heatmap[r0:r1, c0:c1],
            0,
        ).astype(heatmap.dtype, copy=False)
        sub_origin = GridOrigin(
            x0=float(origin.x0 + c0 * cs),
            y0=float(origin.y0 + r0 * cs),
            z0=float(origin.z0),
            cell_size=cs,
            w=int(c1 - c0),
            h=int(r1 - r0),
        )
        sub_grid = WalkableGrid(
            origin=sub_origin,
            mask=(sub_heatmap > 0),
            observation_count=sub_heatmap.astype(np.uint16, copy=False),
        )
        out.append((sub_grid, (r0, c0, r1, c1)))
    return out


def estimate_dominant_axis_from_links(
    link_segments_xy: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    min_best_bin_ratio: float = 0.10,
    bin_deg: float = 5.0,
) -> tuple[float, float] | None:
    """RTABMap link segment list 에서 length-weighted dominant angle.

    dominant_angle_hint 의 RTABMap 게이트보다 완화된 (0.10) 임계.
    Returns:
        (folded_angle_deg, best_bin_ratio) 또는 None.
    """
    if not link_segments_xy:
        return None
    n_bins = max(1, int(round(90.0 / bin_deg)))
    weights = np.zeros(n_bins, dtype=np.float64)
    angle_sum = np.zeros(n_bins, dtype=np.float64)
    total_length = 0.0
    for (ax, ay), (bx, by) in link_segments_xy:
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length < 0.05:
            continue
        total_length += length
        # fold to [-45, 45]
        raw = math.degrees(math.atan2(dy, dx))
        # axis fold (mod 90, center 0)
        folded = ((raw + 45.0) % 90.0) - 45.0
        idx = int(np.floor((folded + 45.0) / bin_deg))
        idx = max(0, min(n_bins - 1, idx))
        weights[idx] += length
        angle_sum[idx] += folded * length
    total_weight = float(weights.sum())
    if total_weight <= 1e-9:
        return None
    best_idx = int(np.argmax(weights))
    best_bin_ratio = float(weights[best_idx]) / total_weight
    if best_bin_ratio < min_best_bin_ratio:
        return None
    folded = float(angle_sum[best_idx] / weights[best_idx])
    return folded, best_bin_ratio


def estimate_dominant_axis_from_obb(
    polygon: Polygon | MultiPolygon,
    *,
    min_aspect_ratio: float = 1.2,
) -> tuple[float, float] | None:
    """polygon OBB long edge angle.

    Returns:
        (folded_angle_deg, aspect_ratio) 또는 None (aspect 부족).
    """
    if polygon is None or polygon.is_empty or polygon.area <= 0:
        return None
    try:
        if hasattr(polygon, "oriented_envelope"):
            obb = polygon.oriented_envelope
        else:
            obb = polygon.minimum_rotated_rectangle
    except Exception:
        return None
    coords = list(obb.exterior.coords)
    if len(coords) < 5:
        return None
    edges: list[tuple[float, float]] = []
    for (ax, ay), (bx, by) in zip(coords, coords[1:], strict=False):
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        edges.append((length, ang))
    if not edges:
        return None
    edges.sort(key=lambda t: t[0], reverse=True)
    long_len, long_ang = edges[0]
    short_len = edges[-1][0]
    aspect = long_len / short_len if short_len > 1e-9 else float("inf")
    if aspect < min_aspect_ratio:
        return None
    folded = ((long_ang + 45.0) % 90.0) - 45.0
    return folded, float(aspect)


def axes_pair_from_primary(primary_deg: float) -> tuple[float, float]:
    """주축 + 90° 직교축 pair (axes_override 주입용).

    primary 는 임의 각도로 들어와도 [0, 180) 으로 정규화 후 pair 반환.
    """
    p = float(primary_deg) % 180.0
    q = (p + 90.0) % 180.0
    return (p, q)


# ── helpers ─────────────────────────────────────────────────────────────────


def _integral_image(arr: np.ndarray) -> np.ndarray:
    values = arr.astype(np.int64, copy=False)
    return np.pad(values, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _union_world_polygons(
    polygons: list[Polygon],
) -> tuple[MultiPolygon | None, list[Polygon]]:
    if not polygons:
        return None, []
    union = unary_union(polygons)
    if isinstance(union, Polygon):
        if union.is_empty:
            return None, []
        return MultiPolygon([union]), [union]
    if hasattr(union, "geoms"):
        geoms = [g for g in union.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not geoms:
            return None, []
        return MultiPolygon(geoms), geoms
    return None, []


def _polygons_to_geojson(union: MultiPolygon | None) -> dict[str, object] | None:
    if union is None or union.is_empty:
        return None
    return dict(mapping(union))


def _world_polygons_to_grid_hit_count(
    *,
    polygons: list[Polygon],
    heatmap: np.ndarray,
    origin: GridOrigin,
) -> tuple[np.ndarray, int]:
    """world polygon list → source grid mask + sum(heatmap[mask]).

    여기서 hits 는 weight 합 (greedy marginal 비교용). cell 점유율 (precision)
    은 호출자가 별도로 (heatmap[mask] > 0).sum() 으로 계산한다.
    """
    import cv2

    h, w = heatmap.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cs = float(origin.cell_size)
    x0 = float(origin.x0)
    y0 = float(origin.y0)
    for polygon in polygons:
        if polygon.is_empty:
            continue
        exterior = np.asarray(
            [
                [
                    int(round((x - x0) / cs)),
                    int(round((y - y0) / cs)),
                ]
                for x, y in polygon.exterior.coords
            ],
            dtype=np.int32,
        )
        if exterior.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [exterior], color=1)
        for interior in polygon.interiors:
            hole = np.asarray(
                [
                    [
                        int(round((x - x0) / cs)),
                        int(round((y - y0) / cs)),
                    ]
                    for x, y in interior.coords
                ],
                dtype=np.int32,
            )
            if hole.shape[0] >= 3:
                cv2.fillPoly(mask, [hole], color=0)
    bool_mask = mask.astype(bool)
    hits_inside = int((bool_mask.astype(np.int64) * heatmap).sum())
    return bool_mask, hits_inside


def _world_polygons_to_hit_cells_count(
    *,
    polygons: list[Polygon],
    heatmap: np.ndarray,
    origin: GridOrigin,
) -> tuple[np.ndarray, int, int]:
    """world polygon list → source grid mask + hit_cells (heatmap>0 & mask).

    반환: (mask, hit_cells_count, hits_weighted_sum).
    precision 은 hit_cells / mask.sum() 으로 호출자가 계산.
    """
    mask, hits_weighted = _world_polygons_to_grid_hit_count(
        polygons=polygons, heatmap=heatmap, origin=origin
    )
    target_mask = heatmap > 0
    hit_cells = int(np.logical_and(mask, target_mask).sum())
    return mask, hit_cells, hits_weighted


def _polygon_to_source_mask(
    polygon: Polygon,
    *,
    shape: tuple[int, int],
    origin: GridOrigin,
) -> np.ndarray:
    """단일 polygon → source grid mask."""
    mask, _ = _world_polygons_to_grid_hit_count(
        polygons=[polygon],
        heatmap=np.zeros(shape, dtype=np.int64),
        origin=origin,
    )
    return mask


def _all_edges_aligned_to_dictionary(
    *,
    polygons: list[Polygon],
    allowed_angles_deg: list[float],
    tol_deg: float,
) -> bool:
    """polygon edges 가 selected angle set + 그 직교축 (±90°) 에 들어 있는가."""
    if not polygons:
        return False
    canonical = set()
    for a in allowed_angles_deg:
        normalized = a % 180.0
        canonical.add(round(normalized, 4))
        canonical.add(round((normalized + 90.0) % 180.0, 4))
    for poly in polygons:
        coords = list(poly.exterior.coords)
        for (x0, y0), (x1, y1) in zip(coords, coords[1:], strict=False):
            dx = x1 - x0
            dy = y1 - y0
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            ok = False
            for ref in canonical:
                diff = abs(angle - ref)
                diff = min(diff, 180.0 - diff)
                if diff <= tol_deg:
                    ok = True
                    break
            if not ok:
                return False
    return True
