"""Multi-view scale calibration step (Sprint 20).

STATUS: Sprint 20 실패 — depth NN relative depth의 noise가 ray 방향으로
나타나고 baseline은 ray와 수직이라 scale 수식 `s = -Σ(diff·offset)/Σ||diff||²`
가 s ≈ 0 으로 수렴. triangulation 경로로 대체됨 (see `triangulation.py`).
이 파일은 추후 multi-view bundle adjustment 재도전 시 참조 역사로 보존.

목표:
    Sprint 19 ScaleCalibrator는 frame 독립적으로 scale을 산출 → 일부 frame 실패,
    성공 frame 간에도 변동 큼 → layout union 삐뚤어짐.
    SuperPoint+LightGlue로 인접 frame 간 matched keypoint를 뽑아,
    매칭된 점을 양쪽 frame에서 back-project한 3D 좌표가 같아야 한다는 제약으로
    per-frame scale을 전역적으로 최적화한다.

관찰:
    relative depth d_rel, pose (R_i, t_i), intrinsics 주어진 상태에서
        P_i = (R_i @ ray_i / ||ray_i||) * (s_i / d_rel_i) + t_i
    scale s_i에 대해 **선형**. 따라서 match residual (P_i - P_j)도 scale에 선형 →
    numpy.linalg.lstsq로 직접 해도 수렴.
    다만 outlier(잘못된 매칭, 움직이는 물체)에 약하므로 scipy.optimize.least_squares +
    Huber loss로 robust 해를 구한다.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from indoor_server.application.building.steps.back_projection import (
    Intrinsics,
    decode_pose_matrix,
    default_intrinsics,
)

if TYPE_CHECKING:
    from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks
    from indoor_server.domain.building.models import DepthMap
    from indoor_server.infrastructure.ml.superpoint_lightglue import SuperPointLightGlueRunner

logger = logging.getLogger(__name__)


class MatchObservation:
    """단일 matched keypoint의 잔차 관측. scale 최적화의 원자 단위."""

    __slots__ = ("i", "j", "coeff_i", "coeff_j", "offset")

    def __init__(
        self,
        i: int,
        j: int,
        coeff_i: np.ndarray,
        coeff_j: np.ndarray,
        offset: np.ndarray,
    ) -> None:
        # residual(x) = coeff_i * s_i - coeff_j * s_j + offset  (3-vec)
        # where coeff = R @ (ray / ||ray||) / d_rel   (3-vec)
        # offset = t_i - t_j
        self.i = i  # frame index
        self.j = j
        self.coeff_i = coeff_i  # (3,)
        self.coeff_j = coeff_j  # (3,)
        self.offset = offset    # (3,)


class MultiViewMatchResult:
    """step.extract_matches 결과 묶음. 시각화 / 디버그에도 재사용."""

    __slots__ = ("observations", "pair_stats", "frames_with_matches")

    def __init__(
        self,
        observations: list[MatchObservation],
        pair_stats: list[dict[str, int | float]],
        frames_with_matches: set[int],
    ) -> None:
        self.observations = observations
        self.pair_stats = pair_stats
        self.frames_with_matches = frames_with_matches


class MultiViewScaleCalibrationStep:
    """SuperPoint+LightGlue로 추출한 매칭으로 per-frame scale을 전역 최적화.

    사용 흐름:
        step = MultiViewScaleCalibrationStep(sp_lg_runner, window=5)
        match_result = await step.extract_matches(masks, depth_maps, storage_root)
        scales = step.optimize_scales(match_result, initial_scales, poses, intrinsics)
    """

    def __init__(
        self,
        sp_lg_runner: SuperPointLightGlueRunner,
        window: int = 5,
        min_match_score: float = 0.8,
        max_pair_matches: int = 32,
    ) -> None:
        self._runner = sp_lg_runner
        self._window = window
        self._min_score = min_match_score
        self._max_pair_matches = max_pair_matches

    async def extract_matches(
        self,
        masks: list[KeyframeMasks],
        depth_maps_by_seq: dict[int, np.ndarray],
        storage_root: Path,
        intrinsics_by_frame: list[Intrinsics],
        progress: Callable[[float], Awaitable[None]] | None = None,
    ) -> MultiViewMatchResult:
        """인접 window 내 pair SuperPoint+LightGlue 매칭 → MatchObservation 리스트.

        masks: frame_idx 순서 list. masks[i].scan_id/seq로 이미지 경로 구성.
        depth_maps_by_seq: seq → (H, W) float32 relative depth.
        """
        import cv2

        n = len(masks)
        observations: list[MatchObservation] = []
        pair_stats: list[dict[str, int | float]] = []
        frames_with_matches: set[int] = set()

        # poses/t pre-decode
        poses = [decode_pose_matrix(m.pose_matrix) for m in masks]

        # 이미지 캐시 (pair i,j 에서 i가 window 내 여러 pair에 공통)
        img_cache: dict[int, np.ndarray] = {}

        def _load(idx: int) -> np.ndarray | None:
            if idx in img_cache:
                return img_cache[idx]
            m = masks[idx]
            img_path = storage_root / f"scans/{m.scan_id}/keyframes/{m.seq:06d}.jpg"
            if not img_path.exists():
                return None
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                return None
            img_cache[idx] = bgr
            if len(img_cache) > (self._window + 2):
                oldest = min(img_cache.keys())
                if oldest < idx - self._window:
                    del img_cache[oldest]
            return bgr

        total_pairs = sum(min(self._window, n - 1 - i) for i in range(n - 1))
        done = 0

        for i in range(n - 1):
            img_a = _load(i)
            if img_a is None:
                continue
            seq_i = masks[i].seq
            if seq_i not in depth_maps_by_seq:
                continue
            depth_i = depth_maps_by_seq[seq_i]
            intrin_i = intrinsics_by_frame[i]
            pose_i = poses[i]

            for delta in range(1, self._window + 1):
                j = i + delta
                if j >= n:
                    break
                img_b = _load(j)
                if img_b is None:
                    continue
                seq_j = masks[j].seq
                if seq_j not in depth_maps_by_seq:
                    continue
                depth_j = depth_maps_by_seq[seq_j]
                intrin_j = intrinsics_by_frame[j]
                pose_j = poses[j]

                mp = await self._runner.match(img_a, img_b)
                done += 1
                if progress is not None and done % 10 == 0:
                    await progress(done / max(total_pairs, 1))

                if mp.matches.shape[0] == 0:
                    pair_stats.append({"i": i, "j": j, "matches": 0, "used": 0})
                    continue

                # score filter
                score_ok = mp.scores >= self._min_score
                sel = mp.matches[score_ok]
                sel_scores = mp.scores[score_ok]

                # top-K by score
                if len(sel) > self._max_pair_matches:
                    order = np.argsort(-sel_scores)[: self._max_pair_matches]
                    sel = sel[order]

                if len(sel) == 0:
                    pair_stats.append({"i": i, "j": j, "matches": int(mp.matches.shape[0]), "used": 0})
                    continue

                # model input → sensor resolution scale
                img_h_a, img_w_a = img_a.shape[:2]
                img_h_b, img_w_b = img_b.shape[:2]
                sx_a = img_w_a / mp.size_a[1]
                sy_a = img_h_a / mp.size_a[0]
                sx_b = img_w_b / mp.size_b[1]
                sy_b = img_h_b / mp.size_b[0]

                used = 0
                for idx_a, idx_b in sel:
                    xa = mp.kpts_a[idx_a, 0] * sx_a
                    ya = mp.kpts_a[idx_a, 1] * sy_a
                    xb = mp.kpts_b[idx_b, 0] * sx_b
                    yb = mp.kpts_b[idx_b, 1] * sy_b

                    # depth 샘플 (bilinear은 비용 큼, 정수 nearest로 충분)
                    uai, vai = int(round(xa)), int(round(ya))
                    ubi, vbi = int(round(xb)), int(round(yb))
                    if not (0 <= vai < depth_i.shape[0] and 0 <= uai < depth_i.shape[1]):
                        continue
                    if not (0 <= vbi < depth_j.shape[0] and 0 <= ubi < depth_j.shape[1]):
                        continue
                    d_rel_i = float(depth_i[vai, uai])
                    d_rel_j = float(depth_j[vbi, ubi])
                    if d_rel_i < 1e-4 or d_rel_j < 1e-4:
                        continue

                    # OpenCV ray convention. diag 결과 pose rotation은 R.T를 써야
                    # world ray가 나온다 (stored pose가 view matrix / cam_T_world 성격).
                    ray_i = np.array(
                        [(xa - intrin_i.cx) / intrin_i.fx,
                         (ya - intrin_i.cy) / intrin_i.fy,
                         1.0],
                        dtype=np.float64,
                    )
                    ray_j = np.array(
                        [(xb - intrin_j.cx) / intrin_j.fx,
                         (yb - intrin_j.cy) / intrin_j.fy,
                         1.0],
                        dtype=np.float64,
                    )
                    ri_norm = ray_i / np.linalg.norm(ray_i)
                    rj_norm = ray_j / np.linalg.norm(ray_j)

                    coeff_i_world = pose_i[:3, :3].T @ (ri_norm / d_rel_i)
                    coeff_j_world = pose_j[:3, :3].T @ (rj_norm / d_rel_j)
                    offset = pose_i[:3, 3] - pose_j[:3, 3]

                    observations.append(
                        MatchObservation(
                            i=i,
                            j=j,
                            coeff_i=coeff_i_world,
                            coeff_j=coeff_j_world,
                            offset=offset,
                        )
                    )
                    frames_with_matches.add(i)
                    frames_with_matches.add(j)
                    used += 1

                pair_stats.append({"i": i, "j": j, "matches": int(mp.matches.shape[0]), "used": int(used)})

        logger.info(
            "multiview match done pairs=%d observations=%d frames_with_matches=%d",
            len(pair_stats),
            len(observations),
            len(frames_with_matches),
        )
        return MultiViewMatchResult(
            observations=observations,
            pair_stats=pair_stats,
            frames_with_matches=frames_with_matches,
        )

    def optimize_scales(
        self,
        match_result: MultiViewMatchResult,
        initial_scales: dict[int, float],
        n_frames: int,
        huber_rounds: int = 3,
        huber_delta: float = 0.5,
    ) -> dict[int, float]:
        """match observations를 기반으로 **단일 global scale** 추정.

        Algorithm: closed-form weighted LSQ + IRLS(Huber) — scipy optimizer 불필요.
            minimize  sum_i w_i * || coeff_diff_i * s + offset_i ||^2
            → s = -sum(w * diff · offset) / sum(w * diff · diff)
        w_i: Huber weight based on previous iteration's residual.

        Returns:
            frame_idx → same global scale (for all frames).
        """
        valid_scales = [v for v in initial_scales.values() if v is not None and v > 0]
        initial_median = float(np.median(valid_scales)) if valid_scales else None
        logger.info(
            "multiview_optimize: %d valid initial scales, median=%s",
            len(valid_scales),
            f"{initial_median:.4f}" if initial_median is not None else "n/a",
        )

        obs = match_result.observations
        if not obs:
            logger.warning("multiview_optimize: no observations → returning init or 1.0")
            fallback = initial_median if initial_median is not None else 1.0
            return {k: fallback for k in range(n_frames)}

        coeffs_i = np.stack([o.coeff_i for o in obs], axis=0)
        coeffs_j = np.stack([o.coeff_j for o in obs], axis=0)
        diff_coeff = coeffs_i - coeffs_j                      # (N, 3)
        offsets = np.stack([o.offset for o in obs], axis=0)   # (N, 3)

        # closed-form LSQ
        num_raw = float(np.sum(diff_coeff * offsets))   # Σ diff · offset (flat dot over N*3)
        den_raw = float(np.sum(diff_coeff * diff_coeff))
        print(
            f"[multiview] obs={len(obs)} diff_coeff p50_norm="
            f"{float(np.median(np.linalg.norm(diff_coeff, axis=1))):.4f} "
            f"offset p50_norm={float(np.median(np.linalg.norm(offsets, axis=1))):.4f} "
            f"num_raw={num_raw:.4f} den_raw={den_raw:.4f}",
            flush=True,
        )
        if den_raw <= 0:
            logger.warning("multiview_optimize: singular design (den=0)")
            fallback = initial_median if initial_median is not None else 1.0
            return {k: fallback for k in range(n_frames)}
        s_ls = -num_raw / den_raw

        # IRLS with Huber weights
        s = s_ls
        weights = np.ones(len(obs), dtype=np.float64)
        for r in range(huber_rounds):
            res = diff_coeff * s + offsets          # (N, 3)
            rn = np.linalg.norm(res, axis=1)        # (N,)
            weights = np.where(rn > huber_delta, huber_delta / (rn + 1e-12), 1.0)
            w_diff = diff_coeff * weights[:, None]
            num = float(np.sum(w_diff * offsets))
            den = float(np.sum(w_diff * diff_coeff))
            if den <= 0:
                break
            s = -num / den

        # 통계 로깅
        res_final = diff_coeff * s + offsets
        rn_final = np.linalg.norm(res_final, axis=1)
        print(
            f"[multiview] s_ls={s_ls:.4f} → s_irls={s:.4f}  "
            f"residual p50={float(np.median(rn_final)):.3f}m p95={float(np.percentile(rn_final, 95)):.3f}m  "
            f"w_mean={float(np.mean(weights)):.2f}",
            flush=True,
        )

        if s <= 0:
            logger.warning("multiview_optimize: negative global scale (%.4g) — geometry convention mismatch", s)

        return {k: float(s) for k in range(n_frames)}
