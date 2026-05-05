"""2-view linear triangulation step (Sprint 20).

SuperPoint+LightGlue로 추출한 matched keypoint pair를 cv2.triangulatePoints로
3D화하여 WalkableGridStep 입력 호환 형식(frame별 (N_i, 3) float64)으로 반환.

depth NN 의존성 없음. ARKit pose metric scale을 그대로 활용.

ARKit → OpenCV projection matrix 변환:
    ARKit ray convention : [(u-cx)/fx, -(v-cy)/fy, -1]  (+X right, +Y up, -Z forward)
    OpenCV ray convention: [(u-cx)/fx,  (v-cy)/fy, +1]  (+X right, +Y down, +Z forward)
    차이: Y-flip + Z-flip → D = diag(1, -1, -1)

    world_T_camera(ARKit) pose[:3,:3] = R_arkit, pose[:3,3] = t_arkit
    R_ocv_world_from_cam = R_arkit @ D
    camera_from_world_R  = R_ocv_world_from_cam.T
    camera_from_world_t  = -camera_from_world_R @ t_arkit
    P = K @ [camera_from_world_R | camera_from_world_t]   (3, 4)

픽셀 (u, v)은 ARKit convention 원본 그대로 넘김 — 변환은 전부 P에 흡수.
cv2.triangulatePoints 반환 world 좌표는 이미 ARKit world frame.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from indoor_server.application.building.steps.back_projection import (
    Intrinsics,
    decode_pose_matrix,
)

if TYPE_CHECKING:
    from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks
    from indoor_server.infrastructure.ml.superpoint_lightglue import SuperPointLightGlueRunner

logger = logging.getLogger(__name__)

# ARKit ↔ OpenCV convention 변환 행렬
_D = np.diag([1.0, -1.0, -1.0])  # diag(1, -1, -1)

# 기본 필터 파라미터
_DEFAULT_MIN_BASELINE_M = 0.05
_DEFAULT_MAX_REPROJ_ERROR_PX = 15.0  # SuperPoint kp가 model 384 → sensor 1920 스케일 5×
                                      # 돼 sub-pixel 1px가 5px로 확대됨. 3px은 97% 제거.
_DEFAULT_MAX_DISTANCE_M = 15.0
_DEFAULT_Z_TOLERANCE_M = 0.30
_DEFAULT_FLOOR_MASK_DILATE_PX = 10
_DEFAULT_MIN_MATCH_SCORE = 0.80
_DEFAULT_MAX_PAIR_MATCHES = 64


class TriangulatedCloud:
    """TriangulationStep 결과. per-frame 점 집합 + 전체 통계."""

    __slots__ = ("points_per_frame", "pair_stats", "total_raw", "total_kept")

    def __init__(
        self,
        points_per_frame: list[np.ndarray],
        pair_stats: list[dict[str, int | float | str]],
        total_raw: int,
        total_kept: int,
    ) -> None:
        self.points_per_frame = points_per_frame  # list[(N_i, 3) float64]
        # [{"i", "j", "matches", "triangulated", "kept", "baseline_m"}]
        self.pair_stats = pair_stats
        self.total_raw = total_raw
        self.total_kept = total_kept


class TriangulationStep:
    """SuperPoint+LightGlue matched kp pair → cv2.triangulatePoints → world 3D.

    출력은 기존 BackProjectionStep 호환 shape: frame별 (N_i, 3) float64.
    WalkableGridStep이 그대로 소비.

    5-stage 필터:
        0. floor-mask gating (dilated 10px)
        1. DLT degeneracy (|w| < 1e-6)
        2. reprojection error (> 3px)
        3. z-tolerance (|z - z0| > 0.30m)
        4. distance cap (dist > 15m)
    """

    def __init__(
        self,
        sp_lg_runner: SuperPointLightGlueRunner,
        window: int = 5,
        min_match_score: float = _DEFAULT_MIN_MATCH_SCORE,
        max_pair_matches: int = _DEFAULT_MAX_PAIR_MATCHES,
        min_baseline_m: float = _DEFAULT_MIN_BASELINE_M,
        max_reprojection_error_px: float = _DEFAULT_MAX_REPROJ_ERROR_PX,
        max_distance_m: float = _DEFAULT_MAX_DISTANCE_M,
        z_tolerance_m: float = _DEFAULT_Z_TOLERANCE_M,
        floor_mask_dilate_px: int = _DEFAULT_FLOOR_MASK_DILATE_PX,
        enable_floor_mask_gate: bool = True,
    ) -> None:
        self._runner = sp_lg_runner
        self._window = window
        self._min_score = min_match_score
        self._max_pair_matches = max_pair_matches
        self._min_baseline_m = min_baseline_m
        self._max_reproj_px = max_reprojection_error_px
        self._max_distance_m = max_distance_m
        self._z_tolerance_m = z_tolerance_m
        self._dilate_px = floor_mask_dilate_px
        self._enable_floor_gate = enable_floor_mask_gate

    async def run(
        self,
        masks: list[KeyframeMasks],
        storage_root: Path,
        intrinsics_by_frame: list[Intrinsics],
        z0: float,
        progress: Callable[[float], Coroutine[object, object, None]] | None = None,
    ) -> TriangulatedCloud:
        """masks 순서대로 window 내 pair 매칭 → triangulate → 필터 → TriangulatedCloud."""
        n = len(masks)
        points_per_frame: list[list[np.ndarray]] = [[] for _ in range(n)]
        pair_stats: list[dict[str, int | float | str]] = []
        total_raw = 0
        total_kept = 0
        total_dlt_drop = 0
        total_reproj_drop = 0
        total_ztol_drop = 0
        total_dist_drop = 0

        poses = [decode_pose_matrix(m.pose_matrix) for m in masks]
        projs = [_build_projection_matrix(poses[i], intrinsics_by_frame[i]) for i in range(n)]

        img_cache: dict[int, np.ndarray] = {}
        dilated_masks: dict[int, np.ndarray] = {}

        def _load_image(idx: int) -> np.ndarray | None:
            if idx in img_cache:
                return img_cache[idx]
            m = masks[idx]
            img_path = storage_root / f"scans/{m.scan_id}/keyframes/{m.seq:06d}.jpg"
            if not img_path.exists():
                return None
            try:
                import cv2
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    return None
                img_cache[idx] = bgr
                # evict old entries (window보다 오래된 frame 있을 때만)
                if len(img_cache) > self._window + 3:
                    evictable = [k for k in img_cache if k < idx - self._window]
                    if evictable:
                        img_cache.pop(min(evictable), None)
                return bgr
            except Exception as e:
                logger.debug("triangulation: image load error idx=%d: %s", idx, e)
                return None

        def _dilated_mask(idx: int) -> np.ndarray:
            if idx in dilated_masks:
                return dilated_masks[idx]
            mask = masks[idx].floor_mask
            if self._dilate_px > 0:
                try:
                    import cv2
                    kernel_size = self._dilate_px * 2 + 1
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                    dilated_u8 = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
                    dilated = dilated_u8.astype(bool)
                except Exception:
                    dilated = mask
            else:
                dilated = mask
            dilated_masks[idx] = dilated
            return dilated

        total_pairs = sum(min(self._window, n - 1 - i) for i in range(n - 1))
        done_pairs = 0

        baselines: list[float] = []
        reproj_errors: list[float] = []

        for i in range(n - 1):
            img_a = _load_image(i)
            if img_a is None:
                continue
            mask_a = _dilated_mask(i)
            proj_i = projs[i]
            t_i = poses[i][:3, 3]

            for delta in range(1, self._window + 1):
                j = i + delta
                if j >= n:
                    break

                # baseline 필터 (pair 전체 skip)
                t_j = poses[j][:3, 3]
                baseline_m = float(np.linalg.norm(t_i - t_j))
                if baseline_m < self._min_baseline_m:
                    pair_stats.append({
                        "i": int(i), "j": int(j), "baseline_m": baseline_m,
                        "matches": 0, "triangulated": 0, "kept": 0,
                        "skip_reason": "baseline_too_small",
                    })
                    done_pairs += 1
                    if progress is not None and done_pairs % 10 == 0:
                        await progress(done_pairs / max(total_pairs, 1))
                    continue

                img_b = _load_image(j)
                if img_b is None:
                    done_pairs += 1
                    continue
                mask_b = _dilated_mask(j)
                proj_j = projs[j]

                mp = await self._runner.match(img_a, img_b)
                done_pairs += 1
                if progress is not None and done_pairs % 10 == 0:
                    await progress(done_pairs / max(total_pairs, 1))

                if mp.matches.shape[0] == 0:
                    pair_stats.append({
                        "i": i, "j": j, "baseline_m": baseline_m,
                        "matches": 0, "triangulated": 0, "kept": 0,
                    })
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
                    pair_stats.append({
                        "i": i, "j": j, "baseline_m": baseline_m,
                        "matches": int(mp.matches.shape[0]), "triangulated": 0, "kept": 0,
                    })
                    continue

                # model → sensor resolution scale
                img_h_a, img_w_a = img_a.shape[:2]
                img_h_b, img_w_b = img_b.shape[:2]
                sx_a = img_w_a / mp.size_a[1]
                sy_a = img_h_a / mp.size_a[0]
                sx_b = img_w_b / mp.size_b[1]
                sy_b = img_h_b / mp.size_b[0]

                # stage 0: floor-mask gating
                kps_i_list: list[tuple[float, float]] = []
                kps_j_list: list[tuple[float, float]] = []
                for idx_a, idx_b in sel:
                    xa = float(mp.kpts_a[idx_a, 0]) * sx_a
                    ya = float(mp.kpts_a[idx_a, 1]) * sy_a
                    xb = float(mp.kpts_b[idx_b, 0]) * sx_b
                    yb = float(mp.kpts_b[idx_b, 1]) * sy_b

                    ui, vi = int(round(xa)), int(round(ya))
                    uj, vj = int(round(xb)), int(round(yb))

                    if self._enable_floor_gate:
                        in_a = (
                            (0 <= vi < mask_a.shape[0])
                            and (0 <= ui < mask_a.shape[1])
                            and mask_a[vi, ui]
                        )
                        in_b = (
                            (0 <= vj < mask_b.shape[0])
                            and (0 <= uj < mask_b.shape[1])
                            and mask_b[vj, uj]
                        )
                        if not (in_a and in_b):
                            continue
                    kps_i_list.append((xa, ya))
                    kps_j_list.append((xb, yb))

                n_matched = len(kps_i_list)
                if n_matched == 0:
                    pair_stats.append({
                        "i": i, "j": j, "baseline_m": baseline_m,
                        "matches": int(mp.matches.shape[0]), "triangulated": 0, "kept": 0,
                    })
                    continue

                # triangulate
                kps_i_arr = np.array(kps_i_list, dtype=np.float32).T  # (2, N)
                kps_j_arr = np.array(kps_j_list, dtype=np.float32).T  # (2, N)

                try:
                    import cv2
                    pts_homo = cv2.triangulatePoints(
                        proj_i.astype(np.float32),
                        proj_j.astype(np.float32),
                        kps_i_arr,
                        kps_j_arr,
                    )  # (4, N)
                except Exception as e:
                    logger.debug("triangulatePoints failed pair(%d,%d): %s", i, j, e)
                    pair_stats.append({
                        "i": i, "j": j, "baseline_m": baseline_m,
                        "matches": n_matched, "triangulated": 0, "kept": 0,
                    })
                    continue

                pts_homo_f64 = pts_homo.astype(np.float64)
                w_vals = pts_homo_f64[3, :]

                # stage 1: DLT degeneracy
                valid_w = np.abs(w_vals) > 1e-6
                stage_dlt_drop = int((~valid_w).sum())
                total_dlt_drop += stage_dlt_drop
                if valid_w.sum() == 0:
                    pair_stats.append({
                        "i": i, "j": j, "baseline_m": baseline_m,
                        "matches": n_matched, "triangulated": 0, "kept": 0,
                    })
                    continue

                pts_world = (pts_homo_f64[:3, valid_w] / w_vals[valid_w][np.newaxis, :]).T  # (M, 3)
                kps_i_valid = kps_i_arr[:, valid_w].T   # (M, 2)
                kps_j_valid = kps_j_arr[:, valid_w].T
                triangulated = int(pts_world.shape[0])
                total_raw += triangulated

                # stage 2: reprojection error
                reproj_ok = _reprojection_filter(
                    pts_world, proj_i, proj_j, kps_i_valid, kps_j_valid, self._max_reproj_px
                )
                total_reproj_drop += int((~reproj_ok).sum())
                pts_world = pts_world[reproj_ok]
                if pts_world.shape[0] > 0:
                    reproj_errors.extend(_compute_reproj_errors(
                        pts_world, proj_i, proj_j,
                        kps_i_valid[reproj_ok], kps_j_valid[reproj_ok]
                    ))

                # stage 3: z-tolerance
                z_ok = np.abs(pts_world[:, 2] - z0) <= self._z_tolerance_m
                total_ztol_drop += int((~z_ok).sum())
                pts_world = pts_world[z_ok]

                # stage 4: distance cap
                dists = np.linalg.norm(pts_world - t_i, axis=1)
                dist_ok = dists <= self._max_distance_m
                total_dist_drop += int((~dist_ok).sum())
                pts_world = pts_world[dist_ok]

                kept = int(pts_world.shape[0])
                total_kept += kept

                if kept > 0:
                    points_per_frame[i].append(pts_world)
                    baselines.append(baseline_m)

                pair_stats.append({
                    "i": i, "j": j, "baseline_m": baseline_m,
                    "matches": n_matched, "triangulated": triangulated, "kept": kept,
                })

        # 통계 로그
        if baselines:
            logger.info(
                "triangulation: pairs=%d kept=%d baseline_m p50=%.3f p95=%.3f",
                len(pair_stats),
                total_kept,
                float(np.median(baselines)),
                float(np.percentile(baselines, 95)),
            )
        if reproj_errors:
            logger.info(
                "triangulation: reprojection_error_px p50=%.2f p95=%.2f",
                float(np.median(reproj_errors)),
                float(np.percentile(reproj_errors, 95)),
            )
        logger.info(
            "triangulation: total_raw=%d total_kept=%d floor_gate_pairs=%d "
            "drops[dlt=%d reproj=%d ztol=%d dist=%d]",
            total_raw,
            total_kept,
            len(pair_stats),
            total_dlt_drop,
            total_reproj_drop,
            total_ztol_drop,
            total_dist_drop,
        )

        # per-frame 결합
        merged: list[np.ndarray] = []
        for _i, pts_list in enumerate(points_per_frame):
            if pts_list:
                merged.append(np.concatenate(pts_list, axis=0))
            else:
                merged.append(np.zeros((0, 3), dtype=np.float64))

        return TriangulatedCloud(
            points_per_frame=merged,
            pair_stats=pair_stats,
            total_raw=total_raw,
            total_kept=total_kept,
        )


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_projection_matrix(pose: np.ndarray, intrin: Intrinsics) -> np.ndarray:
    """ARKit world_T_camera pose → OpenCV convention projection matrix p (3,4).

    p = k @ [r_cfw | t_cfw]  where:
        r_cfw = (r_arkit @ D).T
        t_cfw = -r_cfw @ t_arkit
    """
    r_arkit = pose[:3, :3]
    t_arkit = pose[:3, 3]

    r_ocv_wfc = r_arkit @ _D          # world-from-camera in OpenCV convention
    r_cfw = r_ocv_wfc.T               # camera-from-world rotation
    t_cfw = -r_cfw @ t_arkit          # camera-from-world translation

    k = np.array([
        [intrin.fx, 0.0,       intrin.cx],
        [0.0,       intrin.fy, intrin.cy],
        [0.0,       0.0,       1.0      ],
    ], dtype=np.float64)

    rt = np.concatenate([r_cfw, t_cfw[:, np.newaxis]], axis=1)  # (3, 4)
    result: np.ndarray = np.asarray(k @ rt, dtype=np.float64)
    return result


def _project_points(pts_world: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """(N, 3) world → (N, 2) image pixel coords via projection matrix proj (3,4)."""
    n = pts_world.shape[0]
    pts_h = np.concatenate([pts_world, np.ones((n, 1), dtype=np.float64)], axis=1)  # (N, 4)
    uvw: np.ndarray = (proj @ pts_h.T).T  # (N, 3)
    w = uvw[:, 2:3]
    valid = np.abs(w[:, 0]) > 1e-8
    px = np.full((n, 2), np.nan, dtype=np.float64)
    px[valid] = uvw[valid, :2] / w[valid]
    return px


def _reprojection_filter(
    pts_world: np.ndarray,
    proj_i: np.ndarray,
    proj_j: np.ndarray,
    kps_i: np.ndarray,
    kps_j: np.ndarray,
    max_error_px: float,
) -> np.ndarray:
    """reprojection error 필터. True mask (N,) 반환."""
    reprojected_i = _project_points(pts_world, proj_i)
    reprojected_j = _project_points(pts_world, proj_j)

    err_i = np.linalg.norm(reprojected_i - kps_i, axis=1)
    err_j = np.linalg.norm(reprojected_j - kps_j, axis=1)
    mean_err = (err_i + err_j) / 2.0

    ok: np.ndarray = (mean_err <= max_error_px) & np.isfinite(mean_err)
    return ok


def _compute_reproj_errors(
    pts_world: np.ndarray,
    proj_i: np.ndarray,
    proj_j: np.ndarray,
    kps_i: np.ndarray,
    kps_j: np.ndarray,
) -> list[float]:
    """kept 점들의 reprojection error 목록 반환 (통계 로그용)."""
    reprojected_i = _project_points(pts_world, proj_i)
    reprojected_j = _project_points(pts_world, proj_j)
    err_i = np.linalg.norm(reprojected_i - kps_i, axis=1)
    err_j = np.linalg.norm(reprojected_j - kps_j, axis=1)
    mean_err = (err_i + err_j) / 2.0
    return [float(e) for e in mean_err if np.isfinite(e)]
