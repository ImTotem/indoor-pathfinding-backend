"""Single frame Segformer polygon + SuperPoint multi-view plane fit layout.

Sprint 21 exploration / diagnostic: layout 단일 frame 기반 정확 back-project.

파이프라인:
    1. 목표 keyframe의 Segformer floor mask + polygon (upright rotation으로 정확)
    2. 목표 ± window 인접 frame SuperPoint+LightGlue 매칭
    3. 목표 frame floor polygon 내부 kp pair만 필터 → cv2.triangulatePoints → 3D 점
    4. 3D 점에 RANSAC plane fit (ax+by+cz+d=0)
    5. 목표 frame polygon vertex를 그 평면과 ray-plane intersection → world 좌표
    6. 결과 world polygon 하나로 walkable_grid → skeleton → composite

z=z0 floor 평면 가정을 버리고 **실측 3D 점으로 fitting한 plane**을 사용하는 게 핵심.
Sprint 20 triangulation의 sparse 문제 우회 — polygon 자체는 dense, plane은 noise 수렴.

사용:
    uv run python scripts/single_frame_plane_layout.py <scan_dir> <seq> [--window 5]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.application.building.steps.back_projection import (
    Intrinsics,
    default_intrinsics,
)
from indoor_server.application.building.steps.node_placement import NodePlacementStep
from indoor_server.application.building.steps.skeletonize import SkeletonizeStep
from indoor_server.application.building.steps.triangulation import (
    _build_projection_matrix,
)
from indoor_server.config import settings
from indoor_server.domain.building.models import WalkableGrid
from indoor_server.infrastructure.ml.model_cache import ModelCache
from indoor_server.infrastructure.ml.segformer_onnx import SegformerOnnxSegmenter
from indoor_server.infrastructure.ml.superpoint_lightglue import SuperPointLightGlueRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("single_frame")

# ARKit Y-up → Z-up 변환 (run_real_scan.py와 동일)
_AXIS_SWAP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
_AXIS_SWAP_4 = np.eye(4, dtype=np.float64)
_AXIS_SWAP_4[:3, :3] = _AXIS_SWAP

# ARKit ray convention flip (triangulation.py와 동일 철학)
_D = np.diag([1.0, -1.0, -1.0])


@dataclass
class KeyframeRow:
    seq: int
    pose: np.ndarray  # 4x4, z-up converted
    image_path: Path


def _convert_pose_yup_to_zup(pose_bytes: bytes) -> np.ndarray:
    values = struct.unpack_from("<16f", pose_bytes)
    pose_old = np.array(values, dtype=np.float64).reshape(4, 4, order="F")
    return _AXIS_SWAP_4 @ pose_old


def _load_keyframes(scan_dir: Path) -> tuple[UUID, list[KeyframeRow]]:
    sidecar = scan_dir / "scan_metadata.db"
    conn = sqlite3.connect(str(sidecar))
    try:
        scan_id = UUID(conn.execute("SELECT id FROM scan_session LIMIT 1").fetchone()[0])
        rows = conn.execute(
            "SELECT seq, image_path, pose_matrix FROM keyframe_meta ORDER BY seq"
        ).fetchall()
    finally:
        conn.close()

    keyframes_dir = scan_dir / "keyframes"
    out: list[KeyframeRow] = []
    for seq, img_path, pose_blob in rows:
        img = keyframes_dir / Path(img_path).name
        if not img.exists():
            continue
        out.append(KeyframeRow(seq=seq, pose=_convert_pose_yup_to_zup(bytes(pose_blob)), image_path=img))
    return scan_id, out


async def _segmenter_real_upright(image: np.ndarray, inner: SegformerOnnxSegmenter) -> np.ndarray:
    """run_real_scan.py의 UprightRotatedSegmenter와 동일. 최종 mask(bool)만 반환."""
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    out = await inner.segment(rotated)
    mask_upright = out.class_mask.copy()
    mask_back = cv2.rotate(mask_upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # ADE20K class 3 = floor, 29 = rug → both walkable
    return (np.isin(mask_back, [3, 29])).astype(bool)


def _extract_polygon(mask: np.ndarray, epsilon_frac: float = 0.01) -> np.ndarray | None:
    """floor mask → cv2.findContours 가장 큰 윤곽 → Douglas-Peucker 단순화."""
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    eps = epsilon_frac * cv2.arcLength(largest, closed=True)
    approx = cv2.approxPolyDP(largest, eps, closed=True)  # (K, 1, 2)
    return approx.reshape(-1, 2).astype(np.float64)  # (K, 2) pixel coords (x, y)


async def _collect_triangulated_floor_points(
    target: KeyframeRow,
    target_mask: np.ndarray,
    adjacent: list[KeyframeRow],
    sp_lg: SuperPointLightGlueRunner,
    intrin: Intrinsics,
    min_match_score: float = 0.5,
    max_per_pair: int = 128,
    max_reproj_px: float = 15.0,
) -> np.ndarray:
    """target과 각 인접 frame에 대해 triangulate. floor mask 내부 kp만 사용."""
    target_img = cv2.imread(str(target.image_path))
    if target_img is None:
        raise RuntimeError(f"cannot read target image {target.image_path}")
    h, w = target_img.shape[:2]
    proj_t = _build_projection_matrix(target.pose, intrin)

    # floor mask dilate 로 polygon 경계 근처 kp도 허용
    dilated = cv2.dilate(target_mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)

    all_points: list[np.ndarray] = []
    stats: list[dict] = []

    for adj in adjacent:
        if adj.seq == target.seq:
            continue
        adj_img = cv2.imread(str(adj.image_path))
        if adj_img is None:
            continue
        proj_a = _build_projection_matrix(adj.pose, intrin)
        mp = await sp_lg.match(target_img, adj_img)
        if mp.matches.shape[0] == 0:
            continue

        score_ok = mp.scores >= min_match_score
        sel = mp.matches[score_ok]
        scores_sel = mp.scores[score_ok]
        if len(sel) == 0:
            continue
        if len(sel) > max_per_pair:
            order = np.argsort(-scores_sel)[:max_per_pair]
            sel = sel[order]

        sx_t, sy_t = w / mp.size_a[1], h / mp.size_a[0]
        sx_a, sy_a = w / mp.size_b[1], h / mp.size_b[0]

        kp_t_list, kp_a_list = [], []
        for idx_t, idx_a in sel:
            xt = float(mp.kpts_a[idx_t, 0]) * sx_t
            yt = float(mp.kpts_a[idx_t, 1]) * sy_t
            xa = float(mp.kpts_b[idx_a, 0]) * sx_a
            ya = float(mp.kpts_b[idx_a, 1]) * sy_a
            ui, vi = int(round(xt)), int(round(yt))
            if not (0 <= vi < h and 0 <= ui < w and dilated[vi, ui]):
                continue
            kp_t_list.append((xt, yt))
            kp_a_list.append((xa, ya))
        if not kp_t_list:
            continue

        kp_t_arr = np.array(kp_t_list, dtype=np.float32).T
        kp_a_arr = np.array(kp_a_list, dtype=np.float32).T
        pts_h = cv2.triangulatePoints(proj_t.astype(np.float32), proj_a.astype(np.float32), kp_t_arr, kp_a_arr)
        w_vals = pts_h[3, :]
        valid = np.abs(w_vals) > 1e-6
        if valid.sum() == 0:
            continue
        pts3d = (pts_h[:3, valid] / w_vals[valid][None, :]).T.astype(np.float64)

        # reprojection filter
        pts_h2 = np.concatenate([pts3d, np.ones((len(pts3d), 1))], axis=1)
        proj_t_uvw = (proj_t @ pts_h2.T).T
        proj_t_uv = proj_t_uvw[:, :2] / proj_t_uvw[:, 2:3]
        proj_a_uvw = (proj_a @ pts_h2.T).T
        proj_a_uv = proj_a_uvw[:, :2] / proj_a_uvw[:, 2:3]
        err_t = np.linalg.norm(proj_t_uv - kp_t_arr.T[valid], axis=1)
        err_a = np.linalg.norm(proj_a_uv - kp_a_arr.T[valid], axis=1)
        reproj_ok = (err_t < max_reproj_px) & (err_a < max_reproj_px)
        pts3d = pts3d[reproj_ok]
        all_points.append(pts3d)
        stats.append({"adj_seq": adj.seq, "matched": len(sel), "in_mask": len(kp_t_list),
                      "triangulated": int(valid.sum()), "kept": int(reproj_ok.sum())})

    for s in stats:
        logger.info(
            "  pair target=%d ↔ adj=%d matched=%d in_mask=%d tri=%d kept=%d",
            target.seq, s["adj_seq"], s["matched"], s["in_mask"], s["triangulated"], s["kept"],
        )
    if not all_points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(all_points, axis=0)


def _ransac_plane_fit(points: np.ndarray, iterations: int = 500, threshold_m: float = 0.10) -> tuple[np.ndarray, float, int] | None:
    """numpy-only RANSAC. plane 방정식 n·X + d = 0, ||n||=1.

    반환: (normal (3,), d_scalar, inlier_count). 실패 시 None.
    """
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed=42)
    best_inliers = 0
    best = None
    n = len(points)
    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        sample = points[idx]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-8:
            continue
        normal /= norm_mag
        d = -float(normal @ sample[0])
        dists = np.abs(points @ normal + d)
        inliers = int((dists < threshold_m).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best = (normal, d)
    if best is None:
        return None

    # inlier 집합으로 final least-squares fit
    normal, d = best
    dists = np.abs(points @ normal + d)
    inlier_mask = dists < threshold_m
    inlier_pts = points[inlier_mask]
    # centroid + SVD refinement
    centroid = inlier_pts.mean(axis=0)
    u, s, vh = np.linalg.svd(inlier_pts - centroid)
    normal_ref = vh[-1]
    if normal_ref @ normal < 0:
        normal_ref = -normal_ref
    d_ref = -float(normal_ref @ centroid)
    return normal_ref, d_ref, int(inlier_mask.sum())


def _horizontal_plane_fit(points: np.ndarray, z_tol_m: float = 0.30) -> tuple[np.ndarray, float, int]:
    """floor 가정(normal=[0,0,1]) 하에 z median 기반 plane.

    triangulation noise로 인한 outlier는 |z - median| > z_tol_m 로 날리고
    inlier들의 z median을 floor height로 채택.
    """
    z_vals = points[:, 2]
    z_med_init = float(np.median(z_vals))
    inlier_mask = np.abs(z_vals - z_med_init) < z_tol_m
    if inlier_mask.sum() < 3:
        z_final = z_med_init
        inliers = int(inlier_mask.sum())
    else:
        z_final = float(np.median(z_vals[inlier_mask]))
        inliers = int(inlier_mask.sum())
    normal = np.array([0.0, 0.0, 1.0])
    d = -z_final
    return normal, d, inliers


def _ray_plane_intersect(
    pixel: tuple[float, float],
    pose: np.ndarray,
    intrin: Intrinsics,
    plane_normal: np.ndarray,
    plane_d: float,
) -> np.ndarray | None:
    """pixel (u, v) → camera ARKit ray → world 공간에서 plane 교차.

    ray_cam_arkit = [(u-cx)/fx, -(v-cy)/fy, -1]
    ray_world = R_pose @ ray_cam_arkit
    origin = pose[:3, 3]
    t = -(n·origin + d) / (n·ray_world)
    intersection = origin + t * ray_world
    """
    u, v = pixel
    ray_cam = np.array([
        (u - intrin.cx) / intrin.fx,
        -(v - intrin.cy) / intrin.fy,
        -1.0,
    ], dtype=np.float64)
    ray_world = pose[:3, :3] @ ray_cam
    origin = pose[:3, 3]
    denom = float(plane_normal @ ray_world)
    if abs(denom) < 1e-8:
        return None
    t = -(float(plane_normal @ origin) + plane_d) / denom
    if t <= 0:
        return None  # 뒤쪽 교차
    return origin + t * ray_world


def _rasterize_polygon_to_grid(
    world_polygon: np.ndarray,  # (K, 3)
    cell_size_m: float,
    z0: float,
) -> WalkableGrid:
    """world polygon → walkable_grid cell raster. origin 자동 계산 (bbox + 1m 패딩)."""
    xy = world_polygon[:, :2]
    x_min, y_min = xy.min(axis=0) - 1.0
    x_max, y_max = xy.max(axis=0) + 1.0
    w = int(np.ceil((x_max - x_min) / cell_size_m))
    h = int(np.ceil((y_max - y_min) / cell_size_m))

    # polygon pixel 좌표
    pixel_xy = ((xy - np.array([x_min, y_min])) / cell_size_m).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pixel_xy], 1)
    mask_bool = mask.astype(bool)
    obs_count = mask.astype(np.uint16)

    from indoor_server.domain.building.models import GridOrigin
    origin = GridOrigin(x0=x_min, y0=y_min, z0=z0, cell_size=cell_size_m, w=w, h=h)
    return WalkableGrid(origin=origin, mask=mask_bool, observation_count=obs_count)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir")
    parser.add_argument("seq", type=int, help="target keyframe sequence number")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--plane-threshold", type=float, default=0.30, help="RANSAC inlier threshold (m)")
    parser.add_argument("--cell-size", type=float, default=0.10)
    parser.add_argument("--reproj-px", type=float, default=30.0, help="triangulation reprojection max (px)")
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.0,
        help="손-바닥 거리 가정 (m). z0_prior = target_tz - camera_height",
    )
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).resolve()
    scan_id, keyframes = _load_keyframes(scan_dir)
    logger.info("loaded %d keyframes", len(keyframes))

    # target + window range
    target_idx = next((i for i, k in enumerate(keyframes) if k.seq == args.seq), None)
    if target_idx is None:
        logger.error("target seq %d not found", args.seq)
        return 1
    target = keyframes[target_idx]
    lo = max(0, target_idx - args.window)
    hi = min(len(keyframes), target_idx + args.window + 1)
    adjacent = keyframes[lo:hi]
    logger.info("target=%d  adjacent=[%d..%d]", target.seq, adjacent[0].seq, adjacent[-1].seq)

    # Segformer
    cache = ModelCache(
        cache_dir=settings.model_cache_dir,
        repo_id=settings.segformer_model_repo_id,
        filename=settings.segformer_model_filename,
    )
    segformer = SegformerOnnxSegmenter(model_path=cache.ensure())

    target_img = cv2.imread(str(target.image_path))
    if target_img is None:
        logger.error("cannot read target image: %s", target.image_path)
        return 1
    h, w = target_img.shape[:2]
    intrin = default_intrinsics(w, h)

    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    target_mask = await _segmenter_real_upright(target_rgb, segformer)
    logger.info("target floor mask pixels=%d / %d (%.1f%%)",
                int(target_mask.sum()), target_mask.size, 100 * target_mask.sum() / target_mask.size)

    polygon_2d = _extract_polygon(target_mask, epsilon_frac=0.01)
    if polygon_2d is None:
        logger.error("polygon extract failed")
        return 1
    logger.info("polygon vertices=%d", len(polygon_2d))

    # SuperPoint+LightGlue
    sp_lg_path = settings.model_cache_dir / "superpoint_lightglue.onnx"
    sp_lg = SuperPointLightGlueRunner(model_path=sp_lg_path, input_size=settings.superpoint_input_size)

    # Triangulate floor points
    floor_points_3d = await _collect_triangulated_floor_points(
        target, target_mask, adjacent, sp_lg, intrin,
        max_reproj_px=args.reproj_px,
    )
    logger.info("triangulated 3D floor points: %d", len(floor_points_3d))
    if len(floor_points_3d) >= 1:
        zs = floor_points_3d[:, 2]
        logger.info("  z stats: min=%.3f  p25=%.3f  median=%.3f  p75=%.3f  max=%.3f",
                    float(zs.min()), float(np.percentile(zs, 25)), float(np.median(zs)),
                    float(np.percentile(zs, 75)), float(zs.max()))
        logger.info("  target camera tz=%.3f  (floor는 대략 tz - 1m)", float(target.pose[2, 3]))
    if len(floor_points_3d) < 10:
        logger.error("insufficient triangulated points for plane fit")
        return 1

    # Plane 결정: SuperPoint+LightGlue triangulation은 floor mask 내부에서도
    # wall/ceiling feature를 잡을 수 있어 **floor 높이 prior**를 강제하는 게 안전.
    #
    # prior: z0 = target_tz - camera_hand_height (기본 1m). run_real_scan.py 방식 재현.
    # prior 기준 ±0.5m 이내 triangulated 점만 inlier로 간주하고 그 median으로 refine.
    target_tz = float(target.pose[2, 3])
    z0_prior = target_tz - args.camera_height
    z_tol_refine = 0.50
    z_near = np.abs(floor_points_3d[:, 2] - z0_prior) < z_tol_refine
    if z_near.sum() >= 3:
        z0_fit = float(np.median(floor_points_3d[z_near, 2]))
        inliers = int(z_near.sum())
    else:
        z0_fit = z0_prior
        inliers = 0
    normal = np.array([0.0, 0.0, 1.0])
    d_plane = -z0_fit
    logger.info(
        "plane (horizontal prior): z0_prior=%.3f refined_z0=%.3f inliers=%d/%d",
        z0_prior, z0_fit, inliers, len(floor_points_3d),
    )

    # 참고: general RANSAC plane도 출력 (경사 floor 데이터용 나중)
    _gen = _ransac_plane_fit(floor_points_3d, iterations=500, threshold_m=args.plane_threshold)
    if _gen is not None:
        n_gen, d_gen, inl_gen = _gen
        logger.info(
            "  (ref general RANSAC: n=[%+.3f,%+.3f,%+.3f] d=%+.3f inl=%d cos_up=%.2f)",
            n_gen[0], n_gen[1], n_gen[2], d_gen, inl_gen, abs(float(n_gen[2])),
        )

    # Ray-plane intersect each polygon vertex
    world_vertices: list[np.ndarray] = []
    for u, v in polygon_2d:
        pt = _ray_plane_intersect((float(u), float(v)), target.pose, intrin, normal, d_plane)
        if pt is not None:
            world_vertices.append(pt)
    if len(world_vertices) < 3:
        logger.error("world polygon has < 3 valid vertices (%d)", len(world_vertices))
        return 1
    world_poly = np.array(world_vertices, dtype=np.float64)
    logger.info("world polygon vertices=%d  bbox=(%.2f,%.2f)-(%.2f,%.2f)  z range=(%.3f,%.3f)",
                len(world_poly),
                world_poly[:, 0].min(), world_poly[:, 1].min(),
                world_poly[:, 0].max(), world_poly[:, 1].max(),
                world_poly[:, 2].min(), world_poly[:, 2].max())

    # Rasterize to grid
    grid = _rasterize_polygon_to_grid(world_poly, args.cell_size, z0=z0_fit)
    logger.info("grid: %dx%d cells  walkable=%d", grid.origin.w, grid.origin.h, int(grid.mask.sum()))

    # Skeleton + graph (재활용)
    skel_step = SkeletonizeStep()
    from skimage.morphology import medial_axis
    sk_result = medial_axis(grid.mask, return_distance=True)
    skel_mask = sk_result[0]
    skel_graph = None
    if int(skel_mask.sum()) > 0:
        g = skel_step._build_graph(skel_mask)  # type: ignore[attr-defined]
        nodes, edges = skel_step._extract_topology(g)  # type: ignore[attr-defined]
        from indoor_server.application.building.steps.skeletonize import SkeletonGraph
        skel_graph = SkeletonGraph(nodes=nodes, edges=edges, skeleton_pixel_count=int(skel_mask.sum()))

    placer = NodePlacementStep(scan_id=scan_id, build_job_id=uuid4())
    if skel_graph is not None and skel_graph.nodes:
        nodes_out, edges_out = placer.run(skel_graph, grid.origin)
    else:
        nodes_out, edges_out = [], []
    logger.info("skeleton pixels=%d  nodes=%d  edges=%d",
                int(skel_mask.sum()), len(nodes_out), len(edges_out))

    # Debug dump (간단)
    repo_root = Path(__file__).resolve().parents[2]
    dump = repo_root / "var" / "debug" / str(scan_id) / f"single_frame_{target.seq:06d}"
    dump.mkdir(parents=True, exist_ok=True)
    np.savez(dump / "walkable_grid.npz",
             mask=grid.mask, observation_count=grid.observation_count,
             origin_x0=grid.origin.x0, origin_y0=grid.origin.y0, origin_z0=grid.origin.z0,
             origin_cell_size=grid.origin.cell_size, origin_w=grid.origin.w, origin_h=grid.origin.h)
    # PNG
    vis = np.zeros((grid.origin.h, grid.origin.w, 3), dtype=np.uint8)
    vis[grid.mask] = (180, 180, 180)
    # skeleton overlay
    if int(skel_mask.sum()) > 0:
        vis[skel_mask] = (0, 255, 255)
    cv2.imwrite(str(dump / "layout.png"), vis)
    # target image + polygon overlay
    overlay = target_img.copy()
    poly_int = polygon_2d.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [poly_int], True, (0, 0, 255), 3)
    cv2.imwrite(str(dump / "target_polygon.png"), overlay)
    logger.info("dump written to %s", dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
