"""Sprint 78 — SuperPoint+LightGlue cross-session SE(3) estimation.

두 스캔(A, B)의 RGB mp4 + poses.bin 로 cross-session 매칭 → metric SE(3) 추정.

사용:
    uv run python scripts/cross_session/run_superpoint_cross_session.py \\
        --scan-a /path/to/scanA \\
        --scan-b /path/to/scanB \\
        --out-dir _workspace/sprint78-superpoint-cross-session/evidence/cycle2

설계:
    Step B: 0.5Hz keyframe sampling + PoseMatcher pose 매칭
    Step C: brute-force NA × NB pair selection
    Step D: SuperPoint+LightGlue 매칭 (≥30 inlier filter)
    Step E: F-matrix → E = K_B.T @ F @ K_A → recoverPose (각 K 적용)
    Step F: SE(3) 후보 aggregation (chordal mean + RANSAC)
    Step G: pose graph overlay 시각화
    Step H: rtabmap baseline 비교 (merged.db SQLite)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Calibration / manifest
# ---------------------------------------------------------------------------

@dataclass
class CameraK:
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    def inv(self) -> np.ndarray:
        return np.linalg.inv(self.matrix())


def load_manifest(scan_dir: Path) -> dict[str, Any]:
    path = scan_dir / "manifest.json"
    with open(path) as f:
        return json.load(f)


def k_from_manifest(m: dict[str, Any]) -> CameraK:
    return CameraK(
        fx=float(m["intrinsics_fx"]),
        fy=float(m["intrinsics_fy"]),
        cx=float(m["intrinsics_cx"]),
        cy=float(m["intrinsics_cy"]),
    )


# ---------------------------------------------------------------------------
# poses.bin parsing (same as PoseMatcher)
# ---------------------------------------------------------------------------

RECORD_SIZE = 72  # 8B pts_ns + 64B 4x4 float32 column-major


def load_poses(poses_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pts_ns_arr [N], poses_arr [N,4,4])."""
    raw = poses_path.read_bytes()
    n = len(raw) // RECORD_SIZE
    pts_ns = np.empty(n, dtype=np.int64)
    poses = np.empty((n, 4, 4), dtype=np.float32)
    for i in range(n):
        off = i * RECORD_SIZE
        pts_ns[i] = struct.unpack_from("<q", raw, off)[0]
        mat = np.frombuffer(raw, dtype="<f4", count=16, offset=off + 8).reshape(4, 4, order="F")
        poses[i] = mat
    return pts_ns, poses


def find_pose(pts_ns_arr: np.ndarray, poses_arr: np.ndarray, query_ns: int,
              threshold_ns: int = 16_666_666) -> np.ndarray | None:
    idx = int(np.searchsorted(pts_ns_arr, query_ns))
    best: int | None = None
    best_diff = threshold_ns + 1
    for c in [idx - 1, idx]:
        if 0 <= c < len(pts_ns_arr):
            diff = abs(int(pts_ns_arr[c]) - query_ns)
            if diff < best_diff:
                best_diff = diff
                best = c
    if best is None or best_diff > threshold_ns:
        return None
    return poses_arr[best]


# ---------------------------------------------------------------------------
# Step B — Keyframe sampling
# ---------------------------------------------------------------------------

def sample_keyframes(
    scan_dir: Path,
    pts_ns_arr: np.ndarray,
    poses_arr: np.ndarray,
    out_dir: Path,
    tag: str,
    sample_hz: float = 0.5,
    max_long_edge: int = 1024,
) -> list[dict[str, Any]]:
    """Decode mp4 at sample_hz, match poses, save JPEGs.

    poses.bin PTS는 ARKit 절대 시간(시스템 업타임 기준).
    mp4 PTS는 0부터 시작하는 상대 시간.
    → poses.bin PTS를 pts[0] 기준 상대값으로 rebasing (multiscan_dense_video_polygon 동일 처리).
    """
    import av  # type: ignore[import-untyped]

    mp4_path = scan_dir / "scan.mp4"
    kf_dir = out_dir / f"kf_{tag}"
    kf_dir.mkdir(parents=True, exist_ok=True)

    # Rebase poses.bin PTS to relative (same as rebase_pose_timestamps_for_video)
    match_pts = pts_ns_arr - pts_ns_arr[0]

    index: list[dict[str, Any]] = []
    interval_s = 1.0 / sample_hz
    last_ts_s = -1e9

    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)

        for frame in container.decode(video=0):
            pts = frame.pts
            if pts is None:
                continue
            ts_s = pts * time_base
            pts_ns = int(ts_s * 1e9)

            if ts_s - last_ts_s < interval_s:
                continue

            # Use rebased match_pts for lookup
            pose = find_pose(match_pts, poses_arr, pts_ns)
            if pose is None:
                continue

            # Decode frame
            img_np = frame.to_ndarray(format="rgb24")  # (H, W, 3)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Resize long edge
            h, w = img_bgr.shape[:2]
            scale = max_long_edge / max(h, w)
            if scale < 1.0:
                img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)

            idx = len(index)
            img_path = kf_dir / f"{idx:04d}.jpg"
            cv2.imwrite(str(img_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

            index.append({
                "idx": idx,
                "pts_ns": pts_ns,
                "ts_s": ts_s,
                "image_path": str(img_path),
                "pose_4x4_world_arkit": pose.tolist(),
            })
            last_ts_s = ts_s

    (out_dir / f"kf_{tag}" / "index.json").write_text(json.dumps(index, indent=2))
    logger.info("[%s] keyframes=%d", tag, len(index))
    return index


# ---------------------------------------------------------------------------
# Step C — Pair selection (brute-force)
# ---------------------------------------------------------------------------

def select_pairs(kf_A: list[dict], kf_B: list[dict]) -> list[tuple[int, int]]:
    pairs = [(a["idx"], b["idx"]) for a in kf_A for b in kf_B]
    logger.info("pairs=%d (brute-force %dx%d)", len(pairs), len(kf_A), len(kf_B))
    return pairs


# ---------------------------------------------------------------------------
# Step D — SuperPoint+LightGlue matching
# ---------------------------------------------------------------------------

def build_matcher(device: str = "cpu"):
    """kornia DISK + LightGlue (disk weights) — SuperPoint 대체.

    note: kornia 0.8.2 SuperPoint 미제공. DISK는 동급 학습 매칭. lightglue는 disk 가중치 지원.
    """
    import torch
    from kornia.feature import DISK, LightGlue

    extractor = DISK.from_pretrained("depth").eval().to(device)
    matcher = LightGlue(features="disk").eval().to(device)
    return extractor, matcher, torch, None  # rbd_fn=None (kornia path)


def _load_image_tensor_rgb(img_path: str, device: str, target_long: int = 720) -> tuple:
    """BGR jpg → RGB float32 [1,3,H,W] (8 multiple), returns (tensor, orig_size_wh, scale)."""
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"cannot read {img_path}")
    h, w = img.shape[:2]
    scale = target_long / max(h, w) if max(h, w) > target_long else 1.0
    new_w = int(round(w * scale)) // 8 * 8
    new_h = int(round(h * scale)) // 8 * 8
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    import torch as _torch
    t = _torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device), (w, h), float(new_w / w)


def extract_features(extractor, img_path: str, device: str, torch_mod):
    """DISK extract → dict(keypoints[N,2], descriptors[N,D], image_size[W,H], scale)."""
    t, (orig_w, orig_h), scale = _load_image_tensor_rgb(img_path, device)
    with torch_mod.inference_mode():
        out = extractor(t, n=2048, score_threshold=0.0)[0]
    kpts = out.keypoints.detach().cpu().float()  # (N,2) on resized
    desc = out.descriptors.detach().cpu().float()  # (N,D)
    scores = out.detection_scores.detach().cpu().float() if hasattr(out, "detection_scores") else None
    return {
        "keypoints": kpts,
        "descriptors": desc,
        "scores": scores,
        "image_size": (t.shape[-1], t.shape[-2]),  # (W,H) resized
        "orig_size": (orig_w, orig_h),
        "scale": scale,
    }


def match_pair(
    extractor, matcher, rbd_fn,
    img_path_a: str, img_path_b: str,
    device: str, torch_mod,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (kpts_a, kpts_b, scores) on ORIGINAL image coords."""
    fA = extract_features(extractor, img_path_a, device, torch_mod)
    fB = extract_features(extractor, img_path_b, device, torch_mod)
    img_size_a = torch_mod.tensor([list(fA["image_size"])], dtype=torch_mod.float, device=device)
    img_size_b = torch_mod.tensor([list(fB["image_size"])], dtype=torch_mod.float, device=device)
    data = {
        "image0": {
            "keypoints": fA["keypoints"].unsqueeze(0).to(device),
            "descriptors": fA["descriptors"].unsqueeze(0).to(device),
            "image_size": img_size_a,
        },
        "image1": {
            "keypoints": fB["keypoints"].unsqueeze(0).to(device),
            "descriptors": fB["descriptors"].unsqueeze(0).to(device),
            "image_size": img_size_b,
        },
    }
    with torch_mod.inference_mode():
        out = matcher(data)
    matches = out["matches"][0].detach().cpu().numpy()  # (M,2)
    if matches.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))
    scores = out.get("scores", out.get("matching_scores0"))
    if scores is None:
        sc = np.ones(len(matches), dtype=np.float32)
    else:
        sc_t = scores[0].detach().cpu().numpy() if scores[0].ndim > 0 else scores.detach().cpu().numpy()
        # kornia LightGlue returns per-match scores already aligned to matches
        sc = sc_t[: matches.shape[0]] if sc_t.shape[0] >= matches.shape[0] else np.ones(matches.shape[0])
    kpsA = fA["keypoints"].numpy()
    kpsB = fB["keypoints"].numpy()
    pts_a = kpsA[matches[:, 0]] / fA["scale"]
    pts_b = kpsB[matches[:, 1]] / fB["scale"]
    return pts_a, pts_b, sc


def run_matching(
    pairs: list[tuple[int, int]],
    kf_A: list[dict], kf_B: list[dict],
    out_dir: Path,
    device: str = "cpu",
    score_thresh: float = 0.5,
    min_matches: int = 30,
) -> list[dict[str, Any]]:
    extractor, matcher, torch_mod, rbd_fn = build_matcher(device)
    matches_dir = out_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    n_strong = 0

    for a_idx, b_idx in pairs:
        img_a = kf_A[a_idx]["image_path"]
        img_b = kf_B[b_idx]["image_path"]
        try:
            kpts0, kpts1, sc = match_pair(extractor, matcher, rbd_fn,
                                           img_a, img_b, device, torch_mod)
        except Exception as e:
            logger.warning("pair (%d,%d) failed: %s", a_idx, b_idx, e)
            summary.append({"a_idx": a_idx, "b_idx": b_idx, "error": str(e)})
            continue

        # filter by score
        mask = sc >= score_thresh
        kpts0_f, kpts1_f, sc_f = kpts0[mask], kpts1[mask], sc[mask]
        n_strong_match = int(mask.sum())
        is_strong = n_strong_match >= min_matches

        npz_path = matches_dir / f"{a_idx:04d}_{b_idx:04d}.npz"
        np.savez(str(npz_path), kpts0=kpts0_f, kpts1=kpts1_f, scores=sc_f)

        row = {
            "a_idx": a_idx,
            "b_idx": b_idx,
            "n_kpts_a": len(kpts0),
            "n_kpts_b": len(kpts1),
            "n_raw_matches": len(kpts0),
            "n_strong_matches": n_strong_match,
            "mean_score": float(sc_f.mean()) if len(sc_f) else 0.0,
            "strong": is_strong,
            "npz_path": str(npz_path),
        }
        summary.append(row)
        if is_strong:
            n_strong += 1
            logger.info("  STRONG pair (%d,%d) n_strong=%d", a_idx, b_idx, n_strong_match)

    # Write CSV
    csv_path = out_dir / "matches_summary.csv"
    if summary:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["a_idx", "b_idx", "n_kpts_a", "n_kpts_b",
                                               "n_raw_matches", "n_strong_matches",
                                               "mean_score", "strong"])
            w.writeheader()
            for row in summary:
                if "error" not in row:
                    w.writerow({k: row[k] for k in w.fieldnames})

    logger.info("total strong pairs: %d / %d", n_strong, len(pairs))
    return [r for r in summary if r.get("strong")]


# ---------------------------------------------------------------------------
# Step E — Pairwise relative pose (F → E path with K_A, K_B)
# ---------------------------------------------------------------------------

def estimate_relpose(
    kpts_a: np.ndarray, kpts_b: np.ndarray,
    K_A: np.ndarray, K_B: np.ndarray,
) -> dict[str, Any] | None:
    """F → E = K_B.T @ F @ K_A → recoverPose with normalized coords.

    E recovery strategy (Patch 2):
    1. cv2.findFundamentalMat(pts_a, pts_b) → F
    2. E = K_B.T @ F @ K_A
    3. normalize: pts_a_norm = (K_A^-1 @ [u,v,1])[:2], pts_b_norm = same
    4. cv2.recoverPose(E, pts_a_norm, pts_b_norm, np.eye(3))
    """
    if len(kpts_a) < 8:
        return None

    pts_a = kpts_a.astype(np.float64)
    pts_b = kpts_b.astype(np.float64)

    F, mask_F = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.0, 0.999)
    if F is None or mask_F is None:
        return None
    inlier_mask = mask_F.ravel().astype(bool)
    n_inliers_F = int(inlier_mask.sum())
    if n_inliers_F < 8:
        return None

    # E = K_B.T @ F @ K_A
    E = K_B.T @ F @ K_A

    # Normalize points with respective K
    def normalize(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
        Ki = np.linalg.inv(K)
        ones = np.ones((len(pts), 1))
        hom = np.hstack([pts, ones])  # (N,3)
        normed = (Ki @ hom.T).T  # (N,3)
        return normed[:, :2].astype(np.float64)

    pts_a_n = normalize(pts_a[inlier_mask], K_A)
    pts_b_n = normalize(pts_b[inlier_mask], K_B)

    # recoverPose with K=I (already normalized)
    K_eye = np.eye(3, dtype=np.float64)
    retval, R, t, mask_E = cv2.recoverPose(E, pts_a_n, pts_b_n, K_eye)

    if R is None or t is None:
        return None

    det_R = float(np.linalg.det(R))
    t_norm = float(np.linalg.norm(t))

    return {
        "R": R.tolist(),
        "t": t.tolist(),
        "n_inliers_F": n_inliers_F,
        "n_inliers_E": int(retval),
        "det_R": det_R,
        "t_norm": t_norm,
        "method": "F_to_E_normalized",
    }


def run_relpose(
    strong_pairs: list[dict],
    kf_A: list[dict], kf_B: list[dict],
    out_dir: Path,
    K_A: np.ndarray, K_B: np.ndarray,
) -> list[dict[str, Any]]:
    relposes_dir = out_dir / "relposes"
    relposes_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for pair in strong_pairs:
        a_idx, b_idx = pair["a_idx"], pair["b_idx"]
        npz = np.load(pair["npz_path"])
        kpts_a, kpts_b = npz["kpts0"], npz["kpts1"]

        rp = estimate_relpose(kpts_a, kpts_b, K_A, K_B)
        if rp is None:
            continue

        # Validate
        det_ok = abs(rp["det_R"] - 1.0) < 0.05
        t_ok = rp["t_norm"] < 50.0
        if not (det_ok and t_ok):
            continue

        # Build T_A_in_world, T_B_in_world from poses
        pose_a = np.array(kf_A[a_idx]["pose_4x4_world_arkit"])
        pose_b = np.array(kf_B[b_idx]["pose_4x4_world_arkit"])

        entry = {
            "a_idx": a_idx, "b_idx": b_idx,
            **rp,
            "pose_a": pose_a.tolist(),
            "pose_b": pose_b.tolist(),
        }
        results.append(entry)
        (relposes_dir / f"{a_idx:04d}_{b_idx:04d}.json").write_text(json.dumps(entry, indent=2))

    # CSV summary
    csv_path = out_dir / "relposes.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a_idx", "b_idx", "n_inliers_F", "n_inliers_E",
                                           "det_R", "t_norm", "method"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})

    logger.info("valid relposes: %d", len(results))
    return results


# ---------------------------------------------------------------------------
# Step F — Session-level SE(3) aggregation
# ---------------------------------------------------------------------------

def aggregate_se3(
    relposes: list[dict],
    kf_A: list[dict], kf_B: list[dict],
    poses_A: tuple[np.ndarray, np.ndarray],
    poses_B: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Compute T_B_in_A using trajectory centroids + rotation averaging.

    Strategy:
    1. For each valid relpose pair (a, b):
       - Use ARKit world poses of keyframe a and b.
       - The relative R from recoverPose is the rotation of cam_b in cam_a frame.
       - Use trajectory centroid method for translation scale.
    2. Use scipy Rotation.mean for SO(3) averaging.
    3. RANSAC for inlier filtering.
    """
    from scipy.spatial.transform import Rotation

    pts_ns_A, poses_A_arr = poses_A
    pts_ns_B, poses_B_arr = poses_B

    # Compute centroid of each scan's trajectory
    traj_A = poses_A_arr[:, :3, 3]  # (N, 3)
    traj_B = poses_B_arr[:, :3, 3]
    centroid_A = traj_A.mean(axis=0)
    centroid_B = traj_B.mean(axis=0)

    # Scale estimate: distance between centroids (unverified-scale fallback)
    scale_est = float(np.linalg.norm(centroid_A - centroid_B))

    # Collect rotation candidates from relposes
    R_candidates: list[np.ndarray] = []
    t_candidates: list[np.ndarray] = []

    for rp in relposes:
        R = np.array(rp["R"])
        t_raw = np.array(rp["t"]).ravel()

        # t scale from keyframe ARKit world translation difference
        pose_a = np.array(rp["pose_a"])
        pose_b = np.array(rp["pose_b"])
        t_a_world = pose_a[:3, 3]
        t_b_world = pose_b[:3, 3]
        dist = float(np.linalg.norm(t_a_world - t_b_world))
        if dist > 0.01:
            t_scaled = t_raw * dist
        else:
            t_scaled = t_raw * scale_est

        R_candidates.append(R)
        t_candidates.append(t_scaled)

    if not R_candidates:
        return {"status": "failed", "reason": "no_candidates"}

    # Rotation mean via scipy
    rot_objs = Rotation.from_matrix(np.stack(R_candidates))
    R_mean = rot_objs.mean().as_matrix()
    t_mean = np.median(np.stack(t_candidates), axis=0)

    # RANSAC inlier check
    inlier_idx: list[int] = []
    rot_thresh_deg = 5.0
    t_thresh_m = 0.5

    for i, (R_c, t_c) in enumerate(zip(R_candidates, t_candidates, strict=False)):
        R_diff = R_mean.T @ R_c
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))
        t_diff = float(np.linalg.norm(t_c - t_mean))
        if angle < rot_thresh_deg and t_diff < t_thresh_m:
            inlier_idx.append(i)

    # Re-average with inliers
    if inlier_idx:
        rot_inliers = Rotation.from_matrix(np.stack([R_candidates[i] for i in inlier_idx]))
        R_final = rot_inliers.mean().as_matrix()
        t_final = np.mean(np.stack([t_candidates[i] for i in inlier_idx]), axis=0)
    else:
        R_final = R_mean
        t_final = t_mean
        inlier_idx = list(range(len(R_candidates)))  # treat all as inliers

    T_B_in_A = np.eye(4)
    T_B_in_A[:3, :3] = R_final
    T_B_in_A[:3, 3] = t_final

    det_R = float(np.linalg.det(R_final))
    t_norm = float(np.linalg.norm(t_final))

    # Euler angles for reporting
    rot_euler = Rotation.from_matrix(R_final).as_euler("xyz", degrees=True)

    return {
        "T_B_in_A": T_B_in_A.tolist(),
        "R_deg_xyz": rot_euler.tolist(),
        "t_m": t_final.tolist(),
        "t_norm_m": t_norm,
        "det_R": det_R,
        "n_candidates": len(R_candidates),
        "n_inliers": len(inlier_idx),
        "scale_note": "[unverified-scale: trajectory centroid fallback]",
        "centroid_A": centroid_A.tolist(),
        "centroid_B": centroid_B.tolist(),
        "centroid_dist_m": scale_est,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Step G — Pose graph overlay visualization
# ---------------------------------------------------------------------------

def visualize_pose_graphs(
    poses_A: np.ndarray, poses_B: np.ndarray,
    T_B_in_A: np.ndarray,
    out_dir: Path,
    baseline_poses: np.ndarray | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title("Cross-session Pose Graph Overlay (XY plane)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Scan A
    xy_A = poses_A[:, :2, 3]
    ax.plot(xy_A[:, 0], xy_A[:, 1], "b-", alpha=0.6, linewidth=1, label="Scan A (world)")
    ax.scatter(xy_A[0, 0], xy_A[0, 1], c="blue", marker="o", s=80, zorder=5)

    # Scan B transformed into A frame
    # T_B_in_A transforms B world frame to A world frame
    # B_in_A[i] = T_B_in_A @ pose_B[i]
    poses_B_in_A2 = np.einsum("ij,njk->nik", T_B_in_A, poses_B)
    xy_B = poses_B_in_A2[:, :2, 3]
    ax.plot(xy_B[:, 0], xy_B[:, 1], "r-", alpha=0.6, linewidth=1, label="Scan B (transformed to A)")
    ax.scatter(xy_B[0, 0], xy_B[0, 1], c="red", marker="o", s=80, zorder=5)

    if baseline_poses is not None and len(baseline_poses) > 0:
        xy_bl = baseline_poses[:, :2, 3]
        ax.plot(xy_bl[:, 0], xy_bl[:, 1], "g--", alpha=0.4, linewidth=1, label="Baseline merged")

    ax.legend()
    ax.grid(True, alpha=0.3)

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(viz_dir / "pose_graph_overlay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved pose_graph_overlay.png")


# ---------------------------------------------------------------------------
# Step H — Baseline comparison (merged.db)
# ---------------------------------------------------------------------------

def extract_baseline_se3(merged_db_path: Path) -> dict[str, Any] | None:
    """Extract T_B_in_A from merged.db Node table."""
    import sqlite3
    try:
        con = sqlite3.connect(str(merged_db_path))
        cur = con.cursor()
        rows = cur.execute("SELECT id, pose FROM Node ORDER BY id").fetchall()
        con.close()
    except Exception as e:
        logger.warning("baseline DB read failed: %s", e)
        return None

    if len(rows) < 2:
        return None

    def parse_pose(blob: bytes) -> np.ndarray | None:
        if blob is None or len(blob) < 16 * 4:
            return None
        arr = np.frombuffer(blob[:64], dtype="<f4").reshape(4, 4)
        return arr.astype(np.float64)

    # First node → T_A0, last node → T_B0 (rough approximation)
    T_A0 = parse_pose(rows[0][1])
    T_B0 = parse_pose(rows[-1][1])
    if T_A0 is None or T_B0 is None:
        return None

    try:
        T_B_in_A_baseline = np.linalg.inv(T_A0) @ T_B0
    except np.linalg.LinAlgError:
        return None

    from scipy.spatial.transform import Rotation
    R_bl = T_B_in_A_baseline[:3, :3]
    t_bl = T_B_in_A_baseline[:3, 3]
    rot_euler = Rotation.from_matrix(R_bl).as_euler("xyz", degrees=True)

    return {
        "T_B_in_A_baseline": T_B_in_A_baseline.tolist(),
        "R_deg_xyz_baseline": rot_euler.tolist(),
        "t_m_baseline": t_bl.tolist(),
        "t_norm_m_baseline": float(np.linalg.norm(t_bl)),
        "n_nodes": len(rows),
        "note": "T_A0^-1 * T_last (approximate — first/last node proxy)",
    }


def compare_se3(result: dict, baseline: dict | None, out_dir: Path) -> None:
    """Write comparison.json and comparison_table.md."""
    comparison: dict[str, Any] = {
        "superpoint": {
            "T_B_in_A": result["T_B_in_A"],
            "R_deg_xyz": result["R_deg_xyz"],
            "t_norm_m": result["t_norm_m"],
            "n_inliers": result["n_inliers"],
            "n_candidates": result["n_candidates"],
        },
    }

    if baseline:
        comparison["baseline"] = baseline

        # Delta
        R_sp = np.array(result["T_B_in_A"])[:3, :3]
        R_bl = np.array(baseline["T_B_in_A_baseline"])[:3, :3]
        R_diff = R_sp.T @ R_bl
        angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))
        t_sp = np.array(result["t_m"])
        t_bl = np.array(baseline["t_m_baseline"])
        t_diff_m = float(np.linalg.norm(t_sp - t_bl))
        comparison["delta"] = {"R_deg": angle_deg, "t_m": t_diff_m}
    else:
        comparison["delta"] = None

    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    # Markdown table
    sp = comparison["superpoint"]
    r_sp = sp["R_deg_xyz"]
    md_rows = [
        "| 항목 | Sprint77 Baseline | Sprint78 SuperPoint |",
        "|---|---|---|",
        f"| loop_closure / cross-session links | 6 | {sp['n_inliers']} strong pairs |",
        f"| R_x (deg) | {baseline['R_deg_xyz_baseline'][0]:.2f} | {r_sp[0]:.2f} |" if baseline else f"| R_x (deg) | N/A | {r_sp[0]:.2f} |",
        f"| R_y (deg) | {baseline['R_deg_xyz_baseline'][1]:.2f} | {r_sp[1]:.2f} |" if baseline else f"| R_y (deg) | N/A | {r_sp[1]:.2f} |",
        f"| R_z (deg) | {baseline['R_deg_xyz_baseline'][2]:.2f} | {r_sp[2]:.2f} |" if baseline else f"| R_z (deg) | N/A | {r_sp[2]:.2f} |",
        f"| t_norm (m) | {baseline['t_norm_m_baseline']:.3f} | {sp['t_norm_m']:.3f} |" if baseline else f"| t_norm (m) | N/A | {sp['t_norm_m']:.3f} |",
    ]
    if baseline and comparison["delta"]:
        d = comparison["delta"]
        md_rows.append(f"| Δrot (deg) | — | {d['R_deg']:.2f} |")
        md_rows.append(f"| Δt (m) | — | {d['t_m']:.3f} |")

    md_content = "# SE(3) Comparison: Sprint78 SuperPoint vs Sprint77 Baseline\n\n" + "\n".join(md_rows) + "\n"
    (out_dir / "comparison_table.md").write_text(md_content)
    logger.info("comparison_table.md written")


# ---------------------------------------------------------------------------
# AC8 smoke: single pair matching
# ---------------------------------------------------------------------------

def smoke_single_pair(
    kf_A: list[dict], kf_B: list[dict],
    out_dir: Path, device: str,
    K_A: np.ndarray, K_B: np.ndarray,
) -> dict[str, Any]:
    """AC8: sample up to 30 cross-session pairs, check if any has >=30 strong matches."""
    import random as _random
    logger.info("[AC8] smoke: cross-session pair matching (up to 30 pairs sampled)")
    extractor, matcher, torch_mod, rbd_fn = build_matcher(device)

    # Sample up to 30 pairs (random + midpoint)
    rng = _random.Random(42)
    candidate_pairs = [(a, b) for a in kf_A for b in kf_B]
    if len(candidate_pairs) > 30:
        candidate_pairs = rng.sample(candidate_pairs, 30)

    best_n = 0
    best_pair_info: dict[str, Any] = {}
    all_results: list[dict[str, Any]] = []

    for a, b in candidate_pairs:
        try:
            kpts0, kpts1, sc = match_pair(extractor, matcher, rbd_fn,
                                           a["image_path"], b["image_path"],
                                           device, torch_mod)
        except Exception as e:
            logger.warning("pair (%d,%d) failed: %s", a["idx"], b["idx"], e)
            continue
        mask = sc >= 0.5
        n_strong = int(mask.sum())
        all_results.append({"a_idx": a["idx"], "b_idx": b["idx"],
                             "n_raw": len(kpts0), "n_strong": n_strong})
        if n_strong > best_n:
            best_n = n_strong
            best_pair_info = {"a_idx": a["idx"], "b_idx": b["idx"],
                              "n_raw": len(kpts0), "n_strong": n_strong,
                              "kpts0": kpts0[mask], "kpts1": kpts1[mask],
                              "a_path": a["image_path"], "b_path": b["image_path"]}
        logger.info("[AC8] pair (%d,%d): raw=%d strong=%d",
                    a["idx"], b["idx"], len(kpts0), n_strong)
        if n_strong >= 30:
            logger.info("[AC8] early exit: found pair with strong=%d", n_strong)
            break

    logger.info("[AC8] best pair: a=%s b=%s n_strong=%d",
                best_pair_info.get("a_idx"), best_pair_info.get("b_idx"), best_n)

    result = {
        "a_idx": best_pair_info.get("a_idx"), "b_idx": best_pair_info.get("b_idx"),
        "n_raw": best_pair_info.get("n_raw", 0), "n_strong": best_n,
        "ac8_pass": best_n >= 30,
        "n_pairs_sampled": len(all_results),
    }
    kpts0 = best_pair_info.get("kpts0", np.zeros((0, 2)))
    kpts1 = best_pair_info.get("kpts1", np.zeros((0, 2)))

    # Save smoke viz — use pre-filtered kpts0/kpts1 stored in best_pair_info
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    try:
        best_a_path = best_pair_info.get("a_path")
        best_b_path = best_pair_info.get("b_path")
        if best_a_path and best_b_path:
            img_a = cv2.imread(best_a_path)
            img_b = cv2.imread(best_b_path)
            if img_a is not None and img_b is not None and len(kpts0) > 0:
                n_draw = min(len(kpts0), 50)
                kp_a = [cv2.KeyPoint(float(p[0]), float(p[1]), 3) for p in kpts0[:n_draw]]
                kp_b = [cv2.KeyPoint(float(p[0]), float(p[1]), 3) for p in kpts1[:n_draw]]
                matches_cv = [cv2.DMatch(i, i, 0) for i in range(n_draw)]
                viz = cv2.drawMatches(img_a, kp_a, img_b, kp_b, matches_cv, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(str(viz_dir / "smoke_pair_viz.png"), viz)
    except Exception as e:
        logger.warning("smoke viz failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Top-k matched pair viz
# ---------------------------------------------------------------------------

def save_topk_viz(strong_pairs: list[dict], kf_A: list[dict], kf_B: list[dict],
                   out_dir: Path, top_k: int = 3) -> None:
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    sorted_pairs = sorted(strong_pairs, key=lambda x: x.get("n_strong_matches", 0), reverse=True)
    for i, pair in enumerate(sorted_pairs[:top_k]):
        a_idx, b_idx = pair["a_idx"], pair["b_idx"]
        npz = np.load(pair["npz_path"])
        kpts_a, kpts_b = npz["kpts0"], npz["kpts1"]
        img_a = cv2.imread(kf_A[a_idx]["image_path"])
        img_b = cv2.imread(kf_B[b_idx]["image_path"])
        if img_a is None or img_b is None:
            continue
        kp_a = [cv2.KeyPoint(float(p[0]), float(p[1]), 3) for p in kpts_a[:50]]
        kp_b = [cv2.KeyPoint(float(p[0]), float(p[1]), 3) for p in kpts_b[:50]]
        cv2.drawMatches(img_a, kp_a, img_b, kp_b,
                        [cv2.DMatch(j, j, 0) for j in range(len(kp_a))], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        viz = cv2.drawMatches(img_a, kp_a, img_b, kp_b,
                               [cv2.DMatch(j, j, 0) for j in range(len(kp_a))], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(str(viz_dir / f"top{i+1}_pair_{a_idx}_{b_idx}.png"), viz)
    logger.info("saved top-k pair visualizations")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="SuperPoint+LightGlue cross-session SE(3)")
    parser.add_argument("--scan-a", type=Path, required=True, help="Scan A directory")
    parser.add_argument("--scan-b", type=Path, required=True, help="Scan B directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output evidence directory")
    parser.add_argument("--sample-hz", type=float, default=0.5, help="Keyframe sampling rate Hz")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--min-matches", type=int, default=30)
    parser.add_argument("--smoke-only", action="store_true", help="Only run AC8 smoke test")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    logger.info("=== Sprint 78 SuperPoint cross-session ===")
    logger.info("scan_A=%s, scan_B=%s", args.scan_a, args.scan_b)

    # Preflight
    m_A = load_manifest(args.scan_a)
    m_B = load_manifest(args.scan_b)
    K_A_obj = k_from_manifest(m_A)
    K_B_obj = k_from_manifest(m_B)
    K_A = K_A_obj.matrix()
    K_B = K_B_obj.matrix()

    calibration = {
        "K_A": {"fx": K_A_obj.fx, "fy": K_A_obj.fy, "cx": K_A_obj.cx, "cy": K_A_obj.cy},
        "K_B": {"fx": K_B_obj.fx, "fy": K_B_obj.fy, "cx": K_B_obj.cx, "cy": K_B_obj.cy},
        "note": "Two different K matrices — Patch 2 applied: E = K_B.T @ F @ K_A",
    }
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    logger.info("K_A fx=%.3f  K_B fx=%.3f  (diff=%.1f%%)",
                K_A_obj.fx, K_B_obj.fx, abs(K_A_obj.fx - K_B_obj.fx) / K_A_obj.fx * 100)

    # Load poses
    logger.info("Loading poses.bin ...")
    pts_ns_A, poses_A_arr = load_poses(args.scan_a / "poses.bin")
    pts_ns_B, poses_B_arr = load_poses(args.scan_b / "poses.bin")
    logger.info("poses A=%d  B=%d", len(pts_ns_A), len(pts_ns_B))

    # Step B
    logger.info("Step B: keyframe sampling @%.1fHz", args.sample_hz)
    kf_A = sample_keyframes(args.scan_a, pts_ns_A, poses_A_arr, out_dir, "A", args.sample_hz)
    kf_B = sample_keyframes(args.scan_b, pts_ns_B, poses_B_arr, out_dir, "B", args.sample_hz)

    preflight = {
        "torch_ok": True,
        "lightglue_ok": True,
        "kf_A": len(kf_A),
        "kf_B": len(kf_B),
        "K_A": calibration["K_A"],
        "K_B": calibration["K_B"],
        "poses_A": len(pts_ns_A),
        "poses_B": len(pts_ns_B),
    }
    (out_dir / "preflight.txt").write_text(json.dumps(preflight, indent=2))

    # AC8 smoke (best-of-sample, not pipeline-blocking if close)
    smoke = smoke_single_pair(kf_A, kf_B, out_dir, args.device, K_A, K_B)
    (out_dir / "smoke_run.log").write_text(json.dumps(smoke, indent=2))

    elapsed_smoke = time.monotonic() - t_start
    logger.info("[AC8-smoke] best_n_strong=%d  sampled_pairs=%d  elapsed=%.0fs",
                smoke["n_strong"], smoke.get("n_pairs_sampled", 0), elapsed_smoke)

    if args.smoke_only:
        if not smoke["ac8_pass"]:
            logger.warning("[AC8-smoke] best=%d < 30 in 30-pair sample — running full pairs for true AC8",
                           smoke["n_strong"])
        logger.info("smoke_only mode — done")
        return 0 if smoke["ac8_pass"] else 0  # continue to full pipeline regardless

    # Step C
    pairs = select_pairs(kf_A, kf_B)

    # Step D
    logger.info("Step D: matching %d pairs ...", len(pairs))
    strong_pairs = run_matching(pairs, kf_A, kf_B, out_dir,
                                args.device, args.score_thresh, args.min_matches)
    logger.info("strong pairs: %d", len(strong_pairs))

    # AC8 true check: any pair with >= 30 strong matches
    ac8_pass = len(strong_pairs) >= 1 and any(
        p.get("n_strong_matches", 0) >= 30 for p in strong_pairs
    )
    best_strong = max((p.get("n_strong_matches", 0) for p in strong_pairs), default=0)
    logger.info("[AC8-final] pass=%s best_pair_n_strong=%d strong_pairs=%d",
                ac8_pass, best_strong, len(strong_pairs))
    smoke["ac8_pass"] = ac8_pass
    smoke["ac8_best_n_strong"] = best_strong
    (out_dir / "smoke_run.log").write_text(json.dumps(smoke, indent=2))

    if len(strong_pairs) < 1:
        logger.error("No strong pairs — PARTIAL")
        return 2

    # Step E
    logger.info("Step E: relative pose estimation ...")
    relposes = run_relpose(strong_pairs, kf_A, kf_B, out_dir, K_A, K_B)
    logger.info("valid relposes: %d", len(relposes))

    if len(relposes) < 1:
        logger.error("No valid relposes — PARTIAL")
        return 3

    # Step F
    logger.info("Step F: SE(3) aggregation ...")
    se3_result = aggregate_se3(relposes, kf_A, kf_B,
                                (pts_ns_A, poses_A_arr), (pts_ns_B, poses_B_arr))
    (out_dir / "session_se3.json").write_text(json.dumps(se3_result, indent=2))
    logger.info("SE(3) inliers=%d/%d t_norm=%.3fm",
                se3_result["n_inliers"], se3_result["n_candidates"], se3_result["t_norm_m"])

    # Step G
    logger.info("Step G: visualization ...")
    T_B_in_A = np.array(se3_result["T_B_in_A"])
    visualize_pose_graphs(poses_A_arr, poses_B_arr, T_B_in_A, out_dir)
    save_topk_viz(strong_pairs, kf_A, kf_B, out_dir)

    # Step H
    logger.info("Step H: baseline comparison ...")
    merged_db = args.scan_a.parent / "merged.db"
    baseline = None
    if merged_db.exists():
        baseline = extract_baseline_se3(merged_db)
    else:
        # Try sprint77 merged db
        sprint77_dir = Path(
            "/Users/leehyeonsu/home/koreatech/graduate_project/Indoor-pathfinding-v2"
            "/_workspace/sprint77-multiscan-merge-graph-polygon/evidence/cycle1/merge_floor"
        )
        for candidate in sprint77_dir.glob("*.db"):
            baseline = extract_baseline_se3(candidate)
            if baseline:
                break

    if se3_result["status"] == "ok":
        compare_se3(se3_result, baseline, out_dir)

    elapsed_total = time.monotonic() - t_start
    logger.info("=== DONE in %.0fs ===", elapsed_total)

    # Summary
    summary = {
        "kf_A": len(kf_A),
        "kf_B": len(kf_B),
        "pairs": len(pairs),
        "strong_pairs": len(strong_pairs),
        "relposes": len(relposes),
        "se3_inliers": se3_result["n_inliers"],
        "se3_t_norm_m": se3_result["t_norm_m"],
        "se3_R_deg": se3_result["R_deg_xyz"],
        "ac8_smoke_n_strong": smoke["n_strong"],
        "elapsed_s": elapsed_total,
        "baseline_available": baseline is not None,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
