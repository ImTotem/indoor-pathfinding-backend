"""Sprint 81 Cycle 2 — Step C: ARKit-pose-based SE(3) alignment + merged.db patch.

두 스캔의 ARKit pose를 RANSAC으로 정렬해 T_AB(B→A 좌표변환)를 추정하고,
merged.db의 Node.pose와 map_id를 실제 값으로 패치한다.

Input:
  - merged.db (rtabmap-reprocess 산출, pose=0/map_id=0)
  - scan_A.db (ARKit pose 보유, map_id=0)
  - scan_B.db (ARKit pose 보유, map_id=1)

Output:
  - merged_v2.db  (pose ≠ 0, map_id={0,1}, B pose aligned to A frame)
  - merge_metrics_v2.json
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ─── pose blob helpers ───────────────────────────────────────────────────────

def blob_to_T(blob: bytes) -> np.ndarray:
    """3x4 float32 row-major blob → 4x4 float64 T."""
    v = struct.unpack("<12f", bytes(blob))
    T = np.eye(4, dtype=np.float64)
    T[0, :3] = v[0:3];  T[0, 3] = v[3]
    T[1, :3] = v[4:7];  T[1, 3] = v[7]
    T[2, :3] = v[8:11]; T[2, 3] = v[11]
    return T


def T_to_blob(T: np.ndarray) -> bytes:
    """4x4 float64 → 3x4 float32 row-major blob (48B)."""
    flat = [
        T[0, 0], T[0, 1], T[0, 2], T[0, 3],
        T[1, 0], T[1, 1], T[1, 2], T[1, 3],
        T[2, 0], T[2, 1], T[2, 2], T[2, 3],
    ]
    return struct.pack("<12f", *[float(x) for x in flat])


# ─── loading ─────────────────────────────────────────────────────────────────

def load_poses(db_path: Path) -> dict[int, np.ndarray]:
    """node_id → 4x4 T (float64)."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, pose FROM Node ORDER BY id")
    poses: dict[int, np.ndarray] = {}
    for nid, blob in cur.fetchall():
        if blob:
            poses[nid] = blob_to_T(blob)
    conn.close()
    return poses


def load_cross_session_links(merged_db: Path, a_count: int) -> list[tuple[int, int]]:
    """merged.db의 GlobalClosure(type=1) 중 A↔B cross-session 쌍 반환.

    merged.db 에서 A node id = 1..a_count, B node id = (a_count+1)..(a_count+b_count).
    양방향 Link 중 from < to인 쌍만 반환.
    """
    conn = sqlite3.connect(str(merged_db))
    cur = conn.cursor()
    cur.execute(
        """SELECT from_id, to_id FROM Link
           WHERE type=1 AND from_id < to_id
             AND from_id <= ? AND to_id > ?""",
        (a_count, a_count),
    )
    pairs = cur.fetchall()
    conn.close()
    return pairs  # (merged_A_id, merged_B_id)


# ─── RANSAC SE(3) estimation ─────────────────────────────────────────────────

def estimate_T_AB_ransac(
    a_poses: dict[int, np.ndarray],
    b_poses: dict[int, np.ndarray],
    cross_pairs: list[tuple[int, int]],
    a_count: int,
) -> tuple[np.ndarray, float, int]:
    """RANSAC으로 T_AB(B→A 좌표변환) 추정.

    각 loop-closure pair (a_node, b_merged_node)에서:
      T_AB = T_A @ inv(T_B)   (A좌표계에서 B origin 위치)

    Returns:
        T_AB (4x4), inlier_rmse, inlier_count
    """
    T_AB_list: list[np.ndarray] = []
    for a_id, b_merged_id in cross_pairs:
        b_id = b_merged_id - a_count   # scan_B 원본 id
        if a_id in a_poses and b_id in b_poses:
            T_a = a_poses[a_id]
            T_b = b_poses[b_id]
            T_AB = T_a @ np.linalg.inv(T_b)
            T_AB_list.append(T_AB)

    if not T_AB_list:
        raise ValueError("No valid cross-session pairs for T_AB estimation")

    translations = np.array([T[:3, 3] for T in T_AB_list])

    # RANSAC: median + MAD로 inlier 선택
    median_t = np.median(translations, axis=0)
    mad = np.median(np.abs(translations - median_t), axis=0)
    thresh = mad * 3.0 + 0.3   # minimum 0.3m threshold
    inlier_mask = np.all(np.abs(translations - median_t) < thresh, axis=1)

    logger.info(
        "T_AB RANSAC: %d/%d inliers (median_t=%s, MAD=%s)",
        inlier_mask.sum(), len(inlier_mask), median_t.round(3), mad.round(3),
    )

    if inlier_mask.sum() < 3:
        # inlier 너무 적으면 전체 사용
        logger.warning("Too few RANSAC inliers — using all pairs")
        inlier_mask = np.ones(len(T_AB_list), dtype=bool)

    inlier_T = np.array(T_AB_list)[inlier_mask]

    # Translation mean
    best_t = inlier_T[:, :3, 3].mean(axis=0)

    # Rotation averaging (Frobenius mean → nearest SO(3))
    R_mean = inlier_T[:, :3, :3].mean(axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R_best = U @ Vt
    if np.linalg.det(R_best) < 0:
        Vt[-1] *= -1
        R_best = U @ Vt

    T_AB = np.eye(4, dtype=np.float64)
    T_AB[:3, :3] = R_best
    T_AB[:3, 3] = best_t

    # RMSE on inliers
    errs = [
        np.linalg.norm((T_AB @ np.linalg.inv(T_AB_list[i]))[:3, 3])
        for i in range(len(T_AB_list)) if inlier_mask[i]
    ]
    rmse = float(np.sqrt(np.mean(np.array(errs) ** 2))) if errs else 0.0

    rot_deg = Rotation.from_matrix(R_best).as_euler("xyz", degrees=True)
    logger.info(
        "T_AB: t=%s, R_euler_xyz_deg=%s, RMSE=%.4f, inliers=%d",
        best_t.round(3), rot_deg.round(2), rmse, inlier_mask.sum(),
    )

    return T_AB, rmse, int(inlier_mask.sum())


# ─── merged.db patching ──────────────────────────────────────────────────────

def patch_merged_db(
    src_merged_db: Path,
    out_db: Path,
    a_poses: dict[int, np.ndarray],
    b_poses: dict[int, np.ndarray],
    T_AB: np.ndarray,
    a_count: int,
) -> dict:
    """merged.db를 복사한 뒤 Node.pose와 map_id를 ARKit+정렬 pose로 패치.

    A nodes (1..a_count): 원본 ARKit pose 그대로 (map_id=0).
    B nodes (a_count+1..a_count+b_count): T_AB @ T_b (map_id=1).

    Returns:
        patch_summary dict
    """
    shutil.copy2(str(src_merged_db), str(out_db))
    logger.info("Copied %s → %s", src_merged_db, out_db)

    conn = sqlite3.connect(str(out_db))
    cur = conn.cursor()

    patched_a = 0
    patched_b = 0
    zero_pose_a = 0
    zero_pose_b = 0

    # A nodes
    cur.execute("SELECT id FROM Node WHERE id <= ?", (a_count,))
    a_node_ids = [r[0] for r in cur.fetchall()]
    for nid in a_node_ids:
        orig_id = nid  # scan_A 원본 id = merged id (1..a_count)
        if orig_id in a_poses:
            blob = T_to_blob(a_poses[orig_id])
            cur.execute(
                "UPDATE Node SET pose=?, map_id=0 WHERE id=?",
                (blob, nid),
            )
            patched_a += 1
        else:
            zero_pose_a += 1

    # B nodes
    cur.execute("SELECT id FROM Node WHERE id > ?", (a_count,))
    b_node_ids = [r[0] for r in cur.fetchall()]
    for nid in b_node_ids:
        b_orig_id = nid - a_count  # scan_B 원본 id
        if b_orig_id in b_poses:
            T_b_in_A = T_AB @ b_poses[b_orig_id]
            blob = T_to_blob(T_b_in_A)
            cur.execute(
                "UPDATE Node SET pose=?, map_id=1 WHERE id=?",
                (blob, nid),
            )
            patched_b += 1
        else:
            zero_pose_b += 1

    conn.commit()
    conn.close()

    logger.info(
        "Patched: A=%d (zero=%d), B=%d (zero=%d)",
        patched_a, zero_pose_a, patched_b, zero_pose_b,
    )
    return {
        "patched_a_nodes": patched_a,
        "patched_b_nodes": patched_b,
        "zero_pose_a": zero_pose_a,
        "zero_pose_b": zero_pose_b,
    }


# ─── metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(
    out_db: Path,
    T_AB: np.ndarray,
    t_ab_rmse: float,
    inlier_count: int,
    a_poses: dict[int, np.ndarray],
    b_poses: dict[int, np.ndarray],
    a_count: int,
) -> dict:
    conn = sqlite3.connect(str(out_db))
    cur = conn.cursor()

    cur.execute("SELECT map_id, COUNT(*) FROM Node GROUP BY map_id")
    map_id_dist = {int(r[0]): int(r[1]) for r in cur.fetchall()}

    cur.execute("SELECT COUNT(*) FROM Node WHERE length(pose)=48 AND pose != zeroblob(48)")
    # SQLite blob comparison needs cast
    cur.execute(
        "SELECT COUNT(*) FROM Node WHERE pose IS NOT NULL"
    )
    total_nodes = cur.fetchone()[0]

    # Non-zero pose check
    cur.execute("SELECT pose FROM Node LIMIT 5")
    sample_poses = cur.fetchall()
    nonzero_count = 0
    for (blob,) in sample_poses:
        if blob:
            v = struct.unpack("<12f", bytes(blob))
            if any(abs(x) > 1e-6 for x in [v[3], v[7], v[11]]):
                nonzero_count += 1

    cur.execute("SELECT COUNT(*) FROM Node WHERE id <= ?", (a_count,))
    a_merged_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM Node WHERE id > ?", (a_count,))
    b_merged_count = cur.fetchone()[0]

    cur.execute("SELECT type, COUNT(*) FROM Link GROUP BY type")
    link_dist = {int(r[0]): int(r[1]) for r in cur.fetchall()}

    cur.execute("SELECT COUNT(*) FROM Link WHERE type=1")
    gc_link_count = cur.fetchone()[0]
    loop_closure_pair_count = gc_link_count // 2

    conn.close()

    # Overlap estimate
    a_t = np.array([T[:3, 3] for T in a_poses.values()])
    b_in_A = np.array([(T_AB @ T)[:3, 3] for T in b_poses.values()])

    ax_range = (float(a_t[:, 0].min()), float(a_t[:, 0].max()))
    ay_range = (float(a_t[:, 1].min()), float(a_t[:, 1].max()))
    bx_range = (float(b_in_A[:, 0].min()), float(b_in_A[:, 0].max()))
    by_range = (float(b_in_A[:, 1].min()), float(b_in_A[:, 1].max()))

    ox = max(0.0, min(ax_range[1], bx_range[1]) - max(ax_range[0], bx_range[0]))
    oy = max(0.0, min(ay_range[1], by_range[1]) - max(ay_range[0], by_range[0]))
    overlap_bbox_m2 = float(ox * oy)

    rot_deg = Rotation.from_matrix(T_AB[:3, :3]).as_euler("xyz", degrees=True).tolist()

    metrics = {
        "node_count": a_merged_count + b_merged_count,
        "a_node_count": a_merged_count,
        "b_node_count": b_merged_count,
        "map_id_distribution": map_id_dist,
        "link_distribution": link_dist,
        "loop_closure_pair_count": loop_closure_pair_count,
        "loop_closure_link_count": gc_link_count,
        "pose_nonzero_sample_ratio": f"{nonzero_count}/5",
        "T_AB_translation_m": T_AB[:3, 3].tolist(),
        "T_AB_rotation_euler_xyz_deg": rot_deg,
        "T_AB_rmse_m": t_ab_rmse,
        "T_AB_inlier_pairs": inlier_count,
        "A_xy_range_m": {"x": ax_range, "y": ay_range},
        "B_in_A_xy_range_m": {"x": (float(b_in_A[:, 0].min()), float(b_in_A[:, 0].max())),
                               "y": (float(b_in_A[:, 1].min()), float(b_in_A[:, 1].max()))},
        "overlap_bbox_m2": overlap_bbox_m2,
    }
    return metrics


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Cycle 2: ARKit SE(3) alignment + merged.db pose patch"
    )
    parser.add_argument("merged_db", type=Path)
    parser.add_argument("scan_a_db", type=Path)
    parser.add_argument("scan_b_db", type=Path)
    parser.add_argument("out_db", type=Path, help="Output patched merged_v2.db")
    parser.add_argument("--metrics-json", type=Path, default=None)
    args = parser.parse_args()

    # Load ARKit poses
    a_poses = load_poses(args.scan_a_db)
    b_poses = load_poses(args.scan_b_db)
    a_count = max(a_poses.keys()) if a_poses else 0
    logger.info("scan_A: %d poses (max_id=%d)", len(a_poses), a_count)
    logger.info("scan_B: %d poses", len(b_poses))

    # Load cross-session links
    cross_pairs = load_cross_session_links(args.merged_db, a_count)
    logger.info("Cross-session GlobalClosure pairs: %d", len(cross_pairs))

    # RANSAC T_AB estimation
    T_AB, rmse, inlier_count = estimate_T_AB_ransac(a_poses, b_poses, cross_pairs, a_count)

    # Patch merged.db
    patch_info = patch_merged_db(
        args.merged_db, args.out_db, a_poses, b_poses, T_AB, a_count
    )

    # Compute metrics
    metrics = compute_metrics(
        args.out_db, T_AB, rmse, inlier_count, a_poses, b_poses, a_count
    )
    metrics.update(patch_info)

    # Write metrics
    metrics_path = args.metrics_json or args.out_db.parent / "merge_metrics_v2.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("\n=== merge_metrics_v2 ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved: {args.out_db}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
