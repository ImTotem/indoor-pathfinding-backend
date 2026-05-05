"""Sprint 81 Cycle 2 — Step 4v2: merged_v2.db → heatmap + floor polygon + visualization.

Cycle 2에서는 merged_v2.db의 Node.pose를 직접 읽는다 (ARKit SE(3) 정렬 완료 상태).
src_dbs fallback 없이 merged.db 단독 pose 사용.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPoint, mapping

logger = logging.getLogger(__name__)


def _blob_to_T(blob: bytes) -> np.ndarray:
    """3x4 float32 row-major → 4x4 T."""
    v = struct.unpack("<12f", bytes(blob))
    T = np.eye(4)
    T[0, :3] = v[0:3];  T[0, 3] = v[3]
    T[1, :3] = v[4:7];  T[1, 3] = v[7]
    T[2, :3] = v[8:11]; T[2, 3] = v[11]
    return T


def extract(merged_v2_db: Path, out_viz: Path, out_polygon: Path) -> dict:
    """merged_v2.db → PNG 시각화 + GeoJSON + metrics."""
    out_viz.mkdir(parents=True, exist_ok=True)
    out_polygon.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(merged_v2_db))
    conn.row_factory = sqlite3.Row

    # Node 읽기 — pose가 직접 들어있음
    nodes = conn.execute("SELECT id, map_id, pose FROM Node ORDER BY id").fetchall()
    logger.info("Loaded %d nodes from merged_v2.db", len(nodes))

    links = conn.execute("SELECT from_id, to_id, type FROM Link").fetchall()
    feature_count = conn.execute("SELECT COUNT(*) FROM Feature").fetchone()[0]
    conn.close()

    # Pose 파싱
    node_ids, map_ids, tx_list, ty_list, tz_list = [], [], [], [], []
    zero_pose_count = 0
    for n in nodes:
        nid = n["id"]
        mid = n["map_id"]
        blob = n["pose"]
        if blob is None or len(blob) < 48:
            zero_pose_count += 1
            continue
        v = struct.unpack("<12f", bytes(blob))
        tx, ty, tz = v[3], v[7], v[11]
        if abs(tx) < 1e-9 and abs(ty) < 1e-9 and abs(tz) < 1e-9 and nid > 1:
            zero_pose_count += 1
            continue
        node_ids.append(nid)
        map_ids.append(mid)
        tx_list.append(tx)
        ty_list.append(ty)
        tz_list.append(tz)

    logger.info("Valid poses: %d, zero-pose skipped: %d", len(node_ids), zero_pose_count)

    if not node_ids:
        raise RuntimeError("No valid poses in merged_v2.db")

    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    map_arr = np.array(map_ids)
    unique_maps = sorted(set(map_ids))

    pose_nonzero_ratio = len(node_ids) / max(len(nodes), 1)
    logger.info("pose_nonzero_ratio: %.3f (%d/%d)", pose_nonzero_ratio, len(node_ids), len(nodes))

    color_map = dict(zip(unique_maps, ["royalblue", "crimson", "green", "orange"]))

    loop_closures = [lk for lk in links if lk["type"] in (1, 2, 3)]
    neighbor_links = [lk for lk in links if lk["type"] == 0]
    per_session = {uid: int((map_arr == uid).sum()) for uid in unique_maps}

    metrics: dict = {
        "node_count": len(node_ids),
        "link_count": len(links),
        "neighbor_link_count": len(neighbor_links),
        "loop_closure_pair_count": len(loop_closures) // 2,
        "loop_closure_link_count": len(loop_closures),
        "per_session_node_count": per_session,
        "map_id_distribution": {str(k): v for k, v in per_session.items()},
        "pose_nonzero_ratio": round(pose_nonzero_ratio, 4),
        "feature_row_count": feature_count,
        "coordinate_system": "merged_v2 (B aligned to A via RANSAC SE3)",
    }

    nid_to_xy = {node_ids[i]: (tx_list[i], ty_list[i]) for i in range(len(node_ids))}

    # ── 1. Pose graph ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    for uid in unique_maps:
        mask = map_arr == uid
        ax.scatter(tx_arr[mask], ty_arr[mask], s=10,
                   c=color_map[uid], label=f"Session {uid}", alpha=0.8, zorder=3)

    # Neighbor links (gray)
    for lk in neighbor_links[:3000]:
        f, t = lk["from_id"], lk["to_id"]
        if f in nid_to_xy and t in nid_to_xy:
            ax.plot([nid_to_xy[f][0], nid_to_xy[t][0]],
                    [nid_to_xy[f][1], nid_to_xy[t][1]],
                    c="lightgray", lw=0.4, zorder=1)

    # Cross-session loop closure links (green)
    cross_drawn = 0
    for lk in loop_closures:
        f, t = lk["from_id"], lk["to_id"]
        if f in nid_to_xy and t in nid_to_xy:
            ax.plot([nid_to_xy[f][0], nid_to_xy[t][0]],
                    [nid_to_xy[f][1], nid_to_xy[t][1]],
                    c="lime", lw=1.2, alpha=0.6, zorder=5)
            cross_drawn += 1

    ax.set_xlabel("X (m) — A coordinate frame")
    ax.set_ylabel("Y (m) — A coordinate frame")
    ax.set_title(
        f"Pose Graph (merged_v2) — {len(node_ids)} nodes, "
        f"{len(loop_closures)//2} loop-closure pairs, "
        f"{cross_drawn} cross-session links drawn\n"
        f"[B aligned to A via RANSAC SE(3) — NOT rtabmap-reprocess optimization]"
    )
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_viz / "pose_graph.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", p)

    # ── 2. Pose density heatmap ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    if len(tx_arr) > 1:
        h, xe, ye = np.histogram2d(tx_arr, ty_arr, bins=80)
        ax.imshow(np.log1p(h.T), origin="lower",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]],
                  cmap="hot", aspect="auto")
    for uid in unique_maps:
        mask = map_arr == uid
        ax.scatter(tx_arr[mask], ty_arr[mask], s=3,
                   c=color_map[uid], alpha=0.5, label=f"Session {uid}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Pose Density Heatmap — {len(tx_arr)} nodes (merged_v2 coordinate)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_viz / "heatmap_pose.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", p)

    # ── 3. Feature density heatmap ────────────────────────────────────────────
    conn2 = sqlite3.connect(str(merged_v2_db))
    feature_per_node: dict[int, int] = {}
    if feature_count > 0:
        rows = conn2.execute(
            "SELECT node_id, COUNT(*) FROM Feature GROUP BY node_id"
        ).fetchall()
        feature_per_node = {r[0]: r[1] for r in rows}
    conn2.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    if feature_per_node:
        sizes = np.array([feature_per_node.get(nid, 0) for nid in node_ids], dtype=float)
        sizes_norm = np.clip(sizes / (sizes.max() + 1e-6) * 200 + 5, 2, 200)
        sc = ax.scatter(tx_arr, ty_arr, s=sizes_norm, c=sizes_norm, cmap="YlOrRd", alpha=0.7)
        plt.colorbar(sc, ax=ax, label="feature count (normalized)")
        ax.set_title(f"Feature Density — {sum(feature_per_node.values())} features total")
    else:
        ax.scatter(tx_arr, ty_arr, s=5, c="steelblue", alpha=0.6)
        ax.set_title("Feature Density (no features — visual-only DB)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_viz / "heatmap_features.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", p)

    # ── 4. Floor polygon ──────────────────────────────────────────────────────
    polygon_area = 0.0
    pts_xy = list(zip(tx_list, ty_list))

    if len(pts_xy) >= 4:
        mp = MultiPoint(pts_xy)
        try:
            hull = mp.concave_hull(ratio=0.3)
            if hull.is_empty or hull.area < 0.1:
                hull = mp.convex_hull
        except Exception:
            hull = mp.convex_hull

        if hull.area < 5.0:
            try:
                hull = mp.concave_hull(ratio=0.5)
            except Exception:
                pass
        if hull.area < 5.0:
            hull = mp.convex_hull

        polygon_area = float(hull.area)
        logger.info("Polygon area (merged_v2): %.2f m²", polygon_area)

        # GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(hull),
                    "properties": {
                        "area_m2": polygon_area,
                        "node_count": len(pts_xy),
                        "session_count": len(unique_maps),
                        "coordinate_system": "merged_v2 (B aligned to A)",
                        "note": "B scan aligned to A coordinate frame via RANSAC SE(3)",
                    },
                }
            ],
        }
        (out_polygon / "floor_polygon.geojson").write_text(json.dumps(geojson, indent=2))

        # Polygon PNG
        fig, ax = plt.subplots(figsize=(12, 10))
        if hull.geom_type == "Polygon":
            xs, ys = hull.exterior.xy
            ax.fill(xs, ys, alpha=0.25, fc="royalblue", ec="navy", lw=1.5, label="Floor polygon")
        elif hull.geom_type == "MultiPolygon":
            for geom in hull.geoms:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=0.25, fc="royalblue", ec="navy", lw=1.5)
        for uid in unique_maps:
            mask = map_arr == uid
            ax.scatter(tx_arr[mask], ty_arr[mask], s=8,
                       c=color_map[uid], label=f"Session {uid}", alpha=0.7, zorder=3)
        ax.set_xlabel("X (m) — A frame")
        ax.set_ylabel("Y (m) — A frame")
        ax.set_title(
            f"Floor Polygon (merged_v2) — area={polygon_area:.1f} m²\n"
            f"[B scan RANSAC-aligned to A; T_AB_tx=1.39m, T_AB_ty=-24.98m, yaw=-178.2°]"
        )
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_polygon / "floor_polygon.png"
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", p)

    metrics["polygon_area_m2"] = polygon_area

    return metrics


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Sprint 81 Cycle 2 Step 4v2: heatmap + polygon from merged_v2.db"
    )
    parser.add_argument("merged_v2_db", type=Path)
    parser.add_argument("out_viz", type=Path)
    parser.add_argument("out_polygon", type=Path)
    parser.add_argument("--metrics-json", type=Path, default=None)
    args = parser.parse_args()

    metrics = extract(args.merged_v2_db, args.out_viz, args.out_polygon)

    metrics_path = args.metrics_json or args.out_viz.parent / "merge_metrics_v2.json"
    # merge with existing if already written
    try:
        existing = json.loads(metrics_path.read_text())
        existing.update(metrics)
        metrics_path.write_text(json.dumps(existing, indent=2))
    except Exception:
        metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n=== Polygon + heatmap metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
