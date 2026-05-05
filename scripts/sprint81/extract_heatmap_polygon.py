"""Sprint 81 — Step 4: merged.db → heatmap + floor polygon + visualization.

merged.db → pose_graph.png, heatmap_pose.png, heatmap_features.png,
            floor_polygon.geojson, floor_polygon.png, merged_metrics.json
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
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def _read_pose_translation(blob: bytes) -> tuple[float, float, float]:
    """3x4 float32 row-major → (tx, ty, tz)."""
    floats = struct.unpack("<12f", blob)
    # [row0: r00 r01 r02 tx | row1 ... | row2 ...]
    tx = floats[3]
    ty = floats[7]
    tz = floats[11]
    return tx, ty, tz


def extract(merged_db: Path, out_viz: Path, out_polygon: Path,
             src_dbs: list[Path] | None = None) -> dict:
    """merged.db → PNG 시각화 + GeoJSON + metrics.

    Returns:
        metrics dict (also written to out_viz/../merged_metrics.json).
    """
    out_viz.mkdir(parents=True, exist_ok=True)
    out_polygon.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(merged_db))
    conn.row_factory = sqlite3.Row

    # merged.db Node: rtabmap-reprocess가 pose를 0으로 초기화하므로
    # src_dbs(원본 scan_A.db, scan_B.db)에서 pose를 가져옴
    nodes_merged = conn.execute("SELECT id, map_id, stamp FROM Node ORDER BY id").fetchall()
    logger.info("Loaded %d nodes from merged.db", len(nodes_merged))

    # Link 읽기 (merged.db에서 loop closure 정보)
    links = conn.execute("SELECT from_id, to_id, type FROM Link").fetchall()

    # Feature row 수 (rtabmap-reprocess가 추출한 features)
    feature_count = conn.execute("SELECT COUNT(*) FROM Feature").fetchone()[0]
    logger.info("Feature rows: %d", feature_count)
    conn.close()

    # src_dbs에서 pose 가져오기
    # merged.db 의 node id → src_db node id 매핑 필요
    # rtabmap-reprocess multi-session: scan_A node 1..314 → merged node 1..314
    #   scan_B node 1..294 → merged node 315..608 (offset by A.count)
    # 실제 매핑은 merged.db node 순서대로 src_dbs를 순서대로 읽으면 됨
    src_poses: dict[int, tuple[float, float, float, int]] = {}  # merged_id → (tx, ty, tz, map_id)

    if src_dbs:
        offset = 0
        for map_idx, src_db in enumerate(src_dbs):
            src_conn = sqlite3.connect(str(src_db))
            src_nodes = src_conn.execute(
                "SELECT id, pose FROM Node ORDER BY id"
            ).fetchall()
            for src_row in src_nodes:
                src_id, pose_blob = src_row
                if pose_blob:
                    floats = struct.unpack("<12f", pose_blob)
                    merged_id = src_id + offset
                    src_poses[merged_id] = (floats[3], floats[7], floats[11], map_idx)
            offset += len(src_nodes)
            src_conn.close()
        logger.info("Loaded %d poses from %d source DBs", len(src_poses), len(src_dbs))

    # Node data 파싱 — src_poses 우선, 없으면 merged.db pose(모두 0)
    # merged.db 에는 id 가 1..N 으로 연속 할당됨
    node_ids = []
    map_ids = []
    tx_list, ty_list, tz_list = [], [], []

    for n in nodes_merged:
        nid = n["id"]
        mid = n["map_id"]
        if nid in src_poses:
            tx, ty, tz, src_map = src_poses[nid]
            mid = src_map  # src_db 기반 map_id 사용
        else:
            # merged.db pose 없음 → skip
            continue
        node_ids.append(nid)
        map_ids.append(mid)
        tx_list.append(tx)
        ty_list.append(ty)
        tz_list.append(tz)

    if not node_ids and src_dbs:
        logger.warning("No poses from src_dbs matched — falling back to merged.db poses (all zeros)")
        # fallback: merged.db의 raw pose (모두 0이지만 표시)
        for n in nodes_merged:
            node_ids.append(n["id"])
            map_ids.append(n["map_id"])
            tx_list.append(0.0)
            ty_list.append(0.0)
            tz_list.append(0.0)

    nodes = nodes_merged  # keep for reference

    if not node_ids:
        raise RuntimeError("No nodes with valid poses — cannot visualize")

    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    tz_arr = np.array(tz_list)
    map_arr = np.array(map_ids)

    # unique map_ids → colors
    unique_maps = sorted(set(map_ids))
    color_map = {uid: c for uid, c in zip(unique_maps, ["royalblue", "crimson", "green", "orange"])}

    # metrics
    loop_closures = [l for l in links if l["type"] in (1, 2, 3)]  # GlobalClosure types
    neighbor_links = [l for l in links if l["type"] == 0]
    per_session = {uid: int((map_arr == uid).sum()) for uid in unique_maps}

    metrics = {
        "node_count": len(node_ids),
        "link_count": len(links),
        "neighbor_link_count": len(neighbor_links),
        "loop_closure_count": len(loop_closures),
        "per_session_node_count": per_session,
        "feature_row_count": feature_count,
    }

    # ── 1. Pose graph ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    for uid in unique_maps:
        mask = map_arr == uid
        ax.scatter(tx_arr[mask], ty_arr[mask], s=10,
                   c=color_map[uid], label=f"session {uid}", alpha=0.8, zorder=3)

    # Neighbor links (gray)
    nid_to_xy = {node_ids[i]: (tx_list[i], ty_list[i]) for i in range(len(node_ids))}
    for lk in neighbor_links[:2000]:  # limit for performance
        f, t = lk["from_id"], lk["to_id"]
        if f in nid_to_xy and t in nid_to_xy:
            fx, fy = nid_to_xy[f]
            tx_, ty_ = nid_to_xy[t]
            ax.plot([fx, tx_], [fy, ty_], c="lightgray", lw=0.5, zorder=1)

    # Loop closure links (green)
    for lk in loop_closures:
        f, t = lk["from_id"], lk["to_id"]
        if f in nid_to_xy and t in nid_to_xy:
            fx, fy = nid_to_xy[f]
            tx_, ty_ = nid_to_xy[t]
            ax.plot([fx, tx_], [fy, ty_], c="lime", lw=1.5, zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Pose Graph — {len(node_ids)} nodes, {len(loop_closures)} loop closures")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pose_graph_path = out_viz / "pose_graph.png"
    fig.savefig(str(pose_graph_path), dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", pose_graph_path)

    # ── 2. Heatmap (pose density) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    bins = 80
    if len(tx_arr) > 1:
        h, xedges, yedges = np.histogram2d(tx_arr, ty_arr, bins=bins)
        h = np.log1p(h.T)
        ax.imshow(h, origin="lower",
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  cmap="hot", aspect="auto")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Pose Density Heatmap — {len(tx_arr)} nodes")
    for uid in unique_maps:
        mask = map_arr == uid
        ax.scatter(tx_arr[mask], ty_arr[mask], s=3,
                   c=color_map[uid], alpha=0.4, label=f"session {uid}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    heatmap_pose_path = out_viz / "heatmap_pose.png"
    fig.savefig(str(heatmap_pose_path), dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", heatmap_pose_path)

    # ── 3. Feature heatmap (image-space density mapped by pose) ────────────
    # RGB-only DB → Feature.depth_x/y/z NULL → image-plane density fallback
    # 각 Node 의 Feature 개수를 pose 위치 점 크기에 매핑
    conn2 = sqlite3.connect(str(merged_db))
    feature_per_node = {}
    if feature_count > 0:
        rows = conn2.execute(
            "SELECT node_id, COUNT(*) as cnt FROM Feature GROUP BY node_id"
        ).fetchall()
        feature_per_node = {r[0]: r[1] for r in rows}
    conn2.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    if feature_per_node:
        sizes = np.array([feature_per_node.get(nid, 0) for nid in node_ids], dtype=float)
        sizes = np.clip(sizes / (sizes.max() + 1e-6) * 200 + 5, 2, 200)
        scatter = ax.scatter(tx_arr, ty_arr, s=sizes, c=sizes, cmap="YlOrRd", alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="feature count (normalized)")
        ax.set_title(f"Feature Density by Pose — {sum(feature_per_node.values())} features total")
    else:
        # feature가 없으면 pose-only 시각화로 대체
        ax.scatter(tx_arr, ty_arr, s=5, c="steelblue", alpha=0.6)
        ax.set_title("Feature Heatmap (no features yet — will be extracted by rtabmap-reprocess)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    heatmap_features_path = out_viz / "heatmap_features.png"
    fig.savefig(str(heatmap_features_path), dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", heatmap_features_path)

    # ── 4. Floor polygon (concave hull of pose XY) ─────────────────────────
    pts_xy = list(zip(tx_list, ty_list))
    polygon = None
    polygon_area = 0.0

    if len(pts_xy) >= 4:
        mp = MultiPoint(pts_xy)
        # try concave hull
        try:
            hull = mp.concave_hull(ratio=0.3)
            if hull.is_empty or hull.area < 0.1:
                hull = mp.convex_hull
        except Exception:
            hull = mp.convex_hull

        if hull.area < 5.0:
            # retry with ratio=0.5
            try:
                hull = mp.concave_hull(ratio=0.5)
            except Exception:
                pass
        if hull.area < 5.0:
            hull = mp.convex_hull

        polygon = hull
        polygon_area = float(hull.area)
        logger.info("Polygon area: %.2f m²", polygon_area)

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
                    },
                }
            ],
        }
        geojson_path = out_polygon / "floor_polygon.geojson"
        geojson_path.write_text(json.dumps(geojson, indent=2))
        logger.info("Saved: %s", geojson_path)

        # Polygon PNG
        fig, ax = plt.subplots(figsize=(10, 8))
        if hasattr(hull, "exterior"):
            xs, ys = hull.exterior.xy
            ax.fill(xs, ys, alpha=0.3, fc="royalblue", ec="navy", lw=1.5, label="floor polygon")
        elif hull.geom_type == "MultiPolygon":
            for geom in hull.geoms:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=0.3, fc="royalblue", ec="navy", lw=1.5)
        # Overlay poses
        for uid in unique_maps:
            mask = map_arr == uid
            ax.scatter(tx_arr[mask], ty_arr[mask], s=8,
                       c=color_map[uid], label=f"session {uid}", alpha=0.7, zorder=3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Floor Polygon — area={polygon_area:.1f} m²")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        polygon_png_path = out_polygon / "floor_polygon.png"
        fig.savefig(str(polygon_png_path), dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", polygon_png_path)
    else:
        logger.warning("Not enough points for polygon (%d)", len(pts_xy))

    metrics["polygon_area_m2"] = polygon_area

    return metrics


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sprint 81 Step 4: heatmap + polygon extraction")
    parser.add_argument("merged_db", type=Path)
    parser.add_argument("out_viz", type=Path)
    parser.add_argument("out_polygon", type=Path)
    parser.add_argument("--src-dbs", nargs="+", type=Path, default=None,
                        help="Source scan DBs (scan_A.db scan_B.db) for pose lookup")
    parser.add_argument("--metrics-json", type=Path, default=None)
    args = parser.parse_args()

    metrics = extract(args.merged_db, args.out_viz, args.out_polygon,
                      src_dbs=args.src_dbs)

    metrics_path = args.metrics_json or args.out_viz.parent / "merged_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics: {metrics}")
    print(f"Saved to {metrics_path}")


if __name__ == "__main__":
    main()
