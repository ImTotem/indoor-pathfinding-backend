from __future__ import annotations

import asyncio
import argparse
import json
import math
import os
import sqlite3
import struct
from pathlib import Path

import asyncpg
import cv2
import numpy as np


SCAN_ID = "9fce079b-17e2-4fa4-b7a1-05a9592917ff"
DB_PATH = Path("/app/var/storage/scans/9FCE079B-17E2-4FA4-B7A1-05A9592917FF/rtabmap_reprocessed.db")
OUT_DIR = Path("/app/var/debug/pointcloud_graph_1f")
HTML_PATH = OUT_DIR / "pointcloud_graph_overlay.html"
PLY_PATH = OUT_DIR / "pointcloud_sample.ply"
META_PATH = OUT_DIR / "pointcloud_graph_overlay.meta.json"
PNG_PATH = OUT_DIR / "pointcloud_graph_overlay.png"

MAX_POINTS = 180_000
PIXEL_STRIDE = 10
MIN_DEPTH_M = 0.25
MAX_DEPTH_M = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render RTABMap depth point cloud with route graph and optional snap overlay."
    )
    parser.add_argument("--scan-id", default=SCAN_ID)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--pose",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional localize pose to project to nearest graph edge.",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=None,
        help="Optional localize yaw in degrees. Adds camera-heading-relative side labels.",
    )
    parser.add_argument(
        "--compare-pose",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional second pose to overlay for comparison, e.g. RTAB-style relocalize.",
    )
    parser.add_argument(
        "--compare-yaw-deg",
        type=float,
        default=None,
        help="Optional yaw in degrees for --compare-pose.",
    )
    parser.add_argument(
        "--compare-label",
        default="RTAB rel",
        help="Label for --compare-pose.",
    )
    parser.add_argument(
        "--foot-z-mode",
        choices=("graph", "pose"),
        default="graph",
        help="graph: project to graph edge including interpolated graph z. pose: keep V_foot.z equal to P.z and use XY projection.",
    )
    parser.add_argument("--max-points", type=int, default=MAX_POINTS)
    parser.add_argument("--pixel-stride", type=int, default=PIXEL_STRIDE)
    return parser.parse_args()


def parse_node_transforms(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for node_id, blob in conn.execute("SELECT id, pose FROM Node WHERE pose IS NOT NULL ORDER BY id"):
        if not blob or len(blob) != 48:
            continue
        vals = struct.unpack("<12f", blob)
        if all(v == 0.0 for v in vals):
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
        result[int(node_id)] = pose
    return result


def parse_calibration(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(blob) < 116:
        raise ValueError(f"calibration blob too short: {len(blob)}")
    k_vals = struct.unpack("<9d", blob[44:116])
    k = np.asarray(k_vals, dtype=np.float64).reshape(3, 3)
    if len(blob) >= 164:
        local_vals = struct.unpack("<12f", blob[116:164])
        local_3x4 = np.asarray(local_vals, dtype=np.float64).reshape(3, 4)
    else:
        local_3x4 = np.zeros((3, 4), dtype=np.float64)
        local_3x4[:3, :3] = np.eye(3)
    local = np.eye(4, dtype=np.float64)
    local[:3, :] = local_3x4
    return k, local


def decode_depth(blob: bytes) -> np.ndarray | None:
    arr = np.frombuffer(bytes(blob), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    return img.view(np.float32).reshape(img.shape[0], img.shape[1])


def decode_rgb(blob: bytes) -> np.ndarray | None:
    arr = np.frombuffer(bytes(blob), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


async def load_graph() -> tuple[list[dict], list[dict]]:
    dsn = os.environ.get("DATABASE_URL") or "postgresql://indoor:indoor@db:5432/indoor"
    dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    conn = await asyncpg.connect(dsn)
    try:
        node_rows = await conn.fetch(
            """
            SELECT node_id::text AS id, node_type::text AS type, COALESCE(label, '') AS label,
                   ST_X(geom)::float8 AS x, ST_Y(geom)::float8 AS y, ST_Z(geom)::float8 AS z
            FROM map_node
            WHERE scan_id = $1::uuid AND is_stale = false
            """,
            SCAN_ID,
        )
        edge_rows = await conn.fetch(
            """
            SELECT edge_id::text AS id, from_node_id::text AS from_id,
                   to_node_id::text AS to_id, edge_type::text AS type,
                   length_m::float8 AS length
            FROM map_edge
            WHERE scan_id = $1::uuid AND is_stale = false
            """,
            SCAN_ID,
        )
    finally:
        await conn.close()

    nodes_by_id = {
        row["id"]: {
            "id": row["id"],
            "type": row["type"],
            "label": row["label"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
        }
        for row in node_rows
    }
    edges = []
    for row in edge_rows:
        a = nodes_by_id.get(row["from_id"])
        b = nodes_by_id.get(row["to_id"])
        if a is None or b is None:
            continue
        edges.append(
            {
                "id": row["id"],
                "type": row["type"],
                "length": float(row["length"]),
                "from": row["from_id"],
                "to": row["to_id"],
                "a": [a["x"], a["y"], a["z"]],
                "b": [b["x"], b["y"], b["z"]],
            }
        )
    return list(nodes_by_id.values()), edges


def compute_snap_overlay(
    nodes: list[dict],
    edges: list[dict],
    pose_xyz: tuple[float, float, float] | None,
    yaw_deg: float | None = None,
    foot_z_mode: str = "graph",
) -> dict | None:
    if pose_xyz is None:
        return None
    px, py, pz = pose_xyz
    nodes_by_id = {node["id"]: node for node in nodes}
    backbone_types = {
        "corridor",
        "junction",
        "endpoint",
        "passage_stairs",
        "passage_elevator",
        "passage_escalator",
    }
    best: tuple[float, dict, dict, dict, float, tuple[float, float, float]] | None = None
    for edge in edges:
        u = nodes_by_id.get(edge["from"])
        v = nodes_by_id.get(edge["to"])
        if u is None or v is None:
            continue
        if u["type"] not in backbone_types or v["type"] not in backbone_types:
            continue
        dx = float(v["x"]) - float(u["x"])
        dy = float(v["y"]) - float(u["y"])
        dz = float(v["z"]) - float(u["z"])
        use_pose_z = foot_z_mode == "pose"
        seg_len2 = dx * dx + dy * dy if use_pose_z else dx * dx + dy * dy + dz * dz
        if seg_len2 < 1e-9:
            continue
        if use_pose_z:
            t = ((px - float(u["x"])) * dx + (py - float(u["y"])) * dy) / seg_len2
        else:
            t = ((px - float(u["x"])) * dx + (py - float(u["y"])) * dy + (pz - float(u["z"])) * dz) / seg_len2
        t = max(0.0, min(1.0, t))
        foot = (
            float(u["x"]) + t * dx,
            float(u["y"]) + t * dy,
            pz if use_pose_z else float(u["z"]) + t * dz,
        )
        dist = float(np.linalg.norm(np.asarray(pose_xyz) - np.asarray(foot)))
        if best is None or dist < best[0]:
            best = (dist, edge, u, v, t, foot)

    if best is None:
        return None

    dist, edge, u, v, t, foot = best
    cross_z = (float(v["x"]) - float(u["x"])) * (py - float(u["y"])) - (
        float(v["y"]) - float(u["y"])
    ) * (px - float(u["x"]))
    if abs(cross_z) < 1e-9:
        edge_side = "on_edge"
    else:
        edge_side = "left" if cross_z > 0 else "right"
    camera_side = None
    camera_cross_z = None
    if yaw_deg is not None:
        yaw_rad = math.radians(float(yaw_deg))
        heading_x = math.cos(yaw_rad)
        heading_y = math.sin(yaw_rad)
        camera_cross_z = heading_x * (py - foot[1]) - heading_y * (px - foot[0])
        if abs(camera_cross_z) < 1e-9:
            camera_side = "on_heading"
        else:
            camera_side = "left" if camera_cross_z > 0 else "right"
    xy_d = float(math.hypot(px - foot[0], py - foot[1]))
    z_d = float(abs(pz - foot[2]))
    return {
        "pose": [px, py, pz],
        "foot": [foot[0], foot[1], foot[2]],
        "edge": {
            "id": edge["id"],
            "from": edge["from"],
            "to": edge["to"],
            "u": [float(u["x"]), float(u["y"]), float(u["z"])],
            "v": [float(v["x"]), float(v["y"]), float(v["z"])],
        },
        "edgeSide": edge_side,
        "crossZ": float(cross_z),
        "yawDeg": None if yaw_deg is None else float(yaw_deg),
        "cameraHeadingSide": camera_side,
        "cameraHeadingCrossZ": None if camera_cross_z is None else float(camera_cross_z),
        "footZMode": foot_z_mode,
        "xyDistanceM": xy_d,
        "zDistanceM": z_d,
        "distance3dM": float(dist),
        "footT": float(t),
    }


def build_cloud() -> tuple[np.ndarray, np.ndarray, dict]:
    conn = sqlite3.connect(str(DB_PATH))
    transforms = parse_node_transforms(conn)
    node_ids = [
        int(row[0])
        for row in conn.execute("SELECT id FROM Data ORDER BY id").fetchall()
        if int(row[0]) in transforms
    ]

    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    decoded_count = 0
    raw_point_count = 0

    for node_id in node_ids:
        row = conn.execute(
            "SELECT image, depth, calibration FROM Data WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row or not row[1] or not row[2]:
            continue
        rgb = decode_rgb(row[0]) if row[0] else None
        depth = decode_depth(row[1])
        if depth is None:
            continue

        k, local = parse_calibration(bytes(row[2]))
        pose = transforms[node_id]
        height, width = depth.shape

        src_w = max(k[0, 2] * 2.0, 1.0)
        src_h = max(k[1, 2] * 2.0, 1.0)
        sx = width / src_w
        sy = height / src_h
        fx = k[0, 0] * sx
        fy = k[1, 1] * sy
        cx = k[0, 2] * sx
        cy = k[1, 2] * sy

        yy, xx = np.meshgrid(
            np.arange(0, height, PIXEL_STRIDE),
            np.arange(0, width, PIXEL_STRIDE),
            indexing="ij",
        )
        z = depth[yy, xx].astype(np.float64)
        valid = np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= MAX_DEPTH_M)
        if not np.any(valid):
            continue

        z_valid = z[valid]
        x_cam = (xx[valid].astype(np.float64) - cx) * z_valid / fx
        y_cam = (yy[valid].astype(np.float64) - cy) * z_valid / fy
        cam = np.column_stack([x_cam, y_cam, z_valid, np.ones(len(z_valid), dtype=np.float64)])
        local_points = (local @ cam.T).T
        world = (pose @ local_points.T).T[:, :3]
        finite = np.all(np.isfinite(world), axis=1)
        world = world[finite]
        if len(world) == 0:
            continue

        if rgb is not None:
            rgb_h, rgb_w = rgb.shape[:2]
            rx = np.clip((xx[valid][finite] * (rgb_w / width)).astype(np.int32), 0, rgb_w - 1)
            ry = np.clip((yy[valid][finite] * (rgb_h / height)).astype(np.int32), 0, rgb_h - 1)
            colors = rgb[ry, rx]
        else:
            colors = np.repeat(np.array([[150, 150, 150]], dtype=np.uint8), len(world), axis=0)

        point_chunks.append(world.astype(np.float32))
        color_chunks.append(colors.astype(np.uint8))
        decoded_count += 1
        raw_point_count += len(world)

    conn.close()
    if not point_chunks:
        raise RuntimeError("no depth points decoded")

    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)

    qlo = np.quantile(points, 0.002, axis=0)
    qhi = np.quantile(points, 0.998, axis=0)
    keep = np.all((points >= qlo) & (points <= qhi), axis=1)
    points = points[keep]
    colors = colors[keep]

    if len(points) > MAX_POINTS:
        rng = np.random.default_rng(7)
        selected = rng.choice(len(points), MAX_POINTS, replace=False)
        points = points[selected]
        colors = colors[selected]

    meta = {
        "scan_id": SCAN_ID,
        "db_path": str(DB_PATH),
        "decoded_frames": decoded_count,
        "raw_points_before_crop": int(raw_point_count),
        "points": int(len(points)),
        "pixel_stride": PIXEL_STRIDE,
        "min_depth_m": MIN_DEPTH_M,
        "max_depth_m": MAX_DEPTH_M,
    }
    return points, colors, meta


def write_ply(points: np.ndarray, colors: np.ndarray) -> None:
    with PLY_PATH.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_html(payload: dict) -> None:
    data_json = json.dumps(payload, separators=(",", ":"))
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>1F point cloud + graph overlay</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #111; color: #eee; }
    #bar { position: fixed; left: 0; right: 0; top: 0; padding: 10px 14px; background: rgba(0,0,0,.78); z-index: 2; font-size: 13px; }
    #canvas { display: block; width: 100vw; height: 100vh; }
    button, label { margin-right: 12px; }
    input[type=range] { width: 140px; vertical-align: middle; }
    .legend { color: #ddd; margin-left: 10px; }
  </style>
</head>
<body>
<div id="bar">
  <b>1F Point Cloud + Route Graph</b>
  <button id="fit">Fit</button>
  <label><input id="pc" type="checkbox" checked> point cloud</label>
  <label><input id="graph" type="checkbox" checked> graph</label>
  <label><input id="nodes" type="checkbox" checked> nodes</label>
  <label>point size <input id="ps" type="range" min="1" max="5" value="2"></label>
  <span class="legend">drag: pan, wheel: zoom. Top-down XY.</span>
  <div id="info"></div>
</div>
<canvas id="canvas"></canvas>
<script>
const DATA = __DATA__;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let W = 0, H = 0, scale = 1, ox = 0, oy = 0, dragging = false, lx = 0, ly = 0;

function resize() {
  W = canvas.width = innerWidth * devicePixelRatio;
  H = canvas.height = innerHeight * devicePixelRatio;
  draw();
}
addEventListener("resize", resize);

function fit() {
  const b = DATA.bounds;
  const sx = (W - 120) / (b.maxX - b.minX || 1);
  const sy = (H - 170) / (b.maxY - b.minY || 1);
  scale = Math.min(sx, sy);
  ox = 60 - b.minX * scale;
  oy = 110 + b.maxY * scale;
  draw();
}

function T(x, y) { return [ox + x * scale, oy - y * scale]; }

function draw() {
  if (!W) return;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, W, H);
  ctx.lineCap = "round";

  if (document.getElementById("pc").checked) {
    const ps = Number(document.getElementById("ps").value) * devicePixelRatio;
    for (let i = 0; i < DATA.points.length; i++) {
      const p = DATA.points[i], q = T(p[0], p[1]), c = DATA.colors[i];
      ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      ctx.fillRect(q[0], q[1], ps, ps);
    }
  }

  if (document.getElementById("graph").checked) {
    ctx.strokeStyle = "#ffcc33";
    ctx.lineWidth = 3 * devicePixelRatio;
    for (const e of DATA.edges) {
      const a = T(e.a[0], e.a[1]), b = T(e.b[0], e.b[1]);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
  }

  if (DATA.snap) {
    const p = T(DATA.snap.pose[0], DATA.snap.pose[1]);
    const f = T(DATA.snap.foot[0], DATA.snap.foot[1]);
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = 4 * devicePixelRatio;
    ctx.setLineDash([10 * devicePixelRatio, 6 * devicePixelRatio]);
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.lineTo(f[0], f[1]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = "#00e5ff";
    ctx.fillStyle = "#111";
    ctx.lineWidth = 3 * devicePixelRatio;
    ctx.beginPath();
    ctx.arc(f[0], f[1], 9 * devicePixelRatio, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = "#ff00ff";
    ctx.lineWidth = 5 * devicePixelRatio;
    ctx.beginPath();
    ctx.moveTo(p[0] - 11 * devicePixelRatio, p[1] - 11 * devicePixelRatio);
    ctx.lineTo(p[0] + 11 * devicePixelRatio, p[1] + 11 * devicePixelRatio);
    ctx.moveTo(p[0] - 11 * devicePixelRatio, p[1] + 11 * devicePixelRatio);
    ctx.lineTo(p[0] + 11 * devicePixelRatio, p[1] - 11 * devicePixelRatio);
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = (14 * devicePixelRatio) + "px Arial";
    ctx.fillText("P localize", p[0] + 14 * devicePixelRatio, p[1] - 12 * devicePixelRatio);
    ctx.fillText("V_foot", f[0] + 14 * devicePixelRatio, f[1] + 5 * devicePixelRatio);
  }

  if (document.getElementById("nodes").checked) {
    for (const n of DATA.nodes) {
      const q = T(n.x, n.y);
      ctx.fillStyle = n.type.includes("passage") ? "#49d17d" : "#ff4d4d";
      ctx.beginPath();
      ctx.arc(q[0], q[1], 4 * devicePixelRatio, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  document.getElementById("info").textContent =
    "points=" + DATA.points.length.toLocaleString() +
    " frames=" + DATA.meta.decoded_frames +
    " graph_edges=" + DATA.edges.length +
    " graph_nodes=" + DATA.nodes.length +
    " scale=" + scale.toFixed(1) + "px/m" +
    (DATA.snap ? " snap=" + DATA.snap.edgeSide +
      " xy=" + DATA.snap.xyDistanceM.toFixed(3) + "m z=" + DATA.snap.zDistanceM.toFixed(3) + "m" : "");
}

canvas.addEventListener("mousedown", e => {
  dragging = true;
  lx = e.clientX * devicePixelRatio;
  ly = e.clientY * devicePixelRatio;
});
addEventListener("mouseup", () => dragging = false);
addEventListener("mousemove", e => {
  if (!dragging) return;
  const x = e.clientX * devicePixelRatio, y = e.clientY * devicePixelRatio;
  ox += x - lx;
  oy += y - ly;
  lx = x;
  ly = y;
  draw();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const mx = e.clientX * devicePixelRatio, my = e.clientY * devicePixelRatio;
  const old = scale;
  scale *= Math.exp(-e.deltaY * 0.001);
  ox = mx - (mx - ox) * (scale / old);
  oy = my - (my - oy) * (scale / old);
  draw();
}, { passive: false });
for (const id of ["pc", "graph", "nodes", "ps"]) {
  document.getElementById(id).addEventListener("input", draw);
}
document.getElementById("fit").onclick = fit;
resize();
fit();
</script>
</body>
</html>
"""
    HTML_PATH.write_text(html.replace("__DATA__", data_json), encoding="utf-8")


def write_png(payload: dict) -> None:
    width, height = 1800, 1100
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    bounds = payload["bounds"]
    sx = (width - 140) / max(bounds["maxX"] - bounds["minX"], 1e-6)
    sy = (height - 190) / max(bounds["maxY"] - bounds["minY"], 1e-6)
    scale = min(sx, sy)
    ox = 70 - bounds["minX"] * scale
    oy = 130 + bounds["maxY"] * scale

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(round(ox + x * scale)), int(round(oy - y * scale))

    # Draw point cloud.
    for point, color in zip(payload["points"], payload["colors"], strict=False):
        x, y = to_px(float(point[0]), float(point[1]))
        if 0 <= x < width and 0 <= y < height:
            img[y, x] = (int(color[2]), int(color[1]), int(color[0]))

    # Draw graph edges and nodes over the cloud.
    for edge in payload["edges"]:
        a = to_px(float(edge["a"][0]), float(edge["a"][1]))
        b = to_px(float(edge["b"][0]), float(edge["b"][1]))
        cv2.line(img, a, b, (51, 204, 255), 4, cv2.LINE_AA)
    for node in payload["nodes"]:
        p = to_px(float(node["x"]), float(node["y"]))
        color = (125, 209, 73) if "passage" in str(node["type"]) else (77, 77, 255)
        cv2.circle(img, p, 6, color, -1, cv2.LINE_AA)

    snap = payload.get("snap")
    if snap:
        p = to_px(float(snap["pose"][0]), float(snap["pose"][1]))
        f = to_px(float(snap["foot"][0]), float(snap["foot"][1]))
        cv2.line(img, p, f, (255, 229, 0), 4, cv2.LINE_AA)
        cv2.circle(img, f, 12, (255, 229, 0), 3, cv2.LINE_AA)
        cv2.line(img, (p[0] - 14, p[1] - 14), (p[0] + 14, p[1] + 14), (255, 0, 255), 5, cv2.LINE_AA)
        cv2.line(img, (p[0] - 14, p[1] + 14), (p[0] + 14, p[1] - 14), (255, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(img, "P localize", (p[0] + 16, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "V_foot", (f[0] + 16, f[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    compare_snap = payload.get("compareSnap")
    if compare_snap:
        p = to_px(float(compare_snap["pose"][0]), float(compare_snap["pose"][1]))
        f = to_px(float(compare_snap["foot"][0]), float(compare_snap["foot"][1]))
        label = str(compare_snap.get("label") or "RTAB rel")
        cv2.line(img, p, f, (80, 255, 80), 4, cv2.LINE_AA)
        cv2.circle(img, f, 10, (80, 255, 80), 3, cv2.LINE_AA)
        cv2.circle(img, p, 14, (80, 255, 80), 4, cv2.LINE_AA)
        cv2.putText(img, label, (p[0] + 16, p[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    text_lines = [
        "1F point cloud + route graph overlay",
        f"points={payload['meta']['points']:,} frames={payload['meta']['decoded_frames']} "
        f"nodes={len(payload['nodes'])} edges={len(payload['edges'])}",
        "yellow/cyan = route graph, red = route nodes, RGB dots = RTABMap depth point cloud",
    ]
    if snap:
        text_lines.append(
            f"snap side={snap['edgeSide']} xy={snap['xyDistanceM']:.3f}m "
            f"z={snap['zDistanceM']:.3f}m d3={snap['distance3dM']:.3f}m "
            f"edge={snap['edge']['from'][:8]}->{snap['edge']['to'][:8]}"
        )
    if compare_snap:
        text_lines.append(
            f"{compare_snap.get('label', 'RTAB rel')} side={compare_snap['edgeSide']} "
            f"xy={compare_snap['xyDistanceM']:.3f}m z={compare_snap['zDistanceM']:.3f}m "
            f"d3={compare_snap['distance3dM']:.3f}m"
        )
    cv2.rectangle(img, (20, 20), (width - 20, 55 + 25 * len(text_lines)), (255, 255, 255), -1)
    for idx, line in enumerate(text_lines):
        cv2.putText(
            img,
            line,
            (35, 48 + idx * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(PNG_PATH), img)


def write_snap_zoom_png(payload: dict) -> Path | None:
    snap = payload.get("snap")
    if not snap:
        return None
    zoom_path = OUT_DIR / "pointcloud_graph_overlay_snap_zoom.png"
    width, height = 1200, 900
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    px, py, _ = [float(v) for v in snap["pose"]]
    fx, fy, _ = [float(v) for v in snap["foot"]]
    compare_snap = payload.get("compareSnap")
    radius_m = 2.2
    focus_x = [px]
    focus_y = [py]
    if compare_snap:
        focus_x.append(float(compare_snap["pose"][0]))
        focus_y.append(float(compare_snap["pose"][1]))
    cx = sum(focus_x) / len(focus_x)
    cy = sum(focus_y) / len(focus_y)
    min_x, max_x = cx - radius_m, cx + radius_m
    min_y, max_y = cy - radius_m, cy + radius_m
    scale = min((width - 120) / (max_x - min_x), (height - 170) / (max_y - min_y))
    ox = 60 - min_x * scale
    oy = 120 + max_y * scale

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(round(ox + x * scale)), int(round(oy - y * scale))

    points = payload["points"]
    colors = payload["colors"]
    for point, color in zip(points, colors, strict=False):
        x, y = float(point[0]), float(point[1])
        if x < min_x or x > max_x or y < min_y or y > max_y:
            continue
        ix, iy = to_px(x, y)
        if 0 <= ix < width and 0 <= iy < height:
            img[iy, ix] = (int(color[2]), int(color[1]), int(color[0]))

    for edge in payload["edges"]:
        a = to_px(float(edge["a"][0]), float(edge["a"][1]))
        b = to_px(float(edge["b"][0]), float(edge["b"][1]))
        cv2.line(img, a, b, (51, 204, 255), 5, cv2.LINE_AA)

    # Selected edge direction and side labels.
    u = snap["edge"]["u"]
    v = snap["edge"]["v"]
    u_px = to_px(float(u[0]), float(u[1]))
    v_px = to_px(float(v[0]), float(v[1]))
    cv2.arrowedLine(img, u_px, v_px, (255, 180, 40), 3, cv2.LINE_AA, tipLength=0.05)
    cv2.putText(img, "u->v", (u_px[0] + 10, u_px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 40), 2, cv2.LINE_AA)

    p = to_px(px, py)
    f = to_px(fx, fy)
    cv2.line(img, p, f, (255, 229, 0), 4, cv2.LINE_AA)
    cv2.circle(img, f, 13, (255, 229, 0), 3, cv2.LINE_AA)
    cv2.line(img, (p[0] - 16, p[1] - 16), (p[0] + 16, p[1] + 16), (255, 0, 255), 6, cv2.LINE_AA)
    cv2.line(img, (p[0] - 16, p[1] + 16), (p[0] + 16, p[1] - 16), (255, 0, 255), 6, cv2.LINE_AA)
    cv2.putText(img, "P localize", (p[0] + 18, p[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "V_foot", (f[0] + 18, f[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if compare_snap:
        cpx, cpy, _ = [float(v) for v in compare_snap["pose"]]
        cfx, cfy, _ = [float(v) for v in compare_snap["foot"]]
        cp = to_px(cpx, cpy)
        cf = to_px(cfx, cfy)
        label = str(compare_snap.get("label") or "RTAB rel")
        cv2.line(img, cp, cf, (80, 255, 80), 4, cv2.LINE_AA)
        cv2.circle(img, cf, 11, (80, 255, 80), 3, cv2.LINE_AA)
        cv2.circle(img, cp, 15, (80, 255, 80), 4, cv2.LINE_AA)
        cv2.putText(img, label, (cp[0] + 18, cp[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    lines = [
        "Snap zoom around localize pose",
        f"edge-side by u->v = {snap['edgeSide']}  reversed-direction side = {'right' if snap['edgeSide'] == 'left' else 'left'}",
        f"xy={snap['xyDistanceM']:.3f}m z={snap['zDistanceM']:.3f}m d3={snap['distance3dM']:.3f}m",
    ]
    if snap.get("cameraHeadingSide") is not None:
        lines.insert(
            2,
            f"camera-yaw side = {snap['cameraHeadingSide']}  yaw={snap['yawDeg']:.1f} deg",
        )
    if compare_snap:
        lines.append(
            f"{compare_snap.get('label', 'RTAB rel')} side={compare_snap['edgeSide']} "
            f"xy={compare_snap['xyDistanceM']:.3f}m z={compare_snap['zDistanceM']:.3f}m"
        )
    lines.append(f"V_foot.z mode = {snap.get('footZMode', 'graph')}")
    cv2.rectangle(img, (20, 20), (width - 20, 55 + len(lines) * 25), (255, 255, 255), -1)
    for idx, line in enumerate(lines):
        cv2.putText(img, line, (35, 48 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(zoom_path), img)
    return zoom_path


async def main() -> None:
    global SCAN_ID, DB_PATH, OUT_DIR, HTML_PATH, PLY_PATH, META_PATH, PNG_PATH
    global MAX_POINTS, PIXEL_STRIDE

    args = parse_args()
    SCAN_ID = args.scan_id
    DB_PATH = args.db_path
    OUT_DIR = args.out_dir
    HTML_PATH = OUT_DIR / "pointcloud_graph_overlay.html"
    PLY_PATH = OUT_DIR / "pointcloud_sample.ply"
    META_PATH = OUT_DIR / "pointcloud_graph_overlay.meta.json"
    PNG_PATH = OUT_DIR / "pointcloud_graph_overlay.png"
    MAX_POINTS = args.max_points
    PIXEL_STRIDE = args.pixel_stride

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points, colors, meta = build_cloud()
    nodes, edges = await load_graph()
    snap = compute_snap_overlay(
        nodes,
        edges,
        tuple(args.pose) if args.pose is not None else None,
        args.yaw_deg,
        args.foot_z_mode,
    )
    compare_snap = compute_snap_overlay(
        nodes,
        edges,
        tuple(args.compare_pose) if args.compare_pose is not None else None,
        args.compare_yaw_deg,
        args.foot_z_mode,
    )
    if compare_snap is not None:
        compare_snap["label"] = args.compare_label
    write_ply(points, colors)

    extra_x: list[float] = []
    extra_y: list[float] = []
    if snap is not None:
        extra_x.extend([float(snap["pose"][0]), float(snap["foot"][0])])
        extra_y.extend([float(snap["pose"][1]), float(snap["foot"][1])])
    if compare_snap is not None:
        extra_x.extend([float(compare_snap["pose"][0]), float(compare_snap["foot"][0])])
        extra_y.extend([float(compare_snap["pose"][1]), float(compare_snap["foot"][1])])
    all_x = np.concatenate(
        [
            points[:, 0],
            np.asarray([n["x"] for n in nodes], dtype=np.float32),
            np.asarray(extra_x, dtype=np.float32),
        ]
    )
    all_y = np.concatenate(
        [
            points[:, 1],
            np.asarray([n["y"] for n in nodes], dtype=np.float32),
            np.asarray(extra_y, dtype=np.float32),
        ]
    )
    payload = {
        "meta": meta,
        "points": np.round(points[:, :3], 3).tolist(),
        "colors": colors.tolist(),
        "nodes": nodes,
        "edges": edges,
        "snap": snap,
        "compareSnap": compare_snap,
        "bounds": {
            "minX": float(all_x.min()),
            "maxX": float(all_x.max()),
            "minY": float(all_y.min()),
            "maxY": float(all_y.max()),
        },
    }
    META_PATH.write_text(
        json.dumps({k: v for k, v in payload.items() if k not in {"points", "colors"}}, indent=2),
        encoding="utf-8",
    )
    write_html(payload)
    write_png(payload)
    zoom_path = write_snap_zoom_png(payload)
    print(
        json.dumps(
            {
                "html": str(HTML_PATH),
                "ply": str(PLY_PATH),
                "png": str(PNG_PATH),
                "zoom_png": str(zoom_path) if zoom_path is not None else None,
                "meta": str(META_PATH),
                "cloud": meta,
                "snap": snap,
                "nodes": len(nodes),
                "edges": len(edges),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
