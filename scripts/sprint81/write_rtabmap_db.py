"""Sprint 81 — Step 2: RGB-only RTAB-Map DB writer.

keyframe JPEGs + index.json → RGB-only sqlite3 RTAB-Map DB

Calibration blob format (164B) — reverse-engineered from RTAB-Map 0.23.5 iOS DB:
  [0..43]   11x int32 header:
              [0]=0, [1]=23(version?), [2]=5, [3]=0,
              [4]=width(int), [5]=height(int), [6]=9, [7..9]=0, [10]=12
  [44..115]  9x float64 (intrinsic matrix K row-major 3x3):
              K[0,0]=fx  K[0,1]=0    K[0,2]=cx
              K[1,0]=0   K[1,1]=fy   K[1,2]=cy
              K[2,0]=0   K[2,1]=0    K[2,2]=1.0
  [116..163] 12x float32 local_transform (3x4 row-major):
              fixed value from iOS scan DB:
              [[0, 0, 1, 0],[-1, 0, 0, 0],[0, -1, 0, 0]]

Node.pose blob (48B): 3x4 float32 row-major, ARKit world coordinates
  (with AXIS_SWAP applied: ARKit Y-up → RTAB-Map Z-up)
  _AXIS_SWAP: [[1,0,0],[0,0,-1],[0,1,0]] (R_x +90°)

Information matrix (288B): 6x6 float64, identity * low_weight (large covariance)

Link.transform (48B): 3x4 float32, relative transform between nodes
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_SERVER_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(_SERVER_SRC))

logger = logging.getLogger(__name__)

# ARKit Y-up → RTAB-Map Z-up (R_x +90°): 기존 sprint77 _AXIS_SWAP_4 동일
_AXIS_SWAP_3 = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [0.0, 1.0, 0.0]], dtype=np.float64)
_AXIS_SWAP_4 = np.eye(4, dtype=np.float64)
_AXIS_SWAP_4[:3, :3] = _AXIS_SWAP_3

# RTAB-Map stored images at half-res: 960x540 (vs iOS 1920x1080)
# Intrinsics scale accordingly
_STORED_WIDTH = 960
_STORED_HEIGHT = 540
_INTRINSICS_SCALE = 0.5  # 960/1920 = 0.5

# local_transform fixed value (from iOS DB reverse-engineering)
_LOCAL_TRANSFORM_3x4 = np.array(
    [[0.0, 0.0, 1.0, 0.0],
     [-1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0]], dtype=np.float32
)

# Neighbor link information matrix: large uncertainty (small weight)
_INFO_MATRIX_LOW = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]).astype(np.float64)
# Pose scale factor: odometry-sourced neighbor links get slightly higher weight
_INFO_MATRIX_ODOM = np.diag([1e2, 1e2, 1e2, 1e4, 1e4, 1e2]).astype(np.float64)


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class WriteSummary:
    db_path: Path
    node_count: int
    link_count: int
    map_id: int


def _pack_calibration(intr: Intrinsics) -> bytes:
    """164B calibration blob 직렬화.

    Layout (확인된 iOS RTAB-Map 0.23.5):
      [0..43]   11x int32 header
      [44..115] 9x float64 = K matrix 3x3 row-major
      [116..163] 12x float32 = local_transform 3x4 row-major
    """
    # intrinsics를 저장 해상도(960x540)로 스케일
    # 입력 intrinsics가 full-res (1920x1080) 기준
    stored_w = _STORED_WIDTH
    stored_h = _STORED_HEIGHT
    scale = stored_w / intr.width  # 960/1920 = 0.5
    fx = intr.fx * scale
    fy = intr.fy * scale
    cx = intr.cx * scale
    cy = intr.cy * scale

    # 11x int32 header — fixed from iOS DB observation
    header = struct.pack('<11i', 0, 23, 5, 0, stored_w, stored_h, 9, 0, 0, 0, 12)
    # 9x float64 = 3x3 K matrix row-major
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    k_bytes = K.astype('<f8').tobytes()
    # 12x float32 = local_transform 3x4 row-major (fixed)
    lt_bytes = _LOCAL_TRANSFORM_3x4.astype('<f4').tobytes()

    blob = header + k_bytes + lt_bytes
    assert len(blob) == 164, f"calibration blob expected 164B got {len(blob)}B"
    return blob


def _pose_arkit_to_rtabmap(T_arkit_4x4: np.ndarray) -> bytes:
    """ARKit 4x4 pose → RTAB-Map 3x4 row-major float32 bytes (48B).

    ARKit Y-up → RTAB-Map Z-up via _AXIS_SWAP_4.
    """
    T = _AXIS_SWAP_4 @ T_arkit_4x4.astype(np.float64)
    T_3x4 = T[:3, :]  # (3, 4)
    return T_3x4.astype('<f4').tobytes()


def _relative_transform(T_i: np.ndarray, T_j: np.ndarray) -> bytes:
    """Node i → j 의 상대 transform 3x4 float32 bytes.

    T_i, T_j: 4x4 ARKit world poses (already axis-swapped).
    rel = inv(T_i) @ T_j
    """
    T_i_inv = np.linalg.inv(T_i)
    T_rel = T_i_inv @ T_j
    return T_rel[:3, :].astype('<f4').tobytes()


def _info_matrix_bytes(mat: np.ndarray) -> bytes:
    """6x6 float64 → 288B."""
    return mat.astype('<f8').tobytes()


def _create_schema(conn: sqlite3.Connection) -> None:
    """RTAB-Map 0.23.x SQLite schema 수동 생성."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS Node (
            id INTEGER NOT NULL,
            map_id INTEGER NOT NULL,
            weight INTEGER,
            stamp FLOAT,
            pose BLOB,
            ground_truth_pose BLOB,
            velocity BLOB,
            label TEXT,
            gps BLOB,
            env_sensors BLOB,
            time_enter DATE,
            PRIMARY KEY (id)
        );
        CREATE TABLE IF NOT EXISTS Data (
            id INTEGER NOT NULL,
            image BLOB,
            depth BLOB,
            depth_confidence BLOB,
            calibration BLOB,
            scan BLOB,
            scan_info BLOB,
            ground_cells BLOB,
            obstacle_cells BLOB,
            empty_cells BLOB,
            cell_size FLOAT,
            view_point_x FLOAT,
            view_point_y FLOAT,
            view_point_z FLOAT,
            user_data BLOB,
            time_enter DATE,
            PRIMARY KEY (id)
        );
        CREATE TABLE IF NOT EXISTS Link (
            from_id INTEGER NOT NULL,
            to_id INTEGER NOT NULL,
            type INTEGER NOT NULL,
            information_matrix BLOB NOT NULL,
            transform BLOB,
            user_data BLOB
        );
        CREATE TABLE IF NOT EXISTS Word (
            id INTEGER NOT NULL,
            descriptor_size INTEGER NOT NULL,
            descriptor BLOB NOT NULL,
            time_enter DATE,
            PRIMARY KEY (id)
        );
        CREATE TABLE IF NOT EXISTS Feature (
            node_id INTEGER NOT NULL,
            word_id INTEGER NOT NULL,
            pos_x FLOAT NOT NULL,
            pos_y FLOAT NOT NULL,
            size INTEGER NOT NULL,
            dir FLOAT NOT NULL,
            response FLOAT NOT NULL,
            octave INTEGER NOT NULL,
            depth_x FLOAT,
            depth_y FLOAT,
            depth_z FLOAT,
            descriptor_size INTEGER,
            descriptor BLOB
        );
        CREATE TABLE IF NOT EXISTS GlobalDescriptor (
            node_id INTEGER NOT NULL,
            type INTEGER,
            info BLOB,
            data BLOB
        );
        CREATE TABLE IF NOT EXISTS Info (
            STM_size INTEGER,
            last_sign_added INTEGER,
            process_mem_used INTEGER,
            database_mem_used INTEGER,
            dictionary_size INTEGER,
            parameters TEXT,
            time_enter DATE
        );
        CREATE TABLE IF NOT EXISTS Statistics (
            id INTEGER PRIMARY KEY,
            stamp FLOAT,
            data BLOB,
            wm_state BLOB,
            time_enter DATE
        );
        CREATE TABLE IF NOT EXISTS Admin (
            version TEXT,
            preview_image BLOB,
            opt_ids BLOB,
            opt_poses BLOB,
            opt_last_localization BLOB,
            opt_graph BLOB,
            opt_map BLOB,
            opt_poses_constraints BLOB,
            opt_last_link_added BLOB,
            opt_map_correction BLOB,
            time_enter DATE
        );
    """)
    conn.commit()


_RTABMAP_PARAMS = (
    "Mem/IncrementalMemory=true;"
    "Mem/UseOdomFeatures=false;"
    "Mem/STMSize=30;"
    "Mem/InitWMWithAllNodes=false;"
    "Reg/Strategy=0;"
    "Vis/EstimationType=1;"
    "Kp/DetectorStrategy=6;"
    "Rtabmap/DetectionRate=0;"
    "Rtabmap/LoopThr=0.11;"
    "RGBD/Enabled=true;"
)


def write_rgb_only_db(
    keyframes_json: Path,
    image_dir: Path,
    intrinsics: Intrinsics,
    out_db_path: Path,
    map_id: int = 0,
) -> WriteSummary:
    """RGB-only RTAB-Map DB 작성.

    Args:
        keyframes_json: index.json (KeyframeRecord list).
        image_dir: JPEG 파일들이 있는 디렉터리 (index.json 과 같은 위치).
        intrinsics: full-res (1920x1080) 기준 camera intrinsics.
        out_db_path: 출력 sqlite3 DB 경로.
        map_id: 세션 번호 (A=0, B=1 등).

    Returns:
        WriteSummary.
    """
    records = json.loads(keyframes_json.read_text())
    logger.info("Writing RGB-only DB: %d keyframes, map_id=%d → %s",
                len(records), map_id, out_db_path)

    out_db_path.parent.mkdir(parents=True, exist_ok=True)
    if out_db_path.exists():
        out_db_path.unlink()

    conn = sqlite3.connect(str(out_db_path))
    _create_schema(conn)

    calib_blob = _pack_calibration(intrinsics)
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Admin row — version 맞추기
    conn.execute(
        "INSERT INTO Admin (version, time_enter) VALUES (?, ?)",
        ("0.23.5", now_str),
    )

    # Info row — RTAB-Map parameters
    conn.execute(
        "INSERT INTO Info (STM_size, last_sign_added, process_mem_used, "
        "database_mem_used, dictionary_size, parameters, time_enter) "
        "VALUES (0,0,0,0,0,?,?)",
        (_RTABMAP_PARAMS, now_str),
    )

    # Node + Data rows
    # Collect axis-swapped 4x4 poses for relative transform calculation
    world_poses: list[np.ndarray] = []

    for rec in records:
        node_id = rec["node_id"]
        ts_s = rec["ts_s"]
        img_filename = rec["image_path"]

        # image blob: JPEG を 960x540 へ resize してから encode
        img_path = image_dir / img_filename
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img_small = cv2.resize(img_bgr, (_STORED_WIDTH, _STORED_HEIGHT),
                                interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError(f"imencode failed for {img_path}")
        image_blob = encoded.tobytes()

        # pose: ARKit 4x4 → RTAB-Map 3x4
        T_arkit = np.array(rec["pose_world_arkit"], dtype=np.float64)
        pose_blob = _pose_arkit_to_rtabmap(T_arkit)

        # axis-swapped world pose 보관 (Link 계산용)
        T_rtabmap = _AXIS_SWAP_4 @ T_arkit
        world_poses.append(T_rtabmap)

        conn.execute(
            "INSERT INTO Node (id, map_id, weight, stamp, pose, time_enter) "
            "VALUES (?,?,?,?,?,?)",
            (node_id, map_id, 1, ts_s, pose_blob, now_str),
        )
        conn.execute(
            "INSERT INTO Data (id, image, depth, depth_confidence, calibration, "
            "scan, scan_info, ground_cells, obstacle_cells, empty_cells, "
            "cell_size, view_point_x, view_point_y, view_point_z, time_enter) "
            "VALUES (?,?,NULL,NULL,?,NULL,NULL,NULL,NULL,NULL,NULL,0,0,0,?)",
            (node_id, image_blob, calib_blob, now_str),
        )

    # Sequential Neighbor Links (type=0)
    info_odom_bytes = _info_matrix_bytes(_INFO_MATRIX_ODOM)
    link_count = 0
    for i, rec in enumerate(records[:-1]):
        from_id = rec["node_id"]
        to_id = records[i + 1]["node_id"]
        T_i = world_poses[i]
        T_j = world_poses[i + 1]
        transform_blob = _relative_transform(T_i, T_j)
        conn.execute(
            "INSERT INTO Link (from_id, to_id, type, information_matrix, transform) "
            "VALUES (?,?,0,?,?)",
            (from_id, to_id, info_odom_bytes, transform_blob),
        )
        link_count += 1

    conn.commit()
    conn.close()

    logger.info("DB written: nodes=%d, neighbor_links=%d, path=%s",
                len(records), link_count, out_db_path)
    return WriteSummary(
        db_path=out_db_path,
        node_count=len(records),
        link_count=link_count,
        map_id=map_id,
    )


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sprint 81 Step 2: RGB-only RTAB-Map DB writer")
    parser.add_argument("keyframes_json", type=Path, help="index.json from step 1")
    parser.add_argument("image_dir", type=Path, help="directory containing JPEGs")
    parser.add_argument("out_db", type=Path, help="output .db path")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--map-id", type=int, default=0)
    args = parser.parse_args()

    intr = Intrinsics(
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        width=args.width, height=args.height,
    )
    summary = write_rgb_only_db(
        keyframes_json=args.keyframes_json,
        image_dir=args.image_dir,
        intrinsics=intr,
        out_db_path=args.out_db,
        map_id=args.map_id,
    )
    print(f"Written: nodes={summary.node_count}, links={summary.link_count} → {summary.db_path}")


if __name__ == "__main__":
    main()
