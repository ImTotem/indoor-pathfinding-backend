"""Sprint 20 pose convention 역공학.

확인 항목:
    (1) 저장된 tx/ty/tz 컬럼 값 = decode_pose_matrix(blob)[:3, 3] 같은가?
        → 같으면 column-major 읽기 컨벤션 OK (transform.columns.3 그대로).
    (2) pose[:3, :3] 직교성 + det = +1
    (3) Y-up → Z-up 변환 후 tz(카메라 높이)가 0.5~2m로 일관되게 양수인가?
    (4) 연속 keyframe간 translation 차가 iOS keyframe throttle 0.3m와 일치하나?
    (5) 카메라 forward 방향 (world frame)이 합리적인가?
        ARKit cam z축(backward)을 world로 보내면 대략 horizontal.

사용:
    uv run python scripts/diag_pose_convention.py <scan_dir> [N=10]
"""
from __future__ import annotations

import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_AXIS_SWAP_3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
_AXIS_SWAP_4 = np.eye(4, dtype=np.float64)
_AXIS_SWAP_4[:3, :3] = _AXIS_SWAP_3


def _decode(pose_bytes: bytes) -> np.ndarray:
    values = struct.unpack_from("<16f", pose_bytes)
    return np.array(values, dtype=np.float64).reshape(4, 4, order="F")


def _convert_yup_to_zup(pose_old: np.ndarray) -> np.ndarray:
    return _AXIS_SWAP_4 @ pose_old


def main() -> int:
    scan_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    conn = sqlite3.connect(str(scan_dir / "scan_metadata.db"))
    rows = conn.execute(
        "SELECT seq, pose_matrix, tx, ty, tz FROM keyframe_meta ORDER BY seq LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()

    print(f"loaded {len(rows)} keyframes\n")
    prev_pos_arkit: np.ndarray | None = None
    prev_pos_zup: np.ndarray | None = None

    for seq, pose_blob, tx, ty, tz in rows:
        p_arkit = _decode(bytes(pose_blob))
        p_zup = _convert_yup_to_zup(p_arkit)

        # (1) translation 열 vs 저장된 tx/ty/tz
        dec_t = p_arkit[:3, 3]
        diff_t = np.array([tx, ty, tz], dtype=np.float64) - dec_t
        print(f"seq={seq:3d}  stored(t)=({tx:+.3f},{ty:+.3f},{tz:+.3f})  "
              f"decoded(t)=({dec_t[0]:+.3f},{dec_t[1]:+.3f},{dec_t[2]:+.3f})  "
              f"diff={np.linalg.norm(diff_t):.6f}")

        # (2) rotation orthogonality
        R = p_arkit[:3, :3]
        orth_err = float(np.linalg.norm(R @ R.T - np.eye(3)))
        det_r = float(np.linalg.det(R))
        # (3) camera height after zup
        cam_height_zup = float(p_zup[2, 3])
        # (4) ARKit camera forward direction in world = -z_cam = -R_arkit[:3,2]
        cam_fwd_arkit_world = -p_arkit[:3, 2]
        # Z-up world에서는 S @ (-z_cam_arkit_world)
        cam_fwd_zup_world = _AXIS_SWAP_3 @ cam_fwd_arkit_world

        print(f"    R orth_err={orth_err:.4f} det={det_r:+.4f}  "
              f"cam_height(zup)={cam_height_zup:+.3f}m  "
              f"cam_fwd(arkit_world)=({cam_fwd_arkit_world[0]:+.2f},{cam_fwd_arkit_world[1]:+.2f},{cam_fwd_arkit_world[2]:+.2f})  "
              f"cam_fwd(zup)=({cam_fwd_zup_world[0]:+.2f},{cam_fwd_zup_world[1]:+.2f},{cam_fwd_zup_world[2]:+.2f})")

        # (5) inter-frame translation magnitudes
        pos_arkit = p_arkit[:3, 3]
        pos_zup = p_zup[:3, 3]
        if prev_pos_arkit is not None:
            d_arkit = np.linalg.norm(pos_arkit - prev_pos_arkit)
            d_zup = np.linalg.norm(pos_zup - prev_pos_zup)
            print(f"    Δ from prev: arkit={d_arkit:.3f}m  zup={d_zup:.3f}m  (keyframe throttle expects ~0.3m)")
        prev_pos_arkit = pos_arkit
        prev_pos_zup = pos_zup
        print()

    # (6) ARKit pose col3 값과 실제 의미 — world_T_camera이면 그게 camera position
    print("=" * 72)
    print("pose[:3, 3] 해석:")
    print("  - 만약 world_T_camera 이면 이 값 = 월드 프레임에서 카메라 위치")
    print("  - 만약 camera_T_world 이면 이 값 = -R @ camera_position_in_world")
    print("  Apple 문서: ARCamera.transform = world_T_camera")
    print("  ⇒ pose[:3, 3] = 카메라 월드 위치. cam_height(zup)가 0.5~1.8m 범위면 맞음.")
    print()
    print("cam_fwd(zup) 해석:")
    print("  - 카메라가 정면(수평)을 보면 z 성분이 ±0.2 이내이어야 함.")
    print("  - 아래 보면 z<0, 위 보면 z>0. 실내 스캔은 수평이므로 |z|<0.3 기대.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
