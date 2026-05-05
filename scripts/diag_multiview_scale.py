"""Sprint 20 진단: multi-view scale observation이 geometrically 맞는지 검증.

과정:
    1. Downloads의 실 scan sidecar를 읽어 처음 N keyframe 선택
    2. 각 frame에 대해 Depth Anything으로 depth_relative 산출
    3. 인접 pair마다 SP+LG 매칭 → matched kp
    4. 각 matched kp를 양쪽 frame에서 back-project (ScaleCalibrator initial_scale 사용)
    5. ||P_i - P_j|| 분포 출력 → 지금 코드의 residual이 작은지/큰지 확인
    6. 동시에 closed-form LSQ로 global scale 해 산출 → 상한/하한 없이 자연스러운 값 확인

사용:
    uv run python scripts/diag_multiview_scale.py <scan_dir> [N]
"""
from __future__ import annotations

import asyncio
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.application.building.steps.back_projection import (
    decode_pose_matrix,
    default_intrinsics,
)
from indoor_server.application.building.steps.depth_back_projection import ScaleCalibrator
from indoor_server.config import settings
from indoor_server.infrastructure.ml.depth_anything import DepthAnythingV2Runner
from indoor_server.infrastructure.ml.superpoint_lightglue import SuperPointLightGlueRunner


_AXIS_SWAP_3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
_AXIS_SWAP_4 = np.eye(4, dtype=np.float64)
_AXIS_SWAP_4[:3, :3] = _AXIS_SWAP_3


def _convert_pose_yup_to_zup(pose_bytes: bytes) -> bytes:
    values = struct.unpack_from("<16f", pose_bytes)
    pose_old = np.array(values, dtype=np.float64).reshape(4, 4, order="F")
    pose_new = _AXIS_SWAP_4 @ pose_old
    flat = pose_new.flatten(order="F").astype(np.float32)
    return struct.pack("<16f", *flat.tolist())


async def main() -> int:
    scan_dir = Path(sys.argv[1])
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    sidecar = scan_dir / "scan_metadata.db"
    keyframes_dir = scan_dir / "keyframes"

    conn = sqlite3.connect(str(sidecar))
    rows = conn.execute(
        "SELECT seq, image_path, pose_matrix, tx, ty, tz FROM keyframe_meta ORDER BY seq LIMIT ?",
        (n_frames,),
    ).fetchall()
    scan_id_row = conn.execute("SELECT id FROM scan_session LIMIT 1").fetchone()
    conn.close()

    print(f"scan_id={scan_id_row[0]}, loaded {len(rows)} keyframes")

    # Load models
    depth_runner = DepthAnythingV2Runner(model_path=settings.model_cache_dir / "depth_anything_v2_small.onnx")
    sp_lg = SuperPointLightGlueRunner(
        model_path=settings.model_cache_dir / "superpoint_lightglue.onnx",
        input_size=settings.superpoint_input_size,
    )
    calibrator = ScaleCalibrator()

    # Pre-compute for each frame
    import cv2

    frames: list[dict] = []
    for seq, image_path, pose_blob, tx, ty, tz in rows:
        img_file = keyframes_dir / Path(image_path).name
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  seq={seq} skip: image missing")
            continue
        h, w = img.shape[:2]
        pose_bytes_zup = _convert_pose_yup_to_zup(bytes(pose_blob))
        pose = decode_pose_matrix(pose_bytes_zup)
        intrin = default_intrinsics(w, h)

        # floor mask: bottom 25% of the rotated upright image, rotated back.
        # For diagnosis, we use a simple "bottom half of sensor frame" mask (real_upright's purpose).
        floor_mask = np.zeros((h, w), dtype=bool)
        floor_mask[:, : int(w * 0.25)] = True  # left 25% 가 upright 후 하단에 해당 (run_real_scan.py 로직)

        # depth
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth_rel = await depth_runner.estimate(rgb)

        # z0: all tz들 5%ile - 1.0
        # 여기선 간단히 mean tz - 1.0로 대체
        z0 = -0.88  # 실 run_real_scan 결과 floor_z0 고정

        scale = calibrator.calibrate(depth_rel, floor_mask, pose, intrin, z0)

        frames.append({
            "seq": seq,
            "img": img,
            "pose": pose,
            "intrin": intrin,
            "depth": depth_rel,
            "scale_init": scale,
            "z0": z0,
        })
        print(f"  seq={seq} size={w}x{h} scale_init={scale!r}")

    if len(frames) < 2:
        print("not enough frames")
        return 1

    # pair (0,1) 매칭
    fa, fb = frames[0], frames[1]
    mp = await sp_lg.match(fa["img"], fb["img"])
    print(f"\npair(0,1): matches={mp.matches.shape[0]} scores[0:5]={mp.scores[:5].tolist()}")

    if mp.matches.shape[0] == 0:
        print("no matches for pair(0,1)")
        return 1

    # 상위 20개 match로 ||P_i - P_j|| 확인
    img_h_a, img_w_a = fa["img"].shape[:2]
    img_h_b, img_w_b = fb["img"].shape[:2]
    sx_a = img_w_a / mp.size_a[1]
    sy_a = img_h_a / mp.size_a[0]
    sx_b = img_w_b / mp.size_b[1]
    sy_b = img_h_b / mp.size_b[0]

    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    top = np.argsort(-mp.scores)[:top_k]
    sel = mp.matches[top]

    initial_s = fa["scale_init"] or fb["scale_init"] or 2.0
    initial_s = float(initial_s)

    diff_coeffs = []
    offsets = []
    residuals_at_init = []

    for idx_a, idx_b in sel:
        xa = mp.kpts_a[idx_a, 0] * sx_a
        ya = mp.kpts_a[idx_a, 1] * sy_a
        xb = mp.kpts_b[idx_b, 0] * sx_b
        yb = mp.kpts_b[idx_b, 1] * sy_b
        d_rel_a = float(fa["depth"][int(round(ya)), int(round(xa))])
        d_rel_b = float(fb["depth"][int(round(yb)), int(round(xb))])
        if d_rel_a < 1e-4 or d_rel_b < 1e-4:
            continue

        intrin_a = fa["intrin"]
        intrin_b = fb["intrin"]
        ray_a = np.array([(xa - intrin_a.cx) / intrin_a.fx, (ya - intrin_a.cy) / intrin_a.fy, 1.0])
        ray_b = np.array([(xb - intrin_b.cx) / intrin_b.fx, (yb - intrin_b.cy) / intrin_b.fy, 1.0])
        ra = ray_a / np.linalg.norm(ray_a)
        rb = ray_b / np.linalg.norm(ray_b)

        coeff_a = fa["pose"][:3, :3] @ (ra / d_rel_a)
        coeff_b = fb["pose"][:3, :3] @ (rb / d_rel_b)
        ta = fa["pose"][:3, 3]
        tb = fb["pose"][:3, 3]

        P_a = ta + initial_s * coeff_a
        P_b = tb + initial_s * coeff_b

        print(
            f"  (idx_a={idx_a}, idx_b={idx_b}) kpA=({xa:.0f},{ya:.0f}) kpB=({xb:.0f},{yb:.0f}) "
            f"dA={d_rel_a:.3f} dB={d_rel_b:.3f}  "
            f"Pa=({P_a[0]:+.2f},{P_a[1]:+.2f},{P_a[2]:+.2f}) "
            f"Pb=({P_b[0]:+.2f},{P_b[1]:+.2f},{P_b[2]:+.2f})  "
            f"||Pa-Pb||={np.linalg.norm(P_a-P_b):.2f}m"
        )

        diff_coeffs.append(coeff_a - coeff_b)
        offsets.append(ta - tb)
        residuals_at_init.append(P_a - P_b)

    if not diff_coeffs:
        print("no valid observations")
        return 1

    diff = np.stack(diff_coeffs)           # (N, 3)
    off = np.stack(offsets)                 # (N, 3)
    r_init = np.stack(residuals_at_init)    # (N, 3)

    # closed-form global scale: s = -Σ (diff · offset) / Σ ||diff||^2
    #   since residual = diff * s + offset, minimizing ||.||^2 over s
    num = -float(np.sum(diff * off))         # Σ over N*3 scalars
    den = float(np.sum(diff * diff))
    s_star = num / den if den > 0 else float("nan")
    print(f"\nclosed-form global scale (L2 baseline): {s_star:.4f}")

    # 부호 컨벤션 테스트 — coeff 전체 부호 반전이 맞는 경우
    def _closed_form(diff_c, off_c):
        n = -float(np.sum(diff_c * off_c))
        d = float(np.sum(diff_c * diff_c))
        return n / d if d > 0 else float("nan")

    # 테스트1: ray의 z 부호 반전 (OpenCV-cam vs ARKit-cam -z forward)
    # coeff = R @ ([x,y,1] / ||.||) / d_rel  →  [x, y, -1] / ||.|| / d_rel
    # 즉 coeff의 y/z 부호 반전 후 R 적용 필요 — 단순 근사로 ray_cam_new = diag(1, -1, -1) @ ray_cam
    # 여기선 diag 단계에선 직접 계산해 보기 위해 재산출
    print("\n[test] applying diag(1,-1,-1) to ray_cam (ARKit y-up cam → OpenCV y-down cam):")
    diff_flip = []
    off_flip = []
    for idx_a, idx_b in sel:
        xa = mp.kpts_a[idx_a, 0] * sx_a
        ya = mp.kpts_a[idx_a, 1] * sy_a
        xb = mp.kpts_b[idx_b, 0] * sx_b
        yb = mp.kpts_b[idx_b, 1] * sy_b
        d_rel_a = float(fa["depth"][int(round(ya)), int(round(xa))])
        d_rel_b = float(fb["depth"][int(round(yb)), int(round(xb))])
        if d_rel_a < 1e-4 or d_rel_b < 1e-4:
            continue
        intrin_a = fa["intrin"]
        intrin_b = fb["intrin"]
        ray_a = np.array([(xa - intrin_a.cx) / intrin_a.fx, -(ya - intrin_a.cy) / intrin_a.fy, -1.0])
        ray_b = np.array([(xb - intrin_b.cx) / intrin_b.fx, -(yb - intrin_b.cy) / intrin_b.fy, -1.0])
        ra = ray_a / np.linalg.norm(ray_a)
        rb = ray_b / np.linalg.norm(ray_b)
        ca = fa["pose"][:3, :3] @ (ra / d_rel_a)
        cb = fb["pose"][:3, :3] @ (rb / d_rel_b)
        diff_flip.append(ca - cb)
        off_flip.append(fa["pose"][:3, 3] - fb["pose"][:3, 3])
    if diff_flip:
        dflip = np.stack(diff_flip)
        oflip = np.stack(off_flip)
        s_flip = _closed_form(dflip, oflip)
        r_flip = dflip * s_flip + oflip
        print(f"  s* = {s_flip:.4f}  residual p50={np.median(np.linalg.norm(r_flip, axis=1)):.3f}m  p95={np.percentile(np.linalg.norm(r_flip, axis=1), 95):.3f}m")

    # 테스트3: ARKit ray + R.T 조합 — 두 수정 동시 적용
    print("\n[test3] ARKit ray + pose R.T 조합:")
    diff_13 = []
    off_13 = []
    for idx_a, idx_b in sel:
        xa = mp.kpts_a[idx_a, 0] * sx_a
        ya = mp.kpts_a[idx_a, 1] * sy_a
        xb = mp.kpts_b[idx_b, 0] * sx_b
        yb = mp.kpts_b[idx_b, 1] * sy_b
        d_rel_a = float(fa["depth"][int(round(ya)), int(round(xa))])
        d_rel_b = float(fb["depth"][int(round(yb)), int(round(xb))])
        if d_rel_a < 1e-4 or d_rel_b < 1e-4:
            continue
        intrin_a = fa["intrin"]
        intrin_b = fb["intrin"]
        ray_a = np.array([(xa - intrin_a.cx) / intrin_a.fx, -(ya - intrin_a.cy) / intrin_a.fy, -1.0])
        ray_b = np.array([(xb - intrin_b.cx) / intrin_b.fx, -(yb - intrin_b.cy) / intrin_b.fy, -1.0])
        ra = ray_a / np.linalg.norm(ray_a)
        rb = ray_b / np.linalg.norm(ray_b)
        ca = fa["pose"][:3, :3].T @ (ra / d_rel_a)
        cb = fb["pose"][:3, :3].T @ (rb / d_rel_b)
        diff_13.append(ca - cb)
        off_13.append(fa["pose"][:3, 3] - fb["pose"][:3, 3])
    if diff_13:
        dt = np.stack(diff_13)
        ot = np.stack(off_13)
        st = _closed_form(dt, ot)
        rt = dt * st + ot
        print(f"  s* = {st:.4f}  residual p50={np.median(np.linalg.norm(rt, axis=1)):.3f}m  p95={np.percentile(np.linalg.norm(rt, axis=1), 95):.3f}m")

    # 테스트2: pose rotation transpose 적용
    print("\n[test] pose rotation transposed (in case world_T_cam is actually cam_T_world):")
    diff_t = []
    off_t = []
    for idx_a, idx_b in sel:
        xa = mp.kpts_a[idx_a, 0] * sx_a
        ya = mp.kpts_a[idx_a, 1] * sy_a
        xb = mp.kpts_b[idx_b, 0] * sx_b
        yb = mp.kpts_b[idx_b, 1] * sy_b
        d_rel_a = float(fa["depth"][int(round(ya)), int(round(xa))])
        d_rel_b = float(fb["depth"][int(round(yb)), int(round(xb))])
        if d_rel_a < 1e-4 or d_rel_b < 1e-4:
            continue
        intrin_a = fa["intrin"]
        intrin_b = fb["intrin"]
        ray_a = np.array([(xa - intrin_a.cx) / intrin_a.fx, (ya - intrin_a.cy) / intrin_a.fy, 1.0])
        ray_b = np.array([(xb - intrin_b.cx) / intrin_b.fx, (yb - intrin_b.cy) / intrin_b.fy, 1.0])
        ra = ray_a / np.linalg.norm(ray_a)
        rb = ray_b / np.linalg.norm(ray_b)
        ca = fa["pose"][:3, :3].T @ (ra / d_rel_a)
        cb = fb["pose"][:3, :3].T @ (rb / d_rel_b)
        diff_t.append(ca - cb)
        off_t.append(fa["pose"][:3, 3] - fb["pose"][:3, 3])
    if diff_t:
        dtt = np.stack(diff_t)
        ott = np.stack(off_t)
        s_t = _closed_form(dtt, ott)
        r_t = dtt * s_t + ott
        print(f"  s* = {s_t:.4f}  residual p50={np.median(np.linalg.norm(r_t, axis=1)):.3f}m  p95={np.percentile(np.linalg.norm(r_t, axis=1), 95):.3f}m")

    print(f"initial_s used: {initial_s:.4f}")
    print(f"residual norms at init: p50={np.median(np.linalg.norm(r_init, axis=1)):.2f}m "
          f"p95={np.percentile(np.linalg.norm(r_init, axis=1), 95):.2f}m")

    # s* 로 재평가
    r_opt = diff * s_star + off
    print(f"residual norms at s*: p50={np.median(np.linalg.norm(r_opt, axis=1)):.2f}m "
          f"p95={np.percentile(np.linalg.norm(r_opt, axis=1), 95):.2f}m")

    # 추가 점검: camera translation 차이 크기
    print(f"\ndiag: ||t_a - t_b||={np.linalg.norm(fa['pose'][:3,3] - fb['pose'][:3,3]):.3f}m")
    print(f"diag: diff_coeff magnitudes median={float(np.median(np.linalg.norm(diff, axis=1))):.3f}")
    print(f"diag: sample coeff_a={coeff_a.tolist()}, coeff_b={coeff_b.tolist()}")
    print(f"diag: pose_a[:3,3]={fa['pose'][:3,3].tolist()}")
    print(f"diag: pose_b[:3,3]={fb['pose'][:3,3].tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
