"""measure_corridor_stats.py — scan 9c481325의 trajectory + corridor axis 분포 측정.

Sprint 34 첫 commit: corridor_box_buffer 구현 가부를 결정하는 사전 측정.

사용법:
    uv run python scripts/measure_corridor_stats.py [--scan-id SCAN_ID] [--out PATH]

출력:
    _workspace/sprint_34/corridor_stats.json (기본값)

측정 항목:
    - trajectory total length
    - segment length distribution (연속 keyframe pair 간격)
    - yaw delta histogram (방향 변화 분포)
    - rotation_residual distribution (quantized axis 대비 잔여각)
    - clamp_max_ratio (adaptive_buffer r==max_buffer_m 비율 — Sprint 24 로그 기반)
    - dominant axis estimate (5° bin histogram)
    - expected rejected_ratio 추정

주의:
    clamp_max_ratio는 Segformer floor mask 없이는 재현 불가.
    Sprint 24 build log (clamp_max=154 / frame_used=197) 를 인용한다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = _PROJECT_ROOT / "_workspace" / "sprint_34" / "corridor_stats.json"
_DEFAULT_SCAN_ID = "9c481325-5953-4bcc-b8d0-07a5a61f6755"

# ── Sprint 24 빌드 로그 인용값 (floor mask 없이 측정 불가한 항목) ─────────────
# sprint_24/feedback.md 참고:
#   "adaptive_buffer가 여전히 r=5m max clamp 비율 높음 (clamp_max=154/197)"
#   frame_used=197, n_clamp_max=154 (축 swap fix 이후)
_LOG_FRAME_USED = 197
_LOG_N_CLAMP_MAX = 154
_LOG_R_MEDIAN_M = 0.90   # CLAUDE.md Sprint 22 결과


def decode_pose_matrix(pose_bytes: bytes) -> np.ndarray:
    """BLOB (16 float32 little-endian column-major) → 4x4 Z-up world-from-camera."""
    values = struct.unpack_from("<16f", pose_bytes)
    arkit_pose = np.array(values, dtype=np.float64).reshape(4, 4, order="F")
    s = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    return s @ arkit_pose


def _segment_length(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _yaw_angle_deg(forward_xy: np.ndarray) -> float:
    """xy forward vector → yaw in [-180, 180]."""
    return float(math.degrees(math.atan2(float(forward_xy[1]), float(forward_xy[0]))))


def _angle_mod90(angle_deg: float) -> float:
    """angle → [0, 90) normalized."""
    a = angle_deg % 180.0
    return a - 90.0 if a >= 90.0 else a


def estimate_dominant_axis(yaw_deltas_deg: list[float], bin_deg: float = 5.0) -> float:
    """연속 segment 방향의 mod-90 histogram peak → dominant axis (0~90)."""
    mod90 = [_angle_mod90(d) for d in yaw_deltas_deg]
    n_bins = int(90.0 / bin_deg)
    counts = [0.0] * n_bins
    for v in mod90:
        # v in [-45, 45) → normalize to [0, 90)
        v_norm = v + 45.0  # [0, 90)
        bin_idx = min(int(v_norm / bin_deg), n_bins - 1)
        counts[bin_idx] += 1.0
    if not any(counts):
        return 0.0
    peak_bin = int(np.argmax(counts))
    return peak_bin * bin_deg - 45.0 + bin_deg / 2.0


def rotation_residuals(
    forwards: list[tuple[float, float]],
    axis_deg: float,
    min_len_threshold: float = 0.01,
) -> list[float]:
    """각 forward vector와 quantized axis (4방향) 사이 최소 잔여각 계산."""
    candidates_deg = [axis_deg, axis_deg + 90.0, axis_deg + 180.0, axis_deg + 270.0]
    candidates_rad = [math.radians(d) for d in candidates_deg]
    residuals = []
    for fx, fy in forwards:
        norm = math.hypot(fx, fy)
        if norm < min_len_threshold:
            continue
        fwd_angle = math.atan2(fy, fx)
        min_residual = min(
            abs(math.degrees(fwd_angle - c_rad) % 360.0)
            for c_rad in candidates_rad
        )
        # normalize to [0, 90]
        while min_residual > 90.0:
            min_residual -= 90.0
        residuals.append(min_residual)
    return residuals


async def fetch_keyframe_data(
    scan_id: str,
    db_url: str,
) -> list[dict[str, Any]]:
    """Postgres에서 keyframe pose 데이터 조회."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(db_url, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT seq, tx, ty, tz, pose_matrix
                    FROM keyframe_meta
                    WHERE scan_id = :scan_id
                    ORDER BY seq
                    """
                ),
                {"scan_id": scan_id},
            )
            rows = result.fetchall()
    finally:
        await engine.dispose()

    keyframes = []
    for seq, tx, ty, tz, pose_bytes in rows:
        pose = decode_pose_matrix(bytes(pose_bytes))
        # forward = -R[:,2] (camera -z in world)
        fwd3 = -pose[:3, 2]
        fwd_xy = (float(fwd3[0]), float(fwd3[1]))
        keyframes.append(
            {
                "seq": int(seq),
                "tx": float(tx),
                "ty": float(ty),
                "tz": float(tz),
                "fwd_xy": fwd_xy,
                "cam_xy": (float(pose[0, 3]), float(pose[1, 3])),
                "cam_z": float(pose[2, 3]),
            }
        )
    return keyframes


def compute_stats(keyframes: list[dict[str, Any]]) -> dict[str, Any]:
    """trajectory 기반 통계 계산."""
    if len(keyframes) < 2:
        return {"error": "keyframe count < 2"}

    # ── trajectory ───────────────────────────────────────────────────────────
    traj_xy = [kf["cam_xy"] for kf in keyframes]
    seg_lengths = [
        _segment_length(traj_xy[i], traj_xy[i + 1])
        for i in range(len(traj_xy) - 1)
    ]
    total_length_m = float(sum(seg_lengths))

    valid_segs = [s for s in seg_lengths if s >= 0.05]  # 5cm 이상만 유효

    # ── yaw angles ────────────────────────────────────────────────────────────
    forwards = [kf["fwd_xy"] for kf in keyframes]
    yaw_angles_deg = [_yaw_angle_deg(np.array(f)) for f in forwards]
    yaw_deltas_deg = [
        yaw_angles_deg[i + 1] - yaw_angles_deg[i]
        for i in range(len(yaw_angles_deg) - 1)
    ]
    # normalize to [-180, 180]
    yaw_deltas_deg = [
        d - 360.0 if d > 180.0 else (d + 360.0 if d < -180.0 else d)
        for d in yaw_deltas_deg
    ]

    # ── dominant axis ─────────────────────────────────────────────────────────
    # trajectory segment 방향으로 axis 추정
    seg_yaws = []
    for i in range(len(traj_xy) - 1):
        dx = traj_xy[i + 1][0] - traj_xy[i][0]
        dy = traj_xy[i + 1][1] - traj_xy[i][1]
        if math.hypot(dx, dy) >= 0.1:  # 10cm 이상 이동한 쌍만
            seg_yaws.append(math.degrees(math.atan2(dy, dx)))

    axis_deg = estimate_dominant_axis(seg_yaws) if seg_yaws else 0.0

    # ── rotation residuals ────────────────────────────────────────────────────
    residuals = rotation_residuals(forwards, axis_deg)
    rejected = [r for r in residuals if r > 25.0]
    rejected_ratio = len(rejected) / len(residuals) if residuals else 1.0

    # ── camera height distribution ─────────────────────────────────────────────
    cam_z_list = [kf["cam_z"] for kf in keyframes]
    cam_z_arr = np.array(cam_z_list)
    z0_estimate = float(np.percentile(cam_z_arr, 5))  # WalkableGridStep.estimate_z0 로직

    # ── clamp_max ratio (Sprint 24 로그 인용) ────────────────────────────────
    # floor mask 없이 재현 불가 — Sprint 24 빌드 결과 인용
    clamp_max_ratio_log = _LOG_N_CLAMP_MAX / _LOG_FRAME_USED

    # ── 통계 요약 ──────────────────────────────────────────────────────────────
    seg_lengths_arr = np.array(valid_segs) if valid_segs else np.array([0.0])
    residuals_arr = np.array(residuals) if residuals else np.array([0.0])
    yaw_abs = np.abs(yaw_deltas_deg)

    return {
        "scan_id": _DEFAULT_SCAN_ID,
        "keyframe_count": len(keyframes),
        "trajectory": {
            "total_length_m": round(total_length_m, 3),
            "valid_segments": len(valid_segs),
            "seg_length_p50_m": round(float(np.percentile(seg_lengths_arr, 50)), 4),
            "seg_length_p75_m": round(float(np.percentile(seg_lengths_arr, 75)), 4),
            "seg_length_p95_m": round(float(np.percentile(seg_lengths_arr, 95)), 4),
        },
        "yaw": {
            "delta_abs_p50_deg": round(float(np.percentile(yaw_abs, 50)), 2),
            "delta_abs_p75_deg": round(float(np.percentile(yaw_abs, 75)), 2),
            "delta_abs_p90_deg": round(float(np.percentile(yaw_abs, 90)), 2),
            "delta_abs_p95_deg": round(float(np.percentile(yaw_abs, 95)), 2),
        },
        "dominant_axis": {
            "axis_deg": round(axis_deg, 2),
            "note": "trajectory segment direction histogram peak (5deg bin)",
        },
        "rotation_residual": {
            "p50_deg": round(float(np.percentile(residuals_arr, 50)), 2),
            "p75_deg": round(float(np.percentile(residuals_arr, 75)), 2),
            "p90_deg": round(float(np.percentile(residuals_arr, 90)), 2),
            "total_frames": len(residuals),
            "rejected_count": len(rejected),
            "rejected_ratio": round(rejected_ratio, 4),
            "threshold_deg": 25.0,
        },
        "camera_height": {
            "z0_estimate_5pct_m": round(z0_estimate, 4),
            "cam_z_p5_m": round(float(np.percentile(cam_z_arr, 5)), 4),
            "cam_z_p50_m": round(float(np.percentile(cam_z_arr, 50)), 4),
            "cam_z_p95_m": round(float(np.percentile(cam_z_arr, 95)), 4),
        },
        "adaptive_radius": {
            "source": "Sprint 24 build log (floor mask required for recomputation)",
            "sprint24_frame_used": _LOG_FRAME_USED,
            "sprint24_n_clamp_max": _LOG_N_CLAMP_MAX,
            "clamp_max_ratio": round(clamp_max_ratio_log, 4),
            "r_median_m": _LOG_R_MEDIAN_M,
            "max_buffer_m": 5.0,
            "note": "clamp_max_ratio = n_clamp_max / frame_used. r==max_buffer_m인 프레임 비율.",
        },
        "acceptance_check": {
            "trajectory_length_ok": total_length_m >= 30.0,
            "trajectory_length_m": round(total_length_m, 1),
            "trajectory_threshold_m": 30.0,
            "clamp_max_ratio_ok": clamp_max_ratio_log < 0.40,
            "clamp_max_ratio": round(clamp_max_ratio_log, 4),
            "clamp_max_threshold": 0.40,
            "rotation_residual_p90_ok": float(np.percentile(residuals_arr, 90)) < 25.0,
            "rotation_residual_p90_deg": round(float(np.percentile(residuals_arr, 90)), 2),
            "rotation_residual_threshold_deg": 25.0,
            "rejected_ratio_ok": rejected_ratio < 0.30,
            "rejected_ratio": round(rejected_ratio, 4),
            "rejected_ratio_threshold": 0.30,
        },
        "verdict": {
            "proceed_with_corridor_box_buffer": (
                total_length_m >= 30.0
                and clamp_max_ratio_log < 0.40
                and float(np.percentile(residuals_arr, 90)) < 25.0
                and rejected_ratio < 0.30
            ),
            "blocking_reasons": _collect_blocking_reasons(
                total_length_m,
                clamp_max_ratio_log,
                float(np.percentile(residuals_arr, 90)),
                rejected_ratio,
            ),
        },
    }


def _collect_blocking_reasons(
    total_length_m: float,
    clamp_max_ratio: float,
    rotation_residual_p90: float,
    rejected_ratio: float,
) -> list[str]:
    reasons = []
    if total_length_m < 30.0:
        reasons.append(
            f"trajectory_too_short: {total_length_m:.1f}m < 30m threshold"
        )
    if clamp_max_ratio >= 0.40:
        reasons.append(
            f"clamp_max_ratio_too_high: {clamp_max_ratio:.1%} >= 40% threshold "
            f"(Sprint 24 log: {_LOG_N_CLAMP_MAX}/{_LOG_FRAME_USED}={clamp_max_ratio:.1%}). "
            "원인: 카메라 height 0.3~0.6m 데이터 특성으로 floor projection이 넓게 퍼져 "
            "r_raw > 5m max clamp 다발. "
            "대책: strip_fraction 조정 / horizon filter 강화 / 다른 approach 검토."
        )
    if rotation_residual_p90 >= 25.0:
        reasons.append(
            f"rotation_residual_p90_too_high: {rotation_residual_p90:.1f}deg >= 25deg threshold"
        )
    if rejected_ratio >= 0.30:
        reasons.append(
            f"rejected_ratio_too_high: {rejected_ratio:.1%} >= 30% threshold"
        )
    return reasons


async def main(scan_id: str, db_url: str, out_path: Path) -> None:
    logger.info("Fetching keyframe data for scan_id=%s", scan_id)
    keyframes = await fetch_keyframe_data(scan_id, db_url)
    logger.info("Fetched %d keyframes", len(keyframes))

    stats = compute_stats(keyframes)
    stats["scan_id"] = scan_id

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("Stats written to %s", out_path)

    # 판정 결과 출력
    verdict = stats["verdict"]
    proceed = verdict["proceed_with_corridor_box_buffer"]
    reasons = verdict["blocking_reasons"]

    print("\n" + "=" * 60)
    print(f"VERDICT: {'PROCEED' if proceed else 'BLOCKED'}")
    print("=" * 60)

    ac = stats["acceptance_check"]
    checks = [
        ("trajectory_length", ac["trajectory_length_ok"],
         f"{ac['trajectory_length_m']}m >= {ac['trajectory_threshold_m']}m"),
        ("clamp_max_ratio", ac["clamp_max_ratio_ok"],
         f"{ac['clamp_max_ratio']:.1%} < {ac['clamp_max_threshold']:.0%}"),
        ("rotation_residual_p90", ac["rotation_residual_p90_ok"],
         f"{ac['rotation_residual_p90_deg']}deg < {ac['rotation_residual_threshold_deg']}deg"),
        ("rejected_ratio", ac["rejected_ratio_ok"],
         f"{ac['rejected_ratio']:.1%} < {ac['rejected_ratio_threshold']:.0%}"),
    ]
    for name, ok, detail in checks:
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}")

    if reasons:
        print("\nBlocking reasons:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure corridor stats for Sprint 34")
    parser.add_argument(
        "--scan-id",
        default=_DEFAULT_SCAN_ID,
        help="scan_id to analyze",
    )
    parser.add_argument(
        "--db-url",
        default="postgresql+asyncpg://indoor:indoor@localhost:5432/indoor",
        help="database URL",
    )
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="output JSON path",
    )
    args = parser.parse_args()
    asyncio.run(main(args.scan_id, args.db_url, Path(args.out)))
