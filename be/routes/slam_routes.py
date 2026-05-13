from __future__ import annotations

import asyncio
import base64
import functools
import io
import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image, ImageOps

from config.settings import Settings
from models.slam_api import (
    HealthResponse,
    MapMetadata,
    MaskDebugImage,
    MaskDebugResponse,
    MatchDebugResponse,
    SLAMLocalizeRequest,
    SLAMLocalizeResponse,
    SLAMProcessRequest,
    SLAMProcessResponse,
)
from models.tracking_api import (
    TrackingFrameResponse,
    TrackingResolveRequest,
    TrackingResolveResponse,
    TrackingSessionStartRequest,
    TrackingSessionStartResponse,
    TrackingSessionStateResponse,
)
from utils import logger

settings = Settings()

SLAM_TAG_PROCESS = "SLAM - 처리"
SLAM_TAG_LOCALIZE = "SLAM - 위치추정"
SLAM_TAG_DEBUG = "SLAM - 디버그"
SLAM_TAG_TRACKING = "SLAM - Tracking VPS"
USER_APP_TAG = "사용자 앱 API"

router = APIRouter(prefix="/api/slam")

postgres_adapter = None
job_queue = None

_sp_engine = None
_sp_engine_lock = threading.Lock()
_tracking_sessions: dict[str, dict[str, Any]] = {}
_tracking_sessions_lock = threading.Lock()


def _pose_quat_to_matrix(pose: dict[str, Any]) -> list[list[float]]:
    qx = float(pose["qx"])
    qy = float(pose["qy"])
    qz = float(pose["qz"])
    qw = float(pose["qw"])
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0:
        raise ValueError("zero quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def _mat3_transpose(R: list[list[float]]) -> list[list[float]]:
    return [[R[j][i] for j in range(3)] for i in range(3)]


def _mat3_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mat3_vec_mul(A: list[list[float]], v: list[float]) -> list[float]:
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _vec_norm(v: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _yaw_from_matrix(R: list[list[float]]) -> float:
    return math.degrees(math.atan2(R[1][0], R[0][0]))


def _angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _parse_ar_transform(ar_pose: Any) -> dict[str, Any] | None:
    if not isinstance(ar_pose, dict):
        return None
    values = ar_pose.get("world_T_camera")
    if not isinstance(values, list) or len(values) != 16:
        return None
    try:
        vals = [float(v) for v in values]
    except Exception:
        return None
    # Client sends row-major world_T_camera: rows [r00 r01 r02 tx]...
    R = [
        [vals[0], vals[1], vals[2]],
        [vals[4], vals[5], vals[6]],
        [vals[8], vals[9], vals[10]],
    ]
    t = [vals[3], vals[7], vals[11]]
    return {
        "R": R,
        "t": t,
        "trackingState": str(ar_pose.get("trackingState") or ""),
    }


def _arkit_camera_to_opencv_camera_rotation() -> list[list[float]]:
    # ARKit camera space is right-handed with +X right, +Y up, and +Z toward
    # the viewer/screen side. OpenCV/PnP camera space is +X right, +Y down,
    # +Z forward into the scene. The fixed camera-frame conversion is a 180
    # degree rotation around X.
    return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]


def _rtab_optical_to_base_rotation() -> list[list[float]]:
    # RTAB-Map CameraModel.localTransform / opticalRotation maps image optical
    # frame (x right, y down, z forward) to robot/base frame
    # (x forward, y left, z up). This is also the rotation stored in our DB
    # calibration blobs.
    return [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]


def _convert_ar_camera_to_rtab_camera(ar: dict[str, Any]) -> dict[str, Any]:
    arkit_to_opencv = _arkit_camera_to_opencv_camera_rotation()
    optical_to_base = _rtab_optical_to_base_rotation()
    fix = _mat3_mul(arkit_to_opencv, _mat3_transpose(optical_to_base))
    return {
        "R": _mat3_mul(ar["R"], fix),
        "t": ar["t"],
        "trackingState": ar.get("trackingState") or "",
    }


def _map_anchor_from_pose_and_ar(map_pose: dict[str, Any], ar: dict[str, Any]) -> dict[str, Any]:
    R_map = _pose_quat_to_matrix(map_pose)
    t_map = [float(map_pose["x"]), float(map_pose["y"]), float(map_pose["z"])]
    R_ar = ar["R"]
    t_ar = ar["t"]
    R_map_ar = _mat3_mul(R_map, _mat3_transpose(R_ar))
    t_map_ar = _vec_sub(t_map, _mat3_vec_mul(R_map_ar, t_ar))
    return {"R": R_map_ar, "t": t_map_ar}


def _predict_map_pose_from_ar(anchor: dict[str, Any], ar: dict[str, Any]) -> dict[str, Any]:
    R_pred = _mat3_mul(anchor["R"], ar["R"])
    t_pred = _vec_add(_mat3_vec_mul(anchor["R"], ar["t"]), anchor["t"])
    return {"R": R_pred, "t": t_pred}


def _pose_translation(pose: dict[str, Any]) -> list[float]:
    return [float(pose["x"]), float(pose["y"]), float(pose["z"])]


def _matrix_to_quat(R: list[list[float]]) -> tuple[float, float, float, float]:
    trace = R[0][0] + R[1][1] + R[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2][1] - R[1][2]) / s
        qy = (R[0][2] - R[2][0]) / s
        qz = (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        qw = (R[2][1] - R[1][2]) / s
        qx = 0.25 * s
        qy = (R[0][1] + R[1][0]) / s
        qz = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        qw = (R[0][2] - R[2][0]) / s
        qx = (R[0][1] + R[1][0]) / s
        qy = 0.25 * s
        qz = (R[1][2] + R[2][1]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        qw = (R[1][0] - R[0][1]) / s
        qx = (R[0][2] + R[2][0]) / s
        qy = (R[1][2] + R[2][1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0:
        return 0.0, 0.0, 0.0, 1.0
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return qx, qy, qz, qw


def _quat_to_matrix(q: tuple[float, float, float, float]) -> list[list[float]]:
    return _pose_quat_to_matrix({"qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3]})


def _weighted_average_quats(
    quats: list[tuple[float, float, float, float]], weights: list[float]
) -> tuple[float, float, float, float]:
    ref = quats[0]
    accum = [0.0, 0.0, 0.0, 0.0]
    for q, w in zip(quats, weights):
        dot = sum(ref[i] * q[i] for i in range(4))
        sign = -1.0 if dot < 0 else 1.0
        for i in range(4):
            accum[i] += sign * float(w) * q[i]
    norm = math.sqrt(sum(v * v for v in accum))
    if norm <= 0:
        return ref
    qx, qy, qz, qw = (accum[0] / norm, accum[1] / norm, accum[2] / norm, accum[3] / norm)
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return qx, qy, qz, qw


def _pose_from_matrix_translation(R: list[list[float]], t: list[float]) -> dict[str, float]:
    qx, qy, qz, qw = _matrix_to_quat(R)
    return {
        "x": float(t[0]),
        "y": float(t[1]),
        "z": float(t[2]),
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
    }


def _robust_initial_anchor(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not candidates:
        return None, {"sourceFrames": 0, "inlierFrames": 0}
    scored = sorted(candidates, key=lambda c: float(c.get("score") or 0.0), reverse=True)
    best_inliers: list[dict[str, Any]] = []
    best_key: tuple[int, float, float] | None = None
    for seed in scored:
        seed_anchor = seed["anchor"]
        seed_t = seed_anchor["t"]
        seed_yaw = _yaw_from_matrix(seed_anchor["R"])
        seed_inliers = []
        for c in scored:
            anchor = c["anchor"]
            dt = _vec_norm(_vec_sub(anchor["t"], seed_t))
            dyaw = abs(_angle_diff_deg(_yaw_from_matrix(anchor["R"]), seed_yaw))
            if dt <= 0.75 and dyaw <= 45.0:
                seed_inliers.append(c)
        total_score = sum(float(c.get("score") or 0.0) for c in seed_inliers)
        seed_score = float(seed.get("score") or 0.0)
        key = (len(seed_inliers), total_score, seed_score)
        if best_key is None or key > best_key:
            best_key = key
            best_inliers = seed_inliers
    inliers = best_inliers
    if not inliers:
        inliers = [scored[0]]

    weights = [max(0.05, float(c.get("score") or 0.0)) for c in inliers]
    total_w = sum(weights)
    avg_t = [
        sum(c["anchor"]["t"][i] * w for c, w in zip(inliers, weights)) / total_w
        for i in range(3)
    ]
    quats = [_matrix_to_quat(c["anchor"]["R"]) for c in inliers]
    avg_q = _weighted_average_quats(quats, weights)
    avg_R = _quat_to_matrix(avg_q)
    spreads_t = [_vec_norm(_vec_sub(c["anchor"]["t"], avg_t)) for c in inliers]
    avg_yaw = _yaw_from_matrix(avg_R)
    spreads_yaw = [abs(_angle_diff_deg(_yaw_from_matrix(c["anchor"]["R"]), avg_yaw)) for c in inliers]
    diagnostics = {
        "sourceFrames": len(candidates),
        "inlierFrames": len(inliers),
        "translationSpreadM": max(spreads_t) if spreads_t else 0.0,
        "translationMeanSpreadM": sum(spreads_t) / max(len(spreads_t), 1),
        "yawSpreadDeg": max(spreads_yaw) if spreads_yaw else 0.0,
        "yawMeanSpreadDeg": sum(spreads_yaw) / max(len(spreads_yaw), 1),
        "confidence": min(0.99, sum(weights) / max(len(inliers), 1)),
        "frames": [c.get("frameIndex") for c in inliers],
    }
    return {"R": avg_R, "t": avg_t}, diagnostics


def _resolve_anchor_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    min_anchor_frames: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    anchor, diagnostics = _robust_initial_anchor(candidates)
    if anchor is None or int(diagnostics.get("inlierFrames") or 0) < int(min_anchor_frames):
        diagnostics["reason"] = "insufficient_consistent_anchor_frames"
        diagnostics["requiredInlierFrames"] = min_anchor_frames
        return None, diagnostics

    optimized_anchor, optimizer_diagnostics = _optimize_anchor_6dof(candidates, anchor)
    diagnostics["optimizer"] = optimizer_diagnostics
    return optimized_anchor, diagnostics


def _append_resolved_anchor_sample(
    session: dict[str, Any],
    *,
    frame_index: int | str,
    timestamp_ms: int | None,
    server_received_ms: int,
    anchor: dict[str, Any],
    diagnostics: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    sample = {
        "frameIndex": frame_index,
        "timestampMs": timestamp_ms,
        "serverReceivedMs": server_received_ms,
        "anchor": anchor,
        "diagnostics": diagnostics,
        "source": source,
        "confidence": diagnostics.get("confidence"),
        "inlierFrames": diagnostics.get("inlierFrames"),
        "sourceFrames": diagnostics.get("sourceFrames"),
    }
    samples = list(session.get("resolved_anchor_samples") or [])
    samples.append(sample)
    samples.sort(
        key=lambda s: (
            int(s.get("timestampMs")) if s.get("timestampMs") is not None else 2**62,
            int(s.get("serverReceivedMs") or 0),
        )
    )
    session["resolved_anchor_samples"] = samples[-80:]
    return sample


def _is_newer_timestamp(candidate_ms: int | None, reference_ms: int | None) -> bool:
    if candidate_ms is None:
        return reference_ms is None
    if reference_ms is None:
        return True
    return int(candidate_ms) >= int(reference_ms)


def _timestamp_distance_ms(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return abs(int(a) - int(b))


def _is_stale_for_state_update(
    session: dict[str, Any],
    timestamp_ms: int | None,
    *,
    max_lag_ms: int = 3000,
) -> bool:
    latest = session.get("latest_frame_timestamp_ms")
    if timestamp_ms is None or latest is None:
        return False
    return int(timestamp_ms) < int(latest) - int(max_lag_ms)


def _select_resolved_anchor_sample(
    samples: list[dict[str, Any]],
    *,
    target_timestamp_ms: int | None,
) -> tuple[dict[str, Any] | None, int | None]:
    valid_samples = [s for s in samples if s.get("anchor") is not None]
    if not valid_samples:
        return None, None
    if target_timestamp_ms is None:
        sample = valid_samples[-1]
        return sample, None

    def _distance(sample: dict[str, Any]) -> tuple[int, int]:
        ts = sample.get("timestampMs")
        if ts is None:
            return (2**62, int(sample.get("serverReceivedMs") or 0))
        return (abs(int(ts) - int(target_timestamp_ms)), int(sample.get("serverReceivedMs") or 0))

    sample = min(valid_samples, key=_distance)
    sample_ts = sample.get("timestampMs")
    if sample_ts is None:
        return sample, None
    return sample, abs(int(sample_ts) - int(target_timestamp_ms))


def _anchor_diagnostics_include_frame(diagnostics: dict[str, Any], frame_index: int) -> bool:
    frame_keys = {frame_index, str(frame_index)}
    frames = list(diagnostics.get("frames") or [])
    optimizer_frames = list((diagnostics.get("optimizer") or {}).get("frames") or [])
    return any(frame in frame_keys for frame in frames + optimizer_frames)


def _optimize_anchor_6dof(
    candidates: list[dict[str, Any]],
    initial_anchor: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine map_T_AR as a full 6DoF transform over anchor candidates.

    This keeps RTAB-Map/map output convention unchanged. The variables are a
    single map_T_AR transform. Each visual/depth frame contributes one measured
    map_T_AR_i anchor with robust translation + rotation residuals.
    """
    try:
        import numpy as _np
        from scipy.optimize import least_squares as _least_squares
        from scipy.spatial.transform import Rotation as _Rotation
    except Exception as exc:
        return initial_anchor, {"optimizer": "fallback_average", "error": str(exc)}

    if not candidates:
        return initial_anchor, {"optimizer": "fallback_average", "error": "no_candidates"}

    init_R = _np.array(initial_anchor["R"], dtype=_np.float64)
    init_t = _np.array(initial_anchor["t"], dtype=_np.float64)
    init_rot = _Rotation.from_matrix(init_R)
    init_yaw = _yaw_from_matrix(initial_anchor["R"])

    filtered: list[dict[str, Any]] = []
    for c in candidates:
        anchor = c.get("anchor")
        if not anchor:
            continue
        dt = _vec_norm(_vec_sub(anchor["t"], initial_anchor["t"]))
        dyaw = abs(_angle_diff_deg(_yaw_from_matrix(anchor["R"]), init_yaw))
        if dt <= 0.85 and dyaw <= 50.0:
            filtered.append(c)
    if len(filtered) < 2:
        filtered = [c for c in candidates if c.get("anchor")]
    if not filtered:
        return initial_anchor, {"optimizer": "fallback_average", "error": "no_valid_candidates"}

    anchor_R = [_np.array(c["anchor"]["R"], dtype=_np.float64) for c in filtered]
    anchor_t = [_np.array(c["anchor"]["t"], dtype=_np.float64) for c in filtered]
    anchor_rot = [_Rotation.from_matrix(R) for R in anchor_R]
    weights = _np.array([max(0.05, float(c.get("score") or 0.0)) for c in filtered], dtype=_np.float64)
    weights = weights / max(float(_np.median(weights)), 1e-6)

    x0 = _np.concatenate([init_t, init_rot.as_rotvec()])
    sigma_t = 0.18
    sigma_r = _np.deg2rad(8.0)

    def residual(x: _np.ndarray) -> _np.ndarray:
        t = x[:3]
        R = _Rotation.from_rotvec(x[3:])
        res: list[float] = []
        for target_t, target_R, w in zip(anchor_t, anchor_rot, weights):
            sw = float(_np.sqrt(w))
            res.extend(((t - target_t) / sigma_t * sw).tolist())
            rot_delta = (target_R.inv() * R).as_rotvec()
            res.extend((rot_delta / sigma_r * sw).tolist())
        return _np.array(res, dtype=_np.float64)

    result = _least_squares(
        residual,
        x0,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=100,
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
    )
    opt_t = result.x[:3]
    opt_R_obj = _Rotation.from_rotvec(result.x[3:])
    opt_R = opt_R_obj.as_matrix()

    trans_errors = [_np.linalg.norm(opt_t - t) for t in anchor_t]
    rot_errors = [
        float(_np.rad2deg(_np.linalg.norm((target_R.inv() * opt_R_obj).as_rotvec())))
        for target_R in anchor_rot
    ]
    robust_inliers = [
        i for i, (te, re) in enumerate(zip(trans_errors, rot_errors))
        if te <= 0.35 and re <= 25.0
    ]
    initial_residual = residual(x0)
    final_residual = residual(result.x)
    diagnostics = {
        "optimizer": "soft_l1_se3",
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nfev),
        "initialLinearCost": float(0.5 * _np.sum(initial_residual ** 2)),
        "finalLinearCost": float(0.5 * _np.sum(final_residual ** 2)),
        "finalRobustCost": float(result.cost),
        "sourceFrames": len(candidates),
        "optimizedFrames": len(filtered),
        "optimizerInlierFrames": len(robust_inliers),
        "translationResidualMedianM": float(_np.median(trans_errors)) if trans_errors else None,
        "translationResidualMaxM": float(_np.max(trans_errors)) if trans_errors else None,
        "rotationResidualMedianDeg": float(_np.median(rot_errors)) if rot_errors else None,
        "rotationResidualMaxDeg": float(_np.max(rot_errors)) if rot_errors else None,
        "frames": [filtered[i].get("frameIndex") for i in robust_inliers],
    }
    anchor = {"R": opt_R.tolist(), "t": opt_t.tolist()}
    return anchor, diagnostics


def _get_sp_engine():
    global _sp_engine
    if _sp_engine is None:
        with _sp_engine_lock:
            if _sp_engine is None:
                from slam_engines.superpoint.engine import SuperPointEngine

                _sp_engine = SuperPointEngine()
    return _sp_engine


@router.post(
    "/process",
    response_model=SLAMProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=[SLAM_TAG_PROCESS],
    summary="건물 SLAM 처리 요청",
    description=(
        "건물 ID에 연결된 활성 RTAB-Map DB를 처리 큐에 넣습니다. "
        "현재 통합 서버에서는 v2 스캔/빌드 테이블을 우선 조회하고, legacy "
        "`scan_sessions` 스키마가 있으면 fallback으로 사용합니다."
    ),
)
async def process_slam(request: SLAMProcessRequest) -> SLAMProcessResponse:
    sessions = await _get_sessions_or_raise(request.building_id)
    pairs = [
        (str(session["id"]), str(session["file_path"]))
        for session in sessions
        if session.get("id") and session.get("file_path")
    ]
    if not pairs:
        raise HTTPException(
            status_code=404,
            detail=f"No processable RTAB-Map DB found for building {request.building_id}",
        )

    if job_queue is not None:
        await job_queue.enqueue(request.building_id, pairs)
        return SLAMProcessResponse(
            map_id=request.building_id,
            status="PROCESSING",
            queue_position=_queue_length(),
        )

    return SLAMProcessResponse(
        map_id=request.building_id,
        status=_overall_status(sessions),
        queue_position=0,
    )


@router.get(
    "/status/{building_id}",
    tags=[SLAM_TAG_PROCESS],
    summary="건물 SLAM 처리 상태 조회",
    description=(
        "건물 ID 기준으로 활성 floor scan 또는 legacy scan session 상태를 조회합니다. "
        "`map_id`를 따로 받지 않고 건물 ID로 연결된 맵 후보를 찾는 통합 기준입니다."
    ),
)
async def get_slam_status(building_id: str) -> dict[str, Any]:
    sessions = await _get_sessions(building_id)
    if not sessions:
        fallback_db = settings.MAPS_DIR / f"{building_id}.db"
        if fallback_db.exists():
            return {
                "building_id": building_id,
                "status": "COMPLETED",
                "sessions": [],
                "source": "maps_dir",
            }
        return {
            "building_id": building_id,
            "status": "NOT_FOUND",
            "sessions": [],
            "source": "postgres" if postgres_adapter is not None else "unavailable",
        }

    return {
        "building_id": building_id,
        "status": _overall_status(sessions),
        "sessions": [_serialize_session(session) for session in sessions],
        "source": "postgres",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=[SLAM_TAG_PROCESS],
    summary="SLAM 모듈 헬스 체크",
    description="PostgreSQL 연결 상태와 SLAM 처리 큐 길이를 반환합니다.",
)
async def get_slam_health() -> HealthResponse:
    postgres_status = "disabled"
    if postgres_adapter is not None:
        postgres_status = await postgres_adapter.health_check()
    return HealthResponse(
        status="healthy" if postgres_status == "connected" else "degraded",
        postgres=postgres_status,
        queue_length=_queue_length(),
    )


@router.get(
    "/maps/{building_id}/metadata",
    response_model=MapMetadata,
    tags=[SLAM_TAG_PROCESS],
    summary="건물 맵 메타데이터 조회",
    description=(
        "건물 ID로 활성 floor map을 찾고, RTAB-Map DB가 있으면 keyframe 수를 직접 읽습니다. "
        "DB 연결이 없으면 `DATA_DIR/maps/{building_id}.db`를 fallback으로 확인합니다."
    ),
)
async def get_map_metadata(building_id: str) -> MapMetadata:
    sessions = await _get_sessions(building_id)
    floor_maps = await _get_floor_maps(building_id)
    candidate_paths = [Path(_resolve_floor_path(fm)["file_path"]) for fm in floor_maps]

    fallback_db = settings.MAPS_DIR / f"{building_id}.db"
    if fallback_db.exists():
        candidate_paths.append(fallback_db)

    keyframes = sum(await asyncio.gather(*[_count_keyframes(path) for path in candidate_paths]))
    if keyframes == 0 and sessions:
        keyframes = sum(int(session.get("total_nodes") or 0) for session in sessions)

    created_at = _first_created_at(sessions)
    if created_at == "" and fallback_db.exists():
        created_at = datetime.fromtimestamp(fallback_db.stat().st_mtime).isoformat()

    if not sessions and not candidate_paths:
        raise HTTPException(status_code=404, detail=f"No map metadata found for building {building_id}")

    return MapMetadata(
        map_id=building_id,
        building_id=building_id,
        num_keyframes=keyframes,
        created_at=created_at,
        status=_overall_status(sessions) if sessions else "COMPLETED",
    )


@router.post(
    "/localize",
    response_model=SLAMLocalizeResponse,
    status_code=status.HTTP_200_OK,
    tags=[SLAM_TAG_LOCALIZE],
    summary="기본 위치 추정",
    description=(
        "호환용 기본 위치 추정 API입니다. `multipart/form-data`로 이미지 파일을 직접 받고, "
        "`building_id` 또는 기존 호환 필드 `map_id`를 건물 ID로 해석해 활성 floor map을 찾습니다."
    ),
)
async def localize_in_map(
    images: list[UploadFile] = File(..., description="위치 추정에 사용할 이미지 파일 목록"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> SLAMLocalizeResponse:
    return await _localize_uploads(building_id=building_id, map_id=map_id, images=images)


@router.post(
    "/v2/localize",
    response_model=SLAMLocalizeResponse,
    status_code=status.HTTP_200_OK,
    tags=[SLAM_TAG_LOCALIZE],
    summary="v2 위치 추정",
    description=(
        "v2 호환 위치 추정 API입니다. 이미지 파일을 직접 업로드하며, 사람 마스킹 경로를 켠 상태로 "
        "현재 SuperPoint + LightGlue 위치 추정기를 호출합니다. 마스킹 모델이 없으면 fail-open으로 동작합니다."
    ),
)
async def localize_in_map_v2(
    images: list[UploadFile] = File(..., description="위치 추정에 사용할 이미지 파일 목록"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> SLAMLocalizeResponse:
    return await _localize_uploads(
        building_id=building_id,
        map_id=map_id,
        images=images,
        mask_persons=True,
    )


@router.post(
    "/v3/localize",
    response_model=SLAMLocalizeResponse,
    status_code=status.HTTP_200_OK,
    tags=[USER_APP_TAG, SLAM_TAG_LOCALIZE],
    summary="v3 위치 추정",
    description=(
        "v3 위치 추정 API입니다. 이미지 파일을 직접 업로드하고, 건물 ID로 찾은 활성 floor map 전체에 "
        "SuperPoint + LightGlue 매칭을 수행한 뒤 confidence가 가장 높은 층 결과를 반환합니다."
    ),
)
async def localize_in_map_v3(
    images: list[UploadFile] = File(..., description="위치 추정에 사용할 이미지 파일 목록"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
    floor_id: str | None = Form(None, description="선택 층 ID. 없으면 전체 층에서 탐색합니다."),
) -> SLAMLocalizeResponse:
    return await _localize_uploads(building_id=building_id, map_id=map_id, images=images, floor_id=floor_id)


@router.post(
    "/v1/debug/matches",
    response_model=MatchDebugResponse,
    tags=[SLAM_TAG_DEBUG],
    summary="v1 매칭 디버그",
    description=(
        "v1 호환 매칭 디버그 API입니다. 이미지 파일 하나와 건물 ID를 받아 현재 통합된 매칭 "
        "시각화 경로로 결과 이미지를 반환합니다."
    ),
)
async def debug_matches_v1(
    image: UploadFile = File(..., description="디버그할 이미지 파일"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> MatchDebugResponse:
    return await _debug_matches_upload(building_id=building_id, map_id=map_id, image=image)


@router.post(
    "/v2/debug/mask",
    response_model=MaskDebugResponse,
    tags=[SLAM_TAG_DEBUG],
    summary="v2 사람 마스킹 디버그",
    description=(
        "업로드한 이미지에서 사람 감지 박스를 표시해 반환합니다. 현재 마스킹 구현은 모델이 없으면 "
        "감지 0개로 fail-open합니다."
    ),
)
async def debug_mask_v2(
    images: list[UploadFile] = File(..., description="마스킹 디버그에 사용할 이미지 파일 목록. 최대 5장"),
) -> MaskDebugResponse:
    image_bytes_list = await _read_upload_images(images, max_images=5)

    from slam_engines.rtabmap.person_masker import PersonMasker

    masker = PersonMasker()
    results: list[MaskDebugImage] = []
    loop = asyncio.get_running_loop()
    for index, image_bytes in enumerate(image_bytes_list):
        boxes = await loop.run_in_executor(None, functools.partial(masker.detect_boxes, image_bytes))
        annotated_b64 = _annotate_person_boxes(image_bytes, boxes)
        results.append(
            MaskDebugImage(
                index=index,
                original_b64=base64.b64encode(image_bytes).decode("ascii"),
                annotated_b64=annotated_b64,
                persons_detected=len(boxes),
            )
        )
    return MaskDebugResponse(total_images=len(results), results=results)


@router.post(
    "/v2/debug/matches",
    response_model=MatchDebugResponse,
    tags=[SLAM_TAG_DEBUG],
    summary="v2 매칭 디버그",
    description=(
        "v2 호환 매칭 디버그 API입니다. 이미지 파일 하나와 건물 ID를 받아 사람 마스킹 경로를 "
        "켠 상태로 매칭 시각화 결과를 반환합니다."
    ),
)
async def debug_matches_v2(
    image: UploadFile = File(..., description="디버그할 이미지 파일"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> MatchDebugResponse:
    return await _debug_matches_upload(
        building_id=building_id,
        map_id=map_id,
        image=image,
        mask_persons=True,
    )


@router.post(
    "/v3/debug/matches",
    response_model=MatchDebugResponse,
    tags=[SLAM_TAG_DEBUG],
    summary="v3 매칭 디버그",
    description=(
        "v3 SuperPoint 매칭 디버그 API입니다. 이미지 파일 하나와 건물 ID를 받아 활성 floor map "
        "후보 중 가장 좋은 매칭 시각화 결과를 반환합니다."
    ),
)
async def debug_matches_v3(
    image: UploadFile = File(..., description="디버그할 이미지 파일"),
    building_id: str | None = Form(None, description="건물 ID. 이 값으로 활성 floor map을 조회합니다."),
    map_id: str | None = Form(None, description="기존 클라이언트 호환용 ID. building_id와 동일하게 처리합니다."),
) -> MatchDebugResponse:
    return await _debug_matches_upload(building_id=building_id, map_id=map_id, image=image)


def _utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _session_thresholds(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "minConfidence": session["min_confidence"],
        "minMatches": session["min_matches"],
        "maxPublishAgeMs": session["max_publish_age_ms"],
    }


def _last_reliable_age_ms(session: dict[str, Any], now_ms: int | None = None) -> int | None:
    if session.get("last_reliable_at_ms") is None:
        return None
    now = _utc_ms() if now_ms is None else now_ms
    return max(0, now - int(session["last_reliable_at_ms"]))


def _parse_json_form(value: str | None, field_name: str) -> Any | None:
    if value is None or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in {field_name}: {exc}")


def _tracking_dump_dir(session_id: str) -> Path:
    import os as _os

    storage_root = Path(_os.environ.get("INDOOR_STORAGE_ROOT", "/app/var/storage"))
    dump_dir = storage_root / "debug" / "tracking_vps" / session_id
    dump_dir.mkdir(parents=True, exist_ok=True)
    return dump_dir


def _tracking_state_response(session_id: str, session: dict[str, Any]) -> TrackingSessionStateResponse:
    age_ms = _last_reliable_age_ms(session)
    status_value = "tracking" if session.get("last_reliable_pose") is not None else "uncertain"
    return TrackingSessionStateResponse(
        session_id=session_id,
        building_id=session["building_id"],
        status=status_value,
        frame_count=session["frame_count"],
        last_reliable_pose=session.get("last_reliable_pose"),
        last_candidate_pose=session.get("last_candidate_pose"),
        last_reliable_timestamp_ms=session.get("last_reliable_capture_ts_ms"),
        last_candidate_timestamp_ms=session.get("last_candidate_capture_ts_ms"),
        last_reliable_age_ms=age_ms,
        thresholds=_session_thresholds(session),
    )


@router.post(
    "/v4/tracking/sessions",
    response_model=TrackingSessionStartResponse,
    tags=[SLAM_TAG_TRACKING],
    summary="Start stateful VPS tracking session",
    description=(
        "Creates an in-memory stateful VPS session. Existing `/api/slam/v3/localize` is unchanged.\n\n"
        "Client flow:\n"
        "1. Call this once when navigation/tracking starts.\n"
        "2. For first localization, submit several `/frames` observations with RGB + depth + "
        "ARKit pose. The server accumulates robust `map_T_AR` anchor candidates from these frames.\n"
        "3. Call `/resolve` with the client's current or target AR pose. The server selects the "
        "nearest stored rolling `map_T_AR` anchor sample and computes "
        "`map_T_base = map_T_AR * AR_T_base` for that exact timestamp in RTAB-Map/map "
        "coordinates.\n"
        "4. During ongoing tracking, poll `/frames` every 1-2 seconds with the current RGB frame "
        "and, when available, aligned depth + ARKit metadata.\n"
        "5. Once enough anchors exist, `/frames` publishes the current-frame pose from the ongoing "
        "full 6DoF optimized `map_T_AR` anchor, not just the single-frame visual pose.\n"
        "6. Use `publish_pose` only when it is non-null. Otherwise keep propagating "
        "`last_reliable_pose` locally using ARKit relative motion.\n\n"
        "The server stores the last reliable pose per session and only updates it when a frame "
        "passes the configured quality gates. It also stores visual/depth/AR anchor candidates "
        "and rolling resolved `map_T_AR` anchor samples for `/resolve`."
    ),
)
async def start_tracking_session(
    request: TrackingSessionStartRequest,
) -> TrackingSessionStartResponse:
    session_id = str(uuid4())
    session = {
        "building_id": request.building_id,
        "min_confidence": request.min_confidence,
        "min_matches": request.min_matches,
        "max_publish_age_ms": request.max_publish_age_ms,
        "created_at_ms": _utc_ms(),
        "frame_count": 0,
        "last_reliable_pose": None,
        "last_reliable_at_ms": None,
        "last_reliable_capture_ts_ms": None,
        "last_reliable_ar": None,
        "map_from_ar_anchor": None,
        "initial_anchor_candidates": [],
        "reanchor_candidates": [],
        "resolved_anchor_samples": [],
        "last_candidate_pose": None,
        "last_candidate_capture_ts_ms": None,
        "last_quality": None,
        "latest_frame_timestamp_ms": None,
    }
    with _tracking_sessions_lock:
        _tracking_sessions[session_id] = session
    return TrackingSessionStartResponse(
        session_id=session_id,
        building_id=request.building_id,
        status="tracking",
        thresholds=_session_thresholds(session),
    )


@router.post(
    "/v4/tracking/sessions/{session_id}/frames",
    response_model=TrackingFrameResponse,
    tags=[SLAM_TAG_TRACKING],
    summary="Submit one RGB/depth tracking frame",
    description=(
        "Submits one client tracking observation. This endpoint is designed for 1-2 second polling, "
        "not for uploading a burst of images.\n\n"
        "Coordinate convention:\n"
        "- `candidate_pose`, `publish_pose`, and `last_reliable_pose` are RTAB-Map/map-frame poses, "
        "same convention as v3 localize.\n"
        "- `ar_pose` is raw client-local ARKit camera input. The server converts ARKit camera axes "
        "to OpenCV optical with `diag(1,-1,-1)`, then to RTAB-Map `base_link` using the same "
        "CameraModel optical rotation stored in the scan DB before relative motion, jump rejection, "
        "near-threshold promotion, and map_T_AR anchor estimation. It is never returned directly as "
        "a map pose.\n\n"
        "How the client should use the response:\n"
        "- If `publish_pose` is non-null, anchor/correct the client map transform at "
        "`publish_timestamp_ms`.\n"
        "- `quality.publishPoseSource` tells whether the pose came from the ongoing optimized "
        "`map_T_AR` anchor (`optimized_map_T_ar_anchor`) or from single-frame visual/depth "
        "localization (`single_frame_visual_depth`).\n"
        "- `quality.resolvedPose` shows the current-frame pose predicted by the optimized anchor "
        "when enough anchor frames exist.\n"
        "- If `publish_pose` is null but `last_reliable_pose` exists, continue tracking from "
        "`last_reliable_timestamp_ms` using local ARKit deltas.\n"
        "- If both are null, do not move the user to `candidate_pose`; show tracking/uncertain state.\n\n"
        "When the user moves far enough that the current anchor becomes stale, high-quality frames "
        "that fail `ar_prior_inconsistent` are accumulated as re-anchor candidates. If a separate "
        "cluster reaches enough visual/depth support, the server publishes `reanchored_map_T_ar_anchor` "
        "and replaces the stale anchor. Session state is updated by client `timestamp_ms`, not by "
        "server processing order, so late out-of-order frames are stored/debugged but cannot overwrite "
        "newer reliable poses or anchor samples.\n\n"
        "Depth is used as a metric consistency/refinement signal when valid matched feature depths "
        "are available; the applied correction and residuals are exposed in `quality.depthFusion`. "
        "For first localization, keep submitting frames until `/resolve` returns `localized`."
    ),
)
async def submit_tracking_frame(
    session_id: str,
    image: UploadFile = File(
        ...,
        description=(
            "Current RGB frame. Send the same orientation/resolution convention as v3 localize. "
            "Prefer the same lens used by the scan map; include `lens` and intrinsics when possible."
        ),
    ),
    depth: UploadFile | None = File(
        None,
        description=(
            "Optional depth frame captured at the same timestamp as `image`. Preferred format is "
            "aligned-to-RGB depth, e.g. uint16 millimeters PNG or float32 depth PNG. If not aligned, "
            "the client must send enough metadata in `depth_intrinsics`/`ar_pose` for later fusion."
        ),
    ),
    camera_intrinsics: str | None = Form(
        None,
        description=(
            "Optional JSON for the RGB image after orientation/resize, e.g. "
            "`{\"width\":1920,\"height\":1440,\"fx\":1333.9,\"fy\":1333.9,\"cx\":969.2,\"cy\":718.7}`."
        ),
    ),
    depth_intrinsics: str | None = Form(
        None,
        description=(
            "Optional JSON for depth, e.g. "
            "`{\"width\":256,\"height\":192,\"fx\":177.8,\"fy\":177.8,\"cx\":128,\"cy\":96,"
            "\"alignedToColor\":true,\"unit\":\"m\"}`."
        ),
    ),
    ar_pose: str | None = Form(
        None,
        description=(
            "Optional JSON ARKit pose for the same capture timestamp. Recommended shape: "
            "`{\"world_T_camera\":[16 row-major floats],\"trackingState\":\"normal\","
            "\"gravity\":[x,y,z]}`. The server uses this as local motion input, not as map coordinates."
        ),
    ),
    lens: str | None = Form(
        None,
        description="Optional lens identifier such as `1x`, `0.5x`, `wide`, or `ultra-wide`.",
    ),
    timestamp_ms: int | None = Form(
        None,
        description=(
            "Client capture timestamp in milliseconds for image/depth/ar_pose. This is returned as "
            "`frame_timestamp_ms`, and if accepted, `publish_timestamp_ms`. Required for the client "
            "to apply ARKit deltas from the published frame to the current render frame."
        ),
    ),
) -> TrackingFrameResponse:
    with _tracking_sessions_lock:
        session = _tracking_sessions.get(session_id)
        if session is not None:
            frame_index = int(session["frame_count"])
            session["frame_count"] = frame_index + 1
    if session is None:
        raise HTTPException(status_code=404, detail=f"Tracking session not found: {session_id}")

    rgb_intrinsics = _parse_json_form(camera_intrinsics, "camera_intrinsics")
    parsed_depth_intrinsics = _parse_json_form(depth_intrinsics, "depth_intrinsics")
    parsed_ar_pose = _parse_json_form(ar_pose, "ar_pose")
    parsed_ar_transform_raw = _parse_ar_transform(parsed_ar_pose)
    parsed_ar_transform = (
        _convert_ar_camera_to_rtab_camera(parsed_ar_transform_raw)
        if parsed_ar_transform_raw is not None
        else None
    )

    server_received_ms = _utc_ms()
    image_bytes = await image.read()
    depth_bytes = await depth.read() if depth is not None else b""
    depth_received = bool(depth_bytes)

    candidate_pose = None
    quality: dict[str, Any] = {
        "timestampMs": timestamp_ms,
        "lens": lens,
        "depthReceived": depth_received,
        "depthBytes": len(depth_bytes),
        "cameraIntrinsicsReceived": rgb_intrinsics is not None,
        "depthIntrinsicsReceived": parsed_depth_intrinsics is not None,
        "arPoseReceived": parsed_ar_pose is not None,
        "arPoseTransformValid": parsed_ar_transform is not None,
        "arTrackingState": None if parsed_ar_transform is None else parsed_ar_transform["trackingState"],
        "arCameraFrame": "arkit_camera_raw",
        "serverCameraFrame": "rtabmap_base_link",
        "arCameraToServerCamera": "ARKit camera -> OpenCV optical diag(1,-1,-1) -> RTAB base_link via DB opticalRotation",
        "depthFusionApplied": False,
        "arPredictionApplied": False,
        "accepted": False,
        "rejectReasons": [],
    }
    latest_frame_timestamp_ms = session.get("latest_frame_timestamp_ms")
    out_of_order_lag_ms = (
        int(latest_frame_timestamp_ms) - int(timestamp_ms)
        if latest_frame_timestamp_ms is not None and timestamp_ms is not None
        else None
    )
    state_update_stale = _is_stale_for_state_update(session, timestamp_ms)
    quality.update(
        {
            "latestFrameTimestampMsBefore": latest_frame_timestamp_ms,
            "outOfOrderLagMs": out_of_order_lag_ms,
            "stateUpdateStale": state_update_stale,
        }
    )
    dump_base: Path | None = None
    try:
        dump_dir = _tracking_dump_dir(session_id)
        dump_base = dump_dir / f"{frame_index:06d}_{server_received_ms}"
        image_suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
        (dump_base.with_name(dump_base.name + f"_image{image_suffix}")).write_bytes(image_bytes)
        if depth is not None:
            depth_suffix = Path(depth.filename or "depth.bin").suffix or ".bin"
            (dump_base.with_name(dump_base.name + f"_depth{depth_suffix}")).write_bytes(depth_bytes)
        request_meta = {
            "sessionId": session_id,
            "frameIndex": frame_index,
            "serverReceivedMs": server_received_ms,
            "timestampMs": timestamp_ms,
            "image": {
                "filename": image.filename,
                "contentType": image.content_type,
                "bytes": len(image_bytes),
            },
            "depth": None
            if depth is None
            else {
                "filename": depth.filename,
                "contentType": depth.content_type,
                "bytes": len(depth_bytes),
            },
            "lens": lens,
            "cameraIntrinsicsRaw": camera_intrinsics,
            "cameraIntrinsics": rgb_intrinsics,
            "depthIntrinsicsRaw": depth_intrinsics,
            "depthIntrinsics": parsed_depth_intrinsics,
            "arPoseRaw": ar_pose,
            "arPose": parsed_ar_pose,
            "buildingId": session["building_id"],
        }
        (dump_base.with_name(dump_base.name + "_request.json")).write_text(
            json.dumps(request_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        quality["debugDumpBase"] = str(dump_base)
    except Exception as exc:
        logger.warning("[TRACKING-VPS] request dump failed session=%s frame=%d: %s", session_id, frame_index, exc)
        quality["debugDumpError"] = str(exc)

    try:
        result = await _localize_uploads(
            building_id=session["building_id"],
            map_id=None,
            images=[],
            image_bytes_override=[image_bytes],
            query_depths=[
                {
                    "bytes": depth_bytes,
                    "filename": depth.filename if depth is not None else None,
                    "content_type": depth.content_type if depth is not None else None,
                    "depth_intrinsics": parsed_depth_intrinsics,
                    "camera_intrinsics": rgb_intrinsics,
                    "lens": lens,
                }
            ] if depth_received else None,
            query_intrinsics=rgb_intrinsics,
            mask_persons=False,
        )
        candidate_pose = result.pose
        depth_fusion = (result.debug or {}).get("depthFusion")
        if depth_fusion is not None:
            quality.update(
                {
                    "depthFusion": depth_fusion,
                    "depthFusionApplied": bool(depth_fusion.get("applied")),
                    "depthFusionMode": depth_fusion.get("mode"),
                    "depthValidMatches": depth_fusion.get("validMatches"),
                    "depthResidualMedianM": depth_fusion.get("residualMedianM"),
                    "depthResidualMadM": depth_fusion.get("residualMadM"),
                    "depthCorrectionM": depth_fusion.get("correctionM"),
                    "depth3dPoseUsed": depth_fusion.get("depth3dPoseUsed"),
                    "depth3dApplied": depth_fusion.get("depth3dApplied"),
                    "depth3dTranslationDeltaM": depth_fusion.get("translationDeltaM"),
                }
            )
        quality.update(
            {
                "localizeStatus": "ok",
                "confidence": result.confidence,
                "numMatches": result.numMatches,
                "matchedImageIndex": result.matchedImageIndex,
                "floorId": result.floorId,
                "floorLevel": result.floorLevel,
            }
        )
    except HTTPException as exc:
        quality.update({"localizeStatus": "failed", "error": exc.detail})
        quality["rejectReasons"].append("localize_failed")
    except Exception as exc:
        quality.update({"localizeStatus": "failed", "error": str(exc)})
        quality["rejectReasons"].append("localize_failed")

    ar_prediction: dict[str, Any] | None = None
    ar_normal = (
        parsed_ar_transform is not None
        and str(parsed_ar_transform.get("trackingState") or "").lower() == "normal"
    )
    anchor_candidate: dict[str, Any] | None = None
    anchor_score: float | None = None
    fixed_depth_error = 999.0
    if candidate_pose is not None and parsed_ar_transform is not None:
        fixed_depth_error = float(
            (((quality.get("depthFusion") or {}).get("depthPose") or {}).get("fixedRotationMedianErrorM") or 999.0)
        )
        if (
            ar_normal
            and float(quality.get("confidence", 0.0)) >= 0.58
            and int(quality.get("numMatches", 0)) >= 80
            and bool(quality.get("depth3dPoseUsed"))
            and fixed_depth_error <= 0.15
        ):
            anchor_candidate = _map_anchor_from_pose_and_ar(candidate_pose, parsed_ar_transform)
            anchor_score = min(0.99, float(quality.get("confidence", 0.0))) * min(
                1.0, int(quality.get("numMatches", 0)) / 240.0
            ) * max(0.1, 1.0 - min(1.0, fixed_depth_error / 0.15))
            anchor_record = {
                "frameIndex": frame_index,
                "timestampMs": timestamp_ms,
                "anchor": anchor_candidate,
                "score": anchor_score,
                "confidence": quality.get("confidence"),
                "numMatches": quality.get("numMatches"),
                "depthFixedErrorM": fixed_depth_error,
            }
            initial_candidates = list(session.get("initial_anchor_candidates") or [])
            initial_candidates.append(anchor_record)
            session["initial_anchor_candidates"] = initial_candidates[-20:]
            quality["initialAnchorCandidateAccepted"] = True
            quality["initialAnchorCandidateCount"] = len(session["initial_anchor_candidates"])

        anchor = session.get("map_from_ar_anchor")
        last_ar = session.get("last_reliable_ar")
        if anchor is not None and last_ar is not None:
            predicted = _predict_map_pose_from_ar(anchor, parsed_ar_transform)
            candidate_t = _pose_translation(candidate_pose)
            predicted_t = predicted["t"]
            last_pose = session.get("last_reliable_pose")
            last_pose_t = _pose_translation(last_pose) if last_pose is not None else predicted_t
            last_ar_t = last_ar.get("t") or parsed_ar_transform["t"]
            ar_delta_m = _vec_norm(_vec_sub(parsed_ar_transform["t"], last_ar_t))
            candidate_delta_m = _vec_norm(_vec_sub(candidate_t, last_pose_t))
            prediction_error_m = _vec_norm(_vec_sub(candidate_t, predicted_t))
            predicted_yaw = _yaw_from_matrix(predicted["R"])
            candidate_yaw = _yaw_from_matrix(_pose_quat_to_matrix(candidate_pose))
            yaw_error_deg = abs(_angle_diff_deg(candidate_yaw, predicted_yaw))
            ar_prediction = {
                "trackingState": parsed_ar_transform["trackingState"],
                "normal": ar_normal,
                "arDeltaM": ar_delta_m,
                "candidateDeltaM": candidate_delta_m,
                "predictionErrorM": prediction_error_m,
                "candidateYawDeg": candidate_yaw,
                "predictedYawDeg": predicted_yaw,
                "yawErrorDeg": yaw_error_deg,
            }
            ar_consistent = ar_normal and prediction_error_m <= max(0.45, ar_delta_m + 0.40) and yaw_error_deg <= 75.0
            ar_prediction["consistent"] = ar_consistent
            ar_prediction["applied"] = True
            quality.update(
                {
                    "arPredictionApplied": True,
                    "arPrediction": ar_prediction,
                    "arDeltaM": ar_delta_m,
                    "candidateDeltaFromLastM": candidate_delta_m,
                    "arPredictionErrorM": prediction_error_m,
                    "arYawErrorDeg": yaw_error_deg,
                    "arPriorConsistent": ar_consistent,
                }
            )

    resolved_pose = None
    resolved_anchor_diagnostics: dict[str, Any] | None = None
    resolved_anchor_for_current_frame: dict[str, Any] | None = None
    resolved_anchor_source = "optimized_map_T_ar_anchor"
    if parsed_ar_transform is not None and ar_normal:
        resolve_candidates = list(session.get("initial_anchor_candidates") or [])
        if session.get("map_from_ar_anchor") is not None:
            resolve_candidates.append(
                {
                    "frameIndex": "lastReliable",
                    "anchor": session["map_from_ar_anchor"],
                    "score": 0.95,
                }
            )
        resolved_anchor, resolved_anchor_diagnostics = _resolve_anchor_from_candidates(
            resolve_candidates,
            min_anchor_frames=2,
        )
        if resolved_anchor is not None:
            predicted = _predict_map_pose_from_ar(resolved_anchor, parsed_ar_transform)
            resolved_pose = _pose_from_matrix_translation(predicted["R"], predicted["t"])
            resolved_anchor_for_current_frame = resolved_anchor
            quality.update(
                {
                    "resolvedPoseAvailable": True,
                    "resolvedPose": resolved_pose,
                    "resolvedPoseSource": "optimized_map_T_ar_anchor",
                    "resolvedAnchor": resolved_anchor_diagnostics,
                    "resolvedAnchorSampleStored": False,
                    "resolvedAnchorSampleCount": len(session.get("resolved_anchor_samples") or []),
                }
            )
        elif resolved_anchor_diagnostics is not None:
            quality.update(
                {
                    "resolvedPoseAvailable": False,
                    "resolvedAnchor": resolved_anchor_diagnostics,
                }
            )

    if candidate_pose is not None:
        if float(quality.get("confidence", 0.0)) < float(session["min_confidence"]):
            quality["rejectReasons"].append("confidence_below_threshold")
        if int(quality.get("numMatches", 0)) < int(session["min_matches"]):
            quality["rejectReasons"].append("matches_below_threshold")
        if depth is not None and not depth_received:
            quality["rejectReasons"].append("empty_depth_payload")
        if ar_prediction is not None and ar_normal and not ar_prediction.get("consistent"):
            quality["rejectReasons"].append("ar_prior_inconsistent")

        if (
            ar_prediction is not None
            and ar_prediction.get("consistent")
            and "confidence_below_threshold" in quality["rejectReasons"]
            and float(quality.get("confidence", 0.0)) >= 0.58
            and int(quality.get("numMatches", 0)) >= int(session["min_matches"])
            and bool(quality.get("depth3dPoseUsed"))
            and float((((quality.get("depthFusion") or {}).get("depthPose") or {}).get("fixedRotationMedianErrorM") or 999.0)) <= 0.12
        ):
            quality["rejectReasons"] = [
                r for r in quality["rejectReasons"] if r != "confidence_below_threshold"
            ]
            quality["arPriorPromoted"] = True

        if (
            "ar_prior_inconsistent" in quality["rejectReasons"]
            and anchor_candidate is not None
            and anchor_score is not None
            and ar_normal
            and not state_update_stale
        ):
            reanchor_record = {
                "frameIndex": frame_index,
                "timestampMs": timestamp_ms,
                "anchor": anchor_candidate,
                "score": anchor_score,
                "confidence": quality.get("confidence"),
                "numMatches": quality.get("numMatches"),
                "depthFixedErrorM": fixed_depth_error,
            }
            reanchor_candidates = list(session.get("reanchor_candidates") or [])
            reanchor_candidates.append(reanchor_record)
            session["reanchor_candidates"] = reanchor_candidates[-30:]
            quality["reanchorCandidateAccepted"] = True
            quality["reanchorCandidateCount"] = len(session["reanchor_candidates"])

            reanchor_anchor, reanchor_diagnostics = _resolve_anchor_from_candidates(
                session["reanchor_candidates"],
                min_anchor_frames=3,
            )
            quality["reanchor"] = reanchor_diagnostics
            if (
                reanchor_anchor is not None
                and _anchor_diagnostics_include_frame(reanchor_diagnostics, frame_index)
                and int((reanchor_diagnostics.get("optimizer") or {}).get("optimizerInlierFrames") or 0) >= 2
            ):
                predicted = _predict_map_pose_from_ar(reanchor_anchor, parsed_ar_transform)
                resolved_pose = _pose_from_matrix_translation(predicted["R"], predicted["t"])
                resolved_anchor_for_current_frame = reanchor_anchor
                resolved_anchor_diagnostics = reanchor_diagnostics
                resolved_anchor_source = "reanchored_map_T_ar_anchor"
                quality.update(
                    {
                        "resolvedPoseAvailable": True,
                        "resolvedPose": resolved_pose,
                        "resolvedPoseSource": resolved_anchor_source,
                        "resolvedAnchor": resolved_anchor_diagnostics,
                        "reanchorApplied": True,
                    }
                )
                quality["rejectReasons"] = [
                    reason for reason in quality["rejectReasons"] if reason != "ar_prior_inconsistent"
                ]

    if (
        resolved_anchor_for_current_frame is not None
        and resolved_anchor_diagnostics is not None
        and bool(quality.get("initialAnchorCandidateAccepted"))
        and _anchor_diagnostics_include_frame(resolved_anchor_diagnostics, frame_index)
        and not quality["rejectReasons"]
        and not state_update_stale
    ):
        resolved_sample = _append_resolved_anchor_sample(
            session,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            server_received_ms=server_received_ms,
            anchor=resolved_anchor_for_current_frame,
            diagnostics=resolved_anchor_diagnostics,
            source=(
                "rolling_reanchor_visual_depth_anchor"
                if resolved_anchor_source == "reanchored_map_T_ar_anchor"
                else "rolling_visual_depth_anchor"
            ),
        )
        quality.update(
            {
                "resolvedAnchorSampleStored": True,
                "resolvedAnchorSampleTimestampMs": resolved_sample.get("timestampMs"),
                "resolvedAnchorSampleCount": len(session.get("resolved_anchor_samples") or []),
                "resolvedAnchorSampleSource": resolved_sample.get("source"),
            }
        )

    now_ms = _utc_ms()
    publish_pose = None
    can_update_reliable_state = _is_newer_timestamp(
        timestamp_ms,
        session.get("last_reliable_capture_ts_ms"),
    ) and not state_update_stale
    if candidate_pose is not None and not quality["rejectReasons"] and can_update_reliable_state:
        publish_pose = resolved_pose or candidate_pose
        quality["accepted"] = True
        quality["publishPoseSource"] = (
            resolved_anchor_source if resolved_pose is not None else "single_frame_visual_depth"
        )
        session["last_reliable_pose"] = publish_pose
        session["last_reliable_at_ms"] = now_ms
        session["last_reliable_capture_ts_ms"] = timestamp_ms
        if parsed_ar_transform is not None and ar_normal:
            session["last_reliable_ar"] = parsed_ar_transform
            if resolved_pose is not None and resolved_anchor_diagnostics is not None:
                session["map_from_ar_anchor"] = resolved_anchor_for_current_frame or _map_anchor_from_pose_and_ar(
                    publish_pose,
                    parsed_ar_transform,
                )
                if resolved_anchor_source == "reanchored_map_T_ar_anchor":
                    session["initial_anchor_candidates"] = list(session.get("reanchor_candidates") or [])[-20:]
                    session["reanchor_candidates"] = []
            else:
                session["map_from_ar_anchor"] = _map_anchor_from_pose_and_ar(publish_pose, parsed_ar_transform)
    elif candidate_pose is not None and not quality["rejectReasons"]:
        quality["accepted"] = False
        quality["stateUpdateSkipped"] = True
        quality["stateUpdateSkipReason"] = "out_of_order_or_stale_timestamp"
        quality["rejectReasons"].append("out_of_order_or_stale_timestamp")

    previous_latest_ts = session.get("latest_frame_timestamp_ms")
    if _is_newer_timestamp(timestamp_ms, previous_latest_ts):
        session["latest_frame_timestamp_ms"] = timestamp_ms

    session["frame_count"] = max(int(session.get("frame_count") or 0), frame_index + 1)
    if _is_newer_timestamp(timestamp_ms, session.get("last_candidate_capture_ts_ms")):
        session["last_candidate_pose"] = candidate_pose
        session["last_candidate_capture_ts_ms"] = timestamp_ms
        session["last_quality"] = quality
    with _tracking_sessions_lock:
        _tracking_sessions[session_id] = session

    last_reliable_age = _last_reliable_age_ms(session, now_ms)
    if publish_pose is not None:
        status_value = "localized"
    elif session.get("last_reliable_pose") is not None:
        status_value = "tracking"
    else:
        status_value = "uncertain"
    publish_timestamp_ms = timestamp_ms if publish_pose is not None else None

    logger.info(
        "[TRACKING-VPS] session=%s frame=%d status=%s accepted=%s conf=%s matches=%s "
        "reasons=%s publish_ts=%s last_ts=%s depth=%s depth_fusion=%s depth_corr=%s ar=%s",
        session_id,
        frame_index,
        status_value,
        quality.get("accepted"),
        quality.get("confidence"),
        quality.get("numMatches"),
        quality.get("rejectReasons"),
        publish_timestamp_ms,
        session.get("last_reliable_capture_ts_ms"),
        quality.get("depthReceived"),
        quality.get("depthFusionApplied"),
        quality.get("depthCorrectionM"),
        quality.get("arPoseReceived"),
    )

    if dump_base is not None:
        response_meta = {
            "sessionId": session_id,
            "frameIndex": frame_index,
            "status": status_value,
            "publishPose": publish_pose,
            "lastReliablePose": session.get("last_reliable_pose"),
            "candidatePose": candidate_pose,
            "quality": quality,
            "frameTimestampMs": timestamp_ms,
            "serverReceivedMs": server_received_ms,
            "publishTimestampMs": publish_timestamp_ms,
            "lastReliableTimestampMs": session.get("last_reliable_capture_ts_ms"),
            "lastReliableAgeMs": last_reliable_age,
            "latestFrameTimestampMs": session.get("latest_frame_timestamp_ms"),
        }
        try:
            (dump_base.with_name(dump_base.name + "_response.json")).write_text(
                json.dumps(response_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[TRACKING-VPS] response dump failed session=%s frame=%d: %s", session_id, frame_index, exc)

    return TrackingFrameResponse(
        session_id=session_id,
        status=status_value,
        publish_pose=publish_pose,
        last_reliable_pose=session.get("last_reliable_pose"),
        candidate_pose=candidate_pose,
        quality=quality,
        frame_index=frame_index,
        frame_timestamp_ms=timestamp_ms,
        server_received_ms=server_received_ms,
        publish_timestamp_ms=publish_timestamp_ms,
        last_reliable_timestamp_ms=session.get("last_reliable_capture_ts_ms"),
        last_reliable_age_ms=last_reliable_age,
    )


@router.get(
    "/v4/tracking/sessions/{session_id}",
    response_model=TrackingSessionStateResponse,
    tags=[SLAM_TAG_TRACKING],
    summary="Get tracking session state",
    description=(
        "Returns the session's last reliable map pose and its client capture timestamp. "
        "Use this endpoint when the client wants to recover state after missed frame responses."
    ),
)
async def get_tracking_session_state(session_id: str) -> TrackingSessionStateResponse:
    with _tracking_sessions_lock:
        session = _tracking_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Tracking session not found: {session_id}")
    return _tracking_state_response(session_id, session)


@router.post(
    "/v4/tracking/sessions/{session_id}/resolve",
    response_model=TrackingResolveResponse,
    tags=[SLAM_TAG_TRACKING],
    summary="Resolve a target AR pose through stored rolling map_T_AR anchors",
    description=(
        "Resolves the client's current AR pose into RTAB-Map/map coordinates using visual/depth "
        "anchors accumulated from prior `/frames` calls in this session.\n\n"
        "This is intended for first localization and ongoing timestamped pose correction. "
        "The server does not average camera poses directly. "
        "For each good frame it estimates `map_T_AR = map_T_base * inverse(AR_T_base)`, robustly "
        "clusters those anchors, refines the selected anchor with a full 6DoF robust SE(3) optimizer "
        "over translation and rotation residuals, and stores the result as an anchor sample. "
        "Each successful `/frames` call stores rolling `map_T_AR` anchor samples. `/resolve` is a "
        "timestamped lookup/conversion API: it selects the nearest stored anchor sample to "
        "`timestamp_ms`, then computes `map_T_base = map_T_AR * AR_T_base` for the `ar_pose` "
        "supplied in this request. It does not return a stored camera pose directly. Therefore the "
        "returned pose corresponds to `pose_timestamp_ms` and remains RTAB-Map/map-frame.\n\n"
        "Client usage:\n"
        "1. Start a v4 tracking session.\n"
        "2. Submit 2-5 initialization frames to `/frames` with RGB, depth, `camera_intrinsics`, "
        "`depth_intrinsics`, `ar_pose`, `lens`, and `timestamp_ms`.\n"
        "3. Call this endpoint with the current AR pose and timestamp.\n"
        "4. If `status=localized`, use `pose` as the RTAB-Map pose for `pose_timestamp_ms` and "
        "initialize the client's map-to-AR transform from that timestamp.\n"
        "5. If `status=tracking_only`, the server returned anchor-propagated pose from an older "
        "sample; use it only for continuity, not as a fresh VPS correction.\n"
        "6. If `status=initializing`, inspect `anchor.reason`, continue sending frames, and retry.\n\n"
        "`min_anchor_frames` controls how many consistent visual/depth frames must agree before "
        "the server resolves. Use `2` or `3` for normal first localization."
    ),
)
async def resolve_tracking_pose(
    session_id: str,
    request: TrackingResolveRequest,
) -> TrackingResolveResponse:
    with _tracking_sessions_lock:
        session = _tracking_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Tracking session not found: {session_id}")

    ar_transform = _parse_ar_transform(request.ar_pose)
    if ar_transform is None:
        raise HTTPException(status_code=422, detail="Invalid ar_pose.world_T_camera")
    ar_transform = _convert_ar_camera_to_rtab_camera(ar_transform)
    if ar_transform["trackingState"].lower() != "normal":
        return TrackingResolveResponse(
            session_id=session_id,
            status="initializing",
            pose=None,
            pose_timestamp_ms=request.timestamp_ms,
            anchor={
                "reason": "ar_tracking_not_normal",
                "trackingState": ar_transform["trackingState"],
                "sourceFrames": len(session.get("initial_anchor_candidates") or []),
                "inlierFrames": 0,
            },
        )

    eligible_samples = [
        s for s in list(session.get("resolved_anchor_samples") or [])
        if int(s.get("inlierFrames") or 0) >= int(request.min_anchor_frames)
    ]
    sample, timestamp_distance_ms = _select_resolved_anchor_sample(
        eligible_samples,
        target_timestamp_ms=request.timestamp_ms,
    )
    if sample is None:
        return TrackingResolveResponse(
            session_id=session_id,
            status="initializing",
            pose=None,
            pose_timestamp_ms=request.timestamp_ms,
            anchor={
                "reason": "no_resolved_anchor_samples",
                "sourceFrames": len(session.get("initial_anchor_candidates") or []),
                "inlierFrames": 0,
                "sampleCount": len(session.get("resolved_anchor_samples") or []),
                "eligibleSampleCount": 0,
                "requiredInlierFrames": int(request.min_anchor_frames),
            },
        )

    anchor = sample["anchor"]
    diagnostics = dict(sample.get("diagnostics") or {})
    predicted = _predict_map_pose_from_ar(anchor, ar_transform)
    pose = _pose_from_matrix_translation(predicted["R"], predicted["t"])
    diagnostics["trackingState"] = ar_transform["trackingState"]
    diagnostics["poseSource"] = "stored_map_T_ar_anchor"
    diagnostics["sampleFrameIndex"] = sample.get("frameIndex")
    diagnostics["sampleTimestampMs"] = sample.get("timestampMs")
    diagnostics["sampleServerReceivedMs"] = sample.get("serverReceivedMs")
    diagnostics["sampleTimestampDistanceMs"] = timestamp_distance_ms
    diagnostics["sampleCount"] = len(session.get("resolved_anchor_samples") or [])
    diagnostics["eligibleSampleCount"] = len(eligible_samples)
    diagnostics["requestedPoseTimestampMs"] = request.timestamp_ms
    diagnostics["maxAnchorAgeMs"] = request.max_anchor_age_ms
    status = "localized"
    if (
        request.max_anchor_age_ms is not None
        and timestamp_distance_ms is not None
        and timestamp_distance_ms > int(request.max_anchor_age_ms)
    ):
        status = "tracking_only"
        diagnostics["reason"] = "nearest_anchor_sample_outside_freshness_window"
    return TrackingResolveResponse(
        session_id=session_id,
        status=status,
        pose=pose,
        pose_timestamp_ms=request.timestamp_ms,
        anchor=diagnostics,
    )


@router.delete(
    "/v4/tracking/sessions/{session_id}",
    tags=[SLAM_TAG_TRACKING],
    summary="End tracking session",
    description="Ends the in-memory tracking session and drops last reliable/candidate state.",
)
async def end_tracking_session(session_id: str) -> dict[str, Any]:
    with _tracking_sessions_lock:
        removed = _tracking_sessions.pop(session_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Tracking session not found: {session_id}")
    return {"session_id": session_id, "status": "ended"}


async def _localize_uploads(
    *,
    building_id: str | None,
    map_id: str | None,
    images: list[UploadFile],
    image_bytes_override: list[bytes] | None = None,
    query_depths: list[dict[str, Any]] | None = None,
    query_intrinsics: dict[str, Any] | None = None,
    mask_persons: bool = False,
    floor_id: str | None = None,
) -> SLAMLocalizeResponse:
    resolved_building_id = _coerce_building_id(building_id, map_id)
    image_bytes_list = image_bytes_override or await _read_upload_images(images)

    # Query image debug dump (사용자 앱 메인 endpoint 의 query 보존)
    try:
        import datetime as _dt
        from pathlib import Path as _Path
        import os as _os
        storage_root = _Path(_os.environ.get("INDOOR_STORAGE_ROOT", "/app/var/storage"))
        dump_dir = storage_root / "debug" / "localize" / str(resolved_building_id)
        dump_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
        for i, b in enumerate(image_bytes_list):
            (dump_dir / f"{ts}_{i:02d}.jpg").write_bytes(b)
        logger.info("v3 localize query dump: %s/ %d images", dump_dir, len(image_bytes_list))
    except Exception as _dump_exc:
        logger.warning("v3 localize query dump 실패: %s", _dump_exc)

    return await _localize_bytes(
        building_id=resolved_building_id,
        image_bytes_list=image_bytes_list,
        query_depths=query_depths,
        query_intrinsics=query_intrinsics,
        mask_persons=mask_persons,
        floor_id=floor_id,
    )


async def _localize_impl(
    request: SLAMLocalizeRequest,
    mask_persons: bool = False,
    engine=None,
    floor_id: str | None = None,
) -> SLAMLocalizeResponse:
    """Compatibility core for internal callers that still pass base64 JSON."""

    image_bytes_list = []
    for index, img_b64 in enumerate(request.images):
        try:
            image_bytes_list.append(base64.b64decode(img_b64))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in image {index + 1}: {exc}")

    return await _localize_bytes(
        building_id=request.map_id,
        image_bytes_list=image_bytes_list,
        mask_persons=mask_persons,
        engine=engine,
        floor_id=floor_id,
    )


async def _localize_bytes(
    *,
    building_id: str,
    image_bytes_list: list[bytes],
    query_depths: list[dict[str, Any]] | None = None,
    query_intrinsics: dict[str, Any] | None = None,
    mask_persons: bool = False,
    engine=None,
    floor_id: str | None = None,
) -> SLAMLocalizeResponse:
    logger.info(
        f"[SLAM-LOCALIZE] building_id: {building_id}, mask_persons: {mask_persons}, "
        f"floor_id (incoming, ignored): {floor_id}"
    )
    # 임시: 클라가 보내는 floor_id 는 무시하고 모든 활성 floor 후보로 검색.
    # (iOS 가 보내는 값이 UUID 가 아니라 LEVEL 숫자 ('-3') 라 매칭 실패하던 문제 회피.)
    floor_id = None

    if not image_bytes_list:
        raise HTTPException(status_code=422, detail="At least one image file is required")

    floor_maps = await _get_floor_maps(building_id)
    if not floor_maps:
        single_db = settings.MAPS_DIR / f"{building_id}.db"
        if single_db.exists():
            floor_maps = [
                {
                    "floor_id": "",
                    "floor_name": "",
                    "level": 0,
                    "file_path": str(single_db),
                }
            ]
        else:
            raise HTTPException(status_code=404, detail=f"No maps found for building {building_id}")

    floor_maps = _filter_floor_maps(floor_maps, floor_id=floor_id, building_id=building_id)

    slam_engine = engine
    if slam_engine is None:
        slam_engine = await asyncio.get_running_loop().run_in_executor(None, _get_sp_engine)
    resolved_floors = [_resolve_floor_path(fm) for fm in floor_maps]

    intrinsics = None
    for fm in resolved_floors:
        try:
            intrinsics = slam_engine.extract_intrinsics_from_db(fm["file_path"])
            break
        except Exception:
            continue

    if intrinsics is None:
        raise HTTPException(status_code=500, detail="Failed to extract intrinsics from any floor DB")

    image_bytes_list = _resize_query_images(
        image_bytes_list,
        width=intrinsics["width"],
        height=intrinsics["height"],
    )
    if query_intrinsics is not None:
        scaled_query_intrinsics = _scale_query_intrinsics(
            query_intrinsics,
            width=int(intrinsics["width"]),
            height=int(intrinsics["height"]),
        )
        if scaled_query_intrinsics is not None:
            intrinsics = scaled_query_intrinsics

    async def _localize_floor(fm: dict) -> dict | None:
        try:
            result = await slam_engine.localize(
                fm["floor_id"] or building_id,
                image_bytes_list,
                intrinsics=intrinsics,
                db_path=fm["file_path"],
                mask_persons=mask_persons,
                query_depths=query_depths,
            )
            result["floor_id"] = fm["floor_id"]
            result["floor_name"] = fm["floor_name"]
            result["floor_level"] = fm["level"]
            return result
        except (FileNotFoundError, ValueError) as exc:
            logger.debug(f"[SLAM-LOCALIZE] Floor {fm['floor_name']}: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"[SLAM-LOCALIZE] Floor {fm['floor_name']} error: {exc}")
            return None

    results = await asyncio.gather(*[_localize_floor(fm) for fm in resolved_floors])
    valid = [r for r in results if r is not None]

    if not valid:
        raise HTTPException(status_code=503, detail="Localization failed on all floors")

    # 우선순위: inlier 절대값 > confidence (작은 graph 의 비율 기반 false positive 방지).
    # 작은 graph (예: 4 keyframe) 가 inliers 6 인데도 confidence 비율이 높아 큰 graph 보다
    # 잘못 선택되는 문제 — 큰 graph 의 inliers 많은 매칭이 더 신뢰할 만.
    best = max(valid, key=lambda r: (r.get("num_matches", 0), r["confidence"]))

    logger.info(
        f"[SLAM-LOCALIZE] Best: floor={best.get('floor_name')}, "
        f"confidence={best['confidence']:.2f}, matches={best.get('num_matches', 0)}"
    )

    return SLAMLocalizeResponse(
        pose=best["pose"],
        confidence=best["confidence"],
        mapId=building_id,
        numMatches=best.get("num_matches", 0),
        matchedImageIndex=best.get("matched_image_index", 0),
        floorId=best.get("floor_id", ""),
        floorLevel=best.get("floor_level", 0),
        debug={
            "matchedNodeId": best.get("matched_node_id"),
            "depthFusion": best.get("depth_fusion"),
            "rtabRelativeCompare": best.get("rtab_relative_compare"),
        },
    )


async def _debug_matches_upload(
    *,
    building_id: str | None,
    map_id: str | None,
    image: UploadFile,
    mask_persons: bool = False,
) -> MatchDebugResponse:
    resolved_building_id = _coerce_building_id(building_id, map_id)
    image_bytes = (await _read_upload_images([image], max_images=1))[0]
    return await _debug_matches(
        building_id=resolved_building_id,
        image_bytes=image_bytes,
        mask_persons=mask_persons,
    )


async def _debug_matches(
    *,
    building_id: str,
    image_bytes: bytes,
    mask_persons: bool = False,
) -> MatchDebugResponse:
    floor_maps = await _get_floor_maps(building_id)
    single_db = settings.MAPS_DIR / f"{building_id}.db"
    if not floor_maps and single_db.exists():
        floor_maps = [
            {
                "floor_id": "",
                "floor_name": "",
                "level": 0,
                "file_path": str(single_db),
            }
        ]
    if not floor_maps:
        raise HTTPException(status_code=404, detail=f"No maps found for building {building_id}")

    slam_engine = await asyncio.get_running_loop().run_in_executor(None, _get_sp_engine)
    best_response: MatchDebugResponse | None = None
    best_score = -1

    for floor_map in [_resolve_floor_path(fm) for fm in floor_maps]:
        db_path = floor_map["file_path"]
        if not Path(db_path).exists():
            continue
        try:
            intrinsics = slam_engine.extract_intrinsics_from_db(db_path)
            resized = _resize_query_images(
                [image_bytes],
                width=intrinsics["width"],
                height=intrinsics["height"],
            )[0]
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    _visualize_superpoint_matches,
                    db_path,
                    floor_map["floor_id"] or building_id,
                    resized,
                    slam_engine,
                    mask_persons,
                ),
            )
        except Exception as exc:
            logger.warning(f"[SLAM-DEBUG] Floor {floor_map.get('floor_name')} error: {exc}")
            continue

        response = _match_debug_response(result, floor_map)
        score = response.num_node_matches or response.num_good_matches
        if score > best_score:
            best_response = response
            best_score = score

    if best_response is None:
        raise HTTPException(status_code=503, detail="Match debug failed on all floors")
    return best_response


def _visualize_superpoint_matches(
    db_path: str,
    map_id: str,
    image_bytes: bytes,
    slam_engine: object,
    mask_persons: bool,
) -> dict:
    _ = mask_persons
    from slam_engines.superpoint.match_debugger import visualize_matches_sp

    return visualize_matches_sp(db_path, map_id, image_bytes, slam_engine)


def _match_debug_response(result: dict, floor_map: dict) -> MatchDebugResponse:
    return MatchDebugResponse(
        query_b64=_bgr_to_jpeg_b64(result["query_bgr"]),
        matches_b64=_bgr_to_jpeg_b64(result["vis_bgr"]),
        db_frame_b64=_bgr_to_jpeg_b64(result["db_bgr"]) if result.get("db_bgr") is not None else None,
        best_node_id=int(result.get("best_node_id") or 0),
        num_good_matches=int(result.get("num_good_matches") or 0),
        num_node_matches=int(result.get("num_node_matches") or 0),
        floor_id=str(floor_map.get("floor_id") or ""),
        floor_name=str(floor_map.get("floor_name") or ""),
        has_db_image=bool(result.get("has_db_image")),
    )


async def _get_sessions_or_raise(building_id: str) -> list[dict]:
    sessions = await _get_sessions(building_id)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"No scan sessions found for building {building_id}")
    return sessions


async def _get_sessions(building_id: str) -> list[dict]:
    if postgres_adapter is None:
        return []
    try:
        return await postgres_adapter.get_sessions_by_building_id(building_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid building_id: {exc}")
    except Exception as exc:
        logger.warning(f"[SLAM] Failed to fetch sessions for building {building_id}: {exc}")
        raise HTTPException(status_code=503, detail=f"Failed to fetch sessions: {exc}")


async def _get_floor_maps(building_id: str) -> list[dict]:
    if postgres_adapter is None:
        return []
    try:
        return await postgres_adapter.get_floor_maps(building_id)
    except ValueError:
        return []
    except Exception as exc:
        logger.warning(f"[SLAM] Failed to fetch floor maps for building {building_id}: {exc}")
        return []


async def _read_upload_images(images: list[UploadFile], *, max_images: int | None = None) -> list[bytes]:
    if not images:
        raise HTTPException(status_code=422, detail="At least one image file is required")
    if max_images is not None and len(images) > max_images:
        raise HTTPException(status_code=422, detail=f"Maximum {max_images} image files allowed")

    image_bytes_list = []
    for index, upload in enumerate(images):
        content_type = upload.content_type or ""
        if content_type and not content_type.startswith("image/") and content_type != "application/octet-stream":
            raise HTTPException(
                status_code=422,
                detail=f"File {index + 1} must be an image, got {content_type}",
            )
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=422, detail=f"File {index + 1} is empty")
        image_bytes_list.append(data)
    return image_bytes_list


def _coerce_building_id(building_id: str | None, map_id: str | None) -> str:
    resolved = (building_id or map_id or "").strip()
    if not resolved:
        raise HTTPException(status_code=422, detail="building_id or map_id form field is required")
    return resolved


def _filter_floor_maps(
    floor_maps: list[dict],
    *,
    floor_id: str | None,
    building_id: str,
) -> list[dict]:
    floor_id = (floor_id or "").strip()
    if not floor_id:
        return floor_maps

    # iOS 클라가 floor LEVEL (예: "-3") 을 floor_id 필드로 보내는 경우가 있어
    # UUID 매칭과 별도로 정수 level 매칭도 함께 시도.
    level_candidate: int | None = None
    try:
        level_candidate = int(floor_id)
    except ValueError:
        level_candidate = None

    filtered = [
        fm for fm in floor_maps
        if floor_id in {
            str(fm.get("floor_id") or ""),
            str(fm.get("scan_id") or ""),
        }
        or (level_candidate is not None and int(fm.get("level", 0)) == level_candidate)
    ]

    if not filtered:
        detail = {
            "message": "No map found for selected floor",
            "building_id": building_id,
            "floor_id": floor_id,
        }
        raise HTTPException(status_code=404, detail=detail)
    return filtered


def _resolve_floor_path(floor_map: dict) -> dict:
    file_path = str(floor_map["file_path"])
    if file_path.startswith("./storage/uploads/") or file_path.startswith("storage/uploads/"):
        file_path = f"/app/storage/uploads/{file_path.split('/')[-1]}"
    return {**floor_map, "file_path": file_path}


def _scale_query_intrinsics(
    intrinsics: dict[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except Exception:
        return None

    src_w = intrinsics.get("width") or intrinsics.get("w")
    src_h = intrinsics.get("height") or intrinsics.get("h")
    try:
        src_w_f = float(src_w)
        src_h_f = float(src_h)
    except Exception:
        src_w_f = float(width)
        src_h_f = float(height)
    if src_w_f <= 0 or src_h_f <= 0:
        return None

    sx = width / src_w_f
    sy = height / src_h_f
    return {
        "width": width,
        "height": height,
        "fx": fx * sx,
        "fy": fy * sy,
        "cx": cx * sx,
        "cy": cy * sy,
    }


def _resize_query_images(images: list[bytes], *, width: int, height: int) -> list[bytes]:
    resized = []
    for img_bytes in images:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)))
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            resized.append(buf.getvalue())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")
    return resized


async def _count_keyframes(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    from slam_engines.rtabmap.database_parser import DatabaseParser

    parsed = await DatabaseParser().parse_database(str(db_path), keyframe_limit=0)
    return int(parsed.get("num_keyframes") or 0)


def _overall_status(sessions: list[dict]) -> str:
    if not sessions:
        return "NOT_FOUND"
    statuses = {str(session.get("status") or "").upper() for session in sessions}
    if "FAILED" in statuses:
        return "FAILED"
    if statuses and statuses <= {"COMPLETED"}:
        return "COMPLETED"
    if statuses & {"PROCESSING", "EXTRACTING"}:
        return "PROCESSING"
    if statuses & {"UPLOADED", "PENDING"}:
        return "UPLOADED"
    return next(iter(statuses), "UNKNOWN")


def _serialize_session(session: dict) -> dict[str, Any]:
    return {
        "id": str(session.get("id") or ""),
        "building_id": str(session.get("building_id") or ""),
        "file_name": session.get("file_name"),
        "file_path": session.get("file_path"),
        "file_size": session.get("file_size"),
        "status": session.get("status"),
        "error_message": session.get("error_message"),
        "total_nodes": session.get("total_nodes"),
        "total_distance": session.get("total_distance"),
        "created_at": _iso(session.get("created_at")),
        "updated_at": _iso(session.get("updated_at")),
    }


def _first_created_at(sessions: list[dict]) -> str:
    values = [session.get("created_at") for session in sessions if session.get("created_at") is not None]
    if not values:
        return ""
    return _iso(min(values))


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _queue_length() -> int:
    if job_queue is None:
        return 0
    try:
        return int(job_queue.get_queue_length())
    except Exception:
        return 0


def _annotate_person_boxes(image_bytes: bytes, boxes: list[tuple[int, int, int, int]]) -> str:
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    annotated = bgr.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return _bgr_to_jpeg_b64(annotated)


def _bgr_to_jpeg_b64(bgr: Any) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode debug image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
