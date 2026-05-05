"""ARKit ↔ RTABMap rigid SE(3) transform 추정 (Sprint 49).

Sprint 35 v4 backfill 결과로 keyframe_meta.rtabmap_node_id 가 채워진 row의
ARKit world frame translation (tx, ty, tz) 와 RTABMap node pose translation
사이의 rigid SE(3) 변환을 Kabsch SVD (Umeyama, no scale) 로 추정한다.

Codex BLOCKER 1/2 (Sprint 49 evaluator-design) 반영:
- pair_count<3 / SVD fail / RMS>0.30m / rank deficient → confidence="low"
  (low 인 경우 호출자 POIProjectionStep 은 raw ARKit 좌표 fallback 금지하고
  graph snap fallback 으로 대체).
- residual RMS = ||rtabmap_xyz - (R @ arkit_xyz + t)||  (centered 변환 후
  t 를 다시 빼지 않는다 — Codex W-6).
- planar trajectory (z 거의 일정) 는 정상 실내 scan 에서 매우 흔하므로
  rank<3 이면 "planar" mode 로 표시하되 high confidence 로 허용한다 (W-7).
  단 horizontal axis 분포가 평면이 아닌 직선(rank<2) 인 경우는 low.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


# default thresholds (config 에서 오버라이드 가능)
DEFAULT_PAIR_COUNT_MIN_FOR_HIGH = 3
DEFAULT_RESIDUAL_RMS_MAX_M = 0.30
DEFAULT_PLANAR_Z_RANGE_M = 0.05  # z range < 0.05m → planar trajectory


TransformConfidence = Literal["high", "low"]


@dataclass(frozen=True)
class ArkitToRtabmapTransform:
    """rigid SE(3) (R, t).

    apply(xyz) = R @ xyz + t.

    Codex BLOCKER 1: confidence != "high" 일 때 호출자는 raw ARKit 좌표
    fallback 을 사용하지 말고 graph snap fallback 경로로 가야 한다.
    """

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    pair_count: int
    residual_rms_m: float
    confidence: TransformConfidence
    planar_mode: bool = False
    fallback_reason: str | None = None
    # debug/audit
    rotation_yaw_pitch_roll_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pair_count_min_for_high_confidence: int = DEFAULT_PAIR_COUNT_MIN_FOR_HIGH
    residual_rms_max_m: float = DEFAULT_RESIDUAL_RMS_MAX_M

    def apply(self, xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        v = np.asarray(xyz, dtype=np.float64).reshape(3)
        out = self.rotation @ v + self.translation
        return (float(out[0]), float(out[1]), float(out[2]))

    def to_metadata(self) -> dict[str, object]:
        return {
            "rotation_deg": [float(v) for v in self.rotation_yaw_pitch_roll_deg],
            "translation_m": [float(self.translation[i]) for i in range(3)],
            "pair_count": int(self.pair_count),
            "residual_rms_m": float(self.residual_rms_m),
            "transform_confidence": str(self.confidence),
            "planar_mode": bool(self.planar_mode),
            "fallback_reason": self.fallback_reason,
            "pair_count_min_for_high_confidence": int(
                self.pair_count_min_for_high_confidence
            ),
            "residual_rms_max_m": float(self.residual_rms_max_m),
        }


@dataclass(frozen=True)
class TransformInputPair:
    """rigid 추정 입력 single pair."""

    arkit_xyz: tuple[float, float, float]
    rtabmap_xyz: tuple[float, float, float]
    keyframe_seq: int = -1  # debug/audit


@dataclass(frozen=True)
class EstimateParams:
    pair_count_min_for_high_confidence: int = DEFAULT_PAIR_COUNT_MIN_FOR_HIGH
    residual_rms_max_m: float = DEFAULT_RESIDUAL_RMS_MAX_M
    planar_z_range_m: float = DEFAULT_PLANAR_Z_RANGE_M


@dataclass(frozen=True)
class EstimateResult:
    transform: ArkitToRtabmapTransform
    matched_pairs: list[TransformInputPair] = field(default_factory=list)


def _identity_transform_low(
    reason: str, pair_count: int, params: EstimateParams
) -> ArkitToRtabmapTransform:
    return ArkitToRtabmapTransform(
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
        pair_count=pair_count,
        residual_rms_m=float("nan"),
        confidence="low",
        planar_mode=False,
        fallback_reason=reason,
        rotation_yaw_pitch_roll_deg=(0.0, 0.0, 0.0),
        pair_count_min_for_high_confidence=params.pair_count_min_for_high_confidence,
        residual_rms_max_m=params.residual_rms_max_m,
    )


def _rotation_to_yaw_pitch_roll_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    """ZYX 순서 (yaw=z, pitch=y, roll=x). 디버그/audit 출력용."""
    r = rotation
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
        pitch = np.degrees(np.arctan2(-r[2, 0], sy))
        roll = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    else:
        # gimbal lock
        yaw = np.degrees(np.arctan2(-r[0, 1], r[1, 1]))
        pitch = np.degrees(np.arctan2(-r[2, 0], sy))
        roll = 0.0
    return float(yaw), float(pitch), float(roll)


def estimate_arkit_to_rtabmap_transform(
    pairs: list[TransformInputPair],
    *,
    params: EstimateParams | None = None,
) -> EstimateResult:
    """matched (arkit_xyz, rtabmap_xyz) pair list 에서 rigid SE(3) 추정.

    Codex W-6: residual = ||rtabmap_xyz - (R @ arkit_xyz + t)|| (centered 변환
    후 t 를 다시 빼지 않는다).

    Codex W-7: trajectory 가 planar (z range < 0.05m) 인 경우는 정상 실내
    scan 이므로 rank=2 deficient 라도 high confidence 로 허용한다 — Kabsch
    SVD 결과 rotation matrix 가 여전히 SO(3) 에 속하기 때문.

    Returns:
        EstimateResult(transform, matched_pairs).
        confidence == "high" 인 경우만 호출자가 transform.apply() 적용.
    """
    p = params if params is not None else EstimateParams()
    pair_count = len(pairs)

    if pair_count < p.pair_count_min_for_high_confidence:
        logger.info(
            "arkit_to_rtabmap_transform: pair_count=%d < min=%d → low",
            pair_count,
            p.pair_count_min_for_high_confidence,
        )
        return EstimateResult(
            transform=_identity_transform_low(
                f"insufficient_matched_pairs:n={pair_count}",
                pair_count,
                p,
            ),
            matched_pairs=list(pairs),
        )

    arkit = np.asarray(
        [pp.arkit_xyz for pp in pairs], dtype=np.float64
    )  # (N, 3)
    rtab = np.asarray(
        [pp.rtabmap_xyz for pp in pairs], dtype=np.float64
    )  # (N, 3)

    centroid_a = arkit.mean(axis=0)
    centroid_b = rtab.mean(axis=0)
    a_centered = arkit - centroid_a  # (N, 3)
    b_centered = rtab - centroid_b

    # rank check — collinear / zero-spread input
    a_z_range = float(arkit[:, 2].max() - arkit[:, 2].min())
    a_x_range = float(arkit[:, 0].max() - arkit[:, 0].min())
    a_y_range = float(arkit[:, 1].max() - arkit[:, 1].min())
    horizontal_spread = max(a_x_range, a_y_range)
    planar_mode = a_z_range < p.planar_z_range_m
    if horizontal_spread < 1e-3:
        # 수평 분포 거의 0 — 한 점 또는 거의 한 점 → rank<1
        return EstimateResult(
            transform=_identity_transform_low(
                f"degenerate_input:horizontal_spread={horizontal_spread:.4f}m",
                pair_count,
                p,
            ),
            matched_pairs=list(pairs),
        )

    h_matrix = a_centered.T @ b_centered  # (3, 3)
    try:
        u_mat, sigma, vt_mat = np.linalg.svd(h_matrix)
    except np.linalg.LinAlgError as e:
        logger.warning("arkit_to_rtabmap_transform: SVD failed: %s", e)
        return EstimateResult(
            transform=_identity_transform_low(
                "svd_failed", pair_count, p
            ),
            matched_pairs=list(pairs),
        )

    # rank: nonzero singular value count (under threshold)
    rank_threshold = max(sigma) * 1e-6 if sigma.size > 0 else 0.0
    rank = int(np.sum(sigma > rank_threshold)) if rank_threshold > 0 else 0
    if rank < 2:
        # 직선상 분포 — 회전 자유도 1 부족
        return EstimateResult(
            transform=_identity_transform_low(
                f"rank_deficient_collinear:rank={rank}",
                pair_count,
                p,
            ),
            matched_pairs=list(pairs),
        )

    # rank == 2: planar (W-7) — 정상 실내 scan. allow high confidence.
    # Kabsch + reflection 보정 (det 검사)
    d = float(np.linalg.det(vt_mat.T @ u_mat.T))
    correction = np.eye(3, dtype=np.float64)
    if d < 0:
        correction[2, 2] = -1.0
    rotation = vt_mat.T @ correction @ u_mat.T
    t = centroid_b - rotation @ centroid_a

    # residual RMS — Codex W-6: centered 변환 적용 후 t 를 다시 빼지 않는다.
    # 원본 좌표 기준 residual 계산.
    residuals = np.linalg.norm(rtab - (arkit @ rotation.T + t), axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    yaw_pitch_roll = _rotation_to_yaw_pitch_roll_deg(rotation)

    if rms > p.residual_rms_max_m:
        logger.info(
            "arkit_to_rtabmap_transform: rms=%.4fm > max=%.4fm → low",
            rms,
            p.residual_rms_max_m,
        )
        return EstimateResult(
            transform=ArkitToRtabmapTransform(
                rotation=rotation,
                translation=t,
                pair_count=pair_count,
                residual_rms_m=rms,
                confidence="low",
                planar_mode=planar_mode,
                fallback_reason=f"residual_rms_exceeded:{rms:.4f}m",
                rotation_yaw_pitch_roll_deg=yaw_pitch_roll,
                pair_count_min_for_high_confidence=p.pair_count_min_for_high_confidence,
                residual_rms_max_m=p.residual_rms_max_m,
            ),
            matched_pairs=list(pairs),
        )

    return EstimateResult(
        transform=ArkitToRtabmapTransform(
            rotation=rotation,
            translation=t,
            pair_count=pair_count,
            residual_rms_m=rms,
            confidence="high",
            planar_mode=planar_mode,
            fallback_reason=None,
            rotation_yaw_pitch_roll_deg=yaw_pitch_roll,
            pair_count_min_for_high_confidence=p.pair_count_min_for_high_confidence,
            residual_rms_max_m=p.residual_rms_max_m,
        ),
        matched_pairs=list(pairs),
    )


# ── helper: keyframe_meta + RTABMap nodes → matched pair list ───────────────


def build_pairs_from_keyframes_and_nodes(
    *,
    keyframe_pose_by_rtabmap_node_id: dict[int, tuple[float, float, float]],
    rtabmap_node_pose_by_id: dict[int, tuple[float, float, float]],
) -> list[TransformInputPair]:
    """ARKit↔RTABMap matched pair 추출.

    Args:
        keyframe_pose_by_rtabmap_node_id: keyframe_meta 에서 rtabmap_node_id IS NOT NULL
            인 row 의 (tx, ty, tz). dict[node_id → arkit_xyz].
        rtabmap_node_pose_by_id: RTABMap nodes list 에서 node_id → (tx, ty, tz)
            (pose matrix의 translation 부분).
    """
    pairs: list[TransformInputPair] = []
    for node_id, arkit_xyz in keyframe_pose_by_rtabmap_node_id.items():
        rtabmap_xyz = rtabmap_node_pose_by_id.get(node_id)
        if rtabmap_xyz is None:
            continue
        pairs.append(
            TransformInputPair(
                arkit_xyz=arkit_xyz,
                rtabmap_xyz=rtabmap_xyz,
                keyframe_seq=-1,
            )
        )
    return pairs
