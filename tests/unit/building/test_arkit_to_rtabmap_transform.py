"""ArkitToRtabmapTransform Kabsch SVD 단위 테스트 (Sprint 49).

Codex BLOCKER 1/2 + W-6/W-7 검증.
"""
# ruff: noqa: N806  (R / R_expected 등 수학 관례 — 테스트 가독성 우선)
from __future__ import annotations

import math

import numpy as np
import pytest

from indoor_server.application.building.arkit_to_rtabmap_transform import (
    EstimateParams,
    TransformInputPair,
    estimate_arkit_to_rtabmap_transform,
)


def _rotation_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c = math.cos(r)
    s = math.sin(r)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _make_pairs(
    *,
    rotation_deg: float,
    translation: tuple[float, float, float],
    n: int = 20,
    z_range: float = 0.30,
) -> list[TransformInputPair]:
    """ARKit positions → R @ x + t = RTABMap positions 인 synthetic pair list."""
    rng = np.random.default_rng(seed=42)
    arkit_pts = rng.uniform(-5.0, 5.0, size=(n, 3))
    arkit_pts[:, 2] = rng.uniform(0.0, z_range, size=n)
    R = _rotation_z(rotation_deg)
    t = np.asarray(translation, dtype=np.float64)
    rtab_pts = arkit_pts @ R.T + t
    pairs: list[TransformInputPair] = []
    for i in range(n):
        pairs.append(
            TransformInputPair(
                arkit_xyz=tuple(float(v) for v in arkit_pts[i]),  # type: ignore[arg-type]
                rtabmap_xyz=tuple(float(v) for v in rtab_pts[i]),  # type: ignore[arg-type]
                keyframe_seq=i,
            )
        )
    return pairs


def test_recovers_rotation_30deg_high_confidence() -> None:
    pairs = _make_pairs(rotation_deg=30.0, translation=(1.0, 2.0, 0.5), n=20)
    res = estimate_arkit_to_rtabmap_transform(pairs)
    tf = res.transform
    assert tf.confidence == "high"
    assert tf.pair_count == 20
    assert tf.residual_rms_m < 1e-6
    # rotation 복원 검증 — yaw component
    R_expected = _rotation_z(30.0)
    np.testing.assert_allclose(tf.rotation, R_expected, atol=1e-6)
    np.testing.assert_allclose(
        tf.translation,
        np.asarray([1.0, 2.0, 0.5]),
        atol=1e-6,
    )


def test_recovers_rotation_neg45deg() -> None:
    pairs = _make_pairs(rotation_deg=-45.0, translation=(0.0, 0.0, 0.0), n=15)
    res = estimate_arkit_to_rtabmap_transform(pairs)
    tf = res.transform
    assert tf.confidence == "high"
    R_expected = _rotation_z(-45.0)
    np.testing.assert_allclose(tf.rotation, R_expected, atol=1e-6)


def test_planar_trajectory_high_confidence() -> None:
    """Codex W-7: 실내 trajectory 는 거의 평면이라도 high 로 인정."""
    pairs = _make_pairs(rotation_deg=10.0, translation=(0.0, 0.0, 0.0), n=15, z_range=0.0)
    res = estimate_arkit_to_rtabmap_transform(pairs)
    tf = res.transform
    assert tf.confidence == "high"
    assert tf.planar_mode is True


def test_pair_count_below_min_low_confidence() -> None:
    """Codex BLOCKER 2: pair_count<3 → low."""
    pairs = _make_pairs(rotation_deg=0.0, translation=(0.0, 0.0, 0.0), n=2)
    res = estimate_arkit_to_rtabmap_transform(pairs)
    tf = res.transform
    assert tf.confidence == "low"
    assert tf.fallback_reason is not None
    assert "insufficient_matched_pairs" in tf.fallback_reason


def test_residual_rms_exceeded_low_confidence() -> None:
    """Codex BLOCKER 2: rms > 0.30m 이면 low confidence."""
    rng = np.random.default_rng(seed=7)
    n = 15
    arkit_pts = rng.uniform(-5.0, 5.0, size=(n, 3))
    arkit_pts[:, 2] = 0.5
    R = _rotation_z(20.0)
    t = np.asarray([2.0, -1.0, 0.3])
    rtab_pts = arkit_pts @ R.T + t
    # 큰 noise 주입 (>30cm)
    rtab_pts += rng.normal(0.0, 1.5, size=rtab_pts.shape)
    pairs = [
        TransformInputPair(
            arkit_xyz=tuple(float(v) for v in arkit_pts[i]),  # type: ignore[arg-type]
            rtabmap_xyz=tuple(float(v) for v in rtab_pts[i]),  # type: ignore[arg-type]
        )
        for i in range(n)
    ]
    res = estimate_arkit_to_rtabmap_transform(pairs)
    tf = res.transform
    assert tf.confidence == "low"
    assert tf.fallback_reason is not None
    assert "residual_rms_exceeded" in tf.fallback_reason


def test_residual_uses_original_coords_not_centered() -> None:
    """Codex W-6: residual = ||rtab - (R @ arkit + t)||  (centered 좌표 변환 후
    t 를 다시 빼지 않는다)."""
    pairs = _make_pairs(rotation_deg=15.0, translation=(7.0, -3.0, 2.0), n=10)
    res = estimate_arkit_to_rtabmap_transform(pairs)
    tf = res.transform
    # noise 0 이므로 residual 0 이어야 함. translation 7m 이 있어도 residual 은 영향
    # 받지 않아야 (centered 좌표 빼기 + apply 하면 0).
    assert tf.confidence == "high"
    assert tf.residual_rms_m < 1e-6


def test_apply_round_trip() -> None:
    pairs = _make_pairs(rotation_deg=22.5, translation=(5.0, 6.0, 0.5), n=10)
    res = estimate_arkit_to_rtabmap_transform(pairs)
    tf = res.transform
    assert tf.confidence == "high"
    arkit = (1.0, 0.0, 0.0)
    out = tf.apply(arkit)
    R = _rotation_z(22.5)
    expected = R @ np.asarray(arkit) + np.asarray([5.0, 6.0, 0.5])
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_collinear_input_rank_deficient_low() -> None:
    """rank<2 (직선 분포) → low."""
    n = 10
    arkit_pts = np.zeros((n, 3))
    arkit_pts[:, 0] = np.linspace(0, 1, n)
    # y, z 0 — collinear along x
    R = _rotation_z(0.0)
    rtab_pts = arkit_pts @ R.T
    pairs = [
        TransformInputPair(
            arkit_xyz=tuple(float(v) for v in arkit_pts[i]),  # type: ignore[arg-type]
            rtabmap_xyz=tuple(float(v) for v in rtab_pts[i]),  # type: ignore[arg-type]
        )
        for i in range(n)
    ]
    res = estimate_arkit_to_rtabmap_transform(pairs)
    tf = res.transform
    assert tf.confidence == "low"
    assert tf.fallback_reason is not None
    assert "rank_deficient" in tf.fallback_reason


def test_metadata_dict_round_trip() -> None:
    pairs = _make_pairs(rotation_deg=10.0, translation=(0.0, 0.0, 0.0), n=10)
    res = estimate_arkit_to_rtabmap_transform(pairs)
    md = res.transform.to_metadata()
    assert "rotation_deg" in md
    assert "translation_m" in md
    assert "pair_count" in md
    assert "residual_rms_m" in md
    assert md["transform_confidence"] == "high"
    assert md["pair_count"] == 10


def test_high_confidence_requires_pair_count_min() -> None:
    """Codex W-5: pair_count_min_for_high_confidence 분리."""
    # default min=3 인 경우 n=3 OK
    pairs = _make_pairs(rotation_deg=0.0, translation=(0.0, 0.0, 0.0), n=3)
    res = estimate_arkit_to_rtabmap_transform(
        pairs, params=EstimateParams(pair_count_min_for_high_confidence=3)
    )
    assert res.transform.confidence == "high"

    # min=10 으로 올리면 동일 input n=3 → low
    res2 = estimate_arkit_to_rtabmap_transform(
        pairs, params=EstimateParams(pair_count_min_for_high_confidence=10)
    )
    assert res2.transform.confidence == "low"


def test_empty_input_low() -> None:
    res = estimate_arkit_to_rtabmap_transform([])
    assert res.transform.confidence == "low"
    assert res.transform.pair_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
