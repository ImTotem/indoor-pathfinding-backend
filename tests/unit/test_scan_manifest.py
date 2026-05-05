"""scan_manifest v4/v5 contract 단위 테스트 (Sprint 49 Codex BLOCKER 4)."""
from __future__ import annotations

import json

import pytest

from indoor_server.application.scan_manifest import (
    ManifestStructureInvalid,
    ManifestVersionMismatch,
    enforce_version_contract,
    parse_manifest_dict,
    parse_manifest_file,
)

SCAN_ID = "00000000-1111-2222-3333-444444444444"


def _v5_manifest(scan_id: str = SCAN_ID) -> dict[str, object]:
    return {
        "metadata_version": 5,
        "scan_id": scan_id,
        "keyframes_included": False,
        "keyframe_image_source": "rtabmap_db_data_table",
        "poi_image_source": "poi_photo_image_blob",
        "rtabmap_accepted_frame_count": 128,
        "sidecar_keyframe_meta_count": 284,
        "dropped_reject_frame_image_count": 156,
        "client_app_version": "1.2.0",
    }


# ── 4 분기 contract 검증 ─────────────────────────────────────────────────────


def test_branch_1_legacy_v4_passes() -> None:
    """manifest absent + sidecar v4 → legacy PASS."""
    enforce_version_contract(manifest=None, sidecar_user_version=4)


def test_branch_2_v5_manifest_v5_sidecar_passes() -> None:
    """manifest v5 + sidecar v5 → PASS."""
    manifest = parse_manifest_dict(_v5_manifest(), expected_scan_id=SCAN_ID)
    enforce_version_contract(manifest=manifest, sidecar_user_version=5)


def test_branch_3_v5_sidecar_no_manifest_422() -> None:
    """manifest absent + sidecar v5 → 422."""
    with pytest.raises(ManifestVersionMismatch) as exc_info:
        enforce_version_contract(manifest=None, sidecar_user_version=5)
    assert exc_info.value.detail["sidecar_user_version"] == 5


def test_branch_4_version_mismatch_422() -> None:
    """manifest v5 + sidecar v4 → 422."""
    manifest = parse_manifest_dict(_v5_manifest(), expected_scan_id=SCAN_ID)
    with pytest.raises(ManifestVersionMismatch):
        enforce_version_contract(manifest=manifest, sidecar_user_version=4)


def test_unsupported_manifest_version_422() -> None:
    raw = _v5_manifest()
    raw["metadata_version"] = 4  # v5 만 지원
    manifest = parse_manifest_dict(raw, expected_scan_id=SCAN_ID)
    with pytest.raises(ManifestVersionMismatch):
        enforce_version_contract(manifest=manifest, sidecar_user_version=4)


# ── parse_manifest_dict ─────────────────────────────────────────────────────


def test_parse_dict_minimum_keys() -> None:
    manifest = parse_manifest_dict(_v5_manifest(), expected_scan_id=SCAN_ID)
    assert manifest.metadata_version == 5
    assert manifest.scan_id == SCAN_ID
    assert manifest.keyframes_included is False
    assert manifest.rtabmap_accepted_frame_count == 128
    assert manifest.sidecar_keyframe_meta_count == 284
    assert manifest.dropped_reject_frame_image_count == 156


def test_parse_dict_scan_id_mismatch_invalid() -> None:
    raw = _v5_manifest("wrong-id")
    with pytest.raises(ManifestStructureInvalid):
        parse_manifest_dict(raw, expected_scan_id=SCAN_ID)


def test_parse_dict_missing_version_invalid() -> None:
    raw = _v5_manifest()
    del raw["metadata_version"]
    with pytest.raises(ManifestStructureInvalid):
        parse_manifest_dict(raw, expected_scan_id=SCAN_ID)


def test_parse_file_round_trip(tmp_path) -> None:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(_v5_manifest()), encoding="utf-8")
    manifest = parse_manifest_file(p, expected_scan_id=SCAN_ID)
    assert manifest.metadata_version == 5


def test_parse_file_missing_404_invalid(tmp_path) -> None:
    with pytest.raises(ManifestStructureInvalid):
        parse_manifest_file(tmp_path / "no.json", expected_scan_id=SCAN_ID)


# ── R-1 evidence ─────────────────────────────────────────────────────────────


def test_r1_evidence_keys_in_metadata() -> None:
    manifest = parse_manifest_dict(_v5_manifest(), expected_scan_id=SCAN_ID)
    md = manifest.to_metadata()
    assert md["keyframes_included"] is False
    assert md["keyframe_image_source"] == "rtabmap_db_data_table"
    assert md["poi_image_source"] == "poi_photo_image_blob"
    assert md["rtabmap_accepted_frame_count"] == 128
    assert md["sidecar_keyframe_meta_count"] == 284
    assert md["dropped_reject_frame_image_count"] == 156


# ── Sprint 67 v7 — raw_video_recording manifest ──────────────────────────────


def _v7_manifest(scan_id: str = SCAN_ID) -> dict[str, object]:
    return {
        "metadata_version": 7,
        "scan_id": scan_id,
        "mode": "raw_video_recording",
        "keyframes_included": False,
        "keyframe_image_source": "video_frames",
        "poi_image_source": "poi_photo_image_blob",
        "rtabmap_accepted_frame_count": 0,
        "sidecar_keyframe_meta_count": 3000,
        "dropped_reject_frame_image_count": 0,
        "rtabmap_reprocessed": False,
        "client_app_version": "2.0.0",
        "video_path": "scan.mp4",
        "poses_path": "poses.bin",
        "video_codec": "hevc",
        "video_fps_nominal": 60,
        "pose_record_count": 36000,
        "intrinsics_fx": 1500.0,
        "intrinsics_fy": 1500.0,
        "intrinsics_cx": 960.0,
        "intrinsics_cy": 540.0,
    }


def test_v7_manifest_parse_includes_video_fields() -> None:
    manifest = parse_manifest_dict(_v7_manifest(), expected_scan_id=SCAN_ID)
    assert manifest.metadata_version == 7
    assert manifest.mode == "raw_video_recording"
    assert manifest.is_video_mode is True
    assert manifest.video_path == "scan.mp4"
    assert manifest.poses_path == "poses.bin"
    assert manifest.video_codec == "hevc"
    assert manifest.video_fps_nominal == 60
    assert manifest.pose_record_count == 36000
    assert manifest.intrinsics_fx == 1500.0
    assert manifest.intrinsics_cx == 960.0


def test_v7_manifest_with_v6_sidecar_passes() -> None:
    """Sprint 67 — v7 manifest 는 sidecar v6 을 그대로 사용한다 (스키마 변화 없음)."""
    manifest = parse_manifest_dict(_v7_manifest(), expected_scan_id=SCAN_ID)
    enforce_version_contract(manifest=manifest, sidecar_user_version=6)


def test_v7_manifest_with_v5_sidecar_rejected_422() -> None:
    """v7 manifest + sidecar v5 → 422 (허용된 sidecar version 집합 위반)."""
    manifest = parse_manifest_dict(_v7_manifest(), expected_scan_id=SCAN_ID)
    with pytest.raises(ManifestVersionMismatch):
        enforce_version_contract(manifest=manifest, sidecar_user_version=5)


def test_v6_manifest_with_v7_implied_sidecar_rejected() -> None:
    """v6 manifest 와 sidecar v7 은 허용 안 됨 (v7 sidecar 자체가 미지원)."""
    raw = _v5_manifest()
    raw["metadata_version"] = 6
    raw["mode"] = "raw_arkit_recording"
    manifest = parse_manifest_dict(raw, expected_scan_id=SCAN_ID)
    with pytest.raises(ManifestVersionMismatch):
        enforce_version_contract(manifest=manifest, sidecar_user_version=7)


def test_v7_evidence_metadata_includes_video_fields() -> None:
    manifest = parse_manifest_dict(_v7_manifest(), expected_scan_id=SCAN_ID)
    md = manifest.to_metadata()
    assert md["video_path"] == "scan.mp4"
    assert md["poses_path"] == "poses.bin"
    assert md["video_codec"] == "hevc"
    assert md["video_fps_nominal"] == 60
    assert md["pose_record_count"] == 36000


def test_is_video_mode_false_for_v6() -> None:
    manifest = parse_manifest_dict(_v5_manifest(), expected_scan_id=SCAN_ID)
    assert manifest.is_video_mode is False
