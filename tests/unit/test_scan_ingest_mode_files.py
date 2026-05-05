from __future__ import annotations

from pathlib import Path

import pytest

from indoor_server.application.scan_ingest_service import _enforce_mode_specific_files
from indoor_server.application.scan_manifest import ScanManifest, parse_manifest_dict
from indoor_server.application.zip_unpacker import UnpackedScan
from indoor_server.domain.scan.errors import MissingRequiredFile

SCAN_ID = "00000000-1111-2222-3333-444444444444"


def _v7_manifest() -> ScanManifest:
    return parse_manifest_dict(
        {
            "metadata_version": 7,
            "scan_id": SCAN_ID,
            "mode": "raw_video_recording",
            "keyframes_included": False,
            "keyframe_image_source": "video_frames",
            "poi_image_source": "poi_photo_image_blob",
            "rtabmap_accepted_frame_count": 0,
            "sidecar_keyframe_meta_count": 3,
            "dropped_reject_frame_image_count": 0,
            "rtabmap_reprocessed": False,
            "client_app_version": "2.0.0",
            "video_path": "scan.mp4",
            "poses_path": "poses.bin",
            "video_codec": "hevc",
            "video_fps_nominal": 60,
            "pose_record_count": 180,
            "intrinsics_fx": 1500.0,
            "intrinsics_fy": 1500.0,
            "intrinsics_cx": 960.0,
            "intrinsics_cy": 540.0,
        },
        expected_scan_id=SCAN_ID,
    )


def _unpacked(*, video: bool = True, poses: bool = True) -> UnpackedScan:
    root = Path("/tmp") / SCAN_ID
    return UnpackedScan(
        scan_id=SCAN_ID,
        rtabmap_db_path=root / "rtabmap.db",
        sidecar_db_path=root / "scan_metadata.db",
        keyframe_paths=[],
        payload_sha256="0" * 64,
        manifest_path=root / "manifest.json",
        video_path=root / "scan.mp4" if video else None,
        poses_path=root / "poses.bin" if poses else None,
    )


def test_raw_video_recording_requires_video_and_poses() -> None:
    manifest = _v7_manifest()

    with pytest.raises(MissingRequiredFile) as exc_info:
        _enforce_mode_specific_files(
            manifest=manifest,
            unpacked=_unpacked(video=False, poses=True),
        )

    assert "scan.mp4" in str(exc_info.value)


def test_raw_video_recording_accepts_rtabmap_video_and_poses() -> None:
    _enforce_mode_specific_files(
        manifest=_v7_manifest(),
        unpacked=_unpacked(video=True, poses=True),
    )
