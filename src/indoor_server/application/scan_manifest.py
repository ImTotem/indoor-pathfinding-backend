"""scan ZIP manifest.json (v5) 파서/검증 (Sprint 49).

Codex BLOCKER 4 (v4/v5 policy 한 문장 contract):
  1) manifest absent + sidecar v4 → legacy PASS (backward compat)
  2) manifest present + manifest_version=5 + sidecar v5 → PASS
  3) manifest absent + sidecar v5 → 422 (manifest 필수)
  4) manifest_version != sidecar version → 422 (mismatch)

Codex BLOCKER 3 (manifest 결정성):
- exported_at, created_at 같은 wall-clock timestamp 제거.
- iOS 측에서 deterministic source (scan_session.ended_at) 또는 timestamp 통째 제거 사용.
- ZIP entry timestamp 도 deterministic 값으로 고정 (별도 처리).

Codex BLOCKER 5 (R-1 측정 가능):
- manifest 가 다음 키를 포함하면 BuildCounts 에 노출되어 evidence 자동 검증 가능:
  - keyframes_included: bool
  - keyframe_image_source: str
  - poi_image_source: str
  - rtabmap_accepted_frame_count: int
  - sidecar_keyframe_meta_count: int
  - dropped_reject_frame_image_count: int
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from indoor_server.domain.scan.errors import ScanDomainError

logger = logging.getLogger(__name__)


SUPPORTED_MANIFEST_VERSIONS = frozenset(range(1, 100))

# 매핑이 정의되지 않은 manifest version 은 모든 sidecar version 허용 (lenient).
MANIFEST_VERSION_TO_SIDECAR_VERSIONS: dict[int, frozenset[int]] = {
    5: frozenset({5}),
    6: frozenset({6}),
    7: frozenset({6}),  # sidecar 는 v6 그대로 (legacy)
}


class ManifestVersionMismatch(ScanDomainError):
    """manifest 와 sidecar PRAGMA user_version 불일치, 또는 manifest 누락."""

    code = "MANIFEST_VERSION_MISMATCH"


class ManifestStructureInvalid(ScanDomainError):
    """manifest.json 형식이 잘못됨."""

    code = "MANIFEST_STRUCTURE_INVALID"


@dataclass(frozen=True)
class ScanManifest:
    """parsed manifest.json + 일관성 검증 통과 산출물."""

    metadata_version: int
    scan_id: str
    keyframes_included: bool
    keyframe_image_source: str
    poi_image_source: str
    rtabmap_accepted_frame_count: int | None = None
    sidecar_keyframe_meta_count: int | None = None
    dropped_reject_frame_image_count: int | None = None
    client_app_version: str | None = None
    # Sprint 65 v6:
    #   - mode: "raw_arkit_recording" | "realtime_slam" (legacy v5).
    #   - rtabmap_reprocessed: 서버가 rtabmap-reprocess 실행 후 true 로 갱신.
    mode: str | None = None
    rtabmap_reprocessed: bool | None = None
    # Sprint 67 v7 (raw_video_recording):
    #   60fps HEVC video + poses.bin. RTAB-Map raw recording 은 유지한다.
    #   서버 main build 는 rtabmap.db 를 사용하고 video/poses 는 dense evidence 보조 입력이다.
    video_path: str | None = None
    poses_path: str | None = None
    video_codec: str | None = None
    video_fps_nominal: int | None = None
    pose_record_count: int | None = None
    intrinsics_fx: float | None = None
    intrinsics_fy: float | None = None
    intrinsics_cx: float | None = None
    intrinsics_cy: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        """BuildCounts 노출용 dict (R-1 evidence)."""
        return {
            "metadata_version": int(self.metadata_version),
            "keyframes_included": bool(self.keyframes_included),
            "keyframe_image_source": str(self.keyframe_image_source),
            "poi_image_source": str(self.poi_image_source),
            "rtabmap_accepted_frame_count": self.rtabmap_accepted_frame_count,
            "sidecar_keyframe_meta_count": self.sidecar_keyframe_meta_count,
            "dropped_reject_frame_image_count": self.dropped_reject_frame_image_count,
            "client_app_version": self.client_app_version,
            "mode": self.mode,
            "rtabmap_reprocessed": self.rtabmap_reprocessed,
            "video_path": self.video_path,
            "poses_path": self.poses_path,
            "video_codec": self.video_codec,
            "video_fps_nominal": self.video_fps_nominal,
            "pose_record_count": self.pose_record_count,
        }

    @property
    def is_video_mode(self) -> bool:
        """Sprint 67 — raw_video_recording 모드 여부."""
        return self.mode == "raw_video_recording"


def parse_manifest_file(path: Path, *, expected_scan_id: str) -> ScanManifest:
    """manifest.json 파일을 읽어 ScanManifest 로 반환.

    Raises:
        ManifestStructureInvalid: 파일이 없거나 JSON 파싱 실패, 필수 키 누락.
    """
    if not path.exists():
        raise ManifestStructureInvalid(
            "manifest.json 파일을 찾을 수 없습니다.",
            detail={"path": str(path)},
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestStructureInvalid(
            f"manifest.json 파싱 실패: {e}",
            detail={"path": str(path)},
        ) from e

    if not isinstance(raw, dict):
        raise ManifestStructureInvalid(
            "manifest.json 최상위가 object 가 아닙니다.",
        )

    return parse_manifest_dict(raw, expected_scan_id=expected_scan_id)


def parse_manifest_dict(raw: dict[str, object], *, expected_scan_id: str) -> ScanManifest:
    metadata_version_raw = raw.get("metadata_version")
    if not isinstance(metadata_version_raw, int):
        raise ManifestStructureInvalid(
            "manifest.json metadata_version 이 정수가 아닙니다.",
            detail={"got": metadata_version_raw},
        )

    scan_id_raw = raw.get("scan_id")
    if not isinstance(scan_id_raw, str) or scan_id_raw.lower() != expected_scan_id.lower():
        raise ManifestStructureInvalid(
            "manifest.json scan_id 가 form 과 일치하지 않습니다.",
            detail={
                "expected_scan_id": expected_scan_id,
                "got_scan_id": scan_id_raw,
            },
        )

    keyframes_included = bool(raw.get("keyframes_included", False))
    keyframe_image_source = str(
        raw.get("keyframe_image_source", "rtabmap_db_data_table")
    )
    poi_image_source = str(
        raw.get("poi_image_source", "poi_photo_image_blob")
    )

    def _opt_int(key: str) -> int | None:
        v = raw.get(key)
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return None

    client_app_version_raw = raw.get("client_app_version")
    client_app_version = (
        client_app_version_raw
        if isinstance(client_app_version_raw, str)
        else None
    )

    mode_raw = raw.get("mode")
    mode = mode_raw if isinstance(mode_raw, str) else None

    reprocessed_raw = raw.get("rtabmap_reprocessed")
    rtabmap_reprocessed = (
        reprocessed_raw if isinstance(reprocessed_raw, bool) else None
    )

    def _opt_str(key: str) -> str | None:
        v = raw.get(key)
        return v if isinstance(v, str) else None

    def _opt_float(key: str) -> float | None:
        v = raw.get(key)
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    known_keys = {
        "metadata_version", "scan_id", "keyframes_included",
        "keyframe_image_source", "poi_image_source",
        "rtabmap_accepted_frame_count", "sidecar_keyframe_meta_count",
        "dropped_reject_frame_image_count", "client_app_version",
        "mode", "rtabmap_reprocessed",
        "video_path", "poses_path", "video_codec", "video_fps_nominal",
        "pose_record_count", "intrinsics_fx", "intrinsics_fy",
        "intrinsics_cx", "intrinsics_cy",
    }

    return ScanManifest(
        metadata_version=int(metadata_version_raw),
        scan_id=scan_id_raw,
        keyframes_included=keyframes_included,
        keyframe_image_source=keyframe_image_source,
        poi_image_source=poi_image_source,
        rtabmap_accepted_frame_count=_opt_int("rtabmap_accepted_frame_count"),
        sidecar_keyframe_meta_count=_opt_int("sidecar_keyframe_meta_count"),
        dropped_reject_frame_image_count=_opt_int("dropped_reject_frame_image_count"),
        client_app_version=client_app_version,
        mode=mode,
        rtabmap_reprocessed=rtabmap_reprocessed,
        video_path=_opt_str("video_path"),
        poses_path=_opt_str("poses_path"),
        video_codec=_opt_str("video_codec"),
        video_fps_nominal=_opt_int("video_fps_nominal"),
        pose_record_count=_opt_int("pose_record_count"),
        intrinsics_fx=_opt_float("intrinsics_fx"),
        intrinsics_fy=_opt_float("intrinsics_fy"),
        intrinsics_cx=_opt_float("intrinsics_cx"),
        intrinsics_cy=_opt_float("intrinsics_cy"),
        extra={k: v for k, v in raw.items() if k not in known_keys},
    )


def enforce_version_contract(
    *,
    manifest: ScanManifest | None,
    sidecar_user_version: int,
) -> None:
    """Codex BLOCKER 4: 4 분기 contract 검증.

    Raises:
        ManifestVersionMismatch (422):
            - manifest absent + sidecar v5
            - manifest present + version != sidecar
            - manifest present + version not in SUPPORTED_MANIFEST_VERSIONS
    """
    if manifest is None:
        if sidecar_user_version == 4:
            return  # legacy PASS
        if sidecar_user_version >= 5:
            raise ManifestVersionMismatch(
                "sidecar v5 이상은 manifest.json 이 필수입니다.",
                detail={
                    "sidecar_user_version": sidecar_user_version,
                    "manifest_present": False,
                },
            )
        raise ManifestVersionMismatch(
            f"지원하지 않는 sidecar user_version={sidecar_user_version}",
            detail={"sidecar_user_version": sidecar_user_version},
        )

    if manifest.metadata_version not in SUPPORTED_MANIFEST_VERSIONS:
        raise ManifestVersionMismatch(
            f"지원하지 않는 manifest metadata_version={manifest.metadata_version}",
            detail={
                "manifest_metadata_version": manifest.metadata_version,
                "supported": sorted(SUPPORTED_MANIFEST_VERSIONS),
            },
        )

    # 정책: manifest version 과 sidecar user_version 의 매핑은 명시된 경우만 검증.
    # 매핑 정의 없는 manifest version 은 모든 sidecar version 허용 (lenient).
    allowed_sidecar = MANIFEST_VERSION_TO_SIDECAR_VERSIONS.get(manifest.metadata_version)
    if allowed_sidecar is not None and sidecar_user_version not in allowed_sidecar:
        # 매핑이 명시된 경우만 strict 검증 (legacy v5/v6/v7 호환성 보장).
        raise ManifestVersionMismatch(
            "manifest metadata_version 과 sidecar user_version 조합이 허용되지 않습니다.",
            detail={
                "manifest_metadata_version": manifest.metadata_version,
                "sidecar_user_version": sidecar_user_version,
                "allowed_sidecar": sorted(allowed_sidecar),
            },
        )
