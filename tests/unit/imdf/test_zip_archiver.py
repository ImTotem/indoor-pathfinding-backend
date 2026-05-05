"""zip_archiver 단위 테스트 — 결정성 + zip 구조.

결정성 보장 계약:
  동일 (build_job_id, generated_at) → 동일 zip bytes (sha256 동일).
  export_service.py가 generated_at=build_job.started_at으로 고정하므로
  같은 build_job에 대해 두 번 export해도 결과가 동일해야 한다.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from uuid import UUID

from indoor_server.application.imdf.builder import ImdfBuilder
from indoor_server.application.imdf.zip_archiver import _FILE_ORDER, build_imdf_zip

_SCAN_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_BUILD_JOB_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_NOW = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)

_FOOTPRINT = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0], [0.0, 0.0]]]
    ],
}


def _make_feature_set():
    builder = ImdfBuilder()
    return builder.build(
        scan_id=_SCAN_ID,
        build_job_id=_BUILD_JOB_ID,
        footprint_geojson=_FOOTPRINT,
        poi_nodes=[],
        poi_labels={},
        floor_z0=1.34,
        generated_at=_NOW,
    )


def test_zip_contains_six_files() -> None:
    """U5: zip 내부에 정확히 6개 파일이 있어야 한다."""
    features = _make_feature_set()
    zip_bytes = build_imdf_zip(features)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

    assert len(names) == 6
    for expected in _FILE_ORDER:
        assert expected in names


def test_zip_file_order() -> None:
    """파일이 _FILE_ORDER 순서로 저장되어야 한다."""
    features = _make_feature_set()
    zip_bytes = build_imdf_zip(features)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

    assert names == _FILE_ORDER


def test_zip_deterministic_same_input() -> None:
    """U6: 동일 입력 → 동일 bytes (timestamp 고정 덕분에 결정적)."""
    features = _make_feature_set()
    zip_bytes_1 = build_imdf_zip(features)
    zip_bytes_2 = build_imdf_zip(features)

    assert zip_bytes_1 == zip_bytes_2


def test_zip_manifest_parseable() -> None:
    """zip 내부 manifest.json이 JSON 파싱 가능하고 필수 키 포함."""
    features = _make_feature_set()
    zip_bytes = build_imdf_zip(features)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    assert manifest["format"] == "indoor-pathfinding-imdf-lite"
    assert manifest["scan_id"] == str(_SCAN_ID)
    assert manifest["build_job_id"] == str(_BUILD_JOB_ID)
    assert "feature_files" in manifest


def test_zip_timestamp_fixed() -> None:
    """ZipInfo.date_time이 (1980,1,1,0,0,0)으로 고정되어 있어야 한다."""
    features = _make_feature_set()
    zip_bytes = build_imdf_zip(features)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0), (
                f"{info.filename}: date_time={info.date_time}"
            )


def test_zip_geojson_valid() -> None:
    """zip 내부 모든 .geojson 파일이 type=FeatureCollection."""
    features = _make_feature_set()
    zip_bytes = build_imdf_zip(features)

    geojson_files = [f for f in _FILE_ORDER if f.endswith(".geojson")]

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for fname in geojson_files:
            fc = json.loads(zf.read(fname).decode("utf-8"))
            assert fc["type"] == "FeatureCollection", f"{fname}: type={fc['type']}"
            assert isinstance(fc["features"], list), f"{fname}: features not list"


def test_zip_sha256_deterministic_same_build_job() -> None:
    """D1: 동일 build_job_id + started_at(고정) → sha256 동일.

    export_service.py가 generated_at=build_job.started_at으로 고정하므로
    같은 build_job에 대해 두 번 build_archive를 호출해도 zip sha256이 같아야 한다.
    이 테스트는 builder+zip_archiver 계층에서 동일 generated_at을 주면
    sha256가 동일함을 보장한다.
    """
    fixed_time = datetime(2026, 4, 25, 9, 0, 0, tzinfo=UTC)

    def _build() -> bytes:
        features = ImdfBuilder().build(
            scan_id=_SCAN_ID,
            build_job_id=_BUILD_JOB_ID,
            footprint_geojson=_FOOTPRINT,
            poi_nodes=[],
            poi_labels={},
            floor_z0=1.34,
            generated_at=fixed_time,
        )
        return build_imdf_zip(features)

    sha1 = hashlib.sha256(_build()).hexdigest()
    sha2 = hashlib.sha256(_build()).hexdigest()

    assert sha1 == sha2, "두 번 export한 zip sha256이 다름 — 결정성 위반"


def test_zip_sha256_differs_for_different_generated_at() -> None:
    """D2: generated_at이 다르면 sha256이 달라야 한다.

    결정성의 역방향 검증: 시각이 다르면 manifest.generated_at도 달라지므로
    zip bytes가 달라야 함을 확인. 이는 generated_at이 실제로 zip에 반영됨을 보장.
    """
    time1 = datetime(2026, 4, 25, 9, 0, 0, tzinfo=UTC)
    time2 = datetime(2026, 4, 25, 10, 0, 0, tzinfo=UTC)

    def _build(ts: datetime) -> bytes:
        features = ImdfBuilder().build(
            scan_id=_SCAN_ID,
            build_job_id=_BUILD_JOB_ID,
            footprint_geojson=_FOOTPRINT,
            poi_nodes=[],
            poi_labels={},
            floor_z0=1.34,
            generated_at=ts,
        )
        return build_imdf_zip(features)

    sha1 = hashlib.sha256(_build(time1)).hexdigest()
    sha2 = hashlib.sha256(_build(time2)).hexdigest()

    assert sha1 != sha2, "generated_at이 다른데 sha256이 동일 — manifest에 시각이 미반영"
