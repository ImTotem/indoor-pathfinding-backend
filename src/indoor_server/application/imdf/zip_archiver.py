"""IMDF zip 아카이버 — FeatureSet → in-memory zip bytes. 순수 함수."""
from __future__ import annotations

import io
import json
import zipfile
from zipfile import ZipInfo

from indoor_server.domain.imdf.models import ImdfFeatureSet

# zip 결정성 보장: 모든 ZipInfo timestamp을 epoch 최솟값으로 고정
# → 동일 입력 == 동일 bytes (단위 테스트 byte-level 비교 + sha256 안정성)
_FIXED_DATETIME = (1980, 1, 1, 0, 0, 0)

_FILE_ORDER = [
    "manifest.json",
    "level.geojson",
    "unit.geojson",
    "footprint.geojson",
    "amenity.geojson",
    "anchor.geojson",
]


def _to_json_bytes(obj: object) -> bytes:
    """sort_keys=True로 결정적 JSON 직렬화 → utf-8 bytes."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")


def build_imdf_zip(features: ImdfFeatureSet) -> bytes:
    """ImdfFeatureSet → in-memory zip bytes.

    ZipInfo timestamp을 (1980,1,1,0,0,0)으로 고정해 결정성을 보장한다.
    compress_type=ZIP_DEFLATED로 압축.
    """
    file_contents: dict[str, bytes] = {
        "manifest.json": _to_json_bytes(features.manifest),
        "level.geojson": _to_json_bytes(features.level_geojson),
        "unit.geojson": _to_json_bytes(features.unit_geojson),
        "footprint.geojson": _to_json_bytes(features.footprint_geojson),
        "amenity.geojson": _to_json_bytes(features.amenity_geojson),
        "anchor.geojson": _to_json_bytes(features.anchor_geojson),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in _FILE_ORDER:
            info = ZipInfo(filename=name)
            info.date_time = _FIXED_DATETIME
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, file_contents[name])

    return buf.getvalue()
