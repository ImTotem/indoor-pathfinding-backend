"""Build Phase 단발 실행 — 실 스캔 sidecar+keyframes에서 그래프 추출 + 시각화.

목적: 사용자가 받은 v1 sidecar(scan_metadata.db) + keyframes/*.jpg를
HTTP/Postgres/worker 거치지 않고 BuildPipeline 직접 호출로 처리하여
var/debug/{scan_id}/{job_id}/ 에 4-패널 composite 생성.

제약/우회:
- HuggingFace Segformer 모델이 401(접근 불가) → bottom-half-floor 휴리스틱 stub.
- ARKit world frame은 Y-up, BuildPipeline은 Z-up 가정 → pose_matrix를 X축 -90° 회전으로 변환.
- v1 schema에는 poi_mark.label/source 컬럼 없음 → POI 없이 walkable graph만 빌드.

사용:
    uv run python scripts/run_real_scan.py <scan_dir> [mode] [--depth-nn]

    --depth-nn: Depth Anything v2 NN back-projection 활성.
                hack(distance cap, mask 25%, opening, obs_threshold) 모두 비활성.

scan_dir 안에 scan_metadata.db, rtabmap.db, keyframes/*.jpg 가 있어야 한다.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import struct
import sys
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indoor_server.application.building.debug.filesystem_sink import FilesystemDebugSink
from indoor_server.application.building.pipeline import BuildPipeline
from indoor_server.application.building.steps.back_projection import BackProjectionStep
from indoor_server.application.building.steps.floor_segmentation import KeyframeRef
from indoor_server.config import settings
from indoor_server.domain.building.enums import BuildStep
from indoor_server.infrastructure.ml.model_cache import ModelCache
from indoor_server.infrastructure.ml.protocol import SegmentationOutput, SemanticSegmenter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# C 검증: 깊이 정보 없이 정확한 영역만 — 카메라 5m 이내 + 이미지 하단만
_MAX_PROJ_DISTANCE_M = 7.0
_MASK_KEEP_BOTTOM_FRACTION = 0.25  # 하단 25%만 floor로 인정


def _patch_back_projection_with_distance_cap() -> None:
    original = BackProjectionStep._back_project

    def patched(self, mask, pose, intrin, z0):  # type: ignore[no-untyped-def]
        pts = original(self, mask, pose, intrin, z0)
        if len(pts) == 0:
            return pts
        cam_origin = pose[:3, 3]
        dist = np.linalg.norm(pts - cam_origin, axis=1)
        keep = dist < _MAX_PROJ_DISTANCE_M
        return pts[keep]

    BackProjectionStep._back_project = patched  # type: ignore[assignment]

_SERVER_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVER_ROOT.parent

# ARKit Y-up → pipeline Z-up. R_x(+90°) = [[1,0,0],[0,0,-1],[0,1,0]].
# (이전 R_x(-90°)은 Z가 down되는 frame이라 z0 부호가 반대로 잡혔음)
_AXIS_SWAP_3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
_AXIS_SWAP_4 = np.eye(4, dtype=np.float64)
_AXIS_SWAP_4[:3, :3] = _AXIS_SWAP_3

# 5%ile 카메라 z(=손높이 최저) → 실제 floor: 추가로 -1.0m 오프셋 (평균 손↔바닥 거리)
_FLOOR_Z_OFFSET_M = -1.0


def _patch_estimate_z0_with_offset() -> None:
    """WalkableGridStep.estimate_z0 결과에 floor_offset 적용."""
    from indoor_server.application.building.steps.walkable_grid import WalkableGridStep
    original = WalkableGridStep.estimate_z0

    def patched(self, tz_values):  # type: ignore[no-untyped-def]
        raw = original(self, tz_values)
        return raw + _FLOOR_Z_OFFSET_M

    WalkableGridStep.estimate_z0 = patched  # type: ignore[assignment]


def _patch_intrinsics_scale(scale: float) -> None:
    """default_intrinsics fx/fy를 scale배 — 카메라 lens 변경 시뮬."""
    from indoor_server.application.building.steps import back_projection as bp
    original = bp.default_intrinsics

    def patched(image_w, image_h):  # type: ignore[no-untyped-def]
        intrin = original(image_w, image_h)
        return bp.Intrinsics(
            fx=intrin.fx * scale,
            fy=intrin.fy * scale,
            cx=intrin.cx,
            cy=intrin.cy,
        )

    bp.default_intrinsics = patched  # type: ignore[assignment]


def _patch_morph_opening(kernel_size: int = 5) -> None:
    """walkable_grid의 morphology를 opening + closing으로 교체 — 작은 톱니 제거."""
    from indoor_server.application.building.steps import walkable_grid as wg

    def patched_morph(self, mask):  # type: ignore[no-untyped-def]
        from scipy.ndimage import binary_closing, binary_opening
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        opened = binary_opening(mask, structure=kernel)
        closed = binary_closing(opened, structure=kernel)
        return closed.astype(bool)

    wg.WalkableGridStep._morph_close = patched_morph  # type: ignore[assignment]


def _patch_morph_closing_only(kernel_size: int = 7) -> None:
    """sparse 포인트용: closing만 (dilate → erode). 흩어진 cell을 연결."""
    from indoor_server.application.building.steps import walkable_grid as wg

    def patched_morph(self, mask):  # type: ignore[no-untyped-def]
        from scipy.ndimage import binary_closing
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        return binary_closing(mask, structure=kernel).astype(bool)

    wg.WalkableGridStep._morph_close = patched_morph  # type: ignore[assignment]


def _patch_depth_calibration_threshold(bottom_fraction: float = 0.85) -> None:
    """ScaleCalibrator의 bottom_thresh 만 변경. 0.85 = 하단 15%, 0.5 = 하단 50%, 0 = 전체."""
    from indoor_server.application.building.steps.depth_back_projection import ScaleCalibrator

    def new_method(self, depth_relative, floor_mask, pose, intrin, z0):  # type: ignore[no-untyped-def]
        h, w = depth_relative.shape
        cam_tz = float(pose[2, 3])
        camera_height = cam_tz - z0
        if camera_height <= 0:
            return None
        ys, xs = np.where(floor_mask)
        if len(ys) == 0:
            return None
        bottom_thresh = int(h * bottom_fraction)
        keep = ys >= bottom_thresh
        ys = ys[keep]
        xs = xs[keep]
        if len(ys) < 50:
            return None
        depths = depth_relative[ys, xs]
        valid = depths > 1e-6
        if valid.sum() < 50:
            return None
        ys, xs, depths = ys[valid], xs[valid], depths[valid]
        rys = (ys - intrin.cy) / intrin.fy
        rxs = (xs - intrin.cx) / intrin.fx
        rays_cam = np.stack([rxs, rys, np.ones_like(rxs)], axis=1)
        rotation = pose[:3, :3]
        rays_world = rays_cam @ rotation.T
        dz = rays_world[:, 2]
        valid2 = np.abs(dz) > 1e-6
        if valid2.sum() < 50:
            return None
        rays_world, depths = rays_world[valid2], depths[valid2]
        ray_lengths = np.abs(camera_height / rays_world[:, 2])
        scales = ray_lengths / depths
        return float(np.median(scales))

    ScaleCalibrator._bottom_floor_median = new_method  # type: ignore[assignment]


def _patch_z_tolerance(value: float) -> None:
    """DepthAwareBackProjectionStep의 z_tolerance default 패치."""
    from indoor_server.application.building.steps.depth_back_projection import DepthAwareBackProjectionStep
    original_init = DepthAwareBackProjectionStep.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["z_tolerance_m"] = value
        original_init(self, *args, **kwargs)

    DepthAwareBackProjectionStep.__init__ = patched_init  # type: ignore[assignment]


def _patch_obs_threshold(value: int) -> None:
    """WalkableGridStep 인스턴스의 _obs_threshold를 강제 적용 (default override)."""
    from indoor_server.application.building.steps.walkable_grid import WalkableGridStep
    original_init = WalkableGridStep.__init__

    def patched_init(self, cell_size=0.10, obs_threshold=3):  # type: ignore[no-untyped-def]
        original_init(self, cell_size=cell_size, obs_threshold=value)

    WalkableGridStep.__init__ = patched_init  # type: ignore[assignment]


class UprightRotatedSegmenter:
    """Segmenter wrapper — 입력 이미지를 90° CW로 회전해서 upright로 만든 뒤
    내부 segmenter 호출, 결과 class_mask는 90° CCW로 되돌려 sensor frame 보존.

    iOS export가 JPG를 upright로 회전해서 보내는 경우를 시뮬레이션.
    """

    def __init__(self, inner: SemanticSegmenter, keep_bottom_fraction: float = _MASK_KEEP_BOTTOM_FRACTION) -> None:
        self._inner = inner
        self._keep_bottom_fraction = keep_bottom_fraction

    async def segment(self, image: np.ndarray) -> SegmentationOutput:
        import cv2
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        out = await self._inner.segment(rotated)
        mask_upright = out.class_mask.copy()
        if self._keep_bottom_fraction < 1.0:
            # upright frame에서 하단 _keep_bottom_fraction 만 keep (depth NN 없이 보수)
            h_up, _ = mask_upright.shape
            keep_h = int(h_up * self._keep_bottom_fraction)
            mask_upright[:h_up - keep_h, :] = 0
        mask_back = cv2.rotate(mask_upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return SegmentationOutput(class_mask=mask_back.astype(np.int32))


class BottomHalfFloorSegmenter:
    """이미지 일부 영역을 ADE20K floor(class 3)로 라벨링하는 stub.

    iOS app이 JPG를 sensor-native landscape로 저장(폰을 세로로 들어도 JPG는
    1920x1440 가로). 폰이 90° CCW 회전된 상태로 찍힌 거라면 실제 floor는
    JPG의 좌측(또는 우측)에 위치 — `--floor-side` 환경변수로 지정.
    """

    def __init__(self, side: str = "left", fraction: float = 0.25) -> None:
        self._side = side
        self._fraction = fraction

    async def segment(self, image: np.ndarray) -> SegmentationOutput:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.int32)
        if self._side == "left":
            mask[:, :int(w * self._fraction)] = 3
        elif self._side == "right":
            mask[:, int(w * (1 - self._fraction)):] = 3
        elif self._side == "top":
            mask[:int(h * self._fraction), :] = 3
        else:  # bottom
            mask[int(h * (1 - self._fraction)):, :] = 3
        return SegmentationOutput(class_mask=mask)


def _convert_pose_yup_to_zup(pose_bytes: bytes) -> bytes:
    values = struct.unpack_from("<16f", pose_bytes)
    pose_old = np.array(values, dtype=np.float64).reshape(4, 4, order="F")
    pose_new = _AXIS_SWAP_4 @ pose_old
    flat = pose_new.flatten(order="F").astype(np.float32)
    return struct.pack("<16f", *flat.tolist())


def _load_keyframes(sidecar_db: Path, scan_uuid: UUID, storage_root: Path) -> list[KeyframeRef]:
    conn = sqlite3.connect(str(sidecar_db))
    try:
        # Sprint 49: rtabmap_node_id 도 함께 SELECT — POI frame transform 의
        # ARKit↔RTABMap pair 매칭 input.
        try:
            rows = conn.execute(
                "SELECT seq, image_path, pose_matrix, tx, ty, tz, rtabmap_node_id "
                "FROM keyframe_meta ORDER BY seq"
            ).fetchall()
            has_node_id = True
        except sqlite3.OperationalError:
            # v1 schema 는 rtabmap_node_id 컬럼 없음 — fallback 으로 None.
            rows = conn.execute(
                "SELECT seq, image_path, pose_matrix, tx, ty, tz "
                "FROM keyframe_meta ORDER BY seq"
            ).fetchall()
            has_node_id = False
    finally:
        conn.close()

    refs: list[KeyframeRef] = []
    for row in rows:
        if has_node_id:
            seq, image_path, pose_blob, tx, ty, tz, node_id = row
        else:
            seq, image_path, pose_blob, tx, ty, tz = row
            node_id = None
        rel = f"scans/{scan_uuid}/keyframes/{Path(image_path).name}"
        full = storage_root / rel
        if not full.exists():
            continue
        new_pose = _convert_pose_yup_to_zup(bytes(pose_blob))
        refs.append(
            KeyframeRef(
                scan_id=scan_uuid,
                seq=seq,
                image_path=rel,
                tx=float(tx),
                ty=float(-tz),    # R_x(+90°): new ty = -old tz
                tz=float(ty),     # R_x(+90°): new tz = old ty
                pose_matrix=new_pose,
                rtabmap_node_id=int(node_id) if node_id is not None else None,
            )
        )
    return refs


def _stage_keyframes(scan_dir: Path, scan_uuid: UUID, storage_root: Path) -> None:
    """원본 keyframes/ 를 storage_root/{scan_uuid}/keyframes/ 로 심볼릭 스테이징."""
    target = storage_root / str(scan_uuid) / "keyframes"
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    src = scan_dir / "keyframes"
    target.symlink_to(src.resolve())


def _dump_floor_pointcloud_evidence(
    *,
    evidence_dir: Path,
    outcome: object,
) -> None:
    """Sprint 46~50: floor_pointcloud_report.json + cad_report.json +
    before_cad.geojson + after_cad.geojson + before_after_cad.png + AC 자동 평가.

    Sprint 50 추가: rectangle_cover_report.json + rectangles_overlay.png +
    before_after_rectangle.png + poi_frame_transform.json + zip_size_report.json.
    """
    import json

    evidence_dir.mkdir(parents=True, exist_ok=True)
    counts = getattr(outcome, "counts", None)
    if counts is None:
        return

    # Sprint 50 — rectangle cover evidence (always dumped if metadata exists).
    rectangle_cover = getattr(counts, "rectangle_cover", None)
    if isinstance(rectangle_cover, dict):
        rc_path = evidence_dir / "rectangle_cover_report.json"
        # iteration_log + params + summary 모두 직렬화.
        with rc_path.open("w") as f:
            json.dump(rectangle_cover, f, indent=2, default=str)
        print(f"[rectangle_cover] evidence saved: {rc_path}")

        # rectangles_overlay.png: rectangles + footprint union.
        # rectangle metadata 의 row0/col0/row1/col1 + cell_size 로 world polygon
        # 을 재구성하기엔 angle/origin 정보가 부족하다. 따라서 BuildCounts 에
        # surface 된 rectangle metadata 기반으로 단순한 footprint preview 만
        # 그려둔다 (evidence_dir/rectangles_overlay.png 는 footprint_geojson
        # 단일 패널).
        rects_meta_list = rectangle_cover.get("rectangles") or []
        footprint_geojson = getattr(counts, "footprint_geojson", None)
        if isinstance(footprint_geojson, dict):
            try:
                _render_rectangle_footprint_panel(
                    output_path=evidence_dir / "rectangles_overlay.png",
                    footprint_geojson=footprint_geojson,
                    rectangle_metadata_list=(
                        rects_meta_list
                        if isinstance(rects_meta_list, list)
                        else []
                    ),
                    summary=rectangle_cover,
                )
                print(
                    f"[rectangle_cover] overlay png saved: "
                    f"{evidence_dir / 'rectangles_overlay.png'}"
                )
            except Exception as e:
                print(f"[rectangle_cover] overlay png FAILED: {e}")

        # before_after_rectangle.png: Sprint 49 hint chain (or raw raster) vs
        # Sprint 50 rectangle union 비교.
        floor_raster_meta = getattr(counts, "floor_raster", None)
        before_geojson = None
        if isinstance(floor_raster_meta, dict):
            before_geojson = floor_raster_meta.get("footprint_geojson")
        if isinstance(before_geojson, dict) and isinstance(footprint_geojson, dict):
            try:
                _render_before_after_rectangle_png(
                    output_path=evidence_dir / "before_after_rectangle.png",
                    before_geojson=before_geojson,
                    after_geojson=footprint_geojson,
                    summary=rectangle_cover,
                )
                print(
                    f"[rectangle_cover] before_after png saved: "
                    f"{evidence_dir / 'before_after_rectangle.png'}"
                )
            except Exception as e:
                print(f"[rectangle_cover] before_after png FAILED: {e}")

    # Sprint 50 — poi_frame_transform.json (Sprint 49 BuildCounts 필드 dump).
    poi_frame_transform = getattr(counts, "poi_frame_transform", None)
    if isinstance(poi_frame_transform, dict):
        pft_path = evidence_dir / "poi_frame_transform.json"
        with pft_path.open("w") as f:
            json.dump(poi_frame_transform, f, indent=2, default=str)
        print(f"[poi_frame_transform] evidence saved: {pft_path}")

    # Sprint 50 — zip_size_report.json (iOS export ratio placeholder; iOS 미수정
    # sprint 라 실 측정값은 이전 sprint 49 evidence 와 동일하게 기록).
    zip_size_report = {
        "source": "ios_export_static_record",
        "sprint": 50,
        "ios_changes": "none — sprint50 server-only",
        "note": (
            "Sprint 49 measured rtabmap_db_bytes_per_keyframe=0.84MB. "
            "iOS export contract unchanged in sprint 50."
        ),
    }
    zsr_path = evidence_dir / "zip_size_report.json"
    with zsr_path.open("w") as f:
        json.dump(zip_size_report, f, indent=2, default=str)
    print(f"[zip_size_report] evidence saved: {zsr_path}")

    rectification = getattr(counts, "rectification", None)
    cad_cleanup = getattr(counts, "polygon_cad_cleanup", None)
    hint_chain = getattr(counts, "dominant_angle_hint", None)
    cad_effect_pass = getattr(counts, "cad_effect_pass", None)
    cad_effect_small_polygon_pass = getattr(
        counts, "cad_effect_small_polygon_pass", None
    )

    report = {
        "build_source": getattr(counts, "build_source", None),
        "passed_quality_gate": getattr(outcome, "passed_quality_gate", None),
        "failure_reason": str(getattr(outcome, "failure_reason", None) or ""),
        "walkable_cells": getattr(counts, "walkable_cells", 0),
        "map_nodes": getattr(counts, "map_nodes", 0),
        "map_edges": getattr(counts, "map_edges", 0),
        "connected_components": getattr(counts, "connected_components", 0),
        "walkable_coverage": getattr(counts, "walkable_coverage", 0.0),
        "floor_z0": getattr(counts, "floor_z0", None),
        "floor_pointcloud": getattr(counts, "floor_pointcloud", None),
        "floor_raster": getattr(counts, "floor_raster", None),
        "rectification": rectification,
        "polygon_cad_cleanup": cad_cleanup,
        "dominant_angle_hint": hint_chain,
        "footprint_geojson_present": (
            getattr(counts, "footprint_geojson", None) is not None
        ),
    }
    out_path = evidence_dir / "floor_pointcloud_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[floor_pointcloud] evidence saved: {out_path}")

    # Sprint 47/48 cad_report.json + AC 자동 평가
    cad_report = _build_cad_report(
        rectification,
        cad_cleanup,
        hint_chain=hint_chain,
        cad_effect_pass=cad_effect_pass,
        cad_effect_small_polygon_pass=cad_effect_small_polygon_pass,
    )
    cad_path = evidence_dir / "cad_report.json"
    with cad_path.open("w") as f:
        json.dump(cad_report, f, indent=2, default=str)
    print(f"[cad_cleanup] evidence saved: {cad_path}")

    # before_cad.geojson — raw raster contour (cleanup 진입 직전).
    # `floor_raster` metadata 의 footprint_geojson 또는 raster contour 기반.
    floor_raster = getattr(counts, "floor_raster", None)
    raw_geojson = None
    if isinstance(floor_raster, dict):
        # FloorRasterStep metadata 가 footprint_geojson 을 포함하지 않을 수
        # 있어 footprint_geojson 자체에서 가져온다 (cleanup 비활성 시 raw).
        raw_geojson = floor_raster.get("footprint_geojson")
    # Sprint 48: rectification 진입 직전 raw 가 별도 보관되지 않으면 cleaned
    # 와 동일 (cleanup OFF 모드). 그래도 before 패널을 raster contour 로 그릴 수
    # 있도록 footprint_geojson (cleanup 후) 와 동일값을 쓰는 fallback.
    if raw_geojson is None:
        raw_geojson = getattr(counts, "footprint_geojson", None)
    if raw_geojson is not None:
        raw_path = evidence_dir / "before_cad.geojson"
        with raw_path.open("w") as f:
            json.dump(raw_geojson, f, indent=2, default=str)
        print(f"[cad_cleanup] before_cad geojson saved: {raw_path}")

    cleaned_geojson = getattr(counts, "footprint_geojson", None)
    if cleaned_geojson is not None:
        cleaned_path = evidence_dir / "after_cad.geojson"
        with cleaned_path.open("w") as f:
            json.dump(cleaned_geojson, f, indent=2, default=str)
        print(f"[cad_cleanup] after_cad geojson saved: {cleaned_path}")

    # Sprint 48 (Codex W-7): before/after PNG. raw raster contour 와 cleaned
    # polygon 을 동일 axis/scale 로 같이 그린다.
    if raw_geojson is not None and cleaned_geojson is not None:
        png_path = evidence_dir / "before_after_cad.png"
        try:
            _render_before_after_cad_png(
                output_path=png_path,
                raw_geojson=raw_geojson,
                cleaned_geojson=cleaned_geojson,
                rectification=rectification,
                cleanup=cad_cleanup,
                hint_chain=hint_chain,
            )
            print(f"[cad_cleanup] before_after_cad png saved: {png_path}")
        except Exception as e:
            print(f"[cad_cleanup] before_after_cad png FAILED: {e}")


def _build_cad_report(
    rectification: object | None,
    cad_cleanup: object | None,
    *,
    hint_chain: object | None = None,
    cad_effect_pass: bool | None = None,
    cad_effect_small_polygon_pass: bool | None = None,
) -> dict[str, object]:
    """Sprint 47/48 AC 자동 평가용 cad_report.json 본문.

    Codex F-2 정의 cad_effect_pass 는 pipeline 에서 계산된 값을 우선 사용하고,
    pipeline 가 없으면 rectification + cleanup metadata 로 재계산.
    """
    import json as _json

    rec_dict: dict[str, object]
    if isinstance(rectification, dict):
        rec_dict = dict(rectification)
    else:
        rec_dict = {}

    cad_dict: dict[str, object]
    if isinstance(cad_cleanup, dict):
        cad_dict = dict(cad_cleanup)
    else:
        cad_dict = {}

    hint_dict: dict[str, object]
    if isinstance(hint_chain, dict):
        hint_dict = dict(hint_chain)
    else:
        hint_dict = {}

    # AC-1 (Codex F-2): hint OR four_way path 정의.
    accepted = bool(rec_dict.get("accepted") is True)
    fallback_used = bool(rec_dict.get("fallback_used") is True)
    forced_rectilinear_used = bool(rec_dict.get("forced_rectilinear_used") is True)
    snap_mode_used = rec_dict.get("snap_mode_used")
    input_source = cad_dict.get("input_source")
    ortho = float(cad_dict.get("corner_orthogonality_ratio", 0.0))  # type: ignore[arg-type]
    collinear_residual = int(cad_dict.get("collinear_residual_count", 0))  # type: ignore[arg-type]
    before = int(cad_dict.get("vertex_count_before", 0))  # type: ignore[arg-type]
    after = int(cad_dict.get("vertex_count_after", 0))  # type: ignore[arg-type]

    ac1_pass = bool(accepted and not fallback_used)

    # AC-2 (Codex F-2 + W-3): orthogonality_strict
    ac2_pass = ortho >= 0.90 and collinear_residual == 0

    # AC-3 (W-2): vertex 감소 분기
    short = int(cad_dict.get("short_edges_pruned_count", 0))  # type: ignore[arg-type]
    near = int(cad_dict.get("near_vertices_merged_count", 0))  # type: ignore[arg-type]
    coll = int(cad_dict.get("collinear_merged_count", 0))  # type: ignore[arg-type]
    if before >= 16:
        ac3_pass = after <= int(before * 0.5)
    elif before > 0:
        ac3_pass = (after <= before) and (short + near + coll > 0)
    else:
        ac3_pass = False

    # Codex F-2 cad_effect_pass — pipeline 값 우선, 없으면 재계산.
    if cad_effect_pass is None:
        cad_effect_pass = bool(
            ac1_pass
            and forced_rectilinear_used
            and snap_mode_used in ("hint", "four_way")
            and input_source == "rectified"
            and ortho >= 0.90
            and collinear_residual == 0
            and before >= 16
            and after <= int(before * 0.5)
        )
    if cad_effect_small_polygon_pass is None:
        cleanup_changed = bool(cad_dict.get("cleanup_changed", False))
        if 0 < before < 16:
            cad_effect_small_polygon_pass = bool(
                cleanup_changed and after <= before and ortho >= 0.90
            )
        else:
            cad_effect_small_polygon_pass = False

    # W-1 shape-preservation guard
    iou = float(cad_dict.get("iou_raw_rectified", 0.0))  # type: ignore[arg-type]
    centroid_shift = float(cad_dict.get("centroid_shift_m", 0.0))  # type: ignore[arg-type]
    bbox_w = float(cad_dict.get("bbox_width_ratio", 1.0))  # type: ignore[arg-type]
    bbox_h = float(cad_dict.get("bbox_height_ratio", 1.0))  # type: ignore[arg-type]
    iou_ok = iou >= 0.60
    centroid_ok = centroid_shift <= 0.30
    bbox_ok = (0.65 <= bbox_w <= 1.35) and (0.65 <= bbox_h <= 1.35)
    shape_guard_pass_count = int(iou_ok) + int(centroid_ok) + int(bbox_ok)
    shape_guard_pass = shape_guard_pass_count >= 2

    # W-7 minimum guards
    component_count_after = int(cad_dict.get("component_count_after", 0))  # type: ignore[arg-type]
    polygon_area = float(cad_dict.get("polygon_area_m2", 0.0))  # type: ignore[arg-type]
    minimum_pass = (component_count_after == 1) and (polygon_area >= 3.0)

    # W-8 dominant angle confidence
    dominant_confidence = float(rec_dict.get("dominant_angle_confidence", 0.0))  # type: ignore[arg-type]
    low_confidence = bool(rec_dict.get("low_angle_confidence", False))

    # Codex F-5 — params dump (모든 threshold contract)
    params_dump = {
        "manhattan_max_area_change_floor_pointcloud": rec_dict.get(
            "area_change_threshold_used"
        ),
        "polygon_cad_params": cad_dict.get("params"),
        "hint_chain_params": hint_dict.get("params"),
        # Codex 권고 9: single_scan_calibrated 표시
        "single_scan_calibrated": True,
    }

    return _json.loads(
        _json.dumps(
            {
                "rectification": rec_dict,
                "polygon_cad_cleanup": cad_dict,
                "dominant_angle_hint_chain": hint_dict,
                "acceptance": {
                    "AC-1_rectification_accepted": ac1_pass,
                    "AC-2_orthogonality_strict": ac2_pass,
                    "AC-3_vertex_reduction": ac3_pass,
                    # Sprint 48 (Codex F-2)
                    "cad_effect_pass": bool(cad_effect_pass),
                    "cad_effect_small_polygon_pass": bool(
                        cad_effect_small_polygon_pass
                    ),
                    "cad_effect_breakdown": {
                        "rectification_accepted": accepted,
                        "fallback_not_used": not fallback_used,
                        "forced_rectilinear_used": forced_rectilinear_used,
                        "snap_mode_in_hint_or_four_way": snap_mode_used
                        in ("hint", "four_way"),
                        "input_source_rectified": input_source == "rectified",
                        "orthogonality_ge_090": ortho >= 0.90,
                        "collinear_residual_zero": collinear_residual == 0,
                        "vertex_reduction_ge_050": (
                            before >= 16 and after <= int(before * 0.5)
                        ),
                    },
                },
                "shape_preservation_guard": {
                    "iou_raw_rectified": iou,
                    "iou_pass": iou_ok,
                    "centroid_shift_m": centroid_shift,
                    "centroid_pass": centroid_ok,
                    "bbox_width_ratio": bbox_w,
                    "bbox_height_ratio": bbox_h,
                    "bbox_pass": bbox_ok,
                    "pass_count": shape_guard_pass_count,
                    "passed": shape_guard_pass,
                },
                "minimum_guard": {
                    "component_count_after": component_count_after,
                    "polygon_area_m2": polygon_area,
                    "passed": minimum_pass,
                },
                "dominant_angle_confidence": {
                    "value": dominant_confidence,
                    "low_confidence": low_confidence,
                },
                "params": params_dump,
            },
            default=str,
        )
    )


def _polygon_coords_iter(geojson: dict[str, object]) -> list[list[tuple[float, float]]]:
    """Polygon/MultiPolygon GeoJSON 에서 outer ring coords list 만 추출."""
    if not isinstance(geojson, dict):
        return []
    typ = geojson.get("type")
    coords = geojson.get("coordinates")
    rings: list[list[tuple[float, float]]] = []
    if typ == "Polygon" and isinstance(coords, list) and coords:
        outer = coords[0]
        if isinstance(outer, list):
            rings.append([(float(p[0]), float(p[1])) for p in outer if len(p) >= 2])
    elif typ == "MultiPolygon" and isinstance(coords, list):
        for poly in coords:
            if isinstance(poly, list) and poly:
                outer = poly[0]
                if isinstance(outer, list):
                    rings.append(
                        [(float(p[0]), float(p[1])) for p in outer if len(p) >= 2]
                    )
    return rings


def _render_before_after_cad_png(
    *,
    output_path: Path,
    raw_geojson: dict[str, object],
    cleaned_geojson: dict[str, object],
    rectification: object | None,
    cleanup: object | None,
    hint_chain: object | None,
) -> None:
    """Sprint 48 (Codex W-7): before/after PNG.

    Pillow + numpy 만 사용 (matplotlib 의존성 추가 없음). 두 패널 동일 axis/scale.
    before 패널 = raw raster contour, after 패널 = cleanup 후 polygon.
    """
    from PIL import Image, ImageDraw, ImageFont

    raw_rings = _polygon_coords_iter(raw_geojson)
    after_rings = _polygon_coords_iter(cleaned_geojson)
    if not raw_rings and not after_rings:
        raise ValueError("no polygon coordinates in raw/cleaned")

    # bounding box 동일 axis/scale
    all_pts: list[tuple[float, float]] = []
    for ring in raw_rings + after_rings:
        all_pts.extend(ring)
    if not all_pts:
        raise ValueError("empty point list")
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    margin = 0.05 * max(span_x, span_y)
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    # canvas 크기 (각 패널 800x800)
    panel_w = 800
    panel_h = 800
    title_h = 40
    img = Image.new(
        "RGB",
        (panel_w * 2 + 20, panel_h + title_h + 20),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    span_x_now = max(x_max - x_min, 1e-6)
    span_y_now = max(y_max - y_min, 1e-6)
    scale = min(panel_w / span_x_now, panel_h / span_y_now) * 0.92

    def _world_to_panel(
        x: float, y: float, panel_offset_x: int
    ) -> tuple[float, float]:
        # 좌상단 origin 으로 변환 (y flip)
        px = panel_offset_x + (x - x_min) * scale + (panel_w - span_x_now * scale) / 2
        py = title_h + 10 + (y_max - y) * scale + (panel_h - span_y_now * scale) / 2
        return (px, py)

    def _draw_rings(
        rings: list[list[tuple[float, float]]],
        panel_offset_x: int,
        color: tuple[int, int, int],
    ) -> None:
        for ring in rings:
            if len(ring) < 2:
                continue
            pts = [_world_to_panel(x, y, panel_offset_x) for x, y in ring]
            # 닫힘
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            draw.line(pts, fill=color, width=2)
            # vertex 작은 점
            for px, py in pts:
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)

    # 회색 frame
    draw.rectangle(
        (5, title_h + 5, panel_w + 5, title_h + panel_h + 5),
        outline=(200, 200, 200),
        width=1,
    )
    draw.rectangle(
        (panel_w + 15, title_h + 5, panel_w * 2 + 15, title_h + panel_h + 5),
        outline=(200, 200, 200),
        width=1,
    )

    # before 패널
    _draw_rings(raw_rings, 5, (200, 60, 60))
    raw_vertex = sum(max(0, len(r) - 1) for r in raw_rings)
    title_left = f"raw raster contour (vertex {raw_vertex})"
    draw.text((20, 10), title_left, fill=(60, 60, 60), font=font)

    # after 패널
    _draw_rings(after_rings, panel_w + 15, (60, 120, 200))
    after_vertex = sum(max(0, len(r) - 1) for r in after_rings)
    rec_dict = rectification if isinstance(rectification, dict) else {}
    cl_dict = cleanup if isinstance(cleanup, dict) else {}
    hint_dict = hint_chain if isinstance(hint_chain, dict) else {}
    chosen = hint_dict.get("chosen_source") if isinstance(hint_dict, dict) else None
    snap_mode = rec_dict.get("snap_mode_used")
    ortho = cl_dict.get("corner_orthogonality_ratio")
    title_right = (
        f"after CAD cleanup (vertex {after_vertex}, "
        f"snap={snap_mode}, hint={chosen}, ortho={ortho})"
    )
    draw.text((panel_w + 30, 10), title_right, fill=(60, 60, 60), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="PNG")


def _render_rectangle_footprint_panel(
    *,
    output_path: Path,
    footprint_geojson: dict[str, object],
    rectangle_metadata_list: list[object],
    summary: dict[str, object],
) -> None:
    """Sprint 50 — single panel preview: footprint union + summary text."""
    from PIL import Image, ImageDraw, ImageFont

    rings = _polygon_coords_iter(footprint_geojson)
    if not rings:
        raise ValueError("footprint_geojson has no rings")
    all_pts = [pt for ring in rings for pt in ring]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    margin = 0.05 * max(span_x, span_y)
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    panel_w = 800
    panel_h = 800
    title_h = 80
    img = Image.new(
        "RGB",
        (panel_w + 20, panel_h + title_h + 20),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    span_x_now = max(x_max - x_min, 1e-6)
    span_y_now = max(y_max - y_min, 1e-6)
    scale = min(panel_w / span_x_now, panel_h / span_y_now) * 0.92

    def _wp(x: float, y: float) -> tuple[float, float]:
        px = 10 + (x - x_min) * scale + (panel_w - span_x_now * scale) / 2
        py = title_h + 10 + (y_max - y) * scale + (panel_h - span_y_now * scale) / 2
        return (px, py)

    # union polygon fill
    for ring in rings:
        if len(ring) < 3:
            continue
        pts = [_wp(x, y) for x, y in ring]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        draw.polygon(pts, fill=(180, 220, 200), outline=(60, 120, 80))

    rcount = int(summary.get("rectangle_count", len(rectangle_metadata_list)))
    recall = summary.get("recall")
    over = summary.get("over_cover_ratio")
    accepted = summary.get("accepted")
    fb = summary.get("fallback_used")
    angles = summary.get("selected_angles_unique", [])
    if isinstance(angles, list):
        angles_str = ",".join(str(a) for a in angles[:6])
    else:
        angles_str = ""
    title = (
        f"rectangle dictionary cover — rects={rcount} "
        f"recall={recall} over={over} accepted={accepted} "
        f"fallback={fb}\n"
        f"angles={angles_str}"
    )
    draw.text((20, 10), title, fill=(60, 60, 60), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="PNG")


def _render_before_after_rectangle_png(
    *,
    output_path: Path,
    before_geojson: dict[str, object],
    after_geojson: dict[str, object],
    summary: dict[str, object],
) -> None:
    """Sprint 50 — before (raw raster contour) vs after (rectangle union)."""
    _render_before_after_cad_png(
        output_path=output_path,
        raw_geojson=before_geojson,
        cleaned_geojson=after_geojson,
        rectification={"snap_mode_used": "rectangle_cover"},
        cleanup={
            "corner_orthogonality_ratio": 1.0,
            "vertex_count_after": sum(
                max(0, len(r) - 1)
                for r in _polygon_coords_iter(after_geojson)
            ),
        },
        hint_chain={"chosen_source": f"rectangle_cover_recall_{summary.get('recall')}"},
    )


def _render_rectangles_overlay_png(
    *,
    output_path: Path,
    raw_heatmap: np.ndarray,
    rectangles_world: list[list[tuple[float, float]]],
    union_geojson: dict[str, object] | None,
    grid_origin: tuple[float, float, float],
    metadata: dict[str, object],
) -> None:
    """Sprint 50 — 3-panel: raw heatmap / rectangles overlay / merged polygon.

    raw_heatmap: source grid observation_count (H, W).
    rectangles_world: each entry is list of (x, y) world-frame corners.
    union_geojson: MultiPolygon GeoJSON.
    grid_origin: (x0, y0, cell_size_m).
    """
    from PIL import Image, ImageDraw, ImageFont

    x0, y0, cs = grid_origin
    h, w = raw_heatmap.shape

    # world bbox (heatmap + rectangles + union 모두 포함)
    all_pts: list[tuple[float, float]] = [
        (x0, y0),
        (x0 + w * cs, y0),
        (x0 + w * cs, y0 + h * cs),
        (x0, y0 + h * cs),
    ]
    for rect_pts in rectangles_world:
        all_pts.extend(rect_pts)
    if union_geojson is not None:
        for ring in _polygon_coords_iter(union_geojson):
            all_pts.extend(ring)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    margin = 0.05 * max(span_x, span_y)
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    panel_w = 600
    panel_h = 600
    title_h = 40
    img = Image.new(
        "RGB",
        (panel_w * 3 + 30, panel_h + title_h + 20),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    span_x_now = max(x_max - x_min, 1e-6)
    span_y_now = max(y_max - y_min, 1e-6)
    scale = min(panel_w / span_x_now, panel_h / span_y_now) * 0.92

    def _world_to_panel(
        x: float, y: float, panel_offset_x: int
    ) -> tuple[float, float]:
        px = panel_offset_x + (x - x_min) * scale + (panel_w - span_x_now * scale) / 2
        py = title_h + 10 + (y_max - y) * scale + (panel_h - span_y_now * scale) / 2
        return (px, py)

    # ── 패널 1: raw heatmap ─────────────────────────────────────────────────
    max_count = max(int(raw_heatmap.max()), 1)
    for r in range(h):
        for c in range(w):
            cnt = int(raw_heatmap[r, c])
            if cnt <= 0:
                continue
            world_x = x0 + c * cs
            world_y = y0 + r * cs
            # cell rectangle (world coords)
            tl = _world_to_panel(world_x, world_y + cs, 5)
            br = _world_to_panel(world_x + cs, world_y, 5)
            ratio = cnt / max_count
            # blue gradient
            color = (
                int(220 - 180 * ratio),
                int(220 - 180 * ratio),
                255,
            )
            draw.rectangle(
                (tl[0], tl[1], br[0], br[1]),
                fill=color,
                outline=None,
            )
    draw.rectangle(
        (5, title_h + 5, panel_w + 5, title_h + panel_h + 5),
        outline=(200, 200, 200),
        width=1,
    )
    title_a = f"raw obs heatmap (max={max_count})"
    draw.text((20, 10), title_a, fill=(60, 60, 60), font=font)

    # ── 패널 2: rectangles overlay ──────────────────────────────────────────
    panel2_x = panel_w + 15
    # heatmap 배경 (light gray)
    for r in range(h):
        for c in range(w):
            if int(raw_heatmap[r, c]) <= 0:
                continue
            world_x = x0 + c * cs
            world_y = y0 + r * cs
            tl = _world_to_panel(world_x, world_y + cs, panel2_x)
            br = _world_to_panel(world_x + cs, world_y, panel2_x)
            draw.rectangle(
                (tl[0], tl[1], br[0], br[1]),
                fill=(230, 230, 230),
                outline=None,
            )
    # rectangle outlines
    for rect_pts in rectangles_world:
        if len(rect_pts) < 3:
            continue
        pts = [_world_to_panel(x, y, panel2_x) for x, y in rect_pts]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        draw.line(pts, fill=(200, 60, 60), width=2)
    draw.rectangle(
        (panel2_x, title_h + 5, panel2_x + panel_w, title_h + panel_h + 5),
        outline=(200, 200, 200),
        width=1,
    )
    rcount = int(metadata.get("rectangle_count", 0))
    title_b = f"rectangles ({rcount} selected)"
    draw.text((panel2_x + 15, 10), title_b, fill=(60, 60, 60), font=font)

    # ── 패널 3: merged polygon ──────────────────────────────────────────────
    panel3_x = panel_w * 2 + 25
    if union_geojson is not None:
        for ring in _polygon_coords_iter(union_geojson):
            if len(ring) < 2:
                continue
            pts = [_world_to_panel(x, y, panel3_x) for x, y in ring]
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            draw.polygon(pts, fill=(180, 220, 200), outline=(60, 120, 80))
    draw.rectangle(
        (panel3_x, title_h + 5, panel3_x + panel_w, title_h + panel_h + 5),
        outline=(200, 200, 200),
        width=1,
    )
    recall = metadata.get("recall")
    over = metadata.get("over_cover_ratio")
    title_c = f"merged polygon (recall={recall}, over={over})"
    draw.text((panel3_x + 15, 10), title_c, fill=(60, 60, 60), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="PNG")


async def _run_cad_parameter_sweep(
    *,
    evidence_dir: Path,
    scan_uuid: UUID,
    segmenter: SemanticSegmenter,
    storage_root: Path,
    rtabmap_nodes: object,
    rtabmap_frames: object,
    scan_dir: Path,
    args: object,
    area_values: list[float],
    short_edge_values: list[float],
) -> None:
    """Sprint 48 (Codex W-6): area_change × short_edge 2축 sweep evidence dump.

    각 (area, short_edge) 조합으로 BuildPipeline 재실행 후
    cad_parameter_sweep.json 에 결과 누적.
    """
    import json as _json

    from indoor_server.application.rtabmap.reader import RtabmapReader

    print(
        f"[cad_sweep] start area={area_values} short_edge={short_edge_values} "
        f"({len(area_values) * len(short_edge_values)} combos)"
    )

    rtabmap_db = scan_dir / "rtabmap.db"
    reader = RtabmapReader()
    sweep_links = reader.load_links(rtabmap_db) if rtabmap_db.exists() else []

    results: list[dict[str, object]] = []
    for area in area_values:
        for short_edge in short_edge_values:
            sweep_job_uuid = uuid4()
            sweep_pipeline = BuildPipeline(
                segmenter=segmenter,
                storage_root=storage_root,
                use_floor_pointcloud=True,
                floor_pointcloud_pixel_stride=args.floor_pc_pixel_stride,  # type: ignore[attr-defined]
                floor_pointcloud_height_tolerance_m=args.floor_pc_height_tolerance,  # type: ignore[attr-defined]
                floor_pointcloud_min_cell_hits=args.floor_pc_min_cell_hits,  # type: ignore[attr-defined]
                rtabmap_image_orientation_mode=args.floor_pc_orientation,  # type: ignore[attr-defined]
                polygon_cad_cleanup_enabled=True,  # sweep 은 cleanup 의도
                floor_raster_cad_morph_close_cells=args.cad_morph_close,  # type: ignore[attr-defined]
                manhattan_floor_pointcloud_max_area_change=float(area),
                polygon_cad_short_edge_min_length_m=float(short_edge),
            )

            async def _sweep_progress(_step: BuildStep, _p: float) -> None:
                pass

            async def _sweep_cancel() -> bool:
                return False

            try:
                sweep_outcome = await sweep_pipeline.execute(
                    scan_id=scan_uuid,
                    build_job_id=sweep_job_uuid,
                    keyframes=[],
                    pois=[],
                    rtabmap_nodes=rtabmap_nodes,  # type: ignore[arg-type]
                    rtabmap_links=sweep_links,
                    rtabmap_frames=rtabmap_frames,  # type: ignore[arg-type]
                    progress_sink=_sweep_progress,
                    cancel_check=_sweep_cancel,
                    debug_sink=None,
                )
            except Exception as e:
                results.append(
                    {
                        "area_change_limit": float(area),
                        "short_edge_min_length_m": float(short_edge),
                        "error": str(e),
                    }
                )
                continue

            sw_counts = sweep_outcome.counts
            rec = sw_counts.rectification or {}
            cl = sw_counts.polygon_cad_cleanup or {}
            hint_chain = sw_counts.dominant_angle_hint or {}
            cad_eff = sw_counts.cad_effect_pass
            cad_small = sw_counts.cad_effect_small_polygon_pass
            entry_report = _build_cad_report(
                rec,
                cl,
                hint_chain=hint_chain,
                cad_effect_pass=cad_eff,
                cad_effect_small_polygon_pass=cad_small,
            )
            results.append(
                {
                    "area_change_limit": float(area),
                    "short_edge_min_length_m": float(short_edge),
                    "rectification": rec,
                    "polygon_cad_cleanup": cl,
                    "dominant_angle_hint_chain": hint_chain,
                    "acceptance": entry_report["acceptance"],
                    "shape_preservation_guard": entry_report[
                        "shape_preservation_guard"
                    ],
                }
            )

    sweep_path = evidence_dir / "cad_parameter_sweep.json"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with sweep_path.open("w") as f:
        _json.dump(
            {
                "scan_id": str(scan_uuid),
                "param_axes": [
                    "manhattan_floor_pointcloud_max_area_change",
                    "polygon_cad_short_edge_min_length_m",
                ],
                "area_values": [float(v) for v in area_values],
                "short_edge_values": [float(v) for v in short_edge_values],
                "results": results,
                "single_scan_calibrated": True,
                "morph_close_fixed_at": getattr(args, "cad_morph_close", 3),
                "morph_close_note": "morph_close fixed, not validated",
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[cad_sweep] saved: {sweep_path} ({len(results)} entries)")


async def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build Phase 단발 실행")
    parser.add_argument("scan_dir", help="스캔 디렉터리 (scan_metadata.db + keyframes/ 포함)")
    parser.add_argument(
        "mode",
        nargs="?",
        default="left",
        help="segmenter 모드: left|right|top|bottom|real|real_upright (기본: left)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--depth-nn",
        action="store_true",
        help="Depth Anything v2 NN back-projection 활성 (Sprint 19 경로)",
    )
    mode_group.add_argument(
        "--triangulate",
        action="store_true",
        help="SuperPoint+LightGlue 2-view triangulation 활성 (Sprint 20 경로)",
    )
    mode_group.add_argument(
        "--adaptive-buffer",
        action="store_true",
        help=(
            "Adaptive per-frame buffer 활성 (Sprint 22). "
            "Segformer floor mask 발밑 50%% strip의 가로/세로 측정으로 per-keyframe disk buffer."
        ),
    )
    mode_group.add_argument(
        "--floor-pointcloud",
        action="store_true",
        help=(
            "Sprint 46: floor segmentation point cloud + per-pixel depth source-of-truth. "
            "rtabmap.db Node/Data/calibration 필수 + Segformer 모델 필수."
        ),
    )
    parser.add_argument(
        "--floor-pc-pixel-stride",
        type=int,
        default=4,
        help="floor_pointcloud pixel stride (default 4). sparse mask 환경에서 2로 낮춤.",
    )
    parser.add_argument(
        "--floor-pc-height-tolerance",
        type=float,
        default=0.30,
        help="floor_pointcloud height tolerance m (default 0.30).",
    )
    parser.add_argument(
        "--floor-pc-min-cell-hits",
        type=int,
        default=2,
        help="floor_raster min_cell_hits (default 2). sparse cloud는 1.",
    )
    parser.add_argument(
        "--floor-pc-orientation",
        type=str,
        default="auto",
        choices=["sensor", "rotate_cw_90", "rotate_ccw_90", "rotate_180", "auto"],
        help="floor_pointcloud RtabmapImageEvidenceStep orientation mode (default auto).",
    )
    parser.add_argument(
        "--floor-pc-evidence-dir",
        type=str,
        default=None,
        help=(
            "Sprint 46 evidence dump directory. 지정 시 floor_pointcloud_report.json 등 저장."
        ),
    )
    # Sprint 47 / Sprint 48 — CAD cleanup CLI.
    # Codex F-3: default OFF (>=2 real scan PASS 시까지). opt-in 으로만 evidence 생성.
    parser.add_argument(
        "--floor-pc-cad-cleanup",
        dest="floor_pc_cad_cleanup",
        action="store_true",
        default=False,
        help=(
            "Sprint 48: floor_pointcloud post-rectification CAD cleanup ON. "
            "default OFF (Codex F-3: 2 real scan PASS 후 회귀 가능)."
        ),
    )
    parser.add_argument(
        "--no-floor-pc-cad-cleanup",
        dest="floor_pc_cad_cleanup",
        action="store_false",
        help="Sprint 47/48: CAD cleanup OFF (default). cad_cleanup_meta = None.",
    )
    # Sprint 48: 2-axis sweep evidence (Codex W-6)
    parser.add_argument(
        "--cad-sweep-area",
        type=float,
        nargs="+",
        default=None,
        metavar="FLOAT",
        help=(
            "Sprint 48: area_change_limit sweep 값 list (예: 0.45 0.55 0.65). "
            "지정 시 --cad-sweep-short-edge 와 곱집합으로 2축 sweep evidence."
        ),
    )
    parser.add_argument(
        "--cad-sweep-short-edge",
        type=float,
        nargs="+",
        default=None,
        metavar="FLOAT",
        help=(
            "Sprint 48: short_edge_min_length 2축 sweep (예: 0.10 0.20). "
            "--cad-sweep-area 와 곱집합. 둘 다 미지정 시 sweep 비활성."
        ),
    )
    parser.add_argument(
        "--cad-morph-close",
        type=int,
        default=3,
        help="Sprint 47: FloorRaster morph_close_radius_cells for floor_pc 모드 (default 3).",
    )
    parser.add_argument(
        "--cad-rectification-area-limit",
        type=float,
        default=0.55,
        help=(
            "Sprint 47: floor_pointcloud 모드 manhattan_max_area_change (default 0.55). "
            "trajectory 등 5경로는 default 0.20 그대로."
        ),
    )
    parser.add_argument(
        "--adaptive-buffer-max",
        type=float,
        default=5.0,
        help="adaptive buffer 최대 반지름 m (default 5.0, horizon 발산 방어)",
    )
    parser.add_argument(
        "--adaptive-buffer-min",
        type=float,
        default=0.3,
        help="adaptive buffer 최소 반지름 m (default 0.3, false-zero 방어)",
    )
    parser.add_argument(
        "--adaptive-buffer-strip",
        type=float,
        default=0.5,
        help="발밑 strip 비율 (default 0.5, ray 각도 안전 영역)",
    )
    parser.add_argument(
        "--multiview",
        action="store_true",
        help="SuperPoint+LightGlue 기반 multi-view scale 전역 최적화 활성 (--depth-nn 필수)",
    )
    parser.add_argument(
        "--mv-window",
        type=int,
        default=5,
        help="multi-view 매칭 인접 window (default 5)",
    )
    parser.add_argument(
        "--triang-window",
        type=int,
        default=5,
        help="triangulation 매칭 인접 window (default 5)",
    )
    parser.add_argument(
        "--triang-no-floor-gate",
        action="store_true",
        help="triangulation 시 floor mask gate 해제 — 모든 kp 매치 사용 후 z-tolerance만 적용",
    )
    parser.add_argument(
        "--triang-min-score",
        type=float,
        default=0.8,
        help="LightGlue 매치 score 최소값 (default 0.8)",
    )
    parser.add_argument(
        "--triang-max-matches",
        type=int,
        default=64,
        help="pair당 사용 매치 수 상한 (default 64)",
    )
    parser.add_argument(
        "--trajectory-buffer",
        type=float,
        default=None,
        metavar="BUFFER_M",
        help=(
            "trajectory buffer fusion 활성 (default off). 값은 buffer 반경 m (예: 0.8). "
            "--triangulate와 함께 쓸 때만 의미 있음."
        ),
    )
    # ── Sprint 50: rectangle dictionary cover (Codex BLOCKER 4) ────────────
    # CLI default ON (evaluation 산출물용). production config (config.py) 는
    # 별도로 default OFF — Settings.rectangle_dictionary_cover_enabled.
    parser.add_argument(
        "--rectangle-cover",
        dest="rectangle_cover",
        action="store_true",
        default=True,
        help=(
            "Sprint 50: rectangle dictionary cover 활성 (eval default ON). "
            "obs heatmap 직사각형 union 으로 footprint 직각 보장."
        ),
    )
    parser.add_argument(
        "--no-rectangle-cover",
        dest="rectangle_cover",
        action="store_false",
        help="Sprint 50: rectangle cover OFF (Sprint 49 hint chain path 만 사용).",
    )
    parser.add_argument(
        "--rectangle-cover-tau-p",
        type=float,
        default=0.85,
        help="Sprint 50: precision threshold τ_p (default 0.85).",
    )
    parser.add_argument(
        "--rectangle-cover-recall-min",
        type=float,
        default=0.65,
        help="Sprint 50: recall_min gate (default 0.65, v2 lowered from 0.70).",
    )
    parser.add_argument(
        "--rectangle-cover-over-cover-max",
        type=float,
        default=0.20,
        help="Sprint 50: over_cover_max gate (default 0.20).",
    )
    parser.add_argument(
        "--rectangle-cover-time-budget-sec",
        type=float,
        default=30.0,
        help="Sprint 50: time budget per cover step (default 30.0s).",
    )
    parser.add_argument(
        "--rectangle-cover-axes-mode",
        choices=("pair", "full18"),
        default="pair",
        help=(
            "Sprint 50 v2: axis selection mode. "
            "'pair' — RTABMap link / OBB dominant axis 추출 후 (primary, +90°) "
            "pair 로만 sweep, multi-component 분리 (default). "
            "'full18' — v1 18 angle full sweep."
        ),
    )
    parser.add_argument(
        "--rectangle-cover-precision-dynamic",
        dest="rectangle_cover_precision_dynamic",
        action="store_true",
        default=True,
        help=(
            "Sprint 50 v2: thickness 비례 dynamic precision threshold (default ON). "
            "thickness 큰 corridor 환경에서 임계 자동 완화."
        ),
    )
    parser.add_argument(
        "--no-rectangle-cover-precision-dynamic",
        dest="rectangle_cover_precision_dynamic",
        action="store_false",
        help="Sprint 50 v2: dynamic precision threshold OFF (fixed tau_p).",
    )
    parser.add_argument(
        "--rectangle-cover-precision-min",
        type=float,
        default=0.65,
        help=(
            "Sprint 50 v2: dynamic precision threshold floor "
            "(default 0.65, base 0.85 까지 thickness 비례 완화)."
        ),
    )
    args = parser.parse_args()

    if args.trajectory_buffer is not None and not args.triangulate:
        parser.error("--trajectory-buffer은 --triangulate와 함께 사용해야 합니다.")
    if args.adaptive_buffer and args.trajectory_buffer is not None:
        parser.error("--adaptive-buffer and --trajectory-buffer are mutually exclusive")

    scan_dir = Path(args.scan_dir).resolve()
    sidecar = scan_dir / "scan_metadata.db"
    keyframes_dir = scan_dir / "keyframes"
    if not sidecar.exists() or not keyframes_dir.exists():
        print(f"missing scan_metadata.db or keyframes/ in {scan_dir}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(sidecar))
    try:
        scan_id_row = conn.execute("SELECT id FROM scan_session LIMIT 1").fetchone()
        # Sprint 49: v3+ schema (poi_mark.label/source) 일 때 POI 로드.
        # v1 schema 면 컬럼 missing 으로 OperationalError → 빈 리스트 fallback.
        try:
            from indoor_server.domain.poi.enums import POISource
            from indoor_server.domain.scan.models import POIMarkRow
            poi_rows_raw = conn.execute(
                "SELECT id, scan_id, keyframe_seq, created_at, pose_matrix, "
                "tx, ty, tz, track_id, label, source FROM poi_mark"
            ).fetchall()
            poi_rows: list[POIMarkRow] = [
                POIMarkRow(
                    id=int(r[0]),
                    scan_id=str(r[1]),
                    keyframe_seq=int(r[2]),
                    created_at=int(r[3]),
                    pose_matrix=bytes(r[4]),
                    tx=float(r[5]),
                    ty=float(r[6]),
                    tz=float(r[7]),
                    track_id=int(r[8]) if r[8] is not None else None,
                    label=r[9] if r[9] is not None else None,
                    source=POISource(r[10]),
                )
                for r in poi_rows_raw
            ]
        except sqlite3.OperationalError as exc:
            print(f"poi_mark load skipped (v1 schema): {exc}", file=sys.stderr)
            poi_rows = []
    finally:
        conn.close()
    if scan_id_row is None:
        print("scan_session row missing", file=sys.stderr)
        return 2
    print(f"loaded POIs: {len(poi_rows)}")
    scan_uuid = UUID(scan_id_row[0])
    job_uuid = uuid4()

    storage_root = _REPO_ROOT / "var" / "storage"
    debug_root = _REPO_ROOT / "var" / "debug"

    _stage_keyframes(scan_dir, scan_uuid, storage_root / "scans")
    _patch_estimate_z0_with_offset()

    use_depth_nn = args.depth_nn
    use_triangulation = args.triangulate
    use_adaptive_buffer = args.adaptive_buffer
    use_floor_pointcloud = args.floor_pointcloud
    if use_floor_pointcloud:
        print(
            f"--floor-pointcloud 활성 (Sprint 46): "
            f"stride={args.floor_pc_pixel_stride} "
            f"height_tol={args.floor_pc_height_tolerance:.2f}m "
            f"min_cell_hits={args.floor_pc_min_cell_hits}"
        )
    if use_adaptive_buffer:
        print(
            f"--adaptive-buffer 활성 (Sprint 22): "
            f"min={args.adaptive_buffer_min:.2f}m max={args.adaptive_buffer_max:.2f}m "
            f"strip={args.adaptive_buffer_strip:.0%}"
        )
        # adaptive buffer는 자체 disk union이 walkable 결정 — obs_threshold/morph 패치 불필요
    elif use_depth_nn:
        print("--depth-nn 활성: z-tolerance 50cm + opening kernel 7 (수학적 spike 제거)")
        _patch_z_tolerance(0.50)         # 30cm → 50cm
        _patch_morph_opening(7)          # opening kernel 7 (70cm 미만 spike 제거)
    elif use_triangulation:
        print("--triangulate 활성: SuperPoint+LightGlue 2-view triangulation (Sprint 20)")
        # triangulation 포인트는 depth 경로 대비 1~2 order sparse → obs_threshold 최소
        _patch_obs_threshold(1)
        # Sprint 21: trajectory_buffer가 복도 폭 담당 시 closing은 작은 gap만 채우면 됨.
        # 큰 closing은 triangulation outlier를 불필요하게 뻥튀기해 삐죽 발생.
        if args.trajectory_buffer:
            _patch_morph_closing_only(5)  # buffer 병행 시 작은 gap만
        else:
            _patch_morph_closing_only(15)  # buffer 없으면 조각 연결용
    else:
        _patch_back_projection_with_distance_cap()
        _patch_obs_threshold(5)       # 3 → 5
        _patch_morph_opening(5)       # B-2: opening kernel 5

    fx_scale = float(os.environ.get("FX_SCALE", "1.0"))
    if fx_scale != 1.0:
        print(f"intrinsics fx scale: {fx_scale}")
        _patch_intrinsics_scale(fx_scale)

    keyframes = _load_keyframes(sidecar, scan_uuid, storage_root)
    print(f"loaded {len(keyframes)} keyframes (scan_id={scan_uuid})")
    if not keyframes:
        print("no usable keyframes", file=sys.stderr)
        return 1

    mode = args.mode
    if mode in ("real", "real_upright"):
        from indoor_server.infrastructure.ml.segformer_onnx import SegformerOnnxSegmenter
        cache = ModelCache(
            cache_dir=settings.model_cache_dir,
            repo_id=settings.segformer_model_repo_id,
            filename=settings.segformer_model_filename,
        )
        model_path = cache.ensure()
        base = SegformerOnnxSegmenter(model_path=model_path)
        if mode == "real_upright":
            if use_depth_nn:
                keep_frac = 1.0
            elif use_triangulation:
                # 발밑부터 50% strip만 floor mask로 인정.
                # 멀리 떨어진 horizon 픽셀은 ray 각도 작아 triangulation 노이즈 큼 → 배제.
                keep_frac = 0.50
            elif args.adaptive_buffer:
                # strip은 AdaptiveBufferStep 내부에서 적용 → segmenter는 자르지 않음
                keep_frac = 1.0
            else:
                keep_frac = _MASK_KEEP_BOTTOM_FRACTION
            print(f"using real Segformer-B0 + 90° CW upright rotation (keep={keep_frac:.0%}): {model_path}")
            segmenter: SemanticSegmenter = UprightRotatedSegmenter(base, keep_bottom_fraction=keep_frac)
        else:
            print(f"using real Segformer-B0 ONNX: {model_path}")
            segmenter = base
    else:
        print(f"using stub: {mode} 25%")
        segmenter = BottomHalfFloorSegmenter(side=mode)

    # depth NN 로드
    depth_runner = None
    if use_depth_nn:
        from indoor_server.infrastructure.ml.depth_anything import DepthAnythingV2Runner

        depth_local = settings.model_cache_dir / "depth_anything_v2_small.onnx"
        if not depth_local.exists():
            print(
                f"Depth Anything 모델이 없습니다: {depth_local}\n"
                "먼저: uv run python scripts/fetch_depth_anything.py",
                file=sys.stderr,
            )
            return 2
        print(f"loading Depth Anything v2-Small from {depth_local}")
        depth_runner = DepthAnythingV2Runner(model_path=depth_local)

    # SuperPoint + LightGlue 로드 (Sprint 20)
    sp_lg_runner = None
    use_multiview = bool(args.multiview)
    if use_triangulation:
        from indoor_server.infrastructure.ml.superpoint_lightglue import SuperPointLightGlueRunner

        sp_lg_local = settings.model_cache_dir / "superpoint_lightglue.onnx"
        if not sp_lg_local.exists():
            print(
                f"SuperPoint+LightGlue 모델이 없습니다: {sp_lg_local}\n"
                "먼저: uv run python scripts/fetch_superpoint_lightglue.py",
                file=sys.stderr,
            )
            return 2
        print(f"loading SuperPoint+LightGlue from {sp_lg_local}")
        sp_lg_runner = SuperPointLightGlueRunner(
            model_path=sp_lg_local,
            input_size=settings.superpoint_input_size,
        )
    elif use_multiview:
        if not use_depth_nn:
            print("--multiview requires --depth-nn", file=sys.stderr)
            return 2
        from indoor_server.infrastructure.ml.superpoint_lightglue import SuperPointLightGlueRunner

        sp_lg_local = settings.model_cache_dir / "superpoint_lightglue.onnx"
        if not sp_lg_local.exists():
            print(
                f"SuperPoint+LightGlue 모델이 없습니다: {sp_lg_local}\n"
                "먼저: uv run python scripts/fetch_superpoint_lightglue.py",
                file=sys.stderr,
            )
            return 2
        print(f"loading SuperPoint+LightGlue from {sp_lg_local}")
        sp_lg_runner = SuperPointLightGlueRunner(
            model_path=sp_lg_local,
            input_size=settings.superpoint_input_size,
        )

    use_trajectory_buffer = args.trajectory_buffer is not None
    trajectory_buffer_m = args.trajectory_buffer if args.trajectory_buffer is not None else 0.8

    # Sprint 46: floor_pointcloud 모드는 rtabmap.db에서 Node + Data를 로딩한다.
    # Sprint 48: hint chain 1순위 (RTABMap link) 산출용으로 links 도 로딩.
    rtabmap_nodes_for_pipeline = None
    rtabmap_links_for_pipeline = None
    rtabmap_frames_for_pipeline = None
    if use_floor_pointcloud:
        rtabmap_db = scan_dir / "rtabmap.db"
        if not rtabmap_db.exists():
            print(
                f"--floor-pointcloud requires rtabmap.db at {rtabmap_db}",
                file=sys.stderr,
            )
            return 2
        from indoor_server.application.rtabmap.reader import RtabmapReader

        reader = RtabmapReader()
        rtabmap_nodes_for_pipeline = reader.load_nodes(rtabmap_db)
        rtabmap_frames_for_pipeline = reader.load_data_frames(rtabmap_db)
        try:
            rtabmap_links_for_pipeline = reader.load_links(rtabmap_db)
        except Exception as e:
            print(f"failed to load RTAB-Map links (hint chain disabled): {e}")
            rtabmap_links_for_pipeline = []
        print(
            f"loaded RTAB-Map: nodes={len(rtabmap_nodes_for_pipeline)} "
            f"frames={len(rtabmap_frames_for_pipeline)} "
            f"links={len(rtabmap_links_for_pipeline) if rtabmap_links_for_pipeline else 0}"
        )

    pipeline = BuildPipeline(
        segmenter=segmenter,
        storage_root=storage_root,
        depth_runner=depth_runner,
        use_depth_nn=use_depth_nn,
        sp_lg_runner=sp_lg_runner,
        use_multiview_scale=use_multiview,
        multiview_window=args.mv_window,
        use_triangulation=use_triangulation,
        triangulation_window=args.triang_window,
        triangulation_floor_gate=not args.triang_no_floor_gate,
        triangulation_min_score=args.triang_min_score,
        triangulation_max_matches=args.triang_max_matches,
        use_trajectory_buffer=use_trajectory_buffer,
        trajectory_buffer_m=trajectory_buffer_m,
        use_adaptive_buffer=use_adaptive_buffer,
        adaptive_buffer_max_m=args.adaptive_buffer_max,
        adaptive_buffer_min_m=args.adaptive_buffer_min,
        adaptive_buffer_strip_fraction=args.adaptive_buffer_strip,
        use_floor_pointcloud=use_floor_pointcloud,
        floor_pointcloud_pixel_stride=args.floor_pc_pixel_stride,
        floor_pointcloud_height_tolerance_m=args.floor_pc_height_tolerance,
        floor_pointcloud_min_cell_hits=args.floor_pc_min_cell_hits,
        rtabmap_image_orientation_mode=args.floor_pc_orientation,
        # Sprint 47 CAD cleanup
        polygon_cad_cleanup_enabled=args.floor_pc_cad_cleanup,
        floor_raster_cad_morph_close_cells=args.cad_morph_close,
        manhattan_floor_pointcloud_max_area_change=args.cad_rectification_area_limit,
        # Sprint 50 — rectangle cover. CLI default ON (eval), production OFF.
        use_rectangle_dictionary_cover=(
            use_floor_pointcloud and args.rectangle_cover
        ),
        rectangle_cover_precision_threshold=args.rectangle_cover_tau_p,
        rectangle_cover_recall_min=args.rectangle_cover_recall_min,
        rectangle_cover_over_cover_max=args.rectangle_cover_over_cover_max,
        rectangle_cover_time_budget_sec=args.rectangle_cover_time_budget_sec,
        # Sprint 50 v2: axis pair + dynamic precision threshold
        rectangle_cover_axes_mode=args.rectangle_cover_axes_mode,
        rectangle_cover_precision_threshold_dynamic=(
            args.rectangle_cover_precision_dynamic
        ),
        rectangle_cover_precision_threshold_min=(
            args.rectangle_cover_precision_min
        ),
    )
    sink = FilesystemDebugSink(
        out_dir=debug_root / str(scan_uuid) / str(job_uuid),
        storage_root=storage_root,
        sample_count=5,  # 빠른 검증: 5장만
    )

    async def progress(step: BuildStep, p: float) -> None:
        print(f"  step={step.value:<18} progress={p:.2f}")

    async def cancel() -> bool:
        return False

    try:
        outcome = await pipeline.execute(
            scan_id=scan_uuid,
            build_job_id=job_uuid,
            keyframes=keyframes,
            pois=poi_rows,
            rtabmap_nodes=rtabmap_nodes_for_pipeline,
            rtabmap_links=rtabmap_links_for_pipeline,
            rtabmap_frames=rtabmap_frames_for_pipeline,
            progress_sink=progress,
            cancel_check=cancel,
            debug_sink=sink,
        )
    finally:
        sink.finalize()

    if use_floor_pointcloud and args.floor_pc_evidence_dir:
        _dump_floor_pointcloud_evidence(
            evidence_dir=Path(args.floor_pc_evidence_dir).resolve(),
            outcome=outcome,
        )

        # Sprint 48 (Codex W-6) — 2축 sweep evidence
        sweep_areas = args.cad_sweep_area
        sweep_short_edges = args.cad_sweep_short_edge
        if sweep_areas or sweep_short_edges:
            sweep_areas = sweep_areas or [args.cad_rectification_area_limit]
            sweep_short_edges = sweep_short_edges or [
                settings.polygon_cad_short_edge_min_length_m
            ]
            await _run_cad_parameter_sweep(
                evidence_dir=Path(args.floor_pc_evidence_dir).resolve(),
                scan_uuid=scan_uuid,
                segmenter=segmenter,
                storage_root=storage_root,
                rtabmap_nodes=rtabmap_nodes_for_pipeline,
                rtabmap_frames=rtabmap_frames_for_pipeline,
                scan_dir=scan_dir,
                args=args,
                area_values=sweep_areas,
                short_edge_values=sweep_short_edges,
            )

    print()
    print("=== build outcome ===")
    print(f"  passed_quality_gate: {outcome.passed_quality_gate}")
    print(f"  failure_reason:      {outcome.failure_reason}")
    c = outcome.counts
    print(f"  keyframes_processed: {c.keyframes_processed}")
    print(f"  walkable_cells:      {c.walkable_cells}")
    print(f"  skeleton_pixels:     {c.skeleton_pixels}")
    print(f"  map_nodes:           {c.map_nodes}")
    print(f"  map_edges:           {c.map_edges}")
    print(f"  walkable_coverage:   {c.walkable_coverage:.3f}")
    print(f"  connected_components: {c.connected_components}")
    print(f"  floor_z0:            {c.floor_z0}")
    print()
    print(f"debug dump: {debug_root / str(scan_uuid) / str(job_uuid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
