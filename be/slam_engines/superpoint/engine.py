"""SuperPoint + LightGlue localization engine.

Map building (process / save_map / load_map) is delegated to RTABMapEngine
unchanged. Only localize() is re-implemented here.
"""
import asyncio
import functools
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from slam_interface.base import SLAMEngineBase
from utils import logger


def _to_gray_float(img_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes → grayscale float [0,1], applying EXIF orientation."""
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)))
        if pil.mode != 'L':
            pil = pil.convert('L')
        return np.array(pil, dtype=np.float32) / 255.0
    except Exception:
        return None


def _normalize_query(
    img_bytes: bytes, K_in: np.ndarray
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Decode + normalize query image to landscape, return (gray, K_for_image).

    DB keyframe 들은 RTABMap 이 raw landscape 로 저장 (EXIF 없음).
    query 가 portrait 으로 들어오면 90° CCW 회전 → landscape 정합.
    K 는 image dim 과 회전에 맞춰 변환:
      1) portrait→landscape 회전 시 fx/fy swap, cx/cy 도 회전 변환
      2) K 의 cx/cy 가 image 중심에서 멀면 native sensor 기준 → image dim 비율로 scale
    """
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)))
    except Exception:
        return None

    K = K_in.astype(np.float64).copy()

    # K 가 어떤 image dim 기준인지 추정 — cx*2, cy*2 로 sensor W, H.
    # iOS RTABMap 의 intrinsics 는 보통 native landscape sensor (예 1920×1440) 기준.
    K_sensor_w = K[0, 2] * 2.0
    K_sensor_h = K[1, 2] * 2.0

    # 1) portrait image → landscape 로 회전 (DB keyframe 도 landscape).
    if pil.height > pil.width:
        pil = pil.transpose(Image.Transpose.ROTATE_90)  # PIL ROTATE_90 = 90° CCW

    # 2) image 와 K sensor frame scale 정합 (image 가 다운스케일됐을 때).
    img_w, img_h = pil.size
    if K_sensor_w > 0 and K_sensor_h > 0:
        sx = img_w / K_sensor_w
        sy = img_h / K_sensor_h
        # K_sensor 가 portrait (W < H) 인데 image 가 landscape 인 경우 — sensor 도 회전 가정.
        # 그 때 sx, sy 가 swap 되어 1.0 근처 안 나옴. fallback 으로 평균 ratio 적용.
        if abs(sx - sy) > 0.5 * max(sx, sy):
            sx_swap = img_w / K_sensor_h
            sy_swap = img_h / K_sensor_w
            if abs(sx_swap - sy_swap) < abs(sx - sy):
                # K_sensor 가 portrait 이고 image 는 landscape — sensor frame 회전 적용.
                K[0, 0], K[1, 1] = K[1, 1], K[0, 0]
                K[0, 2], K[1, 2] = K[1, 2], K[0, 2]
                sx, sy = sx_swap, sy_swap
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    if pil.mode != 'L':
        pil = pil.convert('L')
    gray = np.array(pil, dtype=np.float32) / 255.0
    return gray, K


def _depth_dims(meta: dict[str, Any] | None, value_count: int) -> Optional[tuple[int, int]]:
    if meta:
        width = meta.get("width") or meta.get("w")
        height = meta.get("height") or meta.get("h")
        try:
            width_i = int(width)
            height_i = int(height)
            if width_i > 0 and height_i > 0 and width_i * height_i == value_count:
                return width_i, height_i
        except Exception:
            pass

    common = ((256, 192), (192, 256), (320, 240), (240, 320), (640, 480), (480, 640))
    for width_i, height_i in common:
        if width_i * height_i == value_count:
            return width_i, height_i
    return None


def _decode_query_depth(item: dict[str, Any] | None) -> Optional[np.ndarray]:
    if not item:
        return None
    data = item.get("bytes") or item.get("data")
    if not data:
        return None
    meta = item.get("depth_intrinsics") or {}
    filename = str(item.get("filename") or "").lower()
    content_type = str(item.get("content_type") or "").lower()

    if "png" in filename or "png" in content_type:
        arr = np.frombuffer(data, dtype=np.uint8)
        depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        original_dtype = depth.dtype
        depth = depth.astype(np.float32)
        unit = str(meta.get("unit") or "").lower()
        if original_dtype == np.uint16 or unit in {"mm", "millimeter", "millimeters"}:
            depth = depth / 1000.0
        return depth

    if len(data) % 4 == 0:
        values = np.frombuffer(data, dtype="<f4").astype(np.float32)
        dims = _depth_dims(meta, int(values.size))
        if dims is not None:
            width, height = dims
            return values.reshape(height, width)
    if len(data) % 2 == 0:
        values = np.frombuffer(data, dtype="<u2").astype(np.float32)
        dims = _depth_dims(meta, int(values.size))
        if dims is not None:
            width, height = dims
            return values.reshape(height, width) / 1000.0
    return None


def _sample_depths(depth: np.ndarray, pts_2d: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    img_h, img_w = image_shape
    depth_h, depth_w = depth.shape[:2]
    if img_w <= 0 or img_h <= 0 or depth_w <= 0 or depth_h <= 0:
        return np.full((len(pts_2d),), np.nan, dtype=np.float64)

    xs = np.round(pts_2d[:, 0] * (depth_w / img_w)).astype(np.int32)
    ys = np.round(pts_2d[:, 1] * (depth_h / img_h)).astype(np.int32)
    sampled = np.full((len(pts_2d),), np.nan, dtype=np.float64)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x < 0 or y < 0 or x >= depth_w or y >= depth_h:
            continue
        x0, x1 = max(0, x - 1), min(depth_w, x + 2)
        y0, y1 = max(0, y - 1), min(depth_h, y + 2)
        patch = depth[y0:y1, x0:x1].astype(np.float64).reshape(-1)
        patch = patch[np.isfinite(patch) & (patch > 0.15) & (patch < 10.0)]
        if patch.size:
            sampled[i] = float(np.median(patch))
    return sampled


def _backproject_query_points(pts_2d: np.ndarray, depths: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    z = depths.astype(np.float64)
    x = (pts_2d[:, 0].astype(np.float64) - cx) * z / fx
    y = (pts_2d[:, 1].astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def _rigid_transform_3d(src: np.ndarray, dst: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if len(src) < 3 or len(dst) < 3:
        return None
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    H = src_centered.T @ dst_centered
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = dst_centroid - R @ src_centroid
    return R.astype(np.float64), t.astype(np.float64)


def _depth_3d_pose_candidate(
    *,
    depth: np.ndarray,
    pts_2d: np.ndarray,
    pts_3d_world: np.ndarray,
    image_shape: tuple[int, int],
    K: np.ndarray,
    pnp_R_oc: np.ndarray,
    pnp_t_cw: np.ndarray,
) -> dict[str, Any]:
    obs_depth = _sample_depths(depth, pts_2d, image_shape)
    valid = np.isfinite(obs_depth) & (obs_depth > 0.15) & (obs_depth < 10.0)
    valid_count = int(np.count_nonzero(valid))
    result: dict[str, Any] = {
        "validMatches": valid_count,
        "totalInliers": int(len(pts_2d)),
        "applied": False,
    }
    if valid_count < 8:
        result["rejectReason"] = "insufficient_valid_depth_matches"
        return result

    q3d = _backproject_query_points(pts_2d[valid], obs_depth[valid], K)
    w3d = pts_3d_world[valid].astype(np.float64)

    # Depth-aware translation solve with PnP rotation fixed. This uses query
    # depth in the pose solve itself, including lateral translation, while
    # keeping orientation stable under noisy mobile depth.
    offsets = w3d - (pnp_R_oc @ q3d.T).T
    fixed_t = np.median(offsets, axis=0)
    fixed_errors = np.linalg.norm((pnp_R_oc @ q3d.T).T + fixed_t - w3d, axis=1)
    fixed_median = float(np.median(fixed_errors))
    fixed_mad = float(np.median(np.abs(fixed_errors - fixed_median)))
    fixed_threshold = max(0.35, fixed_median + 3.0 * fixed_mad)
    fixed_inliers = fixed_errors <= fixed_threshold
    if np.count_nonzero(fixed_inliers) >= 8:
        fixed_t = np.median(offsets[fixed_inliers], axis=0)
        fixed_errors = np.linalg.norm((pnp_R_oc @ q3d.T).T + fixed_t - w3d, axis=1)
        fixed_median = float(np.median(fixed_errors[fixed_inliers]))
        fixed_mad = float(np.median(np.abs(fixed_errors[fixed_inliers] - fixed_median)))

    result.update(
        {
            "fixedRotationInliers": int(np.count_nonzero(fixed_inliers)),
            "fixedRotationInlierRatio": float(np.count_nonzero(fixed_inliers) / max(valid_count, 1)),
            "fixedRotationMedianErrorM": fixed_median,
            "fixedRotationMadErrorM": fixed_mad,
            "fixedRotationTranslation": [float(v) for v in fixed_t],
            "fixedRotationDeltaM": float(np.linalg.norm(fixed_t - pnp_t_cw)),
        }
    )

    # Full 3D-3D rigid candidate for diagnostics and future promotion. It can
    # be noisier in corridors, so we only publish the fixed-rotation translation
    # unless the full transform is very consistent.
    rng = np.random.default_rng(42)
    best_inliers: np.ndarray | None = None
    best_R: np.ndarray | None = None
    best_t: np.ndarray | None = None
    best_median = float("inf")
    iterations = min(192, max(48, valid_count * 4))
    for _ in range(iterations):
        sample_idx = rng.choice(valid_count, size=3, replace=False)
        transform = _rigid_transform_3d(q3d[sample_idx], w3d[sample_idx])
        if transform is None:
            continue
        R, t = transform
        errors = np.linalg.norm((R @ q3d.T).T + t - w3d, axis=1)
        inliers = errors <= 0.35
        n_in = int(np.count_nonzero(inliers))
        if n_in < 8:
            continue
        med = float(np.median(errors[inliers]))
        if best_inliers is None or n_in > int(np.count_nonzero(best_inliers)) or (
            n_in == int(np.count_nonzero(best_inliers)) and med < best_median
        ):
            best_inliers = inliers
            best_R = R
            best_t = t
            best_median = med

    if best_inliers is not None and best_R is not None and best_t is not None:
        refined = _rigid_transform_3d(q3d[best_inliers], w3d[best_inliers])
        if refined is not None:
            best_R, best_t = refined
            best_errors = np.linalg.norm((best_R @ q3d.T).T + best_t - w3d, axis=1)
            best_median = float(np.median(best_errors[best_inliers]))
            result.update(
                {
                    "rigid3dInliers": int(np.count_nonzero(best_inliers)),
                    "rigid3dInlierRatio": float(np.count_nonzero(best_inliers) / max(valid_count, 1)),
                    "rigid3dMedianErrorM": best_median,
                    "rigid3dTranslation": [float(v) for v in best_t],
                    "rigid3dDeltaM": float(np.linalg.norm(best_t - pnp_t_cw)),
                }
            )

    fixed_ratio = float(result["fixedRotationInlierRatio"])
    fixed_delta = float(result["fixedRotationDeltaM"])
    if fixed_ratio >= 0.55 and fixed_median <= 0.45 and fixed_delta <= 1.25:
        result["applied"] = True
        result["mode"] = "depth_3d_fixed_rotation_translation"
        result["translation"] = [float(v) for v in fixed_t]
    else:
        result["rejectReason"] = "depth_3d_unstable"
    return result


def _rotation_to_quat(R: np.ndarray):
    """Convert 3x3 rotation matrix to (qx, qy, qz, qw), qw >= 0."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / (trace + 1.0) ** 0.5
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * (1 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * (1 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return float(qx), float(qy), float(qz), float(qw)


def _quat_to_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))


def _angle_diff_deg(a: float, b: float) -> float:
    return float((a - b + 180.0) % 360.0 - 180.0)


def _pose_yaw_deg(pose: dict) -> float:
    return _quat_to_yaw_deg(
        float(pose["qx"]), float(pose["qy"]), float(pose["qz"]), float(pose["qw"])
    )


def _rtab_relative_pose_candidate(
    *,
    q_kps: np.ndarray,
    db_kps: np.ndarray,
    matches: np.ndarray,
    K: np.ndarray,
    node_pose: np.ndarray,
    optical_to_base: np.ndarray,
    pnp_pose: dict,
) -> Optional[dict]:
    """RTAB-Map-style comparison pose.

    Estimate query-vs-keyframe relative rotation from 2D matches, compose it with
    the matched RTAB-Map Node.pose, and keep the keyframe translation. The
    recovered relative translation has unknown scale, so it is logged only as a
    direction cue and not used as metric output.
    """
    if len(matches) < 8:
        return None

    pts_query = q_kps[matches[:, 0]].astype(np.float64)
    pts_db = db_kps[matches[:, 1]].astype(np.float64)

    try:
        E, mask = cv2.findEssentialMat(
            pts_query,
            pts_db,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=2.0,
        )
    except Exception:
        return None
    if E is None:
        return None

    try:
        n_in, R_q_to_db, t_q_to_db, pose_mask = cv2.recoverPose(
            E, pts_query, pts_db, K, mask=mask
        )
    except Exception:
        return None
    if int(n_in) < 8:
        return None

    R_base_to_world_db = node_pose[:3, :3]
    t_base_to_world_db = node_pose[:3, 3]
    R_opt_to_world_db = R_base_to_world_db @ optical_to_base
    R_opt_to_world_query = R_opt_to_world_db @ R_q_to_db
    R_base_to_world_query = R_opt_to_world_query @ optical_to_base.T
    qx, qy, qz, qw = _rotation_to_quat(R_base_to_world_query)

    pose = {
        "x": float(t_base_to_world_db[0]),
        "y": float(t_base_to_world_db[1]),
        "z": float(t_base_to_world_db[2]),
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
    }
    pnp_t = np.array([pnp_pose["x"], pnp_pose["y"], pnp_pose["z"]], dtype=np.float64)
    keyframe_t = t_base_to_world_db.astype(np.float64)
    yaw_delta = _angle_diff_deg(_pose_yaw_deg(pose), _pose_yaw_deg(pnp_pose))
    return {
        "pose": pose,
        "relative_inliers": int(n_in),
        "relative_inlier_ratio": float(n_in / max(len(matches), 1)),
        "position_delta_m": float(np.linalg.norm(pnp_t - keyframe_t)),
        "yaw_delta_deg": yaw_delta,
        "relative_translation_unit": [
            float(t_q_to_db[0, 0]),
            float(t_q_to_db[1, 0]),
            float(t_q_to_db[2, 0]),
        ],
    }


class SuperPointEngine(SLAMEngineBase):
    """Localization engine using SuperPoint + LightGlue."""

    def __init__(self):
        from lightglue import LightGlue, SuperPoint

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._extractor = SuperPoint(max_num_keypoints=1024).eval().to(self._device)
        # batch matching 시 adaptive pruning (depth/width_confidence) 가 per-sample 분기로
        # batch 내부에서 IndexError 일으킴 → -1 로 비활성. throughput 우선.
        self._matcher = LightGlue(
            features='superpoint',
            depth_confidence=-1,
            width_confidence=-1,
            mp=True,  # FP16 mixed precision (LightGlue 공식 지원)
        ).eval().to(self._device)
        logger.info(f"[SuperPoint] Engine ready on {self._device}")

    # --- map building: delegate unchanged to RTABMapEngine ---

    async def process(self, *args, **kwargs):
        from slam_engines.rtabmap.engine import RTABMapEngine
        return await RTABMapEngine().process(*args, **kwargs)

    def save_map(self, *args, **kwargs):
        from slam_engines.rtabmap.engine import RTABMapEngine
        return RTABMapEngine().save_map(*args, **kwargs)

    def load_map(self, *args, **kwargs):
        from slam_engines.rtabmap.engine import RTABMapEngine
        return RTABMapEngine().load_map(*args, **kwargs)

    def scale_intrinsics(self, original: dict, new_width: int, new_height: int) -> dict:
        from slam_engines.rtabmap.engine import RTABMapEngine
        return RTABMapEngine().scale_intrinsics(original, new_width, new_height)

    def extract_intrinsics_from_db(self, db_path: str) -> dict:
        from slam_engines.rtabmap.engine import RTABMapEngine
        return RTABMapEngine().extract_intrinsics_from_db(db_path)

    # --- feature extraction & matching ---

    def _extract(self, gray: np.ndarray) -> dict:
        tensor = torch.from_numpy(gray)[None, None].to(self._device)
        # FP16 autocast — SuperPoint 도 mixed precision 으로 가속.
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=self._device.type == 'cuda'):
            out = self._extractor.extract(tensor)
        # descriptors 등 일부는 LightGlue 가 다시 .half() 하므로 그대로 둠.
        return out

    def _extract_batch(self, grays: List[np.ndarray]) -> List[dict]:
        """5 query image 의 SuperPoint feature 를 한 번의 forward 로 추출.

        SuperPoint.extract() 가 하는 것 (resize + forward + scale 보정) 을 batch 화.
        모든 image 의 H, W 동일해야 함 (burst 는 같은 카메라이므로 보장).
        """
        if not grays:
            return []
        # SuperPoint can return fewer than max_num_keypoints for low-texture frames.
        # Its batched forward path stacks keypoints internally, so mixed counts like
        # [1024, 2] and [997, 2] fail. Five query frames are small enough to extract
        # independently and keep localization reliable.
        return [self._extract(g) for g in grays]
        first_shape = grays[0].shape
        if any(g.shape != first_shape for g in grays):
            return [self._extract(g) for g in grays]

        from lightglue.utils import ImagePreprocessor
        B = len(grays)
        H, W = first_shape
        tensor = torch.from_numpy(np.stack(grays))[:, None].to(self._device)  # [B, 1, H, W]

        # SuperPoint extract() 가 하는 preprocess 동일 적용 (resize=1024 등).
        preproc = ImagePreprocessor(**self._extractor.preprocess_conf)
        img_resized, scales = preproc(tensor)  # img: [B,1,h',w'], scales: [2] (sx, sy)

        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=self._device.type == 'cuda'):
            out = self._extractor({'image': img_resized})

        result: List[dict] = []
        kps = out['keypoints']
        scs = out['keypoint_scores']
        descs = out['descriptors']
        img_size_tensor = torch.tensor([[float(W), float(H)]], device=self._device)
        for i in range(B):
            kp_i = kps[i] if isinstance(kps, list) else kps[i]
            sc_i = scs[i] if isinstance(scs, list) else scs[i]
            de_i = descs[i] if isinstance(descs, list) else descs[i]
            # extract() 의 scale 보정과 동일: keypoints 를 원본 image frame 으로.
            kp_i = (kp_i + 0.5) / scales - 0.5
            result.append({
                'keypoints': kp_i.unsqueeze(0) if kp_i.dim() == 2 else kp_i,
                'keypoint_scores': sc_i.unsqueeze(0) if sc_i.dim() == 1 else sc_i,
                'descriptors': de_i.unsqueeze(0) if de_i.dim() == 2 else de_i,
                'image_size': img_size_tensor,
            })
        return result

    def _match(self, feats0: dict, feats1: dict) -> np.ndarray:
        from lightglue.utils import rbd
        f0 = {k: v.to(self._device) for k, v in feats0.items()}
        f1 = {k: v.to(self._device) for k, v in feats1.items()}
        with torch.no_grad():
            result = self._matcher({'image0': f0, 'image1': f1})
        return rbd(result)['matches'].cpu().numpy()  # (M, 2)

    def _match_batch(
        self, q_feats: dict, db_feats_list: List[dict]
    ) -> List[np.ndarray]:
        """1 query × B db candidates 를 한 번의 forward 로 매칭.

        keypoint count 가 candidate 마다 다를 수 있어 max_N 으로 zero-padding.
        반환은 candidate 별 (P_i, 2) numpy array list. padding 위치 매치는 자동 제거.
        """
        B = len(db_feats_list)
        if B == 0:
            return []
        device = self._device

        # query 는 expand (메모리 공유)
        q_kp = q_feats['keypoints'].to(device)        # [1, Nq, 2]
        q_desc = q_feats['descriptors'].to(device)    # [1, Nq, D]
        q_size = q_feats['image_size'].to(device)     # [1, 2]
        # repeat (메모리 복사) — expand 는 LightGlue 내부 in-place 와 충돌 가능.
        q_kp_b = q_kp.repeat(B, 1, 1)
        q_desc_b = q_desc.repeat(B, 1, 1)
        q_size_b = q_size.repeat(B, 1)

        # db 는 padding 후 stack
        Ns = [d['keypoints'].shape[1] for d in db_feats_list]
        Nmax = max(Ns) if Ns else 0
        D = q_desc.shape[-1]
        db_kp = torch.zeros(B, Nmax, 2, dtype=q_kp.dtype, device=device)
        db_desc = torch.zeros(B, Nmax, D, dtype=q_desc.dtype, device=device)
        db_size = torch.zeros(B, 2, dtype=q_size.dtype, device=device)
        for i, d in enumerate(db_feats_list):
            n = Ns[i]
            db_kp[i, :n] = d['keypoints'][0].to(device)
            db_desc[i, :n] = d['descriptors'][0].to(device)
            db_size[i] = d['image_size'][0].to(device)

        with torch.no_grad():
            out = self._matcher({
                'image0': {'keypoints': q_kp_b, 'descriptors': q_desc_b, 'image_size': q_size_b},
                'image1': {'keypoints': db_kp, 'descriptors': db_desc, 'image_size': db_size},
            })

        # 'matches': List[B] of [Si, 2] tensors (이미 candidate 별 분리)
        result: List[np.ndarray] = []
        for i, m in enumerate(out['matches']):
            arr = m.detach().cpu().numpy()
            if arr.size == 0:
                result.append(arr.reshape(0, 2))
                continue
            # padded keypoint (db side index >= Ns[i]) 매칭 제거
            valid = arr[:, 1] < Ns[i]
            result.append(arr[valid])
        return result

    # --- localization core (runs in thread executor) ---

    def _localize_sync(
        self,
        map_id: str,
        images: List[bytes],
        intrinsics: Optional[Dict],
        db_path: Optional[str],
        query_depths: Optional[List[dict[str, Any]]] = None,
    ) -> dict:
        from .map_manager import SuperPointMapManager

        if intrinsics is None:
            raise ValueError("intrinsics required for SuperPoint localization")

        K_native = np.array([
            [intrinsics['fx'], 0,              intrinsics['cx']],
            [0,               intrinsics['fy'], intrinsics['cy']],
            [0,               0,               1              ],
        ], dtype=np.float64)

        mgr = SuperPointMapManager()
        loaded = mgr.get_or_load(map_id, db_path)

        if not loaded.node_ids:
            raise ValueError("No keyframes with stored images in this map")

        # 응답 R 의 axes = RTABMap camera convention (X forward, Y left, Z up) in world.
        # graph map_node / step.position / translation 와 동일 frame 라
        # 클라가 R_rtab_to_arkit 변환행렬을 일관되게 적용 가능.
        # 변환은 R_cw 계산 시점에 column 재배치로 직접 처리 (C 매트릭스 우회).
        C = None  # 사용 안 함

        best: Optional[dict] = None
        burst_best_per_image: List[dict] = []  # 각 image 의 best candidate 만 (burst average 용)

        # 5 query image 를 미리 normalize + SuperPoint batch extract (B=5).
        normalized_list = [_normalize_query(b, K_native) for b in images]
        valid_indices = [i for i, n in enumerate(normalized_list) if n is not None]
        if not valid_indices:
            raise ValueError("SuperPoint+LightGlue: no valid images")
        grays = [normalized_list[i][0] for i in valid_indices]
        Ks = [normalized_list[i][1] for i in valid_indices]
        depth_maps = [
            _decode_query_depth(query_depths[i]) if query_depths and i < len(query_depths) else None
            for i in valid_indices
        ]
        feats_list = self._extract_batch(grays)

        from .global_descriptor import GlobalDescExtractor
        gd = GlobalDescExtractor(self._device)

        for local_idx, img_idx in enumerate(valid_indices):
            image_best: Optional[dict] = None  # 이 image 안에서 candidate keyframe 들 중 best
            gray = grays[local_idx]
            K = Ks[local_idx]
            query_depth = depth_maps[local_idx]
            q_feats = feats_list[local_idx]
            q_kps = q_feats['keypoints'][0].detach().cpu().numpy()  # (N, 2)

            gray_uint8 = (gray * 255).clip(0, 255).astype(np.uint8)
            q_global = gd.extract(gray_uint8)  # (384,)

            candidates = loaded.top_k_candidates(q_global)
            if not candidates:
                continue

            db_feats_list = [loaded.keyframe_feats[nid] for nid in candidates]
            batched_matches = self._match_batch(q_feats, db_feats_list)

            for node_id, matches in zip(candidates, batched_matches):
                world3d = loaded.keyframe_world3d[node_id]          # (M, 3)

                if len(matches) < 4:
                    continue

                pts_2d, pts_3d = [], []
                for qi, di in matches:
                    w = world3d[di]
                    if np.any(np.isnan(w)):
                        continue
                    pts_2d.append(q_kps[qi])
                    pts_3d.append(w)

                if len(pts_3d) < 4:
                    continue

                pts_2d = np.array(pts_2d, dtype=np.float64)
                pts_3d = np.array(pts_3d, dtype=np.float64)

                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d, pts_2d, K, None,
                    flags=cv2.SOLVEPNP_EPNP,
                    reprojectionError=8.0,
                    confidence=0.99,
                    iterationsCount=1000,
                )

                if not ok or inliers is None or len(inliers) < 8:
                    continue

                n_in = len(inliers)
                # confidence = inlier ratio. wrong-keyframe match 자동 reject.
                # 이번 세션 검증: lenient (4) 시 wrong location 자신있게 응답.
                # 8 + 0.30 이 wrong-match 차단의 안전 임계.
                _conf_local = n_in / max(len(pts_3d), 1)
                if _conf_local < 0.35:
                    continue

                # LM refinement — RANSAC 결과를 inlier subset 으로 LM 최적화 (sub-pixel 정밀).
                # inliers 안 줄임. EPNP 의 coarse pose 를 더 정확하게 만든다.
                inlier_idx = inliers.flatten()
                pts_3d_in = pts_3d[inlier_idx]
                pts_2d_in = pts_2d[inlier_idx]
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        pts_3d_in, pts_2d_in, K, None, rvec, tvec,
                    )
                except Exception:
                    pass  # refine 실패 시 EPNP 결과 그대로

                # PnP 결과 (OpenCV optical) → RTABMap base convention 변환.
                # base→optical transform M 은 device/scan 의 calibration 따라 다를 수 있으므로
                # SuperPointMapManager 가 cache 한 DB calibration 사용 (하드코딩 금지).
                # R_base_in_world = R_oc @ M  (optical→world × base→optical = base axes in world)
                R_w2c, _ = cv2.Rodrigues(rvec)
                R_oc = R_w2c.T
                M_optical_to_base = loaded.optical_to_base if loaded.optical_to_base is not None else np.array([
                    [0,  0,  1],
                    [-1, 0,  0],
                    [0, -1,  0],
                ], dtype=np.float64)  # fallback (옛 데이터)
                R_cw = R_oc @ M_optical_to_base.T
                t_cw = (-R_w2c.T @ tvec).flatten()
                depth_fusion = None
                if query_depth is not None:
                    depth_pose = _depth_3d_pose_candidate(
                        depth=query_depth,
                        pts_2d=pts_2d_in,
                        pts_3d_world=pts_3d_in,
                        image_shape=gray.shape,
                        K=K,
                        pnp_R_oc=R_oc,
                        pnp_t_cw=t_cw,
                    )
                    pred_cam = (R_w2c @ pts_3d_in.T + tvec).T
                    pred_depth = pred_cam[:, 2]
                    obs_depth = _sample_depths(query_depth, pts_2d_in, gray.shape)
                    valid_depth = (
                        np.isfinite(obs_depth)
                        & np.isfinite(pred_depth)
                        & (obs_depth > 0.15)
                        & (pred_depth > 0.15)
                        & (pred_depth < 10.0)
                    )
                    depth_fusion = {
                        "received": True,
                        "applied": False,
                        "mode": "depth_3d_fixed_rotation_translation",
                        "validMatches": int(np.count_nonzero(valid_depth)),
                        "totalInliers": int(len(pts_2d_in)),
                        "depthPose": depth_pose,
                    }
                    if np.count_nonzero(valid_depth) >= 8:
                        residual = obs_depth[valid_depth] - pred_depth[valid_depth]
                        median = float(np.median(residual))
                        mad = float(np.median(np.abs(residual - median)))
                        robust = np.abs(residual - median) <= max(0.25, 3.0 * mad)
                        inlier_ratio = float(np.count_nonzero(robust) / max(len(residual), 1))
                        correction = float(np.clip(median, -0.50, 0.50))
                        depth_fusion.update(
                            {
                                "inlierRatio": inlier_ratio,
                                "residualMedianM": median,
                                "residualMadM": mad,
                                "forwardCorrectionM": correction,
                            }
                        )
                        if depth_pose.get("applied") and depth_pose.get("translation") is not None:
                            depth_t = np.array(depth_pose["translation"], dtype=np.float64)
                            delta = float(np.linalg.norm(depth_t - t_cw))
                            t_cw = depth_t
                            material_correction = delta > 0.01
                            depth_fusion.update(
                                {
                                    "applied": material_correction,
                                    "correctionM": delta,
                                    "translationDeltaM": delta,
                                    "depth3dPoseUsed": True,
                                    "depth3dApplied": material_correction,
                                }
                            )
                        elif inlier_ratio >= 0.55 and abs(median) <= 1.25 and mad <= 0.65:
                            forward_world = R_oc[:, 2]
                            t_cw = t_cw - correction * forward_world
                            depth_fusion.update(
                                {
                                    "applied": abs(correction) > 0.01,
                                    "mode": "pnp_forward_depth_bias",
                                    "correctionM": correction,
                                    "depth3dApplied": False,
                                    "forwardCorrectionApplied": abs(correction) > 0.01,
                                }
                            )
                        else:
                            depth_fusion["rejectReason"] = "depth_residual_unstable"
                    else:
                        depth_fusion["rejectReason"] = "insufficient_valid_depth_matches"

                qx, qy, qz, qw = _rotation_to_quat(R_cw)
                confidence = min(0.99, max(0.01, n_in / len(pts_3d)))

                candidate = {
                    'matched_node_id': int(node_id),
                    'num_matches': n_in,
                    'confidence': confidence,
                    'matched_image_index': img_idx,
                    'pose': {
                        'x': float(t_cw[0]), 'y': float(t_cw[1]), 'z': float(t_cw[2]),
                        'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
                    },
                }
                if depth_fusion is not None:
                    candidate["depth_fusion"] = depth_fusion
                node_pose = loaded.node_poses.get(node_id)
                if node_pose is not None and loaded.optical_to_base is not None:
                    db_kps = (
                        loaded.keyframe_feats[node_id]['keypoints'][0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    rtab_cmp = _rtab_relative_pose_candidate(
                        q_kps=q_kps,
                        db_kps=db_kps,
                        matches=matches,
                        K=K,
                        node_pose=node_pose,
                        optical_to_base=loaded.optical_to_base,
                        pnp_pose=candidate['pose'],
                    )
                    if rtab_cmp is not None:
                        if rtab_cmp['relative_inlier_ratio'] < 0.30:
                            continue
                        candidate['rtab_relative_compare'] = rtab_cmp
                if image_best is None or n_in > image_best['num_matches']:
                    image_best = candidate

            if image_best is not None:
                burst_best_per_image.append(image_best)
                if best is None or image_best['num_matches'] > best['num_matches']:
                    best = image_best

        if best is None:
            raise ValueError("SuperPoint+LightGlue: insufficient matches")

        # burst 의 valid pose 들을 inliers-가중 평균 (translation). rotation 은 best 의 quat 사용.
        # 서로 다른 keyframe/hypothesis 를 평균하면 복도 반대편 후보가 섞여 실제로 없는 중간 pose 가 된다.
        # 따라서 best 와 같은 matched node 이면서 가까운 pose 만 평균한다.
        if len(burst_best_per_image) >= 2:
            best_t = np.array(
                [best['pose']['x'], best['pose']['y'], best['pose']['z']],
                dtype=np.float64,
            )
            best_node_id = best.get('matched_node_id')
            average_candidates = []
            for c in burst_best_per_image:
                c_t = np.array(
                    [c['pose']['x'], c['pose']['y'], c['pose']['z']],
                    dtype=np.float64,
                )
                same_node = c.get('matched_node_id') == best_node_id
                close_to_best = np.linalg.norm(c_t - best_t) <= 0.75
                if same_node and close_to_best:
                    average_candidates.append(c)

            if len(average_candidates) < 2:
                average_candidates = [best]

            total_w = float(sum(c['num_matches'] for c in average_candidates))
            tx_avg = sum(c['pose']['x'] * c['num_matches'] for c in average_candidates) / total_w
            ty_avg = sum(c['pose']['y'] * c['num_matches'] for c in average_candidates) / total_w
            tz_avg = sum(c['pose']['z'] * c['num_matches'] for c in average_candidates) / total_w
            best = {
                **best,
                'pose': {**best['pose'], 'x': tx_avg, 'y': ty_avg, 'z': tz_avg},
            }
            rtab_cmp = best.get('rtab_relative_compare')
            if rtab_cmp is not None:
                sp_t = np.array([tx_avg, ty_avg, tz_avg], dtype=np.float64)
                rtab_pose = rtab_cmp['pose']
                rtab_t = np.array(
                    [rtab_pose['x'], rtab_pose['y'], rtab_pose['z']], dtype=np.float64
                )
                rtab_cmp['position_delta_m'] = float(np.linalg.norm(sp_t - rtab_t))

        logger.info(
            f"[SuperPoint] map={map_id} inliers={best['num_matches']} "
            f"confidence={best['confidence']:.3f} burst_valid={len(burst_best_per_image)}/{len(images)}"
        )
        rtab_cmp = best.get('rtab_relative_compare')
        if rtab_cmp is not None:
            rtab_pose = rtab_cmp['pose']
            sp_pose = best['pose']
            logger.info(
                "[LOCALIZE-COMPARE] map=%s node=%s img=%s "
                "sp_xyz=(%.3f,%.3f,%.3f) sp_yaw=%.1f "
                "rtab_rel_xyz=(%.3f,%.3f,%.3f) rtab_rel_yaw=%.1f "
                "pos_delta=%.3fm yaw_delta=%.1fdeg rel_inliers=%d ratio=%.3f",
                map_id,
                best.get('matched_node_id'),
                best.get('matched_image_index'),
                sp_pose['x'], sp_pose['y'], sp_pose['z'], _pose_yaw_deg(sp_pose),
                rtab_pose['x'], rtab_pose['y'], rtab_pose['z'], _pose_yaw_deg(rtab_pose),
                rtab_cmp['position_delta_m'],
                rtab_cmp['yaw_delta_deg'],
                rtab_cmp['relative_inliers'],
                rtab_cmp['relative_inlier_ratio'],
            )
        return {**best, 'map_id': map_id, 'method': 'SuperPoint+LightGlue'}

    async def localize(
        self,
        map_id: str,
        images: List[bytes],
        intrinsics: Optional[Dict] = None,
        initial_pose: Optional[Dict] = None,
        db_path: Optional[str] = None,
        **kwargs,
    ) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._localize_sync,
                map_id,
                images,
                intrinsics,
                db_path,
                kwargs.get("query_depths"),
            ),
        )
