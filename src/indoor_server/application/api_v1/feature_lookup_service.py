"""V1 feature-points lookup service.

클라가 좌표 배열을 보내면, 각 좌표에 가까운 keyframe 들의 SuperPoint
feature pack 을 반환. route bundle 의 generic 한 superset 으로, 클라가
경로 전체 좌표를 보내면 route bundle 효과 동등.
"""
from __future__ import annotations

import base64
import math
import sqlite3
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from indoor_server.application.api_v1.building_floor_service import (
    ActiveScan,
    BuildingFloorService,
)
from indoor_server.application.api_v1.errors import V1ServiceError
from indoor_server.config import settings
from indoor_server.interfaces.api.v1_schemas import (
    FeatureLookupIntrinsics,
    FeatureLookupKeyframe,
    FeatureLookupModel,
    FeatureLookupRequest,
    FeatureLookupResponse,
    FeatureLookupStats,
)


_MAX_QUERIES = 64
_MAX_KEYFRAMES_PER_QUERY = 16
_MAX_TOTAL_KEYFRAMES = 128


@dataclass(frozen=True)
class _FloorIndex:
    scan: ActiveScan
    rtab_db_path: str
    intrinsics: FeatureLookupIntrinsics
    node_poses: dict[int, np.ndarray]  # rtabmap node_id → 4x4


class FeaturePointLookupService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def lookup(
        self,
        *,
        building_id: UUID,
        request: FeatureLookupRequest,
    ) -> FeatureLookupResponse:
        if not request.queries:
            raise V1ServiceError(422, "EMPTY_QUERIES", "queries must not be empty.")
        if len(request.queries) > _MAX_QUERIES:
            raise V1ServiceError(
                422,
                "TOO_MANY_QUERIES",
                f"queries exceed limit {_MAX_QUERIES}.",
                {"queryCount": len(request.queries)},
            )

        opts = request.options
        max_per_query = min(opts.max_keyframes_per_query, _MAX_KEYFRAMES_PER_QUERY)

        scans = await BuildingFloorService(self._session).get_active_scans_for_building(
            building_id
        )
        if not scans:
            raise V1ServiceError(
                404, "ACTIVE_SCAN_NOT_FOUND", "building has no active scans."
            )
        scan_by_level: dict[int, ActiveScan] = {s.floor_level: s for s in scans}

        # query index 를 floor_level 별 그룹화
        queries_by_level: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for idx, q in enumerate(request.queries):
            if q.floor_level not in scan_by_level:
                raise V1ServiceError(
                    404,
                    "FLOOR_NOT_FOUND",
                    "queried floorLevel has no active scan.",
                    {"floorLevel": q.floor_level, "queryIndex": idx},
                )
            queries_by_level[q.floor_level].append((idx, q))

        # 각 floor 에 대해: SuperPoint cache + rtab pose 로드, query → keyframe 후보 수집
        # node_id 는 floor 별로 unique 하지만 floor 간 충돌 가능 → (floor_level, node_id) 로 dedup
        candidates: dict[
            tuple[int, int], dict[str, Any]
        ] = {}  # key: (floor_level, node_id)
        for level, queries in queries_by_level.items():
            scan = scan_by_level[level]
            floor_idx = self._load_floor_index(scan)
            loaded = self._load_superpoint_map(scan, floor_idx.rtab_db_path)

            for q_idx, q in queries:
                hits = self._select_keyframes_for_query(
                    q=q,
                    loaded=loaded,
                    node_poses=floor_idx.node_poses,
                    radius_m=opts.radius_m,
                    max_per_query=max_per_query,
                    view_cone_deg=opts.view_cone_deg,
                )
                for dist, node_id in hits:
                    key = (level, node_id)
                    if key not in candidates:
                        candidates[key] = {
                            "scan": scan,
                            "floor_level": level,
                            "node_id": node_id,
                            "loaded": loaded,
                            "intrinsics": floor_idx.intrinsics,
                            "pose": floor_idx.node_poses[node_id],
                            "matched_queries": [],
                            "distances": [],
                        }
                    candidates[key]["matched_queries"].append(q_idx)
                    candidates[key]["distances"].append(dist)

        if len(candidates) > _MAX_TOTAL_KEYFRAMES:
            raise V1ServiceError(
                422,
                "TOO_MANY_KEYFRAMES",
                f"resulting keyframe set exceeds limit {_MAX_TOTAL_KEYFRAMES}. "
                "Tighten radiusM or maxKeyframesPerQuery.",
                {"keyframeCount": len(candidates)},
            )

        # 직렬화
        out_keyframes: list[FeatureLookupKeyframe] = []
        total_keypoints = 0
        total_bytes = 0
        for info in candidates.values():
            kf, kp_count, byte_size = self._serialize_keyframe(info)
            out_keyframes.append(kf)
            total_keypoints += kp_count
            total_bytes += byte_size

        return FeatureLookupResponse(
            building_id=building_id,
            keyframes=out_keyframes,
            model=FeatureLookupModel(),
            stats=FeatureLookupStats(
                query_count=len(request.queries),
                keyframe_count=len(out_keyframes),
                total_keypoints=total_keypoints,
                byte_size=total_bytes,
            ),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_floor_index(self, scan: ActiveScan) -> _FloorIndex:
        rtab_db = settings.storage_root / "scans" / scan.scan_id / "rtabmap.db"
        if not rtab_db.exists():
            raise V1ServiceError(
                404,
                "RTABMAP_DB_NOT_FOUND",
                "rtabmap.db missing for active scan.",
                {"scanId": scan.scan_id},
            )

        intrinsics, node_poses = self._read_rtab_db(str(rtab_db))
        return _FloorIndex(
            scan=scan,
            rtab_db_path=str(rtab_db),
            intrinsics=intrinsics,
            node_poses=node_poses,
        )

    @staticmethod
    def _read_rtab_db(
        rtab_db_path: str,
    ) -> tuple[FeatureLookupIntrinsics, dict[int, np.ndarray]]:
        con = sqlite3.connect(rtab_db_path)
        try:
            row = con.execute(
                "SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1"
            ).fetchone()
            if row is None or row[0] is None:
                raise V1ServiceError(
                    422, "CALIBRATION_MISSING", "rtabmap.db has no calibration row."
                )
            calib_blob = bytes(row[0])
            from be.slam_engines.superpoint.map_manager import (
                _parse_calibration_K_and_local,
            )

            K, _ = _parse_calibration_K_and_local(calib_blob)
            # blob layout: width @ offset 16, height @ 20 (int32) — see export_route_bundle.py.
            width = struct.unpack("<i", calib_blob[16:20])[0]
            height = struct.unpack("<i", calib_blob[20:24])[0]
            intrinsics = FeatureLookupIntrinsics(
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                width=int(width),
                height=int(height),
            )

            node_poses: dict[int, np.ndarray] = {}
            for nid, blob in con.execute(
                "SELECT id, pose FROM Node WHERE pose IS NOT NULL"
            ):
                if blob and len(blob) == 48:
                    vals = struct.unpack("<12f", blob)
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)
                    node_poses[int(nid)] = T
            return intrinsics, node_poses
        finally:
            con.close()

    @staticmethod
    def _load_superpoint_map(scan: ActiveScan, rtab_db_path: str):
        from be.slam_engines.superpoint.map_manager import SuperPointMapManager

        # map_id = floor_id (singleton cache key — localize 와 일관)
        return SuperPointMapManager().get_or_load(scan.floor_id, rtab_db_path)

    @staticmethod
    def _select_keyframes_for_query(
        *,
        q,
        loaded,
        node_poses: dict[int, np.ndarray],
        radius_m: float,
        max_per_query: int,
        view_cone_deg: float | None,
    ) -> list[tuple[float, int]]:
        cone_cos = (
            math.cos(math.radians(view_cone_deg / 2.0))
            if (view_cone_deg is not None and q.view_direction is not None)
            else None
        )
        view_dir = None
        if cone_cos is not None and q.view_direction is not None:
            v = np.array(q.view_direction, dtype=np.float64)
            n = np.linalg.norm(v)
            if n > 1e-9:
                view_dir = v / n

        candidates: list[tuple[float, int]] = []
        for nid in loaded.node_ids:
            pose = node_poses.get(nid)
            if pose is None:
                continue
            tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
            dx, dy, dz = tx - q.x, ty - q.y, tz - q.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > radius_m:
                continue
            if view_dir is not None:
                # camera forward = pose 의 +Z 축 (rotation column 2)
                kf_forward = pose[:3, 2]
                kf_forward = kf_forward / (np.linalg.norm(kf_forward) + 1e-9)
                if float(np.dot(view_dir, kf_forward)) < cone_cos:
                    continue
            candidates.append((dist, int(nid)))

        candidates.sort(key=lambda t: t[0])
        return candidates[:max_per_query]

    @staticmethod
    def _serialize_keyframe(info: dict[str, Any]) -> tuple[FeatureLookupKeyframe, int, int]:
        loaded = info["loaded"]
        node_id = info["node_id"]
        scan: ActiveScan = info["scan"]
        pose: np.ndarray = info["pose"]

        feats = loaded.keyframe_feats[node_id]
        kp = feats["keypoints"][0].cpu().numpy().astype(np.float32)  # (N, 2)
        desc = feats["descriptors"][0].cpu().numpy().astype(np.float16)  # (N, 256)
        world3d = loaded.keyframe_world3d[node_id].astype(np.float32)  # (N, 3)

        gdesc_b64: str | None = None
        if loaded.global_descs is not None:
            try:
                gidx = loaded.node_ids.index(node_id)
                gdesc = (
                    loaded.global_descs[gidx].cpu().numpy().astype(np.float16)
                )  # (384,)
                gdesc_b64 = _b64(gdesc)
            except (ValueError, IndexError):
                gdesc_b64 = None

        kp_b64 = _b64(kp)
        desc_b64 = _b64(desc)
        w3d_b64 = _b64(world3d)

        byte_size = (
            len(kp_b64) + len(desc_b64) + len(w3d_b64) + (len(gdesc_b64) if gdesc_b64 else 0)
        )

        kf = FeatureLookupKeyframe(
            kf_id=f"{scan.scan_id}:{node_id}",
            scan_id=UUID(scan.scan_id),
            floor_level=info["floor_level"],
            rtabmap_node_id=node_id,
            pose=pose.astype(np.float32).tolist(),
            intrinsics=info["intrinsics"],
            matched_query_indices=list(info["matched_queries"]),
            distances_m=[round(float(d), 4) for d in info["distances"]],
            keypoint_count=int(kp.shape[0]),
            keypoints=kp_b64,
            descriptors=desc_b64,
            world3d=w3d_b64,
            global_descriptor=gdesc_b64,
        )
        return kf, int(kp.shape[0]), byte_size


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")
