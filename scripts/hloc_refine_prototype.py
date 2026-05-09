"""hloc-driven SfM refinement prototype.

표준 흐름 (hloc 가 다 처리):
  1. RTABMap rtabmap.db Data.data blob → image dump (PNG)
  2. ARKit pose → COLMAP reference_model (empty points)
  3. extract_features (SuperPoint) → features.h5
  4. pairs_from_retrieval (또는 sequential)
  5. match_features (SuperPoint+LightGlue) → matches.h5
  6. triangulation.main(reference_model, ...) → BA-refined Reconstruction
  7. recon.points3D → keyframe_world3d 갱신
  8. 4건 query 재측위 + 비교
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hloc_refine")

RTAB_DB = "/app/var/storage/scans/bc5e28b9-5075-4a58-b948-31ab1e277a02/rtabmap.db"
MAP_ID = "125b153d-3475-46f9-9690-e703951fb83d"
DEBUG_DIR = Path("/app/var/storage/debug/localize/e30f31ea-5bbe-42df-9031-fa371bb7a7b3")
WORK_DIR = Path("/tmp/hloc_refine")

QUERIES = [
    ("20260508_125358_467012", 0.643, 9, ( 0.527, -21.743, 0.230)),
    ("20260508_131737_173495", 0.205, 9, ( 0.030, -18.300, 1.032)),
    ("20260508_131801_428982", 0.323, 10, ( 0.131,  -6.501, -0.044)),
    ("20260508_132936_089777", 0.611, 11, ( 0.680, -21.669, 0.194)),
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — RTABMap blob → image dump + metadata
# ─────────────────────────────────────────────────────────────────────────────


def dump_images_and_metadata(
    rtab_db: str, work_dir: Path, allowed_node_ids: set[int] | None = None
) -> dict:
    """RTABMap rtabmap.db 의 image blob 을 디스크에 dump + intrinsics/pose 추출.

    `allowed_node_ids` 가 주어지면 그 집합에 속한 frame 만 dump + pose 등록 —
    SuperPoint cache 의 keyframe selection 결과와 일치시키기 위함.
    """
    from be.slam_engines.superpoint.map_manager import _parse_calibration_K_and_local

    images_dir = work_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(rtab_db)
    try:
        cb = bytes(con.execute(
            "SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1"
        ).fetchone()[0])
        K, local_transform = _parse_calibration_K_and_local(cb)
        width = struct.unpack("<i", cb[16:20])[0]
        height = struct.unpack("<i", cb[20:24])[0]

        node_poses: dict[int, np.ndarray] = {}
        for nid, blob in con.execute(
            "SELECT id, pose FROM Node WHERE pose IS NOT NULL ORDER BY id"
        ):
            if blob and len(blob) == 48:
                if allowed_node_ids is not None and int(nid) not in allowed_node_ids:
                    continue
                vals = struct.unpack("<12f", blob)
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(vals).reshape(3, 4)
                node_poses[int(nid)] = T

        n_dumped = 0
        image_names: list[str] = []
        for nid in node_poses.keys():
            row = con.execute(
                "SELECT image FROM Data WHERE id = ? AND image IS NOT NULL", (nid,)
            ).fetchone()
            if row is None or row[0] is None:
                continue
            blob = bytes(row[0])
            name = f"frame_{nid:06d}.jpg"
            (images_dir / name).write_bytes(blob)
            image_names.append(name)
            n_dumped += 1
        logger.info(f"  dumped {n_dumped} images to {images_dir}")
    finally:
        con.close()

    return {
        "K": K, "local_transform": local_transform,
        "width": int(width), "height": int(height),
        "node_poses": node_poses,
        "images_dir": images_dir,
        "image_names": sorted(image_names),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — ARKit pose → COLMAP reference_model (no points)
# ─────────────────────────────────────────────────────────────────────────────


def arkit_pose_to_colmap_qt(
    T_b_w: np.ndarray, local_transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ARKit base→world (4x4) + base→optical (3x4) → world→optical 의
    quaternion (w,x,y,z) + translation."""
    T_b_opt = np.eye(4, dtype=np.float64)
    T_b_opt[:3, :3] = local_transform[:3, :3]
    T_b_opt[:3, 3] = local_transform[:3, 3]
    T_w_opt = T_b_opt @ np.linalg.inv(T_b_w)
    R = T_w_opt[:3, :3]
    t = T_w_opt[:3, 3]
    import scipy.spatial.transform as st
    quat_xyzw = st.Rotation.from_matrix(R).as_quat()
    qvec = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return qvec, t


def build_reference_model(meta: dict, work_dir: Path) -> Path:
    """ARKit pose → COLMAP text-format empty model (cameras + images, no points)."""
    K = meta["K"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    ref = work_dir / "reference_model"
    ref.mkdir(parents=True, exist_ok=True)

    # cameras.txt
    (ref / "cameras.txt").write_text(
        f"# Camera list with one line of data per camera:\n"
        f"#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {meta['width']} {meta['height']} {fx} {fy} {cx} {cy}\n"
    )

    # images.txt — format:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[]
    lines: list[str] = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    n_images = 0
    for nid, T_b_w in meta["node_poses"].items():
        if f"frame_{nid:06d}.jpg" not in meta["image_names"]:
            continue
        qvec, tvec = arkit_pose_to_colmap_qt(T_b_w, meta["local_transform"])
        qw, qx, qy, qz = qvec.tolist()
        tx, ty, tz = tvec.tolist()
        lines.append(
            f"{nid} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
            f"{tx:.10f} {ty:.10f} {tz:.10f} 1 frame_{nid:06d}.jpg"
        )
        lines.append("")  # 빈 POINTS2D 행
        n_images += 1
    (ref / "images.txt").write_text("\n".join(lines) + "\n")

    # points3D.txt — empty
    (ref / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    )

    logger.info(f"  reference model: {n_images} images → {ref}")
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# Step 3-5 — hloc 표준 흐름
# ─────────────────────────────────────────────────────────────────────────────


def export_features_h5(loaded, image_name_for_node: dict[int, str], features_path: Path):
    """우리 SuperPoint cache → hloc 형식 features.h5."""
    import h5py
    with h5py.File(features_path, 'w', libver='latest') as f:
        for nid in loaded.node_ids:
            if nid not in image_name_for_node:
                continue
            name = image_name_for_node[nid]
            grp = f.create_group(name)
            kps = loaded.keyframe_feats[nid]['keypoints'][0].cpu().numpy().astype(np.float32)
            grp.create_dataset('keypoints', data=kps)
            desc = loaded.keyframe_feats[nid]['descriptors'][0].cpu().numpy().astype(np.float32)  # (N, 256)
            # hloc 표준은 (D, N) — descriptor 첫 dim
            grp.create_dataset('descriptors', data=desc.T)
            if 'image_size' in loaded.keyframe_feats[nid]:
                img_size = loaded.keyframe_feats[nid]['image_size'][0].cpu().numpy().astype(np.int64)
                grp.create_dataset('image_size', data=img_size)


def compute_pair_matches_lightglue(loaded, neighbor_window: int = 5, min_matches: int = 12):
    """이웃 ±window LightGlue 매칭 결과 반환."""
    import torch
    from lightglue import LightGlue
    from lightglue.utils import rbd
    matcher = LightGlue(features='superpoint').eval().to(loaded.device)

    pair_matches: dict[tuple[int, int], np.ndarray] = {}
    pair_scores: dict[tuple[int, int], np.ndarray] = {}
    n = len(loaded.node_ids)
    seen: set[tuple[int, int]] = set()
    with torch.no_grad():
        for i in range(n):
            nid_i = loaded.node_ids[i]
            feats_i = {k: v.to(loaded.device) for k, v in loaded.keyframe_feats[nid_i].items()}
            for off in range(1, neighbor_window + 1):
                for j in (i - off, i + off):
                    if not (0 <= j < n) or j == i:
                        continue
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    nid_j = loaded.node_ids[j]
                    feats_j = {k: v.to(loaded.device) for k, v in loaded.keyframe_feats[nid_j].items()}
                    out = matcher({'image0': feats_i, 'image1': feats_j})
                    out = rbd(out)
                    matches = out['matches'].cpu().numpy()
                    if len(matches) < min_matches:
                        continue
                    scores = out.get('scores', None)
                    if scores is not None:
                        scores = scores.cpu().numpy().astype(np.float32)
                    else:
                        scores = np.ones(len(matches), dtype=np.float32)
                    pair_matches[(nid_i, nid_j)] = matches
                    pair_scores[(nid_i, nid_j)] = scores
    return pair_matches, pair_scores


def export_matches_h5(
    pair_matches: dict, pair_scores: dict, image_name_for_node: dict[int, str],
    n_kp_per_node: dict[int, int], matches_path: Path,
):
    """LightGlue 매칭 결과 → hloc 형식 matches.h5.

    pair_key = "{name0}/{name1}"  (names_to_pair)
    matches0 : (N0,) int32 — image0 kp idx 별 matching image1 kp idx 또는 -1
    matching_scores0 : (N0,) float32
    """
    import h5py
    with h5py.File(matches_path, 'w', libver='latest') as f:
        for (nid_i, nid_j), matches in pair_matches.items():
            name0 = image_name_for_node[nid_i]
            name1 = image_name_for_node[nid_j]
            pair_key = f"{name0}/{name1}"
            grp = f.create_group(pair_key)
            n0 = n_kp_per_node[nid_i]
            matches0 = np.full(n0, -1, dtype=np.int32)
            scores0 = np.full(n0, 0.0, dtype=np.float32)
            scores = pair_scores[(nid_i, nid_j)]
            for k, (ki, kj) in enumerate(matches):
                matches0[int(ki)] = int(kj)
                scores0[int(ki)] = float(scores[k]) if k < len(scores) else 1.0
            grp.create_dataset('matches0', data=matches0)
            grp.create_dataset('matching_scores0', data=scores0)


def run_hloc_pipeline(meta: dict, work_dir: Path, ref_model_dir: Path):
    """우리 SuperPoint cache + LightGlue 매칭 → h5 export → hloc.triangulation.

    핵심 수정 — pycolmap.triangulate_points 직접 호출 (hloc.triangulation.main 우회):
    - triangulation.max_transitivity = 5 (default 1 → 5, multi-view chain 활성)
    - triangulation.ignore_two_view_tracks = False (2-view 도 keep)
    - mapper.fix_existing_frames = True (ARKit pose 고정, BA drift 차단)
    - mapper.ba_local/filter_min_tri_angle 완화
    """
    from hloc import triangulation
    import pycolmap
    from be.slam_engines.superpoint.map_manager import SuperPointMapManager

    images_dir = meta["images_dir"]
    image_names = meta["image_names"]

    sfm_dir = work_dir / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    features_path = work_dir / "features.h5"
    matches_path = work_dir / "matches.h5"
    pairs_path = work_dir / "pairs.txt"

    # 0) SuperPoint cache
    mgr = SuperPointMapManager()
    loaded = mgr.get_or_load(MAP_ID, RTAB_DB)
    image_name_for_node = {nid: f"frame_{nid:06d}.jpg" for nid in loaded.node_ids}
    n_kp_per_node = {
        nid: loaded.keyframe_feats[nid]['keypoints'].shape[1]
        for nid in loaded.node_ids
    }

    # 1) features.h5
    logger.info("export_features_h5 (cache → h5)")
    t0 = time.time()
    export_features_h5(loaded, image_name_for_node, features_path)
    logger.info(f"   features.h5 in {time.time()-t0:.1f}s")

    # 2) matching (LightGlue, neighbor window=5)
    logger.info("LightGlue pair matching (±5)")
    t0 = time.time()
    pair_matches, pair_scores = compute_pair_matches_lightglue(loaded, neighbor_window=15)
    logger.info(f"   pairs: {len(pair_matches)} in {time.time()-t0:.1f}s")

    # 3) matches.h5
    logger.info("export_matches_h5 (LightGlue → h5)")
    export_matches_h5(pair_matches, pair_scores, image_name_for_node,
                      n_kp_per_node, matches_path)

    # 4) pairs.txt
    pairs_path.write_text("\n".join(
        f"{image_name_for_node[i]} {image_name_for_node[j]}"
        for (i, j) in pair_matches.keys()
    ))
    logger.info(f"   pairs.txt: {len(pair_matches)} pairs")

    # 5) triangulation
    logger.info("hloc.triangulation.main (track + BA)")
    t0 = time.time()
    # hloc.triangulation.main 의 pre-processing (DB 생성 + import) 만 활용 후
    # pycolmap.triangulate_points 는 직접 호출 (옵션 객체 직접 구성).
    from hloc.triangulation import (
        create_db_from_model, import_features, import_matches,
        estimation_and_geometric_verification,
    )

    database_path = sfm_dir / "database.db"
    if database_path.exists():
        database_path.unlink()
    reference = pycolmap.Reconstruction(str(ref_model_dir))
    image_ids = create_db_from_model(reference, database_path)
    with pycolmap.Database.open(str(database_path)) as db:
        import_features(image_ids, db, features_path)
        import_matches(image_ids, db, pairs_path, matches_path,
                       min_match_score=None, skip_geometric_verification=False)
    estimation_and_geometric_verification(database_path, pairs_path, verbose=False)

    # IncrementalPipelineOptions — 우리 데이터에 맞춤
    opts = pycolmap.IncrementalPipelineOptions()
    opts.triangulation.max_transitivity = 5
    opts.triangulation.ignore_two_view_tracks = False
    opts.triangulation.complete_max_transitivity = 5
    opts.mapper.fix_existing_frames = True              # ARKit pose 고정
    opts.mapper.filter_min_tri_angle = 0.5              # default 1.5 완화
    opts.mapper.ba_local_min_tri_angle = 0.5            # default 6.0 → 0.5 (우리 3.4°)

    logger.info("pycolmap.triangulate_points (custom options)")
    t0 = time.time()
    recon = pycolmap.triangulate_points(
        reference, str(database_path), str(images_dir), str(sfm_dir / "model_out"),
        clear_points=True, options=opts,
    )
    logger.info(f"   reconstruction in {time.time()-t0:.1f}s: "
                f"{len(recon.images)} images, {len(recon.points3D)} points3D")
    logger.info(f"   summary: {recon.summary()[:300]}")
    logger.info(f"   reconstruction in {time.time()-t0:.1f}s: "
                f"{len(recon.images)} images, {len(recon.points3D)} points3D")
    return recon


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — recon → keyframe_world3d
# ─────────────────────────────────────────────────────────────────────────────


def export_to_world3d(recon, loaded) -> int:
    n_updated = 0
    new_w3d: dict[int, np.ndarray] = {}
    for nid in loaded.node_ids:
        kp_count = loaded.keyframe_feats[nid]['keypoints'].shape[1]
        new_w3d[nid] = np.full((kp_count, 3), np.nan, dtype=np.float32)

    for pid, pt3d in recon.points3D.items():
        xyz = np.array(pt3d.xyz, dtype=np.float32)
        for elem in pt3d.track.elements:
            nid = int(elem.image_id)
            kp = int(elem.point2D_idx)
            if nid in new_w3d and 0 <= kp < new_w3d[nid].shape[0]:
                new_w3d[nid][kp] = xyz
                n_updated += 1
    loaded.keyframe_world3d = new_w3d
    return n_updated


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # SuperPoint cache 먼저 로드 — keyframe selection 적용된 node_ids 알아내기.
    logger.info("=" * 78)
    logger.info("Step 0 — SuperPoint cache 로드 (smart keyframe selection 결과 확인)")
    from be.slam_engines.superpoint.map_manager import SuperPointMapManager
    mgr = SuperPointMapManager()
    if MAP_ID in mgr._maps:
        del mgr._maps[MAP_ID]
    loaded = mgr.get_or_load(MAP_ID, RTAB_DB)
    cache_node_ids: set[int] = set(int(nid) for nid in loaded.node_ids)
    logger.info(f"  cache frames: {len(cache_node_ids)}")

    logger.info("=" * 78)
    logger.info("Step 1 — RTABMap → image dump + metadata (cache frames only)")
    meta = dump_images_and_metadata(RTAB_DB, WORK_DIR, allowed_node_ids=cache_node_ids)

    logger.info("=" * 78)
    logger.info("Step 2 — ARKit pose → COLMAP reference model")
    ref_model_dir = build_reference_model(meta, WORK_DIR)

    logger.info("=" * 78)
    logger.info("Step 3-5 — hloc pipeline (features + match + triangulate + BA)")
    recon = run_hloc_pipeline(meta, WORK_DIR, ref_model_dir)

    logger.info("=" * 78)
    logger.info("Step 6 — keyframe_world3d 갱신 (SuperPointMapManager 캐시)")
    from be.slam_engines.superpoint.map_manager import SuperPointMapManager
    mgr = SuperPointMapManager()
    if MAP_ID in mgr._maps:
        del mgr._maps[MAP_ID]
    loaded = mgr.get_or_load(MAP_ID, RTAB_DB)
    n_upd = export_to_world3d(recon, loaded)
    n_with_3d = sum(
        1 for w in loaded.keyframe_world3d.values()
        if not np.all(np.isnan(w))
    )
    total_kp = sum(w.shape[0] for w in loaded.keyframe_world3d.values())
    total_3d = sum(int((~np.isnan(w).any(axis=1)).sum()) for w in loaded.keyframe_world3d.values())
    logger.info(f"   updated obs: {n_upd}")
    logger.info(f"   3D coverage frames : {n_with_3d}/{len(loaded.node_ids)} "
                f"({100*n_with_3d/len(loaded.node_ids):.1f}%)")
    logger.info(f"   3D-mapped keypoints: {total_3d}/{total_kp} "
                f"({100*total_3d/total_kp:.2f}%)")

    logger.info("=" * 78)
    logger.info("Step 7 — 4건 query 재측위")
    from be.slam_engines.superpoint.engine import SuperPointEngine
    eng = SuperPointEngine()
    intrinsics = {
        "fx": float(meta["K"][0, 0]), "fy": float(meta["K"][1, 1]),
        "cx": float(meta["K"][0, 2]), "cy": float(meta["K"][1, 2]),
    }
    print(f"\n{'time':25}{'recorded':>20}{'new conf':>10}{'inliers':>9}{'pose':>22}")
    for ts, old_c, old_n, _ in QUERIES:
        imgs = []
        for k in range(5):
            p = DEBUG_DIR / f"{ts}_{k:02d}.jpg"
            if p.exists():
                imgs.append(p.read_bytes())
        if not imgs:
            continue
        try:
            res = eng._localize_sync(MAP_ID, imgs, intrinsics, RTAB_DB)
            nx, ny, nz = res['pose']['x'], res['pose']['y'], res['pose']['z']
            rec = f"c={old_c:.2f} n={old_n}"
            print(f"{ts:25}{rec:>20}{res['confidence']:>10.3f}{res['num_matches']:>9}"
                  f"{f'({nx:.2f}, {ny:.2f}, {nz:.2f})':>22}")
        except Exception as e:
            print(f"  {ts}: FAIL {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
