"""
BuildPipeline 디버그 시각화 CLI.

사용법:
  # 합성 walkable grid로 검증
  python scripts/visualize_build.py synthetic --shape t
  python scripts/visualize_build.py synthetic --shape plus --out /tmp/composite.png

  # 워커가 떨군 디렉터리 → composite 재렌더
  python scripts/visualize_build.py dump-dir var/debug/{scan_id}/{build_job_id}/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

# server/src 를 sys.path에 추가 (scripts/ 에서 직접 실행 시)
_SERVER_ROOT = Path(__file__).resolve().parent.parent
_SRC = _SERVER_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── 합성 grid 정의 ────────────────────────────────────────────────────────────

def build_synthetic_grid(shape: str) -> tuple[np.ndarray, np.ndarray, object]:
    """
    합성 WalkableGrid 생성.
    반환: (walkable_mask, observation_count, GridOrigin)
    """
    from indoor_server.domain.building.models import GridOrigin

    h, w = 80, 80
    mask = np.zeros((h, w), dtype=bool)

    if shape == "t":
        mask[30:50, 0:80] = True   # 가로
        mask[0:50, 30:50] = True   # 세로
    elif shape == "plus":
        mask[30:50, 0:80] = True   # 가로
        mask[0:80, 30:50] = True   # 세로
    elif shape == "l":
        mask[30:50, 0:50] = True   # 가로
        mask[0:50, 30:50] = True   # 세로
    else:
        raise ValueError(f"Unknown shape: {shape!r}")

    obs = (mask.astype(np.uint16)) * 5
    origin = GridOrigin(x0=0.0, y0=0.0, z0=0.0, cell_size=0.10, w=w, h=h)
    return mask, obs, origin


def run_synthetic(shape: str, out: Path) -> int:
    """synthetic 모드 실행. 0=OK, 1=error."""
    import cv2

    try:
        from indoor_server.application.building.debug.composite import render_composite_panel
        from indoor_server.application.building.steps.node_placement import NodePlacementStep
        from indoor_server.application.building.steps.skeletonize import SkeletonizeStep
        from indoor_server.domain.building.models import WalkableGrid
    except ImportError as e:
        logger.error("import failed: %s", e)
        return 1

    try:
        walkable_mask, obs, origin = build_synthetic_grid(shape)

        # WalkableGrid 구성
        grid = WalkableGrid(origin=origin, mask=walkable_mask, observation_count=obs)

        # Skeleton
        step = SkeletonizeStep()
        from skimage.morphology import medial_axis as _medial_axis

        result = _medial_axis(grid.mask, return_distance=True)
        skeleton_bool: np.ndarray = result[0]
        graph = step._build_graph(skeleton_bool)  # type: ignore[attr-defined]
        nodes_topo, edges_topo = step._extract_topology(graph)  # type: ignore[attr-defined]
        from indoor_server.application.building.steps.skeletonize import SkeletonGraph
        skel = SkeletonGraph(
            nodes=nodes_topo,
            edges=edges_topo,
            skeleton_pixel_count=int(skeleton_bool.sum()),
        )

        # NodePlacement
        scan_id = uuid4()
        job_id = uuid4()
        placer = NodePlacementStep(scan_id=scan_id, build_job_id=job_id)
        nodes, edges = placer.run(skel, origin)

        logger.info(
            "synthetic shape=%s walkable_cells=%d skeleton_px=%d nodes=%d edges=%d",
            shape, int(walkable_mask.sum()), skel.skeleton_pixel_count, len(nodes), len(edges),
        )

        # Composite 렌더
        composite = render_composite_panel(
            walkable_mask=walkable_mask,
            observation_count=obs,
            skeleton_mask=skeleton_bool,
            nodes=nodes,
            edges=edges,
            origin=origin,
            scan_id=str(scan_id),
            job_id=str(job_id),
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), composite)
        logger.info("composite written to %s", out)
        return 0

    except Exception as e:
        logger.error("render failed: %s", e, exc_info=True)
        return 1


def run_floor_polygons_synthetic(shape: str, out: Path) -> int:
    """
    합성 floor mask → FloorPolygonSimplifyStep → per-frame PNG 1장 생성.
    shape: 'rect' | 'l' | 'noisy'
    """
    import struct

    import cv2

    try:
        from indoor_server.application.building.debug.floor_layout_render import (
            render_floor_polygon_overlay,
        )
        from indoor_server.application.building.steps.floor_polygon import FloorPolygonSimplifyStep
        from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks
    except ImportError as e:
        logger.error("import failed: %s", e)
        return 1

    try:
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=bool)
        if shape == "rect":
            mask[30:170, 30:170] = True
        elif shape == "l":
            mask[30:170, 30:80] = True
            mask[30:80, 30:170] = True
        else:  # noisy
            rng = np.random.default_rng(42)
            mask[30:170, 30:170] = True
            noise = rng.random((140, 140)) > 0.85
            mask[30:170, 30:170] &= ~noise

        identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        pose_bytes = struct.pack("<16f", *identity)

        km = KeyframeMasks(
            scan_id=__import__("uuid").uuid4(),
            seq=0,
            floor_mask=mask,
            stair_mask=np.zeros((h, w), dtype=bool),
            tx=0.0, ty=0.0, tz=0.0,
            pose_matrix=pose_bytes,
        )
        step = FloorPolygonSimplifyStep(min_area_px=50)
        polys = step.run([km])

        logger.info("floor-polygons synthetic shape=%s contours=%d", shape, len(polys))

        img = render_floor_polygon_overlay(None, mask, polys, seq=0)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        logger.info("floor-polygons written to %s", out)
        return 0

    except Exception as e:
        logger.error("render failed: %s", e, exc_info=True)
        return 1


def run_floor_polygons_dump(dump_dir: Path, out: Path | None) -> int:
    """dump-dir에서 floor_polygons.json + PNG 재렌더."""
    import cv2

    json_path = dump_dir / "floor_polygons.json"
    if not json_path.exists():
        logger.error("floor_polygons.json not found in %s", dump_dir)
        return 2

    try:
        import json as _json

        from indoor_server.application.building.debug.floor_layout_render import (
            render_floor_polygon_overlay,
        )
        from indoor_server.domain.building.models import FloorPolygon

        data = _json.loads(json_path.read_text(encoding="utf-8"))
        # 빈 pose_bytes — vertex 없이 overlay만 검증
        import struct
        identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        pose_bytes = struct.pack("<16f", *identity)

        polys_by_seq: dict[int, list[FloorPolygon]] = {}
        for item in data:
            fp = FloorPolygon(
                seq=item["seq"],
                polygon_index=item["polygon_index"],
                vertices_px=[tuple(v) for v in item["vertices_px"]],  # type: ignore[misc]
                image_w=item["image_w"],
                image_h=item["image_h"],
                area_px=item["area_px"],
                pose_matrix=pose_bytes,
                tx=0.0, ty=0.0, tz=0.0,
            )
            polys_by_seq.setdefault(fp.seq, []).append(fp)

        if not polys_by_seq:
            logger.error("no polygons found in %s", json_path)
            return 1

        # 첫 seq 렌더
        first_seq = sorted(polys_by_seq.keys())[0]
        fps = polys_by_seq[first_seq]
        fp0 = fps[0]
        mask = np.zeros((fp0.image_h, fp0.image_w), dtype=bool)
        img = render_floor_polygon_overlay(None, mask, fps, seq=first_seq)

        target = out or (dump_dir / f"floor_polygon_seq{first_seq:06d}.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), img)
        logger.info("floor-polygons dump-dir written to %s", target)
        return 0

    except Exception as e:
        logger.error("render failed: %s", e, exc_info=True)
        return 1


def run_floor_layout_synthetic(out: Path) -> int:
    """합성 polygon 3개 → FloorLayoutStep → top-down PNG."""

    import cv2

    try:
        from indoor_server.application.building.debug.floor_layout_render import (
            render_floor_layout_png,
        )
        from indoor_server.domain.building.models import FloorLayout
    except ImportError as e:
        logger.error("import failed: %s", e)
        return 1

    try:
        # 합성 polygon — shapely 직접 생성, FloorLayoutStep 우회
        from shapely.geometry import MultiPolygon
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union

        p1 = ShapelyPolygon([(0, 0), (5, 0), (5, 10), (0, 10)])
        p2 = ShapelyPolygon([(4, 4), (12, 4), (12, 10), (4, 10)])  # 겹침
        p3 = ShapelyPolygon([(20, 0), (25, 0), (25, 6), (20, 6)])  # 분리

        union = unary_union([p1, p2, p3])
        if not hasattr(union, "geoms"):
            union = MultiPolygon([union]) if not union.is_empty else MultiPolygon()

        layout = FloorLayout(
            multipolygon=union,
            z0=0.0,
            sub_polygon_count=len(list(union.geoms)),
            total_area_m2=float(union.area),
        )
        trajectory = [(float(i) * 0.5, float(i) * 0.5) for i in range(30)]

        img = render_floor_layout_png(layout=layout, trajectory_xy=trajectory)

        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        logger.info("floor-layout synthetic written to %s", out)
        return 0

    except Exception as e:
        logger.error("render failed: %s", e, exc_info=True)
        return 1


def run_floor_layout_dump(dump_dir: Path, out: Path | None) -> int:
    """dump-dir에서 floor_layout.geojson → PNG 재렌더."""
    import cv2

    geojson_path = dump_dir / "floor_layout.geojson"
    if not geojson_path.exists():
        logger.error("floor_layout.geojson not found in %s", dump_dir)
        return 2

    try:
        import json as _json

        from shapely.geometry import shape

        from indoor_server.application.building.debug.floor_layout_render import (
            render_floor_layout_png,
        )
        from indoor_server.domain.building.models import FloorLayout

        geom_dict = _json.loads(geojson_path.read_text(encoding="utf-8"))
        geom = shape(geom_dict)

        count = len(list(geom.geoms)) if hasattr(geom, "geoms") else (1 if not geom.is_empty else 0)
        layout = FloorLayout(
            multipolygon=geom,
            z0=0.0,
            sub_polygon_count=count,
            total_area_m2=float(geom.area),
        )

        img = render_floor_layout_png(layout=layout, trajectory_xy=[])
        target = out or (dump_dir / "floor_layout_rerender.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), img)
        logger.info("floor-layout dump-dir written to %s", target)
        return 0

    except Exception as e:
        logger.error("render failed: %s", e, exc_info=True)
        return 1


def run_dump_dir(dump_dir: Path, out: Path, panel: str) -> int:
    """dump-dir 모드 실행. 0=OK, 1=error, 2=input not found."""
    import cv2

    grid_path = dump_dir / "walkable_grid.npz"
    if not grid_path.exists():
        logger.error("walkable_grid.npz not found in %s", dump_dir)
        return 2

    try:
        from indoor_server.application.building.debug.composite import render_composite_panel
        from indoor_server.application.building.debug.filesystem_sink import (
            _dicts_to_edges,
            _dicts_to_nodes,
        )
        from indoor_server.domain.building.models import GridOrigin

        grid_npz = np.load(str(grid_path))
        walkable_mask: np.ndarray = grid_npz["mask"].astype(bool)
        obs: np.ndarray = grid_npz["observation_count"]
        origin = GridOrigin(
            x0=float(grid_npz["origin_x0"]),
            y0=float(grid_npz["origin_y0"]),
            z0=float(grid_npz["origin_z0"]),
            cell_size=float(grid_npz["origin_cell_size"]),
            w=int(grid_npz["origin_w"]),
            h=int(grid_npz["origin_h"]),
        )

        skel_path = dump_dir / "skeleton.npz"
        skeleton_mask: np.ndarray | None = None
        if skel_path.exists():
            skel_npz = np.load(str(skel_path))
            skeleton_mask = skel_npz["skeleton"].astype(bool)

        nodes_path = dump_dir / "nodes.json"
        edges_path = dump_dir / "edges.json"
        nodes = None
        edges = None
        if nodes_path.exists() and edges_path.exists():
            nodes = _dicts_to_nodes(json.loads(nodes_path.read_text()))
            edges = _dicts_to_edges(json.loads(edges_path.read_text()))

        if panel != "all":
            logger.warning("--panel=%s not fully implemented; rendering 'all'", panel)

        composite = render_composite_panel(
            walkable_mask=walkable_mask,
            observation_count=obs,
            skeleton_mask=skeleton_mask,
            nodes=nodes,
            edges=edges,
            origin=origin,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), composite)
        logger.info("composite written to %s", out)
        return 0

    except Exception as e:
        logger.error("render failed: %s", e, exc_info=True)
        return 1


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visualize_build",
        description="BuildPipeline 디버그 시각화 CLI",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_syn = sub.add_parser("synthetic", help="합성 walkable grid로 skeleton/node 검증")
    p_syn.add_argument(
        "--shape", choices=["t", "plus", "l"], default="t",
        help="합성 grid 모양 (기본: t)",
    )
    p_syn.add_argument(
        "--out", type=Path, default=None,
        help="출력 PNG 경로. 기본: var/debug/synthetic/{shape}/composite.png",
    )

    p_dump = sub.add_parser("dump-dir", help="워커가 떨군 디렉터리 → composite 재렌더")
    p_dump.add_argument(
        "dump_dir", type=Path,
        help="var/debug/{scan_id}/{build_job_id}/",
    )
    p_dump.add_argument(
        "--out", type=Path, default=None,
        help="출력 PNG 경로. 기본: <dump_dir>/composite.png",
    )
    p_dump.add_argument(
        "--panel",
        choices=["all", "mask", "skeleton", "graph", "obs"],
        default="all",
        help="렌더할 패널 (현재 'all'만 완전 구현)",
    )

    p_fp = sub.add_parser(
        "floor-polygons", help="per-frame floor polygon overlay 재렌더 / 합성"
    )
    p_fp.add_argument(
        "dump_dir", type=Path, nargs="?",
        help="dump 디렉터리 (floor_polygons.json 위치)",
    )
    p_fp.add_argument("--synthetic", action="store_true", help="합성 mask로 검증")
    p_fp.add_argument("--shape", choices=["rect", "l", "noisy"], default="rect")
    p_fp.add_argument("--out", type=Path, default=None)
    p_fp.add_argument("--samples", type=int, default=8)

    p_fl = sub.add_parser("floor-layout", help="multi-frame floor layout 재렌더 / 합성")
    p_fl.add_argument(
        "dump_dir", type=Path, nargs="?",
        help="dump 디렉터리 (floor_layout.geojson 위치)",
    )
    p_fl.add_argument("--synthetic", action="store_true", help="합성 polygon으로 검증")
    p_fl.add_argument("--out", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "synthetic":
        out: Path = args.out or (
            _SERVER_ROOT / "var" / "debug" / "synthetic" / args.shape / "composite.png"
        )
        sys.exit(run_synthetic(args.shape, out))

    elif args.mode == "dump-dir":
        dump_dir: Path = args.dump_dir
        out = args.out or (dump_dir / "composite.png")
        sys.exit(run_dump_dir(dump_dir, out, args.panel))

    elif args.mode == "floor-polygons":
        if args.synthetic:
            out = args.out or (
                _SERVER_ROOT / "var" / "debug" / "synthetic" / "floor_polygon.png"
            )
            sys.exit(run_floor_polygons_synthetic(args.shape, out))
        elif args.dump_dir:
            sys.exit(run_floor_polygons_dump(args.dump_dir, args.out))
        else:
            logger.error("floor-polygons: --synthetic 또는 dump_dir 중 하나 필요")
            sys.exit(2)

    elif args.mode == "floor-layout":
        if args.synthetic:
            out = args.out or (
                _SERVER_ROOT / "var" / "debug" / "synthetic" / "floor_layout.png"
            )
            sys.exit(run_floor_layout_synthetic(out))
        elif args.dump_dir:
            sys.exit(run_floor_layout_dump(args.dump_dir, args.out))
        else:
            logger.error("floor-layout: --synthetic 또는 dump_dir 중 하나 필요")
            sys.exit(2)


if __name__ == "__main__":
    main()
