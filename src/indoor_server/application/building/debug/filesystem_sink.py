"""FilesystemDebugSink — 빌드 단계별 결과를 파일로 덤프."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from indoor_server.application.building.steps.wall_polygon import (
        WallPolygonResult,
    )

import numpy as np

from indoor_server.application.building.steps.floor_segmentation import KeyframeMasks
from indoor_server.application.building.steps.skeletonize import SkeletonGraph
from indoor_server.domain.building.models import (
    DepthAwareLayout,
    DepthMap,
    FloorLayout,
    FloorPolygon,
    GridOrigin,
    MapEdgeVO,
    MapNodeVO,
    WalkableGrid,
)

logger = logging.getLogger(__name__)


def _uniform_sample(items: list[KeyframeMasks], n: int) -> list[KeyframeMasks]:
    """균등 샘플링 n장: 첫/끝/3등분점."""
    if len(items) <= n:
        return items
    if n <= 1:
        return [items[0]]
    indices: list[int] = []
    for i in range(n):
        idx = int(round(i * (len(items) - 1) / (n - 1)))
        if idx not in indices:
            indices.append(idx)
    return [items[i] for i in sorted(set(indices))]


def _uniform_sample_seqs(seqs: list[int], n: int) -> list[int]:
    """seq 목록에서 균등 n개 샘플링."""
    if len(seqs) <= n:
        return seqs
    if n <= 1:
        return [seqs[0]]
    indices: list[int] = []
    for i in range(n):
        idx = int(round(i * (len(seqs) - 1) / (n - 1)))
        if idx not in indices:
            indices.append(idx)
    return [seqs[i] for i in sorted(set(indices))]


def _node_to_dict(node: MapNodeVO) -> dict[str, object]:
    return {
        "node_id": str(node.node_id),
        "scan_id": str(node.scan_id),
        "build_job_id": str(node.build_job_id),
        "type": node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
        "x": node.x,
        "y": node.y,
        "z": node.z,
        "label": node.label,
    }


def _edge_to_dict(edge: MapEdgeVO) -> dict[str, object]:
    return {
        "edge_id": str(edge.edge_id),
        "scan_id": str(edge.scan_id),
        "build_job_id": str(edge.build_job_id),
        "from": str(edge.from_node_id),
        "to": str(edge.to_node_id),
        "type": edge.edge_type.value if hasattr(edge.edge_type, "value") else str(edge.edge_type),
        "length_m": edge.length_m,
        "polyline": [list(pt) for pt in edge.polyline],
    }


class FilesystemDebugSink:
    """빌드 파이프라인 단계별 결과를 out_dir에 파일로 덤프."""

    def __init__(
        self,
        out_dir: Path,
        storage_root: Path,
        sample_count: int = 5,
    ) -> None:
        self._out_dir = out_dir
        self._storage_root = storage_root
        self._sample_count = sample_count
        out_dir.mkdir(parents=True, exist_ok=True)

        # finalize()에서 composite 생성 시 필요한 메타
        self._scan_id: str = ""
        self._job_id: str = ""

    def on_floor_segmentation(self, samples: list[KeyframeMasks]) -> None:
        """균등 샘플링 N장 → floor_masks/ 디렉터리에 PNG 저장."""
        import cv2

        if samples:
            self._scan_id = str(samples[0].scan_id)
        chosen = _uniform_sample(samples, self._sample_count)
        mask_dir = self._out_dir / "floor_masks"
        mask_dir.mkdir(exist_ok=True)

        for km in chosen:
            # 원본 jpg: storage_root / scans / scan_id / keyframes/{seq:06d}.jpg
            kf_name = f"keyframes/{km.seq:06d}.jpg"
            orig_path = self._storage_root / "scans" / str(km.scan_id) / kf_name
            mask_bgr = np.zeros((*km.floor_mask.shape, 3), dtype=np.uint8)
            mask_bgr[km.floor_mask] = (255, 255, 0)  # cyan(BGR)

            if orig_path.exists():
                bgr = cv2.imread(str(orig_path))
                if bgr is not None:
                    # 마스크를 원본 크기로 리사이즈
                    if bgr.shape[:2] != km.floor_mask.shape:
                        h, w = km.floor_mask.shape
                        bgr = cv2.resize(bgr, (w, h))
                    overlay: np.ndarray = cv2.addWeighted(bgr, 0.6, mask_bgr, 0.4, 0)
                    out_path = mask_dir / f"keyframe_{km.seq:06d}.png"
                    cv2.imwrite(str(out_path), overlay)
                    continue

            # 원본 없으면 마스크만
            out_path = mask_dir / f"keyframe_{km.seq:06d}.png"
            cv2.imwrite(str(out_path), mask_bgr)

    def on_back_projection(self, points: np.ndarray, z0: float) -> None:
        """back_projection.npz — 100k cap random subsample."""
        rng = np.random.default_rng(42)
        cap = 100_000
        if len(points) > cap:
            idx = rng.choice(len(points), cap, replace=False)
            points = points[idx]
        out = self._out_dir / "back_projection.npz"
        np.savez_compressed(str(out), points=points.astype(np.float32), z0=np.float64(z0))
        logger.debug("debug sink: back_projection.npz points=%d z0=%.3f", len(points), z0)

    def on_walkable_grid(self, grid: WalkableGrid) -> None:
        """walkable_grid.npz + origin 분리 키 저장."""
        out = self._out_dir / "walkable_grid.npz"
        np.savez_compressed(
            str(out),
            mask=grid.mask,
            observation_count=grid.observation_count,
            origin_x0=np.float64(grid.origin.x0),
            origin_y0=np.float64(grid.origin.y0),
            origin_z0=np.float64(grid.origin.z0),
            origin_cell_size=np.float64(grid.origin.cell_size),
            origin_w=np.int64(grid.origin.w),
            origin_h=np.int64(grid.origin.h),
        )
        logger.debug("debug sink: walkable_grid.npz cells=%d", int(grid.mask.sum()))

    def on_skeleton(self, skeleton_mask: np.ndarray, skel: SkeletonGraph) -> None:
        """skeleton.npz 저장."""
        out = self._out_dir / "skeleton.npz"
        np.savez_compressed(str(out), skeleton=skeleton_mask)
        logger.debug("debug sink: skeleton.npz pixels=%d", int(skeleton_mask.sum()))

    def on_node_placement(
        self,
        nodes: list[MapNodeVO],
        edges: list[MapEdgeVO],
        origin: GridOrigin,
    ) -> None:
        """nodes.json / edges.json 저장."""
        nodes_path = self._out_dir / "nodes.json"
        edges_path = self._out_dir / "edges.json"

        if nodes:
            # scan_id/build_job_id 메타 추출
            self._scan_id = str(nodes[0].scan_id)
            self._job_id = str(nodes[0].build_job_id)

        nodes_path.write_text(json.dumps([_node_to_dict(n) for n in nodes], ensure_ascii=False))
        edges_path.write_text(json.dumps([_edge_to_dict(e) for e in edges], ensure_ascii=False))
        logger.debug("debug sink: nodes.json n=%d edges.json n=%d", len(nodes), len(edges))

    def on_floor_polygons(self, polygons: list[FloorPolygon]) -> None:
        """
        per-frame 단순화 polygon 전량 → floor_polygons.json + PNG 샘플 8장.
        """
        import cv2

        from indoor_server.application.building.debug.floor_layout_render import (
            polygons_to_json,
            render_floor_polygon_overlay,
        )

        # 전량 JSON 저장
        json_path = self._out_dir / "floor_polygons.json"
        json_path.write_text(polygons_to_json(polygons), encoding="utf-8")
        logger.debug("debug sink: floor_polygons.json written polygons=%d", len(polygons))

        # seq별로 그룹화
        from collections import defaultdict
        seq_to_polys: dict[int, list[FloorPolygon]] = defaultdict(list)
        for fp in polygons:
            seq_to_polys[fp.seq].append(fp)

        # 균등 샘플링 (seq 단위)
        all_seqs = sorted(seq_to_polys.keys())
        sampled_seqs = _uniform_sample_seqs(all_seqs, self._sample_count)

        poly_dir = self._out_dir / "floor_polygons"
        poly_dir.mkdir(exist_ok=True)

        for seq in sampled_seqs:
            fps = seq_to_polys[seq]
            if not fps:
                continue

            fp0 = fps[0]
            h, w = fp0.image_h, fp0.image_w

            # 합성 floor_mask 복원 (vertices가 없으면 빈 mask)
            floor_mask = np.zeros((h, w), dtype=bool)

            # 원본 jpg는 on_floor_segmentation에서 캐시한 scan_id로 로드 시도
            jpg: np.ndarray | None = None
            if self._scan_id:
                orig_path = (
                    self._storage_root / "scans" / self._scan_id /
                    "keyframes" / f"{seq:06d}.jpg"
                )
                if orig_path.exists():
                    bgr = cv2.imread(str(orig_path))
                    if bgr is not None:
                        if bgr.shape[:2] != (h, w):
                            bgr = cv2.resize(bgr, (w, h))
                        jpg = bgr

            overlay = render_floor_polygon_overlay(jpg, floor_mask, fps, seq)
            out_path = poly_dir / f"keyframe_{seq:06d}.png"
            cv2.imwrite(str(out_path), overlay)

        logger.debug("debug sink: floor_polygons/ PNGs written seqs=%d", len(sampled_seqs))

    def on_floor_layout(
        self,
        layout: FloorLayout,
        trajectory_pts: list[tuple[float, float]],
    ) -> None:
        """
        FloorLayout → floor_layout.png + floor_layout.geojson.
        """
        import cv2

        from indoor_server.application.building.debug.floor_layout_render import (
            layout_to_geojson,
            render_floor_layout_png,
        )

        # GeoJSON
        geojson_path = self._out_dir / "floor_layout.geojson"
        geojson_path.write_text(layout_to_geojson(layout), encoding="utf-8")
        logger.debug(
            "debug sink: floor_layout.geojson written polygons=%d",
            layout.sub_polygon_count,
        )

        # PNG
        img = render_floor_layout_png(
            layout=layout,
            trajectory_xy=trajectory_pts,
            scan_id=self._scan_id,
            job_id=self._job_id,
        )
        png_path = self._out_dir / "floor_layout.png"
        cv2.imwrite(str(png_path), img)
        logger.info("debug sink: floor_layout.png written to %s", png_path)

    def on_floor_layout_raster(
        self,
        layout: FloorLayout,
        trajectory_pts: list[tuple[float, float]],
    ) -> None:
        """
        Fix 3: raster-fit FloorLayout → floor_layout_raster.png + floor_layout_raster.geojson.
        render_floor_layout_png 재사용.
        """
        import cv2

        from indoor_server.application.building.debug.floor_layout_render import (
            layout_to_geojson,
            render_floor_layout_png,
        )

        geojson_path = self._out_dir / "floor_layout_raster.geojson"
        geojson_path.write_text(layout_to_geojson(layout), encoding="utf-8")
        logger.debug(
            "debug sink: floor_layout_raster.geojson written polygons=%d",
            layout.sub_polygon_count,
        )

        img = render_floor_layout_png(
            layout=layout,
            trajectory_xy=trajectory_pts,
            scan_id=self._scan_id,
            job_id=self._job_id,
        )
        png_path = self._out_dir / "floor_layout_raster.png"
        cv2.imwrite(str(png_path), img)
        logger.info("debug sink: floor_layout_raster.png written to %s", png_path)

    def on_depth_maps(self, depth_maps: list[DepthMap]) -> None:
        """균등 8장 샘플 → depth_maps/keyframe_*.png + depth_maps.json."""
        import cv2

        if not depth_maps:
            return

        depth_dir = self._out_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)

        # 균등 샘플링 (최대 8장)
        n_sample = 8
        if len(depth_maps) <= n_sample:
            sampled = depth_maps
        else:
            indices: list[int] = []
            for i in range(n_sample):
                idx = int(round(i * (len(depth_maps) - 1) / (n_sample - 1)))
                if idx not in indices:
                    indices.append(idx)
            sampled = [depth_maps[i] for i in sorted(set(indices))]

        from indoor_server.application.building.debug.depth_render import render_depth_colormap_png

        meta_list: list[dict[str, object]] = []
        for dm in sampled:
            # 원본 JPG 로드 시도
            orig_path: Path | None = None
            if self._scan_id:
                p = (
                    self._storage_root
                    / "scans"
                    / self._scan_id
                    / "keyframes"
                    / f"{dm.seq:06d}.jpg"
                )
                if p.exists():
                    orig_path = p

            orig_bgr: np.ndarray | None = None
            if orig_path is not None:
                orig_bgr = cv2.imread(str(orig_path))

            combined = render_depth_colormap_png(dm, orig_bgr)
            out_path = depth_dir / f"keyframe_{dm.seq:06d}.png"
            cv2.imwrite(str(out_path), combined)

            meta_list.append(
                {
                    "seq": dm.seq,
                    "scale": dm.scale,
                    "valid_pixel_count": dm.valid_pixel_count,
                }
            )

        import json

        json_path = self._out_dir / "depth_maps.json"
        json_path.write_text(json.dumps(meta_list, ensure_ascii=False))
        logger.info("debug sink: depth_maps/ %d PNGs written", len(sampled))

    def on_depth_aware_layout(
        self,
        layout: DepthAwareLayout,
        trajectory_pts: list[tuple[float, float]],
    ) -> None:
        """depth-aware layout → depth_aware_layout.png."""
        import cv2

        from indoor_server.application.building.debug.depth_render import (
            render_depth_aware_layout_png,
        )

        img = render_depth_aware_layout_png(
            layout=layout,
            trajectory_xy=trajectory_pts,
            scan_id=self._scan_id,
            job_id=self._job_id,
        )
        png_path = self._out_dir / "depth_aware_layout.png"
        cv2.imwrite(str(png_path), img)
        logger.info("debug sink: depth_aware_layout.png written to %s", png_path)

    def on_wall_polygon(
        self,
        result: WallPolygonResult,
        *,
        scan_id: str,
        floor_polygon_geojson: dict[str, object] | None,
    ) -> None:
        """Sprint 51 — render 7-PNG evidence + json report into wall_polygon/."""
        from indoor_server.application.building.steps.wall_polygon.evidence import (
            render_wall_polygon_evidence,
        )

        out_dir = self._out_dir / "wall_polygon"
        try:
            render_wall_polygon_evidence(
                result,
                output_dir=out_dir,
                scan_id=scan_id,
                floor_polygon_geojson=floor_polygon_geojson,
            )
        except Exception as e:
            logger.warning("debug sink on_wall_polygon failed: %s", e)

    def finalize(self) -> None:
        """walkable_grid.npz + skeleton.npz + nodes/edges.json → composite.png."""
        import cv2

        from indoor_server.application.building.debug.composite import render_composite_panel

        grid_path = self._out_dir / "walkable_grid.npz"
        skel_path = self._out_dir / "skeleton.npz"
        nodes_path = self._out_dir / "nodes.json"
        edges_path = self._out_dir / "edges.json"

        if not grid_path.exists():
            logger.warning("debug sink finalize: walkable_grid.npz not found, skip composite")
            return

        grid_npz = np.load(str(grid_path))
        walkable_mask: np.ndarray = grid_npz["mask"].astype(bool)
        observation_count: np.ndarray = grid_npz["observation_count"]
        origin = GridOrigin(
            x0=float(grid_npz["origin_x0"]),
            y0=float(grid_npz["origin_y0"]),
            z0=float(grid_npz["origin_z0"]),
            cell_size=float(grid_npz["origin_cell_size"]),
            w=int(grid_npz["origin_w"]),
            h=int(grid_npz["origin_h"]),
        )

        skeleton_mask: np.ndarray | None = None
        if skel_path.exists():
            skel_npz = np.load(str(skel_path))
            skeleton_mask = skel_npz["skeleton"].astype(bool)

        nodes: list[MapNodeVO] | None = None
        edges: list[MapEdgeVO] | None = None
        if nodes_path.exists() and edges_path.exists():
            nodes_raw = json.loads(nodes_path.read_text())
            edges_raw = json.loads(edges_path.read_text())
            nodes = _dicts_to_nodes(nodes_raw)
            edges = _dicts_to_edges(edges_raw)

        composite = render_composite_panel(
            walkable_mask=walkable_mask,
            observation_count=observation_count,
            skeleton_mask=skeleton_mask,
            nodes=nodes,
            edges=edges,
            origin=origin,
            scan_id=self._scan_id,
            job_id=self._job_id,
        )

        out_png = self._out_dir / "composite.png"
        cv2.imwrite(str(out_png), composite)
        logger.info("debug sink: composite.png written to %s", out_png)


def _dicts_to_nodes(raw: list[dict[str, object]]) -> list[MapNodeVO]:
    from indoor_server.domain.building.enums import NodeType

    result: list[MapNodeVO] = []
    for d in raw:
        node_type = NodeType(str(d.get("type", "corridor")))
        _nil = "00000000-0000-0000-0000-000000000000"
        result.append(
            MapNodeVO(
                node_id=UUID(str(d["node_id"])),
                scan_id=UUID(str(d.get("scan_id", _nil))),
                build_job_id=UUID(str(d.get("build_job_id", _nil))),
                node_type=node_type,
                x=float(str(d["x"])),
                y=float(str(d["y"])),
                z=float(str(d["z"])),
                label=str(d["label"]) if d.get("label") is not None else None,
            )
        )
    return result


def _dicts_to_edges(raw: list[dict[str, object]]) -> list[MapEdgeVO]:
    from indoor_server.domain.building.enums import EdgeType

    result: list[MapEdgeVO] = []
    for d in raw:
        edge_type = EdgeType(str(d.get("type", "skeleton")))
        raw_pl_obj = d.get("polyline")
        polyline: list[tuple[float, float, float]] = []
        if isinstance(raw_pl_obj, list):
            for pt in raw_pl_obj:
                if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                    polyline.append((float(pt[0]), float(pt[1]), float(pt[2])))
        _nil = "00000000-0000-0000-0000-000000000000"
        result.append(
            MapEdgeVO(
                edge_id=UUID(str(d["edge_id"])),
                scan_id=UUID(str(d.get("scan_id", _nil))),
                build_job_id=UUID(str(d.get("build_job_id", _nil))),
                from_node_id=UUID(str(d["from"])),
                to_node_id=UUID(str(d["to"])),
                edge_type=edge_type,
                polyline=polyline,
                length_m=float(str(d.get("length_m", 0.0))),
            )
        )
    return result
