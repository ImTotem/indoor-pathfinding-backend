"""BuildDebugSink Protocol + NoopDebugSink."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

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

if TYPE_CHECKING:
    from indoor_server.application.building.steps.wall_polygon import (
        WallPolygonResult,
    )


@runtime_checkable
class BuildDebugSink(Protocol):
    """빌드 파이프라인 각 단계 결과를 수신하는 sink Protocol."""

    def on_floor_segmentation(self, samples: list[KeyframeMasks]) -> None:
        """floor segmentation 완료 후 샘플 N장 수신."""
        ...

    def on_back_projection(self, points: np.ndarray, z0: float) -> None:
        """back projection point cloud 수신. 생략 가능(on_walkable_grid로 대체)."""
        ...

    def on_walkable_grid(self, grid: WalkableGrid) -> None:
        """walkable grid 완성 후 수신."""
        ...

    def on_skeleton(self, skeleton_mask: np.ndarray, skel: SkeletonGraph) -> None:
        """skeletonize 완료 후 bool mask + SkeletonGraph 수신."""
        ...

    def on_node_placement(
        self,
        nodes: list[MapNodeVO],
        edges: list[MapEdgeVO],
        origin: GridOrigin,
    ) -> None:
        """노드/엣지 배치 완료 후 수신."""
        ...

    def on_floor_polygons(self, polygons: list[FloorPolygon]) -> None:
        """per-frame 단순화 polygon 전체 목록 수신.
        구현체는 N장 균등 샘플링 + 원본 jpg overlay PNG 저장.
        """
        ...

    def on_floor_layout(
        self,
        layout: FloorLayout,
        trajectory_pts: list[tuple[float, float]],
    ) -> None:
        """multi-frame union 결과 + 카메라 trajectory 수신.
        구현체는 top-down PNG + GeoJSON dump.
        """
        ...

    def on_floor_layout_raster(
        self,
        layout: FloorLayout,
        trajectory_pts: list[tuple[float, float]],
    ) -> None:
        """Fix 3: raster-fit FloorLayout (walkable_grid 기반, spike 없음).
        구현체는 floor_layout_raster.png + floor_layout_raster.geojson dump.
        """
        ...

    def on_depth_maps(self, depth_maps: list[DepthMap]) -> None:
        """per-frame depth map 수신 — 균등 샘플링 PNG 저장."""
        ...

    def on_depth_aware_layout(
        self,
        layout: DepthAwareLayout,
        trajectory_pts: list[tuple[float, float]],
    ) -> None:
        """depth-aware walkable grid 기반 layout PNG."""
        ...

    def on_wall_polygon(
        self,
        result: WallPolygonResult,
        *,
        scan_id: str,
        floor_polygon_geojson: dict[str, object] | None,
    ) -> None:
        """Sprint 51: 7장 PNG + json report dump (wall_polygon observer)."""
        ...

    def finalize(self) -> None:
        """모든 단계 완료 후 composite.png 생성 등 마무리."""
        ...


class NoopDebugSink:
    """모든 hook이 no-op인 디폴트 sink. 오버헤드 0."""

    def on_floor_segmentation(self, samples: list[KeyframeMasks]) -> None:
        pass

    def on_back_projection(self, points: np.ndarray, z0: float) -> None:
        pass

    def on_walkable_grid(self, grid: WalkableGrid) -> None:
        pass

    def on_skeleton(self, skeleton_mask: np.ndarray, skel: SkeletonGraph) -> None:
        pass

    def on_node_placement(
        self,
        nodes: list[MapNodeVO],
        edges: list[MapEdgeVO],
        origin: GridOrigin,
    ) -> None:
        pass

    def on_floor_polygons(self, polygons: list[FloorPolygon]) -> None:
        pass

    def on_floor_layout(
        self,
        layout: FloorLayout,
        trajectory_pts: list[tuple[float, float]],
    ) -> None:
        pass

    def on_floor_layout_raster(
        self,
        layout: FloorLayout,
        trajectory_pts: list[tuple[float, float]],
    ) -> None:
        pass

    def on_depth_maps(self, depth_maps: list[DepthMap]) -> None:
        pass

    def on_depth_aware_layout(
        self,
        layout: DepthAwareLayout,
        trajectory_pts: list[tuple[float, float]],
    ) -> None:
        pass

    def on_wall_polygon(
        self,
        result: WallPolygonResult,
        *,
        scan_id: str,
        floor_polygon_geojson: dict[str, object] | None,
    ) -> None:
        pass

    def finalize(self) -> None:
        pass
