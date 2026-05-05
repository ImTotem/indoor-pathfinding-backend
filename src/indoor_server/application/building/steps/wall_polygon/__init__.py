"""Sprint 51 — Wall-fitting polygon pipeline (default OFF, observer only).

7-step CAD-style polygon assembly from RTABMap obstacle heatmap (non-LiDAR).
Public surface: facade `WallPolygonFromObstacleStep` + result dataclass.
Sub-step internals are exported for testing only.
"""
from __future__ import annotations

from indoor_server.application.building.steps.wall_polygon.assembly import (
    AssembledPolygon,
    PolygonAssemblyParams,
    PolygonAssemblyStep,
)
from indoor_server.application.building.steps.wall_polygon.components import (
    ComponentSplitParams,
    ComponentSplitStep,
    FragmentSet,
    WallFragment,
)
from indoor_server.application.building.steps.wall_polygon.density import (
    DensityRefined,
    DensityRefineParams,
    DensityRefineStep,
)
from indoor_server.application.building.steps.wall_polygon.facade import (
    WallPolygonFromObstacleStep,
    WallPolygonResult,
    WallPolygonStepParams,
)
from indoor_server.application.building.steps.wall_polygon.heatmap_boundary import (
    HeatmapBoundaryParams,
    HeatmapBoundaryResult,
    HeatmapBoundaryStep,
)
from indoor_server.application.building.steps.wall_polygon.hough_fallback import (
    HoughFallbackParams,
    HoughFallbackResult,
    HoughFallbackStep,
)
from indoor_server.application.building.steps.wall_polygon.interior_fill import (
    WallInteriorFillParams,
    WallInteriorFillResult,
    WallInteriorFillStep,
)
from indoor_server.application.building.steps.wall_polygon.line_fit import (
    LineFitParams,
    LineFitStep,
    LineSegment,
    LineSegmentSet,
)
from indoor_server.application.building.steps.wall_polygon.merge import (
    CollinearMergeStep,
    MergeParams,
    WallLine,
    WallLineSet,
)
from indoor_server.application.building.steps.wall_polygon.obstacle_source import (
    ObstacleHeatmap,
    ObstacleSourceStep,
    ObstacleSourceStepParams,
)
from indoor_server.application.building.steps.wall_polygon.skeleton_split import (
    SkeletonSplitParams,
    SkeletonSplitResult,
    SkeletonSplitStep,
)
from indoor_server.application.building.steps.wall_polygon.snap import (
    AngleSnapParams,
    AngleSnapStep,
    SnappedSegment,
    SnappedSegmentSet,
)
from indoor_server.application.building.steps.wall_polygon.validate import (
    ValidateParams,
    ValidateStep,
    WallPolygonValidation,
)

__all__ = [
    "AssembledPolygon",
    "AngleSnapParams",
    "AngleSnapStep",
    "CollinearMergeStep",
    "ComponentSplitParams",
    "ComponentSplitStep",
    "DensityRefineParams",
    "DensityRefineStep",
    "DensityRefined",
    "FragmentSet",
    "HoughFallbackParams",
    "HoughFallbackResult",
    "HoughFallbackStep",
    "HeatmapBoundaryParams",
    "HeatmapBoundaryResult",
    "HeatmapBoundaryStep",
    "WallInteriorFillParams",
    "WallInteriorFillResult",
    "WallInteriorFillStep",
    "LineFitParams",
    "LineFitStep",
    "LineSegment",
    "LineSegmentSet",
    "MergeParams",
    "ObstacleHeatmap",
    "ObstacleSourceStep",
    "ObstacleSourceStepParams",
    "PolygonAssemblyParams",
    "PolygonAssemblyStep",
    "SkeletonSplitParams",
    "SkeletonSplitResult",
    "SkeletonSplitStep",
    "SnappedSegment",
    "SnappedSegmentSet",
    "ValidateParams",
    "ValidateStep",
    "WallFragment",
    "WallLine",
    "WallLineSet",
    "WallPolygonFromObstacleStep",
    "WallPolygonResult",
    "WallPolygonStepParams",
    "WallPolygonValidation",
]
