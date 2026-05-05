"""Compatibility stub for removed legacy RTAB-Map descriptor localization.

`/api/slam/v3/localize` uses `slam_engines.superpoint.map_manager` instead.
This module stays importable so older debug helpers fail explicitly rather than
breaking application startup.
"""

from __future__ import annotations


class MapManager:
    def get_or_load(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError("RTAB-Map descriptor localization is not exposed")
