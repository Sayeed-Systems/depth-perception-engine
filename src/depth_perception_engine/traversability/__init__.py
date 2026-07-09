"""Traversability layer: per-region grid classification and navigation decision."""

from depth_perception_engine.traversability.region_analyzer import RegionAnalyzer
from depth_perception_engine.traversability.scene_interpreter import (
    SceneInterpreter,
    SceneState,
)
from depth_perception_engine.traversability.types import (
    NavigationDecision,
    RegionClass,
    RegionStats,
    TextureClass,
)

__all__ = [
    "RegionAnalyzer",
    "SceneInterpreter",
    "SceneState",
    "NavigationDecision",
    "RegionClass",
    "RegionStats",
    "TextureClass",
]
