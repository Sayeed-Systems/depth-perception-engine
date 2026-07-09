"""
Public pipeline entry points.

DepthPerceptionPipeline (stateful, recommended for repeated frames) and the
individual stateless functions (compute_disparity, estimate_depth,
classify_traversability, detect_obstacles, process_stereo_pair) — see each
module's docstring for when to use which.
"""

from depth_perception_engine.pipeline.api import (
    classify_traversability,
    compute_disparity,
    detect_obstacles,
    estimate_depth,
    process_stereo_pair,
)
from depth_perception_engine.pipeline.pipeline import DepthPerceptionPipeline

__all__ = [
    "DepthPerceptionPipeline",
    "process_stereo_pair",
    "compute_disparity",
    "estimate_depth",
    "classify_traversability",
    "detect_obstacles",
]
