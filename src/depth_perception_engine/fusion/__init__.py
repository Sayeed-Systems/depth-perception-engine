"""Combines per-stage outputs into one DepthPerceptionResult."""

from depth_perception_engine.fusion.result_builder import (
    aggregate_confidence,
    build_result,
    to_obstacle_assessment,
)

__all__ = ["aggregate_confidence", "build_result", "to_obstacle_assessment"]
