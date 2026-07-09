"""Structured output types returned by the pipeline."""

from depth_perception_engine.models.result import (
    BeamReading,
    DepthPerceptionResult,
    ObstacleAssessment,
    TraversabilityResult,
)

__all__ = [
    "BeamReading",
    "ObstacleAssessment",
    "TraversabilityResult",
    "DepthPerceptionResult",
]
