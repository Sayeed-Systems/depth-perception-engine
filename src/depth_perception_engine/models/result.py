"""
Structured output models for the Depth Perception Engine's public pipeline.

These replace the loose dicts the original standalone scripts passed
around internally — every pipeline entry point (pipeline.api functions and
DepthPerceptionPipeline.process()) returns one of these, never a bare dict.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from depth_perception_engine.traversability.types import NavigationDecision, RegionStats


@dataclass(frozen=True, slots=True)
class BeamReading:
    """One vertical column-slice of the obstacle scan.

    Mirrors obstacles.ThreatAssessor.assess()'s per-beam dict, typed.
    """
    index: int
    x1: int
    x2: int
    distance_m: float
    status: str  # ThreatAssessor.CLEAR / CAUTION / BLOCKED / NO_DATA


@dataclass(frozen=True, slots=True)
class ObstacleAssessment:
    """Full-width obstacle scan: one BeamReading per beam, plus the safest one."""
    beams: List[BeamReading]
    safest_beam: Optional[BeamReading]


@dataclass(frozen=True, slots=True)
class TraversabilityResult:
    """Per-region grid classification, plus the derived global decision.

    `regions` is a name -> RegionStats grid (e.g. "TL".."BR" for the default
    3x3 grid), not a pixel-aligned boolean array — that is what the
    underlying algorithm (traversability.SceneInterpreter) actually
    produces, and this preserves it exactly rather than forcing it into a
    same-shape-as-image mask that would misrepresent the real granularity.
    """
    regions: Dict[str, RegionStats]
    decision: NavigationDecision


@dataclass(frozen=True, slots=True)
class DepthPerceptionResult:
    """The Depth Perception Engine's single top-level output shape.

    Returned by both DepthPerceptionPipeline.process() and
    pipeline.api.process_stereo_pair().
    """
    disparity_map: np.ndarray
    depth_map: np.ndarray
    traversability_mask: TraversabilityResult
    obstacles: ObstacleAssessment
    confidence: float
    processing_time_ms: float
