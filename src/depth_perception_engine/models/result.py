"""
Structured output models for the Depth Perception Engine's public pipeline.

These replace the loose dicts the original standalone scripts passed
around internally — every pipeline entry point (pipeline.api functions and
DepthPerceptionPipeline.process()) returns one of these, never a bare dict.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.traversability.types import NavigationDecision, RegionStats


@dataclass(frozen=True, slots=True)
class StereoObservation:
    """One stereo capture, as a single self-contained value instead of loose
    positional arguments.

    Optional convenience for callers that want to carry timestamps/frame
    identity alongside the images through their own code — not required by
    DepthPerceptionPipeline.process(), which still takes left_image/
    right_image directly. Use process_observation() to hand one of these to
    the pipeline instead.

    left_timestamp/right_timestamp are opaque caller-defined floats (e.g.
    seconds since epoch or a monotonic clock) — this library performs no
    unit conversion or synchronization logic on them; that is a caller
    concern (e.g. mp01_perception's stereo sync_slop check).

    calibration is reserved for future multi-rig/multi-camera use — NOT
    currently consumed by DepthPerceptionPipeline.process_observation(),
    which still always uses the calibration the pipeline itself was
    constructed with. None (the default) means "use the pipeline's own
    calibration", not "no calibration exists". A future caller carrying
    per-observation calibration (e.g. a multi-camera fusion producer
    selecting between rigs) has somewhere to put it without a public API
    change — see docs/LEVEL3_PUBLIC_API.md.
    """
    left_image: np.ndarray
    right_image: np.ndarray
    left_timestamp: Optional[float] = None
    right_timestamp: Optional[float] = None
    calibration: Optional[StereoCalibration] = None
    frame_id: Optional[str] = None


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

    Known naming wart, deliberately not fixed here: `traversability_mask`
    holds a TraversabilityResult, not a pixel mask — see
    docs/IMPLEMENTATION_STATUS.md. Left as-is because mp01_perception reads
    this field by name today; renaming it is a breaking change out of this
    recovery task's scope.
    """
    disparity_map: np.ndarray
    depth_map: np.ndarray
    traversability_mask: TraversabilityResult
    obstacles: ObstacleAssessment
    confidence: float
    processing_time_ms: float
    valid_disparity_mask: Optional[np.ndarray] = None
    valid_depth_mask: Optional[np.ndarray] = None
    timestamp: Optional[float] = None


@dataclass(frozen=True, slots=True)
class PipelineHealth:
    """A point-in-time snapshot returned by DepthPerceptionPipeline.health().

    Deliberately minimal — this reports the pipeline's own lifecycle state
    and the last frame's summary metrics, not a re-diagnosis of the scene
    (that's what DepthPerceptionResult.confidence/traversability/obstacles
    are for, per-frame).
    """
    is_closed: bool
    frames_processed: int
    last_confidence: Optional[float] = None
    last_processing_time_ms: Optional[float] = None
