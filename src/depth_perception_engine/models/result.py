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
from depth_perception_engine.geometry.types import FreeSpaceRays, GeometryMetrics, ObstacleCloud, PointCloud
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

    geometry (Level 3, Phase E3): camera-optical-frame 3D point cloud —
    None unless PipelineConfig.enable_geometry is True. When present,
    geometry.frame_id == frames.FrameId.CAMERA_OPTICAL_LEFT always.
    geometry.valid_mask marks which pixels have a real (non-NaN) point,
    mirroring valid_depth_mask's role one level up. Added at the end,
    defaulted to None, so this field is purely additive — every existing
    caller (including mp01_perception, and every pre-E3 test) is
    byte-for-byte unaffected. See docs/DATA_CONTRACTS.md for the full
    contract and docs/LEVEL3_ARCHITECTURE.md for why this is PointCloud
    directly, not the undefined GeometryResult composite
    docs/E2_IMPLEMENTATION_PLAN.md had sketched.

    geometry_body (Level 3, Phase E4): body-frame 3D point cloud — None
    unless *both* PipelineConfig.enable_geometry is True *and* a
    body_T_camera_left extrinsic was supplied to
    DepthPerceptionPipeline's constructor (there is no camera-frame
    `geometry` to transform otherwise, and this library never assumes a
    missing extrinsic means identity — see
    calibration.contracts.RigCalibration's docstring). When present,
    geometry_body.frame_id == frames.FrameId.BODY, and it is the exact
    same points as `geometry` with frames.RigidTransform's own frozen
    convention applied (rotation @ p + translation) — same valid_mask
    (copied, not recomputed), same timestamp. `geometry` itself is never
    replaced or mutated by this — both coexist so a caller that only
    trusts the E2/E3-verified camera frame is unaffected by whether body
    extrinsics happen to be configured. Reuses geometry.PointCloud
    directly, the same "don't invent a new type" reasoning as `geometry`
    above — see docs/LEVEL3_ARCHITECTURE.md's E4 update.

    obstacle_cloud / free_space_rays / geometry_metrics (Level 3, Phase
    E5): structured spatial evidence derived from `geometry_body` — None
    unless `geometry_body` itself is present (PipelineConfig.enable_geometry
    True and a body_T_camera_left extrinsic supplied) and the
    corresponding PipelineConfig.enable_obstacle_geometry /
    enable_free_space_rays flag is True. `obstacle_cloud`/`free_space_rays`
    both use FrameId.BODY, same as `geometry_body`. Not renamed/reused
    from `obstacles` (the existing per-beam ObstacleAssessment field,
    Level 0-2) — that field is a different, older, 2D-beam-scan concept
    this pass does not touch or replace; `obstacle_cloud` is a distinct,
    3D, geometric concept. `geometry_metrics` is populated whenever
    `geometry_body` exists, independent of the other two flags (cheap
    scalar aggregation, no separate gate — see
    geometry.build_geometry_metrics's docstring for why its
    `min_obstacle_distance_m`/`mean_free_space_m` fields are simply None
    when the corresponding cloud/rays were not computed this call).
    Reuses geometry.ObstacleCloud/FreeSpaceRays/GeometryMetrics directly —
    same "don't invent a new type" reasoning as `geometry`/`geometry_body`
    above. See docs/LEVEL3_ARCHITECTURE.md's E5 update and
    docs/DATA_CONTRACTS.md for the full contract, including the frozen
    field-set mismatches (no timestamp on ObstacleCloud/FreeSpaceRays; no
    max-distance/count fields on GeometryMetrics) reported rather than
    silently resolved by adding fields to those frozen types.
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
    geometry: Optional[PointCloud] = None
    geometry_body: Optional[PointCloud] = None
    obstacle_cloud: Optional[ObstacleCloud] = None
    free_space_rays: Optional[FreeSpaceRays] = None
    geometry_metrics: Optional[GeometryMetrics] = None


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
