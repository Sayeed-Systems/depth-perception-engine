"""
Structured output models for the Depth Perception Engine's public pipeline.

These replace the loose dicts the original standalone scripts passed
around internally — every pipeline entry point (pipeline.api functions and
DepthPerceptionPipeline.process()) returns one of these, never a bare dict.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.geometry.boundary import BoundaryEvidence
from depth_perception_engine.geometry.opening import OpeningEvidence
from depth_perception_engine.geometry.provider import GeometryFrame
from depth_perception_engine.geometry.surface import SurfaceEvidence
from depth_perception_engine.geometry.types import FreeSpaceRays, GeometryMetrics, ObstacleCloud, PointCloud
from depth_perception_engine.temporal.types import (
    MotionAwareReliability,
    MotionHint,
    TemporalConsistency,
    TemporalPersistence,
    TemporalStabilization,
)
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

    motion_hint: an optional temporal.MotionHint (a short-duration
    rotational motion measurement) a caller may attach to this specific
    stereo capture. Reserved at Level 4, Phase E1 (not yet consumed then);
    as of Phase E2, DepthPerceptionPipeline.process_observation() forwards
    this straight through to process()'s own motion_hint parameter, which
    associates it with the frame's temporal.TemporalRecord if
    PipelineConfig.enable_temporal is True — pure bookkeeping association,
    still no integration/alignment/algorithm reads its contents (see
    docs/E2_TEMPORAL_HISTORY_PLAN.md). None (the default) means "no motion
    hint was supplied for this frame" — legal and unremarkable: a missing
    motion hint never blocks temporal-history admission or Level 3
    perception (see docs/LEVEL4_ARCHITECTURE.md's degradation semantics).

    motion_hints (Level 4, Phase E5): an optional, bounded SEQUENCE of
    temporal.MotionHint samples spanning the interval leading up to this
    capture — distinct from the singular `motion_hint` field above, which
    remains for TemporalRecord association only (E1-E4, untouched).
    Forwarded by process_observation() into process()'s own motion_hints
    parameter, which DepthPerceptionPipeline.process() uses (only when
    PipelineConfig.enable_rotation_compensation is True) to integrate a
    short-window relative rotation and compensate the previous comparable
    frame's geometry before E3/E4 run — see
    docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md. None/empty (the default) means
    "no motion samples for this interval" — legal and unremarkable: E5
    falls back cleanly, never blocking Level 3/4 perception.
    """
    left_image: np.ndarray
    right_image: np.ndarray
    left_timestamp: Optional[float] = None
    right_timestamp: Optional[float] = None
    calibration: Optional[StereoCalibration] = None
    frame_id: Optional[str] = None
    motion_hint: Optional[MotionHint] = None
    motion_hints: Optional[Sequence[MotionHint]] = None


@dataclass(frozen=True, slots=True)
class BeamReading:
    """One vertical column-slice of the obstacle scan.

    Mirrors obstacles.ThreatAssessor.assess()'s per-beam dict, typed.

    valid_count/total_pixels (Phase D7): raw valid-pixel coverage for this
    beam's own depth-map column, captured before IQR outlier filtering —
    the smallest backward-compatible extension to ThreatAssessor's return
    shape needed to expose geometric support on
    geometry.provider.ClearanceEvidence (see docs/DPE_V1_PROVIDER_CONTRACT.md's
    D7 record). Defaulted to 0 so any pre-D7 direct construction of a
    BeamReading (bypassing ThreatAssessor.assess()) remains valid; every
    real beam ThreatAssessor itself produces always sets both explicitly.
    distance_m/status derivation is completely unchanged by this addition.
    """
    index: int
    x1: int
    x2: int
    distance_m: float
    status: str  # ThreatAssessor.CLEAR / CAUTION / BLOCKED / NO_DATA
    valid_count: int = 0
    total_pixels: int = 0


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

    temporal_admission_status (Level 4, Phase E2): None unless
    PipelineConfig.enable_temporal is True (default False — see
    docs/E2_TEMPORAL_HISTORY_PLAN.md). When present, one of
    temporal.TemporalAdmissionStatus's plain string constants, reporting
    whether this frame's own timestamp was admitted into the pipeline's
    internal temporal.TemporalHistory buffer (ACCEPTED/
    ACCEPTED_NEW_SEQUENCE) or rejected for chronology reasons
    (REJECTED_INVALID_TIMESTAMP/REJECTED_OLDER_TIMESTAMP/
    REJECTED_DUPLICATE_TIMESTAMP) — never a reinterpretation of this
    frame's own disparity/depth/geometry/obstacle evidence, all of which
    are completely unaffected by temporal-history admission either way.
    Pure metadata about chronology bookkeeping, not a "temporal result" —
    see docs/LEVEL4_ARCHITECTURE.md section 9's raw-vs-temporal authority
    rule.

    temporal_consistency (Level 4, Phase E3): None unless
    PipelineConfig.enable_temporal is True, and even then None on a frame
    whose own timestamp was rejected by temporal chronology (see
    temporal_admission_status above) or when no comparable prior frame
    exists yet (first frame, first after reset(), or the first frame of a
    gap-triggered new sequence — see docs/E3_IMPLEMENTATION_PLAN.md's
    Decision 2). When present, a temporal.TemporalConsistency reporting
    whether this frame's depth_map agrees with the single most recent
    comparable prior frame's — READ-ONLY: computing this never rewrites
    depth_map, geometry, geometry_body, obstacle_cloud, free_space_rays,
    or any other field on this same result, under any outcome (including
    CONTRADICTORY) — see docs/E3_IMPLEMENTATION_PLAN.md's Decision 4.

    temporal_stabilization (Level 4, Phase E4): None unless BOTH
    PipelineConfig.enable_temporal and
    PipelineConfig.enable_temporal_stabilization are True, and even then
    None (INSUFFICIENT_EVIDENCE state, to be precise — see
    temporal.TemporalStabilization's own docstring) whenever
    temporal_consistency itself is None/not-comparable/not-confident-
    enough this frame. When present, a temporal.TemporalStabilization —
    a deterministic, confidence-weighted blend of this frame's own depth
    with the single most recent comparable prior frame's, ADDITIVE
    alongside (never a replacement of) depth_map/geometry/geometry_body
    above, all of which remain byte-identical whether or not this field
    is populated. See docs/E4_IMPLEMENTATION_PLAN.md.

    rotation_compensation_status (Level 4, Phase E5): None unless BOTH
    PipelineConfig.enable_temporal and
    PipelineConfig.enable_rotation_compensation are True. When present,
    one of temporal.RotationCompensationStatus's two plain string
    constants (APPLIED/NOT_APPLIED) — reports whether a short-window
    relative rotation was successfully integrated from this call's
    motion_hints and used to re-express the previous comparable frame's
    geometry before temporal_consistency/temporal_stabilization were
    computed. The relative rotation itself is never exposed here (see
    docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md's Decision 4) — this is a
    pass/fail marker only. Never affects disparity_map/depth_map/geometry/
    etc. above, all of which remain byte-identical regardless.

    motion_aware_reliability (Level 4, Phase E6): None unless BOTH
    PipelineConfig.enable_temporal and
    PipelineConfig.enable_motion_aware_reliability are True. When present,
    a temporal.MotionAwareReliability — a deterministic, EXPLICIT (never
    one opaque blended score) assessment of whether temporal_consistency/
    temporal_stabilization remain trustworthy given this frame's motion
    conditions and rotation_compensation_status. Computed regardless of
    whether E3 actually ran this frame (a chronology-rejected frame is
    still classified, as INSUFFICIENT_EVIDENCE) — but purely read-only of
    every other field on this result: computing it never modifies
    disparity_map, depth_map, any geometry.* field,
    temporal_consistency, temporal_stabilization, or
    rotation_compensation_status, all of which remain byte-identical
    whether or not this field is populated. See
    docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md.

    temporal_persistence (Level 4, Phase E7): None unless ALL THREE of
    PipelineConfig.enable_temporal, enable_motion_aware_reliability, and
    enable_temporal_persistence are True. When present, a
    temporal.TemporalPersistence — a deterministic, per-cell
    classification (NEW/PERSISTENT/DISAPPEARING, plus a frame-level
    CLASSIFIED/UNRELIABLE/INSUFFICIENT_EVIDENCE gate state) of geometric
    evidence across time, requiring repeated chronological support and
    never fabricating free space from history (UNKNOWN stays UNKNOWN —
    see temporal.TemporalPersistence's own docstring). Computed by a
    small, bounded, stateful collaborator
    (temporal.persistence.TemporalPersistenceTracker) the pipeline owns —
    purely read-only of every other field on this result: computing it
    never modifies disparity_map, depth_map, any geometry.* field,
    temporal_consistency, temporal_stabilization,
    rotation_compensation_status, or motion_aware_reliability, all of
    which remain byte-identical whether or not this field is populated.
    See docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md.

    geometry_frame (Phase D2): None unless PipelineConfig.enable_geometry_frame
    is True (default False — every existing caller, including
    mp01_perception, is byte-for-byte unaffected). When present, a
    geometry.GeometryFrame referencing this SAME result's own
    disparity_map/depth_map/valid_*_mask/geometry/geometry_body/
    obstacle_cloud/free_space_rays/geometry_metrics/temporal_consistency/
    temporal_stabilization/rotation_compensation_status/
    motion_aware_reliability/temporal_persistence fields — built by
    fusion.result_builder.build_geometry_frame() purely by reading them
    off this object, never by recomputing anything. This field exists
    only as a migration/backward-compatibility path for existing callers
    of DepthPerceptionResult; GeometryFrame itself, not
    DepthPerceptionResult, is the authoritative DPE V1 provider contract
    a future external perception system should consume — see
    docs/DPE_V1_PROVIDER_CONTRACT.md.

    surface_evidence (Phase D4): None unless PipelineConfig.
    enable_surface_geometry is True AND geometry_body itself exists (same
    enable_geometry + body_T_camera_left extrinsic dependency
    obstacle_cloud/free_space_rays already have — there is no body-frame
    cloud to fit a plane from otherwise). When present, a list of
    geometry.surface.SurfaceEvidence — one deterministic, bounded local
    plane fit (centroid/normal/planarity, or explicitly None when a cell
    has insufficient support) per cell of a small, independently
    configured surface grid (PipelineConfig.surface_grid_rows/
    surface_grid_cols). Purely geometric: answers "what surface geometry
    is observable here," never "can a vehicle traverse/land/navigate
    here" — no occupancy, traversability, or vehicle-specific judgment.
    Computed once by geometry.surface.build_surface_evidence() from the
    same geometry_body PointCloud already built for obstacle_cloud/
    free_space_rays/geometry_metrics — no second body-frame transform, no
    disparity/depth recomputation. Read-only of every other field: never
    modifies disparity_map, depth_map, or any other geometry.* field, all
    of which remain byte-identical whether or not this field is
    populated. See docs/DPE_V1_PROVIDER_CONTRACT.md's D4 record.

    boundary_evidence (Phase D5): None unless PipelineConfig.
    enable_boundary_geometry is True. Unlike surface_evidence, this has NO
    dependency on enable_geometry/geometry_body — its baseline signal
    (depth discontinuity) needs only depth_map, a Level 0 output present
    on every result. When present, a list of
    geometry.boundary.BoundaryEvidence — one evaluated adjacency (RIGHT or
    DOWN neighbor) per cell of a small, independently configured boundary
    grid (PipelineConfig.boundary_grid_rows/boundary_grid_cols), each
    reporting OBSERVED_DISCONTINUITY/NO_DISCONTINUITY/INSUFFICIENT_EVIDENCE
    from a metric depth step and/or (opportunistically, when
    surface_evidence's own grid happens to match) a surface-orientation
    change — never a semantic edge class, never a behavioral judgment,
    never an inferred opening. A missing/insufficient depth measurement on
    either side of a pair is never reinterpreted as a discontinuity — see
    geometry.boundary's own module docstring for the full contract.
    Computed once by geometry.boundary.build_boundary_evidence() directly
    from depth_map (plus surface_evidence when both are enabled and their
    grids align) — no disparity/depth/surface recomputation. Read-only of
    every other field: never modifies disparity_map, depth_map, geometry,
    or surface_evidence, all of which remain byte-identical whether or not
    this field is populated. See docs/DPE_V1_PROVIDER_CONTRACT.md's D5
    record.

    opening_evidence (Phase D6): None unless PipelineConfig.
    enable_opening_geometry is True AND enable_boundary_geometry is ALSO
    True (opening inference structurally requires boundary_evidence to
    confirm flanking structure — there is nothing to build from
    otherwise). When present, a list of geometry.opening.OpeningEvidence
    — zero or more confirmed "geometrically supported gap between
    surrounding physical structures" findings, never
    passable/traversable/safe/doorway/window/fly-through claims, no
    vehicle-size assumption, no platform-specific clearance threshold.
    Unlike boundary_evidence/surface_evidence (one record per grid
    cell/pair, always, including explicit no-finding states), this is a
    POSITIVE-FINDINGS-ONLY list — an empty list means no opening was
    confirmed this frame, not that nothing was evaluated. Computed once
    by geometry.opening.build_opening_evidence() from this SAME frame's
    already-computed boundary_evidence plus depth_map (only to recover
    per-cell absolute depth, which boundary_evidence's own unsigned
    depth_step_m cannot give) — no disparity/depth/boundary
    classification recomputation. Read-only of every other field: never
    modifies disparity_map, depth_map, or boundary_evidence, all of which
    remain byte-identical whether or not this field is populated. See
    docs/DPE_V1_PROVIDER_CONTRACT.md's D6 record.
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
    temporal_admission_status: Optional[str] = None
    temporal_consistency: Optional[TemporalConsistency] = None
    temporal_stabilization: Optional[TemporalStabilization] = None
    rotation_compensation_status: Optional[str] = None
    motion_aware_reliability: Optional[MotionAwareReliability] = None
    temporal_persistence: Optional[TemporalPersistence] = None
    geometry_frame: Optional[GeometryFrame] = None
    surface_evidence: Optional[List[SurfaceEvidence]] = None
    boundary_evidence: Optional[List[BoundaryEvidence]] = None
    opening_evidence: Optional[List[OpeningEvidence]] = None


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
