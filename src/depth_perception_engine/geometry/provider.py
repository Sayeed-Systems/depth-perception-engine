"""
GeometryFrame — Level 4, Phase D2/D3/D4/D5/D6/D7/D8: the FINAL authoritative
DPE V1 provider contract.

CRITICAL ARCHITECTURAL INVARIANT: GeometryFrame is the one, final,
authoritative shape any external perception system (present or future —
e.g. a future hybrid_perception_engine) is meant to consume DPE geometric
evidence through. Such a consumer must never need to import or inspect
DPE algorithm internals, temporal.TemporalHistory/TemporalRecord,
traversability.RegionAnalyzer/ThreatAssessor internals, traversability
.types.NavigationDecision, intermediate pipeline state, or legacy
DepthPerceptionResult fields beyond `geometry_frame` itself.
GeometryFrame being attached to DepthPerceptionResult
(`DepthPerceptionResult.geometry_frame`) is ONLY a migration/backward-
compatibility mechanism for existing callers — architecturally,
GeometryFrame IS the DPE provider boundary. It exposes only evidence DPE
itself genuinely owns and produces: pure geometry (Level 3), pure
temporal-evidence result contracts (Level 4), and — as of Phase D3 —
neutral geometric evidence extracted from the traversability/obstacle
layers. It carries no traversability policy, no NavigationDecision, no
threat-assessment thresholds, and no speculative HPE/neural fields.

Frame convention: `frame_id` is the frame every top-level metric field on
this object (`disparity_map`, `depth_map`, `valid_disparity_mask`,
`valid_depth_mask`, `geometry`) is expressed in — always
frames.FrameId.CAMERA_OPTICAL_LEFT in this pipeline, since disparity/depth
are always camera-optical-frame rasters. Nested evidence objects that are
expressed in a different frame declare it themselves on their own
`.frame_id` attribute (e.g. `geometry_body`/`obstacle_cloud`/
`free_space_rays` all carry `frame_id == frames.FrameId.BODY`) rather than
this object silently reinterpreting them — see each field's own type for
its own frame contract. `geometry_metrics`, `region_evidence`,
`clearance_evidence`, and every `temporal_*` field are scalar/state
aggregates, not spatial data, and have no frame of their own.

Phase D3 — neutral evidence extraction, no unified confidence: RegionEvidence
and ClearanceEvidence below extract already-computed, genuinely geometric
measurements (spatial extent, depth statistics, validity/coverage,
texture/observability, nearest supported distance) out of the
traversability/obstacle layers, deliberately WITHOUT their behavioral
labels (RegionClass's CLEAR/OBSTACLE/PROBABLE_WALL/etc., ThreatAssessor's
CLEAR/CAUTION/BLOCKED proximity-policy status). Neither type imports
RegionClass, NavigationDecision, or ThreatAssessor — see
tests/test_geometry_frame.py's TestNoBehavioralLeakage for the enforced
proof. Per the architect's explicit D3 decision, no unified/blended
GeometryFrame-level confidence score was introduced: `geometry_metrics.
valid_fraction` remains coverage/validity evidence, and RegionEvidence.
confidence below is RegionAnalyzer's own already-existing PER-REGION
blend, reused as-is — not a new frame-level score.

Phase D4 — the first genuinely NEW geometric algorithm: `surface_evidence`
below is a list of `geometry.surface.SurfaceEvidence` (imported from that
module, not defined here — unlike RegionEvidence/ClearanceEvidence, this
is a real computation, not an extraction, so it lives alongside its
algorithm in surface.py, consistent with this subpackage's own
type+producer-co-located convention for post-Level-3-Phase-E1 additions;
see geometry/geometry_metrics.py's GeometryQuality for the precedent).
Deterministic, bounded local-plane/surface-normal estimation — see
surface.py's own module docstring for the full frame/normalization/
orientation/invalid-support contract. Answers "what surface geometry is
observable here," never "can a vehicle traverse/land/navigate here" — no
"road"/"ground"/"wall"/"landing surface"/"traversable" concept appears
anywhere in this capability.

Phase D5 — geometric boundaries/discontinuities: `boundary_evidence`
below is a list of `geometry.boundary.BoundaryEvidence` (imported from
that module, not defined here — same "lives alongside its algorithm"
reasoning as SurfaceEvidence above). Detects depth and/or
surface-orientation discontinuities between adjacent cells of its OWN
independent grid, reusing depth_map directly (Level 0, always present)
and SurfaceEvidence's already-computed normals opportunistically (only
when its grid happens to match — see boundary.py's own "Grid
independence" section; RegionEvidence's/SurfaceEvidence's own grids are
never redesigned or forced to align). Never a semantic edge class
("curb"/"shoreline"/"road edge"/"wall edge"), never a behavioral judgment
("obstacle to avoid"/"safe"/"traversable"), and never an inferred opening
— two adjacent boundaries do not, by themselves, imply a passage; that
reasoning is out of scope, reserved for a future phase.

Phase D6 — geometric openings/passage structure: `opening_evidence` below
is a list of `geometry.opening.OpeningEvidence` (imported from that
module, not defined here — same "lives alongside its algorithm"
reasoning as SurfaceEvidence/BoundaryEvidence above). A "geometrically
supported gap between surrounding physical structures" — built from
already-confirmed `boundary_evidence` (never re-derived) plus depth_map
(the one signal an unsigned depth step cannot give), on the SAME grid
boundary_evidence itself used. Never triggered by two image edges alone;
never PASSABLE/TRAVERSABLE/SAFE/DOORWAY/WINDOW/FLY_THROUGH, no vehicle
size, no platform-specific clearance threshold — see opening.py's own
module docstring for the full admission-rule/invalid-vs-opening/
coordinate contract.

Phase D7 — ClearanceEvidence refinement: `clearance_evidence`'s own
element type gained geometric coverage/support (`valid_count`/
`total_pixels`/`coverage_fraction`/`support_state`, sourced from a
minimal, backward-compatible extension to obstacles.ThreatAssessor's own
return shape — see ClearanceEvidence's own docstring) and calibrated
horizontal bearing (`bearing_center_rad`/`bearing_min_rad`/
`bearing_max_rad`, derived from the pipeline's own already-computed
rectified focal length/principal point via the standard pinhole bearing
formula — no new camera-geometry algorithm). `nearest_distance_m`'s own
derivation is completely unchanged. No unified GeometryFrame confidence
was introduced (the architect's explicit D7 decision 5) — `support_state`
is a per-sector classification of one already-neutral signal
(coverage_fraction), the same discipline classify_geometry_quality
already established for GeometryMetrics.valid_fraction, not a blended
score.

Phase D8 — geometric quality/uncertainty contract: `quality` below is a
`GeometryFrameQuality` (defined here, not a separate file — a pure,
cheap, deterministic rollup of already-computed state fields, no new
algorithm, no new PipelineConfig threshold beyond the two
geometry_healthy_min_valid_fraction/geometry_degraded_min_valid_fraction
already used by classify_geometry_quality). Summarizes DPE's four
frame-level state-bearing evidence sources (geometry_metrics,
temporal_consistency, motion_aware_reliability, temporal_persistence)
into one shared VALID/DEGRADED/INSUFFICIENT vocabulary
(GeometryFrameQualityState) — deliberately NOT a blended numeric score
(architect decision 4). Also added `RegionEvidence.frame_id` (architect
decision 1) — the one other D3-era evidence type that still lacked one.
See GeometryFrameQuality's own docstring for the full per-dimension
derivation, the absent-vs-degraded distinction, and the explicit
statement that legacy DepthPerceptionResult.confidence is NOT superseded
or redefined by this type.

This module defines the GeometryFrame/RegionEvidence/ClearanceEvidence/
GeometryFrameQuality data shapes only (SurfaceEvidence/BoundaryEvidence/
OpeningEvidence are defined in surface.py/boundary.py/opening.py and
merely re-typed here) — it holds no state, performs no computation
itself, and imports nothing from depth_perception_engine.models or
depth_perception_engine.fusion (which would create a circular import,
since DepthPerceptionResult itself carries a
`geometry_frame: Optional[GeometryFrame]` field). See
fusion.result_builder.build_geometry_frame() /
build_region_evidence() / build_clearance_evidence() /
build_geometry_frame_quality() for how the D2/D3/D8 fields are actually
constructed (by reading already-computed pipeline outputs) and
pipeline.pipeline.DepthPerceptionPipeline.process() for where
geometry.surface.build_surface_evidence() /
geometry.boundary.build_boundary_evidence() /
geometry.opening.build_opening_evidence() are actually called (real new
pipeline stages, gated by PipelineConfig.enable_surface_geometry /
enable_boundary_geometry / enable_opening_geometry).

See docs/DPE_V1_PROVIDER_CONTRACT.md for the full D1-D8 design record.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from depth_perception_engine.geometry.boundary import BoundaryEvidence
from depth_perception_engine.geometry.opening import OpeningEvidence
from depth_perception_engine.geometry.surface import SurfaceEvidence
from depth_perception_engine.geometry.types import FreeSpaceRays, GeometryMetrics, ObstacleCloud, PointCloud
from depth_perception_engine.temporal.types import (
    MotionAwareReliability,
    TemporalConsistency,
    TemporalPersistence,
    TemporalStabilization,
)
from depth_perception_engine.traversability.types import TextureClass


@dataclass(frozen=True, slots=True)
class RegionEvidence:
    """Neutral geometric evidence for one spatial grid region — Phase D3/D8.

    Extracted from traversability.RegionAnalyzer's already-computed
    RegionStats, deliberately WITHOUT RegionStats.classification
    (RegionClass — a navigation-relevant occupancy/behavior label, e.g.
    CLEAR/OBSTACLE/PROBABLE_WALL). Every field here is a physical
    measurement or an evidence-quality signal, never an occupancy
    judgment or a maneuver recommendation: `depth_min_m` reports a
    distance, not "blocked"; `valid_fraction`/`valid_count` report
    coverage, not "unknown"; `texture_class`/`texture_score`/`entropy`/
    `gradient_magnitude` report observability (how much genuine texture
    exists for stereo matching to trust), not a reliability verdict.
    `confidence` is RegionAnalyzer's own already-computed per-region
    blend of the three (reused as-is, not recomputed) — a per-region
    evidence-quality score, not a frame-level unified confidence (the
    architect's D3 decision explicitly excluded introducing one of those).

    A caller wanting the full RegionClass/NavigationDecision-bearing
    interpretation of this same data continues to read
    DepthPerceptionResult.traversability_mask directly — RegionEvidence
    is an additional, neutral view of the same underlying measurements,
    not a replacement.

    `frame_id` (Phase D8): generic, copied from the caller — always
    FrameId.CAMERA_OPTICAL_LEFT in this pipeline's actual usage, matching
    the frame gray/raw_disparity/depth_map (RegionAnalyzer's own inputs)
    already operate in. Added in D8 for the identical reason
    ClearanceEvidence gained one in D7: making spatial-frame semantics
    explicit and consistent with every other evidence type in
    GeometryFrame's own type graph (SurfaceEvidence/BoundaryEvidence/
    OpeningEvidence/ClearanceEvidence all already had one).
    RegionEvidence's own field *set* is otherwise completely unchanged —
    no other redesign.
    """

    frame_id: str
    name: str
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int

    valid_count: int
    total_pixels: int
    valid_fraction: float

    depth_avg_m: float
    depth_median_m: float
    depth_min_m: float
    depth_max_m: float

    texture_score: float
    entropy: float
    gradient_magnitude: float
    texture_class: TextureClass
    confidence: float


class ClearanceSupportState:
    """Plain string constants for ClearanceEvidence.support_state — Phase D7.

    Not a closed Enum, matching this codebase's ThreatAssessor/
    GeometryQuality/BoundaryState precedent for exactly this kind of
    lightweight categorical state. Driven by a single, already-neutral
    signal (`coverage_fraction`) — the same "classify one precisely-
    defined fraction, don't blend unrelated signals" discipline
    geometry.geometry_metrics.classify_geometry_quality already
    established; this is a per-sector counterpart to that per-frame
    classifier, not a step toward a unified GeometryFrame confidence
    score (explicitly out of scope for D7 — see the architect's decision
    5 in docs/DPE_V1_PROVIDER_CONTRACT.md's D7 record).
    """

    SUPPORTED = "SUPPORTED"
    """has_evidence is True (this beam cleared ThreatAssessor's own
    min_valid admission floor) AND coverage_fraction >=
    PipelineConfig.clearance_min_coverage_fraction — a genuinely
    well-supported metric clearance reading."""

    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    """has_evidence is True but coverage_fraction <
    PipelineConfig.clearance_min_coverage_fraction — a real distance WAS
    computed (enough valid pixels existed to clear ThreatAssessor's own
    admission floor), but the underlying support is thinner than the
    configured "trust this fully" bar. nearest_distance_m is still the
    same real, unfabricated value — this state is a coverage-quality
    signal, not a claim the distance itself is wrong."""

    NO_EVIDENCE = "NO_EVIDENCE"
    """has_evidence is False — no distance was produced at all this
    beam. Never reinterpreted as "far"/"free"/infinite clearance."""


@dataclass(frozen=True, slots=True)
class ClearanceEvidence:
    """Neutral directional distance evidence for one obstacle-scan sector — Phase D3/D7.

    Extracted from obstacles.ThreatAssessor's already-computed per-beam
    scan (via the existing, Tier 1 BeamReading), deliberately WITHOUT
    BeamReading.status (CLEAR/CAUTION/BLOCKED/NO_DATA — proximity
    classifications derived from configurable, vehicle-specific
    caution_distance_m/clear_distance_m policy thresholds, not neutral
    physical measurements). `nearest_distance_m` is the same
    IQR-filtered, EMA-smoothed distance BeamReading.distance_m already
    reports, re-expressed as None (never a fabricated 0.0) when no real
    evidence exists this beam — the same "None means no data" convention
    used throughout this codebase (e.g. GeometryMetrics.
    min_obstacle_distance_m). `has_evidence` makes that same fact
    explicit as a boolean rather than requiring a caller to infer it from
    `nearest_distance_m is None`. Unchanged since D3; D7 only adds fields.

    Phase D7 additions:

    `frame_id`: generic, copied from the caller (always
    FrameId.CAMERA_OPTICAL_LEFT in this pipeline's actual usage, matching
    the frame disparity_map/depth_map/ThreatAssessor's own beam scan
    already operate in) — matches SurfaceEvidence/BoundaryEvidence/
    OpeningEvidence's own per-object frame_id convention.

    `valid_count`/`total_pixels`/`coverage_fraction`/`support_state`:
    geometric coverage/support — see ClearanceSupportState's own
    docstring for the 3-way SUPPORTED/PARTIALLY_SUPPORTED/NO_EVIDENCE
    split. `valid_count`/`total_pixels` are BeamReading's own new D7
    fields, reused directly (ThreatAssessor.assess() computes them once,
    not recomputed here); `coverage_fraction = valid_count / total_pixels`
    (0.0 when total_pixels is 0) is a trivial unit conversion, matching
    GeometryMetrics.valid_fraction's own convention, not a new
    computation. Never fabricated: a beam with total_pixels == 0 (a
    degenerate configuration) reads coverage_fraction == 0.0 and
    support_state == NO_EVIDENCE, exactly matching an ordinary
    zero-support beam — geometry that was never observed is never
    silently promoted into "infinite/free clearance."

    `bearing_center_rad`/`bearing_min_rad`/`bearing_max_rad`: calibrated
    horizontal direction, derived from the pipeline's own already-computed
    rectified focal length and principal point (the same Q-matrix-derived
    values rotation compensation and surface/boundary/opening geometry
    already use) — NOT a new camera-geometry computation, a direct
    application of the standard pinhole bearing formula
    `atan2(pixel_x - principal_point_x_px, focal_length_px)` to this
    sector's own x1/x2/(x1+x2)/2 pixel columns. Frame: expressed about the
    optical axis of `frame_id` (CAMERA_OPTICAL_LEFT in practice). Units:
    radians. Sign convention: POSITIVE means the column lies toward
    increasing image-x (visually to the right in the rectified left
    image, and toward positive X in FrameId.CAMERA_OPTICAL_LEFT's own X
    right / Y down / Z forward convention); NEGATIVE means toward
    decreasing image-x (left); exactly 0.0 means the column sits exactly
    on the principal point's own x-coordinate. `bearing_center_rad` is
    the sector's midpoint column's bearing; `bearing_min_rad`/
    `bearing_max_rad` are its two edge columns' bearings (x1 and x2
    respectively — always `bearing_min_rad <= bearing_center_rad <=
    bearing_max_rad`, since the pinhole bearing formula is strictly
    monotonic in pixel column for a fixed, positive focal length). Both
    bounds are exposed, not just the center, for the identical reason
    x1/x2 (pixel bounds) were already exposed rather than only a midpoint
    column — a caller needing the sector's own angular EXTENT, not just
    its centerline, has it directly without reconstructing camera
    intrinsics itself.
    """

    frame_id: str
    index: int
    x1: int
    x2: int
    nearest_distance_m: Optional[float]
    has_evidence: bool

    valid_count: int
    total_pixels: int
    coverage_fraction: float
    support_state: str

    bearing_center_rad: float
    bearing_min_rad: float
    bearing_max_rad: float


class GeometryFrameQualityState:
    """Plain string constants shared by every GeometryFrameQuality
    dimension — Phase D8. Not a closed Enum, matching this codebase's
    established lightweight-categorical-state precedent (GeometryQuality/
    BoundaryState/ClearanceSupportState/etc.). Deliberately a SHARED,
    3-value vocabulary applied uniformly across all dimensions (rather
    than each dimension keeping its own distinct source vocabulary) so a
    consumer reading GeometryFrameQuality never has to learn four
    different state systems — see GeometryFrameQuality's own docstring
    for the full per-dimension derivation and the explicit mapping from
    each dimension's own underlying (and still separately readable, on
    GeometryFrame itself) evidence-specific state.
    """

    VALID = "VALID"
    """This dimension's underlying evidence was computed and found
    trustworthy (e.g. GeometryQuality.HEALTHY, TemporalConsistencyState.
    CONSISTENT, MotionAwareReliabilityState.RELIABLE,
    TemporalPersistenceState.CLASSIFIED)."""

    DEGRADED = "DEGRADED"
    """This dimension's underlying evidence was computed and found an
    ACTIVE problem — not merely an absence of information, a real,
    confirmed issue (e.g. GeometryQuality.DEGRADED,
    TemporalConsistencyState.CONTRADICTORY,
    MotionAwareReliabilityState.DEGRADED/UNRELIABLE,
    TemporalPersistenceState.UNRELIABLE)."""

    INSUFFICIENT = "INSUFFICIENT"
    """This dimension's underlying evidence was computed but there was
    not enough of it to reach a judgment either way (e.g.
    GeometryQuality.NO_USABLE_GEOMETRY, TemporalConsistencyState.
    NOT_COMPARABLE/INSUFFICIENT_EVIDENCE, MotionAwareReliabilityState.
    INSUFFICIENT_EVIDENCE, TemporalPersistenceState.
    INSUFFICIENT_EVIDENCE). Distinct from the dimension being `None`
    (the underlying DPE capability was never enabled/computed this frame
    at all) — see GeometryFrameQuality's own docstring for that
    absent-vs-bad distinction, the literal requirement behind this
    separate value existing."""


@dataclass(frozen=True, slots=True)
class GeometryFrameQuality:
    """Structured geometric quality/uncertainty summary for one
    GeometryFrame — Phase D8 (see docs/DPE_V1_PROVIDER_CONTRACT.md's D8
    record for the full design rationale).

    Answers "what geometric evidence is trustworthy, what is degraded or
    unsupported, and why" by summarizing DPE's four FRAME-LEVEL,
    ALREADY-COMPUTED, state-bearing evidence sources into one shared
    VALID/DEGRADED/INSUFFICIENT vocabulary (GeometryFrameQualityState) —
    never a blended numeric score (the architect's explicit D8 decision
    4: "do NOT blindly collapse all quality information into one scalar
    confidence float" — if an aggregate could not be mathematically
    justified, none was invented; see `overall_state`'s own docstring
    below for exactly what IS and is not claimed by the one aggregate
    this type does define).

    The four dimensions, each `Optional[str]` (see "Absent vs degraded"
    below for what `None` means) using GeometryFrameQualityState's shared
    vocabulary:

        geometry_validity_state: derived from `geometry_metrics.
            valid_fraction` via the EXISTING geometry.geometry_metrics.
            classify_geometry_quality() classifier and the EXISTING
            PipelineConfig.geometry_healthy_min_valid_fraction/
            geometry_degraded_min_valid_fraction thresholds — no new
            threshold was introduced. GeometryQuality.HEALTHY -> VALID,
            DEGRADED -> DEGRADED, NO_USABLE_GEOMETRY -> INSUFFICIENT.

        temporal_consistency_state: derived from `temporal_consistency.
            state` (temporal.TemporalConsistencyState). CONSISTENT ->
            VALID, CONTRADICTORY -> DEGRADED, NOT_COMPARABLE /
            INSUFFICIENT_EVIDENCE -> INSUFFICIENT.

        motion_reliability_state: derived from `motion_aware_reliability.
            state` (temporal.MotionAwareReliabilityState). RELIABLE ->
            VALID, DEGRADED / UNRELIABLE -> DEGRADED,
            INSUFFICIENT_EVIDENCE -> INSUFFICIENT. Deliberately the ONE
            dimension standing in for BOTH Phase E4 (temporal
            stabilization) and Phase E5 (rotation compensation)'s own
            contribution too — MotionAwareReliability was ITSELF already
            designed (Phase E6) as "a deterministic, EXPLICIT assessment
            of whether temporal_consistency/temporal_stabilization remain
            trustworthy given this frame's motion conditions," so a
            separate temporal_stabilization_state dimension would be
            redundant with, not additional to, this one — temporal
            stabilization's own state remains directly readable on
            `GeometryFrame.temporal_stabilization.state` regardless.

        persistence_state: derived from `temporal_persistence.state`
            (temporal.TemporalPersistenceState, the frame-level gate, NOT
            the per-cell state_grid). CLASSIFIED -> VALID, UNRELIABLE ->
            DEGRADED, INSUFFICIENT_EVIDENCE -> INSUFFICIENT.

    Deliberately NOT included as dimensions: RegionEvidence/
    ClearanceEvidence/SurfaceEvidence/BoundaryEvidence/OpeningEvidence.
    Each of these is a per-cell/per-sector/per-span LIST, not a single
    frame-level state — rolling one up into a frame-level dimension would
    require DPE to invent a NEW aggregation (e.g. "fraction of boundary
    cells that are non-INSUFFICIENT") that is not already a precisely-
    defined DPE metric, exactly the kind of not-mathematically-justified
    aggregate decision 4's IMPORTANT note warns against. Each of these
    evidence families already self-describes its own per-item support
    directly (valid_fraction/coverage_fraction/support_fraction/state) —
    a consumer needing a frame-level rollup of any of them can compute it
    directly from the list DPE already exposes, without DPE inventing a
    blended aggregation it cannot uniquely justify.

    Absent vs degraded — the literal test requirement this type exists to
    satisfy: a dimension reads `None` when the underlying DPE capability
    was never enabled/computed this frame at all (e.g.
    PipelineConfig.enable_temporal is False, so `temporal_consistency`
    itself is `None` on GeometryFrame — there is no evidence to assess).
    A dimension reads `GeometryFrameQualityState.INSUFFICIENT` when the
    capability WAS enabled and DID compute a real result object, but that
    object's own state says there wasn't enough evidence to judge (e.g.
    the very first frame of a sequence, before any comparable prior
    exists — `temporal_consistency.state ==
    TemporalConsistencyState.INSUFFICIENT_EVIDENCE`, a real, computed
    outcome, not an absence of computation). These are deliberately
    different values (`None` vs `"INSUFFICIENT"`) so a consumer can tell
    "nothing to assess" apart from "assessed, and there wasn't enough,"
    which is itself different again from `"DEGRADED"` ("assessed, and it
    actively disagrees/is untrustworthy").

    overall_state: a single, simple, fully-transparent DETERMINISTIC
    priority rule over the four dimensions above — NOT a blended score,
    a discrete 3-value rollup: DEGRADED if ANY defined (non-None)
    dimension reads DEGRADED; else INSUFFICIENT if ANY defined dimension
    reads INSUFFICIENT (or if EVERY dimension is None — "no quality-
    bearing evidence exists to assess this frame's geometric
    trustworthiness at all" is itself a meaningful, actionable INSUFFICIENT
    finding, not a silently-omitted state); else VALID (every defined
    dimension read VALID). This is a fixed, documented, "an active
    problem always takes priority over an absence of judgment" policy
    choice, applied identically regardless of WHICH dimension triggered
    it — it does not claim any dimension is inherently "worse" than
    another in general, only that this is the priority DPE applies when
    summarizing across all of them into one field. A consumer who wants a
    different priority, or wants to know the true severity within a
    SINGLE dimension's own vocabulary, reads that dimension (or the
    underlying GeometryFrame.temporal_consistency/etc. field) directly —
    nothing is hidden by this rollup.

    degradation_reasons: a deterministic list of `"{DIMENSION}:{STATE}"`
    strings (fixed order: GEOMETRY_VALIDITY, TEMPORAL_CONSISTENCY,
    MOTION_RELIABILITY, PERSISTENCE) for every DEFINED dimension that is
    NOT VALID — pure convenience derived from the same four fields above,
    zero new information, so a consumer can quickly scan "why" without
    checking each field individually.

    Relationship to existing evidence-specific measurements: this type
    NEVER replaces or modifies geometry_metrics.valid_fraction/
    temporal_consistency.agreement_fraction/motion_aware_reliability.
    motion_coverage_fraction/temporal_persistence.persistent_fraction (or
    any other already-existing precise metric) — every one of those
    remains exactly as before, directly readable on GeometryFrame. This
    type is a purely ADDITIVE, DERIVED summary view over already-computed
    state fields, computed via simple dict lookups and boolean checks —
    zero stereo/depth/surface/boundary/opening recomputation, fully
    deterministic, platform-neutral, independent of navigation behavior
    or vehicle type.

    Relationship to DepthPerceptionResult.confidence: NONE.
    DepthPerceptionResult.confidence (the mean per-region RegionAnalyzer
    confidence blend, unchanged since before Level 3) remains exactly
    what it always was — legacy compatibility behavior for existing
    callers (mp01_perception included). GeometryFrameQuality is the
    authoritative V1 quality/uncertainty contract; legacy `confidence` is
    NOT silently redefined, superseded, or reinterpreted as
    GeometryFrameQuality by this phase, and no code path derives one from
    the other.
    """

    overall_state: str
    geometry_validity_state: Optional[str]
    temporal_consistency_state: Optional[str]
    motion_reliability_state: Optional[str]
    persistence_state: Optional[str]
    degradation_reasons: List[str]


@dataclass(frozen=True, slots=True)
class GeometryFrame:
    """The final, authoritative DPE V1 provider contract — see module docstring.

    Every field here is a reference to (or a neutral, non-recomputing
    extraction of) evidence already produced elsewhere in the pipeline
    (geometry.*, temporal.*, traversability.*, obstacles.*) — GeometryFrame
    itself never recomputes, filters, or reinterprets any of it. Every
    field beyond `timestamp`/`frame_id` is Optional because it reflects
    whatever the pipeline actually had available this frame, exactly
    mirroring the corresponding DepthPerceptionResult field's own
    optionality (None when the relevant PipelineConfig.enable_* flag was
    off, or when no comparable prior frame existed yet — never a
    fabricated placeholder value). `region_evidence`/`clearance_evidence`/
    `quality` are the one exception in practice: `quality` (Phase D8) is a
    pure, cheap rollup of already-computed fields with no enable_* flag of
    its own, so — like region_evidence/clearance_evidence — it is
    populated whenever GeometryFrame itself is
    (`PipelineConfig.enable_geometry_frame is True`).
    """

    timestamp: Optional[float]
    frame_id: str

    disparity_map: np.ndarray
    depth_map: np.ndarray
    valid_disparity_mask: Optional[np.ndarray]
    valid_depth_mask: Optional[np.ndarray]

    geometry: Optional[PointCloud]
    geometry_body: Optional[PointCloud]
    obstacle_cloud: Optional[ObstacleCloud]
    free_space_rays: Optional[FreeSpaceRays]
    geometry_metrics: Optional[GeometryMetrics]

    temporal_consistency: Optional[TemporalConsistency]
    temporal_stabilization: Optional[TemporalStabilization]
    rotation_compensation_status: Optional[str]
    motion_aware_reliability: Optional[MotionAwareReliability]
    temporal_persistence: Optional[TemporalPersistence]

    region_evidence: Optional[Dict[str, RegionEvidence]]
    clearance_evidence: Optional[List[ClearanceEvidence]]

    surface_evidence: Optional[List[SurfaceEvidence]]

    boundary_evidence: Optional[List[BoundaryEvidence]]

    opening_evidence: Optional[List[OpeningEvidence]]

    quality: Optional[GeometryFrameQuality]
