"""
Depth Perception Engine — a standalone, ROS-free stereo depth/traversability/
obstacle library.

Repository/library name: depth_perception_engine. Canonical processing
object: DepthPerceptionPipeline — the repository is not named after a
class, and the class is not named after the repository; there is no
`DepthPerceptionEngine` symbol and none is planned (see docs/PUBLIC_API.md).

Canonical usage — every symbol below imports directly from this package
root; nothing external should need to import from a submodule:

    from depth_perception_engine import (
        DepthPerceptionPipeline,
        PipelineConfig,
        StereoObservation,
        load_stereo_calibration,
    )

    calibration = load_stereo_calibration("/path/to/stereo_calibration.xml")
    pipeline = DepthPerceptionPipeline(PipelineConfig(), calibration)   # build once
    result = pipeline.process(left_image, right_image)                 # per frame

Three API tiers — full reference in docs/PUBLIC_API.md:

    Tier 1 (primary):  DepthPerceptionPipeline, PipelineConfig,
                        StereoCalibration, StereoObservation,
                        DepthPerceptionResult, PipelineHealth,
                        TraversabilityResult, ObstacleAssessment,
                        BeamReading, NavigationDecision, RegionClass,
                        RegionStats, TextureClass, load_stereo_calibration,
                        GeometryFrame, TemporalConsistency,
                        TemporalConsistencyState, TemporalStabilization,
                        TemporalStabilizationState,
                        RotationCompensationStatus, MotionAwareReliability,
                        MotionAwareReliabilityState, TemporalPersistence,
                        TemporalPersistenceState, TemporalPersistenceCellState,
                        PointCloud, ObstacleCloud, FreeSpaceRays,
                        GeometryMetrics, RegionEvidence, ClearanceEvidence,
                        SurfaceEvidence, BoundaryEvidence, BoundaryState,
                        BoundaryDirection, OpeningEvidence,
                        ClearanceSupportState, GeometryFrameQuality,
                        GeometryFrameQualityState, FrameId, MotionHint
    Tier 2 (advanced):  process_stereo_pair, compute_disparity,
                        estimate_depth, detect_obstacles,
                        classify_traversability
    Tier 3 (internal):  everything else — submodules remain importable
                        (e.g. depth_perception_engine.pipeline) but are not
                        part of the documented, stable contract.

GeometryFrame (Phase D2) is the FINAL, authoritative DPE V1 provider
contract for any external perception system — see
docs/DPE_V1_PROVIDER_CONTRACT.md. DepthPerceptionResult.geometry_frame is
only a migration/compatibility path onto it, gated by
PipelineConfig.enable_geometry_frame (default False). TemporalConsistency/
TemporalStabilization/RotationCompensationStatus/MotionAwareReliability/
TemporalPersistence (plus their *State/*CellState constant classes) are
promoted to Tier 1 here because GeometryFrame carries them directly — a
caller reading a GeometryFrame's temporal fields must be able to interpret
them without an internal temporal.* import.

Phase D3 promoted PointCloud/ObstacleCloud/FreeSpaceRays/GeometryMetrics
to Tier 1 for the identical reason (GeometryFrame's own `geometry`/
`geometry_body`/`obstacle_cloud`/`free_space_rays`/`geometry_metrics`
fields are typed against them) and added two new neutral evidence
contracts, RegionEvidence and ClearanceEvidence — geometric measurements
(spatial extent, depth statistics, validity/coverage, texture/
observability, nearest supported distance) extracted from the
traversability/obstacle layers WITHOUT their behavioral labels
(RegionClass, NavigationDecision, ThreatAssessor's CLEAR/CAUTION/BLOCKED).
No unified/blended GeometryFrame-level confidence score was introduced —
see docs/DPE_V1_PROVIDER_CONTRACT.md's D3 record. Builders/algorithms
(PointCloudBuilder, build_obstacle_cloud, build_free_space_rays,
build_geometry_metrics, GeometryQuality, classify_geometry_quality) stay
Tier 3 — only the result *types* were promoted.

Phase D4 added the first genuinely NEW geometric algorithm: deterministic,
bounded local surface-normal (plane) estimation over the existing
body-frame PointCloud, exposed as SurfaceEvidence (Tier 1) — a list of
per-cell centroid/normal/planarity evidence, or explicitly None fields
when a cell has insufficient support. Gated by
PipelineConfig.enable_surface_geometry (default False). Answers "what
surface geometry is observable here," never "can a vehicle traverse/
land/navigate here" — no occupancy, traversability, or vehicle-specific
concept appears anywhere in it. geometry.surface.build_surface_evidence
(the algorithm) stays Tier 3, same "promote the type, not the producer"
rule as Phase D3. See docs/DPE_V1_PROVIDER_CONTRACT.md's D4 record for
the full frame/normalization/orientation/invalid-support contract.

Phase D5 added geometric boundary/discontinuity evidence: BoundaryEvidence
(Tier 1, plus its BoundaryState/BoundaryDirection state-constant classes)
reports depth and/or surface-orientation discontinuities between adjacent
cells of its own independent grid — reusing depth_map directly (Level 0,
always present, independent of enable_geometry) and SurfaceEvidence's
normals opportunistically (only when its grid matches). Three explicit
states — OBSERVED_DISCONTINUITY / NO_DISCONTINUITY / INSUFFICIENT_EVIDENCE
— so a missing depth measurement is never silently promoted into a
physical boundary. Never a semantic edge class ("curb"/"shoreline"/"road
edge"/"wall edge"), never a behavioral judgment ("obstacle to avoid"/
"safe"/"traversable"), and never an inferred opening. Gated by
PipelineConfig.enable_boundary_geometry (default False).
geometry.boundary.build_boundary_evidence (the algorithm) stays Tier 3,
same "promote the type, not the producer" rule. See
docs/DPE_V1_PROVIDER_CONTRACT.md's D5 record for the full discontinuity-
definition/invalid-vs-boundary/coordinate contract.

Phase D6 added geometric opening/passage-structure evidence: OpeningEvidence
(Tier 1) reports zero or more confirmed "geometrically supported gap[s]
between surrounding physical structures" — built from this frame's already-
confirmed BoundaryEvidence (never re-derived) plus depth_map (only to
recover the per-cell absolute depth an unsigned depth step cannot give),
on the SAME grid boundary_evidence itself used. Never triggered by two
image edges alone; never PASSABLE/TRAVERSABLE/SAFE/DOORWAY/WINDOW/
FLY_THROUGH, no vehicle-size assumption, no platform-specific clearance
threshold. Unlike BoundaryEvidence/SurfaceEvidence (one record per grid
cell/pair always), OpeningEvidence is a positive-findings-only list — an
empty list means no opening was confirmed, not that nothing was
evaluated. Gated by PipelineConfig.enable_opening_geometry (default
False, nested under enable_boundary_geometry). geometry.opening.
build_opening_evidence (the algorithm) stays Tier 3, same "promote the
type, not the producer" rule. See docs/DPE_V1_PROVIDER_CONTRACT.md's D6
record for the full admission-rule/invalid-vs-opening/coordinate contract.

Phase D7 refined ClearanceEvidence into a complete neutral directional-
geometry product: geometric coverage/support (valid_count/total_pixels/
coverage_fraction/support_state — the last a new Tier 1 state-constant
class, ClearanceSupportState, with a SUPPORTED/PARTIALLY_SUPPORTED/
NO_EVIDENCE 3-way split, mirroring classify_geometry_quality's own
single-signal classification discipline, not a step toward unified
GeometryFrame confidence) and calibrated horizontal bearing
(bearing_center_rad/bearing_min_rad/bearing_max_rad, derived from
already-available rectified focal length/principal point via the
standard pinhole formula — no new camera-geometry algorithm).
`nearest_distance_m`'s own derivation is completely unchanged.
obstacles.ThreatAssessor's own return shape gained the smallest possible
backward-compatible extension (valid_count/total_pixels per beam,
BeamReading defaulted to 0) to make this coverage available without a
second scan of depth_map. See docs/DPE_V1_PROVIDER_CONTRACT.md's D7
record for the full bearing/coverage/frame contract.

Phase D8 established coherent quality/uncertainty semantics across
GeometryFrame: GeometryFrameQuality (Tier 1, plus its
GeometryFrameQualityState VALID/DEGRADED/INSUFFICIENT state-constant
class) rolls up DPE's four frame-level state-bearing evidence sources
(geometry_metrics, temporal_consistency, motion_aware_reliability,
temporal_persistence) into one shared vocabulary — deliberately NOT a
blended numeric confidence float (if an aggregate could not be
mathematically justified, none was invented). Each dimension is `None`
when its own underlying capability was never enabled/computed this frame
(absent), distinct from `INSUFFICIENT` (the capability ran but found too
little evidence to judge) and `DEGRADED` (the capability ran and found an
active problem) — `overall_state` is a single, fully-transparent priority
rule (DEGRADED beats INSUFFICIENT beats VALID), never a score, and
`degradation_reasons` lists exactly which dimensions triggered it. Named
deliberately distinctly from the pre-existing, narrower
geometry.GeometryQuality (a single-signal HEALTHY/DEGRADED/
NO_USABLE_GEOMETRY classification of GeometryMetrics.valid_fraction alone,
unchanged since Phase E6, and the direct source of
GeometryFrameQuality's own geometry_validity_state dimension — not
replaced or redefined). DepthPerceptionResult.confidence (legacy,
unchanged) is explicitly NOT superseded or redefined by
GeometryFrameQuality — see docs/DPE_V1_PROVIDER_CONTRACT.md's D8 record
for the full contract. RegionEvidence also gained a `frame_id` field
(the one other D3-era evidence type that still lacked one).

Phase D9 audited GeometryFrame's complete public type graph for freeze-
readiness and found exactly one inconsistency: `frames.FrameId` — the
source of every `frame_id` string value across the whole type graph —
was the one categorical-value source never promoted to Tier 1, unlike
every other state-bearing field's own constant class. Phase D10 resolved
this: `FrameId` is now Tier 1 — a pure API/export/test/doc change, zero
behavioral change (every `frame_id` field already held the exact same
string values before and after; this only adds a named, importable
vocabulary for comparing against them instead of hardcoding magic
strings). `frames.RigidTransform` remains Tier 3 — it is a pipeline
CONSTRUCTOR input (`body_T_camera_left`), never part of GeometryFrame's
own output type graph, so it was not in scope for this promotion. See
docs/DPE_V1_PROVIDER_CONTRACT.md's D9/D10 records.

Phase D13 audited the complete public API surface for provider/consumer
hardening: GeometryFrame's own recursive type graph was confirmed already
fully Tier 1 end to end (every field's type — directly or via
Optional/List/Dict — resolves to a Tier 1 symbol; zero gap found). One
real INPUT-side gap was found and closed: MotionHint — required to
construct a complete StereoObservation.motion_hint/.motion_hints or to
call DepthPerceptionPipeline.process()'s own motion_hint/motion_hints
parameters — was Tier 3 only, forcing an internal
depth_perception_engine.temporal import for any caller wanting to supply
motion data. Promoted here: a pure API/export/test/doc change, zero
behavioral change (MotionHint's own shape/validation is untouched). It is
not part of GeometryFrame's own output type graph (D2 deliberately did
not promote it for that reason) — it is promoted now as one of D13's own
named Tier 1 INPUT contracts, symmetric to how D10 promoted FrameId on
the output side. See docs/DPE_V1_PROVIDER_CONTRACT.md's D13 record.

Existing subpackage imports (`from depth_perception_engine.pipeline import
DepthPerceptionPipeline`, etc.) continue to work and resolve to the exact
same objects as the top-level import — see
tests/test_public_api.py::TestImportIdentity. They remain valid
compatibility paths, not the canonical style for new code.

No module in this package imports rclpy, sensor_msgs, cv_bridge, opens a
camera device, or calls any cv2.imshow/waitKey/GUI function — see
tests/test_no_ros_dependency.py. Importing this package performs no I/O
and has no runtime side effects.
"""

from depth_perception_engine.calibration import StereoCalibration, load_stereo_calibration
from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry import (
    BoundaryDirection,
    BoundaryEvidence,
    BoundaryState,
    ClearanceEvidence,
    ClearanceSupportState,
    FreeSpaceRays,
    GeometryFrame,
    GeometryFrameQuality,
    GeometryFrameQualityState,
    GeometryMetrics,
    ObstacleCloud,
    OpeningEvidence,
    PointCloud,
    RegionEvidence,
    SurfaceEvidence,
)
from depth_perception_engine.models import (
    BeamReading,
    DepthPerceptionResult,
    ObstacleAssessment,
    PipelineHealth,
    StereoObservation,
    TraversabilityResult,
)
from depth_perception_engine.pipeline import (
    DepthPerceptionPipeline,
    classify_traversability,
    compute_disparity,
    detect_obstacles,
    estimate_depth,
    process_stereo_pair,
)
from depth_perception_engine.temporal import (
    MotionAwareReliability,
    MotionAwareReliabilityState,
    MotionHint,
    RotationCompensationStatus,
    TemporalConsistency,
    TemporalConsistencyState,
    TemporalPersistence,
    TemporalPersistenceCellState,
    TemporalPersistenceState,
    TemporalStabilization,
    TemporalStabilizationState,
)
from depth_perception_engine.traversability import (
    NavigationDecision,
    RegionClass,
    RegionStats,
    TextureClass,
)

__version__ = "1.0.1"

__all__ = [
    "__version__",
    # --- Tier 1: primary stable API ---
    # calibration
    "StereoCalibration",
    "load_stereo_calibration",
    # config
    "PipelineConfig",
    # results
    "DepthPerceptionResult",
    "TraversabilityResult",
    "ObstacleAssessment",
    "BeamReading",
    "StereoObservation",
    "PipelineHealth",
    # traversability result shape (embedded in DepthPerceptionResult;
    # promoted to Tier 1 because a caller cannot meaningfully interpret
    # result.traversability_mask without these — see docs/PUBLIC_API.md)
    "NavigationDecision",
    "RegionClass",
    "RegionStats",
    "TextureClass",
    # pipeline
    "DepthPerceptionPipeline",
    # Phase D2: GeometryFrame — the final, authoritative DPE V1 provider
    # contract (see docs/DPE_V1_PROVIDER_CONTRACT.md) — plus the Level 4
    # temporal result contracts it carries, promoted to Tier 1 alongside
    # it (a caller cannot meaningfully interpret geometry_frame's temporal
    # fields without these).
    "GeometryFrame",
    "TemporalConsistency",
    "TemporalConsistencyState",
    "TemporalStabilization",
    "TemporalStabilizationState",
    "RotationCompensationStatus",
    "MotionAwareReliability",
    "MotionAwareReliabilityState",
    "TemporalPersistence",
    "TemporalPersistenceState",
    "TemporalPersistenceCellState",
    # Phase D3: the four Level 3 geometry result types GeometryFrame's own
    # fields are typed against, plus the two new neutral evidence
    # contracts extracted from traversability/obstacles.
    "PointCloud",
    "ObstacleCloud",
    "FreeSpaceRays",
    "GeometryMetrics",
    "RegionEvidence",
    "ClearanceEvidence",
    # Phase D4: the first genuinely new geometric evidence type — local
    # surface-normal/planarity estimation over the body-frame PointCloud.
    "SurfaceEvidence",
    # Phase D5: geometric boundary/discontinuity evidence, plus its two
    # state-constant classes (needed to interpret .state/.direction).
    "BoundaryEvidence",
    "BoundaryState",
    "BoundaryDirection",
    # Phase D6: geometric opening/passage-structure evidence.
    "OpeningEvidence",
    # Phase D7: ClearanceEvidence's own coverage/support state-constant
    # class (ClearanceEvidence itself was already Tier 1 since D3).
    "ClearanceSupportState",
    # Phase D8: the structured geometric quality/uncertainty rollup.
    "GeometryFrameQuality",
    "GeometryFrameQualityState",
    # Phase D10: the canonical frame-name vocabulary for every frame_id
    # field across GeometryFrame's own type graph.
    "FrameId",
    # Phase D13: MotionHint — required to construct a complete public
    # INPUT contract (StereoObservation.motion_hint/.motion_hints,
    # DepthPerceptionPipeline.process()'s own motion_hint/motion_hints
    # parameters) without an internal depth_perception_engine.temporal
    # import. Not part of GeometryFrame's own output type graph (D2
    # deliberately did not promote it for that reason) — promoted now
    # because it is one of D13's own named Tier 1 INPUT contracts.
    "MotionHint",
    # --- Tier 2: advanced functional API ---
    "process_stereo_pair",
    "compute_disparity",
    "estimate_depth",
    "classify_traversability",
    "detect_obstacles",
]
