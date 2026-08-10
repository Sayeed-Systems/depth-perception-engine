"""
Pipeline configuration.

A single, plain, typed dataclass bundling every tunable threshold the
pipeline needs — the same values that used to live as loose module-level
constants in the original standalone main.py. Defaults here match that
file's tuned values exactly, so building a pipeline with PipelineConfig()
reproduces the original desk-tested behavior.

Deliberately excluded: anything specific to flight-velocity planning
(corridor fraction, max forward speed, yaw rate, creep speed). Those drive
examples/navigation/velocity_planner.py, which is a downstream, demo-only
consumer of this engine's output — not part of the Depth Perception Engine
itself. See docs/INTEGRATION_READINESS.md.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PipelineConfig:
    # --- stereo disparity (StereoSGBM) ---
    min_disparity: int = 0
    num_disparities: int = 128       # must be divisible by 16; ~0.31m-10m @ 64mm baseline
    block_size: int = 13             # must be odd; larger = smoother disparity

    # --- shared distance thresholds (obstacles + traversability) ---
    caution_distance_m: float = 0.60   # nearer than this: BLOCKED / OBSTACLE
    clear_distance_m: float = 1.20     # farther than this: CLEAR

    # --- obstacles (per-beam threat assessment) ---
    n_beams: int = 20
    obstacle_percentile: int = 15
    obstacle_min_valid_px: int = 5
    obstacle_blocked_invalid_ratio: float = 0.85
    obstacle_ema_alpha: float = 0.30
    obstacle_debounce_frames: int = 3
    # None => resolved to num_disparities at pipeline construction time,
    # matching the original main.py's ThreatAssessor(dead_zone_px=NUM_DISPARITIES).
    obstacle_dead_zone_px: Optional[int] = None

    # --- traversability (region grid) ---
    traversability_grid_rows: int = 3
    traversability_grid_cols: int = 3
    traversability_ambiguous_fraction_thresh: float = 0.50

    # --- geometry (Level 3, Phase E3) ---
    # Gates DepthPerceptionPipeline.process()'s camera-optical-frame
    # PointCloud stage (geometry.PointCloudBuilder — see
    # docs/LEVEL3_ARCHITECTURE.md). Default is explicitly False: geometry
    # is additional compute cost (see docs/VALIDATION_REPORT.md's E3
    # benchmark) that every existing caller — including mp01_perception,
    # which does not know this field exists — must not start paying for
    # or receiving without opting in. False also reproduces pre-E3
    # behavior exactly: DepthPerceptionResult.geometry stays None, and no
    # PointCloudBuilder is even constructed. No __post_init__ check is
    # added below for this field: it is a plain bool with no invalid
    # *value* to reject (unlike e.g. block_size, which has a real
    # constraint), so a value check would be validation theater.
    enable_geometry: bool = False

    # --- E5 spatial evidence (ObstacleCloud / FreeSpaceRays) ---
    # Both default False — additional compute cost on top of enable_geometry,
    # same "opt-in, existing callers unaffected" discipline as
    # enable_geometry itself. Neither has any effect unless enable_geometry
    # is also True AND a body_T_camera_left extrinsic was supplied to the
    # pipeline (E5 operates on the E4 body-frame cloud only — see
    # docs/LEVEL3_ARCHITECTURE.md's E5 update); with no body-frame cloud
    # there is nothing for either to filter/cast rays from.
    enable_obstacle_geometry: bool = False
    enable_free_space_rays: bool = False

    # Range window (Euclidean distance from the camera's own origin, not
    # the body origin — matches ObstacleCloud.distances_m's frozen
    # docstring) an ObstacleCloud point must fall within. Defaults are
    # deliberately unbounded ("no additional restriction beyond what the
    # sensor itself already enforces upstream" — every valid point is
    # already clamped to [DepthEstimator.MIN_DEPTH_M, MAX_DEPTH_M] in
    # camera-frame depth before this stage ever runs), not a new
    # MP01-specific or hardware-specific number.
    obstacle_min_range_m: float = 0.0
    obstacle_max_range_m: float = float("inf")

    # 2D grid decimation applied identically by both build_obstacle_cloud
    # and build_free_space_rays (see their shared `stride` semantics) —
    # one shared knob rather than two independent ones, since both
    # producers read the same source PointCloud and "keep every Nth
    # pixel" means the same thing for either. 1 = no downsampling
    # (deterministic default — every pixel considered, same as every
    # other geometry stage's default behavior).
    geometry_sampling_stride: int = 1

    # --- E6 geometry quality classification thresholds ---
    # Consumed by geometry.classify_geometry_quality(), an opt-in helper
    # (not auto-invoked by process() — see docs/IMPLEMENTATION_STATUS.md's
    # E6 addendum for why) that maps GeometryMetrics.valid_fraction — a
    # single, already-precisely-defined metric, not a new blended score —
    # onto one of GeometryQuality.HEALTHY/DEGRADED/NO_USABLE_GEOMETRY.
    # These two fractions are a POLICY choice, not a physical constant:
    # there is no universally correct "enough valid geometry" threshold —
    # it depends on sensor, scene, and how the caller intends to use the
    # geometry. The defaults below are conservative, undocumented-against-
    # any-real-dataset placeholders, not tuned/validated values — a real
    # deployment is expected to override them for its own sensor/scene.
    # valid_fraction >= geometry_healthy_min_valid_fraction        -> HEALTHY
    # geometry_degraded_min_valid_fraction <= valid_fraction < ... -> DEGRADED
    # valid_fraction < geometry_degraded_min_valid_fraction        -> NO_USABLE_GEOMETRY
    geometry_healthy_min_valid_fraction: float = 0.5
    geometry_degraded_min_valid_fraction: float = 0.05

    # --- Level 4, Phase E2: bounded temporal history ---
    # Gates DepthPerceptionPipeline's temporal.TemporalHistory buffer.
    # Default False, same "opt-in, existing callers unaffected" discipline
    # as enable_geometry: False reproduces pre-E2 behavior exactly — no
    # TemporalHistory is even constructed, DepthPerceptionResult.
    # temporal_admission_status stays None, zero added cost. See
    # docs/E2_TEMPORAL_HISTORY_PLAN.md.
    enable_temporal: bool = False

    # Hard cap on retained TemporalRecord count, independent of timestamp
    # semantics entirely — protects against a caller feeding unexpectedly
    # dense/rapid timestamps where the time-window bound alone might
    # retain far more records than intended. Not tuned against any
    # dataset; a policy default in the same spirit as
    # geometry_healthy_min_valid_fraction above (see
    # docs/E2_TEMPORAL_HISTORY_PLAN.md's Decision 1 for the full
    # reasoning, including why this is deliberately not derived from an
    # assumed frame rate).
    temporal_max_records: int = 30

    # Maximum age (TemporalRecord.timestamp units, relative to the newest
    # accepted record — never wall-clock time) a retained record may have.
    # A conservative "recent past" window: comfortably shorter than would
    # risk unbounded retention, comfortably longer than any single
    # frame-to-frame interval this project has actually measured (see
    # docs/VALIDATION_REPORT.md's E7 real-hardware run, ~42 FPS).
    temporal_max_age_s: float = 1.0

    # Gap (same units) between two consecutively accepted records beyond
    # which the older history is treated as a genuine timestamp
    # discontinuity and cleared, rather than bridged — see
    # docs/E2_TEMPORAL_HISTORY_PLAN.md's Decision 7. Deliberately smaller
    # than, and independent of, temporal_max_age_s: this answers "is this
    # arrival continuous with the last one," not "is this record still
    # recent enough to keep."
    temporal_gap_limit_s: float = 0.5

    # --- Level 4, Phase E3: read-only temporal consistency ---
    # 2D grid decimation applied to depth_map before it's retained on a
    # TemporalRecord (temporal.TemporalRecord.depth_snapshot_m) — reuses
    # the exact array[::stride, ::stride] pattern
    # geometry_sampling_stride already established for
    # build_obstacle_cloud/build_free_space_rays, but is a DIFFERENT,
    # independent knob: geometry_sampling_stride decimates a
    # geometry.PointCloud for E5 spatial-evidence products;
    # temporal_consistency_sampling_stride decimates depth_map for E3
    # temporal comparison, and has no logical connection to whether
    # enable_geometry is even True. See docs/E3_IMPLEMENTATION_PLAN.md's
    # Decision 5 for the exact memory-cost accounting this bounds.
    temporal_consistency_sampling_stride: int = 4

    # Max absolute depth difference (metres) for one comparable pixel to
    # count as "agreeing" between the current frame and the most recent
    # comparable prior frame — see temporal.compute_temporal_consistency().
    # A policy choice, not a physical constant — "same surface, sensor
    # noise" tolerance, not derived from any specific rig — same
    # "conservative, undocumented-against-any-real-dataset placeholder"
    # discipline as geometry_healthy_min_valid_fraction above.
    temporal_consistency_agreement_tolerance_m: float = 0.05

    # Minimum agreement_fraction (agreeing_count / comparable_count) for
    # temporal.TemporalConsistencyState.CONSISTENT; below this is
    # CONTRADICTORY. Same policy-choice discipline as the fraction above —
    # see docs/E3_IMPLEMENTATION_PLAN.md's Decision 1.
    temporal_consistency_min_agreement_fraction: float = 0.7

    # --- Level 4, Phase E4: deterministic, confidence-aware temporal stabilization ---
    # Gates DepthPerceptionPipeline's stabilization stage. Additional
    # compute cost ON TOP OF enable_temporal (mirrors
    # enable_obstacle_geometry/enable_free_space_rays's own precedent of
    # being gated underneath enable_geometry for the identical "opt-in,
    # existing callers unaffected" reason) — has no effect unless
    # enable_temporal is also True (there is no TemporalHistory/
    # TemporalConsistency to stabilize from otherwise). Default False:
    # even a caller who already opted into enable_temporal for E2/E3
    # should not start paying E4's per-pixel array-construction cost
    # without a second, explicit opt-in.
    enable_temporal_stabilization: bool = False

    # Minimum previous_record.confidence (Level 0-2's own existing
    # per-frame scalar, reused as-is — not a new "temporal trust" concept)
    # for that historical record to be considered at all this frame — see
    # docs/E4_IMPLEMENTATION_PLAN.md's Decision 2. A policy choice, same
    # discipline as every other threshold in this file.
    temporal_stabilization_min_history_confidence: float = 0.3

    # Minimum comparable_count / current_snapshot.size (Phase E3's own
    # already-computed comparable_count, reused directly) for history to
    # be considered at all this frame — "sufficient overlapping valid
    # geometry exists" per docs/E4_IMPLEMENTATION_PLAN.md's Decision 2.
    # Distinct from E3's own bare comparable_count > 0 check: E3 asks "can
    # I classify at all," this asks "is there enough to bother producing
    # a stabilization claim."
    temporal_stabilization_min_comparable_fraction: float = 0.1

    # --- Level 4, Phase E5: short-window rotational motion compensation ---
    # Gates DepthPerceptionPipeline's rotation-compensation stage. Nested
    # underneath enable_temporal (no effect unless it is also True — there
    # is no TemporalHistory/previous record to compensate otherwise) and
    # independent of enable_temporal_stabilization (E5 can usefully feed
    # E3-only consumers too, not just E4's stabilization) — same "opt-in
    # flag nested under a broader one" precedent as
    # enable_obstacle_geometry/enable_free_space_rays under enable_geometry,
    # and enable_temporal_stabilization under enable_temporal. Default
    # False: even a caller who already opted into enable_temporal for
    # E2/E3/E4 should not start paying E5's per-pixel reprojection cost
    # without a third, explicit opt-in. No separate "short window" size
    # threshold exists — temporal_gap_limit_s above already bounds how
    # large a previous-to-current interval can be while still being
    # "continuous" (see docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md's Decision 7).
    enable_rotation_compensation: bool = False

    # --- Level 4, Phase E6: deterministic motion-aware reliability ---
    # Gates DepthPerceptionPipeline's reliability-assessment stage.
    # Nested underneath enable_temporal (no effect unless it is also
    # True — there is no TemporalConsistency/RotationCompensationStatus
    # to assess otherwise), but INDEPENDENT of enable_temporal_
    # stabilization and enable_rotation_compensation: E6 assesses
    # reliability using whatever E3/E4/E5 state actually exists this
    # frame, including "E5 disabled" (a fully legitimate configuration,
    # see docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md section 3's branch 4c) —
    # it does not require either of those flags to be on. Default False:
    # a fourth, explicit opt-in, same "opt-in nested under a broader one"
    # discipline as every prior Level 4 phase's own flag.
    enable_motion_aware_reliability: bool = False

    # Maximum integrated rotation angle (radians) for a frame with
    # rotation_compensation_status == APPLIED to still be considered
    # RELIABLE — beyond this, E5's own zero-order-hold/discretized-
    # reprojection small-rotation assumptions are no longer trusted. A
    # policy choice, not a physical constant — 5 degrees, conservative,
    # not derived from any specific rig's calibration or frame rate. See
    # docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md section 4 for the full
    # justification.
    reliability_max_angular_motion_rad: float = 0.0873  # ~5 degrees

    # Minimum motion_coverage_fraction (fraction of the true previous-to-
    # current interval actually spanned by accepted MotionHint samples,
    # as opposed to left as E5's own documented uncompensated residual
    # tail) for a frame with rotation_compensation_status == APPLIED to
    # still be considered RELIABLE rather than DEGRADED. Same
    # policy-choice discipline as the angular threshold above.
    reliability_min_motion_coverage_fraction: float = 0.5

    # --- Level 4, Phase E7: deterministic per-cell temporal persistence ---
    # Gates DepthPerceptionPipeline's persistence-classification stage
    # (temporal.persistence.TemporalPersistenceTracker). Nested underneath
    # BOTH enable_temporal and enable_motion_aware_reliability (no effect
    # unless both are also True — persistence requires E6's own
    # reliability verdict to gate whether a frame may create/reinforce
    # support, see docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md's Rule 7), same
    # "opt-in flag nested under a broader one" precedent as every prior
    # Level 4 phase's own flag. Default False: a fifth, explicit opt-in.
    enable_temporal_persistence: bool = False

    # Minimum per-cell support_count_grid value (consecutive
    # chronologically-agreeing observations) for
    # temporal.persistence.TemporalPersistenceCellState to read PERSISTENT
    # rather than NEW. Must be >= 2 — a single observation can never
    # become PERSISTENT (an explicit, hard architectural rule, not a
    # policy default that could be tuned down to 1). 2 is the smallest
    # legal value: exactly one repeat is the minimum meaningful definition
    # of "repeated chronological support."
    persistence_min_support_count: int = 2

    # Consecutive absent frames (current decimated depth <= 0.0 at that
    # cell) a previously-PERSISTENT cell may tolerate before reading
    # DISAPPEARING instead — the grace window a single dropout frame must
    # not immediately erase persistence. A policy choice, not a physical
    # constant, same discipline as every other threshold in this file.
    persistence_max_dropout_frames: int = 1

    # Consecutive absent frames after which a cell's tracked persistence
    # state is fully, deterministically cleared (reverts to NO_EVIDENCE —
    # never silently reinterpreted as FREE). Must exceed
    # persistence_max_dropout_frames — expiration is a strictly later
    # event than the dropout grace window, not the same boundary.
    persistence_expiration_absence_frames: int = 5

    def resolved_obstacle_dead_zone_px(self) -> int:
        """obstacle_dead_zone_px if set, else num_disparities."""
        return (
            self.obstacle_dead_zone_px
            if self.obstacle_dead_zone_px is not None
            else self.num_disparities
        )

    def __post_init__(self) -> None:
        """Validate at construction time, not three layers deeper inside
        DisparityEngine/ThreatAssessor/SceneInterpreter — mirrors the
        validation pattern StereoCalibration and RegionAnalyzer already use
        elsewhere in this codebase."""
        if self.num_disparities % 16 != 0:
            raise ValueError(
                f"num_disparities ({self.num_disparities}) must be divisible by 16."
            )
        if self.block_size % 2 == 0 or self.block_size < 1:
            raise ValueError(
                f"block_size ({self.block_size}) must be a positive odd integer."
            )
        if self.caution_distance_m <= 0.0:
            raise ValueError(
                f"caution_distance_m ({self.caution_distance_m}) must be positive."
            )
        if not (self.caution_distance_m < self.clear_distance_m):
            raise ValueError(
                "Require caution_distance_m < clear_distance_m, got "
                f"caution_distance_m={self.caution_distance_m}, "
                f"clear_distance_m={self.clear_distance_m}."
            )
        if self.n_beams < 1:
            raise ValueError(f"n_beams ({self.n_beams}) must be at least 1.")
        if not (0 <= self.obstacle_percentile <= 100):
            raise ValueError(
                f"obstacle_percentile ({self.obstacle_percentile}) must be in [0, 100]."
            )
        if self.obstacle_min_valid_px < 0:
            raise ValueError(
                f"obstacle_min_valid_px ({self.obstacle_min_valid_px}) must be >= 0."
            )
        if not (0.0 <= self.obstacle_blocked_invalid_ratio <= 1.0):
            raise ValueError(
                "obstacle_blocked_invalid_ratio "
                f"({self.obstacle_blocked_invalid_ratio}) must be in [0, 1]."
            )
        if not (0.0 < self.obstacle_ema_alpha <= 1.0):
            raise ValueError(
                f"obstacle_ema_alpha ({self.obstacle_ema_alpha}) must be in (0, 1]."
            )
        if self.obstacle_debounce_frames < 1:
            raise ValueError(
                f"obstacle_debounce_frames ({self.obstacle_debounce_frames}) must be at least 1."
            )
        if self.obstacle_dead_zone_px is not None and self.obstacle_dead_zone_px < 0:
            raise ValueError(
                f"obstacle_dead_zone_px ({self.obstacle_dead_zone_px}) must be >= 0 when set."
            )
        if self.traversability_grid_rows < 1 or self.traversability_grid_cols < 1:
            raise ValueError(
                "traversability_grid_rows/cols must both be at least 1, got "
                f"rows={self.traversability_grid_rows}, cols={self.traversability_grid_cols}."
            )
        if not (0.0 <= self.traversability_ambiguous_fraction_thresh <= 1.0):
            raise ValueError(
                "traversability_ambiguous_fraction_thresh "
                f"({self.traversability_ambiguous_fraction_thresh}) must be in [0, 1]."
            )
        if self.obstacle_min_range_m < 0.0:
            raise ValueError(
                f"obstacle_min_range_m ({self.obstacle_min_range_m}) must be >= 0."
            )
        if not (self.obstacle_min_range_m <= self.obstacle_max_range_m):
            raise ValueError(
                "Require obstacle_min_range_m <= obstacle_max_range_m, got "
                f"obstacle_min_range_m={self.obstacle_min_range_m}, "
                f"obstacle_max_range_m={self.obstacle_max_range_m}."
            )
        if self.geometry_sampling_stride < 1:
            raise ValueError(
                f"geometry_sampling_stride ({self.geometry_sampling_stride}) must be >= 1."
            )
        if not (0.0 <= self.geometry_degraded_min_valid_fraction <= 1.0):
            raise ValueError(
                "geometry_degraded_min_valid_fraction "
                f"({self.geometry_degraded_min_valid_fraction}) must be in [0, 1]."
            )
        if not (0.0 <= self.geometry_healthy_min_valid_fraction <= 1.0):
            raise ValueError(
                "geometry_healthy_min_valid_fraction "
                f"({self.geometry_healthy_min_valid_fraction}) must be in [0, 1]."
            )
        if not (self.geometry_degraded_min_valid_fraction <= self.geometry_healthy_min_valid_fraction):
            raise ValueError(
                "Require geometry_degraded_min_valid_fraction <= geometry_healthy_min_valid_fraction, got "
                f"geometry_degraded_min_valid_fraction={self.geometry_degraded_min_valid_fraction}, "
                f"geometry_healthy_min_valid_fraction={self.geometry_healthy_min_valid_fraction}."
            )
        if self.temporal_max_records < 1:
            raise ValueError(
                f"temporal_max_records ({self.temporal_max_records}) must be >= 1."
            )
        if not (self.temporal_max_age_s > 0.0):
            raise ValueError(
                f"temporal_max_age_s ({self.temporal_max_age_s}) must be > 0."
            )
        if not (self.temporal_gap_limit_s > 0.0):
            raise ValueError(
                f"temporal_gap_limit_s ({self.temporal_gap_limit_s}) must be > 0."
            )
        if self.temporal_consistency_sampling_stride < 1:
            raise ValueError(
                "temporal_consistency_sampling_stride "
                f"({self.temporal_consistency_sampling_stride}) must be >= 1."
            )
        if not (self.temporal_consistency_agreement_tolerance_m >= 0.0):
            raise ValueError(
                "temporal_consistency_agreement_tolerance_m "
                f"({self.temporal_consistency_agreement_tolerance_m}) must be >= 0."
            )
        if not (0.0 <= self.temporal_consistency_min_agreement_fraction <= 1.0):
            raise ValueError(
                "temporal_consistency_min_agreement_fraction "
                f"({self.temporal_consistency_min_agreement_fraction}) must be in [0, 1]."
            )
        if not (0.0 <= self.temporal_stabilization_min_history_confidence <= 1.0):
            raise ValueError(
                "temporal_stabilization_min_history_confidence "
                f"({self.temporal_stabilization_min_history_confidence}) must be in [0, 1]."
            )
        if not (0.0 <= self.temporal_stabilization_min_comparable_fraction <= 1.0):
            raise ValueError(
                "temporal_stabilization_min_comparable_fraction "
                f"({self.temporal_stabilization_min_comparable_fraction}) must be in [0, 1]."
            )
        if not (self.reliability_max_angular_motion_rad > 0.0):
            raise ValueError(
                "reliability_max_angular_motion_rad "
                f"({self.reliability_max_angular_motion_rad}) must be > 0."
            )
        if not (0.0 <= self.reliability_min_motion_coverage_fraction <= 1.0):
            raise ValueError(
                "reliability_min_motion_coverage_fraction "
                f"({self.reliability_min_motion_coverage_fraction}) must be in [0, 1]."
            )
        if self.persistence_min_support_count < 2:
            raise ValueError(
                "persistence_min_support_count "
                f"({self.persistence_min_support_count}) must be >= 2 — one "
                "observation can never become PERSISTENT."
            )
        if self.persistence_max_dropout_frames < 0:
            raise ValueError(
                "persistence_max_dropout_frames "
                f"({self.persistence_max_dropout_frames}) must be >= 0."
            )
        if not (self.persistence_expiration_absence_frames > self.persistence_max_dropout_frames):
            raise ValueError(
                "Require persistence_expiration_absence_frames > persistence_max_dropout_frames, got "
                f"persistence_expiration_absence_frames={self.persistence_expiration_absence_frames}, "
                f"persistence_max_dropout_frames={self.persistence_max_dropout_frames}."
            )
