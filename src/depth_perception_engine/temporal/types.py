"""
Level 4 temporal/motion-input contracts — Level 4 (Phases E1-E6).

Pure data only. Nothing in this module compares observations across time,
computes temporal consistency, estimates a stabilized value, integrates a
rotation, or classifies reliability — every type here is immutable data;
the actual algorithms live in temporal/consistency.py (Phase E3),
temporal/stabilization.py (Phase E4), temporal/rotation_compensation.py
(Phase E5), and temporal/reliability.py (Phase E6), mirroring how
geometry/types.py holds data while geometry/geometry_metrics.py holds
classification logic. See docs/LEVEL4_CONTRACTS.md for the full contract
rationale, docs/E2_TEMPORAL_HISTORY_PLAN.md for how TemporalRecord is
produced/consumed, docs/E3_IMPLEMENTATION_PLAN.md for TemporalConsistency,
docs/E4_IMPLEMENTATION_PLAN.md for TemporalStabilization,
docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md for rotation compensation, and
docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md for MotionAwareReliability.

Not exported from the top-level depth_perception_engine package — reachable
only via `from depth_perception_engine.temporal import ...`, matching
geometry/types.py's own precedent from Level 3, Phase E1: promoting this to
the top-level public API would overclaim capability relative to what this
library's documented Tier 1/2 contract covers — see docs/LEVEL4_PUBLIC_API.md.

Scope discipline (see docs/LEVEL4_ARCHITECTURE.md section 2):
MotionHint (E1) was frozen first because its shape was already
well-constrained without an algorithm design decision behind it — the
external motion-input boundary. TemporalRecord (E2), TemporalConsistency
(E3), TemporalStabilization (E4), MotionAwareReliability (E6), and
TemporalPersistence (E7) are now frozen too, once each phase's own design
decisions made their minimal shapes concrete. E5 deliberately added no
new type here at all — it reuses TemporalRecord via dataclasses.replace()
and exposes only a status string, RotationCompensationStatus
(rotation_compensation.py). E4 deliberately did NOT extend TemporalRecord
again — docs/E4_IMPLEMENTATION_PLAN.md's Decision 5 confirms the E2/E3
fields already carry everything the stabilization estimator needs. E7
likewise does NOT extend TemporalRecord — its own per-cell chronology
state lives entirely inside temporal.persistence.TemporalPersistenceTracker
(a small, fixed-size, bounded collaborator, mirroring
obstacles.ThreatAssessor's own cross-frame state precedent), never inside
TemporalHistory's own records — see docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class MotionHint:
    """An optional, externally supplied, short-duration rotational motion
    measurement — a PERCEPTION AID only. Never a pose, never a velocity or
    position estimate, never a localization authority.

    Purpose: future Level 4 phases (E5+) will use the angular rate carried
    here, together with the time interval between two consecutive frame
    timestamps, to estimate a short-duration rotation for aligning/
    comparing recent perception evidence across frames (e.g. "the sensor
    yawed ~2 degrees between these two frames, so don't treat a shifted
    wall as new evidence of change"). This module defines the value only;
    no such alignment/comparison logic exists anywhere in this repository
    yet — see docs/E2_TEMPORAL_HISTORY_PLAN.md.

    Producer: NOT built in this phase. A future producer may be a
    simulated-IMU utility living in tests/examples/simulation tooling (see
    docs/LEVEL4_SIMULATED_IMU.md — explicitly not implemented here) or a
    real IMU driver upstream of this library. This type carries the value;
    it does not create it, and nothing in this repository constructs one
    outside test fixtures.

    Consumer: NOT built in this phase. A future Level 4 temporal-
    consistency component (E5+) is the anticipated consumer — see
    docs/E2_TEMPORAL_HISTORY_PLAN.md for the ownership decision.

    Deliberately gyro-only — no linear velocity, acceleration, or position
    field exists here, and none should be added without a dedicated
    redesign discussion. A translation-capable field would invite exactly
    the pose/localization creep this library's Level 4 scope forbids (see
    docs/LEVEL4_ARCHITECTURE.md section 2) — translation cannot be safely
    inferred from a gyro-only measurement, and this type must not imply
    otherwise by its own shape. There is likewise no `source`/`is_simulated`
    field: a consumer must not branch on where a MotionHint came from (see
    docs/LEVEL4_SIMULATED_IMU.md) — simulated and real producers must both
    resolve to this exact same contract.

    Attributes:
        timestamp: opaque, caller-defined float — same pass-through
            convention as StereoObservation.left_timestamp/
            DepthPerceptionResult.timestamp (no unit conversion or
            synchronization performed by this library). Required (not
            Optional): a MotionHint's entire purpose is anchoring "when"
            an angular rate applies, so a producer with no reliable
            timestamp should not construct one at all — represent "no
            motion hint available" by omitting the value at the point of
            use (e.g. a future `Optional[MotionHint] = None` parameter),
            not by passing a placeholder timestamp here.
        angular_velocity_rad_s: (3,) float32/float64 ndarray, radians per
            second — rotation rate about `frame_id`'s X, Y, Z axes. An
            already-integrated rotation (e.g. a quaternion or rotation
            matrix delta) is deliberately NOT an alternative representation
            here — see this module's docstring and
            docs/LEVEL4_CONTRACTS.md for why raw angular velocity was
            chosen and who will own integrating it into a short-duration
            rotation (the future E5 consumer, not this type).
        frame_id: coordinate frame the angular velocity is expressed in —
            see frames.FrameId. Expected to be frames.FrameId.BODY for a
            body-mounted IMU (the physically natural case, and the only
            one this repository's Level 3 body-frame convention already
            supports without a further, uncalibrated camera-to-IMU
            extrinsic) but not hardcoded to that value — this library
            performs no silent axis conversion or frame inference for a
            MotionHint, matching frames.RigidTransform's own explicit-
            frame-naming convention.
        valid: True (default) means this measurement should be treated at
            face value by a future consumer. False means the producer
            itself flagged this measurement as untrustworthy (e.g. a
            sensor fault flag, a saturated reading, a value the producer
            chose to flag rather than withhold) — distinct from simply not
            supplying a MotionHint at all (represented by
            `Optional[MotionHint] = None` at the point of use, not by this
            field). A future consumer must treat `valid=False` identically
            to "no motion hint supplied" for any compensation purpose —
            never partially trust an invalid hint.
    """

    timestamp: float
    angular_velocity_rad_s: np.ndarray
    frame_id: str
    valid: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.angular_velocity_rad_s, np.ndarray)
            or self.angular_velocity_rad_s.shape != (3,)
        ):
            raise ValueError(
                "MotionHint.angular_velocity_rad_s must be a (3,) ndarray, "
                f"got {getattr(self.angular_velocity_rad_s, 'shape', type(self.angular_velocity_rad_s).__name__)}."
            )
        if not self.frame_id:
            raise ValueError("MotionHint.frame_id must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class TemporalRecord:
    """One minimal chronology entry in a temporal.TemporalHistory buffer —
    Level 4, Phase E2.

    Deliberately NOT a DepthPerceptionResult and does not contain one.
    DepthPerceptionResult may carry expensive full-resolution products
    (disparity/depth arrays, masks, point clouds, obstacle/free-space
    products) — retaining several of those per historical frame would
    multiply memory use for no proven benefit at this phase. This type
    stores only the four fields E2's own chronology/bookkeeping needs;
    see docs/E2_TEMPORAL_HISTORY_PLAN.md's "Additional authoritative
    memory decision" section for why each field, specifically, earns its
    place here.

    Producer: DepthPerceptionPipeline.process(), once per accepted frame,
    when PipelineConfig.enable_temporal is True. Consumer: temporal.
    TemporalHistory (admission/eviction bookkeeping only — E2 performs no
    temporal algorithm over these records; a future E3+ consistency
    algorithm is the anticipated real consumer).

    Attributes:
        timestamp: opaque, caller-defined float, same convention as every
            other timestamp in this library — Optional (not required,
            unlike MotionHint.timestamp) specifically so
            TemporalHistory.admit() can centralize "missing/invalid
            timestamp" rejection itself rather than requiring
            DepthPerceptionPipeline to pre-validate before constructing a
            record (see docs/E2_TEMPORAL_HISTORY_PLAN.md's "Buffer
            behavior" section — timestamp validation must not be
            scattered across pipeline code).
        confidence: DepthPerceptionResult.confidence for this frame —
            float, [0, 1], always present regardless of whether geometry
            is enabled. The one evidence-quality signal guaranteed to
            exist on every frame; without it, a caller who never enables
            geometry would have a temporal record carrying zero
            evidence-quality information at all.
        geometry_quality: one of geometry.GeometryQuality.HEALTHY/
            DEGRADED/NO_USABLE_GEOMETRY (reusing the existing frozen
            Level 3, Phase E6 classification directly — no parallel
            quality system invented), or None if geometry_metrics was not
            computed this call (PipelineConfig.enable_geometry False, or
            no body-frame cloud). None is a fourth, distinct state from
            all three GeometryQuality values — "not computed this call,"
            never conflated with "computed and found unusable."
        motion_hint: the MotionHint (if any) associated with this frame
            via StereoObservation.motion_hint / process()'s motion_hint
            parameter — carried through unread; E2 performs no
            integration, alignment, or validation of its contents beyond
            what MotionHint.__post_init__ itself already enforces.
        depth_snapshot_m: Level 4, Phase E3. A decimated
            (`depth_map[::stride, ::stride]`,
            PipelineConfig.temporal_consistency_sampling_stride) copy of
            this frame's own depth_map, float32, metres, reusing
            depth_map's own existing 0.0-is-invalid convention exactly (no
            second validity array — one array, not two, is the minimum
            representation temporal.compute_temporal_consistency() needs).
            None unless PipelineConfig.enable_temporal is True (whenever
            it is True, this is always populated — depth_map always
            exists). Deliberately NOT the full-resolution depth_map, and
            NOT a geometry.PointCloud/ObstacleCloud/FreeSpaceRays — see
            docs/E3_IMPLEMENTATION_PLAN.md's Decision 5 for the exact
            memory-cost accounting and why this specific representation,
            and only this one, was added.
    """

    timestamp: Optional[float]
    confidence: float
    geometry_quality: Optional[str] = None
    motion_hint: Optional[MotionHint] = None
    depth_snapshot_m: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class TemporalConsistency:
    """A read-only judgement of how consistent the current frame's
    geometry is with the most recent comparable prior frame — Level 4,
    Phase E3.

    Purpose: NOT a filtered/fused/stabilized geometry — this never
    modifies, replaces, or blends any Level 3 field. It only reports
    whether current evidence agrees with what was observed one frame ago,
    so a future consumer (E4, "confidence-aware temporal estimation/
    stabilization" — not built here) has an honest signal to act on. See
    docs/E3_IMPLEMENTATION_PLAN.md's Decision 4 for the safety-enforcement
    argument (this type cannot rewrite anything, structurally, because
    nothing holds a mutable reference to any Level 3 output).

    Producer: temporal.compute_temporal_consistency() (temporal/
    consistency.py), called once per frame by DepthPerceptionPipeline.
    process() when PipelineConfig.enable_temporal is True and this
    frame's timestamp was accepted by temporal.TemporalHistory.

    Attributes:
        state: one of temporal.TemporalConsistencyState's four plain
            string constants (CONSISTENT/CONTRADICTORY/
            INSUFFICIENT_EVIDENCE/NOT_COMPARABLE) — see that class's own
            docstring for the precise trigger for each.
        comparable_count: number of pixels where both the current and
            previous depth snapshots report a valid (> 0.0) measurement —
            the population agreement_fraction is computed over. 0 exactly
            when state is INSUFFICIENT_EVIDENCE or NOT_COMPARABLE.
        agreeing_count: subset of comparable_count where the two depth
            values differ by no more than
            PipelineConfig.temporal_consistency_agreement_tolerance_m.
        disagreement_count: comparable_count - agreeing_count. Always a
            real, non-negative integer — never left implicit as "whatever
            agreement_fraction doesn't cover", stated as its own field per
            Decision 1's explicit ask.
        agreement_fraction: agreeing_count / comparable_count, or None
            when comparable_count is 0. Never 0/0, never silently
            defaulted to 0.0 or 1.0 — None is the only way "no evidence to
            judge from" is represented, so it is never confused with "full
            contradiction" (agreement_fraction near 0.0 with
            comparable_count > 0), per Decision 1's explicit warning.
    """

    state: str
    comparable_count: int
    agreeing_count: int
    disagreement_count: int
    agreement_fraction: Optional[float]


@dataclass(frozen=True, slots=True)
class TemporalStabilization:
    """A deterministic, confidence-aware, read-only-of-current-evidence
    temporal estimate — Level 4, Phase E4.

    Purpose: an OPTIONAL, additive alternative view alongside (never a
    replacement of) the current frame's own raw depth_map — see
    docs/E4_IMPLEMENTATION_PLAN.md's Decision 1. Combines the current
    frame's decimated depth snapshot with the single most recent
    comparable prior temporal.TemporalRecord's own snapshot via one
    closed-form, bounded weighted average, per-pixel — never a neural
    model, Kalman filter, or any learned/opaque component (Decision 9).
    Strong current evidence that contradicts history always wins outright
    at the pixel level and is never blended away (Decision 3).

    Producer: temporal.compute_temporal_stabilization()
    (temporal/stabilization.py), called once per frame by
    DepthPerceptionPipeline.process() when both
    PipelineConfig.enable_temporal and
    PipelineConfig.enable_temporal_stabilization are True, and only when
    temporal.compute_temporal_consistency() (Phase E3) already ran this
    frame (i.e. this frame's own timestamp was accepted by
    temporal.TemporalHistory).

    Same resolution as temporal.TemporalRecord.depth_snapshot_m —
    decimated, NOT full pixel resolution. See
    docs/E4_IMPLEMENTATION_PLAN.md's Decision 1 for why upsampling back to
    full resolution was deliberately not done (it would fabricate
    geometry between real sample points).

    Attributes:
        state: one of temporal.TemporalStabilizationState's three plain
            string constants (STABILIZED/CURRENT_ONLY_FALLBACK/
            INSUFFICIENT_EVIDENCE) — see that class's own docstring for
            the precise trigger for each.
        stabilized_depth_m: the decimated stabilized array, float32,
            metres, same 0.0-is-invalid convention as depth_map/
            TemporalRecord.depth_snapshot_m. None only when state is
            INSUFFICIENT_EVIDENCE. Populated (falling back to the current
            snapshot's own values at every pixel) when state is
            CURRENT_ONLY_FALLBACK too, so a caller wanting "the best
            available estimate" never needs to branch on state first
            whenever state != INSUFFICIENT_EVIDENCE.
        eligible_count: number of pixels valid (> 0.0) in the current
            frame's own decimated snapshot — the denominator both
            fractions below are computed over. 0 exactly when state is
            INSUFFICIENT_EVIDENCE.
        stabilized_count: subset of eligible_count where a genuine
            confidence-weighted blend with history actually occurred this
            frame (current and previous both valid and in agreement).
        contradiction_count: subset of eligible_count where current and a
            valid prior value disagreed beyond tolerance, and current was
            therefore kept unchanged (the safety override, Decision 3) —
            an explicit, honest count of "this many pixels' history was
            overridden," never silently absorbed into state alone (an
            aggregate STABILIZED frame can still report a nonzero
            contradiction_fraction for the pixels that individually
            disagreed).
        stabilized_fraction: stabilized_count / eligible_count, or None
            when state is INSUFFICIENT_EVIDENCE. Never 0/0.
        contradiction_fraction: contradiction_count / eligible_count, or
            None under the same rule.
    """

    state: str
    stabilized_depth_m: Optional[np.ndarray]
    eligible_count: int
    stabilized_count: int
    contradiction_count: int
    stabilized_fraction: Optional[float]
    contradiction_fraction: Optional[float]


@dataclass(frozen=True, slots=True)
class MotionAwareReliability:
    """A deterministic, explicit assessment of whether E3/E4's temporal
    results remain trustworthy given this frame's motion conditions and
    E5's compensation outcome — Level 4, Phase E6.

    Purpose: NOT a blended/opaque confidence score — every underlying
    signal is exposed as its own field, and `state` is a deterministic
    function of them (see temporal.reliability's own docstring for the
    exact decision table). E6 never modifies geometry, disparity, depth,
    E3 consistency evidence, E4 stabilized geometry, or E5's relative
    rotation — it only reads already-computed values and classifies them.
    See docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md for the full derivation.

    Producer: temporal.compute_motion_aware_reliability()
    (temporal/reliability.py), called once per frame by
    DepthPerceptionPipeline.process() when PipelineConfig.enable_temporal
    and PipelineConfig.enable_motion_aware_reliability are both True —
    regardless of whether E3 actually ran this frame (a chronology-
    rejected frame is still classified, as INSUFFICIENT_EVIDENCE).

    Attributes:
        state: one of temporal.MotionAwareReliabilityState's four plain
            string constants (RELIABLE/DEGRADED/UNRELIABLE/
            INSUFFICIENT_EVIDENCE).
        motion_sample_count: number of MotionHint samples
            temporal.rotation_compensation.select_motion_hint_samples()
            found admissible for this frame's (previous_frame_timestamp,
            current_frame_timestamp] interval — "motion-data
            availability." 0 whenever no comparison interval exists this
            frame (e.g. a chronology-rejected frame).
        motion_coverage_fraction: (last_accepted_sample.timestamp -
            previous_frame_timestamp) / (current_frame_timestamp -
            previous_frame_timestamp), or None when motion_sample_count
            is 0. Dimensionless, [0, 1] by construction. Never 0.0
            masquerading as "measured and found zero" — see
            docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md's threshold section.
        rotation_compensation_status: temporal.RotationCompensationStatus's
            value for this frame (APPLIED/NOT_APPLIED), or None if E5
            itself was not enabled/reachable this frame — reused exactly
            as E5 produced it, never reclassified.
        angular_motion_magnitude_rad: the SO(3) rotation angle extracted
            from this frame's ΔR_prev_to_curr (θ = arccos(clip((trace(ΔR)
            - 1) / 2, -1, 1)), the standard closed-form relationship, not
            an invented heuristic), radians. Only computed when
            rotation_compensation_status is APPLIED (a real ΔR was
            actually integrated this frame); None otherwise. The matrix
            itself is never retained or exposed anywhere — only this one
            derived, discarded-after-use scalar.
        temporal_consistency_state: temporal.TemporalConsistencyState's
            value for this frame, or None if E3 did not run (chronology
            rejected the frame's own timestamp) — reused exactly as E3
            produced it.
        temporal_stabilization_state: temporal.TemporalStabilizationState's
            value for this frame, or None if E4 did not run/was not
            enabled — reused exactly as E4 produced it.
    """

    state: str
    motion_sample_count: int
    motion_coverage_fraction: Optional[float]
    rotation_compensation_status: Optional[str]
    angular_motion_magnitude_rad: Optional[float]
    temporal_consistency_state: Optional[str]
    temporal_stabilization_state: Optional[str]


@dataclass(frozen=True, slots=True)
class TemporalPersistence:
    """A deterministic, per-cell classification of geometric evidence
    across time — Level 4, Phase E7.

    Purpose: NOT a filtered/fused/smoothed geometry (like E4's
    TemporalStabilization) and NOT a single frame-level verdict (like
    E3/E6) — persistence is inherently spatial: one region of a frame can
    be freshly NEW while another is long PERSISTENT and a third is
    DISAPPEARING, all in the same call. Classifies each cell of the same
    decimated grid TemporalRecord.depth_snapshot_m already uses (E3's own
    PipelineConfig.temporal_consistency_sampling_stride — no new grid
    invented) into exactly one of temporal.persistence.
    TemporalPersistenceCellState's four codes (NO_EVIDENCE/NEW/PERSISTENT/
    DISAPPEARING), using repeated chronological support tracked by
    temporal.persistence.TemporalPersistenceTracker — a small, bounded,
    per-cell collaborator (support count, absence streak, first-observed
    timestamp; see that module for the full algorithm) reusing E2's own
    admitted-history discipline (nothing unbounded, no complete
    DepthPerceptionResult ever retained) and E5's own rotation-
    compensation function (compensate_prior_geometry_with_payload(), so a
    physically stationary surface stays spatially tracked across a
    rotating sensor — never a second, independent motion-compensation
    implementation).

    UNKNOWN != FREE, always: this type only ever classifies OCCUPIED
    evidence (a cell with a valid — greater than 0.0 — decimated depth
    value) as NEW/PERSISTENT, or a cell that WAS occupied and has since
    gone quiet as DISAPPEARING. A cell with no current evidence and no
    tracked history is NO_EVIDENCE (state_grid code 0) — this type never
    reports, infers, or fabricates FREE space from history, matching
    Level 3's own frozen spatial-evidence invariant
    (docs/DATA_CONTRACTS.md) extended into the temporal dimension.

    Producer: temporal.persistence.TemporalPersistenceTracker.update(),
    called once per frame by DepthPerceptionPipeline.process() when
    PipelineConfig.enable_temporal, enable_motion_aware_reliability, AND
    enable_temporal_persistence are all True — E7 is layered directly on
    top of E6's own reliability verdict (rule: an E6 UNRELIABLE frame can
    never create or reinforce persistence, see `state` below), which is
    itself layered on E3/E4/E5's outputs; persistence is not computed
    from raw depth alone.

    Attributes:
        state: one of temporal.persistence.TemporalPersistenceState's
            three plain string constants:
            - CLASSIFIED: this frame's own current evidence was used to
              update the tracker; state_grid/support_count_grid/
              age_s_grid all reflect THIS frame's fresh classification.
            - UNRELIABLE: this frame's own MotionAwareReliability.state
              was UNRELIABLE (Phase E6) — per the architect's own Rule 7,
              this frame's evidence must not create or reinforce
              persistence. The tracker is NOT updated at all this call;
              state_grid/support_count_grid are the tracker's own
              unchanged snapshot from the most recent CLASSIFIED frame
              (never erased by an unreliable frame — see Rule 4's
              "temporary dropout/degradation must not immediately erase
              previously persistent evidence"), while age_s_grid is
              still freshly computed against this frame's own timestamp
              (elapsed real time does not require reinforcement to be
              true).
            - INSUFFICIENT_EVIDENCE: a genuine structural incompatibility
              between this frame's decimated grid shape and the shape the
              tracker has been accumulating against — mirrors
              temporal.TemporalConsistencyState.NOT_COMPARABLE's own rare,
              defensive trigger exactly (this library has no motion/pose
              estimation capability, so it cannot and does not attempt to
              infer correspondence any other way). state_grid/
              support_count_grid/age_s_grid are all None; every count is
              0; persistent_fraction is None. In ordinary operation
              (fixed calibration, fixed
              temporal_consistency_sampling_stride) this is unreachable —
              the decimated grid shape never changes within one pipeline
              instance.
        state_grid: (H_dec, W_dec) int8 array, same shape as
            TemporalRecord.depth_snapshot_m, or None (see `state` above).
            Each cell holds one of temporal.persistence.
            TemporalPersistenceCellState's four codes — NO_EVIDENCE (0,
            the overwhelming background case: no current evidence, no
            tracked history), NEW (1), PERSISTENT (2), or DISAPPEARING
            (3). A cell can never hold a fifth code for "free" — see this
            type's own UNKNOWN != FREE paragraph above.
        support_count_grid: (H_dec, W_dec) int32 array, or None under the
            same condition as state_grid. Per-cell count of consecutive
            chronologically-agreeing observations (0 = never tracked or
            fully expired). A cell reaching
            PipelineConfig.persistence_min_support_count is exactly the
            threshold at which state_grid transitions from NEW to
            PERSISTENT — one observation (count == 1) can never read
            PERSISTENT, structurally, because
            PipelineConfig.persistence_min_support_count is validated
            >= 2 at construction (Rule 2).
        age_s_grid: (H_dec, W_dec) float32 array, or None under the same
            condition as state_grid. Per-cell elapsed time (this frame's
            own timestamp minus the timestamp of that cell's own first
            observation in its current, unexpired support streak) —
            0.0 for an untracked cell (support_count_grid == 0) and,
            not coincidentally, also 0.0 for a cell classified NEW this
            exact frame (first-observed timestamp == this frame's own
            timestamp) — both correctly represent "zero elapsed support
            time," never conflated with a real non-zero age.
        new_count / persistent_count / disappearing_count: cell counts
            for each of state_grid's three non-background codes — always
            non-negative integers, always 0 when state_grid is None.
        expired_count: number of cells whose absence streak crossed
            PipelineConfig.persistence_expiration_absence_frames THIS
            frame — the deterministic moment a cell's tracked state is
            fully cleared (support_count_grid entry resets to 0,
            state_grid entry reverts to NO_EVIDENCE) per Rule 5. 0 when
            state is not CLASSIFIED (no update ran).
        eligible_count: new_count + persistent_count + disappearing_count
            — the denominator persistent_fraction is computed over;
            NO_EVIDENCE cells are deliberately excluded (matching every
            other *_fraction field in this library's own "exclude
            background/non-participating population" convention).
        persistent_fraction: persistent_count / eligible_count, or None
            when eligible_count is 0. Never 0/0.
    """

    state: str
    state_grid: Optional[np.ndarray]
    support_count_grid: Optional[np.ndarray]
    age_s_grid: Optional[np.ndarray]
    new_count: int
    persistent_count: int
    disappearing_count: int
    expired_count: int
    eligible_count: int
    persistent_fraction: Optional[float]
