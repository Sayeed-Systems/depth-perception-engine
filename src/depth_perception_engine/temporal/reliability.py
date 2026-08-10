"""
Deterministic, motion-aware reliability assessment — Level 4, Phase E6.

Mirrors temporal/consistency.py's, temporal/stabilization.py's, and
temporal/rotation_compensation.py's own precedent: a small class of plain
string constants (MotionAwareReliabilityState) plus one pure, stateless
function (compute_motion_aware_reliability) that classifies already-
computed values — no new geometric computation, no motion compensation
(compensate_prior_geometry() is never called from this module), and no
mutation of anything it's given.

E6 reuses two of Phase E5's own pure functions —
select_motion_hint_samples() and integrate_angular_velocity() — a SECOND
time, to derive two diagnostic scalars (motion_sample_count,
angular_motion_magnitude_rad) that RotationCompensationStatus alone does
not expose (E5's own Decision 6 deliberately collapsed every failure
reason into one NOT_APPLIED state). This is reuse of E5's existing,
frozen, tested logic, not a new algorithm, and it is categorically not
"motion compensation": neither function warps any geometry.
temporal/rotation_compensation.py is not modified by this phase — see
docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md's audit section.

E6 is read-only of every upstream Level 4 value, structurally, not just
by convention: compute_motion_aware_reliability() receives only already-
computed enums/scalars and a small, bounded MotionHint sequence, and
returns a new, small, immutable MotionAwareReliability value. It cannot
rewrite TemporalConsistency, TemporalStabilization, or any Level 3 field,
because nothing here holds a mutable reference to any of them.
"""

import math
from typing import Optional, Sequence

import numpy as np

from depth_perception_engine.temporal.consistency import TemporalConsistencyState
from depth_perception_engine.temporal.rotation_compensation import (
    RotationCompensationStatus,
    integrate_angular_velocity,
    select_motion_hint_samples,
)
from depth_perception_engine.temporal.types import MotionAwareReliability, MotionHint


class MotionAwareReliabilityState:
    """Plain string constants for compute_motion_aware_reliability()'s
    MotionAwareReliability.state — mirrors geometry.GeometryQuality's and
    temporal.TemporalConsistencyState's/TemporalStabilizationState's/
    RotationCompensationStatus's existing precedent (plain string
    constants, not a new Enum type)."""

    RELIABLE = "RELIABLE"
    """Either E3 found current/prior evidence CONSISTENT and E5's
    compensation (if enabled) covered the interval adequately with small
    enough angular motion, or E5 was not enabled at all and E3's own
    CONSISTENT verdict stands undegraded — see
    docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md section 3's branches 4a/4c."""

    DEGRADED = "DEGRADED"
    """E3 found current/prior evidence CONSISTENT, but either (a) motion
    could not be validated this frame (missing/invalid/stale/insufficient
    MotionHint coverage — E5 itself enabled but NOT_APPLIED), or (b)
    compensation was applied but covered less than the configured minimum
    fraction of the true interval. The temporal comparison itself
    succeeded; the motion conditions under which it was made could not be
    fully confirmed favorable."""

    UNRELIABLE = "UNRELIABLE"
    """Either E3 itself found current/prior evidence CONTRADICTORY, or
    compensation was applied but the integrated angular motion exceeded
    the configured maximum — beyond either point, the temporal result
    should not be trusted at face value."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    """No real E3 comparison exists to assess reliability of — E3 did not
    run this frame (chronology rejected the frame's own timestamp), or E3
    ran but found no comparable evidence itself (first frame, post-reset,
    gap-restart, or a structural incompatibility)."""


def compute_motion_aware_reliability(
    temporal_consistency_state: Optional[str],
    temporal_stabilization_state: Optional[str],
    rotation_compensation_status: Optional[str],
    motion_hints: Optional[Sequence[MotionHint]],
    previous_timestamp: Optional[float],
    current_timestamp: Optional[float],
    max_angular_motion_rad: float,
    min_motion_coverage_fraction: float,
) -> MotionAwareReliability:
    """Classify this frame's motion-aware reliability from already-
    computed E3/E4/E5 signals.

    Args:
        temporal_consistency_state: this frame's own
            temporal.TemporalConsistency.state, or None if E3 did not run.
        temporal_stabilization_state: this frame's own
            temporal.TemporalStabilization.state, or None if E4 did not
            run/was not enabled.
        rotation_compensation_status: this frame's own
            temporal.RotationCompensationStatus value, or None if E5 was
            not enabled/reachable this frame.
        motion_hints: the same bounded MotionHint sequence passed to
            process() this call (or None) — used only to derive
            motion_sample_count/motion_coverage_fraction/
            angular_motion_magnitude_rad; never mutated, never used to
            compensate any geometry.
        previous_timestamp: the previous comparable frame's own
            timestamp, or None if no comparison interval exists this
            frame (matches E3/E5's own "no prior record" convention).
        current_timestamp: this frame's own timestamp.
        max_angular_motion_rad: PipelineConfig.
            reliability_max_angular_motion_rad — see
            docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md section 4 for the exact
            justification.
        min_motion_coverage_fraction: PipelineConfig.
            reliability_min_motion_coverage_fraction — see the same
            section.

    Returns:
        A MotionAwareReliability. Never raises on ordinary inputs.
    """
    accepted = select_motion_hint_samples(motion_hints, previous_timestamp, current_timestamp)
    motion_sample_count = len(accepted)

    motion_coverage_fraction = None
    if accepted and previous_timestamp is not None and current_timestamp is not None:
        interval = current_timestamp - previous_timestamp
        if interval > 0.0:
            motion_coverage_fraction = (accepted[-1].timestamp - previous_timestamp) / interval

    angular_motion_magnitude_rad = None
    if rotation_compensation_status == RotationCompensationStatus.APPLIED and accepted:
        delta_r = integrate_angular_velocity(accepted, previous_timestamp)
        cos_theta = float(np.clip((np.trace(delta_r) - 1.0) / 2.0, -1.0, 1.0))
        angular_motion_magnitude_rad = math.acos(cos_theta)

    state = _classify(
        temporal_consistency_state, rotation_compensation_status,
        angular_motion_magnitude_rad, motion_coverage_fraction,
        max_angular_motion_rad, min_motion_coverage_fraction,
    )

    return MotionAwareReliability(
        state=state,
        motion_sample_count=motion_sample_count,
        motion_coverage_fraction=motion_coverage_fraction,
        rotation_compensation_status=rotation_compensation_status,
        angular_motion_magnitude_rad=angular_motion_magnitude_rad,
        temporal_consistency_state=temporal_consistency_state,
        temporal_stabilization_state=temporal_stabilization_state,
    )


def _classify(
    temporal_consistency_state: Optional[str],
    rotation_compensation_status: Optional[str],
    angular_motion_magnitude_rad: Optional[float],
    motion_coverage_fraction: Optional[float],
    max_angular_motion_rad: float,
    min_motion_coverage_fraction: float,
) -> str:
    """The exact decision table from docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md
    section 3 — kept as its own tiny, pure function so the priority order
    is easy to read top-to-bottom and to unit-test independently of the
    scalar-derivation logic above."""
    if temporal_consistency_state is None:
        return MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE
    if temporal_consistency_state in (
        TemporalConsistencyState.INSUFFICIENT_EVIDENCE, TemporalConsistencyState.NOT_COMPARABLE,
    ):
        return MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE
    if temporal_consistency_state == TemporalConsistencyState.CONTRADICTORY:
        return MotionAwareReliabilityState.UNRELIABLE

    # temporal_consistency_state == CONSISTENT from here on.
    if rotation_compensation_status == RotationCompensationStatus.APPLIED:
        if angular_motion_magnitude_rad is not None and angular_motion_magnitude_rad > max_angular_motion_rad:
            return MotionAwareReliabilityState.UNRELIABLE
        if motion_coverage_fraction is not None and motion_coverage_fraction < min_motion_coverage_fraction:
            return MotionAwareReliabilityState.DEGRADED
        return MotionAwareReliabilityState.RELIABLE
    if rotation_compensation_status == RotationCompensationStatus.NOT_APPLIED:
        return MotionAwareReliabilityState.DEGRADED
    return MotionAwareReliabilityState.RELIABLE  # rotation_compensation_status is None: E5 not enabled
