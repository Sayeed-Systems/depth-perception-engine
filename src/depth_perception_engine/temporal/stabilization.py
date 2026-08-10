"""
Deterministic, confidence-aware temporal stabilization — Level 4, Phase E4.

Mirrors geometry/geometry_metrics.py's GeometryQuality/
classify_geometry_quality and temporal/consistency.py's
TemporalConsistencyState/compute_temporal_consistency precedent exactly: a
small class of plain string constants (TemporalStabilizationState) plus
one pure, stateless, vectorized function (compute_temporal_stabilization)
that combines already-computed values — no new geometric computation, no
reprojection, no re-derivation of validity, and no mutation of anything
it's given.

E4 is read-only of current Level 3 evidence, structurally, not just by
convention: this function never receives a mutable reference to
DepthPerceptionResult or any Level 3 array, and returns a new, small,
immutable TemporalStabilization value. It cannot rewrite depth_map,
geometry, geometry_body, obstacle_cloud, or free_space_rays under any
outcome — see docs/E4_IMPLEMENTATION_PLAN.md's Decision 1/3.

E4 performs no IMU integration, no motion/ego-motion estimation, no
rotation/translation compensation, and no learned/neural/opaque
component of any kind — see docs/E4_IMPLEMENTATION_PLAN.md's Decisions
8-9. It reads temporal.MotionHint nowhere.
"""

from typing import Optional

import numpy as np

from depth_perception_engine.temporal.consistency import TemporalConsistencyState
from depth_perception_engine.temporal.types import TemporalConsistency, TemporalRecord, TemporalStabilization


class TemporalStabilizationState:
    """Plain string constants for compute_temporal_stabilization()'s
    TemporalStabilization.state — mirrors geometry.GeometryQuality's and
    temporal.TemporalConsistencyState's existing precedent (plain string
    constants, not a new Enum type)."""

    STABILIZED = "STABILIZED"
    """The Decision-2 frame-level gate passed AND at least one pixel was
    genuinely confidence-blended with history this frame
    (stabilized_count > 0)."""

    CURRENT_ONLY_FALLBACK = "CURRENT_ONLY_FALLBACK"
    """The Decision-2 frame-level gate passed, but zero pixels actually
    qualified for blending (e.g. every overlapping pixel individually
    contradicted, or history had no overlap with any of current's valid
    pixels) — stabilized_depth_m falls back to the current snapshot's own
    values everywhere, never fabricated from history."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    """The Decision-2 frame-level gate failed — no prior record, E3 found
    the frames not legitimately comparable (NOT_COMPARABLE) or found no
    evidence to judge from (INSUFFICIENT_EVIDENCE), the prior record's
    own confidence fell below the configured minimum, or too little of
    the frame overlapped validly. stabilized_depth_m is None; every count
    is 0; every fraction is None. Current observation remains fully
    authoritative — no stabilization claim of any kind."""


def compute_temporal_stabilization(
    current_depth_snapshot: np.ndarray,
    previous_record: Optional[TemporalRecord],
    consistency: Optional[TemporalConsistency],
    agreement_tolerance_m: float,
    min_history_confidence: float,
    min_comparable_fraction: float,
) -> TemporalStabilization:
    """Combine current_depth_snapshot with previous_record's own snapshot
    into a deterministic, confidence-weighted stabilized estimate.

    Args:
        current_depth_snapshot: this frame's own decimated depth snapshot
            (same convention/shape as TemporalRecord.depth_snapshot_m —
            0.0-is-invalid, float array).
        previous_record: the TemporalRecord to stabilize against, or
            None. Same "caller decides what counts as a real prior"
            convention as compute_temporal_consistency() — this function
            does not know about admission status or chronology itself.
        consistency: this frame's own already-computed TemporalConsistency
            (Phase E3), reused directly (its comparable_count) rather than
            recomputed. None is treated identically to a failing gate.
        agreement_tolerance_m: the exact same tolerance
            compute_temporal_consistency() uses — one definition of
            "agree," shared, never duplicated with drift.
        min_history_confidence: PipelineConfig.
            temporal_stabilization_min_history_confidence — the minimum
            previous_record.confidence for history to be considered at
            all this frame.
        min_comparable_fraction: PipelineConfig.
            temporal_stabilization_min_comparable_fraction — the minimum
            comparable_count / current_depth_snapshot.size for history to
            be considered at all this frame.

    Returns:
        A TemporalStabilization. Never raises on ordinary inputs — a
        gate-failure case is a normal, valid outcome
        (INSUFFICIENT_EVIDENCE), not an error.
    """
    gate_failed = (
        previous_record is None
        or previous_record.depth_snapshot_m is None
        or consistency is None
        or consistency.state not in (TemporalConsistencyState.CONSISTENT, TemporalConsistencyState.CONTRADICTORY)
        or previous_record.confidence < min_history_confidence
        or (consistency.comparable_count / current_depth_snapshot.size) < min_comparable_fraction
    )
    if gate_failed:
        return TemporalStabilization(
            state=TemporalStabilizationState.INSUFFICIENT_EVIDENCE,
            stabilized_depth_m=None,
            eligible_count=0, stabilized_count=0, contradiction_count=0,
            stabilized_fraction=None, contradiction_fraction=None,
        )

    previous_snapshot = previous_record.depth_snapshot_m
    current_valid = current_depth_snapshot > 0.0
    previous_valid = previous_snapshot > 0.0
    both_valid = current_valid & previous_valid
    agrees = both_valid & (np.abs(current_depth_snapshot - previous_snapshot) <= agreement_tolerance_m)
    contradicts = both_valid & ~agrees

    eligible_count = int(current_valid.sum())
    if eligible_count == 0:
        # Defensive: in real pipeline usage consistency.comparable_count
        # always reflects the same arrays being compared here, so
        # comparable_count > 0 (already required by the gate above)
        # guarantees eligible_count > 0 too (both_valid <= current_valid).
        # This guard exists for callers who pass a consistency value
        # computed against different arrays than current_depth_snapshot —
        # degrade cleanly rather than dividing by zero.
        return TemporalStabilization(
            state=TemporalStabilizationState.INSUFFICIENT_EVIDENCE,
            stabilized_depth_m=None,
            eligible_count=0, stabilized_count=0, contradiction_count=0,
            stabilized_fraction=None, contradiction_fraction=None,
        )

    weight = previous_record.confidence
    blended = (current_depth_snapshot + weight * previous_snapshot) / (1.0 + weight)
    stabilized_depth_m = np.where(agrees, blended, current_depth_snapshot).astype(np.float32)

    stabilized_count = int(agrees.sum())
    contradiction_count = int(contradicts.sum())

    state = (
        TemporalStabilizationState.STABILIZED
        if stabilized_count > 0
        else TemporalStabilizationState.CURRENT_ONLY_FALLBACK
    )
    return TemporalStabilization(
        state=state,
        stabilized_depth_m=stabilized_depth_m,
        eligible_count=eligible_count,
        stabilized_count=stabilized_count,
        contradiction_count=contradiction_count,
        stabilized_fraction=stabilized_count / eligible_count,
        contradiction_fraction=contradiction_count / eligible_count,
    )
