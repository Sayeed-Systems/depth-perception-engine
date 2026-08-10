"""
Read-only temporal consistency — Level 4, Phase E3.

Mirrors geometry/geometry_metrics.py's GeometryQuality/
classify_geometry_quality precedent exactly: a small class of plain string
constants (TemporalConsistencyState) plus one pure, stateless, vectorized
function (compute_temporal_consistency) that classifies already-computed
values — no new geometric computation, no reprojection, no re-derivation
of validity, and (unlike PointCloudBuilder/geometry_metrics, and like
classify_geometry_quality) no mutation of anything it's given.

E3 is read-only, structurally, not just by convention: compute_temporal_
consistency() receives two small decimated depth arrays and returns a new,
small, immutable TemporalConsistency value. It holds no state, calls no
cv2/algorithm code, and has no way to write back into DepthPerceptionResult,
TemporalRecord, or TemporalHistory — see docs/E3_IMPLEMENTATION_PLAN.md's
Decision 4 for the full safety-enforcement argument.

E3 performs no temporal filtering, fusion, IMU integration, rotation
compensation, or persistence classification — see docs/E3_IMPLEMENTATION_PLAN.md.
"""

from typing import Optional

import numpy as np

from depth_perception_engine.temporal.types import TemporalConsistency, TemporalRecord


class TemporalConsistencyState:
    """Plain string constants for compute_temporal_consistency()'s
    TemporalConsistency.state — mirrors geometry.GeometryQuality's and
    temporal.TemporalAdmissionStatus's existing precedent (plain string
    constants, not a new Enum type) for this kind of lightweight
    categorical outcome in this codebase. Audited against both before
    being added: neither matches this semantics (GeometryQuality
    classifies a single frame's own evidence coverage; TemporalAdmissionStatus
    reports chronology-admission outcome; this reports agreement between
    two frames), so this is a genuinely new, not a duplicated, concept."""

    CONSISTENT = "CONSISTENT"
    """Comparable evidence exists (comparable_count > 0) and at least
    PipelineConfig.temporal_consistency_min_agreement_fraction of it
    agrees within PipelineConfig.temporal_consistency_agreement_tolerance_m."""

    CONTRADICTORY = "CONTRADICTORY"
    """Comparable evidence exists (comparable_count > 0) but agreement_
    fraction falls below the configured threshold. Reported honestly, not
    acted on — E3 never suppresses, overrides, or removes the current
    frame's own evidence because of this; see docs/E3_IMPLEMENTATION_PLAN.md's
    Decision 4."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    """No comparable evidence exists to judge from — covers both "no prior
    record to compare against" (first frame, first after reset(), or the
    first frame of a gap-triggered new sequence — docs/E3_IMPLEMENTATION_PLAN.md's
    Decision 2) and "a prior record exists but zero pixels are valid in
    both snapshots" (comparable_count == 0). agreement_fraction is always
    None in this state — never 0.0, never confused with CONTRADICTORY's
    real low-but-nonzero-population fraction."""

    NOT_COMPARABLE = "NOT_COMPARABLE"
    """A prior record exists but its depth_snapshot_m has a different
    shape than the current frame's — a genuine structural incompatibility,
    not an inferred one. This library has no motion/pose-estimation
    capability, so it cannot and does not attempt to detect "the camera
    moved, so don't compare" any other way — see docs/E3_IMPLEMENTATION_PLAN.md's
    "Additional rule" section for why inventing such a check would itself
    be inventing alignment."""


def compute_temporal_consistency(
    current_depth_snapshot: np.ndarray,
    previous_record: Optional[TemporalRecord],
    agreement_tolerance_m: float,
    min_agreement_fraction: float,
) -> TemporalConsistency:
    """Compare current_depth_snapshot against previous_record's own
    depth_snapshot_m and classify the result.

    Args:
        current_depth_snapshot: this frame's own decimated depth snapshot
            (same convention/shape as TemporalRecord.depth_snapshot_m —
            0.0-is-invalid, float array).
        previous_record: the TemporalRecord to compare against, or None.
            The caller (DepthPerceptionPipeline.process()) is responsible
            for passing None whenever docs/E3_IMPLEMENTATION_PLAN.md's
            Decision 2 trigger table says no real prior exists for this
            frame — this function does not itself know about admission
            status, chronology, or sequence restarts; it only knows
            "was I given a record to compare against or not."
        agreement_tolerance_m: max absolute depth difference (metres) for
            one comparable pixel to count as agreeing.
        min_agreement_fraction: minimum agreement_fraction for
            TemporalConsistencyState.CONSISTENT; below it is CONTRADICTORY.

    Returns:
        A TemporalConsistency. Never raises on ordinary inputs — an
        incomparable or evidence-free case is a normal, valid classified
        outcome (NOT_COMPARABLE/INSUFFICIENT_EVIDENCE), not an error.
    """
    if previous_record is None or previous_record.depth_snapshot_m is None:
        return TemporalConsistency(
            state=TemporalConsistencyState.INSUFFICIENT_EVIDENCE,
            comparable_count=0, agreeing_count=0, disagreement_count=0,
            agreement_fraction=None,
        )

    previous_snapshot = previous_record.depth_snapshot_m
    if current_depth_snapshot.shape != previous_snapshot.shape:
        return TemporalConsistency(
            state=TemporalConsistencyState.NOT_COMPARABLE,
            comparable_count=0, agreeing_count=0, disagreement_count=0,
            agreement_fraction=None,
        )

    both_valid = (current_depth_snapshot > 0.0) & (previous_snapshot > 0.0)
    comparable_count = int(both_valid.sum())
    if comparable_count == 0:
        return TemporalConsistency(
            state=TemporalConsistencyState.INSUFFICIENT_EVIDENCE,
            comparable_count=0, agreeing_count=0, disagreement_count=0,
            agreement_fraction=None,
        )

    agreeing_mask = both_valid & (np.abs(current_depth_snapshot - previous_snapshot) <= agreement_tolerance_m)
    agreeing_count = int(agreeing_mask.sum())
    disagreement_count = comparable_count - agreeing_count
    agreement_fraction = agreeing_count / comparable_count

    state = (
        TemporalConsistencyState.CONSISTENT
        if agreement_fraction >= min_agreement_fraction
        else TemporalConsistencyState.CONTRADICTORY
    )
    return TemporalConsistency(
        state=state,
        comparable_count=comparable_count,
        agreeing_count=agreeing_count,
        disagreement_count=disagreement_count,
        agreement_fraction=agreement_fraction,
    )
