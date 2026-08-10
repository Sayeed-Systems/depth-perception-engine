"""
Level 4 temporal/motion-aware perception — Phases E1-E6.

LEVEL 4 E6 — DETERMINISTIC MOTION-AWARE RELIABILITY ASSESSMENT IMPLEMENTED.
NO GEOMETRY MODIFICATION IMPLEMENTED (E6 is read-only of every upstream value).
NO ADDITIONAL MOTION COMPENSATION IMPLEMENTED (E6 never calls compensate_prior_geometry()).
NO IMU BIAS ESTIMATION IMPLEMENTED.
NO ACCUMULATED ORIENTATION IMPLEMENTED.
NO TRANSLATION COMPENSATION IMPLEMENTED.
NO PERSISTENCE CLASSIFICATION IMPLEMENTED.
NO VIO/SLAM/LOCALIZATION IMPLEMENTED.
NO NEURAL/LEARNED COMPONENT OF ANY KIND.

MotionHint (E1, types.py): a generic, producer-agnostic, short-duration
rotational motion measurement a caller may optionally supply as a
PERCEPTION AID.

TemporalRecord (E2, types.py) / TemporalHistory / TemporalAdmissionStatus
(E2, history.py): a minimal, bounded, deterministic chronology of recent
perception observations — infrastructure only.

TemporalConsistency / TemporalConsistencyState (E3, types.py/consistency.py):
a read-only comparison of current geometry against the single most recent
comparable prior TemporalRecord.

TemporalStabilization / TemporalStabilizationState (E4, types.py/
stabilization.py): a deterministic, confidence-weighted blend of current
and the single most recent comparable prior frame's depth.

RotationCompensationStatus / compute_rotation_compensation() (E5,
rotation_compensation.py): short-window gyro integration (SO(3), via
cv2.Rodrigues) plus prior-geometry rotation — re-expresses the previous
frame's depth snapshot in the current frame's own viewpoint before E3/E4
run. Computes only an ephemeral relative rotation, used once and
discarded.

MotionAwareReliability / MotionAwareReliabilityState (E6, types.py/
reliability.py): a deterministic, explicit (never blended) assessment of
whether E3/E4's temporal results remain trustworthy given this frame's
motion conditions — RELIABLE/DEGRADED/UNRELIABLE/INSUFFICIENT_EVIDENCE,
derived from already-computed E3/E4/E5 signals plus two exactly-defined,
justified thresholds. Read-only of everything it inspects; never modifies
geometry or performs additional motion compensation.

TemporalPersistence / TemporalPersistenceState / TemporalPersistenceCellState
/ TemporalPersistenceTracker (E7, types.py/persistence.py): a deterministic,
bounded, per-cell classification of geometric evidence across time —
NEW/PERSISTENT/DISAPPEARING (state_grid, per cell) plus CLASSIFIED/
UNRELIABLE/INSUFFICIENT_EVIDENCE (state, frame-level gate). Requires
repeated chronological support from E2's bounded history discipline, E3's
agreement tolerance, E5's rotation compensation (reused, not
reimplemented), and E6's reliability verdict (an UNRELIABLE frame can
never create or reinforce persistence). Never fabricates free space from
history — UNKNOWN stays UNKNOWN.

See docs/LEVEL4_ARCHITECTURE.md, docs/LEVEL4_CONTRACTS.md,
docs/E2_TEMPORAL_HISTORY_PLAN.md, docs/E3_IMPLEMENTATION_PLAN.md,
docs/E4_IMPLEMENTATION_PLAN.md, docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md,
docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md, and
docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md for the full contract and
design-decision record.

Deliberately not re-exported from the top-level depth_perception_engine
package — see this subpackage's types.py module docstring for why.
"""

from depth_perception_engine.temporal.consistency import TemporalConsistencyState, compute_temporal_consistency
from depth_perception_engine.temporal.history import TemporalAdmissionStatus, TemporalHistory
from depth_perception_engine.temporal.persistence import (
    TemporalPersistenceCellState,
    TemporalPersistenceState,
    TemporalPersistenceTracker,
)
from depth_perception_engine.temporal.reliability import (
    MotionAwareReliabilityState,
    compute_motion_aware_reliability,
)
from depth_perception_engine.temporal.rotation_compensation import (
    RotationCompensationStatus,
    compensate_prior_geometry_with_payload,
    compute_rotation_compensation,
)
from depth_perception_engine.temporal.stabilization import (
    TemporalStabilizationState,
    compute_temporal_stabilization,
)
from depth_perception_engine.temporal.types import (
    MotionAwareReliability,
    MotionHint,
    TemporalConsistency,
    TemporalPersistence,
    TemporalRecord,
    TemporalStabilization,
)

__all__ = [
    "MotionHint",
    "TemporalRecord",
    "TemporalHistory",
    "TemporalAdmissionStatus",
    "TemporalConsistency",
    "TemporalConsistencyState",
    "compute_temporal_consistency",
    "TemporalStabilization",
    "TemporalStabilizationState",
    "compute_temporal_stabilization",
    "RotationCompensationStatus",
    "compute_rotation_compensation",
    "compensate_prior_geometry_with_payload",
    "MotionAwareReliability",
    "MotionAwareReliabilityState",
    "compute_motion_aware_reliability",
    "TemporalPersistence",
    "TemporalPersistenceState",
    "TemporalPersistenceCellState",
    "TemporalPersistenceTracker",
]
