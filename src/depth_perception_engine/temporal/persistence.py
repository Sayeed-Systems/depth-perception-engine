"""
Deterministic, per-cell temporal-persistence classification — Level 4, Phase E7.

Classifies geometric evidence, per cell of the same decimated grid E3/E4
already use, into NEW / PERSISTENT / DISAPPEARING (state_grid codes) plus
one frame-level gate state (CLASSIFIED / UNRELIABLE / INSUFFICIENT_EVIDENCE)
that governs whether/how the grid updated this call — see
temporal.types.TemporalPersistence's own docstring for the full field
contract and docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md for the complete
decision record.

E7 is a stateful, bounded collaborator (TemporalPersistenceTracker),
mirroring obstacles.ThreatAssessor's own cross-frame EMA/debounce state
precedent — NOT a pure function like E3/E4/E6, because persistence
fundamentally requires memory beyond what a single previous record can
represent. That memory is deliberately small and FIXED SIZE (four arrays,
each exactly the shape of one decimated grid — never growing with frame
count, never storing a full TemporalRecord or DepthPerceptionResult): this
is not a second, competing history buffer alongside temporal.TemporalHistory
(E2) — TemporalHistory remains the one chronology-admission buffer; this
tracker holds only this module's own per-cell support bookkeeping, and
reuses temporal.TemporalHistory/TemporalRecord/MotionAwareReliability
exactly as E2/E3/E4/E5/E6 already produce them, never re-deriving any of
their own judgements.

Rotation is handled by reusing E5's own compensate_prior_geometry_with_payload()
(temporal/rotation_compensation.py) to warp the tracker's own per-cell
state through the identical rotation E5 already computes for
TemporalRecord.depth_snapshot_m — never a second, independent
motion-compensation implementation (see that function's own docstring).

E7 performs no semantics, neural/learned inference, object recognition,
planning, trajectory prediction, VIO, SLAM, or localization — it only
counts repeated agreement/absence of already-computed decimated depth
evidence against its own tracked state.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from depth_perception_engine.temporal.reliability import MotionAwareReliabilityState
from depth_perception_engine.temporal.rotation_compensation import (
    compensate_prior_geometry_with_payload,
    integrate_angular_velocity,
    select_motion_hint_samples,
)
from depth_perception_engine.temporal.types import MotionHint, TemporalPersistence


class TemporalPersistenceCellState:
    """Plain int8 codes for temporal.TemporalPersistence.state_grid — a
    compact per-cell array, so this is a small integer code class rather
    than the plain-string-constant convention every frame-level Level 4
    result uses (TemporalAdmissionStatus/TemporalConsistencyState/etc.) —
    a (H_dec, W_dec) array of Python strings would defeat the point of a
    dense NumPy classification grid. See temporal.types.TemporalPersistence's
    own docstring for the precise trigger for each code."""

    NO_EVIDENCE = 0
    """No current occupied evidence at this cell AND no tracked history
    (support_count_grid == 0) — the overwhelming background case. NEVER
    means "free" — see TemporalPersistence's own UNKNOWN != FREE
    paragraph. Also the value every expired cell reverts to (Rule 5)."""

    NEW = 1
    """Occupied this frame (current decimated depth > 0.0) AND this
    cell's own support_count_grid entry is below
    PipelineConfig.persistence_min_support_count — covers both a
    genuinely first-ever observation AND a strong current observation
    that CONTRADICTED previously-tracked history (support resets to 1 in
    either case, per Rule 3: history must never suppress or delay new
    occupied evidence)."""

    PERSISTENT = 2
    """support_count_grid >= PipelineConfig.persistence_min_support_count
    AND either (a) occupied and agreeing this frame, or (b) within a
    dropout grace window (PipelineConfig.persistence_max_dropout_frames)
    of an absence — Rule 4: one dropout frame must not immediately erase
    previously persistent evidence."""

    DISAPPEARING = 3
    """Was tracked (support_count_grid > 0) but has now been absent
    (current decimated depth <= 0.0) for longer than
    PipelineConfig.persistence_max_dropout_frames, and not yet past
    PipelineConfig.persistence_expiration_absence_frames (that boundary
    is EXPIRED, which reverts the cell to NO_EVIDENCE rather than
    remaining DISAPPEARING forever) — repeated absence, trending toward
    removal, reported honestly rather than silently held or erased."""


class TemporalPersistenceState:
    """Plain string constants for temporal.TemporalPersistence.state —
    the frame-level gate outcome, mirroring every other Level 4 phase's
    plain-string-constant precedent. See temporal.types.TemporalPersistence's
    own docstring for the precise trigger for each."""

    CLASSIFIED = "CLASSIFIED"
    """This frame's own current evidence was used to update the tracker —
    state_grid/support_count_grid/age_s_grid all reflect a fresh
    classification."""

    UNRELIABLE = "UNRELIABLE"
    """This frame's own MotionAwareReliability.state (Phase E6) was
    UNRELIABLE — per Rule 7, this frame's evidence must not create or
    reinforce persistence. The tracker is not updated at all; the exposed
    grids are the tracker's own unchanged snapshot from the most recent
    CLASSIFIED frame."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    """A genuine structural incompatibility between this frame's
    decimated grid shape and the shape the tracker has been accumulating
    against — mirrors TemporalConsistencyState.NOT_COMPARABLE's own rare,
    defensive trigger. Unreachable in ordinary single-pipeline-instance
    operation."""


class TemporalPersistenceTracker:
    """Bounded, deterministic, per-cell chronological-support tracker.

    Fixed-size state (four (H_dec, W_dec) arrays, shape fixed on first
    update() call and never resized thereafter) — no growth with frame
    count, no storage of any TemporalRecord/DepthPerceptionResult. One
    instance is owned by DepthPerceptionPipeline, constructed once,
    mirroring how it already owns temporal.TemporalHistory and
    obstacles.ThreatAssessor.
    """

    def __init__(
        self,
        min_support_count: int,
        max_dropout_frames: int,
        expiration_absence_frames: int,
        agreement_tolerance_m: float,
    ) -> None:
        """
        Args:
            min_support_count: PipelineConfig.persistence_min_support_count
                — minimum support_count_grid value for state_grid to read
                PERSISTENT rather than NEW. Must be >= 2 (Rule 2: one
                observation can never become PERSISTENT) — validated by
                PipelineConfig.__post_init__, not re-validated here (this
                class trusts its caller, matching TemporalHistory's own
                discipline of validating once at the PipelineConfig
                boundary).
            max_dropout_frames: PipelineConfig.persistence_max_dropout_frames
                — consecutive absent frames a previously-PERSISTENT cell
                may tolerate before reading DISAPPEARING instead.
            expiration_absence_frames: PipelineConfig.
                persistence_expiration_absence_frames — consecutive absent
                frames after which a cell's tracked state is fully
                cleared (reverts to NO_EVIDENCE). Must exceed
                max_dropout_frames (validated by PipelineConfig).
            agreement_tolerance_m: the same
                PipelineConfig.temporal_consistency_agreement_tolerance_m
                E3/E4 already use — one shared definition of "agrees,"
                never a second, independent tolerance concept.
        """
        self._min_support_count = min_support_count
        self._max_dropout_frames = max_dropout_frames
        self._expiration_absence_frames = expiration_absence_frames
        self._agreement_tolerance_m = agreement_tolerance_m

        self._grid_shape: Optional[Tuple[int, int]] = None
        self._tracked_depth: Optional[np.ndarray] = None
        self._support_count: Optional[np.ndarray] = None
        self._absence_streak: Optional[np.ndarray] = None
        self._first_observed_timestamp: Optional[np.ndarray] = None
        self._last_state_grid: Optional[np.ndarray] = None
        self._last_update_timestamp: Optional[float] = None

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Discard all tracked per-cell state. The next update() call
        starts a brand-new, empty grid (shape re-derived from that call's
        own current_depth_snapshot) — called by
        DepthPerceptionPipeline.reset() (Rule: reset clears persistence
        state) and internally by the pipeline on a temporal-history
        gap-restart (TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE — Rule:
        a large timestamp gap starts a fresh persistence sequence too,
        matching TemporalHistory's own "complete amnesia" behavior)."""
        self._grid_shape = None
        self._tracked_depth = None
        self._support_count = None
        self._absence_streak = None
        self._first_observed_timestamp = None
        self._last_state_grid = None
        self._last_update_timestamp = None

    # ------------------------------------------------------------------
    def update(
        self,
        current_depth_snapshot: np.ndarray,
        current_timestamp: float,
        motion_aware_reliability_state: Optional[str],
        motion_hints: Optional[Sequence[MotionHint]],
        enable_rotation_compensation: bool,
        stride: int,
        focal_length_px: float,
        principal_point_px: Tuple[float, float],
    ) -> TemporalPersistence:
        """Classify this frame's per-cell persistence and update tracked
        state accordingly.

        Args:
            current_depth_snapshot: this frame's own decimated depth
                snapshot — same array/convention as
                TemporalRecord.depth_snapshot_m (0.0-is-invalid).
            current_timestamp: this frame's own timestamp.
            motion_aware_reliability_state: this frame's own
                temporal.MotionAwareReliability.state (Phase E6), or None
                if E6 did not run. UNRELIABLE triggers Rule 7 (see
                TemporalPersistenceState.UNRELIABLE); every other value
                (including None, RELIABLE, DEGRADED, and E6's own
                INSUFFICIENT_EVIDENCE — a first frame naturally has no E3
                comparison either, which is exactly the ordinary "NEW"
                case, not a reason to withhold E7's own independent
                per-cell bookkeeping) permits a normal update.
            motion_hints: the same bounded MotionHint sequence passed to
                process() this call, forwarded to E5's own
                select_motion_hint_samples()/integrate_angular_velocity()
                exactly as E5/E6 already use them.
            enable_rotation_compensation: PipelineConfig.
                enable_rotation_compensation — when False, the tracker's
                own state is compared against the current grid with zero
                rotation warp (identity), same "E5 disabled is a
                legitimate configuration" precedent E6 already
                established.
            stride, focal_length_px, principal_point_px: same rectified
                intrinsics/decimation E5 already uses for
                compensate_prior_geometry().

        Returns:
            A TemporalPersistence. Never raises on ordinary inputs.
        """
        if self._grid_shape is None:
            self._grid_shape = current_depth_snapshot.shape
            self._tracked_depth = np.zeros(self._grid_shape, dtype=np.float32)
            self._support_count = np.zeros(self._grid_shape, dtype=np.int32)
            self._absence_streak = np.zeros(self._grid_shape, dtype=np.int32)
            self._first_observed_timestamp = np.zeros(self._grid_shape, dtype=np.float64)

        if current_depth_snapshot.shape != self._grid_shape:
            return _empty_result(TemporalPersistenceState.INSUFFICIENT_EVIDENCE)

        if motion_aware_reliability_state == MotionAwareReliabilityState.UNRELIABLE:
            return self._unreliable_result(current_timestamp)

        warped_depth, warped_support, warped_absence, warped_first_ts = self._warp_tracked_state(
            current_timestamp, motion_hints, enable_rotation_compensation, stride, focal_length_px, principal_point_px,
        )

        curr_occ = current_depth_snapshot > 0.0
        curr_depth = current_depth_snapshot.astype(np.float32)
        prev_tracked = warped_support > 0

        agrees = prev_tracked & curr_occ & (np.abs(curr_depth - warped_depth) <= self._agreement_tolerance_m)
        contradicts = prev_tracked & curr_occ & ~agrees
        newly_seen = curr_occ & ~prev_tracked
        dropout = prev_tracked & ~curr_occ
        fresh = contradicts | newly_seen

        new_support = np.zeros(self._grid_shape, dtype=np.int32)
        new_support[agrees] = warped_support[agrees] + 1
        new_support[fresh] = 1
        new_support[dropout] = warped_support[dropout]

        new_absence = np.zeros(self._grid_shape, dtype=np.int32)
        new_absence[dropout] = warped_absence[dropout] + 1

        new_tracked_depth = np.where(agrees | fresh, curr_depth, warped_depth).astype(np.float32)
        new_first_ts = np.where(fresh, current_timestamp, warped_first_ts).astype(np.float64)

        expired_mask = dropout & (new_absence > self._expiration_absence_frames)
        new_support[expired_mask] = 0
        new_absence[expired_mask] = 0
        new_tracked_depth[expired_mask] = 0.0
        new_first_ts[expired_mask] = 0.0
        expired_count = int(expired_mask.sum())

        state_grid = np.zeros(self._grid_shape, dtype=np.int8)
        state_grid[curr_occ & (new_support < self._min_support_count)] = TemporalPersistenceCellState.NEW
        state_grid[curr_occ & (new_support >= self._min_support_count)] = TemporalPersistenceCellState.PERSISTENT
        persistent_grace = dropout & (new_support >= self._min_support_count) & (new_absence <= self._max_dropout_frames) & ~expired_mask
        state_grid[persistent_grace] = TemporalPersistenceCellState.PERSISTENT
        disappearing = dropout & (new_absence > self._max_dropout_frames) & ~expired_mask
        state_grid[disappearing] = TemporalPersistenceCellState.DISAPPEARING

        age_s_grid = np.where(new_support > 0, current_timestamp - new_first_ts, 0.0).astype(np.float32)

        self._tracked_depth = new_tracked_depth
        self._support_count = new_support
        self._absence_streak = new_absence
        self._first_observed_timestamp = new_first_ts
        self._last_state_grid = state_grid
        self._last_update_timestamp = current_timestamp

        return _result_from_grid(
            TemporalPersistenceState.CLASSIFIED, state_grid, new_support, age_s_grid, expired_count,
        )

    # ------------------------------------------------------------------
    def _warp_tracked_state(
        self, current_timestamp, motion_hints, enable_rotation_compensation, stride, focal_length_px, principal_point_px,
    ):
        """Re-express the tracker's own per-cell state in the current
        frame's grid via E5's exact rotation-compensation function — see
        compensate_prior_geometry_with_payload()'s own docstring for why
        this is reuse, not a second motion-compensation path. Falls back
        to the tracker's own unwarped arrays (identity) whenever
        rotation compensation is disabled, has never been updated before,
        or no admissible motion samples exist for the interval — exactly
        E5's own NOT_APPLIED fallback discipline."""
        has_tracked = bool((self._support_count > 0).any())
        if has_tracked and enable_rotation_compensation and self._last_update_timestamp is not None:
            accepted = select_motion_hint_samples(motion_hints, self._last_update_timestamp, current_timestamp)
            if accepted:
                delta_r = integrate_angular_velocity(accepted, self._last_update_timestamp)
                warped_depth, (warped_support_f, warped_absence_f, warped_first_ts) = compensate_prior_geometry_with_payload(
                    self._tracked_depth,
                    [
                        self._support_count.astype(np.float64),
                        self._absence_streak.astype(np.float64),
                        self._first_observed_timestamp,
                    ],
                    delta_r, stride, focal_length_px, principal_point_px,
                )
                warped_support = np.round(warped_support_f).astype(np.int32)
                warped_absence = np.round(warped_absence_f).astype(np.int32)
                return warped_depth, warped_support, warped_absence, warped_first_ts.astype(np.float64)

        return self._tracked_depth, self._support_count, self._absence_streak, self._first_observed_timestamp

    # ------------------------------------------------------------------
    def _unreliable_result(self, current_timestamp: float) -> TemporalPersistence:
        if self._last_state_grid is None:
            return _empty_result(TemporalPersistenceState.UNRELIABLE)

        age_s_grid = np.where(
            self._support_count > 0, current_timestamp - self._first_observed_timestamp, 0.0,
        ).astype(np.float32)
        result = _result_from_grid(
            TemporalPersistenceState.UNRELIABLE,
            self._last_state_grid, self._support_count, age_s_grid, expired_count=0,
        )
        return result


# ----------------------------------------------------------------------
def _result_from_grid(
    state: str, state_grid: np.ndarray, support_count_grid: np.ndarray, age_s_grid: np.ndarray, expired_count: int,
) -> TemporalPersistence:
    new_count = int((state_grid == TemporalPersistenceCellState.NEW).sum())
    persistent_count = int((state_grid == TemporalPersistenceCellState.PERSISTENT).sum())
    disappearing_count = int((state_grid == TemporalPersistenceCellState.DISAPPEARING).sum())
    eligible_count = new_count + persistent_count + disappearing_count
    persistent_fraction = (persistent_count / eligible_count) if eligible_count > 0 else None

    return TemporalPersistence(
        state=state,
        state_grid=state_grid.copy(),
        support_count_grid=support_count_grid.copy(),
        age_s_grid=age_s_grid.copy(),
        new_count=new_count,
        persistent_count=persistent_count,
        disappearing_count=disappearing_count,
        expired_count=expired_count,
        eligible_count=eligible_count,
        persistent_fraction=persistent_fraction,
    )


def _empty_result(state: str) -> TemporalPersistence:
    return TemporalPersistence(
        state=state,
        state_grid=None,
        support_count_grid=None,
        age_s_grid=None,
        new_count=0,
        persistent_count=0,
        disappearing_count=0,
        expired_count=0,
        eligible_count=0,
        persistent_fraction=None,
    )
