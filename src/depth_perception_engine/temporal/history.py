"""
Bounded temporal history — Level 4, Phase E2.

The one canonical chronology component for Level 4: TemporalHistory admits
TemporalRecord values in timestamp order, evicts by count and by age (both
measured against TemporalRecord.timestamp values only — never wall-clock
time), and detects/handles timestamp discontinuities. Every admission
invariant lives here; DepthPerceptionPipeline.process() constructs one
TemporalRecord per frame and calls admit() exactly once — no timestamp
comparison logic exists in pipeline.py itself (see
docs/E2_TEMPORAL_HISTORY_PLAN.md's "Buffer behavior" section).

E2 is infrastructure only. Nothing in this module smooths depth, fuses
geometry, averages frames, computes temporal consistency, classifies
persistence, compensates rotation, integrates IMU data, or reasons about
individual pixels in any way — it only decides which TemporalRecord
values are chronologically admissible and how long to keep them.
"""

import math
from typing import List, Optional, Tuple

from depth_perception_engine.temporal.types import TemporalRecord


class TemporalAdmissionStatus:
    """Plain string constants describing what TemporalHistory.admit() did
    with one incoming TemporalRecord — mirrors obstacles.ThreatAssessor's
    CLEAR/CAUTION/BLOCKED/NO_DATA and geometry.GeometryQuality's existing
    precedent (plain string constants, not a new Enum type) for exactly
    this kind of lightweight categorical outcome in this codebase. Not a
    second competing health system: PipelineHealth (lifecycle) and
    GeometryQuality (per-frame geometry evidence) each answer a different
    question; this answers "was this frame's timestamp admitted to
    temporal chronology," which neither of those contracts covers."""

    ACCEPTED = "ACCEPTED"
    """Admitted normally: chronologically newer than (or the first record
    after) the previous newest, within the configured gap limit."""

    ACCEPTED_NEW_SEQUENCE = "ACCEPTED_NEW_SEQUENCE"
    """Admitted, but only after the existing history was cleared because
    the gap since the previous newest record exceeded
    PipelineConfig.temporal_gap_limit_s — a genuine timestamp
    discontinuity, not ordinary frame-to-frame jitter. The admitted
    record is the sole entry in a freshly started sequence."""

    REJECTED_INVALID_TIMESTAMP = "REJECTED_INVALID_TIMESTAMP"
    """Rejected: record.timestamp is None or not finite (NaN/±Inf).
    History is unchanged."""

    REJECTED_OLDER_TIMESTAMP = "REJECTED_OLDER_TIMESTAMP"
    """Rejected: record.timestamp is strictly older than the current
    newest record's timestamp (this includes an out-of-order arrival).
    History is unchanged — never reordered, never mutated."""

    REJECTED_DUPLICATE_TIMESTAMP = "REJECTED_DUPLICATE_TIMESTAMP"
    """Rejected: record.timestamp exactly equals the current newest
    record's timestamp. The existing record is kept unchanged — never
    replaced, merged, or duplicated."""


class TemporalHistory:
    """A bounded, deterministic chronology of recent TemporalRecord values.

    Two independent eviction bounds apply simultaneously — whichever is
    tighter wins (docs/E2_TEMPORAL_HISTORY_PLAN.md, Decision 1):

        len(history) <= max_records
        (newest.timestamp - record.timestamp) <= max_age_s  for every record

    A new, chronologically-admissible observation is never rejected merely
    because the buffer is already full — eviction always runs *after*
    admission, never as a pre-admission rejection reason.

    Age-based eviction is computed purely from TemporalRecord.timestamp
    values (the newest accepted record's own timestamp is the reference
    point) — time.time()/time.perf_counter() is never called anywhere in
    this module, so this behaves identically for simulated, recorded, or
    real-hardware timestamps.
    """

    def __init__(self, max_records: int, max_age_s: float, gap_limit_s: float) -> None:
        """
        Args:
            max_records: hard cap on retained record count. Must be >= 1.
            max_age_s: maximum age (in TemporalRecord.timestamp units) a
                retained record may have relative to the newest accepted
                record. Must be > 0.
            gap_limit_s: maximum gap (same units) between two
                consecutively accepted records before the older history is
                treated as a discontinuity and cleared. Must be > 0. Not
                required to be <= max_age_s (they answer different
                questions — see docs/E2_TEMPORAL_HISTORY_PLAN.md's
                Decision 7) — a gap_limit_s larger than max_age_s is legal,
                if somewhat redundant with age-based eviction, and is not
                treated as a construction-time error.
        """
        if max_records < 1:
            raise ValueError(f"max_records ({max_records}) must be >= 1.")
        if not (max_age_s > 0.0):
            raise ValueError(f"max_age_s ({max_age_s}) must be > 0.")
        if not (gap_limit_s > 0.0):
            raise ValueError(f"gap_limit_s ({gap_limit_s}) must be > 0.")

        self._max_records = max_records
        self._max_age_s = max_age_s
        self._gap_limit_s = gap_limit_s
        self._records: List[TemporalRecord] = []

    # ------------------------------------------------------------------
    @property
    def records(self) -> Tuple[TemporalRecord, ...]:
        """Oldest-first, read-only snapshot of the current history. A
        fresh tuple every call — mutating it never affects internal
        state."""
        return tuple(self._records)

    @property
    def latest(self) -> Optional[TemporalRecord]:
        """The most recently accepted record, or None if history is
        empty."""
        return self._records[-1] if self._records else None

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Discard all retained records. history is empty afterward —
        the next accepted record starts a brand-new temporal sequence
        with zero historical influence. Called by
        DepthPerceptionPipeline.reset() (docs/E2_TEMPORAL_HISTORY_PLAN.md,
        Decision 3)."""
        self._records.clear()

    # ------------------------------------------------------------------
    def admit(self, record: TemporalRecord) -> str:
        """Attempt to admit `record` into history.

        Every chronology invariant (invalid/older/duplicate timestamp,
        gap-triggered sequence restart, count/age eviction) is decided
        here — the caller does not pre-validate timestamps itself. Never
        inspects `record.confidence`/`record.geometry_quality`: admission
        is a pure function of timestamps only, so a degraded or
        NO_USABLE_GEOMETRY observation is admitted exactly like a healthy
        one, and no degraded/poisoned buffer state can ever result from
        evidence quality alone (docs/E2_TEMPORAL_HISTORY_PLAN.md,
        Decision 12).

        Returns:
            One of TemporalAdmissionStatus's constants.
        """
        ts = record.timestamp
        if ts is None or not math.isfinite(ts):
            return TemporalAdmissionStatus.REJECTED_INVALID_TIMESTAMP

        status = TemporalAdmissionStatus.ACCEPTED
        if self._records:
            newest_ts = self._records[-1].timestamp
            if ts < newest_ts:
                return TemporalAdmissionStatus.REJECTED_OLDER_TIMESTAMP
            if ts == newest_ts:
                return TemporalAdmissionStatus.REJECTED_DUPLICATE_TIMESTAMP
            if (ts - newest_ts) > self._gap_limit_s:
                self._records.clear()
                status = TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE

        self._records.append(record)
        self._evict(reference_timestamp=ts)
        return status

    # ------------------------------------------------------------------
    def _evict(self, reference_timestamp: float) -> None:
        """Enforce both bounds, oldest-first, deterministic. Runs after
        every successful admission."""
        while len(self._records) > self._max_records:
            self._records.pop(0)
        while self._records and (reference_timestamp - self._records[0].timestamp) > self._max_age_s:
            self._records.pop(0)
