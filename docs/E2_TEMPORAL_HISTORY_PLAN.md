> **Superseded for current usage (2026-08-09, Phase E8):** see `docs/LEVEL4_CANONICAL_REFERENCE.md` section 3 ("Temporal history") for the current, authoritative description. This document remains as the historical decision record.

# E2 plan — RESOLVED, IMPLEMENTED

**LEVEL 4 E2 — BOUNDED TEMPORAL HISTORY IMPLEMENTED.**
**NO TEMPORAL FILTERING IMPLEMENTED. NO TEMPORAL FUSION IMPLEMENTED. NO IMU INTEGRATION IMPLEMENTED. NO ROTATIONAL COMPENSATION IMPLEMENTED. NO PERSISTENCE CLASSIFICATION IMPLEMENTED.**

This document originally recorded twelve open design questions E1 left for E2 to resolve (`docs/LEVEL4_ARCHITECTURE.md` sections 6, 12, 13). All twelve have now been resolved by explicit project-architect decision, verified against the frozen E1 repository state for contradictions (none found), and implemented exactly as decided. This document is the permanent record of those decisions — it is no longer a list of open questions.

## Where E1 left off (unchanged, for context)

`temporal.MotionHint` (`docs/LEVEL4_CONTRACTS.md`) is the frozen input contract. `StereoObservation.motion_hint` was a reserved, unconsumed field at E1 — **as of E2, it is consumed**: `DepthPerceptionPipeline.process_observation()` now forwards it into the temporal-admission path (see "Motion hints" below). `DepthPerceptionPipeline.reset()`/`.health()` already existed and are the lifecycle hooks the E2 history buffer integrates with, exactly as anticipated.

## History ownership (unchanged, confirmed, now implemented)

`DepthPerceptionPipeline` owns one optional `temporal.TemporalHistory` instance, constructed in `__init__` only when `PipelineConfig.enable_temporal` is `True` (mirroring `PointCloudBuilder`'s "only construct if enabled" discipline). The pipeline does not manipulate raw history directly — it constructs a `temporal.TemporalRecord` per accepted frame and delegates admission entirely to `TemporalHistory.admit()`. Exposed read-only via `DepthPerceptionPipeline.temporal_history` (mirrors the existing `.config`/`.calibration` properties); `None` when disabled.

## The twelve decisions, as resolved

### Decision 1 — Maximum history semantics

**Both** a record-count bound and a time-window bound apply; the tighter constraint wins at eviction time. On every accepted observation: append the newest record first, then evict the oldest records until *both* `len(history) <= temporal_max_records` and every retained record's age relative to the newest (`newest.timestamp - record.timestamp`) is `<= temporal_max_age_s`. A new, chronologically-valid observation is never rejected merely because the buffer is full — eviction always happens after admission, never as a pre-admission rejection reason.

**Implemented:** `PipelineConfig.temporal_max_records: int = 30`, `PipelineConfig.temporal_max_age_s: float = 1.0`. Defaults are conservative and audited against this project's own measured numbers, not arbitrarily chosen: `docs/VALIDATION_REPORT.md`'s E7 real-hardware run measured ~42 FPS (23.84 ms mean latency) at the project's actual calibration resolution — `temporal_max_age_s=1.0` s is deliberately *not* derived from an assumed frame rate (this library performs no FPS estimation and must not start doing so for a config default); it is a conservative "recent past" window, comfortably shorter than would ever risk unbounded growth and comfortably longer than any single-frame-to-next-frame interval at any reasonable processing rate this project has actually measured. `temporal_max_records=30` is a hard safety cap independent of timestamp semantics entirely (protects against a caller feeding unexpectedly dense/rapid timestamps where the time-window bound alone might retain far more records than intended) — not tuned against any dataset, a policy default in the same spirit as `PipelineConfig.geometry_healthy_min_valid_fraction`'s own documented "policy choice, not a physical constant."

### Decision 2 — Initialization state

Confirmed, implemented exactly as specified: immediately after `DepthPerceptionPipeline.__init__`, `temporal_history.records == ()` and `temporal_history.latest is None`. No fake initial observation, no identity geometry, no zero-depth observation, no fabricated timestamp. The first accepted observation establishes the start of a temporal sequence.

### Decision 3 — Reset semantics

`DepthPerceptionPipeline.reset()` now also calls `self._temporal_history.clear()` when temporal history is enabled, in addition to its existing `ThreatAssessor` state rebuild. `reset() => history is empty` is enforced unconditionally. Ordinary Level 3 processing is unaffected — `reset()`'s existing behavior (rebuilding `ThreatAssessor`, resetting `frames_processed`/`last_confidence`/`last_processing_time_ms`, leaving calibration/config/rectification maps untouched) is unchanged. Regression test: `tests/test_temporal_history.py::TestPipelineIntegration::test_reset_clears_temporal_history`.

### Decision 4 — Older timestamps

If an incoming observation's timestamp is strictly less than the newest accepted record's timestamp, `TemporalHistory.admit()` returns `TemporalAdmissionStatus.REJECTED_OLDER_TIMESTAMP` — the observation is **not** inserted, existing history is **not** reordered or mutated. The current frame's ordinary Level 3 result (`disparity_map`, `depth_map`, `traversability_mask`, `obstacles`, `confidence`, `geometry`/`geometry_body`/etc.) is completely unaffected — only temporal-history admission fails, never single-frame perception. The rejection is represented via the new `DepthPerceptionResult.temporal_admission_status: Optional[str]` field, using `temporal.TemporalAdmissionStatus`'s plain string constants — the same "plain string constants, not a new Enum" convention `geometry.GeometryQuality`/`obstacles.ThreatAssessor.CLEAR/CAUTION/BLOCKED/NO_DATA` already established. This is **not** a second competing health system: `PipelineHealth` remains lifecycle-only and untouched, `GeometryQuality` remains a geometry-evidence classification and untouched — `TemporalAdmissionStatus` answers a third, disjoint question ("did this frame's timestamp get admitted to temporal chronology") that neither existing contract answers, so neither "supports it" in the sense Decision 4 anticipated (no conflict, and no field was repurposed to force-fit this).

### Decision 5 — Duplicate timestamps

If an incoming observation's timestamp exactly equals the newest accepted record's timestamp, `admit()` returns `TemporalAdmissionStatus.REJECTED_DUPLICATE_TIMESTAMP`. The existing record is kept unchanged — no replace, no merge, no dual storage, no timestamp perturbation. The incoming frame's Level 3 result is still returned to the caller normally.

### Decision 6 — Out-of-order arrival

Handled by the identical `REJECTED_OLDER_TIMESTAMP` path as Decision 4 (an out-of-order arrival *is* an older-than-newest arrival, by definition, once at least one record exists) — it invalidates only that one incoming observation for temporal-history purposes. Existing history is never cleared, reordered, or corrupted by a single out-of-order arrival. No fault counter, no repeated-violation tracking, no complex diagnostic state was added — E1 froze no such counter, and this phase does not invent one (Decision 6's own instruction).

### Decision 7 — Large timestamp gaps

If `incoming.timestamp - newest.timestamp > temporal_gap_limit_s`, the old history is cleared *before* the incoming observation is admitted, and the incoming observation (assuming it is otherwise admissible — i.e. not itself an invalid timestamp, which is impossible here since it's already been proven newer than a real previous timestamp) starts a fresh sequence as the buffer's sole record. `admit()` returns the distinct status `TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE` (not plain `ACCEPTED`) so this discontinuity is honestly distinguishable from ordinary continuous admission. For gaps `<= temporal_gap_limit_s`, chronology is preserved and the observation is appended normally (`ACCEPTED`). **Implemented:** `PipelineConfig.temporal_gap_limit_s: float = 0.5` — deliberately smaller than, and independent of, `temporal_max_age_s`: they answer different questions (`temporal_gap_limit_s`: "is this arrival continuous with the last one?"; `temporal_max_age_s`: "is this retained record still recent enough to keep?"). No confidence-decay-by-`dt` behavior was implemented — explicitly deferred, per this phase's own instruction, to a later Level 4 phase. **Degraded geometry alone never clears history** — only an actual timestamp discontinuity does; `admit()`'s gap check is computed purely from `TemporalRecord.timestamp` values and never inspects `TemporalRecord.geometry_quality`/`confidence` at all.

### Decision 8 — Missing observations / missing motion hint

A valid Level 3 observation with no motion hint (`motion_hint=None`) is still a fully valid temporal observation and is admitted normally — `TemporalRecord.motion_hint` is simply `None` on that record. Missing IMU/motion hint never blocks temporal-history admission and never blocks Level 3 perception; later rotational compensation (a future phase) will simply have no motion data for that interval. If Level 3 geometry itself is degraded, its real quality is recorded honestly via `TemporalRecord.geometry_quality` (see Decision 9) — history never silently upgrades degraded evidence.

### Decision 9 — Invalid / NO_USABLE_GEOMETRY observations

An observation at a valid, chronologically-admissible timestamp that produced `geometry.GeometryQuality.NO_USABLE_GEOMETRY` (or `DEGRADED`) is **still admitted and represented** in history — never silently omitted. `TemporalRecord.geometry_quality: Optional[str]` distinguishes exactly the two cases Decision 9 named: **(A) no observation occurred at all** — no `TemporalRecord` exists for that timestamp because chronology admission itself failed (Decisions 4/5) — from **(B) an observation occurred at timestamp T but produced no usable geometric evidence** — a `TemporalRecord` exists, with `geometry_quality == GeometryQuality.NO_USABLE_GEOMETRY`. `geometry_quality` is computed by reusing the already-frozen, already-tested `geometry.classify_geometry_quality()` (Level 3, Phase E6) against the frame's own `geometry_metrics` and the pipeline's own configured thresholds — no new classification algorithm was written. When geometry isn't enabled/computed at all this call (`PipelineConfig.enable_geometry=False`, or no `geometry_metrics` for any other reason), `geometry_quality` is `None` — a fourth, distinct state from all three `GeometryQuality` values, meaning "not computed this call," never conflated with "computed and found unusable." No geometry is fabricated; no `UNKNOWN` pixel is ever converted to `FREE` by this recording (E2 performs no per-pixel reasoning of any kind — `TemporalRecord` never carries pixel-level data). No giant arrays are stored for any record, valid or invalid — see the memory analysis below.

### Decision 10 — Configuration changes mid-run

**Not applicable, confirmed.** Level 4 temporal configuration (`enable_temporal`, `temporal_max_records`, `temporal_max_age_s`, `temporal_gap_limit_s`) is construction-time-only, exactly like every other `PipelineConfig` field — there is no public setter, and E2 adds no dynamic reconfiguration mechanism. The preferred future architecture, if runtime reconfiguration is ever needed, remains transactional (stop temporal processing → reset temporal state → apply new configuration → begin new sequence) — named here as the intended future shape, not built now.

### Decision 11 — Temporal state after reset

**No temporal state survives `reset()`**, confirmed and implemented (see Decision 3). At E2, "all Level 4 temporal state" means exactly one thing — the `TemporalHistory` buffer — since no temporal confidence state, persistence state, alignment state, or other algorithm state exists yet. `reset()`'s implementation clears exactly what exists today; it is not written to anticipate future state that doesn't exist yet.

### Decision 12 — Degradation and recovery

Confirmed, and true by construction rather than by any special-cased recovery logic: `TemporalHistory.admit()`'s chronology decision (accept/reject-older/reject-duplicate/gap-restart) is a pure function of `TemporalRecord.timestamp` values alone — it never inspects `confidence` or `geometry_quality`. A `GOOD, INVALID, GOOD` (or longer) sequence with monotonically increasing, gap-free timestamps is admitted in full, every record retained (subject only to the ordinary count/age bounds) — there is no "degraded" or "poisoned" buffer state to recover from, because degradation quality never participates in the admission decision at all. Only an actual chronology violation (older/duplicate timestamp, or a gap exceeding `temporal_gap_limit_s`) changes admission behavior — never a quality signal. Regression test: `tests/test_temporal_history.py::TestDegradationRecovery`.

## Additional authoritative memory decision — implemented

**`TemporalRecord` is not, and does not contain, a `DepthPerceptionResult`.** It is a new, deliberately minimal, frozen dataclass (`temporal/types.py`) with exactly four fields:

| Field | Type | Why temporal history needs it |
|---|---|---|
| `timestamp` | `Optional[float]` | The entire ordering/eviction/gap-detection key — without it, none of Decisions 1, 4-7 (bounding, chronology, discontinuity detection) are possible. `Optional` (not required) so `TemporalHistory.admit()` can centralize the "missing/invalid timestamp" rejection itself (Decision-driven: no scattered validation elsewhere — see "Buffer behavior" below) rather than requiring the pipeline to pre-validate before constructing a record. |
| `confidence` | `float` | The one per-frame evidence-quality signal **guaranteed to exist on every `DepthPerceptionResult` regardless of whether geometry is enabled** (`fusion.result_builder.aggregate_confidence`, Level 0-2). Without it, a caller who never enables geometry (the common case — `mp01_perception` does not enable it today) would have a temporal record carrying zero evidence-quality information at all, defeating Decision 8/9's intent to "record its real quality/status" honestly. |
| `geometry_quality` | `Optional[str]` | `geometry.GeometryQuality.HEALTHY`/`DEGRADED`/`NO_USABLE_GEOMETRY`, or `None` if not computed this call — see Decision 9. Reuses the existing frozen E6 classification instead of inventing a parallel one. |
| `motion_hint` | `Optional[temporal.MotionHint]` | Preserves the E1-frozen association per Decision 8/section 7 of this phase's instructions. Retaining the full object (not just a boolean) is deliberately cheap and safe: `MotionHint` carries exactly one `(3,)` float array plus two scalars and a short string — nowhere near the "expensive full-resolution product" (disparity/depth/geometry arrays) this phase's memory decision warns against, so there is no proportionate reason to store anything less than the whole value a genuinely future E5 consumer would need to read. |

No disparity array, depth array, mask, `PointCloud`, `ObstacleCloud`, or `FreeSpaceRays` is ever retained by a `TemporalRecord` — E2 does not store, and has no proven need to store, any of Level 3's expensive per-pixel products. If a concrete future algorithm (E3+) demonstrates a real need for more, that is an E3+ decision to make deliberately, not something E2 anticipates speculatively.

## Buffer behavior — implemented

One canonical component, `temporal.TemporalHistory` (`temporal/history.py`) — not a competing second manager. It owns `admit(record) -> TemporalAdmissionStatus`, `records` (read-only tuple view), `latest` (`Optional[TemporalRecord]`), `__len__`, and `clear()`. Every timestamp-chronology invariant (older/duplicate/gap/count/age) is enforced inside `TemporalHistory.admit()`/`_evict()` — `DepthPerceptionPipeline.process()` only constructs a `TemporalRecord` from already-computed frame values and calls `.admit()` once; no timestamp comparison logic exists in `pipeline.py` itself.

## Time-window eviction — implemented

Age-based eviction is computed as `newest_accepted_timestamp - record.timestamp`, using only `TemporalRecord.timestamp` values — `time.time()`/`time.perf_counter()` is never called anywhere in `temporal/history.py`. This holds for simulated, recorded, or real-hardware timestamps identically, exactly as required.

## Motion hints — implemented, still no algorithm

`StereoObservation.motion_hint` is now forwarded by `DepthPerceptionPipeline.process_observation()` into a new, additive, defaulted `process(..., motion_hint: Optional[MotionHint] = None)` keyword parameter, and attached to the resulting `TemporalRecord` if one is admitted. This is pure association/bookkeeping — E2 does not read `angular_velocity_rad_s`, does not integrate it, does not compute a rotation, and does not branch on whether a hint is present, valid, or which producer supplied it. No simulated-IMU generation was added. `tests/test_level4_architecture_guards.py` (E1, unmodified) continues to guard against source-specific branching.

## Readiness assessment

**E2 was BLOCKED on design decisions; those decisions are now resolved and implemented.** All twelve questions above have concrete, implemented answers with regression tests. See `docs/E3_IMPLEMENTATION_PLAN.md` for what comes next.
