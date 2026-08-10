> **Superseded for current usage (2026-08-09, Phase E8):** see `docs/LEVEL4_CANONICAL_REFERENCE.md` section 4 ("Temporal consistency") for the current, authoritative description. This document remains as the historical decision record.

# E3 plan — RESOLVED, IMPLEMENTED

**LEVEL 4 E3 — READ-ONLY TEMPORAL CONSISTENCY IMPLEMENTED.**
**NO TEMPORAL SMOOTHING, FUSION, IMU INTEGRATION, ROTATION COMPENSATION, PERSISTENCE, VIO, SLAM, OR LOCALIZATION IMPLEMENTED.**

This document originally listed five open design questions blocking E3. All five have now been resolved by explicit project-architect decision and implemented exactly as decided (no contradiction was found against the frozen E1/E2 repository state). This document is the permanent record of those decisions.

## E3 is about TEMPORAL CONSISTENCY, not filtering or fusion (restated, unchanged)

E3 answers "how consistent is current evidence with the most recent comparable prior evidence" — a read-only judgement — never "here is smoothed/fused evidence." Nothing in E3 replaces, blends, or averages any Level 3 field. **E4's objective, as specified by the architect, is confidence-aware temporal estimation/stabilization** — that is a distinct, not-yet-started future phase; E3 does not anticipate or partially implement it.

## Decision 1 — Consistency definition

Temporal consistency = agreement between CURRENT Level 3 geometry and the most recent PRIOR admissible comparable geometry — exactly one prior frame, never a multi-frame blend or window.

**Substrate chosen: `depth_map`/its own `0.0`-is-invalid convention**, not `geometry`/`geometry_body` (Level 3, Phase E3, gated by `PipelineConfig.enable_geometry`). Reasoning: `depth_map` is the one per-pixel geometric evidence guaranteed to exist on every frame regardless of whether geometry is enabled — the same "always present, usable from stereo alone" argument E2 used to choose `confidence` for `TemporalRecord`. Consistency evaluation must not silently stop working the moment a caller leaves `enable_geometry` at its default `False`.

**Categorical state — `temporal.TemporalConsistencyState`** (new; audited against existing naming first — neither `geometry.GeometryQuality` (HEALTHY/DEGRADED/NO_USABLE_GEOMETRY, a single-frame evidence-coverage judgement) nor `temporal.TemporalAdmissionStatus` (a chronology-admission outcome) matches this semantics, so a new plain-string-constants class was warranted, not a repurposing of either):

- `CONSISTENT` — comparable evidence exists and mostly agrees.
- `CONTRADICTORY` — comparable evidence exists and mostly disagrees.
- `INSUFFICIENT_EVIDENCE` — no comparable evidence exists to judge from (covers both "no prior record" and "prior record exists but zero pixels overlap validly with current").
- `NOT_COMPARABLE` — a prior record exists but is structurally incompatible with the current frame (shape mismatch).

**Explicit metrics — `temporal.TemporalConsistency`** (new, frozen dataclass, mirrors `geometry.GeometryMetrics`'s own "bundle a few related, precisely-defined scalars" precedent): `state`, `comparable_count`, `agreeing_count`, `disagreement_count`, `agreement_fraction`.

**The "score=0 ambiguity" warning, resolved explicitly:** `comparable_count == 0` is checked and short-circuited to `state=INSUFFICIENT_EVIDENCE` with `agreement_fraction=None` *before* any fraction is computed — `agreement_fraction` is never `0/0`, never silently defaulted to `0.0` or `1.0`, and is only ever a real number when `comparable_count > 0`. "No evidence" (`INSUFFICIENT_EVIDENCE`, `agreement_fraction=None`) and "full contradiction" (`CONTRADICTORY`, `agreement_fraction` a real low number) are structurally distinguishable by both `state` and by whether `agreement_fraction` is `None` at all — never conflated.

**Exact math** (`temporal/consistency.py`'s `compute_temporal_consistency()`):

```
both_valid       = (current_snapshot > 0.0) & (previous_snapshot > 0.0)
comparable_count = int(both_valid.sum())
agreeing_mask    = both_valid & (abs(current_snapshot - previous_snapshot) <= agreement_tolerance_m)
agreeing_count   = int(agreeing_mask.sum())
disagreement_count = comparable_count - agreeing_count
agreement_fraction = agreeing_count / comparable_count   # only when comparable_count > 0

state = CONSISTENT   if agreement_fraction >= min_agreement_fraction
        CONTRADICTORY otherwise
```

Two new `PipelineConfig` fields carry the two policy thresholds this math needs (both undocumented-against-any-real-dataset placeholders, same "policy choice, not a physical constant" discipline as `geometry_healthy_min_valid_fraction`): `temporal_consistency_agreement_tolerance_m: float = 0.05` (5 cm — "same surface, sensor noise" tolerance, not derived from any specific rig) and `temporal_consistency_min_agreement_fraction: float = 0.7`.

## Decision 2 — Trigger

E3 is evaluated exactly when `PipelineConfig.enable_temporal` is `True` **and** this frame's own timestamp was `ACCEPTED`/`ACCEPTED_NEW_SEQUENCE` by `TemporalHistory.admit()` (Level 4, Phase E2). Implemented as a single, uniform rule in `DepthPerceptionPipeline.process()`, capturing `previous_latest = self._temporal_history.latest` **before** calling `admit()` (admission mutates the buffer):

```
previous_for_comparison = previous_latest if admission_status == ACCEPTED else None
```

This one line correctly implements every case the trigger table names, with no separate special-case branch needed:

| Case | `admission_status` | `previous_for_comparison` | Result |
|---|---|---|---|
| First frame ever / first after `reset()` | `ACCEPTED` (history was empty) | `previous_latest` is already `None` | `INSUFFICIENT_EVIDENCE` |
| Gap-triggered new sequence | `ACCEPTED_NEW_SEQUENCE` | forced `None` — the pre-gap record must never be compared against, even though it was technically `TemporalHistory`'s `.latest` a moment earlier (`docs/E2_TEMPORAL_HISTORY_PLAN.md`'s Decision 7: "historical geometry from the old sequence must never influence the new sequence") | `INSUFFICIENT_EVIDENCE` |
| Duplicate/older/invalid timestamp (chronology rejection) | `REJECTED_*` | E3 does not run at all | `temporal_consistency` stays `None` — "no comparison," not a categorical value; `temporal_admission_status` already reports why, so this avoids two overlapping status concepts answering the same question |
| Ordinary continuous frame with real prior history | `ACCEPTED`, non-empty history | the real previous record | Real comparison — `CONSISTENT`/`CONTRADICTORY`/`INSUFFICIENT_EVIDENCE` (zero pixel overlap)/`NOT_COMPARABLE` (shape mismatch) |
| Missing `MotionHint` | any of the above | unaffected | `compute_temporal_consistency()` never reads `motion_hint` at all — missing/invalid/present all behave identically |

## Decision 3 — Result location

`DepthPerceptionResult.temporal_consistency: Optional[temporal.TemporalConsistency] = None` — new, additive, appended last (after `temporal_admission_status`). `None` unless `PipelineConfig.enable_temporal` is `True`; even then, `None` specifically on a chronology-rejected frame (see Decision 2's table) — never a mutation of any Level 3 field, never a replacement of `depth_map`/`geometry`/etc. Name chosen after auditing existing sibling fields (`geometry_metrics`, `temporal_admission_status`) — `temporal_consistency` is the smallest name that doesn't collide with, or need disambiguation from, either.

## Decision 4 — Safety enforcement

`compute_temporal_consistency()` is a pure function: it reads two decimated depth snapshots and returns a new, small, immutable `TemporalConsistency` value. It has no side effects, no write access to any Level 3 array, and cannot rewrite depth, rewrite geometry, remove obstacles, promote `UNKNOWN` to `FREE`, or average historical geometry into current geometry — there is no code path through which it could, since it never receives a mutable reference to any Level 3 output and never returns anything larger than four integers/a float/a string. Proven, not just claimed: `tests/test_temporal_consistency.py::TestCurrentEvidenceUnchanged` constructs a pipeline with `enable_temporal=True` across a `CONTRADICTORY`-triggering scene change and asserts `disparity_map`/`depth_map`/`geometry`/`geometry_metrics`/`obstacle_cloud`/`free_space_rays` are identical to an otherwise-identical pipeline with `enable_temporal=False`. E4 (confidence-aware temporal estimation/stabilization) owns any future policy that actually *acts* on a `CONTRADICTORY` verdict — E3 only reports it.

## Decision 5 — TemporalRecord extension

`TemporalRecord` could not compare geometry with its E2-frozen 4 fields (none of `timestamp`/`confidence`/`geometry_quality`/`motion_hint` carries per-pixel data) — extension was required, exactly as anticipated by E2's own memory-decision table ("if a concrete future algorithm demonstrates a real need for more, that is an E3+ decision to make deliberately").

**Added:** `depth_snapshot_m: Optional[np.ndarray] = None` — a decimated (`depth_map[::stride, ::stride]`), `float32` copy of the frame's own `depth_map`, reusing `depth_map`'s existing `0.0`-is-invalid convention exactly (no second validity array needed — one array, not two, keeps this the minimum representation). `Optional`, defaulted `None`, appended after the three existing optional fields — every pre-E3 `TemporalRecord` construction (including the whole of `tests/test_temporal_history.py`'s chronology-only test suite) is unaffected.

**Why this field, and only this field:** it is the one piece of information `compute_temporal_consistency()` structurally requires and nothing else — no `PointCloud`, no `ObstacleCloud`, no `DepthPerceptionResult`. **Not the full-resolution `depth_map`** — a new, dedicated `PipelineConfig.temporal_consistency_sampling_stride: int = 4` (independent of `geometry_sampling_stride`, a different E5-specific knob with no logical connection to this concern) decimates it first, reusing the exact `array[::stride, ::stride]` pattern `build_obstacle_cloud`/`build_free_space_rays` already established.

**Memory cost, bounded and computed, not guessed:** at the project's own real-hardware calibration resolution (320×240, `docs/VALIDATION_REPORT.md`'s E7 addendum) and the default stride of 4, one snapshot is `80×60` `float32` ≈ 19.2 KB; at `temporal_max_records=30` (E2's default), the worst-case total addition is ≈ 576 KB — small next to a single full-resolution `depth_map` (307 KB) and nowhere near "retains full-resolution copies of every intermediate product," the anti-pattern `docs/LEVEL4_ARCHITECTURE.md` section 13 warns against. At a larger synthetic 640×480 resolution (E2-E5's own benchmark precedent), the same math gives ≈ 2.3 MB worst case.

**Only the single most recent record is ever read for comparison** (Decision 1: exactly one prior frame, never a window) — every other buffered record's `depth_snapshot_m` is inert, carried only because `TemporalHistory`'s bounded buffer doesn't distinguish "the record E3 will actually read" from "a record kept for E2's own chronology purposes." This is accepted as the correct trade-off, not reworked into a separate side-channel outside `TemporalRecord`/`TemporalHistory`, because the architect decision explicitly named `TemporalRecord` as the extension point — inventing a separate single-slot cache outside that model would be reinterpreting the decision without a real contradiction to justify it.

**Ownership/safety:** `depth_snapshot_m` is a fresh array built once per `process()` call (`depth_map[::stride, ::stride]` — a NumPy view, but immediately never written to by anything after construction, since `TemporalRecord` is frozen and nothing downstream of `admit()` mutates a stored record's fields) — no aliasing hazard with `DepthPerceptionResult.depth_map` in practice, since neither `TemporalHistory` nor any E3 code ever writes back into it.

## Additional rule — no invented alignment

"Without motion compensation, compare only observations that are legitimately comparable in the existing frame/image domain. If camera motion makes direct comparison invalid, return `NOT_COMPARABLE` rather than inventing alignment."

Implemented literally, within what this repository can actually know: a single `DepthPerceptionPipeline` instance has one fixed calibration/resolution for its entire lifetime, so any two of its own frames already share the same rectified-left-camera pixel grid by construction — pixel-indexed comparison is always in the same *image domain*. This library has zero pose/motion-estimation capability (by design — `docs/LEVEL4_ARCHITECTURE.md` section 3) and therefore no way to *detect* that the physical camera moved between two frames, only a way to *observe its consequence* (the same pixel index now reporting a materially different depth) — which is exactly what `CONTRADICTORY` already reports honestly. Inventing a motion-detection check to redirect that case to `NOT_COMPARABLE` would itself be "inventing alignment" one level removed (silently deciding when to trust pixel-indexed comparison based on guessed motion, with no real motion data to guess from) — the more honest design lets a real scene/camera change surface as `CONTRADICTORY`, and reserves `NOT_COMPARABLE` for genuine, checkable structural incompatibility: `current_snapshot.shape != previous_record.depth_snapshot_m.shape` (defensive — would only trigger if resolution changed mid-pipeline-lifetime, not supported/expected today, but checked rather than allowed to crash or silently misalign).

## Readiness assessment

**E3 was BLOCKED on design decisions; those decisions are now resolved and implemented**, with regression tests proving each one. See `docs/E4_IMPLEMENTATION_PLAN.md` (if/when written) for what confidence-aware temporal estimation/stabilization — E4's stated objective — will need to build on top of this.
