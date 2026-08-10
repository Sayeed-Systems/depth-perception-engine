> **Superseded for current usage (2026-08-09, Phase E8):** see `docs/LEVEL4_CANONICAL_REFERENCE.md` section 5 ("Temporal stabilization") for the current, authoritative description. This document remains as the historical decision record.

# E4 plan — RESOLVED, IMPLEMENTED

**LEVEL 4 E4 — DETERMINISTIC, CONFIDENCE-AWARE TEMPORAL STABILIZATION IMPLEMENTED.**
**NO IMU INTEGRATION. NO MOTION WARPING. NO EGO-MOTION ESTIMATION. NO ROTATION/TRANSLATION COMPENSATION. NO NEURAL NETWORKS, LEARNED WEIGHTS, OR OPAQUE BLENDED SCORES. NO E5+ IMPLEMENTED.**

Objective (as specified by the architect): deterministic, confidence-aware temporal stabilization. This document records the ten resolved decisions and the exact math they produced — verified against the frozen E1-E3 repository state before implementation began; no contradiction was found.

## Decision 1 — What E4 stabilizes

`DepthPerceptionResult` keeps the raw current observation exactly as-is (`depth_map`, `geometry`, `geometry_body`, etc. — untouched, byte-identical whether or not stabilization runs) **plus** one new, additive, `Optional`, default-`None` field: `temporal_stabilization: Optional[temporal.TemporalStabilization]`. Nothing E4 computes ever overwrites, mutates, or replaces a Level 3 field — the same raw-vs-temporal authority rule E1 froze and E2/E3 already upheld.

**Resolution granularity, decided and stated explicitly (not left implicit):** the stabilized estimate is computed over the same decimated grid as `TemporalRecord.depth_snapshot_m` (Level 4, Phase E3) — **not** full pixel resolution. Producing a full-resolution stabilized map would require retaining full-resolution history, which Decision 5 below forbids without a proven mathematical need; there is none, since the already-retained decimated snapshot is sufficient. `temporal_stabilization.stabilized_depth_m` is honestly documented as decimated, never silently upsampled/interpolated back to full resolution (that would be fabricating geometry between real sample points).

## Decision 2 — When history may contribute

All four conditions the architect listed are implemented as an explicit, AND-ed frame-level gate, evaluated fresh every `process()` call (no persistent "armed" state):

1. **"E3 says observations are legitimately comparable"** → `temporal_consistency.state in {CONSISTENT, CONTRADICTORY}` (both require `comparable_count > 0`, i.e. a real pixel-level comparison happened this frame — only `INSUFFICIENT_EVIDENCE`/`NOT_COMPARABLE` fail this). **Note, stated explicitly:** `CONTRADICTORY` at the *aggregate* level still passes this specific gate — E3's `CONTRADICTORY` means "comparable, but disagreeing on balance," which is different from `NOT_COMPARABLE` ("not legitimately comparable at all"). Whether an *individual* pixel is safe to blend is decided separately, per-pixel, in Decision 3 below — an aggregate-`CONTRADICTORY` frame can still have some genuinely agreeing pixels that benefit from stabilization, and the per-pixel mechanism (not the aggregate label) is what actually protects against bad blends.
2. **"Sufficient overlapping valid geometry exists"** → new, explicit, separate threshold: `comparable_count / current_snapshot.size >= PipelineConfig.temporal_stabilization_min_comparable_fraction` (default `0.1`) — reuses E3's own already-computed `comparable_count`, no recomputation. Distinct from E3's own bare `comparable_count > 0` check: E3 asks "can I classify at all," E4 asks "is there *enough* to bother producing a stabilization claim."
3. **"Historical evidence is recent under E2"** → inherited for free: the only historical evidence E4 ever reads is `previous_record`, the exact same "single most recent comparable prior record" E3 already selected (`None` on the first frame, after `reset()`, or the first frame of a gap-triggered new sequence — E2/E3's own chronology rules already enforce recency; E4 adds no separate age check).
4. **"Required quality/confidence thresholds are satisfied"** → new, explicit, separate threshold: `previous_record.confidence >= PipelineConfig.temporal_stabilization_min_history_confidence` (default `0.3`) — reuses Level 0-2's own already-defined `confidence` scalar (`TemporalRecord.confidence`, E2), not a new "temporal trust" concept.

If any condition fails: `state = TemporalStabilizationState.INSUFFICIENT_EVIDENCE`, `stabilized_depth_m = None`, every count `0`, every fraction `None` — current observation remains fully authoritative, no stabilization claim of any kind, mirroring `TemporalConsistency`'s own established "all-zero, fraction-`None`" convention for exactly this situation.

## Decision 3 — Contradiction safety

Enforced **per pixel**, on the decimated grid, using the exact same tolerance E3 already uses (`PipelineConfig.temporal_consistency_agreement_tolerance_m` — one definition of "agree," never two):

```
current_valid  = current_snapshot > 0.0
previous_valid = previous_snapshot > 0.0
both_valid     = current_valid & previous_valid
agrees         = both_valid & (|current_snapshot - previous_snapshot| <= agreement_tolerance_m)
contradicts    = both_valid & ~agrees
```

- **`~current_valid` (current itself has no measurement here):** stabilized value stays invalid (`0.0`, `depth_map`'s own convention) — **never filled in from history.** This is a deliberate, conservative reading of Decision 2's "sufficient *overlapping* valid geometry" condition: there is no overlap at a pixel current doesn't see, so it is structurally out of scope for stabilization, not merely policy-excluded. This is also exactly how "UNKNOWN never becomes FREE from history" and "history must never erase newly observed occupied geometry" are satisfied *structurally*: a currently-unknown pixel is never touched by this mechanism at all, and a currently-occupied (valid, low-depth) pixel with no matching prior evidence (`~previous_valid`) falls to the next case, current wins outright.
- **`current_valid & ~previous_valid` (current has fresh evidence, history has none here):** stabilized value = `current_snapshot` value, unchanged. Current wins because there is nothing to blend with.
- **`contradicts` (both valid, but disagree beyond tolerance):** stabilized value = `current_snapshot` value, unchanged — **"strong current contradictory geometry always wins," implemented as a structural no-op, not a policy exception.** These pixels are counted in `contradiction_count`/`contradiction_fraction`, an explicit, honest record that an override happened — but the array value itself is never touched.
- **`agrees` (both valid, within tolerance):** genuine confidence-weighted blend (Decision 4).

"When uncertain, prefer current occupied/unknown evidence over historical free-space evidence" is satisfied by the same structural rule from two directions: an unknown current pixel is never filled from history (first bullet), and an occupied (valid) current pixel that disagrees with a "farther/freer" historical value is a `contradicts` pixel, current wins (third bullet) — there is no code path in which a historical "it was farther/clearer back then" value can ever overwrite a nearer/occupied current reading.

## Decision 4 — Estimator

A single, deterministic, bounded weighted average — no Kalman filter, no learned weighting, no opaque blended score:

```
w = previous_record.confidence                      # already bounded [0, 1] by confidence's own existing definition
stabilized_value[agrees] = (current_snapshot[agrees] + w * previous_snapshot[agrees]) / (1 + w)
```

**Every weight has exactly one documented meaning:**
- Current's own weight is fixed at `1.0` — current evidence is never assigned less trust than history, meaning history can never numerically outweigh or dominate current within the blend (the blend is a weighted average with the historical weight capped at `1.0` and the current weight fixed at `1.0`, so the historical contribution is mathematically bounded to at most a 50/50 split, never more).
- `w`, the historical sample's weight, **is** `previous_record.confidence` directly — Level 0-2's own single already-defined per-frame evidence-quality scalar, reused as-is. No new "temporal trust," no separate scale/multiplier, no arbitrary "AI confidence" invented — this is the one existing scalar this repository already computes and already means "how good was this frame's overall evidence," repurposed for exactly what it already measures.
- At `w = 0` (the minimum a record admitted past Decision 2's `>= 0.3` gate can technically approach, since the gate uses `>=`, so `w >= 0.3` in practice — but the formula itself degrades gracefully to `stabilized_value = current_snapshot` at `w = 0` regardless, satisfying Decision 7's graceful-degradation requirement even if the gate threshold were ever configured down to `0.0`).

This is the smallest estimator that is simultaneously deterministic, bounded, mathematically inspectable (a single closed-form fraction, no iteration, no hidden state), and genuinely confidence-aware (the blend ratio literally *is* the historical frame's own confidence).

## Decision 5 — History

**`TemporalRecord` is NOT extended.** Everything the estimator needs already exists on the E2/E3-frozen contract: `previous_record.depth_snapshot_m` (E3's decimated snapshot) and `previous_record.confidence` (E2's Level 0-2 scalar). Extension was checked and found mathematically unnecessary — the opposite conclusion from E3's own Decision 5, which *did* need to extend the record because no field carried per-pixel data at all; E3 already solved that problem, and E4 simply reuses the solution.

**No `DepthPerceptionResult` is ever stored.** `compute_temporal_stabilization()` reads exactly two already-existing values (`previous_record.depth_snapshot_m`, `previous_record.confidence`) and the current frame's own already-computed decimated snapshot — nothing new is retained in `TemporalHistory`'s bounded buffer at all. The stabilized output array itself is a normal, per-call `DepthPerceptionResult` field with the same lifetime as `depth_map` — computed fresh every call, never cached, never buffered.

**Memory cost of this phase: zero additional bytes retained in history.** The only new array (`temporal_stabilization.stabilized_depth_m`) lives on the returned per-call result, exactly like `depth_map`/`disparity_map`/every other Level 3 array already does — not a new retained-history cost, matching Decision 5's "do not store full `DepthPerceptionResult` objects" and "minimum recent evidence necessary" instructions exactly (the minimum turned out to be *nothing new*).

## Decision 6 — Output

New, additive, frozen `temporal.TemporalStabilization` (`temporal/types.py`) plus `temporal.TemporalStabilizationState` (`temporal/stabilization.py`, mirroring `TemporalConsistencyState`'s plain-string-constant precedent — audited first, no existing name/type matches this semantics):

| Field | Meaning |
|---|---|
| `state` | One of `STABILIZED` / `CURRENT_ONLY_FALLBACK` / `INSUFFICIENT_EVIDENCE`. |
| `stabilized_depth_m` | The decimated stabilized array, or `None` only when `state == INSUFFICIENT_EVIDENCE`. Populated (falling back to the current snapshot's own values) in `CURRENT_ONLY_FALLBACK` too — a caller wanting "the best available estimate" never needs to branch on `state` before reading it when `state != INSUFFICIENT_EVIDENCE`. |
| `eligible_count` | `current_valid.sum()` on the decimated grid — the denominator both fractions below are computed over. `0` when `state == INSUFFICIENT_EVIDENCE`. |
| `stabilized_count` | `agrees.sum()` — pixels genuinely confidence-blended this frame. |
| `contradiction_count` | `contradicts.sum()` — pixels where current overrode a disagreeing prior value this frame. |
| `stabilized_fraction` | `stabilized_count / eligible_count`, or `None` when `state == INSUFFICIENT_EVIDENCE`. Answers "amount/fraction of geometry actually stabilized," precisely. |
| `contradiction_fraction` | `contradiction_count / eligible_count`, or `None` under the same rule. Answers "contradiction/current-evidence override," precisely, as its own explicit metric rather than folded into `state` (an aggregate-`STABILIZED` frame can still report a nonzero `contradiction_fraction` for the pixels that individually disagreed — `state` alone cannot represent that, the explicit count can). |

`state` derivation, once the Decision 2 gate passes: `STABILIZED` if `stabilized_count > 0`, else `CURRENT_ONLY_FALLBACK` (gate passed, but zero pixels actually agreed closely enough to blend — e.g. every overlapping pixel individually contradicted). `eligible_count > 0` is guaranteed whenever the gate has passed (`comparable_count > 0` implies `both_valid` has at least one `True`, and `both_valid ⊆ current_valid`), so no division-by-zero exists in either fraction.

**Existing Level 3 result semantics are untouched** — `temporal_stabilization` is appended as the very last field on `DepthPerceptionResult`, after `temporal_consistency`; no existing field's meaning, shape, or value changes.

## Decision 7 — Failure / recovery

Satisfied by construction, not by added exception handling (matching this repository's own established discipline: Level 3's E3-E5 stages are deliberately *not* wrapped in `try/except` either — a genuine bug should surface, not be silently swallowed):

- Missing history (`previous_record is None`), invalid prior geometry (shape mismatch → already `NOT_COMPARABLE` at E3, fails the Decision 2 gate), timestamp discontinuity (gap-restart → `previous_for_comparison` is already forced `None` by E3's own trigger rule), missing `MotionHint` (never read by this function at all — see Decision 8), non-comparable observations (`NOT_COMPARABLE`/`INSUFFICIENT_EVIDENCE` at E3) — every one of these collapses to the same `INSUFFICIENT_EVIDENCE` gate-failure path. None of them can raise: the function either takes the gate-failure early return or proceeds through arithmetic already proven division-by-zero-safe.
- **E4 failure can never make Level 3 perception unusable** because `temporal_stabilization` is purely additive and never read by, or fed back into, any Level 3 computation — `disparity_map` through `geometry_metrics` are computed entirely before this stage runs and are never revisited afterward.
- **Recovery is automatic** because there is no persistent E4-specific state to get "stuck" in — every `process()` call re-evaluates the Decision 2 gate fresh, from the current frame's own values and whatever `TemporalHistory.latest` happens to be at that moment. A `STABILIZED, INSUFFICIENT_EVIDENCE, STABILIZED` sequence (history briefly dips below the confidence gate, then recovers) behaves exactly like E2's already-proven `GOOD, INVALID, GOOD` recovery — regression-tested in `tests/test_temporal_stabilization.py::TestFailureRecovery`.

## Decision 8 — Motion

`compute_temporal_stabilization()`'s signature accepts `current_depth_snapshot`, `previous_record`, `consistency`, and three scalar thresholds — **no `MotionHint` parameter exists, and `previous_record.motion_hint` is never read.** There is no IMU integration, no geometry warping, no ego-motion estimate, and no rotation/translation compensation anywhere in this module — verified structurally (`tests/test_temporal_stabilization.py::TestScopeDiscipline::test_stabilization_never_reads_motion_hint`, an AST attribute-access scan, the same technique `tests/test_temporal_history.py::TestDegradationRecovery::test_admission_never_inspects_quality_or_confidence` already established for an analogous claim). "If geometry cannot honestly be compared without compensation, do not stabilize it" is inherited directly from E3's `NOT_COMPARABLE` state, which already refuses to compare across a structural incompatibility without inventing alignment — E4 adds nothing new here, it just respects E3's existing refusal by gating on it.

## Decision 9 — Deterministic MP01 rule

No neural network, learned depth, learned weighting, learned filtering, semantic model, or transformer appears anywhere in `temporal/stabilization.py` or `temporal/types.py`'s `TemporalStabilization`. The entire estimator is one closed-form arithmetic expression over two small arrays and one existing scalar — fully reproducible (`tests/test_temporal_stabilization.py::TestDeterminism`), fully inspectable by reading the four lines of Decision 3/4's math above, nothing hidden in a trained parameter.

## Decision 10 — Test coverage

See `tests/test_temporal_stabilization.py` for the full suite; every scenario the architect listed is covered — enumerated in the final report.

## What this phase explicitly did not do

Implement E5 or any part of it (motion compensation, IMU integration, rotation/translation warping, ego-motion estimation). Modify `mp01_perception`, `mp01_localization`, `state_estimation_engine`, `mp01_mapping`, or any other repository. Change any Level 0-3 algorithm, or any E1-E3 contract's existing shape/semantics. Commit or push.

## Readiness assessment

**E4 was BLOCKED on design decisions; those decisions are now resolved and implemented**, with regression tests proving each one. Unresolved E5 decisions (motion compensation itself — the one thing this phase was explicitly forbidden from touching) are listed in the final report delivered alongside this document.
