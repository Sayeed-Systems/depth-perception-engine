> **Superseded for current usage (2026-08-09, Phase E8):** see `docs/LEVEL4_CANONICAL_REFERENCE.md` section 7 ("Reliability") for the current, authoritative description. This document remains as the historical decision record.

# Level 4, Phase E6 plan — RESOLVED, IMPLEMENTED

**Naming note, resolved before writing anything else:** `docs/E6_IMPLEMENTATION_PLAN.md` already exists and is unrelated to this document — same situation as E5's own naming collision (`docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md`'s own note). It is a pre-existing, stale **Level 3** planning stub (occupancy/temporal-fusion speculation, itself superseded in place by a 2026-08-06 update note pointing to `docs/E7_IMPLEMENTATION_PLAN.md`), unrelated to Level 4's own independently-numbered Phase E6. Left untouched, not overwritten. This document uses a disambiguated filename, matching the E5 precedent.

**LEVEL 4 E6 — MOTION-AWARE RELIABILITY ASSESSMENT IMPLEMENTED.**
**NO GEOMETRY MODIFICATION. NO ADDITIONAL MOTION COMPENSATION. NO NEURAL METHODS.**

Objective (as specified by the architect): assess, deterministically and explicitly, whether E3/E4's temporal results remain trustworthy given the motion conditions and E5's compensation outcome — never blend these into one opaque score, never touch geometry.

## Audit of existing contracts (performed before any design decision below)

What Level 4 already freezes that E6 can reuse directly, with zero new algorithm:

- `temporal.TemporalConsistency.state` (E3): `CONSISTENT`/`CONTRADICTORY`/`INSUFFICIENT_EVIDENCE`/`NOT_COMPARABLE` — already exactly "temporal comparability quality."
- `temporal.TemporalStabilization.state` (E4): `STABILIZED`/`CURRENT_ONLY_FALLBACK`/`INSUFFICIENT_EVIDENCE` — already exactly "stabilization applicability."
- `temporal.RotationCompensationStatus` (E5): `APPLIED`/`NOT_APPLIED` — already exactly "rotation-compensation validity," but deliberately collapses every failure reason (missing/invalid/stale/insufficient hints, no prior record) into one state (E5's own Decision 6: "no richer taxonomy of *why* `NOT_APPLIED`"). This means "motion-data availability" and "rotation-compensation coverage" are **not** separately observable from `RotationCompensationStatus` alone — a real gap E6 must fill, not invent from nothing.
- `temporal.rotation_compensation.select_motion_hint_samples()` / `integrate_angular_velocity()` (E5): already-frozen, pure, tested functions. `select_motion_hint_samples()` returns exactly the admissible-sample list for an interval; `integrate_angular_velocity()` returns exactly `ΔR_prev_to_curr`. Neither performs "motion compensation" in the sense E6 is forbidden from doing — that verb specifically means `compensate_prior_geometry()` (warping geometry), which E6 never calls.
- `ΔR_prev_to_curr` itself: computed inside `compute_rotation_compensation()` but, by E5's own explicit Decision 4, never returned to its caller or exposed on `DepthPerceptionResult` ("so a caller cannot be tempted to log/accumulate a rotation trajectory externally"). E6 needs a scalar rotation *magnitude* for one frame's reliability classification — this is satisfied by calling `integrate_angular_velocity()` a second time (E5's own pure, frozen function, unmodified), immediately converting its result to one discarded scalar, never retaining or exposing the matrix itself. This preserves E5's own boundary in spirit (no trajectory-shaped state, no pose exposed) while satisfying E6's stated requirement to represent "angular-motion magnitude" explicitly. `temporal/rotation_compensation.py` itself is not modified — verified: `git diff` for that file is empty.

**No existing contract already represents "motion-data availability" or "angular-motion magnitude" as their own explicit values** — these two are the only genuinely new pieces of information this phase computes, and both are derived by calling E5's own existing pure functions a second time, not by inventing new algorithm.

## 1. Explicit representation (not one opaque blended confidence)

New, additive, frozen `temporal.MotionAwareReliability` (`temporal/types.py`), with exactly six fields — one per signal the architect's Design section 1 named, "at least":

| Field | Represents | Source |
|---|---|---|
| `motion_sample_count: int` | motion-data availability | `len(select_motion_hint_samples(...))` — E5's own function, reused |
| `motion_coverage_fraction: Optional[float]` | rotation-compensation coverage | derived (see below) — new, well-defined metric |
| `rotation_compensation_status: Optional[str]` | rotation-compensation validity | `temporal.RotationCompensationStatus`, reused as-is |
| `angular_motion_magnitude_rad: Optional[float]` | angular-motion magnitude | derived from `integrate_angular_velocity()`'s own `ΔR_prev_to_curr` — E5's function, reused |
| `temporal_consistency_state: Optional[str]` | temporal comparability quality | `temporal.TemporalConsistencyState`, reused as-is |
| `temporal_stabilization_state: Optional[str]` | stabilization applicability | `temporal.TemporalStabilizationState`, reused as-is |
| `state: str` | resulting reliability state | computed by E6 (section 2 below) |

No field is a blend of the others — a caller can read any one signal independently, exactly as the architect required.

## 2. Reliability states

`temporal.MotionAwareReliabilityState` (`temporal/reliability.py`, mirroring every prior phase's plain-string-constants precedent): `RELIABLE`, `DEGRADED`, `UNRELIABLE`, `INSUFFICIENT_EVIDENCE` — the architect's own suggested names, used verbatim (no repository-existing equivalent was found to reuse instead, per the audit above).

## 3. Exact decision logic (deterministic, every branch traced to an existing or newly-justified signal)

Evaluated fresh every `process()` call, in this exact priority order:

```
1. IF temporal_consistency is None:                     -> INSUFFICIENT_EVIDENCE
   (E3 did not run this frame at all — chronology
    rejected the timestamp; nothing to assess)

2. ELIF temporal_consistency.state in
       (INSUFFICIENT_EVIDENCE, NOT_COMPARABLE):          -> INSUFFICIENT_EVIDENCE
   (E3 itself found no real comparison to judge —
    first frame, post-reset, gap-restart, or a
    structural incompatibility)

3. ELIF temporal_consistency.state == CONTRADICTORY:     -> UNRELIABLE
   (E3 itself already found current and prior evidence
    disagree on balance — "poor temporal comparability",
    the architect's own listed degrading factor, in its
    strongest already-frozen form)

4. ELSE (temporal_consistency.state == CONSISTENT):
   4a. IF rotation_compensation_status == APPLIED:
       IF angular_motion_magnitude_rad >
          reliability_max_angular_motion_rad:            -> UNRELIABLE
          ("excessive angular displacement" — beyond this
           angle, E5's own zero-order-hold/discretized-
           reprojection assumptions are no longer trusted)
       ELIF motion_coverage_fraction <
            reliability_min_motion_coverage_fraction:     -> DEGRADED
          (compensation applied, but only over a minority
           of the true interval — "insufficient MotionHint
           coverage", a large uncompensated residual
           remains per E5's own documented limitation)
       ELSE:                                              -> RELIABLE
          (consistent, and the motion that was compensated
           for was both small and well-covered)

   4b. ELIF rotation_compensation_status == NOT_APPLIED:  -> DEGRADED
       (E3 is consistent on RAW, uncompensated evidence
        alone — "missing/insufficient MotionHint coverage"
        or invalid/stale hints meant motion could not be
        validated for this frame; the comparison itself
        still succeeded, but without the additional
        motion-conditions confirmation the architect asked
        E6 to require)

   4c. ELSE (rotation_compensation_status is None,
             i.e. E5 was not even enabled):                -> RELIABLE
       (motion assessment was never requested by the
        caller's own configuration — E3's already-strong
        CONSISTENT verdict stands on its own merits,
        undegraded; this is a configuration choice, not a
        failure, and is the one case Design section 5's
        "never block otherwise valid single-frame
        perception" most directly protects — a caller who
        never enabled E5 gets exactly E3's own judgement,
        nothing withheld)
```

## 4. Thresholds — exact definition, units, justification (Design section 3's explicit requirement)

Two new `PipelineConfig` fields, both documented with the same "policy choice, not a physical constant" honesty this repository already uses for every threshold of this kind (`geometry_healthy_min_valid_fraction`, `temporal_consistency_min_agreement_fraction`, etc. — none of those are derived from first principles either, and this repository has never pretended otherwise):

- **`reliability_max_angular_motion_rad: float = 0.0873`** (5°). Units: radians, the same unit `angular_motion_magnitude_rad` (and `MotionHint.angular_velocity_rad_s`, once integrated) is already expressed in throughout Level 4 — no new unit introduced. Algorithmic justification: E5's own compensation is a zero-order-hold SO(3) integration composed with a discrete, grid-snapped (nearest-neighbor, not interpolated) re-projection — both are well-understood *small-rotation* approximations; as the true rotation grows, (a) the zero-order-hold's own discretization error grows, and (b) more re-projected points fall outside the current frame's decimated grid entirely (observed directly during E5's own test-parameter tuning — see `docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md`), silently shrinking the comparable population rather than failing loudly. 5° is a conservative, order-of-magnitude "still small" bound for a short inter-frame interval — not derived from any specific rig's calibration or frame rate, exactly like every sibling threshold in this file.
- **`reliability_min_motion_coverage_fraction: float = 0.5`** (dimensionless, `[0, 1]`). Justification: `motion_coverage_fraction` (defined below) measures what portion of the true `(previous_frame_timestamp, current_frame_timestamp]` interval was actually spanned by accepted samples, versus left as an unmeasured, uncompensated residual tail (E5's own Decision 1: "the residual gap between the *last* accepted sample and `current_frame_timestamp` is deliberately left uncompensated"). Below 50% coverage, more than half the interval's true rotation is entirely unaccounted for by the zero-order-hold model — a policy choice for "no longer confidently 'mostly compensated'," not a derived constant.

**`motion_coverage_fraction`, exact definition:** `(last_accepted_sample.timestamp - previous_frame_timestamp) / (current_frame_timestamp - previous_frame_timestamp)`, computed only when at least one sample was accepted (`None` otherwise, matching every other "no evidence" field in this library's own convention — never `0.0` masquerading as "measured and found zero"). Dimensionless, `[0, 1]` by construction (accepted samples are bounded within the interval by `select_motion_hint_samples()`'s own frozen selection rule).

**`angular_motion_magnitude_rad`, exact definition:** the standard, canonical SO(3) rotation angle extracted from `ΔR_prev_to_curr`: `θ = arccos(clip((trace(ΔR_prev_to_curr) - 1) / 2, -1, 1))` — not an invented heuristic; this is the textbook closed-form relationship between a rotation matrix and its own rotation angle, independent of axis. Computed only when `rotation_compensation_status == APPLIED` (a real `ΔR_prev_to_curr` was actually integrated this frame); `None` otherwise.

## 5. Raw-vs-derived separation

E6 reads `temporal_consistency`, `temporal_stabilization`, `rotation_compensation_status`, and `motion_hints` — all already-computed, already-frozen values — and writes nothing back into any of them. `compute_motion_aware_reliability()`'s signature accepts no mutable reference to `DepthPerceptionResult`, `disparity_map`, `depth_map`, any `geometry.*` type, `TemporalConsistency`, `TemporalStabilization`, or a rotation matrix capable of being mutated in place (verified: `tests/test_motion_aware_reliability.py::TestScopeDiscipline::test_signature_accepts_no_mutable_upstream_type`, mirroring E3/E4's own "structurally read-only" proof pattern). `temporal/consistency.py`, `temporal/stabilization.py`, and `temporal/rotation_compensation.py` are not modified by this phase at all.

## 6. Failure/recovery — missing motion data degrades honestly, never blocks

Every one of the architect's named cases resolves to a specific, already-covered branch in section 3's decision table — no separate mechanism needed: missing/insufficient `MotionHint` coverage → branch 4b (`DEGRADED`) or, if E3 itself found no comparable evidence, branch 2 (`INSUFFICIENT_EVIDENCE`); invalid compensation → branch 4b; excessive angular displacement → branch 4a's first check (`UNRELIABLE`); poor temporal comparability → branch 3 (`UNRELIABLE`); insufficient overlapping evidence → branch 2. `compute_motion_aware_reliability()` never raises on any ordinary input (missing timestamps, empty motion hints, `None` inputs) — every combination is a valid, classified, non-exceptional outcome. **Level 3 single-frame perception is never blocked**: `motion_aware_reliability` is purely additive on `DepthPerceptionResult`, computed entirely after `disparity_map` through `geometry_metrics` are already finalized, and its own computation (even in the worst case) touches nothing upstream. Recovery is automatic — no persistent E6 state exists to get stuck in; every `process()` call re-derives `state` fresh from that call's own `temporal_consistency`/`rotation_compensation_status`/motion samples.

## 7. No platform assumptions

`temporal/reliability.py` contains no aerial/ground/marine/drone/rover/boat concept, and no dependency on `state_estimation_engine` — guarded structurally by the existing, unmodified `tests/test_level4_architecture_guards.py` (Phase E1), which scans all of `src/depth_perception_engine/` including this new module.

## 8. No neural methods

The entire reliability computation is one deterministic branch table over already-computed scalars/enums plus one closed-form trigonometric formula (`arccos`) — no learned weight, no trained model, no opaque score, verified by the same AST-based "no forbidden learned/neural identifier" scan `tests/test_temporal_stabilization.py` already established as a pattern.

## Output contract

`DepthPerceptionResult.motion_aware_reliability: Optional[temporal.MotionAwareReliability] = None`, appended last (after `rotation_compensation_status`), additive. `None` unless `PipelineConfig.enable_temporal` and `PipelineConfig.enable_motion_aware_reliability` are both `True`. `PipelineConfig.enable_motion_aware_reliability: bool = False`, nested under `enable_temporal`, independent of `enable_temporal_stabilization`/`enable_rotation_compensation` (E6 assesses reliability using whatever E3/E4/E5 state actually exists this frame — including "E5 disabled," a fully legitimate configuration per branch 4c above — so it does not require either of the other two flags to be on).

## Where E6 runs in `process()`

Immediately after E4's stabilization block, but **outside** the narrower "admitted this frame" gate that scopes E3/E4/E5 — computed whenever `enable_temporal` and `enable_motion_aware_reliability` are both `True`, regardless of `temporal_admission_status`, so a chronology-rejected frame is correctly classified `INSUFFICIENT_EVIDENCE` too (matching decision branch 1, which specifically depends on being reachable even when E3 never ran).

## What this phase explicitly did not do

Implement E7 or any part of it. Modify `temporal/consistency.py`, `temporal/stabilization.py`, or `temporal/rotation_compensation.py`. Modify any Level 0-3 algorithm. Modify `mp01_perception`, `mp01_localization`, `state_estimation_engine`, `mp01_mapping`, or any other repository. Perform any additional motion compensation (no call to `compensate_prior_geometry()` anywhere in this module). Use any neural/learned component. Introduce a platform/vehicle-class concept. Commit or push. Modify or delete the pre-existing, unrelated `docs/E6_IMPLEMENTATION_PLAN.md` (Level 3's own stale planning stub).

## Readiness assessment

**E6 was BLOCKED on design decisions; those decisions are now resolved and implemented**, each one traced to either an existing frozen contract or a newly-justified, exactly-defined threshold — no policy was invented without justification. Unresolved E7 decisions are listed in the final report delivered alongside this document.
