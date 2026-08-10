> **Superseded for current usage (2026-08-09, Phase E8):** see `docs/LEVEL4_CANONICAL_REFERENCE.md` section 6 ("Motion compensation") for the current, authoritative description. This document remains as the historical decision record.

# Level 4, Phase E5 plan — RESOLVED, IMPLEMENTED

**Naming note, resolved before writing anything else:** `docs/E5_IMPLEMENTATION_PLAN.md` already exists in this repository and is unrelated to this document — it is a pre-existing, stale **Level 3** planning stub (`ObstacleCloud`/`FreeSpaceRays`/`GeometryMetrics`), Level 3's own independently-numbered "Phase E5," whose actual implementation happened via a different path and is already complete (see `docs/LEVEL3_ARCHITECTURE.md`'s 2026-08-06 Phase E5 update, `docs/IMPLEMENTATION_STATUS.md`). That file was left un-archived after that work finished — a genuine, pre-existing documentation gap, not something this pass caused. It is **not modified or deleted by this pass** — overwriting a historical planning record without being asked to is exactly the kind of thing to avoid; this is a naming collision between two independently-numbered phase sequences (Level 3's E1-E7 vs. Level 4's E1-E5), not an architectural contradiction. This document uses a disambiguated filename instead. Flagged explicitly in the final report so a future pass can decide whether to rename either file for clarity.

**LEVEL 4 E5 — ROTATIONAL MOTION COMPENSATION IMPLEMENTED.**
**NO IMU BIAS ESTIMATION. NO ACCUMULATED ORIENTATION. NO TRANSLATION. NO VIO. NO SLAM. NO LOCALIZATION. NO WORLD-FRAME OUTPUT. NO NEURAL METHODS.**

Objective (as specified by the architect): compensate for short-duration camera rotation between the previous comparable frame and the current frame, so E3/E4's existing comparison/stabilization machinery can work against geometry that has been honestly re-expressed in the current frame's own viewpoint — never a pose estimate. Ten decisions resolved below; verified against the frozen E1-E4 repository state before implementation began, no contradiction found.

## Decision 1 — SO(3) integration math

For a chronologically-ordered, already-filtered sequence of accepted `MotionHint` samples `s_1, ..., s_n` (selection rule: Decision 3), each contributing an increment

```
ΔR_k = Exp([ω_k]_x · Δt_k)
```

computed via `cv2.Rodrigues(ω_k · Δt_k)` — reusing this project's existing OpenCV dependency for the standard Rodrigues exponential map, rather than hand-rolling small-angle edge cases (`cv2.Rodrigues` already handles `θ → 0` correctly via its own internal series). `Δt_k = t_k - t_{k-1}`, with `t_0 := previous_frame_timestamp`. **Zero-order hold, not extrapolation**: each sample's angular velocity is assumed constant only over the span *actually bounded by* the previous sample (or the previous frame, for the first sample) and its own timestamp — never extended past its own timestamp. The residual gap between the *last* accepted sample and `current_frame_timestamp` is deliberately left uncompensated (Decision 7 forbids assuming one sample represents the whole interval; extrapolating the last sample forward would violate that for exactly the same reason a single sample covering the whole interval would).

**Composed chronologically** via post-multiplication (`ΔR_total = ΔR_1 @ ΔR_2 @ ... @ ΔR_n`) — the standard "compose in the local/body frame" rule, correct because a gyroscope reports angular velocity in its own instantaneous (rotating) frame, not a fixed external one.

**The actually-useful quantity, `ΔR_prev_to_curr`, is the transpose of that composition, not the composition itself** — derived, not assumed:

Let `R_cam(t)` be the (unobserved, never computed or stored) camera orientation relative to a fixed external reference, so `p_world = R_cam(t) · p_cam(t)`. The standard gyro kinematics give `R_cam(t_curr) = R_cam(t_prev) · ΔR_total`. For a physically fixed point, `p_curr = R_cam(t_curr)^T · R_cam(t_prev) · p_prev = ΔR_total^T · p_prev`. So:

```
ΔR_prev_to_curr = ΔR_total^T
```

This is the rotation applied to 3D points reconstructed from the *previous* frame to express them in the *current* frame's own coordinates — verified empirically, not just derived on paper, by `tests/test_rotation_compensation.py::TestSyntheticRotationImprovesComparability` (a synthetic scene rotated by a known `ΔR_true`, fed a `MotionHint` sequence encoding that same physical rotation, confirmed to raise `agreement_fraction` after compensation) and by dedicated analytical 90°-yaw/pitch/roll cases checking the *sign* of the resulting point displacement matches physical intuition (a camera that yaws right sees a fixed forward point shift left in its own frame).

`R_cam(t)` itself is never computed, stored, or returned anywhere — only the ephemeral `ΔR_prev_to_curr` for the one comparison at hand (Decision 4).

## Decision 2 — One internal component, prior-geometry rotation only

New module, `temporal/rotation_compensation.py`. It owns exactly two things, corresponding to the two verbs in the architect's own phrasing:

- **"short-window gyro integration"** → `select_motion_hint_samples()` + `integrate_angular_velocity()`.
- **"prior-geometry rotation"** → `compensate_prior_geometry()`, which back-projects the previous frame's decimated depth snapshot into 3D camera-frame points, rotates them by `ΔR_prev_to_curr`, and re-projects them onto the current frame's own decimated pixel grid.

**Why reprojection, not an in-place depth-value nudge:** `temporal.TemporalRecord.depth_snapshot_m` (Phase E3) is a decimated *depth* array — a scalar per fixed pixel index, not a full 3D point cloud. A camera rotation's dominant geometric effect is that a fixed physical point projects to a *different pixel* after the rotation, not merely a different depth value at the same pixel. Adjusting depth values in place without re-indexing pixels would not honestly represent "prior-geometry rotation" — it would silently misattribute rotated points to the wrong pixel, which is worse than not compensating at all. Full back-projection/rotate/re-projection is the smallest design that is still geometrically honest about what a rotation does to observed geometry.

**Why this does not become a pose estimator:** `R_cam(t)` (an actual orientation) is never computed. `ΔR_prev_to_curr` is computed fresh every `process()` call from only the current interval's samples, used immediately to re-express one previous record's geometry, and discarded — nothing about it is retained across calls, returned to the caller, or accumulated into any running estimate (verified: `tests/test_rotation_compensation.py::TestScopeDiscipline` confirms no orientation-shaped state survives between two `process()` calls with the pipeline instance held constant).

**Intrinsics reused, not reinvented:** the back-projection/re-projection uses `f = abs(Q[2,3])`, `cx = -Q[0,3]`, `cy = -Q[1,3]` — the exact same rectified-intrinsics derivation `depth.DepthEstimator`/`calibration.contracts.StereoExtrinsics` already use from the pipeline's own `Q` matrix, computed once in `DepthPerceptionPipeline.__init__` (same "build once, reuse" discipline as every other engine). No new intrinsics representation, no distortion handling needed (all of E2-E5's arrays are already post-rectification, where a plain pinhole model is exact).

**Occlusion/collision handling in the re-projection:** deterministic nearest-wins (smallest re-projected depth) at each target decimated cell, via a stable sort plus fancy-index assignment (last write wins) — no interpolation, no hole-filling, no fabricated geometry between real sample points. A target cell with no re-projected point stays invalid (`0.0`), honestly representing "nothing from the prior frame reprojects here."

## Decision 3 — Timestamp contract and sample selection

**Frozen for E5's own math specifically** (a narrower, additional precondition on top of E1-E4's fully opaque "caller-defined float" timestamp convention, which remains unchanged for every other purpose): `MotionHint.timestamp` and frame timestamps must be **seconds**, in one common **monotonic sensor-time domain**, whenever a caller wants E5's compensation to activate. This is required because `Δt` is used directly as physical seconds in `ω [rad/s] × Δt [s] = rotation vector [rad]` — E1-E4 never did arithmetic on timestamp *values* (only comparisons), so this requirement did not previously exist and does not retroactively apply to E1-E4's own contracts.

**Interval:** a `MotionHint` sample is admissible only if `previous_frame_timestamp < sample.timestamp <= current_frame_timestamp` — a half-open interval, chosen so a sample landing exactly on the current frame's own timestamp counts (it plausibly *is* this frame's own synchronized reading) while a sample landing exactly on the previous frame's timestamp does not (it was already available, and by construction, at the previous frame's own processing).

**Rejection rule, implemented as one deterministic pass over the sequence *in the order given* (not pre-sorted — silently reordering a misbehaving caller's sequence would mask a real bug, not fix one):** a sample is rejected if it is `None`, `MotionHint.valid` is `False`, its timestamp is missing/non-finite, it falls outside the interval above, or its timestamp does not strictly exceed the last *accepted* sample's timestamp (starting from `previous_frame_timestamp`) — this single rule catches non-monotonic and duplicate-timestamp samples identically. "Stale" (the architect's own word, Decision 6) is not a separate mechanism — a stale sample is simply one that fails the interval check above (its timestamp is not within `(previous_frame_timestamp, current_frame_timestamp]`).

## Decision 4 — Scope: ephemeral relative rotation only

`compute_rotation_compensation()`'s only numerically meaningful output consumed further is `ΔR_prev_to_curr` (Decision 1), used once, immediately, then discarded. No accumulated orientation, no translation (the reprojection's back-projection/rotation/re-projection formulas contain no translation term anywhere — verified structurally, `tests/test_rotation_compensation.py::TestScopeDiscipline::test_no_translation_terms`), no velocity, no bias estimation, no VIO/SLAM/localization, and no world-frame output. `ΔR_prev_to_curr` itself is **not** exposed on `DepthPerceptionResult` — only a small status marker is (Decision 6's output) — specifically so a caller cannot be tempted to log/accumulate a rotation trajectory externally, which would reintroduce exactly the "accumulated orientation" this decision forbids, just one layer removed from this library's own boundary.

## Decision 5 — Upstream of E3/E4, zero duplication

Implemented as a pipeline-level *substitution*, not a change to E3/E4's own code:

```
previous_for_comparison (E3's existing selection, unchanged)
        |
        v
compute_rotation_compensation()  — NEW, E5
        |
        v
compensated_previous_for_comparison  (a TemporalRecord with the SAME
                                       timestamp/confidence/geometry_quality/
                                       motion_hint, but depth_snapshot_m
                                       replaced via dataclasses.replace —
                                       or, unchanged, the original record,
                                       when compensation was not applied)
        |
        v
compute_temporal_consistency(...)     — E3, byte-identical code
        |
        v
compute_temporal_stabilization(...)   — E4, byte-identical code
```

`temporal/consistency.py` and `temporal/stabilization.py` are not modified by this phase at all — verified: `git diff` for those two files is empty. E5 only changes which `TemporalRecord` value `pipeline.py` passes into their existing, frozen signatures.

## Decision 6 — Fallback semantics

Every one of the architect's named failure cases collapses to the same `RotationCompensationStatus.NOT_APPLIED` outcome, and in every case `previous_for_comparison` is passed into E3/E4 **completely unchanged** — E3/E4 behave exactly as they did before E5 existed:

| Case | Handling |
|---|---|
| Missing motion hints (`motion_hints=None` or empty) | `select_motion_hint_samples()` returns `[]` → `NOT_APPLIED` |
| Invalid motion hint (`MotionHint.valid is False`) | Filtered out by selection, same as missing |
| Stale motion hint | Filtered out by the interval check (Decision 3) |
| Out-of-interval / non-monotonic motion hint | Filtered out by the interval/monotonicity check (Decision 3) |
| Insufficient motion hints (zero survive filtering) | `select_motion_hint_samples()` returns `[]` → `NOT_APPLIED` |
| No prior record (`previous_for_comparison is None`) | `compute_rotation_compensation()`'s first check → `NOT_APPLIED`, nothing to compensate |
| `enable_rotation_compensation=False` | E5 is not invoked at all — `pipeline.py` never calls into `temporal/rotation_compensation.py` |

**Motion data never blocks otherwise-valid stereo perception:** `compute_rotation_compensation()` and everything it calls can raise only on a genuine programming error (bad shapes passed by the caller), never on "ordinary" missing/invalid/stale input — every one of those is a valid, non-exceptional `NOT_APPLIED` outcome, and even a `NOT_APPLIED` outcome has zero effect on `disparity_map`/`depth_map`/`geometry`/every other Level 3 field, which are computed entirely before E5 ever runs.

## Decision 7 — Bounded sample sequences, no single-sample extrapolation

`DepthPerceptionPipeline.process()` gains a new, additive `motion_hints: Optional[Sequence[MotionHint]] = None` parameter — distinct from the existing singular `motion_hint` parameter (E1-E4, still used for `TemporalRecord.motion_hint` association only, untouched). `StereoObservation` gains the matching additive `motion_hints: Optional[Sequence[MotionHint]] = None` field, forwarded by `process_observation()`. "Bounded" is satisfied by construction, not by a new arbitrary count-limit config field: the sequence is used and discarded within one `process()` call — never retained in `TemporalHistory`'s buffer, never accumulated across calls — so its "boundedness" is about lifetime (ephemeral, per-call), not an artificial maximum length. `integrate_angular_velocity()` never assumes a single sample covers more than the span actually bounded by its own neighbors (Decision 1's zero-order-hold, not-extrapolated rule).

No separate "short window" config threshold was added: a gap wide enough to make the previous/current interval no longer "short" already triggers `TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE` at Phase E2 (`PipelineConfig.temporal_gap_limit_s`, default `0.5` s), which already forces `previous_for_comparison = None` before E5 ever runs — E5 inherits this bound rather than duplicating it with a second, potentially inconsistent threshold.

## Decision 8 — Simulated IMU stays outside core

`temporal/rotation_compensation.py` accepts exactly the same `temporal.MotionHint` sequence type every other Level 4 phase already accepts — no simulator, no synthetic-sample generator, and no source-specific branch of any kind was added anywhere in `src/depth_perception_engine/`. `tests/test_level4_architecture_guards.py` (unmodified since Phase E1) continues to guard this structurally.

## Output — new metadata

`DepthPerceptionResult.rotation_compensation_status: Optional[str] = None`, appended last, additive. `None` unless `PipelineConfig.enable_temporal` and `PipelineConfig.enable_rotation_compensation` are both `True`; otherwise one of `temporal.RotationCompensationStatus.APPLIED` / `NOT_APPLIED` — two states only (not a richer taxonomy of *why* `NOT_APPLIED`, since Decision 6 already treats every failure case identically and `ΔR_prev_to_curr` itself is deliberately not exposed to explain further — see Decision 4). `PipelineConfig.enable_rotation_compensation: bool = False`, nested under `enable_temporal`, independent of `enable_temporal_stabilization` (E5 can usefully feed E3-only consumers too, not just E4).

## What this phase explicitly did not do

Implement E6 or beyond. Modify `temporal/consistency.py`, `temporal/stabilization.py`, or any of their frozen semantics. Modify any Level 0-3 algorithm. Modify `mp01_perception`, `mp01_localization`, `state_estimation_engine`, `mp01_mapping`, or any other repository. Estimate camera pose, position, velocity, or IMU bias. Compensate translation. Accumulate orientation across calls. Use any neural/learned component. Commit or push. Modify or delete the pre-existing, unrelated `docs/E5_IMPLEMENTATION_PLAN.md` (Level 3's own stale planning stub — see the naming note at the top of this document).

## Readiness assessment

**E5 was BLOCKED on design decisions; those decisions are now resolved and implemented.** Unresolved E6 decisions are listed in the final report delivered alongside this document.
