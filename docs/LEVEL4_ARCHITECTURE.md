> **Superseded for current usage (2026-08-09, Phase E8):** `docs/LEVEL4_CANONICAL_REFERENCE.md` is now the single authoritative, current description of Level 4, organized by concept rather than by build order. This document remains as the historical development-process record (the audited pre-change state, the section-by-section design reasoning, the twelve-decisions-per-phase level of detail) — nothing below is deleted or rewritten, but a reader wanting "what Level 4 currently is" should start at the canonical reference, not here.

# Level 4 architecture report (Phases E1-E2, updated through E7)

**LEVEL 4 CURRENT STATUS: E7 — DETERMINISTIC, PER-CELL TEMPORAL-PERSISTENCE CLASSIFICATION IMPLEMENTED.**
**NO TRACKING, VIO, SLAM, LOCALIZATION, MAPPING, PLANNING, CONTROL, OR SEMANTIC/OBJECT-RECOGNITION IS IMPLEMENTED, AND NONE IS PLANNED IN THIS REPOSITORY.**

> **Update (2026-08-09, Phase E7):** persistence classification (NEW/PERSISTENT/DISAPPEARING per grid cell, plus a CLASSIFIED/UNRELIABLE/INSUFFICIENT_EVIDENCE frame-level gate) is now implemented — see `docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md` for the complete decision record. Section 7's frozen safety rules ("history may support weak current evidence, never override strong contradictory current evidence" and "UNKNOWN != FREE") are now enforced by real code, not just stated: a bounded, fixed-size collaborator (`temporal.persistence.TemporalPersistenceTracker`, owned by `DepthPerceptionPipeline` exactly like `temporal.TemporalHistory`/`obstacles.ThreatAssessor`) tracks per-cell support count/absence streak against the same decimated grid E3/E4 already use, gated hard on Phase E6's own `MotionAwareReliability.state` (an `UNRELIABLE` frame can neither create nor reinforce persistence), and reuses Phase E5's own rotation-compensation function (extended additively with a payload-carrying sibling, `compensate_prior_geometry_with_payload()` — not a second, independent reprojection implementation) to keep tracked state spatially valid under a rotating sensor. Sections 3 and 12 below are updated in place; nothing in sections 1-2's architectural boundary changed. E3-E6 (temporal consistency, stabilization, rotation compensation, motion-aware reliability) were implemented between this document's original E2 update and this one — see `docs/E3_IMPLEMENTATION_PLAN.md`, `docs/E4_IMPLEMENTATION_PLAN.md`, `docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md`, `docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md`, and `docs/IMPLEMENTATION_STATUS.md`'s per-phase addenda for their own complete records; this document's prose below was not rewritten for each of those phases individually (they did not change this document's own architectural boundary), only for E2 (which resolved this document's own open design questions) and now E7 (which fulfills this document's own frozen section 7 safety rules for the first time).

> **Update (2026-08-08, Phase E2):** the twelve open design questions this document's original (E1) sections 6/12/13 left for E2 have all been resolved by explicit project-architect decision and implemented — see `docs/E2_TEMPORAL_HISTORY_PLAN.md` for the complete, permanent record of each decision. Summary: `temporal.TemporalHistory` (`temporal/history.py`) is a bounded, deterministic chronology component, owned by `DepthPerceptionPipeline` and constructed only when `PipelineConfig.enable_temporal` (new, default `False`) is `True`. It admits one `temporal.TemporalRecord` per accepted frame — a deliberately minimal 4-field value (`timestamp`, `confidence`, `geometry_quality`, `motion_hint`; see `docs/E2_TEMPORAL_HISTORY_PLAN.md`'s memory-decision table for why each field earns its place) — enforcing a count bound, a time-window bound, and timestamp-discontinuity (gap) detection, all computed from `TemporalRecord.timestamp` values alone, never wall-clock time. `DepthPerceptionResult` gained one additive field, `temporal_admission_status: Optional[str] = None`, reporting per-frame chronology outcome via `temporal.TemporalAdmissionStatus`'s plain string constants. `DepthPerceptionPipeline.reset()` now also clears temporal history (E1's frozen obligation, section 11 below, now fulfilled). `process()` gained one additive `motion_hint: Optional[MotionHint] = None` parameter, and `process_observation()` now forwards `StereoObservation.motion_hint` into it — the field E1 left reserved-but-unconsumed is, as of E2, consumed for pure bookkeeping association (no integration, no algorithm). Sections 4-13 below are updated in place to reflect what's now implemented; nothing in sections 1-3's architectural boundary changed.

This document is the Level 4 counterpart to `docs/LEVEL3_ARCHITECTURE.md`, written at the start of Level 4 with Level 3 frozen (`docs/IMPLEMENTATION_STATUS.md`'s 2026-08-07 E7 addendum) as a regression baseline. It records what E1 audited, what it built, what it deliberately did not build, and why.

## 1. Pre-change state (audited before anything was changed)

- Branch: `main`, commit `169b7e6` ("Recapture the Level 3 README snapshot/GIF with a richer real scene"), working tree clean.
- Full suite: `pytest tests/ -q` — **422 passed**, 0 failed, 0 skipped.
- Canonical processing object: `DepthPerceptionPipeline` (`pipeline/pipeline.py`) — already has `process()`, `process_observation()`, `reset()`, `close()`, `health()`. No `DepthPerceptionEngine` symbol exists or is introduced here.
- `PipelineConfig` (`config/pipeline_config.py`): plain validated dataclass, no temporal fields.
- `DepthPerceptionResult` (`models/result.py`): `disparity_map`, `depth_map`, `traversability_mask`, `obstacles`, `confidence`, `processing_time_ms`, `valid_disparity_mask`, `valid_depth_mask`, `timestamp`, `geometry`, `geometry_body`, `obstacle_cloud`, `free_space_rays`, `geometry_metrics` — all Level 0-3, no temporal field.
- `StereoObservation` (`models/result.py`): `left_image`, `right_image`, `left_timestamp`, `right_timestamp`, `calibration` (reserved, unconsumed), `frame_id` (reserved, unconsumed) — already has precedent for "reserved, additive, not-yet-consumed" fields.
- Frame/timestamp conventions: `frames.FrameId` (`CAMERA_OPTICAL_LEFT`, `BODY`), `frames.RigidTransform` (static rotation+translation, named `from_frame`/`to_frame`, convention `p_out = rotation @ p_in + translation`). Timestamps everywhere in this library are opaque, caller-defined floats — no unit conversion or synchronization is ever performed by the library itself.
- Health/degradation contracts: `PipelineHealth` is explicitly lifecycle-only ("not a per-frame diagnosis," its own pre-existing docstring) — per-frame degradation is read off `DepthPerceptionResult` (`confidence`, `geometry_metrics`, `geometry.GeometryQuality`/`classify_geometry_quality()`), not off `PipelineHealth`.
- Zero references anywhere in `src/` to `state_estimation_engine`, `rclpy`, `sensor_msgs`, `cv_bridge`, `aerial`, `drone`, `ground` (as a platform-mode word), `marine`, `rover`, `boat`, `MAVLink`, or `Pixhawk` — confirmed by direct grep before this pass began, and re-verified after (see `docs/E7_IMPLEMENTATION_PLAN.md`'s own precedent for this style of check; guarded going forward by `tests/test_no_ros_dependency.py` and this pass's new `tests/test_level4_architecture_guards.py`).

Every design decision below adapts to this audited state rather than assuming names/APIs from the Level 4 task prose that turned out not to exist (none did — the repository already matched the prose closely, e.g. `reset()`/`health()` already existed, so section 11's "design the smallest clean public contract" resolved to "document the existing one," not build a new one).

## 2. Architectural purpose (restated, not changed)

Level 3 answers "what geometric environment evidence is visible NOW." Level 4 will eventually answer "how trustworthy and temporally consistent is that evidence across recent observations." Level 4 must remain **perception** — it does not become, and must never become, localization, mapping, planning, or control.

`depth_perception_engine` remains standalone, ROS-free, platform-agnostic, and agent-agnostic. Nothing in Level 4 (present or future) may know whether its consumer is aerial, ground, marine, or any other vehicle class — that concept must never appear in core Level 4 logic, config, or types. This is enforced going forward by `tests/test_level4_architecture_guards.py`, not merely stated.

## 3. Strict responsibility boundary (frozen, not just described)

Level 4 MAY eventually (not yet, not in E1): retain bounded recent perception history, compare geometric observations through time, compute temporal consistency, stabilize noisy depth/geometry, represent temporal confidence, distinguish persistent/transient geometric evidence, use an optional externally supplied short-duration rotational motion hint, degrade perception confidence during unfavorable motion. **All of the above are now implemented (E2-E7)** — see this section's own update note above and `docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md`. The forbidden list immediately below remains completely unimplemented and unplanned throughout.

Level 4 MUST NOT, ever, in this repository: estimate authoritative vehicle/agent pose, estimate global position, estimate velocity as localization state, estimate IMU biases, implement visual odometry, VIO, or SLAM, create a world map, perform world-frame persistence, plan trajectories, calculate vehicle clearance, know vehicle geometry, issue movement commands, know actuator types, or contain aerial/ground/marine modes.

**Boundary with `state_estimation_engine`:** there is zero dependency between `depth_perception_engine` and `state_estimation_engine` in either direction, and none is introduced by this pass. `depth_perception_engine` does not import, reference, or assume the existence of `state_estimation_engine` anywhere — not in source, not in `pyproject.toml`, not in an optional extra. A future integration (if one is ever built) is a downstream consumer's concern: something outside this repository may read `DepthPerceptionResult`/a future Level 4 temporal result and something outside this repository may produce a `MotionHint` from `state_estimation_engine`'s own IMU pipeline — but `depth_perception_engine` itself must never import from it, call into it, or special-case its presence. `tests/test_level4_architecture_guards.py` greps for the literal string `state_estimation_engine` anywhere under `src/` and fails if found (a documentation *mention* of the boundary, like this paragraph, lives in `docs/`, not `src/`, so it does not trip the guard).

## 4. What E1 actually built

```
src/depth_perception_engine/
├── temporal/                 # NEW package — interfaces only, mirrors geometry/'s E1 precedent
│   ├── __init__.py           # exports MotionHint only; explicit "E1 contracts only" banner
│   └── types.py              # MotionHint — the one contract, see docs/LEVEL4_CONTRACTS.md
└── models/result.py           # StereoObservation gains one Optional field (motion_hint);
                                # nothing else changes — same pattern as the E1-era `calibration`
                                # field addition (docs/LEVEL3_ARCHITECTURE.md)
```

Nothing else in `src/depth_perception_engine/` was touched. `PipelineConfig`, `DepthPerceptionResult`, `DepthPerceptionPipeline`, and every Level 0-3 stage module are byte-for-byte unchanged.

### Why this is the minimal correct E1 footprint

- **`temporal/` as a new top-level package, not inside `models/`** — same reasoning `geometry/` used at Level 3 E1: this represents a genuinely distinct future capability tier, not another output shape for the existing pipeline. Keeping it physically separate makes "this is not real yet" legible from the directory structure alone.
- **Only `MotionHint` is frozen as code.** Level 3 E1 froze four result types (`PointCloud`, `ObstacleCloud`, `FreeSpaceRays`, `GeometryMetrics`) with zero producers — but their shapes were already fully determined by pre-existing depth/disparity math (an organized cloud is obviously `(H, W, 3)` because `depth_map` is already `(H, W)`; a filtered obstacle set is obviously `(N, 3)`). Level 4's output-side concepts (a temporal-consistency result, a temporal-quality classification, a bounded-history entry) do **not** have that same pre-existing determinism — their shape is a real design decision that belongs to whichever phase actually designs the algorithm consuming them (E2+), not to E1 guessing ahead of that design. `MotionHint` is different: it is an *input* contract, and the task itself already specifies its natural minimal shape (timestamp, angular rate, validity, frame) closely enough that freezing it now is a genuine architecture decision, not a guess. See `docs/LEVEL4_CONTRACTS.md` for the full reasoning and for how the deferred output-side concepts are documented instead of coded.
- **`StereoObservation.motion_hint` is additive only**, defaulting to `None`, not consumed by `process_observation()` — identical discipline to the `calibration` field Level 3 E1 added for the same reason (reserve a place for a future caller without requiring a public API change later). Every existing construction of `StereoObservation(left_image=..., right_image=...)` continues to work unmodified.
- **Zero `PipelineConfig` fields added.** Level 3 E1 also added zero `PipelineConfig` fields — `enable_geometry` was not added until E3, once a real producer (`PointCloudBuilder`) existed to gate. Adding `enable_temporal` now, ahead of any producer it would gate, would be exactly the "large configuration surface prematurely" this phase's own instructions warn against. The planned future fields (`enable_temporal`, history length, timestamp-gap limits, consistency thresholds) are designed on paper in `docs/E2_TEMPORAL_HISTORY_PLAN.md`, not coded here.
- **Zero `DepthPerceptionResult` fields added.** Mirrors the same reasoning: Level 3 E1 did not add `DepthPerceptionResult.geometry` either — that came in E3 once `PointCloudBuilder` existed. A future additive field (tentatively `DepthPerceptionResult.temporal`, holding a not-yet-designed result type) is documented as planned in `docs/LEVEL4_PUBLIC_API.md`, not added now.
- **Nothing new is exported from the top-level `depth_perception_engine` package.** `temporal.MotionHint` is reachable only via `from depth_perception_engine.temporal import MotionHint` — same Tier 3 discipline as `geometry.*`/`calibration.contracts.*`/`frames.*`. Promoting an unproduced, unconsumed contract to the top-level namespace would overclaim capability this repository does not have.

## 5. Raw vs. temporal outputs — how the distinction will be represented (not implemented)

Level 3 outputs (`geometry`, `geometry_body`, `obstacle_cloud`, `free_space_rays`, `geometry_metrics`) remain the authoritative single-frame observation and are not touched by this pass. When a future phase adds temporal interpretation, the frozen intent is:

- A future temporal result is carried on a **new, additive, `Optional`, default-`None`** field of `DepthPerceptionResult` — never a replacement of an existing field, never a mutation of `geometry`/`geometry_body` in place. This is the exact pattern every Level 3 phase already used (E3's `geometry`, E4's `geometry_body`, E5's `obstacle_cloud`/`free_space_rays`/`geometry_metrics` all coexist; none replaced an earlier one).
- The raw Level 3 fields remain readable and meaningful with Level 4 fully disabled, and remain readable and meaningful *alongside* a populated temporal field once Level 4 exists — a caller must always be able to compare "what the sensor showed this frame" against "what the temporal layer believes" for debugging, benchmarking, and failure analysis.
- No field is silently replaced by a "smoothed" version of itself. This is a hard architectural rule, not a style preference — see section 7's temporal safety semantics.

## 6. History ownership (decision, not implementation)

```
DepthPerceptionPipeline
        |
        +-- single-frame Level 3 stages (unchanged)
        |
        +-- optional Level 4 temporal component (NOT built in E1)
                |
                +-- bounded recent history (NOT built in E1; belongs to E2)
```

`DepthPerceptionPipeline` remains the one stateful, reusable entry point — this is unchanged and is not redesigned. A future Level 4 temporal component is a construction-time-optional collaborator the pipeline owns (mirroring how it already owns `ThreatAssessor`'s cross-frame EMA/debounce state today), not a second top-level class and not a `DepthPerceptionEngine`. The temporal component itself — once it exists — owns the bounded history buffer; the pipeline does not reach into raw history directly, the same way it does not reach into `ThreatAssessor`'s internal EMA state directly today.

History must never grow indefinitely. Exact bounding semantics (max length, eviction policy, what happens on a timestamp gap/duplicate/out-of-order arrival) are **E2's job to define and implement**, not E1's — see `docs/E2_TEMPORAL_HISTORY_PLAN.md` for the full list of open questions E2 must resolve before writing any buffer code.

**Implemented at E2:** `temporal.TemporalHistory` (`temporal/history.py`) is exactly this component — bounded by both record count (`PipelineConfig.temporal_max_records`) and time window (`PipelineConfig.temporal_max_age_s`), whichever is tighter; timestamp gaps beyond `PipelineConfig.temporal_gap_limit_s` clear old history and start a fresh sequence; duplicate/older/out-of-order timestamps are rejected without mutating existing history. `DepthPerceptionPipeline` owns exactly one instance, constructed in `__init__` only when `PipelineConfig.enable_temporal` is `True`, exposed read-only via the new `.temporal_history` property. The pipeline never reaches into raw history directly — it constructs one `temporal.TemporalRecord` per frame and calls `TemporalHistory.admit()` once; every chronology decision lives inside `TemporalHistory` itself. See `docs/E2_TEMPORAL_HISTORY_PLAN.md` for the full per-decision record.

## 7. Temporal safety semantics (frozen rule, not implemented)

Frozen explicitly, to bind every future E2+ phase:

> **HISTORY MAY SUPPORT WEAK CURRENT EVIDENCE. HISTORY MUST NEVER OVERRIDE STRONG CONTRADICTORY CURRENT EVIDENCE.**

A wall observed consistently for several frames may support a temporarily weak/invalid observation of that same wall. But if a new strong surface suddenly appears where history showed free space, temporal smoothing must not erase the new surface — recency and strength of current evidence must win over historical priors whenever they conflict.

> **UNKNOWN != FREE.**

A pixel/region with no current valid measurement is `UNKNOWN`, not `FREE`, regardless of what history shows. Historical observations must never convert currently-unknown space into freshly-reported free space without explicitly representing the age and reduced confidence of that historical evidence — an aged, historical "this was free 40 frames ago" must remain visibly distinct from "this is free right now," never silently collapsed into the same value. This directly extends Level 3's own already-enforced invariant (`docs/DATA_CONTRACTS.md`'s spatial-evidence table: unknown space produces no obstacle point and no ray, never inferred either way) into the temporal dimension.

Neither rule was implemented by any code at E1 — there was no history to violate them yet. They were frozen here so E2+ would be designed against them from the start, not retrofitted later.

**Fulfilled at E7:** both rules are now enforced by real code, not just stated. "History may support weak current evidence, never override strong contradictory current evidence" — `temporal.persistence.TemporalPersistenceTracker` always classifies a currently-occupied cell as `NEW`/`PERSISTENT` based on that cell's own current evidence; a contradicting historical value never suppresses or delays that classification, it only resets the cell's own support count (`tests/test_temporal_persistence.py::TestBasicClassification::test_contradiction_resets_to_new_not_suppressed`). "UNKNOWN != FREE" — `temporal.persistence.TemporalPersistenceCellState` has exactly four codes and none of them means or implies FREE space; an expired or absent cell reverts to `NO_EVIDENCE`, never to a fabricated "known clear" value (`tests/test_temporal_persistence.py::TestUnknownNeverFree`). See `docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md` sections 1, 5, and 6 for the complete record.

## 8. Motion-hint boundary (frozen, see docs/LEVEL4_CONTRACTS.md for the contract itself)

```
Allowed (future):                          Forbidden (always):

angular measurements                        IMU
      |                                       |
      v                                       v
short interval rotation                pose estimator
      |                                       |
      v                                       v
align/compare recent perception        position/velocity output
      |
      v
temporal consistency
```

`temporal.MotionHint` (this pass) carries raw angular velocity only — never an already-integrated rotation, never a translation. Integration of angular velocity into a short-duration rotation, if and when it happens, is owned by the future E5 temporal-consistency consumer, not by `MotionHint` itself and not by any code added in this pass. See `docs/LEVEL4_CONTRACTS.md` for the full field-by-field rationale.

## 9. Frame semantics (reused, not reinvented)

`MotionHint.frame_id` reuses `frames.FrameId` exactly as-is — no new frame constant was added, no new axis convention was invented, and no drone/ground/marine-specific frame assumption exists anywhere. `frames.FrameId.BODY` is documented as the expected value (the physically natural mounting frame for a body-mounted IMU, and the only frame this repository's existing body-frame convention already supports without requiring a further, uncalibrated camera-to-IMU extrinsic), but `MotionHint.frame_id` is not hardcoded to it — a `MotionHint` must always state its own frame explicitly, matching `frames.RigidTransform`'s existing convention.

## 10. Configuration design

Zero new `PipelineConfig` fields in this pass — see section 4's reasoning. Planned future fields (`enable_temporal`, a history-length bound, a timestamp-gap limit, a consistency threshold) are named and scoped, not coded, in `docs/E2_TEMPORAL_HISTORY_PLAN.md`. None of the planned fields encode a platform mode (`mode = aerial`, `mode = ground`, `mode = marine`) — no such concept exists anywhere in this design.

## 11. Reset / lifecycle contract

`DepthPerceptionPipeline.reset()` already exists and is unchanged by this pass — it clears `ThreatAssessor`'s cross-frame EMA/debounce state and resets `frames_processed`/`last_confidence`/`last_processing_time_ms`, raising `RuntimeError` if called after `close()`. No new lifecycle method was needed or added: this satisfies section 11's request ("determine whether the current pipeline already has an appropriate reset/lifecycle mechanism" — it does).

**Fulfilled at E2:** `reset()` now also calls `TemporalHistory.clear()` when temporal history is enabled, so the next `process()` call after a `reset()` behaves as the start of a brand-new temporal sequence with zero historical influence — exactly like `ThreatAssessor`'s state is cleared. `reset()`'s pre-existing Level 3 behavior (clearing `ThreatAssessor` state, leaving calibration/config/rectification maps untouched) is unchanged; regression-tested by `tests/test_temporal_history.py::TestPipelineIntegration::test_reset_clears_temporal_history`/`test_reset_does_not_affect_ordinary_level3_lifecycle`.

## 12. Failure/degradation semantics

E2 status per row (updated from E1's "not yet implemented" table — most rows are now implemented as pure timestamp-chronology bookkeeping; the motion-hint *interpretation* rows remain future, since E2 performs no motion-hint algorithm):

| Condition | E1-frozen requirement | E2 status |
|---|---|---|
| No history yet (first frame(s) after construction/reset) | Operate on Level 3 evidence alone; never block or error | **Implemented** — `TemporalHistory.latest is None`/`len() == 0`; `process()` returns a full Level 3 result regardless |
| Insufficient history for a temporal judgement | Degrade to a documented state, never fabricate | Not yet applicable — no algorithm reads history yet (E3+) |
| Invalid timestamp | Reject/flag for temporal purposes only; never affect Level 3 output | **Implemented** — `TemporalAdmissionStatus.REJECTED_INVALID_TIMESTAMP`; `DepthPerceptionResult`'s other fields are computed identically regardless |
| Timestamp discontinuity (large gap) | Treat as history-invalidating, not silently bridged | **Implemented** — `PipelineConfig.temporal_gap_limit_s`; `TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE` |
| Missing motion hint (`None`) | Operate on stereo evidence alone | **Implemented** — legal, admitted normally, `TemporalRecord.motion_hint = None` |
| Invalid motion hint (`MotionHint.valid is False`) | Treated identically to missing | **Confirmed unchanged in effect**: E2 does not branch on `.valid` at all — an invalid hint is stored and admitted exactly like a valid one (no special-casing either way); a future consumer, not E2, is where `.valid` must actually be honored |
| Motion hint outside expected time interval | Discarded for alignment, logged, never misapplied | Not yet applicable — no interval-alignment algorithm exists (E5) |
| Excessive angular motion | Degrade temporal confidence | Not yet applicable — no motion-hint algorithm exists (E5) |
| Weak stereo geometry | Reflected via `GeometryQuality`; temporal layer must not override upward | **Implemented** — `TemporalRecord.geometry_quality` records `classify_geometry_quality()`'s verdict honestly, admitted regardless of value (Decision 9/12) |
| Temporal disagreement (history vs. current) | Current strong evidence wins | Not yet applicable — no comparison algorithm exists yet; structurally guaranteed not to be violated because none exists |
| Recovery after degraded input | Recover cleanly, no stuck state | **Implemented** — proven structurally: `admit()` never inspects quality/confidence, so no degraded state can ever accumulate (`tests/test_temporal_history.py::TestDegradationRecovery`) |
| Persistence classification (E7) | Distinguish persistent/transient geometric evidence, deterministically, never fabricating free space | **Implemented** — `temporal.persistence.TemporalPersistenceTracker`; see `docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md` |
| E6 `UNRELIABLE` frame reaching E7 | Must not create or reinforce persistence | **Implemented** — `TemporalPersistenceTracker.update()` skips every per-cell update on `MotionAwareReliabilityState.UNRELIABLE`, returning the tracker's own unchanged prior snapshot (`tests/test_temporal_persistence.py::TestReliabilityGating`) |

The core principle from E1 still holds and is now enforced, not just stated: **the temporal layer is additive insurance, not a single point of failure** — `tests/test_temporal_history.py::TestZeroRegressionOnLevel3Output` (E2) and `tests/test_temporal_persistence.py::TestPipelineOutputsUnchanged` (E7) prove enabling/disabling each Level 4 phase produces byte-identical Level 0-3 numeric output either way.

## 13. Performance / memory contract

**Implemented at E2:** `TemporalRecord` retains exactly 4 small fields — no disparity/depth array, mask, `PointCloud`, `ObstacleCloud`, or `FreeSpaceRays` is ever stored (see `docs/E2_TEMPORAL_HISTORY_PLAN.md`'s memory-decision table for the per-field justification). History is bounded by both count (`temporal_max_records`) and age (`temporal_max_age_s`); `tests/test_temporal_history.py::TestMemoryBounding` proves both the hard bound (1000 insertions into a 5-record buffer never exceeds 5) and actual release (an evicted record's only remaining reference is the test's own local variable, verified via `sys.getrefcount`). Temporal-admission latency is logged separately at `DEBUG` level (`pipeline.py`'s `"Temporal admission stage: %.2f ms"`, mirroring every other per-stage log line) while still being included in the overall `processing_time_ms` total, matching every other Level 3 stage's existing convention. No Jetson-specific code was added; Jetson Orin Nano remains a deployment-target consideration, not a code branch.

## 14. Public API

See `docs/LEVEL4_PUBLIC_API.md` for the full Tier 1/2/3 accounting. Summary: `StereoObservation.motion_hint` (E1, now consumed by `process_observation()` as of E2); `DepthPerceptionPipeline.process()` gained an additive `motion_hint` parameter and a new `.temporal_history` read-only property (E2); `DepthPerceptionResult` gained one additive field, `temporal_admission_status` (E2); `PipelineConfig` gained four additive fields, `enable_temporal`/`temporal_max_records`/`temporal_max_age_s`/`temporal_gap_limit_s` (E2). `temporal.MotionHint`/`TemporalRecord`/`TemporalHistory`/`TemporalAdmissionStatus` are all Tier 3 (not exported at top level). No other public surface changed; no `DepthPerceptionEngine` class exists or is planned; `DepthPerceptionPipeline` remains the one canonical processing object.

## 15. What Phase E2 explicitly did not do

Implement temporal filtering, temporal fusion, IMU integration/compensation, rotation-from-gyro computation, persistence classification, dynamic-object reasoning, optical flow, feature tracking, VIO, SLAM, localization, mapping, planning, control, a simulated IMU, or any temporal-consistency/temporal-quality algorithm. Did not change any Level 0-3 stereo/disparity/depth/geometry/obstacle/traversability algorithm, `mp01_perception`, or `state_estimation_engine`. Did not add a fault counter for repeated chronology violations (no such counter was frozen at E1, and E2 does not invent one, per this phase's own instruction). Verified: `pytest tests/ -q` — see `docs/IMPLEMENTATION_STATUS.md`'s Level 4 Phase E2 addendum for the exact before/after count.
