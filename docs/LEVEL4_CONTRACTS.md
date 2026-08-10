> **Superseded for current usage (2026-08-09, Phase E8):** `docs/LEVEL4_CANONICAL_REFERENCE.md` is now the single authoritative, current per-concept contract reference. This document remains as the historical per-phase decision record.

# Level 4 contracts (Phases E1-E2; see docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md for E3-E7)

**LEVEL 4 CURRENT STATUS: E7 — DETERMINISTIC, PER-CELL TEMPORAL-PERSISTENCE CLASSIFICATION IMPLEMENTED.**

Companion to `docs/LEVEL4_ARCHITECTURE.md` (the "why"/"what's next") — this document is the per-contract reference, mirroring `docs/LEVEL3_CONTRACTS.md`'s role for Level 3. See `docs/E2_TEMPORAL_HISTORY_PLAN.md` for the full record of the twelve E2 design decisions these contracts implement; this document's own body below was not rewritten for every phase after E2 (each later phase — E3 `docs/E3_IMPLEMENTATION_PLAN.md`, E4 `docs/E4_IMPLEMENTATION_PLAN.md`, E5 `docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md`, E6 `docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md`, E7 `docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md` — is the authoritative per-contract reference for its own new types). The one addition below is the "Deliberately deferred" section's own update, since E7 resolved the "persistent/transient" candidate that section had left open since E1.

## `temporal.MotionHint` (`temporal/types.py`) — frozen this pass

| Property | Value |
|---|---|
| **Purpose** | An optional, externally supplied, short-duration rotational motion measurement — a PERCEPTION AID only. Carries a raw angular velocity reading (and the moment it applies to) so a future Level 4 temporal component can estimate a short-duration rotation between two frames and use it to align/compare recent perception evidence. Never a pose, velocity, or position estimate. |
| **Producer** | None in this repository today. A future producer may be a simulated-IMU utility living in tests/examples/simulation tooling (explicitly not built in this pass — see `docs/LEVEL4_SIMULATED_IMU.md`) or a real IMU driver upstream of this library. |
| **Consumer** | None in this repository today. `StereoObservation.motion_hint` (see below) is the one place a `MotionHint` can currently be attached to a frame — it is carried through unread. |
| **Units** | `angular_velocity_rad_s`: radians per second, 3-vector (rotation rate about `frame_id`'s X, Y, Z axes). `timestamp`: opaque, caller-defined — same convention as every other timestamp in this library (no unit conversion or synchronization performed here). |
| **Coordinate frame** | `frame_id: str` — any `frames.FrameId`-style string, explicitly required (no default, no inference). Expected value: `frames.FrameId.BODY` (a body-mounted IMU is the physically natural case, and the only one this repository's existing body-frame convention already supports without a further, uncalibrated camera-to-IMU extrinsic) — not hardcoded to it. No new `FrameId` constant was added; `MotionHint` reuses the exact naming convention `frames.RigidTransform` already established. |
| **Timestamp semantics** | Required `float` (not `Optional`) — a `MotionHint`'s entire purpose is anchoring "when" an angular rate applies. "No motion hint available" is represented by omitting the value entirely at the point of use (e.g. a future `Optional[MotionHint] = None` parameter), not by constructing a `MotionHint` with a placeholder/sentinel timestamp. |
| **Validity semantics** | `valid: bool = True`. `True` (default): treat at face value. `False`: the producer itself flagged this specific measurement as untrustworthy (sensor fault, saturation, an out-of-range value the producer chose to flag rather than withhold). Distinct from "not supplied" — a future consumer must treat `valid=False` identically to "no motion hint supplied," never partially trusting it. |
| **Immutability** | `@dataclass(frozen=True, slots=True)` — same convention as every other Level 3/4 contract type (`RigidTransform`, `PointCloud`, etc.). A producer allocates a fresh instance per reading; nothing mutates one in place. |
| **Relationship to existing Level 3 contracts** | Reuses `frames.FrameId` directly — no new frame invented. Deliberately **not** merged into or built on top of `frames.RigidTransform`: a `RigidTransform` is a static mounting/extrinsic relationship between two fixed frames (e.g. camera-to-body), while a `MotionHint` is a time-varying rate measurement — different concepts with different lifetimes, not the same type wearing two hats. |

### Why raw angular velocity, not an integrated rotation

The task that motivated this contract explicitly raised the alternative: a `MotionHint` could instead carry an already-integrated rotation (e.g. a quaternion or rotation-matrix delta) computed by the producer. This contract deliberately chose **raw angular velocity** instead, for three reasons:

1. **Matches the natural IMU output.** A gyroscope reports angular velocity; integrating it into a rotation delta requires knowing the exact interval to integrate over, which this contract alone cannot know (it only knows the interval *between* two `MotionHint` readings, not which two frame timestamps a future consumer will want to align against).
2. **Keeps integration ownership unambiguous.** If `MotionHint` carried an already-integrated rotation, "integrated over what interval, by whom, using what method" would be an implicit, undocumented assumption baked into every producer. By carrying only the raw rate, the integration step — and the interval it's computed over — is an explicit, visible decision the future E5 consumer makes, using the same timestamps it already has (frame timestamps and `MotionHint.timestamp`), not a hidden assumption inside this type.
3. **One representation, not two.** The task explicitly warned against creating "two competing representations unless genuinely necessary." Freezing raw angular velocity now and documenting that integration is the future consumer's job avoids ever needing a second, integrated-rotation sibling type.

**Frozen ownership statement:** integrating `angular_velocity_rad_s` into a short-duration rotation is the responsibility of whatever future component consumes `MotionHint` (anticipated: an E5 temporal-consistency component) — not `MotionHint` itself, not `StereoObservation`, not `DepthPerceptionPipeline` in this pass.

### Why no linear velocity/acceleration/position field

Explicitly and permanently out of scope for this type, restated from `temporal/types.py`'s own docstring: a translation-capable field on `MotionHint` would invite exactly the pose/localization creep Level 4 is architecturally forbidden from performing (`docs/LEVEL4_ARCHITECTURE.md` section 3). Translation cannot be safely inferred from a gyro-only measurement, and this type must not imply otherwise by its own shape. Any future need for translation-aware motion information is a `state_estimation_engine`-side concern, explicitly outside this repository's boundary (`docs/LEVEL4_ARCHITECTURE.md` section 3) — not a field to add here.

### Why no `source`/`is_simulated` field

A `MotionHint` must be indistinguishable in shape whether it came from a simulated-IMU test utility or a real IMU driver — see `docs/LEVEL4_SIMULATED_IMU.md`. A `source` field would be an open invitation for a future implementation to branch on it (`if simulated_imu: ...`), which this phase's own instructions explicitly forbid in core algorithm code. If a caller needs to know a `MotionHint`'s provenance for its own bookkeeping, that is the caller's concern to track alongside the value, not this library's concern to encode into the contract.

## `StereoObservation.motion_hint` (`models/result.py`) — additive field, this pass

Not a new type — a new `Optional[MotionHint] = None` field on the existing, already-Tier-1 `StereoObservation`, following the exact precedent `calibration: Optional[StereoCalibration] = None` set at Level 3 E1:

- **Reserved, not consumed.** `DepthPerceptionPipeline.process_observation()` does not read `motion_hint` — it still unpacks exactly `left_image`, `right_image`, `left_timestamp`, `right_timestamp`, unchanged since before this pass (verified by `tests/test_temporal_contracts.py::TestStereoObservationMotionHintNotConsumed`).
- **Backward compatible by construction.** Every existing call `StereoObservation(left_image=..., right_image=...)` (with or without `calibration`/`frame_id`) continues to work unmodified — `motion_hint` defaults to `None` like every other optional field on this type.
- **`None` means "no motion hint for this frame," not an error.** A future Level 4 consumer must treat an observation with `motion_hint=None` as "operate on stereo evidence alone" — see `docs/LEVEL4_ARCHITECTURE.md` section 12's degradation table.

## `temporal.TemporalRecord` (`temporal/types.py`) — frozen at Phase E2

| Property | Value |
|---|---|
| **Purpose** | One minimal chronology entry in a `temporal.TemporalHistory` buffer. Records that a frame occurred, when, and how good its evidence was — nothing about the frame's actual pixel-level content. |
| **Producer** | `DepthPerceptionPipeline.process()`, once per frame, only when `PipelineConfig.enable_temporal` is `True`. |
| **Consumer** | `temporal.TemporalHistory` (admission/eviction bookkeeping only). No E3+ algorithm consumes it yet. |
| **Units/frame** | No units or frame of its own — it carries a `timestamp` (opaque, caller-defined, same convention as everywhere else in this library) and, if present, a `motion_hint` whose own frame/units are exactly `MotionHint`'s (see above). |
| **Fields** | `timestamp: Optional[float]`; `confidence: float` (Level 0-2's `DepthPerceptionResult.confidence`, always present); `geometry_quality: Optional[str]` (one of `geometry.GeometryQuality`'s three values, or `None` if not computed this call); `motion_hint: Optional[temporal.MotionHint]`. |
| **Validity semantics** | `timestamp` being `None`/`NaN`/`±Inf` is a rejection condition for `TemporalHistory.admit()`, not a construction-time error on `TemporalRecord` itself — a record can be constructed with any value; `admit()` alone decides admissibility. This centralizes every chronology invariant in one place (`docs/E2_TEMPORAL_HISTORY_PLAN.md`'s "Buffer behavior" section). |
| **Immutability** | `@dataclass(frozen=True, slots=True)`, same convention as every other contract type in this library. |
| **Relationship to existing contracts** | Deliberately **not** a `DepthPerceptionResult` and does not embed one — see the memory-decision table in `docs/E2_TEMPORAL_HISTORY_PLAN.md` for exactly why each of its 4 fields, and only those 4, earns a place. Reuses `geometry.GeometryQuality` and `temporal.MotionHint` directly rather than inventing parallel types. |

## `temporal.TemporalHistory` / `temporal.TemporalAdmissionStatus` (`temporal/history.py`) — frozen at Phase E2

Not a data contract — a stateful component, in the same sense `obstacles.ThreatAssessor`/`traversability.SceneInterpreter` are stateful engine classes rather than dataclasses. Full interface and behavior contract: `docs/E2_TEMPORAL_HISTORY_PLAN.md`'s "Buffer behavior"/"Time-window eviction" sections and the twelve per-decision writeups above them. Summary:

- `admit(record) -> str` — the only place any timestamp-chronology decision is made. Returns one of `TemporalAdmissionStatus.ACCEPTED`/`ACCEPTED_NEW_SEQUENCE`/`REJECTED_INVALID_TIMESTAMP`/`REJECTED_OLDER_TIMESTAMP`/`REJECTED_DUPLICATE_TIMESTAMP` — plain string constants, mirroring `obstacles.ThreatAssessor.CLEAR/CAUTION/BLOCKED/NO_DATA` and `geometry.GeometryQuality`'s existing precedent, not a new Enum type and not a competing health system (`PipelineHealth` stays lifecycle-only; `GeometryQuality` stays a geometry-evidence classification; `TemporalAdmissionStatus` answers the third, disjoint question of chronology admission).
- `records: Tuple[TemporalRecord, ...]`, `latest: Optional[TemporalRecord]`, `__len__`, `clear()` — the rest of the minimal interface this phase's own instructions required.
- Construction validates `max_records >= 1`, `max_age_s > 0`, `gap_limit_s > 0`.

## `DepthPerceptionResult.temporal_admission_status` (`models/result.py`) — additive field, Phase E2

`Optional[str]`, default `None`. `None` unless `PipelineConfig.enable_temporal` is `True`, in which case it is always one of `TemporalAdmissionStatus`'s five constants for every processed frame. Pure metadata about this frame's own chronology-admission outcome — never a reinterpretation of `disparity_map`/`depth_map`/`geometry`/etc., all of which are completely unaffected by temporal-history admission either way (`tests/test_temporal_history.py::TestZeroRegressionOnLevel3Output`). This is how Decision 4 ("represent/report the temporal rejection") is satisfied without inventing a second competing health system — see `docs/E2_TEMPORAL_HISTORY_PLAN.md`.

## `DepthPerceptionPipeline.process()`'s `motion_hint` parameter and `.temporal_history` property — additive, Phase E2

`process(..., motion_hint: Optional[temporal.MotionHint] = None)` — new, additive, keyword parameter; every pre-E2 call site is unaffected. `process_observation()` now forwards `StereoObservation.motion_hint` into it (was reserved/unconsumed at E1). Neither parameter nor the resulting association triggers any computation beyond attaching the value to a `TemporalRecord` — no integration, no alignment, no validation beyond `MotionHint.__post_init__`'s own checks.

`.temporal_history: Optional[temporal.TemporalHistory]` — new read-only property, mirroring `.config`/`.calibration`'s existing exposure pattern. `None` unless `PipelineConfig.enable_temporal` is `True`.

## Deliberately deferred (documented, not coded) — candidate output-side contracts

The Level 4 task prose named several candidate concepts this phase's own instructions say are "NOT mandatory" and warn against freezing "merely because [they were] listed." Each is addressed here narratively so a future phase has this reasoning available, without inventing a concrete type ahead of the algorithm design that should actually determine its shape:

- **A temporal-consistency / stabilized-geometry result** (raw-vs-temporal distinction, `docs/LEVEL4_ARCHITECTURE.md` section 5). Anticipated to eventually live as one new, additive, `Optional`, default-`None` field on `DepthPerceptionResult` (tentatively `temporal`), coexisting with — never replacing — `geometry`/`geometry_body`/`obstacle_cloud`/`free_space_rays`/`geometry_metrics`. Not frozen now because its internal shape (what exactly gets "stabilized," at what granularity, with what confidence representation) is a real algorithm design decision, not yet made.
- **A temporal-quality classification distinct from `geometry.GeometryQuality`.** `GeometryQuality` itself is now reused directly by `TemporalRecord.geometry_quality` (see above) rather than duplicated — no separate "temporal quality" enum was needed for E2's own purposes. A genuinely new temporal-specific classification (e.g. "is this sequence internally consistent over the last N frames") still depends on an E3+ algorithm that doesn't exist yet, and remains deferred for the same reason as before: defining tier boundaries now would be guessing at semantics an unbuilt algorithm hasn't produced.
- **A bounded-history entry shape** — **resolved at E2.** This was the one deferred item E2's own charter was to resolve; see `temporal.TemporalRecord` above and `docs/E2_TEMPORAL_HISTORY_PLAN.md`'s memory-decision table for the answer (4 small fields, no full-resolution product retained).

## Deferred items above, as actually resolved (E3-E7)

Each candidate this section named above was resolved as its own separate, additive field — never as one unified `temporal` composite field, and never inside `TemporalRecord` itself (see `docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md` section 5 for why E7 in particular does not extend `TemporalRecord` again): `DepthPerceptionResult.temporal_consistency` (E3, `temporal.TemporalConsistency`), `temporal_stabilization` (E4, `temporal.TemporalStabilization`), `rotation_compensation_status` (E5, `temporal.RotationCompensationStatus`), `motion_aware_reliability` (E6, `temporal.MotionAwareReliability`), and `temporal_persistence` (E7, `temporal.TemporalPersistence` — the "distinguish persistent/transient geometric evidence" candidate this section originally left open at E1). A genuinely new temporal-specific quality classification distinct from `geometry.GeometryQuality` was never needed — every phase reused `GeometryQuality`/`TemporalConsistencyState`/etc. directly where applicable rather than inventing a parallel system.

## Not exported at the top level

`temporal.MotionHint`/`TemporalRecord`/`TemporalHistory`/`TemporalAdmissionStatus` are reachable only via `from depth_perception_engine.temporal import ...` — not from the package root. See `docs/LEVEL4_PUBLIC_API.md`.
