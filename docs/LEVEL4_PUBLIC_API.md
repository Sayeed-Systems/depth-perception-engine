> **Superseded for current usage (2026-08-09, Phase E8):** `docs/LEVEL4_CANONICAL_REFERENCE.md` sections 10-11 are now the single authoritative, current `PipelineConfig`/`DepthPerceptionResult` field reference, organized by concept. This document remains as the historical per-phase API-diff record.

# Public API impact (Level 4, Phases E1-E2, plus E7 below)

Companion to `docs/PUBLIC_API.md` (the general, authoritative Tier 1/2/3 reference) — this document records exactly what Level 4 changed there, mirroring `docs/LEVEL3_PUBLIC_API.md`'s role at Level 3. E3-E6 each made analogous additive changes (one new `DepthPerceptionResult` field, one or two new `PipelineConfig` flags/thresholds, new Tier 3 `temporal.*` symbols) not individually tabulated in this document — see `docs/E3_IMPLEMENTATION_PLAN.md`, `docs/E4_IMPLEMENTATION_PLAN.md`, `docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md`, `docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md` for each phase's own complete API accounting. E7's own table is below.

## What changed at Phase E7

| Surface | Change |
|---|---|
| `DepthPerceptionResult` (Tier 1) | Gained one additive field, appended last: `temporal_persistence: Optional[temporal.TemporalPersistence] = None`. |
| `PipelineConfig` (Tier 1) | Gained four additive fields: `enable_temporal_persistence: bool = False` (nested under both `enable_temporal` and `enable_motion_aware_reliability`), `persistence_min_support_count: int = 2` (validated `>= 2`), `persistence_max_dropout_frames: int = 1` (validated `>= 0`), `persistence_expiration_absence_frames: int = 5` (validated `> persistence_max_dropout_frames`). |
| `temporal.TemporalPersistence`, `temporal.TemporalPersistenceState`, `temporal.TemporalPersistenceCellState`, `temporal.persistence.TemporalPersistenceTracker` | **New.** All Tier 3 — reachable only via `from depth_perception_engine.temporal import ...`, never from the package root. |
| `temporal.rotation_compensation.compensate_prior_geometry_with_payload` | **New**, additive sibling to E5's own `compensate_prior_geometry` (same module, shares its exact reprojection math via a refactored private helper — `compensate_prior_geometry`'s own pre-E7 behavior is unchanged, proven by the full, unmodified E5 test suite). Tier 3. |
| `DepthPerceptionPipeline.__init__`/`.process()`/`.reset()` | No signature change. `__init__` now additionally constructs a `TemporalPersistenceTracker` when all three E7 gates are `True`; `process()` calls its `.update()` once per frame in that case; `reset()` also calls its `.clear()`. |

`depth_perception_engine.__all__` is unchanged by E7 — every addition is either a new field on an already-Tier-1 type or a Tier 3 symbol within `temporal`. `tests/test_public_api.py`'s `INTERNAL_SYMBOLS` regression list gained `"compensate_prior_geometry_with_payload"`, `"TemporalPersistence"`, `"TemporalPersistenceState"`, `"TemporalPersistenceCellState"`, `"TemporalPersistenceTracker"`.

## What changed at Phase E1

| Surface | Change |
|---|---|
| `StereoObservation` (Tier 1) | Gained one additive field: `motion_hint: Optional[temporal.MotionHint] = None`. Every existing construction/usage is unaffected — see `docs/LEVEL4_CONTRACTS.md`. |
| `temporal.MotionHint` | **New.** Tier 3 — reachable only via `from depth_perception_engine.temporal import MotionHint`, never from the package root. |

## What changed at Phase E2

| Surface | Change |
|---|---|
| `PipelineConfig` (Tier 1) | Gained four additive fields, all defaulted to preserve pre-E2 behavior exactly: `enable_temporal: bool = False`, `temporal_max_records: int = 30`, `temporal_max_age_s: float = 1.0`, `temporal_gap_limit_s: float = 0.5`. `__post_init__` validates the latter three; `enable_temporal=False` reproduces pre-E2 behavior exactly (no `TemporalHistory` constructed, `temporal_admission_status` stays `None`). |
| `DepthPerceptionResult` (Tier 1) | Gained one additive field, appended last: `temporal_admission_status: Optional[str] = None`. |
| `DepthPerceptionPipeline.process()` (Tier 1) | Gained one additive keyword parameter: `motion_hint: Optional[temporal.MotionHint] = None`. Every existing positional/keyword call site is unaffected. |
| `DepthPerceptionPipeline.process_observation()` (Tier 1) | Now forwards `observation.motion_hint` into `process()`'s new parameter — was reserved/unconsumed at E1, consumed (bookkeeping-only) as of E2. |
| `DepthPerceptionPipeline.reset()` (Tier 1) | Now also clears temporal history when enabled — no signature change, no change to its pre-existing Level 3 behavior. |
| `DepthPerceptionPipeline.temporal_history` | **New** read-only property, mirroring `.config`/`.calibration`. Returns `Optional[temporal.TemporalHistory]`. |
| `temporal.TemporalRecord`, `temporal.TemporalHistory`, `temporal.TemporalAdmissionStatus` | **New.** All Tier 3 — reachable only via `from depth_perception_engine.temporal import ...`, never from the package root. |

`depth_perception_engine.__all__` is unchanged by either phase — no new symbol was added to it (every E2 addition is either a new field/parameter/property on an *already*-Tier-1 type, or a Tier 3 symbol within `temporal`). `tests/test_public_api.py`'s `EXPECTED_ALL`/`TIER_1_SYMBOLS`/`TIER_2_SYMBOLS` sets required no edits; its `INTERNAL_SYMBOLS` regression list (Tier 3 leak guard) gained `"MotionHint"` at E1 and `"TemporalRecord"`/`"TemporalHistory"`/`"TemporalAdmissionStatus"` at E2, matching how every prior Level 3 phase extended that same list as new Tier-3 symbols were added (E2's `PointCloudBuilder`, E4's `transform_point_cloud`, E5's `build_obstacle_cloud`/`build_free_space_rays`/`build_geometry_metrics`, E6's `GeometryQuality`/`classify_geometry_quality`).

## Why `temporal.MotionHint` stays Tier 3

Identical reasoning to `docs/LEVEL3_PUBLIC_API.md`'s treatment of `geometry.*`/`calibration.contracts.*`/`frames.*` at Level 3 E1: nothing in this repository produces a `MotionHint` and nothing consumes one. Promoting an unproduced, unconsumed type to the top-level namespace would overclaim capability — a caller seeing `depth_perception_engine.MotionHint` would reasonably assume something in this library creates or reads one. Neither is true yet.

## No `DepthPerceptionEngine` class

Restated explicitly per this phase's own instruction: the canonical processing object remains `DepthPerceptionPipeline`. No `DepthPerceptionEngine` symbol was created, and none is planned — see `docs/PUBLIC_API.md`'s own "Naming: repository vs. class" section, unchanged and still authoritative.

## What Phase E2 built, vs. what was anticipated at E1

E1's own "what a future phase will need to add" list, checked against what E2 actually did:

- `PipelineConfig.enable_temporal: bool = False` — **built exactly as anticipated**, plus three sibling threshold fields (`temporal_max_records`/`temporal_max_age_s`/`temporal_gap_limit_s`) E1 had named but not designed — see `docs/E2_TEMPORAL_HISTORY_PLAN.md`'s Decision 1/7 for the resolved values and reasoning.
- `DepthPerceptionResult.temporal: Optional[<not-yet-named-type>] = None` — **not built**, and still correctly deferred: E1 anticipated this as the eventual home for a *temporal-consistency result* (stabilized/compared evidence). E2 built no such algorithm, so there is nothing to put there yet. What E2 *did* add instead — `temporal_admission_status: Optional[str] = None` — is a narrower, different thing: pure chronology-admission metadata, not a temporal interpretation of evidence. Both can coexist once a real temporal-consistency algorithm exists.
- `.process()`/`.process_observation()`'s call signature — **`process()` did gain one new keyword parameter** (`motion_hint`), which E1 had explicitly said it did *not* anticipate needing, since `StereoObservation.motion_hint` alone seemed sufficient. In practice, wiring `process_observation()` to actually forward the value required *somewhere* on `process()` to receive it — `StereoObservation` cannot hand a value to `process()` without `process()` accepting it. This is still additive/backward-compatible (a new defaulted keyword parameter), so no existing call site broke, but it is a small correction to E1's own prediction, recorded here rather than silently glossed over.

## What a future phase will still need to add here (not built at E2 either)

- `DepthPerceptionResult.temporal: Optional[<not-yet-named-type>] = None` — see above; depends on an E3+ temporal-consistency algorithm that doesn't exist yet.
- A temporal-quality classification distinct from `geometry.GeometryQuality` (if one ever proves necessary) — see `docs/LEVEL4_CONTRACTS.md`'s "Deliberately deferred" section.

## Future extension check (E1's own success criteria, re-verified at E2)

- **A simulated-then-real IMU swap without redesigning the public API:** still plausible, now exercised for real — `tests/test_temporal_history.py::TestMotionHintAssociation` constructs `MotionHint` values directly (no simulator exists) and proves they flow through unmodified to `TemporalRecord.motion_hint`; nothing in the path branches on provenance. See `docs/LEVEL4_SIMULATED_IMU.md`.
- **A future temporal result without redesigning `DepthPerceptionResult`:** still plausible — `DepthPerceptionResult` has now absorbed six additive fields across Level 3 (E3-E5) and Level 4 (E2) without a breaking change; a seventh (`temporal`) follows the identical, proven pattern.
