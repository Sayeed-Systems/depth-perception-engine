# Level 4 canonical reference (FROZEN)

**This is the single authoritative, current description of Level 4 — depth_perception_engine's temporal perception layer.** It supersedes `docs/LEVEL4_ARCHITECTURE.md`, `docs/LEVEL4_CONTRACTS.md`, and `docs/LEVEL4_PUBLIC_API.md` for describing current behavior (those documents remain, unmodified in substance, as the historical development-process record — see "Development history," below). Organized by concept, not by build order: nothing below is named after an internal development-phase label, because those labels were never part of the public API to begin with (no `PipelineConfig` field, `DepthPerceptionResult` field, or `temporal.*` class name contains one) — this document simply also never narrates by them.

Companion documents: `docs/LEVEL3_ARCHITECTURE.md`/`docs/LEVEL3_CONTRACTS.md`/`docs/LEVEL3_PUBLIC_API.md` (Level 3, unaffected — see "Level 3 compatibility" below), `docs/LEVEL4_HARDWARE_VALIDATION_PENDING.md` (what remains unverified against real sensors), `docs/LEVEL4_E8_INTEGRATION_REPORT.md` (the validation pass that produced this freeze).

## 1. What Level 4 is

Level 3 answers "what geometric environment evidence is visible right now." Level 4 answers "how trustworthy and temporally consistent is that evidence across recent observations, and which of it has been repeated over time versus just appeared or just vanished." Level 4 is **perception only** — never localization, mapping, planning, or control (see section 9).

`depth_perception_engine` remains standalone, ROS-free, platform-agnostic, and agent-agnostic throughout Level 4: nothing in it knows whether its caller is aerial, ground, or marine, and nothing in it depends on `state_estimation_engine` in either direction (both guarded structurally by `tests/test_level4_architecture_guards.py`, which scans all of `src/depth_perception_engine/`).

`DepthPerceptionPipeline` (`pipeline/pipeline.py`) remains the one canonical, stateful processing object — no `DepthPerceptionEngine` symbol exists or is planned. Every Level 4 capability is opt-in, gated by its own `PipelineConfig` flag(s), and additive: with every Level 4 flag left at its default (`False`), Level 4 costs nothing and changes nothing — `DepthPerceptionResult`'s Level 0-3 fields, and every Level 4 field, are `None`.

## 2. Current / raw evidence

The Level 3 fields Level 4 reads from and never modifies: `DepthPerceptionResult.disparity_map`, `depth_map`, `geometry` (camera-optical-frame point cloud), `geometry_body` (body-frame point cloud), `obstacle_cloud`, `free_space_rays`, `geometry_metrics`, `confidence`. Every Level 4 computation is strictly downstream and read-only of these — no Level 4 stage ever writes back into any of them, under any outcome, proven not just stated (`tests/test_level4_integration_e8.py::TestRawLevel3EvidenceUnchanged` runs a full multi-frame sequence with every Level 4 capability enabled and shows byte-identical Level 0-3 output against an otherwise-identical pipeline with Level 4 entirely disabled).

## 3. Temporal history

A bounded, deterministic chronology of recent per-frame summaries — infrastructure every other Level 4 concept builds on.

- **Gate:** `PipelineConfig.enable_temporal: bool = False`.
- **Component:** `temporal.TemporalHistory` (`temporal/history.py`), one instance owned by `DepthPerceptionPipeline`, admitting one `temporal.TemporalRecord` per frame via `.admit()`.
- **`temporal.TemporalRecord`:** `timestamp`, `confidence`, `geometry_quality` (reuses `geometry.GeometryQuality` directly), `motion_hint`, `depth_snapshot_m` (a decimated — `PipelineConfig.temporal_consistency_sampling_stride` — copy of `depth_map`, the one array-shaped piece of data retained; never a full-resolution array, mask, or point cloud).
- **Bounding:** `PipelineConfig.temporal_max_records` (hard count cap) and `temporal_max_age_s` (timestamp-based, never wall-clock), whichever is tighter.
- **Chronology outcome:** `DepthPerceptionResult.temporal_admission_status`, one of `temporal.TemporalAdmissionStatus.ACCEPTED`/`ACCEPTED_NEW_SEQUENCE`/`REJECTED_INVALID_TIMESTAMP`/`REJECTED_OLDER_TIMESTAMP`/`REJECTED_DUPLICATE_TIMESTAMP`. `ACCEPTED_NEW_SEQUENCE` — a gap beyond `PipelineConfig.temporal_gap_limit_s` — clears all prior history (and, if enabled, persistence state, see section 8) and starts fresh, rather than bridging the discontinuity.
- **Reset:** `DepthPerceptionPipeline.reset()` clears history completely (and persistence tracking, if enabled) — complete temporal amnesia; the next frame behaves as the first frame of a brand-new sequence.

## 4. Temporal consistency

A read-only comparison of the current frame's geometry against the single most recent comparable prior frame.

- **Component:** `temporal.compute_temporal_consistency()` (`temporal/consistency.py`), a pure function.
- **Output:** `DepthPerceptionResult.temporal_consistency: Optional[temporal.TemporalConsistency]` — `None` unless `enable_temporal` is `True` and this frame's timestamp was itself admitted to history. Fields: `state` (`temporal.TemporalConsistencyState.CONSISTENT`/`CONTRADICTORY`/`INSUFFICIENT_EVIDENCE`/`NOT_COMPARABLE`), `comparable_count`, `agreeing_count`, `disagreement_count`, `agreement_fraction` (`None`, never 0/0, when nothing comparable exists).
- **Thresholds:** `PipelineConfig.temporal_consistency_agreement_tolerance_m` (per-pixel agreement tolerance, metres), `temporal_consistency_min_agreement_fraction` (population threshold for `CONSISTENT` vs. `CONTRADICTORY`).
- **Safety rule:** structurally read-only — this stage cannot rewrite `depth_map`/`geometry`/anything else, under any verdict including `CONTRADICTORY`. Current evidence always wins; this is a report, never an override.

## 5. Temporal stabilization

A deterministic, confidence-weighted, ADDITIVE alternative depth view — never a replacement of the current frame's own raw depth.

- **Gate:** `PipelineConfig.enable_temporal_stabilization: bool = False`, nested under `enable_temporal`.
- **Component:** `temporal.compute_temporal_stabilization()` (`temporal/stabilization.py`), a pure function combining the current frame's decimated depth with the single most recent comparable prior record's own snapshot via one closed-form, bounded weighted average, per pixel.
- **Output:** `DepthPerceptionResult.temporal_stabilization: Optional[temporal.TemporalStabilization]`. Fields: `state` (`temporal.TemporalStabilizationState.STABILIZED`/`CURRENT_ONLY_FALLBACK`/`INSUFFICIENT_EVIDENCE`), `stabilized_depth_m` (decimated, same 0.0-invalid convention as `depth_map`; `None` only under `INSUFFICIENT_EVIDENCE`), `eligible_count`/`stabilized_count`/`contradiction_count`, `stabilized_fraction`/`contradiction_fraction`.
- **Safety rule:** strong current evidence that contradicts history is kept unchanged at the pixel level, never blended away — an aggregate `STABILIZED` frame can still report a nonzero `contradiction_fraction` for the pixels that individually disagreed.

## 6. Motion compensation

Short-window rotational alignment of the previous comparable frame's geometry into the current frame's own viewpoint, so genuine sensor rotation is not mistaken for scene change.

- **Input contract:** `temporal.MotionHint` — an optional, externally supplied, short-duration rotational motion measurement (`timestamp`, `angular_velocity_rad_s` (3-vector, rad/s), `frame_id`, `valid`). Gyro-only by design: no linear velocity/acceleration/position field exists or is planned (translation cannot be safely inferred from a gyro-only reading, and adding one would invite exactly the pose/localization creep Level 4 is architecturally forbidden from performing). Carried on `StereoObservation.motion_hint` (single, per-frame association) and `StereoObservation.motion_hints`/`process()`'s `motion_hints` parameter (a bounded sequence spanning the interval leading up to a frame, for integration).
- **Gate:** `PipelineConfig.enable_rotation_compensation: bool = False`, nested under `enable_temporal`.
- **Components** (`temporal/rotation_compensation.py`, all pure functions): `select_motion_hint_samples()` (interval/monotonicity/validity filtering), `integrate_angular_velocity()` (zero-order-hold SO(3) integration via Rodrigues), `compensate_prior_geometry()` (back-project → rotate → re-project the previous frame's decimated depth onto the current frame's own grid; deterministic nearest-wins occlusion resolution; no translation term anywhere), and `compensate_prior_geometry_with_payload()` (the same exact reprojection, additionally carrying auxiliary per-cell channels through the identical mapping — used by persistence tracking, section 8).
- **Output:** `DepthPerceptionResult.rotation_compensation_status: Optional[str]` — `temporal.RotationCompensationStatus.APPLIED`/`NOT_APPLIED`. Every failure/fallback condition (missing/invalid/stale/insufficient hints, no prior record, disabled) collapses to `NOT_APPLIED` with the uncompensated record passed through completely unchanged — no richer failure taxonomy, and the relative rotation itself is never exposed (never retained, logged, or accumulated into a trajectory by this library).
- **Scope boundary:** never a pose estimator. No camera orientation is ever computed, stored, or returned — only an ephemeral relative rotation between two specific frames, used once per call and discarded.

## 7. Reliability

A deterministic, explicit (never one opaque blended score) assessment of whether the consistency/stabilization results for this frame remain trustworthy given its motion conditions and compensation outcome.

- **Gate:** `PipelineConfig.enable_motion_aware_reliability: bool = False`, nested under `enable_temporal`; independent of `enable_temporal_stabilization`/`enable_rotation_compensation` (reliability assesses whatever state actually exists, including "compensation disabled" — a legitimate configuration, not a failure).
- **Component:** `temporal.compute_motion_aware_reliability()` (`temporal/reliability.py`), a pure function over already-computed signals only.
- **Output:** `DepthPerceptionResult.motion_aware_reliability: Optional[temporal.MotionAwareReliability]`. `state` (`temporal.MotionAwareReliabilityState.RELIABLE`/`DEGRADED`/`UNRELIABLE`/`INSUFFICIENT_EVIDENCE`) plus six explicit, never-blended signal fields: `motion_sample_count`, `motion_coverage_fraction`, `rotation_compensation_status`, `angular_motion_magnitude_rad`, `temporal_consistency_state`, `temporal_stabilization_state`.
- **Computed even when consistency did not run this frame** (a chronology-rejected frame is still classified, as `INSUFFICIENT_EVIDENCE`), so a caller always has a reliability verdict to read when the flag is on.
- **Thresholds:** `PipelineConfig.reliability_max_angular_motion_rad` (default ≈5°), `reliability_min_motion_coverage_fraction` (default 0.5).

## 8. Persistence

A deterministic, per-cell classification of geometric evidence across time — the only Level 4 concept whose own output is inherently spatial (one region of a frame can be freshly new while another is long-persistent and a third is fading, all in the same call).

- **Gate:** `PipelineConfig.enable_temporal_persistence: bool = False`, requiring **both** `enable_temporal` and `enable_motion_aware_reliability` also `True` (persistence gates its own updates on the reliability verdict — see the safety rule below).
- **Component:** `temporal.persistence.TemporalPersistenceTracker`, a small, bounded, stateful collaborator (four fixed-size arrays, shape fixed on first use, never resized — never a second, unbounded history buffer, never a store of full `TemporalRecord`/`DepthPerceptionResult` objects) owned by `DepthPerceptionPipeline`, mirroring `TemporalHistory`'s/`ThreatAssessor`'s own cross-frame-state precedent.
- **Output:** `DepthPerceptionResult.temporal_persistence: Optional[temporal.TemporalPersistence]`. `state` (`temporal.persistence.TemporalPersistenceState.CLASSIFIED`/`UNRELIABLE`/`INSUFFICIENT_EVIDENCE` — the frame-level gate outcome) and, per cell of the same decimated grid consistency/stabilization use, `state_grid` (`temporal.persistence.TemporalPersistenceCellState.NO_EVIDENCE`/`NEW`/`PERSISTENT`/`DISAPPEARING`), `support_count_grid`, `age_s_grid`, plus `new_count`/`persistent_count`/`disappearing_count`/`expired_count`/`eligible_count`/`persistent_fraction`.
- **Support/expiration policy:** `PipelineConfig.persistence_min_support_count` (≥2, enforced — one observation can never read `PERSISTENT`), `persistence_max_dropout_frames` (grace window before an absent cell reads `DISAPPEARING`), `persistence_expiration_absence_frames` (beyond which a cell's tracked state is fully, deterministically cleared, reverting to `NO_EVIDENCE`).
- **Motion compensation reuse:** the tracker's own per-cell state is re-expressed in the current frame's grid via `compensate_prior_geometry_with_payload()` (section 6) — the identical rotation-compensated mapping the depth channel itself uses, never a second reprojection implementation.

### Frozen safety rules (enforced structurally, proven by test, not merely stated)

1. **History may support weak current evidence; it can never override strong contradictory current evidence.** A currently-occupied cell is always classified from its own current value; a contradicting historical value only resets that cell's own support count, never suppresses or delays its classification.
2. **UNKNOWN never becomes FREE.** Exactly four per-cell codes exist and none of them means or implies free space — an expired or long-absent cell reverts to `NO_EVIDENCE`, the same code a cell with no history at all carries.
3. **One observation can never become `PERSISTENT`.** `persistence_min_support_count` is validated `>= 2` at construction.
4. **An `UNRELIABLE` frame (reliability, section 7) can neither create nor reinforce persistence.** The tracker performs no update at all on such a frame — no support increment, no new classification, no contradiction-reset — re-exposing its own unchanged prior snapshot instead. It does not erase persistence either.
5. **A single dropout frame does not instantly erase previously persistent evidence** (the grace window, above); **stale evidence expires deterministically** (the expiration threshold, above) rather than persisting forever or being silently reinterpreted as free.

## 9. Health / timing

- **`DepthPerceptionPipeline.health() -> PipelineHealth`:** lifecycle-only (`is_closed`, `frames_processed`, `last_confidence`, `last_processing_time_ms`) — explicitly not a per-frame diagnosis; unaffected by Level 4.
- **`DepthPerceptionResult.processing_time_ms`:** total per-frame latency, including every enabled Level 4 stage's own cost — each stage additionally logs its own sub-cost separately at `DEBUG` level (`pipeline.py`'s `"... stage: %.2f ms"` lines, one per stage, consistent format throughout Level 3 and Level 4).
- **`DepthPerceptionPipeline.temporal_history`:** read-only property, `Optional[temporal.TemporalHistory]`, mirroring `.config`/`.calibration`'s own exposure pattern — query `.records`/`.latest`/`len()` to inspect current chronology.
- Measured marginal cost of the entire Level 4 chain (all seven capabilities) on top of Level 3 alone: **~1.7% overhead** on this development container's own CPU — see `docs/LEVEL4_E8_INTEGRATION_REPORT.md` for the full measurement and its explicit "not Jetson" caveat.

## 10. Configuration reference (by concept)

| Concept | `PipelineConfig` fields | Default |
|---|---|---|
| Temporal history | `enable_temporal`, `temporal_max_records`, `temporal_max_age_s`, `temporal_gap_limit_s` | `False`, `30`, `1.0`, `0.5` |
| Temporal consistency | `temporal_consistency_sampling_stride`, `temporal_consistency_agreement_tolerance_m`, `temporal_consistency_min_agreement_fraction` | `4`, `0.05`, `0.7` |
| Temporal stabilization | `enable_temporal_stabilization`, `temporal_stabilization_min_history_confidence`, `temporal_stabilization_min_comparable_fraction` | `False`, `0.3`, `0.1` |
| Motion compensation | `enable_rotation_compensation` | `False` |
| Reliability | `enable_motion_aware_reliability`, `reliability_max_angular_motion_rad`, `reliability_min_motion_coverage_fraction` | `False`, `0.0873`, `0.5` |
| Persistence | `enable_temporal_persistence`, `persistence_min_support_count`, `persistence_max_dropout_frames`, `persistence_expiration_absence_frames` | `False`, `2`, `1`, `5` |

Every threshold above is an explicit policy choice, not a derived physical constant — validated at construction (`PipelineConfig.__post_init__`), documented with its own justification inline in `config/pipeline_config.py`, and expected to be re-tuned per deployment/sensor rather than trusted as a universal default.

## 11. `DepthPerceptionResult` field reference (by concept)

| Concept | Field | Type |
|---|---|---|
| Current evidence | `disparity_map`, `depth_map`, `traversability_mask`, `obstacles`, `confidence`, `processing_time_ms`, `valid_disparity_mask`, `valid_depth_mask`, `timestamp`, `geometry`, `geometry_body`, `obstacle_cloud`, `free_space_rays`, `geometry_metrics` | (Level 3, unchanged) |
| Temporal history | `temporal_admission_status` | `Optional[str]` |
| Temporal consistency | `temporal_consistency` | `Optional[temporal.TemporalConsistency]` |
| Temporal stabilization | `temporal_stabilization` | `Optional[temporal.TemporalStabilization]` |
| Motion compensation | `rotation_compensation_status` | `Optional[str]` |
| Reliability | `motion_aware_reliability` | `Optional[temporal.MotionAwareReliability]` |
| Persistence | `temporal_persistence` | `Optional[temporal.TemporalPersistence]` |

Every field above defaults to `None` and is purely additive — every pre-Level-4 construction of `DepthPerceptionResult` continues to work unmodified.

## 12. Level 3 compatibility

Nothing above changes a Level 3 field's name, type, shape, or computed value. Every Level 3 test (`tests/test_pipeline_geometry.py` and the rest of the pre-Level-4 suite) passes unmodified. `docs/LEVEL3_ARCHITECTURE.md`/`docs/LEVEL3_CONTRACTS.md`/`docs/LEVEL3_PUBLIC_API.md` remain fully authoritative for Level 3 and are not superseded by this document.

## 13. Non-goals (frozen, unchanged since Level 4's own inception)

Level 4 never estimates authoritative vehicle/agent pose, global position, or velocity as localization state; never estimates IMU biases; never implements visual odometry, VIO, or SLAM; never creates a world map or performs world-frame persistence; never plans trajectories, calculates vehicle clearance, or knows vehicle geometry; never issues movement commands or knows actuator types; never contains an aerial/ground/marine mode; never depends on `state_estimation_engine`; never uses a neural or learned component of any kind (every Level 4 computation is closed-form, deterministic, and traceable to an explicit formula or threshold — see `temporal/reliability.py`'s `arccos` rotation-angle extraction and `temporal/rotation_compensation.py`'s `cv2.Rodrigues` integration for the two least-trivial examples, neither of which is learned). Guarded structurally, not just by convention, by `tests/test_level4_architecture_guards.py`.

## 14. Development history (for archaeology only — not part of the public API)

Level 4 was built in eight incremental passes. This section exists solely so a reader auditing a specific design decision can find the record of it — none of the labels below appear anywhere in `src/depth_perception_engine/`'s actual code, and this section is the only place in this canonical reference where they appear at all.

| Concept (this document's own organization) | Decision record |
|---|---|
| Motion-hint input contract | `docs/E2_TEMPORAL_HISTORY_PLAN.md` (frozen alongside temporal history) |
| Temporal history | `docs/E2_TEMPORAL_HISTORY_PLAN.md` |
| Temporal consistency | `docs/E3_IMPLEMENTATION_PLAN.md` |
| Temporal stabilization | `docs/E4_IMPLEMENTATION_PLAN.md` |
| Motion compensation | `docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md` |
| Reliability | `docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md` |
| Persistence | `docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md` |
| Integration validation, visual proof, this freeze | `docs/LEVEL4_E8_IMPLEMENTATION_PLAN.md` (the plan), `docs/LEVEL4_E8_INTEGRATION_REPORT.md` (the result) |
| Original architectural boundary | `docs/LEVEL4_ARCHITECTURE.md`, `docs/LEVEL4_CONTRACTS.md`, `docs/LEVEL4_PUBLIC_API.md` (superseded by this document for current usage; retained for full process history) |
| Simulated-vs-real motion-input parity | `docs/LEVEL4_SIMULATED_IMU.md` |
| What remains unverified against real sensors | `docs/LEVEL4_HARDWARE_VALIDATION_PENDING.md` |

## Freeze statement

As of this document, **Level 4 software is frozen**: every capability described above (sections 3-8) is implemented, tested (715 tests passing, `pytest tests/ -q`), integrated (all seven capabilities proven to compose correctly in one pipeline over realistic multi-frame sequences — `docs/LEVEL4_E8_INTEGRATION_REPORT.md`), and documented in this single canonical reference. No further Level 4 capability is planned. A standalone live OpenCV validation tool (`examples/visualize_level4_live.py`) and a real-camera capture of the full chain (`docs/assets/11_level4_live_demo.gif`, `docs/VALIDATION_REPORT.md`'s Level 4 addendum) close the "real stereo capture" row of the hardware checklist below; real IMU, genuine rotation, measured extrinsics, and Jetson performance remain explicitly pending — see `docs/LEVEL4_HARDWARE_VALIDATION_PENDING.md` — and none of this blocks the software freeze.
