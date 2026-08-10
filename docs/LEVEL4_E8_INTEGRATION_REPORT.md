# Level 4, Phase E8 — integration report

Companion to `docs/LEVEL4_CANONICAL_REFERENCE.md` (the frozen, authoritative "what Level 4 is") and `docs/LEVEL4_HARDWARE_VALIDATION_PENDING.md` (what E8 explicitly did NOT and could not validate). This document records what E8's own validation pass actually did and measured.

## A. Full-chain integration — scenarios and results

`tests/test_level4_integration_e8.py`, 17 tests, one pipeline configuration with every Level 3 geometry flag and every Level 4 flag enabled simultaneously (`enable_geometry`, `enable_obstacle_geometry`, `enable_free_space_rays`, `enable_temporal`, `enable_temporal_stabilization`, `enable_rotation_compensation`, `enable_motion_aware_reliability`, `enable_temporal_persistence`), run over realistic multi-frame synthetic sequences (not single frames) — all passing:

| Claim | Test class | Result |
|---|---|---|
| Current raw Level-3 evidence remains unchanged | `TestRawLevel3EvidenceUnchanged` | Byte-identical `disparity_map`/`depth_map`/`geometry`/`geometry_body`/`obstacle_cloud`/`free_space_rays`/`confidence` across a 5-frame mixed sequence, with vs. without the entire Level 4 chain enabled |
| Strong new occupied evidence wins immediately | `TestNewEvidenceWinsImmediately` | A new object is classified `NEW` on its very first frame, both in isolation and in the same frame as unrelated, already-`PERSISTENT` evidence elsewhere in the grid |
| UNKNOWN never becomes FREE | `TestUnknownNeverBecomesFree` | Zero obstacle/ray evidence on a textureless frame (re-verifies the frozen Level 3 invariant still holds with the full Level 4 chain active); `state_grid` never contains a code beyond the four defined (no "FREE"); expired cells revert to `NO_EVIDENCE`, never fabricate free space |
| Temporal stabilization behaves correctly | `TestTemporalStabilizationBehavesCorrectly` | First frame `INSUFFICIENT_EVIDENCE`; second (repeated) frame `STABILIZED`, finite values, >90% stabilized fraction; `depth_map` itself untouched |
| Rotation compensation wiring | `TestRotationCompensationWiring` | `rotation_compensation_status` reads `APPLIED`/`NOT_APPLIED` correctly through the full 7-stage chain; a negligible true rotation does not degrade an otherwise-good comparison. **Quantitative "improves agreement over a real rotation" is proven at the exact-function level** by the pre-existing `tests/test_rotation_compensation.py`/`tests/test_temporal_persistence.py` suites (same functions the full chain itself calls) — a genuinely photorealistic rotated capture requires real hardware, see Part C |
| Unreliable evidence cannot reinforce persistence | `TestUnreliableEvidenceCannotReinforcePersistence` | A deliberately excessive injected rotation forces `MotionAwareReliabilityState.UNRELIABLE`; `temporal_persistence.state == UNRELIABLE`, `persistent_count` and `support_count_grid` provably unchanged; the very next ordinary frame recovers cleanly |
| Persistent evidence survives bounded dropout; stale evidence expires | `TestDropoutSurvivalAndExpiration` | Full cycle on a real pipeline run: `NEW` → `PERSISTENT` → one dropout frame (still `PERSISTENT`, grace) → second dropout (`DISAPPEARING`) → third dropout (`EXPIRED`, `eligible_count == 0`) → reappearance (`NEW` again, not resumed) |
| Reset/gaps clear temporal assumptions | `TestResetAndGapClearTemporalAssumptions` | `reset()` empties `temporal_history` and zeroes `persistent_count`; a gap beyond `temporal_gap_limit_s` produces `ACCEPTED_NEW_SEQUENCE`, `len(temporal_history) == 1`, and `persistent_count == 0` |
| Degradation → recovery works | `TestDegradationRecovery` | A textureless frame sandwiched between two identical real scenes recovers persistence exactly where it left off (dropout grace); a genuinely different scene after degradation reads `NEW`, not stuck |
| Memory remains bounded | `TestBoundedMemory` | 80-frame run, `temporal_max_records=10`: `len(temporal_history) == 10` throughout; `temporal_persistence.state_grid.shape` identical on every one of the 80 frames |
| Deterministic replay | `TestDeterministicReplay` | Two independently-constructed pipelines given the identical 4-frame sequence (including motion hints) produce byte-identical `disparity_map`/`depth_map` and identical state/count values at every stage, every frame |

## Total regression count

- Before E8: 698 passed (E1-E7 complete).
- After E8 (17 new integration tests): **715 passed, 0 failed**, `pytest tests/ -q`.

## Visual validation command/output

```
pip install -e ".[viz]"
python examples/visualize_level4_temporal.py --output <path>.png
```

Produces one multi-panel filmstrip PNG (8 rows, one per synthetic frame: `flat → scene A appears → scene A repeats → scene A + simulated MotionHint → 3× flat dropout → scene B appears`). Each row shows, side by side: raw `depth_map`, `temporal_stabilization.stabilized_depth_m` (or `N/A`), `temporal_persistence.state_grid` (color-coded NO_EVIDENCE/NEW/PERSISTENT/DISAPPEARING), and a text panel with `temporal_admission_status`, `temporal_consistency.state`, `temporal_stabilization.state`, `rotation_compensation_status`, `motion_aware_reliability.state`, `temporal_persistence.state`, and the new/persistent/disappearing/expired counts. Console output during the run:

```
frame processed: flat (baseline)                     t=  0.00  persistence=CLASSIFIED     new=    0 persistent=    0 disappearing=    0 expired=    0
frame processed: scene A appears                     t=  1.00  persistence=CLASSIFIED     new= 1796 persistent=    0 disappearing=    0 expired=    0
frame processed: scene A repeats                     t=  2.00  persistence=CLASSIFIED     new=    0 persistent= 1796 disappearing=    0 expired=    0
frame processed: scene A + simulated MotionHint      t=  3.10  persistence=CLASSIFIED     new=    0 persistent= 1796 disappearing=    0 expired=    0
frame processed: flat (dropout 1)                    t=  4.00  persistence=CLASSIFIED     new=    0 persistent= 1796 disappearing=    0 expired=    0
frame processed: flat (dropout 2)                    t=  5.00  persistence=CLASSIFIED     new=    0 persistent=    0 disappearing= 1796 expired=    0
frame processed: flat (dropout 3)                    t=  6.00  persistence=CLASSIFIED     new=    0 persistent=    0 disappearing=    0 expired= 1796
frame processed: scene B appears                     t=  7.00  persistence=CLASSIFIED     new= 1762 persistent=    0 disappearing=    0 expired=    0
```

The script (`examples/visualize_level4_temporal.py`) is standalone (outside `src/depth_perception_engine/`), calls only `DepthPerceptionPipeline.process()` and reads public `DepthPerceptionResult` fields — it computes no consistency/stabilization/compensation/reliability/persistence itself, matching `examples/visualize_level3.py`'s own established discipline. The one simulated `MotionHint` is a plain, directly-constructed dataclass value (`docs/LEVEL4_SIMULATED_IMU.md`).

## Determinism proof

`TestDeterministicReplay` (Part A) plus `tests/test_temporal_persistence.py`'s and `tests/test_rotation_compensation.py`'s own pre-existing determinism tests: two independently-constructed pipelines given identical inputs (including identical `MotionHint` sequences) produce byte-identical numeric output and identical state-string/count output at every one of the seven Level 4 stages, every frame. No randomness, no wall-clock dependency (`time.time()`/`time.perf_counter()` is never used for any chronology decision — verified structurally at E2, unchanged since).

## Bounded-memory result

`TestBoundedMemory`: an 80-frame run with `temporal_max_records=10` keeps `len(temporal_history) == 10` throughout, and `temporal_persistence`'s own four backing arrays never change shape, regardless of frame count — consistent with every phase's own bounded-memory proof (E2's `TestMemoryBounding`, E7's `TestBoundedMemory`/`TestPipelineBoundedMemory`).

## Latency/performance impact

Measured on this development container's own CPU (explicitly NOT Jetson — see `docs/LEVEL4_HARDWARE_VALIDATION_PENDING.md`), 40-frame run (5-frame warmup discarded) on a 320×240 real-noise synthetic stereo pair, `PipelineConfig`'s default resolution/stride settings:

| Configuration | Mean `processing_time_ms` | p95 |
|---|---|---|
| Level 3 only (geometry + obstacle cloud + free space rays) | 23.04 ms | 25.06 ms |
| Level 3 + full Level 4 chain (all 7 capabilities) | 23.44 ms | 24.96 ms |
| Marginal Level 4 overhead | **+0.40 ms (~1.7%)** | — |

Consistent with each phase's own individually-measured marginal cost (E2-E7's own `docs/*_IMPLEMENTATION_PLAN.md` addenda) summing to a small fraction of the Level 3 baseline — no combinatorial blowup from running all seven stages together.

## Documentation canonicalization

See `docs/LEVEL4_CANONICAL_REFERENCE.md` (new, authoritative, organized by concept — current/raw evidence, temporal consistency, temporal stabilization, motion compensation, reliability, persistence, health/timing — with no E1-E7 phase label anywhere in its own body). `docs/LEVEL4_ARCHITECTURE.md`, `docs/LEVEL4_CONTRACTS.md`, `docs/LEVEL4_PUBLIC_API.md`, and every `docs/*_IMPLEMENTATION_PLAN.md`/`docs/LEVEL4_E*_IMPLEMENTATION_PLAN.md` decision record now carry a superseded-pointer banner at the top (added, not deleted — full decision history remains available for archaeology) directing a reader to the canonical reference for current usage.

## Hardware-validation checklist

See `docs/LEVEL4_HARDWARE_VALIDATION_PENDING.md` — real stereo, real IMU, genuine rotation, measured extrinsics, and Jetson performance are all explicitly checked off as **pending**, not fabricated.

## Genuine blockers to Level 4 freeze

None found in software. The one real, acknowledged limitation: Part A's rotation-compensation "improves comparison" claim is proven at the exact-function level (same code the full pipeline calls) rather than through a photorealistic full-pipeline rotated capture, because no synthetic random-noise image pair can represent a real rotated view of the same scene — this requires real hardware (Part C) and is not a software defect or an open design question, it is the explicitly-scoped boundary of what synthetic validation can ever prove. Level 4 SOFTWARE freeze does not require closing this gap; the frozen canonical reference documents it as a known, non-fabricated limitation instead.
