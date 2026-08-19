# DPE v1.1.1 — Validation Matrix

Every row below is traceable to a committed benchmark script, a test file, or
`src/`'s own live config defaults — see the **Artifact/source** column.
Result files under `benchmarks/*/results/` are `.gitignore`d (regeneratable,
not committed); every number here was re-derived fresh during the v1.1.1
documentation pass via `benchmarks/reporting/release_metrics_manifest.py`,
which cites its own source for each value — that module, not this table, is
the canonical source of truth if the two ever disagree.

**Status legend** — used deliberately instead of a blanket PASS/FAIL:

- **PASS** — meets or exceeds a stated target, or is a clean before/after improvement with no regression.
- **CHARACTERIZED LIMITATION** — a real, measured, unresolved gap. Not a bug being hidden; a boundary of the current algorithm, documented so a consumer can design around it.
- **HARDWARE PENDING** — validated only in software/simulation; requires physical stereo-camera/Jetson evidence before being claimed as hardware-qualified.

## Stereo / disparity

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Decorrelated (zero-correspondence) false-valid disparity | 41.788% | 0.267% | — | PASS | Synthetic, deterministic seeds | `benchmarks/i1_stereo_accuracy/{measure.py, results/baseline_current.json, results/final_after.json}` |
| Depth error @ 1 m (median) | 0.561% | 0.405% | — | PASS | Synthetic | same |
| Depth error @ 2 m (median) | 0.561% | 0.561% | — | PASS (unchanged) | Synthetic | same |
| Depth error @ 3 m (median) | 1.501% | 1.018% | — | PASS | Synthetic | same |
| Depth error @ 5 m (median) | 0.561% | 0.561% | — | PASS (unchanged) | Synthetic | same |
| Depth error @ 6 m (median) | 4.443% | 3.574% | — | PASS | Synthetic | same |
| Depth error @ 6 m, weak texture (scenario C, median) | — | 5.296% | — | CHARACTERIZED LIMITATION | Synthetic | `benchmarks/i1_stereo_accuracy/results/final_after.json` (scenario C, 6.0m) |
| SGBM `P1`/`P2` channel-count correctness | 3-channel constant applied to grayscale-only input (over-smoothing) | 1-channel constant (correct) | matches `cv2.StereoSGBM` channel count actually used | PASS | Source-level fix | `src/depth_perception_engine/stereo/disparity_engine.py` (git `c2b1906`) |

## Depth validity / observable ROI

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Structural SGBM dead zone (left `numDisparities`-wide band, 320px frame) | ~40% of width unmatchable | unchanged — physical limit, not a defect | — | CHARACTERIZED LIMITATION | Synthetic, geometric | `benchmarks/i2_depth_validity/results/step1_2_roi_and_gates.json` (`theoretical_whole_frame_observable_fraction=0.60`) |
| Whole-frame valid fraction (unscoped) | ~59.9% | ~59.9% (unchanged — same physical limit) | not meaningful alone | CHARACTERIZED (metric is a red herring on its own) | Synthetic | same, `whole_frame_valid_mean` |
| Observable-ROI valid fraction (columns ≥128px only) | not separately measured pre-I2 | 99.34-99.96% | ≥70% (pre-existing target) | PASS | Synthetic | same, `roi_valid_mean` across the full texture/depth sweep |

## Occlusion / dis-occlusion safety (shadow-zone reliability)

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Occlusion-strip contamination into `obstacle_cloud`/`free_space_rays`/`BoundaryEvidence` | present, unflagged | reduced ~13.6%, now flagged via `compute_shadow_zone_mask` | — | PASS (mitigated, not eliminated) | Synthetic, geometric mechanism | `src/depth_perception_engine/geometry/reliability.py`; `docs/VALIDATION_REPORT.md` I1-I6 addendum |
| `geometry_shadow_zone_lookahead_px` | — | 8 px | — | (config value) | live source default | `src/depth_perception_engine/config/pipeline_config.py` |
| `geometry_shadow_zone_gradient_threshold_px` | — | 3.0 px | — | (config value) | live source default | same |
| `geometry_shadow_zone_max_width_px` | — | 40 px | — | (config value) | live source default | same |

## Boundary

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Precision | 87.8% | 100.0% (TP=126, FP=0) | — | PASS | Synthetic, deterministic fixtures | `docs/VALIDATION_REPORT.md` I1-I6 addendum (baseline, not re-derivable on frozen source); `benchmarks/i4_boundary_precision/collect.py` (v1.1.1, fresh rerun this pass) |
| Recall | 100.0% | 100.0% (FN=0) | — | PASS (unchanged) | Synthetic | same |
| Noise-driven low-support false positives | present | 0 | — | PASS | Synthetic | `benchmarks/i4_boundary_precision/collect.py` |
| Genuine TN (correctly-quiet uniform pairs) | — | 54 | — | PASS | Synthetic | same |
| A note on a claimed "96.9%" historical figure | — | — | — | **NOT VERIFIED** | — | searched across all committed docs/benchmarks/git history; not found anywhere in this repository — omitted from all v1.1.1 documentation rather than guessed |

## Surface

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Angular error, high-planarity (≥0.95) + full-coverage cells, p95 | — | 1.417° | ≤10° (preferred ≤5°) | PASS | Synthetic, analytic ground truth, 32 cells (fronto/yaw/pitch/combined slant × texture × range × seed) | `benchmarks/i5_surface_opening_clearance/surface/measure.py`, `results/measure.json` — see `release_metrics_manifest.surface_normal_high_planarity_stats()`'s own provenance note on excluding the separate partial-coverage fixture family |
| Angular error, same subset, median / max | — | 0.515° / 1.655° | — | PASS | same | same |
| Angular error, partial-coverage/mixed-surface cells (support_fraction ~0.2-0.75) | — | mean 36.7°, up to 81.5°, regardless of self-reported planarity | — | CHARACTERIZED LIMITATION | Synthetic, deliberately adversarial fixtures | same file, `mixed_surface_cell`/`partial_invalid_cell` labels |

## Opening

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Precision | 100.0% | 100.0% (TP=50, FP=0) | — | PASS (unchanged) | Synthetic, deterministic fixtures | `docs/VALIDATION_REPORT.md` I1-I6 addendum (baseline); `benchmarks/i5_surface_opening_clearance/opening/measure.py`, fresh rerun this pass |
| Recall | 54.5% | 90.9% (FN=5, all `partial_invalid` scenario) | — | PASS | Synthetic | same |
| Width error, median (confirmed openings) | — | 0.561% | — | PASS | Synthetic | same |
| Range error, median (confirmed openings) | — | 0.561% | — | PASS | Synthetic | same |
| Negative-fixture false openings (decorrelated noise) | 0 | 0 | — | PASS (unchanged) | Synthetic | `benchmarks/i3_occlusion_safety/validate.py`, fresh rerun this pass — `grid_3x3`/`grid_6x6` both 0 before/after |

## Clearance

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Transition/narrow-obstacle sector magnitude error | 13-94% relative | unchanged — unresolved, accepted | — | CHARACTERIZED LIMITATION | Synthetic | `docs/VALIDATION_REPORT.md` I1-I6 addendum; `benchmarks/i5_surface_opening_clearance/clearance/measure.py` |
| False-clear `SUPPORTED` sectors — initial measurement | — | 28/252 (11.1%) | 0 | (superseded by methodology fix below) | Synthetic | `benchmarks/results/i6_final_qualification.json` |
| False-clear sectors — after benchmark-methodology fix (fresh pipeline per fixture) | — | 4/252 (1.6%) | 0 | (superseded by gating fix below) | Synthetic | same |
| False-clear sectors — final, after shadow-zone + ramp-zone gating | — | **0/252 (0.0%)**, reconfirmed fresh this pass | 0 | **PASS** | Synthetic, 630+ sectors across scenario families incl. 0/630 on pure noise | `benchmarks/i5_surface_opening_clearance/clearance/measure.py`, fresh rerun this pass |
| Worst-case `SUPPORTED`-sector error | 139.252% | 4.443% | — | PASS | Synthetic | `benchmarks/results/i6_final_qualification.json` |
| Sectors downgraded `SUPPORTED`→`PARTIALLY_SUPPORTED` (accepted conservative-direction cost) | — | ~30/252 | — | (accepted trade-off, not a defect) | Synthetic | same |
| `clearance_shadow_zone_contamination_threshold` | — | 0.30 | — | (config value) | live source default | `src/depth_perception_engine/config/pipeline_config.py` |
| `clearance_ramp_zone_window_px` | — | 24 px | — | (config value) | live source default | same |
| `clearance_ramp_zone_gradient_threshold_px` | — | 2.0 px | — | (config value) | live source default | same |
| **`ClearanceEvidence` is evidence with support/quality semantics, not an unconditional safety guarantee** | — | reaffirmed | — | see caveat | policy statement | `docs/RELEASE_NOTES_V1.md` "Known caveats"; prompted directly by an external MP01-sim integration finding false clears on a scene outside this qualified 252-sector set |

## Temporal (Level 4)

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Static-scene temporal agreement | — | confirmed | — | PASS | Synthetic | `benchmarks/i6_temporal/measure.py` |
| Rotation-compensation status correctness (`NOT_APPLIED` with no motion, `APPLIED` with a real MotionHint) | — | confirmed both paths | — | PASS | Synthetic + one real-camera live capture (Level 4 addendum) | `docs/VALIDATION_REPORT.md` Level 4 addendum (2026-08-10) |
| `TemporalHistory`/`TemporalPersistenceTracker` bounded-resource behavior, 500-frame run | — | confirmed bounded, front-loaded ramp only | — | PASS | dev-container run | `docs/VALIDATION_REPORT.md` D14 addendum, `results/d14_performance_validation.json` |
| Real IMU/measured-rotation/Jetson temporal accuracy | not evaluated | not evaluated | — | **HARDWARE PENDING** | — | `docs/LEVEL4_HARDWARE_VALIDATION_PENDING.md` |

## Performance (standalone DPE, dev-container — NOT hardware-qualified)

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Mean latency, 320×240, full V1 evidence config | historically 23.6-49.4 ms across recorded sessions (real, disclosed environment variance) | 38.39 ms (fresh, this pass) | no documented real-time requirement exists for this library | PASS (characterized, not judged against an invented target) | dev-container, not cherry-picked | `benchmarks/i0_baseline/compare_to_baseline.py`, run live this pass |
| p95 / p99 | — | 45.87 ms / 52.17 ms | — | PASS (characterized) | same | same |
| FPS (mean-based) | — | 26.0 | — | PASS (characterized) | same | same |
| Simulation/ROS integration end-to-end latency (a SEPARATE measurement boundary — external MP01 integration, not this repository) | — | not this library's own compute cost | — | out of scope for this repo | external | see `docs/ENGINEERING_EVOLUTION.md`'s performance-provenance note — not re-cited as a DPE number here |
| Physical stereo-camera + Jetson real-time performance | not measured | not measured | — | **HARDWARE PENDING** | — | — |

## Packaging / release

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| Isolated `pip install`/`pip wheel` (Python 3.10, Ubuntu dist-packages) | silently empty wheel (v1.1.0) | 59-module wheel, correct | — | PASS | reproducible execution, `mp01_ros2` container | `docs/DPE_V1_PROVIDER_CONTRACT.md` D18 record |
| Editable install (`pip install -e .`) | — | resolves to source tree | — | PASS | same | same |
| Normal install | — | correct version/imports | — | PASS | same | same |
| Git-tag install (`git+...@v1.1.1`) | — | correct, genuinely isolated build | — | PASS | same | same |
| Version consistency (`pyproject.toml`/`setup.py`/`__init__.py`) | — | all `1.1.1` | — | PASS | `tests/test_packaging_metadata.py::TestVersionConsistency` | same |

## Regression

| Metric | Baseline | v1.1.1 | Target | Status | Validation type | Artifact/source |
|---|---|---|---|---|---|---|
| `pytest tests/ -q` | 950 (D17) | 953 | 0 failures | PASS | full suite | run this pass |
| `compare_to_baseline` leaf-metric agreement | — | 69/79 exact zero-delta (10 latency-only metrics differ, expected/disclosed) | no unexplained accuracy delta | PASS | `benchmarks/i0_baseline/compare_to_baseline.py` | run this pass |
| Public API / `GeometryFrame` contract | frozen at D13/D16 | unchanged | no breaking change | PASS | `tests/test_public_api.py` | unchanged since D13 |
