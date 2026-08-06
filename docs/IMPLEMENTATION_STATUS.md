# Implementation status

Written as part of a baseline-recovery/audit pass (see `docs/VALIDATION_REPORT.md` for the full report). States what's actually true today — not aspirational.

## Implemented and verified

- Standalone, ROS-free, pip-installable stereo depth library (`pip install -e .` — real venv, real hardware, no MP01 workspace needed).
- One canonical execution path: `DepthPerceptionPipeline.process()`/`.process_observation()` (stateful, recommended) and `pipeline.api`'s five stateless functions (one-shot use) — call the same underlying stage classes, no divergent logic.
- Real algorithm, not a stub: rectification → StereoSGBM disparity → closed-form Q-matrix depth → 3×3 region classification (Laplacian-variance texture, Sobel gradient, Shannon entropy, blended confidence) → 20-beam obstacle scan (IQR-percentile, EMA-smoothed, debounced).
- Math verified two ways: differentially against `cv2.reprojectImageTo3D` across real + synthetic Q matrices and invalid/negative/near-zero/huge-disparity edge cases (`test_depth_estimator.py::TestZOnlyMatchesFullReprojection`), and — added in this pass — a pure hand-computed analytic check (`TestAnalyticKnownDepth`) independent of any OpenCV reprojection function, `depth = f × baseline / disparity`, across four known focal-length/baseline/disparity combinations.
- Verified on real hardware (2026-08-05, this recovery pass's preceding session): a real USB global-shutter stereo camera, `/dev/video0`, produced a stable 1.21 m reading (±2 mm across 8 consecutive frames) against a real wall, with physically sensible region classification (wall correctly `PROBABLE_WALL`/`UNKNOWN`, a cable correctly `OBSTACLE`) and one live catch of the anti-false-clear safety design (a noisy long-range disparity blip correctly downgraded to `BLOCKED` rather than trusted as `CLEAR`).
- 131 tests, `pytest tests/ -q`, all passing as of this pass — up from 67 before it. Real behavioral coverage (a genuine rectification-failure-propagates regression test, real classification-logic tests for the `UNKNOWN` hard gate and `evidence_lost` debounce bypass), not just structural/shape assertions.
- Config validated at construction (`PipelineConfig.__post_init__`, added this pass) — bad thresholds fail immediately with a readable message instead of three layers deeper inside `DisparityEngine`.
- Calibration validated at construction (`StereoCalibration.__post_init__`, pre-existing) — shape and positivity checks.
- Pipeline lifecycle surface added this pass: `from_config()` (alternate constructor), `process_observation()` (accepts the new `StereoObservation` contract), `reset()` (clears `ThreatAssessor`'s cross-frame EMA/debounce state without discarding calibration/config), `close()` + post-close `RuntimeError` on further `process()`/`reset()` calls, `health()` (typed `PipelineHealth` snapshot).
- `DepthPerceptionResult` gained `timestamp` (opaque pass-through from `StereoObservation`/`process()` kwargs) and `valid_disparity_mask`/`valid_depth_mask` (explicit boolean arrays, computed once, rather than left for every caller to re-derive the `<=0`/`==0` sign convention themselves).

## Partially implemented / known limitations (not fixed in this pass — documented, not hidden)

- **`traversability_mask` naming wart.** The field holds a `TraversabilityResult` (region grid + navigation decision), not a pixel mask. Not renamed because `mp01_perception` reads this field by name today, and this recovery task's explicit scope excludes touching `mp01_perception`. A rename would need a coordinated two-repo change with a documented migration.
- **No NaN/Inf validation on `StereoCalibration`'s matrices.** Shape and positivity are checked; individual finite-value validation is not.
- **Calibration loading assumes pre-rectified input** (`R1`/`R2`/`P1`/`P2`/`Q` already computed by an external tool) — no path to compute these from raw intrinsics + extrinsics via `cv2.stereoRectify()` inside this library. Pre-existing design choice, not a defect from this pass, but worth knowing before wiring in a calibration source that only provides raw `R`/`T`.
- **`DistanceReader` is real, tested (added this pass — was previously the one real gap in test coverage), and used by `examples/live_demo.py`, but not part of `DepthPerceptionResult`.** A caller wanting a single-point distance reading must construct and call it separately; it is not merged into the canonical per-frame output, by design (see `docs/ARCHITECTURE.md`).
- **`docs/INTEGRATION_READINESS.md` describes a *planned* `mp01_perception` integration that has, in fact, already happened** — flagged with a status-update note at the top of that file rather than rewritten, since rewriting it would mean describing `mp01_perception`'s current internals in detail, and this pass's scope is this repo, not that one.
- **`mp01_perception` currently imports one internal module directly** (`depth_perception_engine.quality.frame_quality.looks_like_garbage_frame`, not re-exported from the top-level `__all__`) — pre-dates this recovery pass, contradicts the long-term goal that `mp01_perception` should only ever import the top-level public API, explicitly out of scope to fix here (would require touching `mp01_perception`).
- **No config serialization/hash support.** `PipelineConfig` is a plain validated dataclass; there's no `.to_dict()`/reproducibility-hash helper. Not needed for anything in this pass's scope; would matter once configs start being logged/compared across research runs.

## Not implemented (explicitly out of scope for this task)

IMU-assisted perception, temporal depth fusion, VIO, state estimation, learned/neural stereo, occupancy mapping beyond the existing region grid, rear/multi-camera rigs, ROS adapters (that's `mp01_perception`'s job), Jetson-specific optimization, CUDA/TensorRT.

## Deferred (real gaps, reasonable to fix later, deliberately not fixed now)

- Fixing the `traversability_mask` naming wart, coordinated with an `mp01_perception` migration.
- Deciding whether `mp01_perception` should get a re-exported `quality` surface at the top level, or continue reaching into the internal module.
- NaN/Inf validation on calibration matrix entries.
