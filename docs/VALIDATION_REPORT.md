# Validation report — baseline recovery pass (2026-08-05)

## Commands run

```bash
# Test suite
cd ~/PycharmProjects/depth_perception_engine
source .venv/bin/activate
pytest tests/ -q                                    # 131 passed
pytest tests/ -q --durations=5                       # slowest test: 0.07s

# Dead-file removal
git rm calibration.yml config/camera.yaml

# colcon build/test (run inside the mp01_ros2 docker container, which has
# the real ROS Humble toolchain — this host sandbox does not)
docker exec mp01_ros2 bash -lc '
  cd /root/mp01_ws && source /opt/ros/humble/setup.bash
  colcon build --packages-select mp01_msgs depth_perception_engine mp01_perception --symlink-install
  source install/setup.bash
  colcon test --packages-select depth_perception_engine mp01_perception
'
```

## Test results

- **Before this pass:** 67 tests, all passing.
- **After this pass:** 131 tests, all passing, 0.89s wall time. +64 new tests: `tests/test_distance_reader.py` (21, new file — closes a real coverage gap on a previously-untested public class), `tests/test_pipeline_config.py` (24, new file — validates the new `__post_init__`), `tests/test_pipeline.py` additions (22 — `from_config`, `process_observation`, timestamp pass-through, validity masks, `reset`/`close`/`health` lifecycle), `tests/test_depth_estimator.py` additions (5 — the new analytic known-value class).
- `colcon test --packages-select depth_perception_engine mp01_perception` inside the `mp01_ros2` container (this repo's own tests plus flake8/pep257 lint): **64 passed / 3 skipped** here, **157 passed / 1 skipped** in `mp01_perception` (zero regression, as expected — every change this pass made was additive, `mp01_perception` was not touched).

## Numerical validation

- **Differential (pre-existing, unchanged this pass):** `test_depth_estimator.py::TestZOnlyMatchesFullReprojection` — the closed-form Z-only depth formula compared against `cv2.reprojectImageTo3D`'s own Z channel, across the real calibration and synthetic Q matrices with a non-zero principal-point-offset term, plus invalid/negative/near-zero/huge-disparity edge cases.
- **Analytic, independent of OpenCV (new this pass):** `TestAnalyticKnownDepth` — hand-built Q matrices for the no-principal-point-offset case, where the general formula collapses exactly to `depth_m = focal_length_px × baseline_m / disparity_px`. Four known combinations checked (e.g. f=600px, baseline=0.065m, disparity=50px → expected exactly 0.78m), asserted to `rtol=1e-5` with **no OpenCV reprojection function on either side of the comparison** — this is the gap the original audit found (see `docs/IMPLEMENTATION_STATUS.md`) and closes it.
- **Real hardware (2026-08-05):** a real USB global-shutter stereo camera produced a stable 1.21m reading, ±2mm across 8 consecutive frames, against a real wall ~1.2m away — physically consistent. Region classification was physically sensible (blank wall → `PROBABLE_WALL`/`UNKNOWN`, a visible cable → `OBSTACLE` at 0.88m). One live catch of the anti-false-clear safety design: a spurious 4.79m disparity blip from stray noisy pixels was correctly downgraded to `BLOCKED` (not trusted as `CLEAR`) by the invalid-ratio override.

## Performance measurements (real hardware, this pass)

15 consecutive real frames, 320×240 per eye, `PipelineConfig()` defaults (128 disparities, SGBM block size 13):

| Metric | Value |
|---|---|
| `processing_time_ms` min | 14.73 ms |
| `processing_time_ms` max | 46.60 ms |
| `processing_time_ms` mean | 26.94 ms |
| `PipelineHealth.frames_processed` after 15 calls | 15 (confirms the new lifecycle bookkeeping is accurate) |
| `valid_disparity_mask` on the last frame | 10,468 / 76,800 pixels valid (13.6% — consistent with a low-texture wall scene) |

No unbounded memory growth observed across repeated calls (consistent with `ThreatAssessor`'s fixed-size per-beam state arrays — no growing buffers anywhere in the per-frame path). Not formally profiled for memory this pass — a stopwatch/`processing_time_ms` measurement, not a memory benchmark.

## Failures found and fixes applied

| # | Finding | Fix |
|---|---|---|
| 1 | No `from_config()`, `health()`, `reset()`, `close()` on `DepthPerceptionPipeline` — target lifecycle contract entirely absent | Added all four, additive (existing `.process(left, right)` two-arg call unaffected) |
| 2 | No `StereoObservation`/timestamp concept anywhere in the engine layer | Added `StereoObservation` + optional `left_timestamp`/`right_timestamp` kwargs on `process()` + `process_observation()` |
| 3 | `DepthPerceptionResult` had no explicit validity masks, no timestamp | Added `valid_disparity_mask`/`valid_depth_mask` (computed once in `fusion/result_builder.py`) + `timestamp` |
| 4 | `PipelineConfig` had zero construction-time validation — bad values only surfaced 3 layers deep inside `DisparityEngine` | Added `__post_init__` validation mirroring `StereoCalibration`'s existing pattern |
| 5 | `DistanceReader` — real, exported, used by `examples/live_demo.py` — had zero tests | Added `tests/test_distance_reader.py`, 21 tests |
| 6 | Depth math only verified differentially against OpenCV, no independent analytic check | Added `TestAnalyticKnownDepth`, 5 tests, hand-computed expected values |
| 7 | `calibration.yml` and `config/camera.yaml` — dead, git-tracked, unused, wrong-format calibration artifacts | Removed (`git rm`) |
| 8 | No `docs/ARCHITECTURE.md`/`DATA_CONTRACTS.md`/`CALIBRATION.md`/`IMPLEMENTATION_STATUS.md`/`VALIDATION_REPORT.md` | Added all five |
| 9 | `docs/INTEGRATION_READINESS.md` described a planned `mp01_perception` integration that has, in fact, already happened | Added a status-update note at the top; left the historical design content in place (still accurate as a record of what was planned and then actually built) |

## Known limitations

See `docs/IMPLEMENTATION_STATUS.md`'s "Partially implemented" section in full. Summary: `traversability_mask`'s misleading field name (not fixed — renaming breaks `mp01_perception` today, out of scope), no NaN/Inf validation on calibration matrices, calibration loading assumes pre-rectified input only, `mp01_perception` still imports one internal module (`quality.frame_quality`) directly rather than only the top-level public API (pre-dates this pass, out of scope to fix here), no config serialization/hash support.

## Deferred work

IMU-assisted perception, temporal depth fusion, VIO, state estimation, learned/neural stereo, rear/multi-camera rigs, ROS adapters, Jetson/CUDA optimization — all explicitly out of this task's scope per its own non-goals list.

## Final decision

**RESTORED WITH DOCUMENTED LIMITATIONS.**

The repository was already substantially healthy going into this pass (genuinely ROS-independent, one canonical execution path, real — not stubbed — algorithm code, mostly-behavioral test coverage, accurate pre-existing docs). This pass closed the real gaps found by the audit (missing lifecycle API surface, missing timestamp/validity-mask contracts, one untested public class, one OpenCV-only-verified math path, two dead files, five missing docs) without any breaking change to the existing public API — every addition is opt-in, and `mp01_perception`'s existing `.process(left, right)` call keeps working unmodified, confirmed by a zero-regression `colcon test` run. Not marked fully "RESTORED AND FROZEN" because two real, known limitations remain by deliberate choice (the `traversability_mask` naming wart and `mp01_perception`'s internal-module import) — both require touching `mp01_perception`, which is explicitly out of this task's scope, and are documented rather than hidden.
