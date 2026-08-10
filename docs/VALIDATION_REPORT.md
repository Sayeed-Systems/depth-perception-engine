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
- `colcon test --packages-select depth_perception_engine mp01_perception` inside the `mp01_ros2` container (this repo's own tests plus flake8/pep257 lint), re-run after syncing this pass's changes into the vendored copy (`git fetch` + `merge --ff-only` from this repo, same discipline as the earlier drift-reconciliation pass): **128 passed / 3 skipped** here (up from 64/3 before this pass — the 64 new tests, plus flake8/pep257 passing clean on every new file), **157 passed / 1 skipped** in `mp01_perception`, unchanged (zero regression, as expected — every change this pass made was additive, `mp01_perception` was not touched).

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

## Level 3, Phase E3 addendum (2026-08-06) — geometry wired into the pipeline

E2 (previous pass) built and validated `geometry.PointCloudBuilder` standalone; this pass wired it into `DepthPerceptionPipeline.process()` behind `PipelineConfig.enable_geometry` (default `False`). Full details: `docs/IMPLEMENTATION_STATUS.md`'s E3 addendum, `docs/DATA_CONTRACTS.md`'s "Geometry" section.

### Test results

`pytest tests/ -q`: **246 passed** (222 before this pass, +24 new — `tests/test_pipeline_geometry.py`). Zero existing test modified.

### Zero-regression evidence

`tests/test_pipeline_geometry.py::TestZeroRegression` — two pipelines built from configs identical in every field except `enable_geometry`, run on the same fixed synthetic stereo pair: `disparity_map`, `depth_map`, `valid_disparity_mask`, `valid_depth_mask`, `confidence`, every `traversability_mask` region's classification/median-depth/confidence/valid-count, every `obstacles` beam's index/bounds/distance/status, and `traversability_mask.decision` across 4 repeated calls (exercising `ThreatAssessor`'s cross-frame EMA/debounce state) — all identical, `enable_geometry=True` vs. `False`.

### Geometry correctness evidence (integration-level, beyond E2's own math validation)

- `test_valid_geometry_count_matches_valid_depth_mask_exactly`: `result.geometry.valid_mask` is bit-identical to `result.valid_depth_mask` for the same frame (both derive from the same `raw_disparity`/`Q`/depth-range clamp).
- `test_matches_e2_verified_builder_called_directly`: `result.geometry` (produced inside `process()`) is bit-identical to calling a freshly-constructed `PointCloudBuilder.build()` directly on `result.disparity_map` — proves the pipeline performs no second, divergent reprojection.
- `TestSingleCanonicalProducer::test_exactly_one_reprojectImageTo3D_call_site_exists`: static AST scan of every file under `src/` — exactly one `cv2.reprojectImageTo3D` call site exists anywhere in the library (`depth/depth_estimator.py`), mirroring `test_no_ros_dependency.py`'s existing scan style.

### Failure/degradation semantics

- Disabled (`enable_geometry=False`): `result.geometry is None`, zero added cost — no `PointCloudBuilder` even constructed.
- No valid pixels (flat, textureless synthetic stereo pair — verified empirically to reliably force `valid_disparity_mask.sum() == 0`, unlike identical-but-textured images which still yield scattered spurious SGBM matches): `result.geometry` is still present, a normal `PointCloud` with `valid_mask` all-`False` and `points` all-`NaN` — a data-quality signal, not an exception, and never a silent zero-fill (`np.any(points == 0.0)` asserted `False`).
- Genuine runtime error inside the geometry stage (simulated via mocking `PointCloudBuilder.build` to raise): propagates uncaught out of `process()`, invalidating the whole frame — identical treatment to an existing rectification failure, not silently swallowed into a fake empty result.

### Performance measurements (synthetic, this pass — no camera)

Two resolutions measured, `PipelineConfig()` defaults (128 disparities, SGBM block size 13), `N_WARMUP=15` before timing:

| Resolution | Calibration | Level 0-2 mean | Level 0-3 mean | Geometry-stage-only (isolated) | Absolute increase | Relative increase |
|---|---|---|---|---|---|---|
| 320×240 | real hardware fixture, `rectify=True` | 12.85 ms (std 1.05) | 16.66 ms (std 0.79) | — | +3.81 ms | +29.6% |
| 640×480 | synthetic, `rectify=False` (matches `examples/benchmark_point_cloud.py`'s E2 resolution) | 46-67 ms (std 5-40, noisy — see below) | 65-100 ms | 17-20 ms (std ~1-2, stable) | +19-33 ms across repeated runs | +30-59% across repeated runs |

Memory: peak traced memory over 20 calls at 640×480, geometry off vs. on: 12,018 KiB → 21,915 KiB (+9,896 KiB — matches the expected `(H×W×3×4) + (H×W×1)` bytes for one live `PointCloud` plus normal per-call churn; no growth across repeated batches in either configuration, confirmed by re-running the batch multiple times).

**Root-cause note on the 640×480 run-to-run variance:** the *baseline* (geometry disabled) measurement itself varies as much as the geometry-enabled one on this shared/virtualized development container (StereoSGBM at 640×480/128-disparities is CPU-heavy and visibly sensitive to scheduling noise here — visible even with geometry never in the loop). Tested and ruled out: Python GC pressure from the new per-call ~4 MB allocation (`gc.disable()` during a controlled back-to-back A/B run changed the measured geometry overhead by under 1 ms — 19.43 ms enabled vs. 20.49 ms disabled-GC, not the cause). The isolated geometry-only measurement (17-20 ms, low variance, consistent with E2's own ~11-19 ms `examples/benchmark_point_cloud.py` result) is the trustworthy number; the wide pipeline-level range above is reported honestly rather than smoothed over, per this task's explicit "measure and document, do not redesign without justification" instruction — no redesign was undertaken, since the isolated measurement shows the geometry stage itself is neither the cause nor unusually expensive.

### Final decision (E3)

**COMPLETE — READY FOR E4.** Camera-frame-only 3D geometry is now a first-class, opt-in, config-gated `DepthPerceptionPipeline.process()` output, additive to every existing Level 0-2 field, with zero measured regression and one canonical, math-verified producer. `ObstacleCloud`, `FreeSpaceRays`, `GeometryMetrics`, body-frame transform, and occupancy mapping remain unbuilt — E4+, not claimed here.

## Level 3, Phase E4 addendum (2026-08-06) — body-frame transformation

E3 (previous pass) produced camera-optical-frame geometry only. This pass adds the rigid transform into `FrameId.BODY`, using the exact convention E1 already froze in `frames.py`/`docs/COORDINATE_FRAMES.md` — verified consistent between code and docs before any implementation began (`p_out = rotation @ p_in + translation`; no mismatch found, no STOP triggered). Full details: `docs/IMPLEMENTATION_STATUS.md`'s E4 addendum, `docs/COORDINATE_FRAMES.md`'s E4 sections.

### Test results

`pytest tests/ -q`: **291 passed** (246 before this pass, +45 new — `tests/test_rigid_transform.py` (26), `tests/test_pipeline_body_frame.py` (19)). Zero existing test modified except one E3 test's exact-field-list assertion, updated to include the new `geometry_body` field appended at the end (`tests/test_pipeline_geometry.py::TestResultContract::test_existing_fields_unchanged_in_name_type_and_order`) — the field list changing is exactly what an additive field is supposed to do; the *order* and every *other* field's presence/default was asserted unchanged by the same test, and still is.

### Mathematical validation (independent of OpenCV — `transform_point_cloud` calls no cv2 function)

`tests/test_rigid_transform.py`, all hand-computed, `atol=1e-5` (`1e-4` for the inverse round-trip, accounting for one float32 truncation in between):

1. **Identity** (`R=I`, `t=0`): output XYZ == input XYZ.
2. **Pure translation**: known XYZ + known translation == exact expected output.
3. **90° rotation about X, Y, and Z independently**: `(1,2,3)` → `(1,-3,2)` / `(3,2,-1)` / `(-2,1,3)` respectively — each hand-derived from the standard rotation-matrix formula, not from the function under test.
4. **Combined rotation + translation**: `Rz(90°) @ (1,2,3) + (10,-5,0.5) = (8,-4,3.5)`, confirmed; plus a 2×2 organized multi-point cloud confirmed per-pixel.
5. **Inverse round-trip**: forward transform (37° rotation about Z + translation) then its mathematical inverse (`R.T`, `-R.T @ t`, computed directly in the test — `RigidTransform` has no `.inverse()` method, none was added, since that would extend a frozen E1 type beyond this task's scope) recovers the original random 4×5 cloud within `1e-4`.
6. **Organized shape preserved**: output `(H, W, 3)` matches input.
7. **`valid_mask` preserved exactly** — bit-identical, and confirmed to be a copy (not the same array object as the source).
8. **NaN stays NaN**: invalid input points remain `(NaN, NaN, NaN)` after transform, with no special-case masking code needed — IEEE-754 arithmetic already guarantees this (a zero `rotation` entry times NaN is NaN, not 0).
9. **`frame_id` changes correctly**: set to `transform.to_frame`; confirmed generic (not hardcoded to `BODY` — an arbitrary `to_frame` string is honored identically), matching Task 3's "zero assumptions about where the rig is mounted."
10. **`timestamp` preserved unchanged** (including the `None` case).

Also covered: `confidence` preservation (copied when present, stays `None` when absent), non-mutation of the source cloud, determinism (identical output across repeated calls on the same input), and rejection of non-finite `rotation`/`translation` (invalid calibration must never silently produce a transformed cloud) and of a `transform.from_frame` that doesn't match `cloud.frame_id`.

### Zero-regression evidence

`tests/test_pipeline_body_frame.py::TestZeroRegression` — two pipelines, configs and calibration identical, one with `body_T_camera_left` supplied and one without: `disparity_map`, `depth_map`, both validity masks, `confidence`, `traversability_mask.decision`, every obstacle beam's status, and — critically — **the E3 camera-frame `geometry` cloud itself** (`points` and `valid_mask`, bit-identical, `frame_id` still `CAMERA_OPTICAL_LEFT` either way) are all identical regardless of whether body-frame transformation is also being computed. Confirms E4 is purely additive, exactly as required.

### Geometry correctness evidence (integration-level, beyond the standalone math validation above)

- `test_matches_e4_verified_transform_called_directly`: `result.geometry_body` (produced inside `process()`) is bit-identical to calling `transform_point_cloud()` directly on `result.geometry` outside the pipeline — proves the pipeline uses the one canonical transform implementation, not a second divergent one.
- `test_valid_mask_identical_to_camera_cloud` / `test_invalid_pixels_stay_nan_in_body_frame`: `geometry_body.valid_mask` matches `geometry.valid_mask` exactly; invalid pixels are NaN in both frames.

### Failure/absent-extrinsic semantics

- No `body_T_camera_left` supplied: `geometry_body` stays `None` — **never** silently treated as identity, even though `geometry` (camera-frame) is still produced normally. Enforced by `calibration.contracts.RigCalibration`'s reused, already-frozen validation, not reimplemented.
- `enable_geometry=False`: both `geometry` and `geometry_body` are `None` regardless of whether an extrinsic was configured — there is no camera-frame cloud to transform.
- Invalid extrinsic (non-finite `rotation`/`translation`, or wrong `from_frame`/`to_frame`): rejected at `DepthPerceptionPipeline` **construction** time (fails fast, before any frame is ever processed), not silently accepted.

### Performance measurements (synthetic organized cloud, 640×480 — same resolution as the E2/E3 benchmarks)

`examples/benchmark_body_transform.py`, transform stage only (no pipeline/SGBM/rectify in the loop), `N_WARMUP=15`, `N_ITERS=200`, three runs:

| Run | mean | std | max |
|---|---|---|---|
| 1 | 15.51 ms | 4.64 ms | 48.07 ms |
| 2 | 17.39 ms | 6.11 ms | 39.39 ms |
| 3 | 16.37 ms | 6.01 ms | 51.77 ms |

Memory: output `PointCloud.points` = 3,600 KiB, `valid_mask` = 300 KiB (both exact, `H×W×3×4`/`H×W×1` bytes). Peak traced memory flat across 5 successive batches of 20 calls in every run (0.0 KiB growth) — bounded, no leak.

**Root-cause note:** a raw-numpy microbenchmark of the identical reshape/cast/matmul/reshape/cast chain, outside this library's function-call/validation overhead entirely, reproduced the same ~9-15 ms magnitude on this shared/virtualized development container — confirming the cost is this environment's memory bandwidth for a ~300K-row float64 matmul + two dtype casts, not an artifact of `transform_point_cloud()`'s own overhead, and not something a Python-loop-vs-vectorized difference would explain (there is no loop). The implementation is already fully vectorized (Task 9: "the implementation should already be vectorized" — confirmed, not redesigned) — no premature optimization was undertaken, per this task's explicit instruction.

### Final decision (E4)

**COMPLETE — READY FOR E5.** Camera-optical-frame geometry now transforms correctly into `FrameId.BODY` using the exact frozen `RigidTransform` convention, calibration-driven via a new additive, defaulted `DepthPerceptionPipeline` parameter, with zero hardcoded rig dimensions anywhere (verified: the transform function is frame-name-agnostic). Camera-frame geometry, all Level 0-2 outputs, and the entire pre-E4 test suite are unchanged. `ObstacleCloud`, `FreeSpaceRays`, `GeometryMetrics`, vehicle envelopes, occupancy mapping, IMU, and temporal fusion remain unbuilt — E5+, see `docs/E5_IMPLEMENTATION_PLAN.md`.

## Level 3, Phase E5 addendum (2026-08-06) — ObstacleCloud, FreeSpaceRays, GeometryMetrics

E4 (previous pass) produced body-frame geometry only. This pass converts it into structured spatial evidence: `ObstacleCloud`, `FreeSpaceRays`, `GeometryMetrics` — all three previously producer-less frozen E1 types. Full details: `docs/IMPLEMENTATION_STATUS.md`'s E5 addendum, `docs/DATA_CONTRACTS.md`'s "Spatial evidence" section.

### Test results

`pytest tests/ -q`: **360 passed** (293 before this pass, +67 new — `tests/test_obstacle_extractor.py` (20), `tests/test_free_space.py` (15), `tests/test_geometry_metrics.py` (9), `tests/test_pipeline_spatial_evidence.py` (23)). Two E3-era tests updated to reflect legitimate E5 additions (not regressions): `TestResultContract`'s exact-field-list assertion (three new fields appended), and `TestSingleCanonicalProducer`'s "ObstacleCloud/FreeSpaceRays must not appear in pipeline.py" check (replaced with an import-provenance check, since those strings appearing is now correct, not a leak).

### A real bug found and fixed during this pass

Both `build_obstacle_cloud` and `build_free_space_rays`'s first draft admitted a contract-violating Inf-valued-but-`valid_mask=True` point under the **actual default configuration** (`obstacle_max_range_m = +inf`): `inf <= inf` is `True` in IEEE-754, so the range-bound check alone did not exclude it, and `build_free_space_rays` would have divided `Inf / Inf`, producing a `NaN` direction. Caught by this pass's own `tests/test_obstacle_extractor.py::TestNaNInfNeverBecomeObstacles::test_inf_point_marked_valid_is_excluded_by_range_not_by_crashing` and `tests/test_free_space.py::TestInvalidPointsGenerateNoRay::test_inf_marked_valid_produces_no_finite_ray` before this ever reached the pipeline. Fixed by adding an explicit `np.isfinite(...)` guard in both functions, independent of the configured range bounds.

### Mismatches found and resolved (Task 1's audit, reported per its explicit instruction)

1. Task prose requested a `timestamp` field on `ObstacleCloud`/`FreeSpaceRays` — neither frozen type has one. Resolved: use the frozen shape exactly; the frame's timestamp remains available via `geometry_body.timestamp`.
2. Task prose describes `FreeSpaceRays` as `(origin, endpoint)` pairs — the frozen type stores `(origins, directions, ranges_m)`. Mathematically equivalent (verified directly, `endpoint = origin + direction * range`); built the frozen parametrization.
3. `GeometryMetrics`' frozen 4 fields are narrower than the task's metric wishlist (max distance, separate obstacle/ray counts, unknown_fraction, coverage). Populated exactly the 4 frozen fields with precise definitions; the rest flagged as E6 candidates, not silently added.

### Core safety rule evidence — "invalid depth must never become free space"

`tests/test_pipeline_spatial_evidence.py::TestUnknownSpaceSafetyRule`, through the real pipeline (not just the unit level):
- A flat, textureless synthetic scene (zero valid disparity anywhere, verified reliable in the E3 test suite) produces `obstacle_cloud.points.shape[0] == 0` and `free_space_rays.ranges_m.shape[0] == 0` — never a crash, never a fabricated claim.
- On a realistic mixed valid/invalid frame: `obstacle_cloud.points.shape[0]` and `free_space_rays.ranges_m.shape[0]` both exactly equal `geometry_body.valid_mask.sum()` — every invalid pixel is excluded from both outputs, exactly, not approximately.
- A point excluded from `ObstacleCloud` by a tight range window still produces a free-space ray (its surface evidence is real regardless of the obstacle-reporting range window) — proving range-filtering and free-space-ray generation are independent, and that excluding a point from obstacle-reporting is never conflated with marking the space in front of it as free or unknown.

### Zero-regression evidence

`tests/test_pipeline_spatial_evidence.py::TestZeroRegression`/`TestCameraAndBodyGeometryUnchanged`: `disparity_map`, `depth_map`, both masks, `confidence`, traversability, obstacles (Level 0-2), and — critically — `result.geometry` (E3) and `result.geometry_body` (E4) are bit-identical whether or not E5's two flags are enabled.

### Geometry correctness evidence

`TestDerivedFromBodyPointCloudOnly`: `result.obstacle_cloud`/`result.free_space_rays` (produced inside `process()`) are bit-identical to calling `build_obstacle_cloud()`/`build_free_space_rays()` directly on `result.geometry_body` outside the pipeline — proves no second, divergent implementation. A sampled check confirms every free-space ray's reconstructed endpoint (`origin + direction * range`) matches an actual valid body-frame surface point to within `1e-3` m.

### Performance measurements (synthetic, this pass — no camera, 640×480, matching E2/E3/E4's benchmark resolution)

`examples/benchmark_spatial_evidence.py`, two runs, ~261,000 valid points / 307,200 total:

| Stage | Run 1 mean | Run 2 mean | std (both runs) |
|---|---|---|---|
| Obstacle-cloud filtering | 19.10 ms | 18.61 ms | ~2.2 ms |
| Free-space ray generation | 28.83 ms | 30.43 ms | ~3-5 ms |
| Geometry-metrics aggregation | 0.37 ms | 0.23 ms | ~0.1-0.2 ms |
| **Total E5 added latency** | **48.30 ms** | **49.28 ms** | ~4-5 ms |

Memory: `ObstacleCloud` output ≈ 4,071 KiB (260,577 points), `FreeSpaceRays` output ≈ 7,146 KiB (261,322 rays), both exact. Peak traced memory flat across 5 successive batches of 20 full-E5 calls (0.2 KiB growth, i.e. bounded, no leak).

### Final decision (E5)

**COMPLETE — READY FOR E6.** `ObstacleCloud`/`FreeSpaceRays`/`GeometryMetrics` are implemented, calibration/config-driven, purely geometric (no semantic/object classification), and derive exclusively from the E4 body-frame `PointCloud` with no parallel geometry chain. The core safety rule — invalid depth never becomes free space or an obstacle — is structurally enforced and tested at both the unit and pipeline-integration level. Camera-frame geometry, body-frame geometry, and every Level 0-2 output are unchanged with E5 enabled. **E5 does NOT produce an occupancy map.** Occupancy/voxel mapping, temporal fusion, vehicle envelopes, collision-risk scoring, and IMU remain unbuilt — E6+, see `docs/E6_IMPLEMENTATION_PLAN.md`.

## Level 3, Phase E6 addendum (2026-08-06) — robustness, degradation semantics, performance hardening

E5 (previous pass) built the geometry chain end to end. This pass hardens it: adversarial input coverage, an explicit UNKNOWN-space safety invariant proof, degradation-quality classification, failure containment, determinism, recovery, and full performance/memory characterization. **No algorithm changed.** Full details: `docs/IMPLEMENTATION_STATUS.md`'s E6 addendum, `docs/LEVEL3_ARCHITECTURE.md`'s E6 update.

### Pre-E6 audit findings (Task 1)

- Zero `try/except` anywhere in the Level 3 chain inside `process()` — any exception already aborted the whole call, atomically, before E6 started.
- NaN/Inf handling was already extensive (E2/E5 built it); E5's own Inf-admission bug fix was the most recent example.
- `PipelineHealth` is explicitly lifecycle-only ("not a per-frame diagnosis" — pre-existing docstring) — it was never the right place for degradation semantics.
- No caching, no cross-call state, anywhere in the E2-E5 chain except `ThreatAssessor`'s pre-existing Level 0-2 EMA/debounce.

### Vulnerabilities / weaknesses actually discovered (Task 1 → Task 2)

One real, non-hypothetical finding: **unsupported image dtype surfaces as a raw, uncaught `cv2.error`**, not this library's own `RuntimeError` wrapper (`DisparityEngine.compute_disparity` wraps `StereoSGBM.compute()` failures into `RuntimeError`, but the grayscale-conversion call site in `pipeline.py` — Level 0-2 code, predates E2-E5 — is unprotected). Safety is not compromised (the frame still fails atomically, no geometry is fabricated), but the exception type is inconsistent. Judged out of E6's "harden the geometry chain, not unrelated Level 0-2 code" scope; documented, not silently patched. No other vulnerability was found — this is stated explicitly, not omitted, per Task 1's "do not assume a vulnerability exists" instruction.

### Files changed

New: `geometry/geometry_metrics.py` additions (`GeometryQuality`, `classify_geometry_quality`), `tests/test_geometry_quality.py`, `tests/test_adversarial_geometry.py`, `tests/test_failure_containment.py`, `tests/test_determinism.py`, `tests/test_state_recovery.py`, `tests/test_performance_guards.py`, `examples/benchmark_e6_full_pipeline.py`, `examples/benchmark_e6_memory_stability.py`, `docs/E7_IMPLEMENTATION_PLAN.md`. Modified: `config/pipeline_config.py` (2 new threshold fields), `geometry/__init__.py` (2 new exports), `tests/test_public_api.py` (`INTERNAL_SYMBOLS` list corrected — a pre-existing staleness bug, not an E6-introduced one). **Zero changes** to `stereo/`, `depth/depth_estimator.py`'s math, `geometry/point_cloud_builder.py`, `geometry/rigid_transform.py`, `geometry/obstacle_extractor.py`, `geometry/free_space.py`'s math, `traversability/`, `obstacles/`, `mp01_perception`, anything ROS.

### Adversarial input matrix (Task 2) — `tests/test_adversarial_geometry.py`

| # | Scenario | Result |
|---|---|---|
| A | Normal scene | PASS — substantial valid geometry, HEALTHY or DEGRADED |
| B | Textureless scene | PASS — valid_fraction == 0.0, zero obstacles/rays, NO_USABLE_GEOMETRY |
| C | Extremely sparse valid depth | PASS — 0 < valid_fraction < 0.05, NO_USABLE_GEOMETRY |
| D | All invalid disparity | PASS — zero valid points, all NaN |
| E | Zero disparity | PASS — rejected identically to D |
| F | Negative disparity | PASS — rejected regardless of magnitude |
| G | NaN values | PASS — excluded, no crash |
| H | ±Inf values | PASS — excluded, no leak into valid points |
| I | Too-near depth | PASS — rejected, not clamped |
| J | Too-far depth | PASS — rejected, not clamped |
| K | Mixed valid + invalid | PASS — counts exactly match valid pixel count |
| L | Empty geometry after range filtering | PASS — obstacle_cloud empty, free_space_rays unaffected |
| M | Malformed image shapes | PASS — ValueError, before any Level 3 stage runs |
| N | Left/right size mismatch | PASS — ValueError (cross-checked against pre-existing `test_pipeline.py` coverage) |
| O | Unsupported dtype | PASS (with the documented caveat above — raises, does not fabricate geometry) |
| P | Invalid calibration/transform | PASS — rejected at pipeline construction, before any frame processed |
| Q | Repeated identical frames | PASS — see Determinism below (`tests/test_determinism.py`) |
| R | Healthy → invalid → healthy | PASS — see Recovery below (`tests/test_state_recovery.py`) |

20/20 tests pass for A-P; Q/R get their own dedicated test files (below) since they require multi-call sequences.

### UNKNOWN-space invariant evidence (Task 3)

`invalid disparity → invalid depth → invalid geometry → NO FreeSpaceRay → UNKNOWN` verified at three levels: (1) analytically, `tests/test_free_space.py`/`test_obstacle_extractor.py` (E5); (2) through the real pipeline on a genuinely mixed frame, `tests/test_adversarial_geometry.py::TestK_MixedValidInvalidGeometry` — obstacle/ray counts exactly equal the valid pixel count, never more; (3) the degenerate all-invalid case, `TestB_TexturelessScene` — zero rays, zero obstacles, `NO_USABLE_GEOMETRY`, not a crash and not a fabricated claim. No code path anywhere in `build_obstacle_cloud`/`build_free_space_rays` adds a point/ray without first ANDing `cloud.valid_mask` — confirmed by direct source reading (Task 1) and now by this test matrix.

### Degradation semantics (Task 4/5)

No new frozen contract — `PipelineHealth` is lifecycle-only by its own pre-existing design; `DepthPerceptionResult` already carries what's needed. New: `geometry.classify_geometry_quality(metrics, healthy_min_valid_fraction, degraded_min_valid_fraction) -> GeometryQuality.{HEALTHY,DEGRADED,NO_USABLE_GEOMETRY}` — one field (`valid_fraction`), two configurable thresholds, boundary-tested exhaustively (`tests/test_geometry_quality.py`, 19 tests: tier assignment, inclusive-lower-bound boundaries, the equal-thresholds edge case, threshold validation). Opt-in only — not auto-computed by `process()`, not a new result field.

### Failure-containment behavior (Task 6)

`tests/test_failure_containment.py`, 6 tests: simulated failure at each of camera point-cloud construction, body transform, obstacle extraction, and free-space generation, plus a generic non-`RuntimeError` exception type. In every case: the exception propagates uncaught, and `PipelineHealth.frames_processed`/`last_confidence`/`last_processing_time_ms` are confirmed unchanged — the failed call had zero observable effect on pipeline state. An already-successfully-computed `obstacle_cloud` is provably discarded (never reaches the caller) if `free_space_rays` then fails — stricter containment than the mission's stated minimum.

### Determinism results (Task 7)

`tests/test_determinism.py`, 4 tests: same-pipeline-instance repeated calls and independent-fresh-instances, both a normal and a textureless (all-invalid) scene. Verified exactly equal (or `np.testing.assert_allclose` within `rtol=atol=1e-6` for float arrays): `disparity_map`, `depth_map`, both validity masks, `confidence`, `geometry`/`geometry_body` points and masks, `obstacle_cloud` points/distances, `free_space_rays` origins/directions/ranges, `geometry_metrics` (exact dataclass equality), and the quality classification. `processing_time_ms` is explicitly never asserted equal (wall-clock).

### Healthy → invalid → healthy recovery results (Task 8)

`tests/test_state_recovery.py`, 5 tests: the invalid frame in the middle produces zero valid geometry (not a blend with the previous healthy frame); the following healthy frame reproduces the *first* healthy frame's geometry exactly; no Level 3 result object's identity is shared across calls (ruled out via `is not` checks, not just value equality); three repeated healthy/invalid cycles remain stable; `ThreatAssessor`'s Level 0-2 debounce is explicitly confirmed to not leak into Level 3 geometry (geometry is instantaneous, zero smoothing, by design).

### Performance table (Task 9)

`examples/benchmark_e6_full_pipeline.py`, warm-up = 15 discarded iterations, 100 timed iterations, per stage:

**320×240 (real hardware calibration, rectify=True — current development resolution):**

| Stage | mean | median | std | p95 | max |
|---|---|---|---|---|---|
| rectify | 0.98 ms | 0.86 ms | 0.32 ms | 1.55 ms | 2.24 ms |
| disparity | 6.71 ms | 6.27 ms | 1.36 ms | 10.16 ms | 11.10 ms |
| depth | 0.78 ms | 0.62 ms | 0.33 ms | 1.38 ms | 1.90 ms |
| camera_cloud | 5.12 ms | 4.91 ms | 1.29 ms | 7.12 ms | 10.55 ms |
| body_transform | 3.11 ms | 3.00 ms | 0.83 ms | 4.53 ms | 4.89 ms |
| obstacle_cloud | 5.02 ms | 4.99 ms | 1.13 ms | 6.82 ms | 10.47 ms |
| free_space_rays | 5.30 ms | 5.20 ms | 1.29 ms | 7.52 ms | 9.48 ms |
| geometry_metrics | 0.15 ms | 0.12 ms | 0.08 ms | 0.36 ms | 0.41 ms |
| **TOTAL `process()`** | **42.62 ms** | **32.97 ms** | **29.10 ms** | **115.45 ms** | **168.18 ms** |

Effective FPS (mean-based): **23.5**. Sum of per-stage means (27.17 ms) is well under the real `process()` total (42.62 ms) — the remainder is Level 0-2 traversability/obstacle-scan cost (not separately instrumented) plus call overhead.

**640×480 (synthetic calibration, rectify=False — higher realistic resolution, same synthetic-calibration technique E3's own benchmark established):**

| Stage | mean | median | std | p95 | max |
|---|---|---|---|---|---|
| disparity | 67.74 ms | 67.83 ms | 10.70 ms | 81.57 ms | 93.83 ms |
| depth | 4.99 ms | 4.52 ms | 1.09 ms | 7.01 ms | 9.28 ms |
| camera_cloud | 24.16 ms | 23.85 ms | 2.60 ms | 29.22 ms | 31.35 ms |
| body_transform | 18.88 ms | 11.08 ms | 13.91 ms | 48.15 ms | 70.92 ms |
| obstacle_cloud | 26.10 ms | 23.82 ms | 6.22 ms | 37.67 ms | 59.08 ms |
| free_space_rays | 31.90 ms | 29.20 ms | 6.63 ms | 46.74 ms | 59.15 ms |
| geometry_metrics | 0.59 ms | 0.43 ms | 0.81 ms | 0.90 ms | 6.91 ms |
| **TOTAL `process()`** | **185.95 ms** | **185.38 ms** | **23.27 ms** | **234.75 ms** | **248.87 ms** |

Effective FPS (mean-based): **5.4**. As in every prior phase's benchmark, this shared/virtualized development container shows substantial run-to-run variance (visible in `body_transform`'s std being ~74% of its own mean) — consistent with the environment noise root-cause already established and verified (via a GC-disabled A/B test) in the E3 addendum above; not re-litigated here.

### Memory / long-run results (Task 10)

`examples/benchmark_e6_memory_stability.py`, 500 frames (20 discarded warm-up), RSS sampled every 50 frames via `/proc/self/status`:

- RSS at frame 1: 82,252 KiB → frame 500: 84,960 KiB (+2,708 KiB total, +5.43 KiB/frame average).
- RSS trend across the 10 checkpoints was **not** strictly non-decreasing (it fluctuated between +1,784 and +5,296 KiB relative to baseline) — allocator/GC noise, not a sustained climb.
- Live object counts for `PointCloud`/`ObstacleCloud`/`FreeSpaceRays`/`GeometryMetrics` stayed flat at 2/1/1/1 throughout all 500 frames (2 `PointCloud` = `geometry` + `geometry_body` of the single most-recent result) — no growth proportional to frame count. After the script's own last reference was deleted and `gc.collect()` ran, all four counts dropped to 0 — confirming the pipeline itself retains no hidden reference to any past result.

**Stated precisely, per Task 10's own instruction: no monotonic memory growth was observed over 500 frames on this platform. This is not a formal proof of leak-freedom under all conditions or longer runs.**

### Full regression test count (Task 12)

`pytest tests/ -q`: **416 passed** (360 before E6, +56 new: `test_geometry_quality.py` 19, `test_adversarial_geometry.py` 20, `test_failure_containment.py` 6, `test_determinism.py` 4, `test_state_recovery.py` 5, `test_performance_guards.py` 2). Zero existing test's assertions weakened or removed; one pre-existing test-infrastructure staleness bug fixed (`INTERNAL_SYMBOLS` list, see above).

### API-freeze verification (Task 13)

`tests/test_public_api.py` (23 tests) + `tests/test_no_ros_dependency.py` (5 tests) + `tests/test_imports.py` (2 tests) re-run clean. Directly verified via a standalone script (not just the test suite) that none of `PointCloud`/`ObstacleCloud`/`FreeSpaceRays`/`GeometryMetrics`/`PointCloudBuilder`/`transform_point_cloud`/`build_obstacle_cloud`/`build_free_space_rays`/`build_geometry_metrics`/`GeometryQuality`/`classify_geometry_quality`/`RigidTransform`/`FrameId`/`RigCalibration` are reachable from the top-level `depth_perception_engine` package or present in `__all__`; `__all__` is still exactly the original 19-symbol Tier 1/2 set; no `DepthPerceptionEngine` alias exists.

### Known limitations

- The `cv2.error`-vs-`RuntimeError` classification gap for unsupported image dtype (documented above), left unfixed as out of E6's scope.
- `GeometryQuality`'s two thresholds are undocumented-against-any-real-dataset placeholders (0.5/0.05) — a policy choice for a real deployment to override, not a validated value.
- Performance numbers reflect this shared/virtualized development container, not dedicated/embedded hardware — directional, not a certification-grade number (see Task 9 above and the E3 addendum's original root-cause note).
- Memory observation is a single 500-frame run on one platform, not a formal leak-freedom proof.

### Final decision (E6)

**COMPLETE — READY FOR E7.** The E2-E5 geometry chain is now hardened: adversarial-input-tested (A-R, 20 dedicated tests plus determinism/recovery), the mandatory UNKNOWN-space safety invariant is proven structurally and behaviorally, degradation is classifiable using existing contracts plus one new opt-in helper (no frozen-contract change), failures are contained atomically with zero fabricated geometry, determinism and healthy→invalid→healthy recovery both hold exactly, and full per-stage performance/memory behavior is characterized (not merely claimed) at two resolutions. No algorithm was changed; the entire historical suite (416 tests) passes; the API freeze is directly verified intact.

## Level 3, Phase E7 addendum (2026-08-07) — integrated validation, visual proof, and freeze

E6 hardened the chain in isolation (synthetic fixtures, mocked failures, a shared dev container). E7 validates it as one coherent system: deterministic synthetic ground truth with hand-computed expected values, then real calibrated stereo hardware, then a standalone visual proof — and freezes Level 3 on the result.

### Pre-E7 state (Task 1)

Branch `main`, 7 commits ahead of `origin/main`, full E2-E6 work uncommitted (unchanged from every prior phase — nothing has been committed in this session unless explicitly requested). Full suite green (416/416) before any E7 change. Public API, ROS-independence, and single-canonical-path tests re-confirmed clean. Real camera hardware (`/dev/video0`) confirmed accessible.

### Synthetic ground-truth acceptance (Task 2) — `tests/test_e7_synthetic_ground_truth.py`, all PASS

Real hardware calibration (`fx≈614.52px`, `baseline≈64.73mm`), hand-computed expected values (not derived from the code under test), numeric tolerance `1e-3 m` (`1e-4 m` for the exact rotation check):

| # | Scenario | Result |
|---|---|---|
| 1 | Flat fronto-parallel plane (50px disparity) | PASS — expected depth 0.7955 m, measured 0.7955 m at centre pixel; full chain (disparity→depth→camera XYZ→body XYZ→obstacle→ray endpoint) all matched |
| 2 | Near (0.3978 m) + far (1.9888 m) background | PASS — ordering correct, every ray endpoint Z matched one of exactly two expected values |
| 3 | Partial invalid region (20 px, 6 invalid) | PASS — exactly 14 valid points, 14 obstacle points, 14 rays |
| 4 | Known 90° rotation + translation body transform | PASS — body point matched `R·p + t` to `1e-4 m`, and was confirmed distinct from the untransformed camera point |
| 5 | Empty/textureless scene | PASS — zero obstacle points, zero rays, `valid_fraction=0.0`, `NO_USABLE_GEOMETRY`, zero fabricated (all-NaN) points |
| 6 | Mixed 4-region scene (near/far/invalid/too-far) | PASS — exact valid count (38,400/76,800 = 50.0%), each region independently correct |

### Real hardware validation (Task 3, 7, 8)

Real calibrated USB global-shutter stereo rig, `/dev/video0`, `examples/e7_live_capture_session.py`. Two sessions were run: the first was judged unrepresentative mid-review (a persistent close object dominated every phase's global-nearest-point metric and prevented the textureless phase from actually reaching a low-texture state) and re-run with the operator's agreement after repositioning the camera — both the judgment call and both datasets are recorded here, not silently discarded. Second session: 7 phases × 24 real frames = 168 frames, zero read errors, zero pipeline errors, no parameter tuned per phase.

| Phase | mean latency | mean valid_fraction | quality states seen |
|---|---|---|---|
| baseline | 34.83 ms | 0.443 | DEGRADED (24/24) |
| toward (object moved toward camera) | 32.39 ms | 0.450 | DEGRADED (24/24) |
| away (object moved away) | 32.98 ms | 0.500 | DEGRADED 9 / HEALTHY 15 |
| left | 33.19 ms | 0.491 | DEGRADED 14 / HEALTHY 10 |
| right | 33.20 ms | 0.398 | DEGRADED (24/24) |
| occlusion (partial) | 33.97 ms | 0.339 | DEGRADED (24/24) |
| textureless | 34.35 ms | 0.361 | DEGRADED (24/24) |

**UNKNOWN-space invariant, directly re-verified on real sensor data** (Task 3's real point of this exercise): for all 7 phases, `int(geometry_body.valid_mask.sum()) == obstacle_cloud.points.shape[0] == free_space_rays.ranges_m.shape[0]` exactly (28,019 to 36,750 depending on phase), and `np.all(np.isnan(geometry_body.points[~valid_mask]))` held with zero exceptions — the same guarantee proven synthetically in E5/E6/Task 2 above, now confirmed on 168 real frames with zero mismatches.

**Degradation semantics on real data**: `occlusion` (0.339) and `textureless` (0.361) show measurably lower valid-geometry fraction than `away`/`left` (~0.49-0.50) — correctly directioned, real evidence, not fabricated. `away`/`left` crossing into `HEALTHY` some frames and `DEGRADED` others is real frame-to-frame variation in a live, imperfectly-controlled human interaction, not noise in the classifier itself (the classifier's own boundary behavior was already exhaustively tested in Task 5/E6).

**Honest limitation, not glossed over**: `GeometryMetrics.min_obstacle_distance_m` (the single global-nearest-point scalar) did **not** cleanly track the physically-moved object's distance during `toward`/`away` — investigated directly (see the `toward`/`away` per-frame trend, both essentially flat at ~0.33 m throughout, with occasional unrelated jumps during `toward` consistent with the object briefly passing in front of a different, closer, persistent surface). Root cause: this specific room/rig has a separate nearby surface that remained the frame's global minimum throughout most of both sessions, structurally masking a single moved object's contribution to that one particular scalar. This is a property of **this test environment and this specific metric**, not a defect in the geometry chain — the chain's per-pixel correctness was independently and exactly proven in Task 2's synthetic scenario 4 (a known transform matched to `1e-4 m`) and holds, mismatch-free, across all 168 real frames above. Level 0-2's existing per-beam `ObstacleAssessment` (unchanged throughout E2-E7) is the more appropriate tool for spatially-localized tracking and was not exercised by this specific check. **This finding is reported rather than hidden or re-run until it looked better** — the retry between sessions fixed the textureless-phase confound (session 1) but not this one, and no third attempt was made.

### Visual validation tool (Task 4/5/6)

`examples/visualize_level3.py` — standalone (optional `viz` extra, `pip install -e ".[viz]"`; `src/depth_perception_engine/` imports matplotlib nowhere). One figure per frame: left image, disparity (invalid pixels blank, not zero-colored), metric depth (same), and a BODY-frame 3D panel. The 3D panel draws an explicit origin+axes triad (X forward/red, Y left/green, Z up/blue — `docs/COORDINATE_FRAMES.md`'s exact convention), the camera's position (from `body_T_camera_left.translation`), the real `ObstacleCloud` (SURFACE/OCCUPIED, solid markers), and a plot-time-sampled subset of the real `FreeSpaceRays` (FREE, line segments) — sampling is applied only in the `matplotlib` call, never to the engine's own output arrays. UNKNOWN space is drawn as nothing at all, with an explicit on-figure caption stating that blank regions are unknown, never free or occupied. **No axis flip anywhere**: `mplot3d` imposes no inherent "up" convention, so BODY X/Y/Z map directly onto plot X/Y/Z with zero data conversion — verified by inspection of the plotting code, not assumed; the only aesthetic choice made is the camera *view angle* (`ax.view_init`), which changes how the same unmodified data is looked at, not the data itself.

### Generated artifacts (Task 14)

Placed in the repository's existing `docs/assets/` structure, continuing its established numbering:
- `docs/assets/07_level3_baseline.png` — baseline real scene, DEGRADED, SLOW_DOWN-adjacent state
- `docs/assets/08_level3_degraded_occlusion.png` — real partial occlusion, navigation decision `SLOW_DOWN`
- `docs/assets/09_level3_healthy_scene.png` / `docs/assets/10_level3_live_demo.gif` — re-captured (post-freeze) desk/monitor scene, `DEGRADED`, navigation decision `STOP` (a real nearby obstacle at ~0.32m drives this, not a defect) — regenerate anytime with `python examples/capture_readme_snapshot.py` / `python examples/generate_demo_gif.py`, both with a 6-second on-screen countdown to reposition the scene first

Reproduce: `pip install -e ".[viz]"` then `python examples/visualize_level3.py --live` (or `--left/--right` with saved `.npy` frames). No large raw recordings were committed — only the three rendered PNGs above.

### BODY-frame orientation proof (Task 5/6)

Mathematical: Task 2 Scenario 4, `1e-4 m` exact match against a hand-computed 90° rotation + translation. Visual: the three artifacts above show the axes triad, camera marker, and point cloud in a single consistent frame, with the point cloud always extending in the `+X` (forward) direction from the camera marker — matching the physical fact that a forward-facing camera observes what's in front of it, visible directly in the rendered images.

### Obstacle/free-space validation (Task 6/7)

`ObstacleCloud` point count and `FreeSpaceRays` count were exactly equal to the real, current frame's valid-pixel count in every one of the 168 real captured frames (see table above) — obstacle evidence tracks observed surfaces by construction (unchanged since E5), not approximately. Free-space rays terminate exactly at their source point (`endpoint = origin + direction·range`, verified to float32 precision in Task 2 and E5/E6's own tests) — never behind a measured surface, never extended past it.

### Degradation / recovery results (Task 8)

Agrees with E6's automated findings (Task 8's explicit requirement): `occlusion`/`textureless` show reduced valid-geometry fraction and predominantly `DEGRADED` classification on real data, matching E6's synthetic `NO_USABLE_GEOMETRY`-under-degradation findings in direction (real degradation here didn't reach `NO_USABLE_GEOMETRY` because the degraded regions never covered the *entire* frame — a real, partial, human-executed occlusion/low-texture attempt, not the synthetic all-invalid case E6 tested exactly). The pipeline remained fully responsive throughout (no stall, no error, stable ~33-35 ms latency across every phase including the degraded ones) and recovered normally in the immediately following phase, consistent with E6's `tests/test_state_recovery.py` proof that no stale geometry persists between frames — reconfirmed here on real data, not just re-asserted.

### Performance table (Task 9) — real hardware, 320×240, the actual rig resolution

`examples/e7_live_long_run.py`, 15-frame warm-up (discarded), 300 timed real frames, continuous:

| Metric | Value |
|---|---|
| Frames requested / processed | 300 / 300 |
| Errors | 0 |
| Mean latency | 23.84 ms |
| Median latency | 23.60 ms |
| Std | 2.12 ms |
| p95 | 27.71 ms |
| Max | 40.35 ms |
| **Effective FPS (measured, mean-based)** | **42.0** |

No specific real-time rate requirement has been defined for this library by any consumer to date (`mp01_perception`'s actual required rate is out of this repository's scope) — 42.0 FPS is reported as a measured fact on this hardware/software combination, not asserted as sufficient for any particular downstream use.

### Long-run stability (Task 10) — real hardware

Same 300-frame run: RSS sampled every 30 frames, range 63,316-77,780 KiB, no monotonic trend across the 11 checkpoints. Zero degraded episodes (valid_fraction never dropped below the 0.05 no-usable-geometry floor during this particular run — a normal indoor scene throughout). **Stated precisely**: no monotonic memory growth observed over 300 continuous real frames on this platform — not a formal proof of leak-freedom, matching E6's own wording discipline, now with real rather than synthetic frames.

### Full regression test count (Task 11)

`pytest tests/ -q`: **422 passed** (416 before E7, +6 new — `tests/test_e7_synthetic_ground_truth.py`). Zero existing assertion weakened. Level 0-2, E2 math, E3 camera geometry, E4 body transform, E5 obstacle/free-space semantics, and E6 robustness tests all re-ran unmodified and clean.

### Public API / standalone verification (Task 12)

Direct repository-wide grep (not just the test suite) confirmed zero `rclpy`/`sensor_msgs`/`cv_bridge`/Gazebo/`mp01_ws`/`mp01_perception` **imports or dependencies** anywhere in `src/depth_perception_engine/` — the handful of textual matches are documentation/docstring mentions of `mp01_perception` by name (explaining design decisions for that downstream consumer), individually inspected and confirmed to be prose, not code. No camera-driver dependency in the engine core (camera I/O lives only in `examples/*.py`, confirmed by the same grep). `matplotlib` (the one new dependency, for the Task 4 visualization tool) is declared only in the new optional `viz` extra in `pyproject.toml` and imported only by `examples/visualize_level3.py` — `pip install -e .` with zero extras remains sufficient for the full core library and its entire test suite. Canonical top-level API (`__all__`, 19 symbols) unchanged since E1's freeze.

### Known limitations

- The `min_obstacle_distance_m` global-scalar motion-tracking demonstration was inconclusive in this specific real-hardware session (documented above) — not re-attempted a third time.
- Real hardware validation covered one specific rig/room/lighting condition, not a systematic sweep of environments.
- Performance numbers reflect this specific USB stereo rig and host machine — not a general real-time capability claim for other hardware.
- `examples/e7_live_capture_session.py` saves only the first raw frame of each phase (not a full per-phase recording) — sufficient for the per-frame CSV telemetry (which covers the whole window) but not for frame-by-frame visual review of a phase's evolution.
- No systematic sweep of lighting/distance/object-type variations — this was one session, on one day, with the objects available.

### Final Level-3 acceptance matrix (Task 15)

| Capability | Result |
|---|---|
| Stereo rectification | PASS |
| Disparity | PASS |
| Metric depth | PASS |
| Confidence/validity | PASS |
| Camera-frame PointCloud | PASS |
| BODY-frame PointCloud | PASS |
| ObstacleCloud | PASS |
| FreeSpaceRays | PASS |
| UNKNOWN preservation | PASS (synthetic + 168/168 real frames, zero exceptions) |
| GeometryMetrics | PASS |
| Degradation handling | PASS |
| Recovery | PASS |
| Determinism | PASS |
| Performance | PASS (measured: 42.0 FPS mean, 320×240, real hardware; no rate claimed beyond what was measured) |
| Long-run stability | PASS (no monotonic growth observed, 300 real frames — not a formal leak-freedom proof) |
| Public API freeze | PASS |
| ROS independence | PASS |
| Real stereo visual validation | PASS, with one documented partial limitation (global-scalar motion-tracking inconclusive in this session; per-pixel/per-frame correctness independently proven) |

No row is FAIL.

### Final decision (E7)

**LEVEL 3 COMPLETE — READY TO FREEZE.** Every acceptance-matrix row passes. The chain was validated as one coherent system — synthetically (exact, hand-computed) and on real calibrated hardware (168 real frames, the core UNKNOWN-space safety invariant re-verified with zero exceptions) — and a standalone visual proof exists showing correct BODY-frame orientation, obstacle evidence, and free-space rays on real data. One narrow, specific limitation (a single global-scalar metric's sensitivity to a moved object, in this specific test environment) is documented rather than hidden, and does not implicate the chain's proven per-pixel/per-frame correctness. No algorithm changed, the entire historical suite passes, and the public API remains exactly as frozen at E1.

**This freeze is not a flight-safety, collision-avoidance, or production-certification claim of any kind.** Level 3 means: metric depth with confidence/validity, camera- and body-frame 3D geometry, obstacle and free-space evidence with explicit UNKNOWN preservation, and geometry-quality metrics — single-frame, stateless (except Level 0-2's pre-existing `ThreatAssessor` debounce), calibration-driven, and ROS-free. It does **not** mean a persistent map, temporal fusion, IMU compensation, a vehicle collision envelope, collision-risk scoring, learned stereo, semantics, localization, or planning — all remain unbuilt, see `docs/E7_IMPLEMENTATION_PLAN.md`.

**This pass makes no claim of flight safety, collision avoidance, or production certification of any kind — E6 characterizes and hardens single-frame geometric perception under a fixed, single-machine, synthetic-input test regime; it is not a safety case.** Occupancy mapping, temporal fusion, vehicle envelopes, and collision-risk scoring remain entirely unbuilt — see `docs/E7_IMPLEMENTATION_PLAN.md`.

## Level 4 addendum (2026-08-10) — real-camera live capture of the full temporal chain

Real, calibrated global-shutter stereo rig (`/dev/video0`, the same "3D Global Shutter Camera" device used by the Level 3 E7 real-hardware session above), driven continuously through `DepthPerceptionPipeline` with **every** Level 4 capability enabled (temporal history, consistency, stabilization, rotation compensation, motion-aware reliability, persistence) via `examples/generate_level4_live_gif.py` (which reuses `examples/visualize_level4_live.py`'s own camera-opening and dashboard-rendering code directly, not a second implementation). 24 real frames at ~3 Hz, zero read errors, zero pipeline errors, no threshold tuned for this capture (every `PipelineConfig` flag left at its frozen default).

**This is real stereo geometry. The MotionHint is not.** IMU remains simulated throughout this repository (`docs/LEVEL4_SIMULATED_IMU.md`) — frames 13-24 of this capture had a directly-constructed, synthetic `temporal.MotionHint` attached (a small fixed yaw rate), and the on-screen dashboard displays an unmistakable red "SIMULATED MOTIONHINT: ON — (NOT real IMU data)" badge for every one of those frames, so the two are never conflated in the artifact itself.

| Metric | Value |
|---|---|
| Frames captured | 24 (real reads, 0 dropped) |
| Mean `processing_time_ms` (full 7-stage chain) | 28.5 ms (min 23.7, max 38.5) |
| Mean FPS (real capture, including display build) | 35.3 |
| Frames reading `motion_aware_reliability.state == UNRELIABLE` | 8 / 24 — real-world lighting/hand-held-camera jitter, not injected |
| `temporal_persistence.state` on every one of those 8 frames | `UNRELIABLE` — `persistent_count`/`support_count_grid` provably unchanged from the immediately preceding `CLASSIFIED` frame in every case (matches `tests/test_level4_integration_e8.py::TestUnreliableEvidenceCannotReinforcePersistence`'s synthetic proof, now also observed on real sensor noise) |

**Rule 7 ("an UNRELIABLE frame can neither create nor reinforce persistence") holds on real data, not just synthetic fixtures**: e.g. frames 5-8 all read `persistent=1515, disappearing=332` — frozen, byte-for-byte identical across four consecutive real `UNRELIABLE` frames — then resumed normal updates the instant reliability recovered (frame 9). This was not constructed to demonstrate the rule; it is what the real camera's own natural frame-to-frame noise happened to trigger, observed after the fact.

**Persistence built up correctly on a real, mostly-static scene**: `new_count` drops from 1733 (frame 1, first-ever observation) toward a few hundred within 3-4 frames as `persistent_count` climbs past 1500 of ~1730-1800 eligible cells — real repeated agreement, not a synthetic construction.

**Honest limitation, not glossed over**: this capture does not exercise real IMU data, genuine controlled camera rotation, or a measured extrinsic — `body_T_camera_left` here is still `examples/visualize_level3.py`'s own illustrative transform, and the attached `MotionHint` is a synthetic, hand-picked value, clearly badged as such. This capture closes only the "real stereo" row of `docs/LEVEL4_HARDWARE_VALIDATION_PENDING.md`'s checklist — real IMU timestamps/angular velocity, genuine rotation, measured extrinsics, and Jetson performance all remain explicitly pending, unchanged by this addendum.

**Generated artifact**: `docs/assets/11_level4_live_demo.gif` — reproduce with `python examples/generate_level4_live_gif.py` (5-second on-screen countdown to reposition the scene first, matching `generate_demo_gif.py`'s own convention).

## Final decision

**RESTORED WITH DOCUMENTED LIMITATIONS.**

The repository was already substantially healthy going into this pass (genuinely ROS-independent, one canonical execution path, real — not stubbed — algorithm code, mostly-behavioral test coverage, accurate pre-existing docs). This pass closed the real gaps found by the audit (missing lifecycle API surface, missing timestamp/validity-mask contracts, one untested public class, one OpenCV-only-verified math path, two dead files, five missing docs) without any breaking change to the existing public API — every addition is opt-in, and `mp01_perception`'s existing `.process(left, right)` call keeps working unmodified, confirmed by a zero-regression `colcon test` run. Not marked fully "RESTORED AND FROZEN" because two real, known limitations remain by deliberate choice (the `traversability_mask` naming wart and `mp01_perception`'s internal-module import) — both require touching `mp01_perception`, which is explicitly out of this task's scope, and are documented rather than hidden.
