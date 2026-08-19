# I0 — DPE Improvement Benchmark Freeze

This directory is the frozen V1.0.1 benchmark baseline for all future
DPE improvement work (Phase I0). It does not modify any perception
algorithm in `src/depth_perception_engine/` — everything here is
read-only instrumentation around the real, public DPE API, following
the same discipline already established by
`tests/test_d10_integrated_ground_truth.py` and
`examples/benchmark_d14_provider_validation.py`.

## Why this directory exists instead of the originally-named files

The task that created this freeze named specific source artifacts
(`DPE_MP01_SIM_VALIDATION_REPORT.pdf`, `DPE_MP01_SIM_MEASUREMENTS.md`,
`measurements.csv`, a Gazebo world) that do not exist anywhere in this
repository or its git history — this codebase has zero Gazebo/ROS
dependency (see `docs/VALIDATION_REPORT.md`). "MP01" in this repo's own
history refers to a downstream ROS2 consumer package (`mp01_perception`),
not a simulation-validation report.

Per the repo owner's direction, this freeze was built instead from what
actually exists: this repo's own checked-in calibration config
(`examples/config/stereo_calibration.xml`) and `PipelineConfig`'s
documented defaults as the sole ground-truth source (this repo's
equivalent of "SDF/URDF/config geometry" — task requirement 4), reusing
the exact scenario geometry already established and reviewed in
`tests/test_d10_integrated_ground_truth.py` (D10), `tests/test_clearance_geometry.py`
(D7), and `examples/benchmark_d14_provider_validation.py` (D14).

## Structure (scenario definition separated from measurement collection)

- `calibration_geometry.py` — derives fx/cx/cy/baseline_m from the
  calibration file's own Q matrix (one shared, read-only derivation).
- `scenarios.py` — **scenario definitions only**. Builds synthetic
  disparity/depth arrays and their analytically-known ground truth.
  Never calls DPE geometry/pipeline code.
- `measure.py` — **measurement collection only**. Runs each scenario
  through DPE's real, unmodified public functions/pipeline and records
  what came out. Asserts nothing; introduces no acceptance threshold.
  `collect_all()` returns the 8 required categories: depth errors by
  target, valid fractions, boundary findings, opening result,
  surface-normal error/planarity, clearance error, degradation state,
  latency/FPS.
- `record_baseline.py` — freezes one `collect_all()` run, plus DPE
  version/git commit/environment metadata, to `baseline_v1.0.1.json`.
  Refuses to overwrite an existing baseline unless `--force` is passed.
- `compare_to_baseline.py` — re-runs `collect_all()` as a "candidate"
  and diffs every leaf metric against the frozen baseline. Reports
  deltas only — no pass/fail gate.
- `baseline_v1.0.1.json` — the frozen artifact itself (DPE v1.0.1).

## Determinism

Every scenario uses either a closed-form analytic construction (no
randomness) or a fixed RNG seed (`numpy.random.default_rng(seed)`,
matching this repo's own existing convention). Re-running
`compare_to_baseline.py` against `baseline_v1.0.1.json` reproduces all
69 non-latency leaf metrics with **exactly zero delta** — verified
during this freeze. Only `latency_fps.*` (wall-clock timing, inherently
run-to-run noisy) varies between runs; this is expected and is not a
correctness signal.

## Reproducible commands

```bash
# Re-freeze from scratch (only if a genuine re-freeze is intended):
python -m benchmarks.i0_baseline.record_baseline --force

# Compare the current working tree against the frozen v1.0.1 baseline:
python -m benchmarks.i0_baseline.compare_to_baseline
python -m benchmarks.i0_baseline.compare_to_baseline --out /tmp/i0_diff.json
```
