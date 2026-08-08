"""
DepthPerceptionPipeline.process() latency/memory benchmark — Level 3,
Phase E3: cost added by the geometry stage.

No camera required — synthetic stereo pair + a synthetic 640x480
StereoCalibration (this repo's own hardware fixture is 320x240; a
synthetic calibration lets this benchmark run at the exact resolution
examples/benchmark_point_cloud.py (E2) used, for a direct comparison).
rectify=False on both pipelines, so RectificationEngine (which needs a
real calibration's rectification maps) is never invoked — SGBM runs
directly on the synthetic "already rectified" pair; this has no bearing
on the geometry stage's own cost, which only depends on raw_disparity's
shape and the Q matrix.

Run:
    pip install -e ..
    python examples/benchmark_pipeline_geometry.py
"""

import time
import tracemalloc

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig
from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.geometry import PointCloudBuilder

WIDTH, HEIGHT = 640, 480
N_WARMUP = 15
N_ITERS = 100


def _synthetic_calibration(width: int, height: int) -> StereoCalibration:
    """Shape-valid, numerically plausible synthetic calibration at
    (width, height) — not tied to any real rig. R1/R2/P1/P2/dist_coeffs
    are never exercised (both benchmarked pipelines use rectify=False),
    so only image_size/camera_matrix_*/Q need to be realistic; Q follows
    the same construction tests/test_depth_estimator.py's TestEstimatePointCloud
    and TestAnalyticKnownDepth already use and independently verified."""
    fx = fy = 600.0
    cx, cy = width / 2.0, height / 2.0
    baseline_m = 0.065
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((1, 5), dtype=np.float64)
    tx = 1.0 / (baseline_m * 1000.0)
    Q = np.array(
        [
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, fx],
            [0.0, 0.0, tx, 0.0],
        ],
        dtype=np.float64,
    )
    return StereoCalibration(
        image_size=(width, height),
        camera_matrix_left=camera_matrix,
        dist_coeffs_left=dist_coeffs,
        camera_matrix_right=camera_matrix,
        dist_coeffs_right=dist_coeffs,
        R1=np.eye(3),
        R2=np.eye(3),
        P1=np.hstack([camera_matrix, np.zeros((3, 1))]),
        P2=np.hstack([camera_matrix, np.zeros((3, 1))]),
        Q=Q,
    )


def _latency_stats_ms(pipeline, left, right, n_iters):
    for _ in range(N_WARMUP):
        pipeline.process(left, right)
    samples = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        t0 = time.perf_counter()
        pipeline.process(left, right)
        samples[i] = (time.perf_counter() - t0) * 1000.0
    return samples


def main() -> None:
    calibration = _synthetic_calibration(WIDTH, HEIGHT)
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)

    config_off = PipelineConfig(enable_geometry=False)
    config_on = PipelineConfig(enable_geometry=True)
    pipeline_off = DepthPerceptionPipeline(config_off, calibration, rectify=False)
    pipeline_on = DepthPerceptionPipeline(config_on, calibration, rectify=False)

    print(f"Benchmark resolution: {WIDTH}x{HEIGHT} (matches examples/benchmark_point_cloud.py's E2 benchmark)\n")

    samples_off = _latency_stats_ms(pipeline_off, left, right, N_ITERS)
    samples_on = _latency_stats_ms(pipeline_on, left, right, N_ITERS)

    print(f"Level 0-2 baseline (enable_geometry=False), {N_ITERS} iterations:")
    print(f"  mean: {samples_off.mean():.3f} ms   std: {samples_off.std():.3f} ms   "
          f"p95: {np.percentile(samples_off, 95):.3f} ms   max: {samples_off.max():.3f} ms")

    print(f"\nLevel 0-3 (enable_geometry=True), {N_ITERS} iterations:")
    print(f"  mean: {samples_on.mean():.3f} ms   std: {samples_on.std():.3f} ms   "
          f"p95: {np.percentile(samples_on, 95):.3f} ms   max: {samples_on.max():.3f} ms")

    abs_increase = samples_on.mean() - samples_off.mean()
    pct_increase = 100.0 * abs_increase / samples_off.mean()
    print(f"\nAbsolute latency increase (mean): {abs_increase:.3f} ms")
    print(f"Percentage latency increase (mean): {pct_increase:.1f}%")

    # Geometry-stage-only latency, isolated: call PointCloudBuilder.build()
    # directly on the exact raw_disparity this pipeline itself computed —
    # cross-check against the (Level 0-3 - Level 0-2) difference above,
    # and against examples/benchmark_point_cloud.py's own E2-only number.
    result_for_disparity = pipeline_off.process(left, right)
    builder = PointCloudBuilder.from_calibration(calibration)
    for _ in range(N_WARMUP):
        builder.build(result_for_disparity.disparity_map)
    geom_samples = np.empty(N_ITERS, dtype=np.float64)
    for i in range(N_ITERS):
        t0 = time.perf_counter()
        builder.build(result_for_disparity.disparity_map)
        geom_samples[i] = (time.perf_counter() - t0) * 1000.0

    print(f"\nGeometry stage only (PointCloudBuilder.build(), isolated), {N_ITERS} iterations:")
    print(f"  mean: {geom_samples.mean():.3f} ms   std: {geom_samples.std():.3f} ms   "
          f"max: {geom_samples.max():.3f} ms")
    print(f"  (cross-check: isolated geometry mean {geom_samples.mean():.3f} ms vs. "
          f"pipeline-level difference {abs_increase:.3f} ms)")

    # -- Memory: peak traced memory across repeated calls, both configs --
    for label, pipeline in (("Level 0-2 (off)", pipeline_off), ("Level 0-3 (on)", pipeline_on)):
        tracemalloc.start()
        for _ in range(20):
            result = pipeline.process(left, right)
        del result
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"\n{label}: peak traced memory over 20 calls: {peak / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
