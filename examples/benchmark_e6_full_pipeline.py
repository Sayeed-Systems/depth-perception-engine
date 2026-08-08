"""
Full Level 0-3 per-stage performance characterization — Level 3, Phase E6
(Task 9).

Two resolutions: the repository's own real hardware calibration (320x240,
"current development resolution", rectify=True) and a synthetic 640x480
calibration (rectify=False — the same synthetic-calibration technique
examples/benchmark_pipeline_geometry.py (E3) already used, chosen because
640x480 is already this repository's established "higher realistic
resolution" benchmark precedent from E2-E5, not a new, unvalidated
assumption).

Two measurements per resolution:
  1. TOTAL pipeline latency — the real, public DepthPerceptionPipeline.process(),
     unmodified, called end to end. This is the number that matters.
  2. PER-STAGE breakdown — fresh, standalone engine instances built
     identically to how DepthPerceptionPipeline.__init__ builds them
     (from the same PipelineConfig/calibration/extrinsic), chained
     together manually so each stage's real output feeds the next. This
     never reaches into DepthPerceptionPipeline's private attributes and
     never modifies pipeline.py — purely additive, read-only
     instrumentation, consistent with every earlier benchmark script in
     this repository.

Warm-up policy (explicit, not hidden, per Task 9): N_WARMUP iterations of
the exact same call are run and discarded before any timed sample is
taken, for both the total-pipeline and the per-stage measurements
separately (per-stage warmup exercises each stage's own first-call cost —
e.g. cv2's internal thread-pool/buffer setup — independently).

Run:
    pip install -e ..
    python examples/benchmark_e6_full_pipeline.py
"""

import time

import cv2
import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import (
    PointCloudBuilder,
    build_free_space_rays,
    build_geometry_metrics,
    build_obstacle_cloud,
    transform_point_cloud,
)
from depth_perception_engine.stereo.disparity_engine import DisparityEngine

N_WARMUP = 15
N_ITERS = 100


def _synthetic_calibration(width: int, height: int) -> StereoCalibration:
    """Identical construction to examples/benchmark_pipeline_geometry.py's
    fixture — see that file for the full rationale."""
    fx = fy = 600.0
    cx, cy = width / 2.0, height / 2.0
    baseline_m = 0.065
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((1, 5), dtype=np.float64)
    tx = 1.0 / (baseline_m * 1000.0)
    Q = np.array(
        [[1.0, 0.0, 0.0, -cx], [0.0, 1.0, 0.0, -cy], [0.0, 0.0, 0.0, fx], [0.0, 0.0, tx, 0.0]],
        dtype=np.float64,
    )
    return StereoCalibration(
        image_size=(width, height),
        camera_matrix_left=camera_matrix, dist_coeffs_left=dist_coeffs,
        camera_matrix_right=camera_matrix, dist_coeffs_right=dist_coeffs,
        R1=np.eye(3), R2=np.eye(3),
        P1=np.hstack([camera_matrix, np.zeros((3, 1))]), P2=np.hstack([camera_matrix, np.zeros((3, 1))]),
        Q=Q,
    )


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _stats(samples_ms):
    samples_ms = np.asarray(samples_ms)
    return {
        "mean": samples_ms.mean(), "median": np.median(samples_ms), "std": samples_ms.std(),
        "p95": np.percentile(samples_ms, 95), "max": samples_ms.max(),
    }


def _print_stats(label, samples_ms, n_frames):
    s = _stats(samples_ms)
    fps = 1000.0 / s["mean"] if s["mean"] > 0 else float("inf")
    print(f"  {label:<28s} mean={s['mean']:7.3f}ms  median={s['median']:7.3f}ms  "
          f"std={s['std']:6.3f}ms  p95={s['p95']:7.3f}ms  max={s['max']:7.3f}ms  "
          f"FPS={fps:6.1f}  (n={n_frames})")


def _run_resolution(label, calibration, rectify, config):
    print(f"\n=== {label} ({calibration.image_size[0]}x{calibration.image_size[1]}, rectify={rectify}) ===")
    w, h = calibration.image_size
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

    # -- 1. Total pipeline latency (real, public process()) --------------
    pipeline = DepthPerceptionPipeline(config, calibration, rectify=rectify, body_T_camera_left=_transform())
    for _ in range(N_WARMUP):
        pipeline.process(left, right)
    total_samples = np.empty(N_ITERS, dtype=np.float64)
    for i in range(N_ITERS):
        t0 = time.perf_counter()
        pipeline.process(left, right)
        total_samples[i] = (time.perf_counter() - t0) * 1000.0
    print(f"Warm-up: {N_WARMUP} discarded iterations before every timed measurement below.")
    _print_stats("TOTAL pipeline (process())", total_samples, N_ITERS)

    # -- 2. Per-stage breakdown, standalone engines, chained -------------
    rectifier = None
    if rectify:
        from depth_perception_engine.stereo.rectification import RectificationEngine
        rectifier = RectificationEngine(calibration)
        rectifier.initialize_rectification()
    disparity_engine = DisparityEngine(
        min_disparity=config.min_disparity, num_disparities=config.num_disparities, block_size=config.block_size,
    )
    depth_estimator = DepthEstimator.from_calibration(calibration)
    point_cloud_builder = PointCloudBuilder.from_calibration(calibration)
    transform = _transform()

    stage_samples = {
        name: np.empty(N_ITERS, dtype=np.float64)
        for name in (
            "rectify" if rectify else None,
            "disparity", "depth", "camera_cloud", "body_transform",
            "obstacle_cloud", "free_space_rays", "geometry_metrics",
        ) if name is not None
    }

    def _one_pass(record: bool, i: int = 0):
        l, r = left, right
        if rectifier is not None:
            t0 = time.perf_counter()
            l, r = rectifier.rectify(l, r)
            if record:
                stage_samples["rectify"][i] = (time.perf_counter() - t0) * 1000.0
        gray = l if l.ndim == 2 else cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()
        raw_disparity, _ = disparity_engine.compute_disparity(l, r, left_gray=gray, compute_visualization=False)
        if record:
            stage_samples["disparity"][i] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        depth_estimator.estimate(raw_disparity)
        if record:
            stage_samples["depth"][i] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        cloud = point_cloud_builder.build(raw_disparity)
        if record:
            stage_samples["camera_cloud"][i] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        body_cloud = transform_point_cloud(cloud, transform)
        if record:
            stage_samples["body_transform"][i] = (time.perf_counter() - t0) * 1000.0

        origin = transform.translation
        t0 = time.perf_counter()
        obstacle_cloud = build_obstacle_cloud(
            body_cloud, origin, min_range_m=config.obstacle_min_range_m, max_range_m=config.obstacle_max_range_m,
        )
        if record:
            stage_samples["obstacle_cloud"][i] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        rays = build_free_space_rays(body_cloud, origin)
        if record:
            stage_samples["free_space_rays"][i] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        build_geometry_metrics(body_cloud, obstacle_cloud, rays)
        if record:
            stage_samples["geometry_metrics"][i] = (time.perf_counter() - t0) * 1000.0

    for _ in range(N_WARMUP):
        _one_pass(record=False)
    for i in range(N_ITERS):
        _one_pass(record=True, i=i)

    print("Per-stage breakdown (standalone engines, chained, same warm-up policy):")
    for name, samples in stage_samples.items():
        _print_stats(name, samples, N_ITERS)
    stage_sum = sum(s.mean() for s in stage_samples.values())
    print(f"  {'sum of stage means':<28s} = {stage_sum:.3f} ms  "
          f"(vs. TOTAL process() mean {total_samples.mean():.3f} ms — "
          f"difference is Level 0-2 traversability/obstacle-scan cost, not measured stage-by-stage here, "
          f"plus call overhead)")


def main() -> None:
    full_config = PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)

    real_calibration = load_stereo_calibration("examples/config/stereo_calibration.xml")
    _run_resolution("Current development resolution (real hardware calibration)", real_calibration, True, full_config)

    synthetic_calibration = _synthetic_calibration(640, 480)
    _run_resolution("Higher realistic resolution (synthetic calibration)", synthetic_calibration, False, full_config)


if __name__ == "__main__":
    main()
