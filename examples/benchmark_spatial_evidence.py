"""
ObstacleCloud / FreeSpaceRays / GeometryMetrics latency/memory benchmark —
Level 3, Phase E5.

No camera, no pipeline, no hardware required — a synthetic organized
body-frame PointCloud at 640x480 (same resolution examples/benchmark_point_cloud.py
(E2), examples/benchmark_pipeline_geometry.py (E3), and
examples/benchmark_body_transform.py (E4) used), each E5 stage measured in
isolation.

Run:
    pip install -e ..
    python examples/benchmark_spatial_evidence.py
"""

import time
import tracemalloc

import numpy as np

from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry import (
    PointCloud,
    build_free_space_rays,
    build_geometry_metrics,
    build_obstacle_cloud,
)

WIDTH, HEIGHT = 640, 480
N_WARMUP = 15
N_ITERS = 200
ORIGIN = np.array([0.08, 0.0, 0.05])  # illustrative — matches the other E4/E5 benchmarks' fixture


def _synthetic_body_cloud(width: int, height: int, seed: int) -> PointCloud:
    """Same construction as examples/benchmark_body_transform.py's fixture
    — a plausible organized cloud with a realistic mix of valid/invalid
    (NaN) pixels."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-5.0, 5.0, size=(height, width, 3)).astype(np.float32)
    points[:, :, 2] = np.abs(points[:, :, 2]) + 0.15
    invalid = rng.random((height, width)) < 0.15
    points[invalid] = np.nan
    valid_mask = ~invalid
    return PointCloud(points=points, frame_id=FrameId.BODY, valid_mask=valid_mask)


def _time_calls(fn, n_iters):
    samples = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1000.0
    return samples


def _report(label, samples):
    print(f"{label}: mean={samples.mean():.3f} ms  std={samples.std():.3f} ms  "
          f"max={samples.max():.3f} ms  p95={np.percentile(samples, 95):.3f} ms")


def main() -> None:
    cloud = _synthetic_body_cloud(WIDTH, HEIGHT, seed=0)
    print(f"Benchmark resolution: {WIDTH}x{HEIGHT} (matches E2/E3/E4 benchmarks)")
    print(f"Valid points in fixture: {int(cloud.valid_mask.sum())} / {cloud.valid_mask.size}\n")

    obstacle_cloud = build_obstacle_cloud(cloud, ORIGIN, min_range_m=0.0, max_range_m=8.0)
    free_space_rays = build_free_space_rays(cloud, ORIGIN)
    print(f"Obstacle points: {obstacle_cloud.points.shape[0]}")
    print(f"Free-space rays: {free_space_rays.ranges_m.shape[0]}\n")

    for _ in range(N_WARMUP):
        build_obstacle_cloud(cloud, ORIGIN, min_range_m=0.0, max_range_m=8.0)
        build_free_space_rays(cloud, ORIGIN)
        build_geometry_metrics(cloud, obstacle_cloud, free_space_rays)

    obstacle_samples = _time_calls(
        lambda: build_obstacle_cloud(cloud, ORIGIN, min_range_m=0.0, max_range_m=8.0), N_ITERS,
    )
    _report("Obstacle-cloud filtering", obstacle_samples)

    rays_samples = _time_calls(lambda: build_free_space_rays(cloud, ORIGIN), N_ITERS)
    _report("Free-space ray generation", rays_samples)

    metrics_samples = _time_calls(
        lambda: build_geometry_metrics(cloud, obstacle_cloud, free_space_rays), N_ITERS,
    )
    _report("Geometry metrics aggregation", metrics_samples)

    total_samples = obstacle_samples + rays_samples + metrics_samples
    print(f"\nTotal E5 added latency (sum of the three stages above): "
          f"mean={total_samples.mean():.3f} ms  std={total_samples.std():.3f} ms  max={total_samples.max():.3f} ms")

    # -- Memory: exact output sizes + bounded-growth check ------------------
    obstacle_bytes = obstacle_cloud.points.nbytes + obstacle_cloud.distances_m.nbytes
    rays_bytes = free_space_rays.origins.nbytes + free_space_rays.directions.nbytes + free_space_rays.ranges_m.nbytes
    print(f"\nObstacleCloud output: {obstacle_bytes / 1024:.1f} KiB "
          f"({obstacle_cloud.points.shape[0]} points)")
    print(f"FreeSpaceRays output: {rays_bytes / 1024:.1f} KiB "
          f"({free_space_rays.ranges_m.shape[0]} rays)")

    tracemalloc.start()
    peaks_kib = []
    for _ in range(5):
        for _ in range(20):
            oc = build_obstacle_cloud(cloud, ORIGIN, min_range_m=0.0, max_range_m=8.0)
            rays = build_free_space_rays(cloud, ORIGIN)
            metrics = build_geometry_metrics(cloud, oc, rays)
        del oc, rays, metrics
        _current, peak = tracemalloc.get_traced_memory()
        peaks_kib.append(peak / 1024)
    tracemalloc.stop()
    print(f"\nPeak traced memory after each successive batch of 20 full-E5 calls (KiB): "
          + ", ".join(f"{p:.1f}" for p in peaks_kib))
    growth = peaks_kib[-1] - peaks_kib[1]
    print(f"Growth from batch 2 to batch 5: {growth:.1f} KiB "
          f"({'bounded — no leak' if abs(growth) < obstacle_bytes / 1024 else 'UNBOUNDED — investigate'})")


if __name__ == "__main__":
    main()
