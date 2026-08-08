"""
transform_point_cloud() latency/memory benchmark — Level 3, Phase E4.

No camera, no pipeline, no hardware required — a synthetic organized
PointCloud at 640x480 (same resolution examples/benchmark_point_cloud.py
(E2) and examples/benchmark_pipeline_geometry.py (E3) used), transformed
in isolation. This measures the transform stage alone, not SGBM/rectify/
depth — see those other two scripts for the rest of the pipeline's cost.

Run:
    pip install -e ..
    python examples/benchmark_body_transform.py
"""

import time
import tracemalloc

import numpy as np

from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import PointCloud, transform_point_cloud

WIDTH, HEIGHT = 640, 480
N_WARMUP = 15
N_ITERS = 200


def _synthetic_camera_cloud(width: int, height: int, seed: int) -> PointCloud:
    """A plausible organized point cloud with a realistic mix of valid/
    invalid (NaN) pixels — mirrors examples/benchmark_point_cloud.py's
    disparity-based synthetic fixture, but built directly as a PointCloud
    since this benchmark targets the transform stage only, not
    reprojection (already benchmarked separately in E2/E3)."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-5.0, 5.0, size=(height, width, 3)).astype(np.float32)
    points[:, :, 2] = np.abs(points[:, :, 2]) + 0.15  # plausible positive depth
    invalid = rng.random((height, width)) < 0.15
    points[invalid] = np.nan
    valid_mask = ~invalid
    return PointCloud(points=points, frame_id=FrameId.CAMERA_OPTICAL_LEFT, valid_mask=valid_mask)


def _illustrative_transform() -> RigidTransform:
    """Synthetic, illustrative only — see docs/COORDINATE_FRAMES.md's E4
    section. Not tied to any real rig's measured extrinsic."""
    angle = np.deg2rad(15.0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    translation = np.array([0.08, 0.0, 0.05])
    return RigidTransform(
        rotation=rotation, translation=translation,
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def main() -> None:
    cloud = _synthetic_camera_cloud(WIDTH, HEIGHT, seed=0)
    transform = _illustrative_transform()
    print(f"Benchmark resolution: {WIDTH}x{HEIGHT} (matches E2/E3 benchmarks)\n")

    for _ in range(N_WARMUP):
        transform_point_cloud(cloud, transform)

    latencies_ms = np.empty(N_ITERS, dtype=np.float64)
    for i in range(N_ITERS):
        t0 = time.perf_counter()
        transform_point_cloud(cloud, transform)
        latencies_ms[i] = (time.perf_counter() - t0) * 1000.0

    print(f"Body-frame transform latency over {N_ITERS} iterations (post-warmup):")
    print(f"  mean: {latencies_ms.mean():.3f} ms")
    print(f"  std:  {latencies_ms.std():.3f} ms")
    print(f"  min:  {latencies_ms.min():.3f} ms")
    print(f"  max:  {latencies_ms.max():.3f} ms")
    print(f"  p95:  {np.percentile(latencies_ms, 95):.3f} ms")

    # -- Memory: exact output size + bounded-growth check ------------------
    out = transform_point_cloud(cloud, transform)
    points_bytes = out.points.nbytes
    mask_bytes = out.valid_mask.nbytes
    print(f"\nOutput PointCloud.points: {points_bytes / 1024:.1f} KiB (exact, H*W*3*4 bytes)")
    print(f"Output PointCloud.valid_mask: {mask_bytes / 1024:.1f} KiB (exact, H*W*1 byte)")

    del out
    tracemalloc.start()
    peaks_kib = []
    for _ in range(5):
        for _ in range(20):
            out = transform_point_cloud(cloud, transform)
        del out
        _current, peak = tracemalloc.get_traced_memory()
        peaks_kib.append(peak / 1024)
    tracemalloc.stop()
    print(f"\nPeak traced memory after each successive batch of 20 calls (KiB): "
          + ", ".join(f"{p:.1f}" for p in peaks_kib))
    growth = peaks_kib[-1] - peaks_kib[1]
    print(f"Growth from batch 2 to batch 5: {growth:.1f} KiB "
          f"({'bounded — no leak' if abs(growth) < points_bytes / 1024 else 'UNBOUNDED — investigate'})")


if __name__ == "__main__":
    main()
