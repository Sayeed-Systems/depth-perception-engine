"""
PointCloudBuilder latency/memory benchmark — Level 3, Phase E2.

No camera, no GUI, no hardware required — synthetic disparity only, same
spirit as examples/synthetic_demo.py. Measures PointCloudBuilder.build()
at 640x480 (this benchmark's fixed target resolution — independent of
whatever resolution the project's own physical rig calibration happens to
be, since reprojection is a per-pixel calculation with no dependency on
image size matching the calibration file).

Run:
    pip install -e ..
    python examples/benchmark_point_cloud.py
"""

import os
import time
import tracemalloc

import numpy as np

from depth_perception_engine.calibration import load_stereo_calibration
from depth_perception_engine.geometry import PointCloudBuilder

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_CALIBRATION_FILE = os.path.join(_EXAMPLES_DIR, "config", "stereo_calibration.xml")

WIDTH, HEIGHT = 640, 480
N_WARMUP = 10
N_ITERS = 200


def _synthetic_disparity(width: int, height: int, seed: int) -> np.ndarray:
    """A disparity map with a realistic mix of valid/invalid pixels, not a
    degenerate all-one-value array — closer to what a real SGBM output's
    validity pattern looks like than a uniform plane would be."""
    rng = np.random.default_rng(seed)
    disp = rng.uniform(4.0, 130.0, size=(height, width)).astype(np.float32)
    # ~15% invalid (no stereo correspondence), scattered — typical of a
    # textured scene with some low-texture regions.
    invalid = rng.random((height, width)) < 0.15
    disp[invalid] = 0.0
    return disp


def main() -> None:
    calibration = load_stereo_calibration(_CALIBRATION_FILE)
    builder = PointCloudBuilder.from_calibration(calibration)
    print(f"Builder ready: {builder}")
    print(f"Benchmark resolution: {WIDTH}x{HEIGHT}\n")

    disparity = _synthetic_disparity(WIDTH, HEIGHT, seed=0)

    # Warmup — first calls pay one-time costs (Python import machinery
    # already paid; this is about cv2/numpy's own internal buffer/thread
    # pool warmup) that would otherwise skew the first few samples.
    for _ in range(N_WARMUP):
        builder.build(disparity)

    # -- Latency ----------------------------------------------------------
    latencies_ms = np.empty(N_ITERS, dtype=np.float64)
    for i in range(N_ITERS):
        t0 = time.perf_counter()
        pc = builder.build(disparity)
        latencies_ms[i] = (time.perf_counter() - t0) * 1000.0

    print("Latency over {} iterations (post-warmup):".format(N_ITERS))
    print(f"  mean:   {latencies_ms.mean():.3f} ms")
    print(f"  std:    {latencies_ms.std():.3f} ms")
    print(f"  min:    {latencies_ms.min():.3f} ms")
    print(f"  max:    {latencies_ms.max():.3f} ms")
    print(f"  p50:    {np.percentile(latencies_ms, 50):.3f} ms")
    print(f"  p95:    {np.percentile(latencies_ms, 95):.3f} ms")
    print(f"  p99:    {np.percentile(latencies_ms, 99):.3f} ms")
    print(f"  -> effective max throughput: {1000.0 / latencies_ms.mean():.1f} Hz (single-threaded)")

    # -- Memory allocation behavior ----------------------------------------
    # Two separate questions: (1) exactly how big is one call's output
    # (deterministic, read directly off the returned arrays — not
    # estimated), and (2) does memory grow without bound across repeated
    # calls (a leak check, via tracemalloc's *peak* traced memory, which
    # unlike a live-snapshot diff isn't blind to same-iteration
    # allocate/free churn from re-binding `pc` every loop iteration).
    pc = builder.build(disparity)
    points_bytes = pc.points.nbytes
    mask_bytes = pc.valid_mask.nbytes
    print("\nMemory allocation behavior:")
    print(f"  PointCloud.points:     {points_bytes / 1024:.1f} KiB  (exact: H*W*3*4 bytes, float32)")
    print(f"  PointCloud.valid_mask: {mask_bytes / 1024:.1f} KiB  (exact: H*W*1 byte, bool)")
    print(f"  -> {(points_bytes + mask_bytes) / 1024:.1f} KiB attributable to the returned PointCloud alone; "
          "build() also allocates comparable-sized intermediates internally "
          "(disp_f cast, cv2's own (H,W,3) reprojection buffer, the invalid mask) "
          "that are freed before build() returns.")

    del pc
    tracemalloc.start()
    peaks_kib = []
    for batch in range(5):
        for _ in range(20):
            pc = builder.build(disparity)
        del pc
        _current, peak = tracemalloc.get_traced_memory()
        peaks_kib.append(peak / 1024)
    tracemalloc.stop()

    print(f"  peak traced memory after each successive batch of 20 calls (KiB): "
          + ", ".join(f"{p:.1f}" for p in peaks_kib))
    growth = peaks_kib[-1] - peaks_kib[1]
    print(f"  growth from batch 2 to batch 5: {growth:.1f} KiB "
          f"({'bounded — no leak' if abs(growth) < points_bytes / 1024 else 'UNBOUNDED — investigate'})")


if __name__ == "__main__":
    main()
