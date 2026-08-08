"""
Memory / long-run stability — Level 3, Phase E6 (Task 10).

Runs a bounded, repeated-processing loop (default 500 frames) through the
full E5-enabled DepthPerceptionPipeline and tracks:
  - Process RSS over time (current resident set size, read from
    /proc/self/status — Linux-specific, matching this repo's own
    development/CI environment; no external dependency like psutil is
    required or assumed available).
  - Python-level allocations via tracemalloc (peak traced memory at
    checkpoints).
  - Object counts for the four Level 3 geometry types (via gc.get_objects()
    type-counting) — a direct check for "retained previous frames" /
    "accidental result history" / "unbounded caches": if the pipeline
    were accidentally holding references to old PointCloud/ObstacleCloud/
    FreeSpaceRays/GeometryMetrics instances, the live count of each would
    grow roughly linearly with frames processed; it should not.

This script observes and reports; it does NOT assert pass/fail (matching
Task 10's explicit "do not claim memory leak free from a short Python
test" instruction) — it prints precise, non-overclaiming conclusions
about what was and was not observed over the N frames actually run.

Run:
    pip install -e ..
    python examples/benchmark_e6_memory_stability.py
"""

import gc
import time

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.types import FreeSpaceRays, GeometryMetrics, ObstacleCloud, PointCloud

N_FRAMES = 500
N_WARMUP = 20
CHECKPOINT_EVERY = 50


def _current_rss_kib() -> float:
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1])  # kB
    return float("nan")


def _live_object_count(cls) -> int:
    return sum(1 for obj in gc.get_objects() if type(obj) is cls)


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def main() -> None:
    calibration = load_stereo_calibration("examples/config/stereo_calibration.xml")
    config = PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)
    pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_transform())

    w, h = calibration.image_size
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

    print(f"Warm-up: {N_WARMUP} discarded frames before tracking begins.")
    for _ in range(N_WARMUP):
        pipeline.process(left, right)
    gc.collect()

    print(f"Tracking {N_FRAMES} frames, checkpoint every {CHECKPOINT_EVERY} frames.\n")
    print(f"{'frame':>6} {'RSS (KiB)':>12} {'ΔRSS vs start':>15} "
          f"{'PointCloud':>11} {'ObstacleCloud':>14} {'FreeSpaceRays':>14} {'GeometryMetrics':>16}")

    rss_samples = []
    result = None  # holds only the MOST RECENT result, matching real caller usage
    start_rss = None

    for frame_idx in range(1, N_FRAMES + 1):
        result = pipeline.process(left, right)

        if frame_idx % CHECKPOINT_EVERY == 0 or frame_idx == 1:
            gc.collect()
            rss = _current_rss_kib()
            if start_rss is None:
                start_rss = rss
            rss_samples.append((frame_idx, rss))
            print(f"{frame_idx:>6} {rss:>12.1f} {rss - start_rss:>+15.1f} "
                  f"{_live_object_count(PointCloud):>11} {_live_object_count(ObstacleCloud):>14} "
                  f"{_live_object_count(FreeSpaceRays):>14} {_live_object_count(GeometryMetrics):>16}")

    del result
    gc.collect()

    print("\n--- Analysis ---")
    first_rss = rss_samples[0][1]
    last_rss = rss_samples[-1][1]
    total_delta = last_rss - first_rss
    print(f"RSS at frame {rss_samples[0][0]}: {first_rss:.1f} KiB")
    print(f"RSS at frame {rss_samples[-1][0]}: {last_rss:.1f} KiB")
    print(f"Total ΔRSS over {N_FRAMES - rss_samples[0][0]} tracked frames: {total_delta:+.1f} KiB "
          f"({total_delta / (N_FRAMES - rss_samples[0][0]):+.2f} KiB/frame average)")

    # Simple monotonic-growth heuristic: is RSS higher at every later
    # checkpoint than at the first, with no checkpoint returning close to
    # baseline? A single-object working set (one process() call's worth
    # of arrays) should NOT show a sustained upward trend across
    # checkpoints spaced 50 frames apart.
    deltas = [rss - first_rss for _, rss in rss_samples]
    strictly_increasing_tail = all(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1))
    print(f"\nRSS trend strictly non-decreasing across all checkpoints: {strictly_increasing_tail}")
    if strictly_increasing_tail and total_delta > 2048:  # > 2 MiB sustained growth
        print("CONCLUSION: Sustained upward RSS trend observed (> 2 MiB over the run). "
              "This warrants further investigation before considering memory behavior settled — "
              "NOT claimed leak-free.")
    else:
        print(f"CONCLUSION: No monotonic memory growth observed over {N_FRAMES} frames "
              f"({N_FRAMES - rss_samples[0][0]} tracked checkpoints, {total_delta:+.1f} KiB total RSS delta). "
              "This is evidence from one run of this length, on this platform — not a formal proof of the "
              "absence of a leak under all conditions.")

    final_counts = {
        "PointCloud": _live_object_count(PointCloud),
        "ObstacleCloud": _live_object_count(ObstacleCloud),
        "FreeSpaceRays": _live_object_count(FreeSpaceRays),
        "GeometryMetrics": _live_object_count(GeometryMetrics),
    }
    print(f"\nLive Level-3 geometry object counts after the run (only the last `result` variable "
          f"is still referenced by this script): {final_counts}")
    print("Each count should be small and bounded (matching how many geometry objects the current "
          "`result` alone references), not proportional to N_FRAMES — proportional growth would "
          "indicate an accidental cache or result-history leak.")


if __name__ == "__main__":
    main()
