"""
E7 real-hardware long-run stability — Level 3, Phase E7 (Task 10).

Continuous real-camera capture+process loop, no physical interaction
required (upgrades E6's synthetic-only memory characterization —
examples/benchmark_e6_memory_stability.py — to real hardware). Tracks
process RSS, per-frame latency, and errors/degraded-episode counts.

Run:
    python examples/e7_live_long_run.py
"""

import time

import cv2
import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

_CALIBRATION_FILE = "examples/config/stereo_calibration.xml"
_N_FRAMES = 300
_CHECKPOINT_EVERY = 30


def _current_rss_kib() -> float:
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1])
    return float("nan")


def main() -> None:
    calibration = load_stereo_calibration(_CALIBRATION_FILE)
    width, height = calibration.image_size
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 0.")

    config = PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)
    transform = RigidTransform(
        rotation=np.eye(3), translation=np.array([0.10, 0.0, 0.05]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )
    pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)

    for _ in range(15):
        cap.read()

    latencies = []
    errors = 0
    degraded_episodes = 0
    was_degraded = False
    start = time.perf_counter()

    print(f"Processing {_N_FRAMES} live frames continuously...")
    print(f"{'frame':>6} {'elapsed_s':>10} {'RSS(KiB)':>10} {'latency_ms':>11} {'valid_frac':>11}")

    for i in range(1, _N_FRAMES + 1):
        ok, frame = cap.read()
        if not ok:
            errors += 1
            continue
        frame_h, frame_w = frame.shape[:2]
        half_w = frame_w // 2
        left, right = frame[:, :half_w], frame[:, half_w:]

        try:
            t0 = time.perf_counter()
            result = pipeline.process(left, right)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        except Exception as exc:  # noqa: BLE001 — deliberately recorded, not silently retried
            errors += 1
            print(f"  frame {i}: ERROR {type(exc).__name__}: {exc}")
            continue

        vf = result.geometry_metrics.valid_fraction if result.geometry_metrics else float("nan")
        is_degraded = vf < 0.05
        if is_degraded and not was_degraded:
            degraded_episodes += 1
        was_degraded = is_degraded

        if i % _CHECKPOINT_EVERY == 0 or i == 1:
            print(f"{i:>6} {time.perf_counter() - start:>10.1f} {_current_rss_kib():>10.1f} "
                  f"{latencies[-1]:>11.2f} {vf:>11.3f}")

    cap.release()

    latencies = np.array(latencies)
    print("\n--- Summary ---")
    print(f"Frames requested: {_N_FRAMES}, processed: {len(latencies)}, errors: {errors}")
    print(f"Degraded episodes (valid_fraction < 0.05, counted on transition): {degraded_episodes}")
    print(f"Latency: mean={latencies.mean():.2f}ms median={np.median(latencies):.2f}ms "
          f"std={latencies.std():.2f}ms p95={np.percentile(latencies, 95):.2f}ms max={latencies.max():.2f}ms")
    print(f"Effective FPS (mean-based): {1000.0 / latencies.mean():.1f}")
    print(f"Final RSS: {_current_rss_kib():.1f} KiB")


if __name__ == "__main__":
    main()
