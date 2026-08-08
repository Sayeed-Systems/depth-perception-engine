"""
E7 live hardware capture session — Level 3, Phase E7 (Tasks 3, 7, 8, 9).

Interactive, phased real-camera capture. Camera I/O lives here, in
examples/, never in the core engine (matches examples/live_demo.py's
existing precedent). For each phase, this script prints clear on-screen
instructions with a countdown, then captures+processes real frames for a
fixed window at a fixed target rate, recording per-frame numeric
telemetry (never just a visual impression) to a CSV for later analysis.

Phases (Task 3 A-G / Task 7 motion / Task 8 degraded):
    0. baseline       — scene stationary, nothing manipulated
    1. toward         — human moves an object gradually toward the camera
    2. away           — human moves the same object away
    3. left           — human moves the object left
    4. right          — human moves the object right
    5. occlusion      — human partially covers the lens/FOV briefly
    6. textureless    — camera pointed at a blank/low-texture surface

No parameter is tuned per phase (Task 3's explicit instruction) — the
same PipelineConfig is used throughout.

Run:
    pip install -e ".[viz]"   # not required for this script itself, only visualize_level3.py
    python examples/e7_live_capture_session.py
"""

import csv
import os
import time

import cv2
import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import classify_geometry_quality

_CALIBRATION_FILE = "examples/config/stereo_calibration.xml"
_OUTPUT_DIR = "/tmp/claude-0/-home-sayeed/0a4f25aa-54b1-4be6-aa17-a868d0915306/scratchpad/e7_session"
_PHASE_DURATION_S = 8.0
_TARGET_HZ = 3.0
_COUNTDOWN_S = 3

_PHASES = [
    ("baseline", "Keep the scene as it is — do not move anything."),
    ("toward", "Hold one clear object ahead of the camera and move it GRADUALLY TOWARD the camera."),
    ("away", "Move that same object GRADUALLY AWAY from the camera."),
    ("left", "Move the object LEFT (camera's left) at roughly constant distance."),
    ("right", "Move the object RIGHT (camera's right) at roughly constant distance."),
    ("occlusion", "PARTIALLY cover part of the camera's field of view with your hand (not the whole lens)."),
    ("textureless", "Point the camera at (or hold a blank sheet/wall in front of) a flat, low-texture surface."),
]


def _illustrative_transform() -> RigidTransform:
    angle = np.deg2rad(10.0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    translation = np.array([0.10, 0.0, 0.05])
    return RigidTransform(
        rotation=rotation, translation=translation,
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"  starting in {remaining}...", flush=True)
        time.sleep(1)
    print("  GO", flush=True)


def main() -> None:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    calibration = load_stereo_calibration(_CALIBRATION_FILE)
    width, height = calibration.image_size

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 0.")

    config = PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)
    transform = _illustrative_transform()
    pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)

    print("Discarding warm-up frames for auto-exposure...")
    for _ in range(15):
        cap.read()

    csv_path = os.path.join(_OUTPUT_DIR, "e7_session_telemetry.csv")
    fieldnames = [
        "phase", "phase_frame_index", "t_since_phase_start_s", "processing_time_ms",
        "valid_disparity_fraction", "valid_depth_fraction", "confidence",
        "geometry_valid_fraction", "min_obstacle_distance_m", "mean_free_space_m",
        "obstacle_point_count", "free_space_ray_count", "quality", "navigation_decision",
    ]
    rows = []

    period_s = 1.0 / _TARGET_HZ
    for phase_name, instruction in _PHASES:
        print(f"\n=== Phase: {phase_name} ===")
        print(f"  INSTRUCTION: {instruction}")
        _countdown(_COUNTDOWN_S)

        phase_start = time.perf_counter()
        frame_idx = 0
        saved_frame = False
        while (time.perf_counter() - phase_start) < _PHASE_DURATION_S:
            loop_t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("  WARNING: frame read failed, skipping.")
                continue

            frame_h, frame_w = frame.shape[:2]
            half_w = frame_w // 2
            left, right = frame[:, :half_w], frame[:, half_w:]

            if not saved_frame:
                np.save(os.path.join(_OUTPUT_DIR, f"{phase_name}_left.npy"), left)
                np.save(os.path.join(_OUTPUT_DIR, f"{phase_name}_right.npy"), right)
                saved_frame = True

            result = pipeline.process(left, right)
            quality = classify_geometry_quality(
                result.geometry_metrics, config.geometry_healthy_min_valid_fraction, config.geometry_degraded_min_valid_fraction,
            ) if result.geometry_metrics is not None else None

            rows.append({
                "phase": phase_name,
                "phase_frame_index": frame_idx,
                "t_since_phase_start_s": round(time.perf_counter() - phase_start, 3),
                "processing_time_ms": round(result.processing_time_ms, 3),
                "valid_disparity_fraction": round(float(result.valid_disparity_mask.mean()), 4),
                "valid_depth_fraction": round(float(result.valid_depth_mask.mean()), 4),
                "confidence": round(result.confidence, 4),
                "geometry_valid_fraction": round(result.geometry_metrics.valid_fraction, 4) if result.geometry_metrics else None,
                "min_obstacle_distance_m": result.geometry_metrics.min_obstacle_distance_m if result.geometry_metrics else None,
                "mean_free_space_m": result.geometry_metrics.mean_free_space_m if result.geometry_metrics else None,
                "obstacle_point_count": result.obstacle_cloud.points.shape[0] if result.obstacle_cloud is not None else None,
                "free_space_ray_count": result.free_space_rays.ranges_m.shape[0] if result.free_space_rays is not None else None,
                "quality": quality,
                "navigation_decision": result.traversability_mask.decision.value,
            })
            frame_idx += 1

            elapsed = time.perf_counter() - loop_t0
            if elapsed < period_s:
                time.sleep(period_s - elapsed)

        print(f"  Captured {frame_idx} frames for phase '{phase_name}'.")

    cap.release()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows of telemetry to {csv_path}")
    print(f"Saved one representative raw stereo frame pair per phase to {_OUTPUT_DIR}/<phase>_{{left,right}}.npy")


if __name__ == "__main__":
    main()
