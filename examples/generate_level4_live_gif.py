"""
Level 4 live demo GIF generator — produces docs/assets/11_level4_live_demo.gif
from a short REAL-camera capture through the real, unmodified
DepthPerceptionPipeline with every Level 4 capability enabled.

Same reuse discipline as examples/generate_demo_gif.py (Level 3's own
GIF generator): this script does not compute disparity/depth/temporal-
consistency/stabilization/rotation-compensation/reliability/persistence
itself — every value comes from a real DepthPerceptionResult. Camera
acquisition and dashboard rendering are both imported directly from
examples/visualize_level4_live.py (the live tool this GIF documents),
not reimplemented a second time.

The captured scene is real; the MotionHint is not. This capture toggles
the simulated MotionHint on partway through, so the GIF itself shows the
"SIMULATED MOTIONHINT: ON" badge appearing — real stereo geometry, a
synthetic motion input, never conflated (see
docs/LEVEL4_HARDWARE_VALIDATION_PENDING.md).

Run:
    python examples/generate_level4_live_gif.py
"""

import os
import sys
import time

import cv2
from PIL import Image

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)

from visualize_level3 import _illustrative_transform  # noqa: E402
from visualize_level4_live import (  # noqa: E402
    _CALIBRATION_FILE,
    _build_dashboard,
    _build_pipeline_config,
    _open_camera,
    _simulated_motion_hint,
    _split_frame,
)

from depth_perception_engine import DepthPerceptionPipeline, load_stereo_calibration  # noqa: E402

_OUTPUT_PATH = os.path.join(_EXAMPLES_DIR, "..", "docs", "assets", "11_level4_live_demo.gif")
_N_FRAMES = 24
_CAPTURE_HZ = 3.0
_PANEL_SIZE = (180, 150)  # grid height = 2*150 = 300px — the minimum that keeps the telemetry
# column (13 text lines + the SIMULATED MotionHint badge, ~290px tall as laid out by
# visualize_level4_live.py's own _build_telemetry_panel()) from being clipped; still much
# smaller than the live tool's own interactive default (320, 240), matching
# generate_demo_gif.py's own "kept small to keep the committed GIF file size modest" discipline.
_FRAME_DURATION_MS = 220
_MOTION_HINT_STARTS_AT_FRAME = _N_FRAMES // 2  # SIMULATED badge turns on partway through the GIF


def main() -> None:
    calibration = load_stereo_calibration(_CALIBRATION_FILE)
    width, height = calibration.image_size
    pipeline = DepthPerceptionPipeline(
        _build_pipeline_config(), calibration, body_T_camera_left=_illustrative_transform(),
    )

    print("Opening camera...")
    cap = _open_camera(camera_index=0, width=width, height=height)

    print("Position the camera/scene now.")
    for remaining in range(5, 0, -1):
        print(f"  capturing in {remaining}...", flush=True)
        time.sleep(1)
    print("  CAPTURING NOW")

    frames = []
    period = 1.0 / _CAPTURE_HZ
    t_start = time.perf_counter()
    print(f"Capturing {_N_FRAMES} real frames at ~{_CAPTURE_HZ} Hz through the full Level 4 chain "
          f"(simulated MotionHint turns on at frame {_MOTION_HINT_STARTS_AT_FRAME + 1})...")

    try:
        for i in range(_N_FRAMES):
            t0 = time.perf_counter()
            ok, raw = cap.read()
            if not ok or raw is None:
                print(f"  frame {i}: read failed, skipping")
                continue

            left, right = _split_frame(raw)
            timestamp = time.perf_counter() - t_start
            motion_hint_enabled = i >= _MOTION_HINT_STARTS_AT_FRAME
            motion_hints = [_simulated_motion_hint(timestamp)] if motion_hint_enabled else None

            proc_t0 = time.perf_counter()
            result = pipeline.process(left, right, left_timestamp=timestamp, motion_hints=motion_hints)
            latency_ms = (time.perf_counter() - proc_t0) * 1000.0
            fps = 1.0 / max(1e-6, time.perf_counter() - t0)

            dashboard = _build_dashboard(left, result, fps, latency_ms, motion_hint_enabled, panel_size=_PANEL_SIZE)
            frames.append(Image.fromarray(cv2.cvtColor(dashboard, cv2.COLOR_BGR2RGB)))
            print(f"  captured frame {i + 1}/{_N_FRAMES}  fps={fps:5.1f}  latency={latency_ms:5.1f}ms  "
                  f"persistence={result.temporal_persistence.state if result.temporal_persistence else None}  "
                  f"new={result.temporal_persistence.new_count if result.temporal_persistence else None}  "
                  f"persistent={result.temporal_persistence.persistent_count if result.temporal_persistence else None}  "
                  f"disappearing={result.temporal_persistence.disappearing_count if result.temporal_persistence else None}  "
                  f"motion_hint={'SIMULATED' if motion_hint_enabled else 'off'}")

            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("No frames captured — cannot build a GIF.")

    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    frames[0].save(
        _OUTPUT_PATH, save_all=True, append_images=frames[1:],
        duration=_FRAME_DURATION_MS, loop=0, optimize=True,
    )
    size_kib = os.path.getsize(_OUTPUT_PATH) / 1024
    print(f"\nSaved {len(frames)}-frame GIF: {_OUTPUT_PATH} ({size_kib:.1f} KiB)")


if __name__ == "__main__":
    main()
