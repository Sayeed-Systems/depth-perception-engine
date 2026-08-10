"""
Level 4 live visual validation tool — standalone example, OpenCV GUI.

Drives the real, unmodified DepthPerceptionPipeline from the real stereo
camera and displays raw + Level 4 temporal outputs continuously, one
frame at a time, in a live dashboard.

REUSE, not a competing camera abstraction: camera acquisition mirrors
examples/live_demo.py's own "open cv2.VideoCapture directly, no forced
codec" approach (this USB stereo camera does not support MJPEG) and
examples/visualize_level3.py's own calibration-derived
width*2/height capture geometry and left/right split logic
(width, height = calibration.image_size; frame[:, :half], frame[:, half:])
— reproduced here, not reimplemented differently, so this script and
visualize_level3.py agree on exactly how this hardware is opened. The
illustrative BODY extrinsic is imported directly from
examples/visualize_level3.py (the same import generate_demo_gif.py
already established), not redefined.

This script owns camera I/O, the on-screen cv2 display, and keyboard
handling only — the same three things live_demo.py's own docstring
already scopes an examples/ script to. Every depth/temporal computation
is delegated to the installed depth_perception_engine package. This
script reads ONLY public DepthPerceptionResult fields
(disparity_map/depth_map/confidence/temporal_consistency/
temporal_stabilization/rotation_compensation_status/
motion_aware_reliability/temporal_persistence) — it does not call any
temporal.* algorithm function directly and does not reimplement
consistency/stabilization/compensation/reliability/persistence itself.

IMU remains simulated. Pressing 'i' toggles an optional, directly-
constructed temporal.MotionHint attached to each subsequent frame — a
plain synthetic value (docs/LEVEL4_SIMULATED_IMU.md), never a hardware
reading. Whenever it is enabled, the dashboard's title bar and an
on-screen badge say so unmistakably ("SIMULATED MOTIONHINT: ON", drawn in
a color no other telemetry line uses) — this must never be mistaken for
real IMU data.

No ROS. No neural methods. No Level 5. No change to
depth_perception_engine's own behavior or public API — this file adds
nothing to src/depth_perception_engine/.

Controls:
    q / ESC = quit
    r       = pipeline reset (DepthPerceptionPipeline.reset())
    i       = toggle simulated MotionHint

Run:
    python examples/visualize_level4_live.py
    python examples/visualize_level4_live.py --camera-index 1
    python examples/visualize_level4_live.py --help
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)

from visualize_level3 import _illustrative_transform  # noqa: E402 — reused, not redefined

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration  # noqa: E402
from depth_perception_engine.frames import FrameId  # noqa: E402
from depth_perception_engine.temporal import MotionHint  # noqa: E402
from depth_perception_engine.temporal.persistence import TemporalPersistenceCellState  # noqa: E402

_CALIBRATION_FILE = os.path.join(_EXAMPLES_DIR, "config", "stereo_calibration.xml")
_WINDOW_NAME = "Level 4 live temporal validation"

# TemporalPersistenceCellState.NO_EVIDENCE/NEW/PERSISTENT/DISAPPEARING = 0/1/2/3 — BGR.
_PERSISTENCE_COLORS_BGR = {
    TemporalPersistenceCellState.NO_EVIDENCE: (40, 40, 40),
    TemporalPersistenceCellState.NEW: (50, 210, 230),
    TemporalPersistenceCellState.PERSISTENT: (80, 170, 60),
    TemporalPersistenceCellState.DISAPPEARING: (30, 90, 220),
}

_WHITE = (255, 255, 255)
_CYAN = (220, 220, 0)
_SIMULATED_BADGE_COLOR = (0, 0, 255)  # pure red — reserved for the SIMULATED MotionHint badge only


def _build_pipeline_config() -> PipelineConfig:
    """Every Level 3 geometry flag and every Level 4 flag enabled, at
    their existing frozen defaults — this script does not tune or
    override a single threshold (no config/behavior change, per this
    pass's own instruction)."""
    return PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
    )


def _open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """Same capture geometry examples/visualize_level3.py's own
    _capture_live_pair() uses (width*2 side-by-side, no forced codec —
    this USB stereo camera does not support MJPEG, matching
    examples/live_demo.py's own documented reason for the same choice),
    adapted for a continuous read loop instead of one single-shot read."""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")
    for _ in range(10):  # discard a few frames for auto-exposure to settle
        cap.read()
    return cap


def _split_frame(frame: np.ndarray):
    frame_h, frame_w = frame.shape[:2]
    half_w = frame_w // 2
    return frame[:, :half_w], frame[:, half_w:]


def _depth_to_bgr(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0.0
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        d_min, d_max = float(depth[valid].min()), float(depth[valid].max())
        if d_max > d_min:
            vis[valid] = np.clip((depth[valid] - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            vis[valid] = 128
    bgr = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)
    bgr[~valid] = (35, 35, 35)
    return bgr


def _persistence_to_bgr(state_grid) -> np.ndarray:
    h, w = state_grid.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    for code, color in _PERSISTENCE_COLORS_BGR.items():
        bgr[state_grid == code] = color
    return bgr


def _na_panel(size, text="N/A") -> np.ndarray:
    h, w = size
    panel = np.full((h, w, 3), 25, dtype=np.uint8)
    cv2.putText(panel, text, (w // 2 - 20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1, cv2.LINE_AA)
    return panel


def _label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _WHITE, 1, cv2.LINE_AA)
    return panel


def _build_dashboard(left_bgr, result, fps, latency_ms, motion_hint_simulated, panel_size):
    pw, ph = panel_size
    left_panel = _label_panel(cv2.resize(left_bgr, (pw, ph)), "left camera")
    depth_panel = _label_panel(cv2.resize(_depth_to_bgr(result.depth_map), (pw, ph)), "raw depth")

    ts = result.temporal_stabilization
    if ts is not None and ts.stabilized_depth_m is not None:
        stabilized_panel = _label_panel(cv2.resize(_depth_to_bgr(ts.stabilized_depth_m), (pw, ph)), "stabilized depth")
    else:
        stabilized_panel = _label_panel(_na_panel((ph, pw)), "stabilized depth")

    tp = result.temporal_persistence
    if tp is not None and tp.state_grid is not None:
        persistence_panel = _label_panel(cv2.resize(_persistence_to_bgr(tp.state_grid), (pw, ph), interpolation=cv2.INTER_NEAREST), "persistence state")
    else:
        persistence_panel = _label_panel(_na_panel((ph, pw)), "persistence state")

    top = np.hstack([left_panel, depth_panel])
    bottom = np.hstack([stabilized_panel, persistence_panel])
    grid = np.vstack([top, bottom])

    telemetry = _build_telemetry_panel(result, fps, latency_ms, motion_hint_simulated, height=grid.shape[0], width=280)
    return np.hstack([grid, telemetry])


def _build_telemetry_panel(result, fps, latency_ms, motion_hint_simulated, height, width):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    tc = result.temporal_consistency
    ts = result.temporal_stabilization
    mar = result.motion_aware_reliability
    tp = result.temporal_persistence

    lines = [
        (f"FPS: {fps:.1f}   latency: {latency_ms:.1f} ms", _WHITE),
        (f"confidence: {result.confidence:.3f}", _WHITE),
        ("", _WHITE),
        (f"consistency:   {tc.state if tc else 'N/A'}", _CYAN),
        (f"stabilization: {ts.state if ts else 'N/A'}", _CYAN),
        (f"rotation comp: {result.rotation_compensation_status}", _CYAN),
        (f"reliability:   {mar.state if mar else 'N/A'}", _CYAN),
        ("", _WHITE),
        ("persistence:", _WHITE),
        (f"  NEW:         {tp.new_count if tp else 'N/A'}", _WHITE),
        (f"  PERSISTENT:  {tp.persistent_count if tp else 'N/A'}", _WHITE),
        (f"  DISAPPEARING:{tp.disappearing_count if tp else 'N/A'}", _WHITE),
        (f"  EXPIRED:     {tp.expired_count if tp else 'N/A'}", _WHITE),
    ]
    y = 20
    for text, color in lines:
        cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y += 18

    y += 10
    if motion_hint_simulated:
        cv2.rectangle(panel, (4, y - 14), (width - 4, y + 6), _SIMULATED_BADGE_COLOR, 2)
        cv2.putText(panel, "SIMULATED MOTIONHINT: ON", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _SIMULATED_BADGE_COLOR, 1, cv2.LINE_AA)
        cv2.putText(panel, "(NOT real IMU data)", (10, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _SIMULATED_BADGE_COLOR, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "simulated MotionHint: off", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(panel, "press 'i' to toggle, 'r' to reset, 'q' to quit", (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1, cv2.LINE_AA)

    return panel


def _simulated_motion_hint(timestamp: float):
    """A plain, directly-constructed temporal.MotionHint — a small,
    fixed synthetic yaw rate, exactly the same "hand-constructed value"
    convention every other example/test in this repository already uses
    (docs/LEVEL4_SIMULATED_IMU.md) — never read from any sensor."""
    return MotionHint(
        timestamp=timestamp, angular_velocity_rad_s=np.array([0.0, 0.01, 0.0]), frame_id=FrameId.BODY,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-index", type=int, default=0, help="cv2.VideoCapture index (default: 0).")
    parser.add_argument(
        "--simulate-motion-hint", action="store_true",
        help="Start with the simulated MotionHint toggle already ON (can still be toggled with 'i').",
    )
    args = parser.parse_args()

    print("Loading calibration and building pipeline...")
    calibration = load_stereo_calibration(_CALIBRATION_FILE)
    width, height = calibration.image_size
    pipeline = DepthPerceptionPipeline(
        _build_pipeline_config(), calibration, body_T_camera_left=_illustrative_transform(),
    )

    print(f"Opening camera index {args.camera_index}...")
    cap = _open_camera(args.camera_index, width, height)

    print("Running — press 'q'/ESC to quit, 'r' to reset, 'i' to toggle simulated MotionHint.")
    motion_hint_enabled = bool(args.simulate_motion_hint)
    frame_count = 0
    fps = 0.0
    t_fps = time.perf_counter()
    t_start = time.perf_counter()

    try:
        while True:
            ok, raw = cap.read()
            if not ok or raw is None:
                continue

            left, right = _split_frame(raw)
            timestamp = time.perf_counter() - t_start
            motion_hints = [_simulated_motion_hint(timestamp)] if motion_hint_enabled else None

            t0 = time.perf_counter()
            result = pipeline.process(left, right, left_timestamp=timestamp, motion_hints=motion_hints)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            frame_count += 1
            elapsed = time.perf_counter() - t_fps
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_fps = time.perf_counter()

            dashboard = _build_dashboard(left, result, fps, latency_ms, motion_hint_enabled, panel_size=(320, 240))
            cv2.imshow(_WINDOW_NAME, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 == ESC
                break
            elif key == ord('r'):
                pipeline.reset()
                print("  pipeline reset()")
            elif key == ord('i'):
                motion_hint_enabled = not motion_hint_enabled
                print(f"  simulated MotionHint: {'ON' if motion_hint_enabled else 'off'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
