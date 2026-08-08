"""
Level 3 demo GIF generator — produces docs/assets/10_level3_live_demo.gif
from a short real-camera capture through the real, unmodified
DepthPerceptionPipeline.

Same reuse discipline as examples/visualize_level3.py: this script does
not compute disparity/depth/reprojection/transform/obstacles/free-space
itself — every value comes from a real DepthPerceptionResult. Uses the
same illustrative BODY transform as examples/visualize_level3.py
(imported, not re-derived).

Run:
    python examples/generate_demo_gif.py
"""

import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_level3 import _illustrative_transform  # noqa: E402

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.depth.depth_estimator import DepthEstimator

_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "stereo_calibration.xml")
_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs", "assets", "10_level3_live_demo.gif",
)
_N_FRAMES = 30
_CAPTURE_HZ = 3.0
_PANEL_SIZE = (220, 165)  # (w, h) per quadrant — kept small to keep the committed GIF file size modest
_FRAME_DURATION_MS = 180


def _disparity_to_bgr(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 0.0
    vis = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        d = disparity[valid]
        lo, hi = float(d.min()), float(d.max())
        if hi > lo:
            vis[valid] = np.clip((disparity[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        else:
            vis[valid] = 128
    bgr = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    bgr[~valid] = (40, 40, 40)
    return bgr


def _depth_to_bgr(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0.0
    vis = np.zeros(depth.shape, dtype=np.uint8)
    lo, hi = DepthEstimator.MIN_DEPTH_M, DepthEstimator.MAX_DEPTH_M
    if np.any(valid):
        vis[valid] = np.clip((depth[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    bgr[~valid] = (40, 40, 40)
    return bgr


def _topdown_bgr(result, size=165) -> np.ndarray:
    panel = np.full((size, size, 3), 25, dtype=np.uint8)
    cx, origin_row, mpp = size // 2, int(size * 0.9), 0.02

    def to_px(x, y):
        return int(cx - y / mpp), int(origin_row - x / mpp)

    body = result.geometry_body
    if body is not None and np.any(body.valid_mask):
        pts = body.points[body.valid_mask]
        stride = max(1, pts.shape[0] // 1500)
        for p in pts[::stride]:
            c, r = to_px(p[0], p[1])
            if 0 <= c < size and 0 <= r < size:
                panel[r, c] = (90, 60, 30)
    if result.obstacle_cloud is not None:
        pts = result.obstacle_cloud.points
        stride = max(1, pts.shape[0] // 1000)
        for p in pts[::stride]:
            c, r = to_px(p[0], p[1])
            if 0 <= c < size and 0 <= r < size:
                cv2.circle(panel, (c, r), 1, (40, 40, 230), -1)
    cv2.circle(panel, (cx, origin_row), 4, (255, 255, 255), -1)
    return panel


def main() -> None:
    calibration = load_stereo_calibration(_CALIBRATION_PATH)
    width, height = calibration.image_size
    config = PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)
    pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 0.")

    for _ in range(15):
        cap.read()

    frames = []
    period = 1.0 / _CAPTURE_HZ
    print(f"Capturing {_N_FRAMES} real frames at ~{_CAPTURE_HZ} Hz...")
    for i in range(_N_FRAMES):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {i}: read failed, skipping")
            continue
        frame_h, frame_w = frame.shape[:2]
        half_w = frame_w // 2
        left, right = frame[:, :half_w], frame[:, half_w:]
        result = pipeline.process(left, right)

        left_bgr = left if left.ndim == 3 else cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        pw, ph = _PANEL_SIZE
        left_panel = cv2.resize(left_bgr, (pw, ph))
        disp_panel = cv2.resize(_disparity_to_bgr(result.disparity_map), (pw, ph))
        depth_panel = cv2.resize(_depth_to_bgr(result.depth_map), (pw, ph))
        topdown_panel = cv2.resize(_topdown_bgr(result), (pw, ph))

        top = np.hstack([left_panel, disp_panel])
        bottom = np.hstack([depth_panel, topdown_panel])
        composite = np.vstack([top, bottom])
        cv2.putText(composite, f"frame {i+1}/{_N_FRAMES}  vf={result.geometry_metrics.valid_fraction:.2f}"
                    if result.geometry_metrics else f"frame {i+1}/{_N_FRAMES}",
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        frames.append(Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)))
        print(f"  captured frame {i+1}/{_N_FRAMES}")

        elapsed = time.perf_counter() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

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
