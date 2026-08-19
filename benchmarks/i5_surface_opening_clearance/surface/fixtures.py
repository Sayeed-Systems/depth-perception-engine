"""
Phase I5 Part A — slanted-plane stereo-PAIR fixtures (through REAL StereoSGBM,
not analytic disparity injection like tests/test_d10_integrated_ground_truth.py's
own Scenario 3). Reuses benchmarks/i1_stereo_accuracy/fixtures.py's proven
low-frequency-canvas / disparity-remap technique directly.
"""
import sys
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from benchmarks.i1_stereo_accuracy.fixtures import (
    W, H, FX, BASELINE_M, _MAX_SHIFT_MARGIN, _low_freq_canvas, _remap_by_disparity, _to_bgr,
)

CX = 155.57466888427734
CY = 185.47198152542114


def slanted_plane_disparity(theta_x_deg: float, theta_y_deg: float, z0_m: float) -> np.ndarray:
    """Analytic disparity map (H, W) for a single plane tilted theta_x
    (about the X axis -> row-wise/pitch gradient) and theta_y (about the
    Y axis -> column-wise/yaw gradient) simultaneously, at the frame
    center's depth z0_m. Normal (camera-optical frame, oriented toward
    the camera along -Z): n = (sin(theta_y), -sin(theta_x), -cos(theta_y)*cos(theta_x))
    approximately for small-to-moderate combined angles — exact for the
    single-axis cases (theta_x=0 or theta_y=0), which is what this
    fixture module actually uses (yaw-only, pitch-only) plus one modest
    combined case validated by construction (both angles applied via the
    same plane equation n.P = d solved per-pixel, not a small-angle
    approximation).
    """
    ty = np.deg2rad(theta_y_deg)
    tx = np.deg2rad(theta_x_deg)
    # Build normal via two sequential axis rotations of camera-frame (0,0,-1):
    # first tilt about Y (yaw) then about X (pitch) -- exact, not linearized.
    n0 = np.array([0.0, 0.0, -1.0])
    cy_, sy_ = np.cos(ty), np.sin(ty)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    cx_, sx_ = np.cos(tx), np.sin(tx)
    Rx = np.array([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]])
    n = Rx @ Ry @ n0
    n = n / np.linalg.norm(n)
    d = float(np.dot(n, [0.0, 0.0, z0_m]))

    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    denom = n[0] * (u - CX) / FX + n[1] * (v - CY) / FX + n[2]
    z = d / denom
    assert np.all(z > 0.0), "fixture must stay in front of the camera everywhere"
    disparity = (FX * BASELINE_M / z).astype(np.float32)
    return disparity, n


def make_slanted_plane_pair(theta_x_deg: float, theta_y_deg: float, z0_m: float,
                             texture_scale: int, seed: int):
    """Real stereo image PAIR (not analytic disparity injection) encoding
    the given slanted plane via image warping, for the REAL DisparityEngine/
    StereoSGBM to match -- characterizes real SGBM noise on top of the
    exact PCA-fit math tests/test_d10_integrated_ground_truth.py's own
    Scenario 3 already proved noise-free (analytic disparity input)."""
    disparity, expected_normal = slanted_plane_disparity(theta_x_deg, theta_y_deg, z0_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, texture_scale, seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]
    # disparity varies per-pixel (not just per-column) for the pitch/
    # combined cases -- _remap_by_disparity already supports a full (H,W)
    # disp_map, matching make_discontinuity_fixture's own usage.
    right_gray = _remap_by_disparity(canvas, disparity, x0)
    return _to_bgr(left_gray), _to_bgr(right_gray), expected_normal, disparity
