"""
Phase I6 Part A — temporal fixtures. Reuses benchmarks/i1_stereo_accuracy/
fixtures.py's own proven low-frequency-canvas/disparity-remap technique.
"""
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from benchmarks.i1_stereo_accuracy.fixtures import (
    W, H, FX, BASELINE_M, _MAX_SHIFT_MARGIN, _low_freq_canvas, _remap_by_disparity, _to_bgr,
    disparity_for_depth,
)
from benchmarks.i5_surface_opening_clearance.opening.fixtures import make_gap_fixture


def static_pair(depth_m=2.0, seed=1, texture_scale=6):
    d = disparity_for_depth(depth_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, texture_scale, seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]
    disp_map = np.full((H, W), d, dtype=np.float32)
    right_gray = _remap_by_disparity(canvas, disp_map, x0)
    return _to_bgr(left_gray), _to_bgr(right_gray)


def two_object_pair(near_m, far_m, seed=1, texture_scale=6, boundary_frac=0.5):
    """A near/far two-region scene (reuses make_discontinuity_fixture's own
    technique inline) -- used for 'obstacle appearing/disappearing' via
    toggling between this and a flat far-only scene."""
    d_near = disparity_for_depth(near_m)
    d_far = disparity_for_depth(far_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, texture_scale, seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]
    boundary_col = int(boundary_frac * W)
    disp_map = np.full((H, W), d_far, dtype=np.float32)
    disp_map[:, :boundary_col] = d_near
    right_gray = _remap_by_disparity(canvas, disp_map, x0)
    return _to_bgr(left_gray), _to_bgr(right_gray)


def flat_pair(depth_m, seed=1, texture_scale=6):
    return static_pair(depth_m=depth_m, seed=seed, texture_scale=texture_scale)


def decorrelated_pair(seed=1):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (H, W), dtype=np.uint8)
    right = rng.integers(0, 255, (H, W), dtype=np.uint8)
    return _to_bgr(left), _to_bgr(right)


def textureless_pair(value=128):
    gray = np.full((H, W), value, dtype=np.uint8)
    return _to_bgr(gray), _to_bgr(gray)


def gap_pair(seed=1):
    """A genuine gap fixture (reused from I5) -- used for opening
    appearing/disappearing."""
    fx, gt = make_gap_fixture(near_left_m=2.0, gap_m=5.0, near_right_m=2.0, seed=seed, gap_cols=(160, 213))
    return fx.left, fx.right, gt


def wall_pair(depth_m=2.0, seed=1):
    """A continuous wall (no gap) -- the 'closed' counterpart to gap_pair."""
    return static_pair(depth_m=depth_m, seed=seed, texture_scale=6)
