"""
Phase I4 — additional deterministic fixtures for boundary-support-distribution
characterization (Steps A-I of the I4 task). Reuses benchmarks/i1_stereo_accuracy/
fixtures.py's own proven low-frequency-canvas / disparity-remap technique
directly rather than reinventing it.
"""
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from benchmarks.i1_stereo_accuracy.fixtures import (
    W, H, FX, BASELINE_M, _MAX_SHIFT_MARGIN, _low_freq_canvas, _remap_by_disparity, _to_bgr,
    disparity_for_depth, Fixture, make_discontinuity_fixture, make_decorrelated_fixture,
    make_repetitive_fixture, make_flat_fixture,
)


def make_narrow_obstacle_fixture(near_m: float, far_m: float, seed: int, texture_scale: int = 6,
                                  box_frac=(0.4, 0.6)) -> Fixture:
    """A compact near 'box' (columns [box_frac[0]*W, box_frac[1]*W)) against
    a far background on both sides — two REAL transitions (far->near,
    near->far), unlike make_discontinuity_fixture's single frame-spanning
    step. Models a genuine, narrow, both-sides-supported obstacle edge."""
    d_near = disparity_for_depth(near_m)
    d_far = disparity_for_depth(far_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, texture_scale, seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]

    c0 = int(box_frac[0] * W)
    c1 = int(box_frac[1] * W)
    disp_map = np.full((H, W), d_far, dtype=np.float32)
    disp_map[:, c0:c1] = d_near
    right_gray = _remap_by_disparity(canvas, disp_map, x0)

    return Fixture(
        name=f"E_narrow_obstacle_{near_m}m-{far_m}m_seed{seed}", scenario="E", depth_m=None,
        true_disparity_px=None, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
        gt_invalid_mask=None, gt_disparity_map=disp_map,
    )


def make_weak_texture_discontinuity_fixture(near_m: float, far_m: float, seed: int) -> Fixture:
    """B/F: a genuine near/far step (same construction as
    make_discontinuity_fixture) but with WEAK (coarse/low-frequency)
    texture on both sides, matching fixtures.py's own 'C' texture scale —
    tests whether a genuine boundary under weak-texture conditions still
    has strong-enough support to be a fair, legitimate low-support case."""
    d_near = disparity_for_depth(near_m)
    d_far = disparity_for_depth(far_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, texture_scale=24, seed=seed)  # 24 = fixtures.py's own "C" (weak) scale
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]

    boundary_col = W // 2
    disp_map = np.full((H, W), d_far, dtype=np.float32)
    disp_map[:, :boundary_col] = d_near
    right_gray = _remap_by_disparity(canvas, disp_map, x0)

    return Fixture(
        name=f"F_weak_texture_{near_m}m-{far_m}m_seed{seed}", scenario="F", depth_m=None,
        true_disparity_px=None, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
        gt_invalid_mask=None, gt_disparity_map=disp_map,
    )


def make_textureless_fixture(depth_m: float = 2.0) -> Fixture:
    """I: fully flat/textureless — zero real correspondence signal
    anywhere (SGBM's data cost is uninformative everywhere), the
    'invalid/no-correspondence region' negative case distinct from G
    (which is i.i.d. noise, still has SOME local texture SGBM can latch
    onto)."""
    gray = np.full((H, W), 128, dtype=np.uint8)
    return Fixture(
        name=f"I_textureless", scenario="I", depth_m=depth_m, true_disparity_px=disparity_for_depth(depth_m),
        left=_to_bgr(gray), right=_to_bgr(gray),
    )
