"""
Phase I5 Part C — clearance-qualification fixtures. Reuses
benchmarks/i1_stereo_accuracy/fixtures.py's proven low-frequency-canvas /
disparity-remap technique directly. New: make_dis_occlusion_fixture, the
far-side mirror of make_discontinuity_fixture(occlusion=True)'s near-side
strip, and make_multi_zone_fixture for multiple discrete obstacle sectors.
"""
import sys
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from benchmarks.i1_stereo_accuracy.fixtures import (
    W, H, _MAX_SHIFT_MARGIN, _TEXTURE_SCALE, _low_freq_canvas, _remap_by_disparity, _to_bgr,
    disparity_for_depth, Fixture,
)


def make_dis_occlusion_fixture(near_m: float, far_m: float, seed: int) -> Fixture:
    """Far-side mirror of make_discontinuity_fixture(occlusion=True)'s
    near-side occlusion strip. Same near(left)/far(right) step at
    boundary_col=W//2. The near-side construction injects noise into the
    RIGHT image at columns [boundary_col - strip_w, boundary_col) (the
    last strip_w columns of the NEAR region, immediately before the
    step). This mirrors that exactly onto the FAR side: noise injected at
    columns [boundary_col, boundary_col + strip_w) (the first strip_w
    columns of the FAR region, immediately after the step) — same width
    (round(d_near - d_far)), same noise-injection technique, opposite
    side of the transition. This is a deliberate stress-test construction
    (not necessarily the true physical dis-occlusion geometry for this
    specific near-left/far-right sign convention), to directly measure
    whether contamination on the FAR side of a transition can ever
    produce a false-clear (a SUPPORTED sector reporting FARTHER than the
    true nearest observable obstacle)."""
    d_near = disparity_for_depth(near_m)
    d_far = disparity_for_depth(far_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, _TEXTURE_SCALE["B"], seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]

    boundary_col = W // 2
    disp_map = np.full((H, W), d_far, dtype=np.float32)
    disp_map[:, :boundary_col] = d_near
    right_gray = _remap_by_disparity(canvas, disp_map, x0)

    strip_w = max(1, int(round(d_near - d_far)))
    occ_rng = np.random.default_rng(seed + 20_000)
    c1 = min(W, boundary_col + strip_w)
    gt_invalid = np.zeros((H, W), dtype=bool)
    right_gray = right_gray.copy()
    right_gray[:, boundary_col:c1] = occ_rng.integers(0, 255, (H, c1 - boundary_col), dtype=np.uint8)
    gt_invalid[:, boundary_col:c1] = True

    return Fixture(
        name=f"DISOCC_{near_m}m-{far_m}m_seed{seed}", scenario="DISOCC", depth_m=None,
        true_disparity_px=None, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
        gt_invalid_mask=gt_invalid, gt_disparity_map=disp_map,
    )


def make_multi_zone_fixture(zones, seed: int) -> Fixture:
    """Multiple discrete obstacle sectors: `zones` is a list of
    (col_frac_start, col_frac_end, depth_m) tuples covering [0,1) of the
    frame width, e.g. [(0.0,0.3,1.5), (0.3,0.6,4.0), (0.6,1.0,2.5)] for
    three distinct bearings/ranges. Background defaults to the last
    zone's own far value where zones don't cover [0,1) fully."""
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, _TEXTURE_SCALE["B"], seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]

    disp_map = np.full((H, W), disparity_for_depth(zones[-1][2]), dtype=np.float32)
    for c0f, c1f, depth_m in zones:
        c0, c1 = int(c0f * W), int(c1f * W)
        disp_map[:, c0:c1] = disparity_for_depth(depth_m)
    right_gray = _remap_by_disparity(canvas, disp_map, x0)

    return Fixture(
        name=f"MULTI_seed{seed}", scenario="MULTI", depth_m=None, true_disparity_px=None,
        left=_to_bgr(left_gray), right=_to_bgr(right_gray),
        gt_invalid_mask=None, gt_disparity_map=disp_map,
    )
