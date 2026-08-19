"""
Phase I5 Part B — genuine-gap ("near-far-near") fixture generator plus
reuse of I1/I4's own proven low-frequency-canvas/disparity-remap technique.
A gap is the OPPOSITE construction from I4's make_narrow_obstacle_fixture
(near box flanked by far background): here a FAR region (the opening) is
flanked by NEAR walls on one or both sides.
"""
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from benchmarks.i1_stereo_accuracy.fixtures import (
    W, H, FX, BASELINE_M, _MAX_SHIFT_MARGIN, _low_freq_canvas, _remap_by_disparity, _to_bgr,
    disparity_for_depth, Fixture,
)


def make_gap_fixture(near_left_m, gap_m, near_right_m, seed, gap_cols, texture_scale=6,
                      invalid_gap_cols=None):
    """near_left_m: depth of the left flank wall (None = flank absent,
    frame starts directly in the gap — for image-edge-truncation cases).
    gap_m: depth of the gap/far region.
    near_right_m: depth of the right flank wall (None = absent).
    gap_cols: (c0, c1) PIXEL column bounds of the gap — caller's
    responsibility to align these with the actual boundary/opening grid's
    own cell boundaries (np.linspace(0, W, grid_cols+1)), since a gap
    that straddles a grid cell along with flank content produces a
    contaminated median, not a clean flank/gap read (this dependency is
    real and load-bearing, not an implementation detail to hide).
    invalid_gap_cols: optional (a,b) PIXEL sub-range (within gap_cols) to
    force textureless/invalid inside the gap (partial-invalid-support case).

    Ground truth: gap width_m = (c1-c0 px) * gap_m / FX (pinhole,
    matching build_opening_evidence's own approx_width_m formula exactly,
    so 'ground truth' and 'measured' use the identical projection model —
    isolating STEREO/GRID error, not a formula mismatch); range_m = gap_m.
    """
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, texture_scale, seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]

    c0, c1 = gap_cols

    d_gap = disparity_for_depth(gap_m)
    disp_map = np.full((H, W), d_gap, dtype=np.float32)
    if near_left_m is not None:
        disp_map[:, :c0] = disparity_for_depth(near_left_m)
    if near_right_m is not None:
        disp_map[:, c1:] = disparity_for_depth(near_right_m)

    right_gray = _remap_by_disparity(canvas, disp_map, x0)

    gt_invalid = np.zeros((H, W), dtype=bool)
    if invalid_gap_cols is not None:
        ic0, ic1 = invalid_gap_cols
        right_gray = right_gray.copy()
        right_gray[:, ic0:ic1] = 128  # flat/textureless -> no correspondence
        left_gray_mod = left_gray.copy()
        left_gray_mod[:, ic0:ic1] = 128
        left_gray = left_gray_mod
        gt_invalid[:, ic0:ic1] = True

    width_m_truth = (c1 - c0) * gap_m / FX
    return Fixture(
        name=f"gap_{near_left_m}-{gap_m}-{near_right_m}_seed{seed}", scenario="OPENING", depth_m=gap_m,
        true_disparity_px=d_gap, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
        gt_invalid_mask=gt_invalid if invalid_gap_cols else None, gt_disparity_map=disp_map,
    ), {"width_m": width_m_truth, "range_m": gap_m, "col_span": (c0, c1)}
