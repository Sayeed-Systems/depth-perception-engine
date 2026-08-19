"""
Phase I3 — deterministic fixtures for occlusion-boundary contamination /
local evidence safety. Reuses benchmarks/i1_stereo_accuracy/fixtures.py's
proven generators (A/B/E below) directly; adds the three constructions I1's
fixtures.py does not already cover (C dis-occlusion far-side, D mixed
valid/invalid cell, F genuine two-sided obstacle boundary), using the exact
same low-frequency-canvas / subpixel-remap technique for consistency.

No i.i.d. per-pixel noise used for "genuine texture" regions anywhere here
(confirmed elsewhere in this repo: it defeats real StereoSGBM correspondence
entirely) — i.i.d. noise is used ONLY where the ground truth explicitly
wants zero true correspondence (scenario E's occlusion strip, scenario G).
"""
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from benchmarks.i1_stereo_accuracy.fixtures import (
    Fixture, W, H, FX, BASELINE_M, _MAX_SHIFT_MARGIN, _TEXTURE_SCALE,
    _low_freq_canvas, _remap_by_disparity, _to_bgr,
    disparity_for_depth, make_discontinuity_fixture, make_flat_fixture, make_decorrelated_fixture,
)


def make_A_clean_discontinuity(seed, near_m=1.5, far_m=5.0):
    """A. Clean depth discontinuity, no occlusion (= I1's scenario D)."""
    return make_discontinuity_fixture(near_m, far_m, seed, occlusion=False)


def make_B_occlusion_strip(seed, near_m=1.5, far_m=5.0):
    """B. Controlled occlusion strip (= I1's scenario E) — strip on the
    NEAR side immediately left of the boundary, right-image content there
    is uncorrelated noise, gt_invalid_mask=True there."""
    return make_discontinuity_fixture(near_m, far_m, seed, occlusion=True)


def make_C_disocclusion_far_side(seed, near_m=1.5, far_m=5.0):
    """C. Dis-occlusion strip on the FAR side, immediately RIGHT of the
    boundary (the symmetric counterpart of B's near-side strip — occurs
    when the near surface is to the right of the far surface instead of
    the left, i.e. the mirror-image scene). Same width formula
    (round(d_near - d_far)), same noise-injection mechanism as B, just the
    near/far column assignment and strip placement are mirrored left<->right
    relative to B so the strip sits on the FAR plateau's own near-boundary
    edge instead of the near plateau's."""
    d_near = disparity_for_depth(near_m)
    d_far = disparity_for_depth(far_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, _TEXTURE_SCALE["B"], seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]

    boundary_col = W // 2
    # Mirror of make_discontinuity_fixture: near is on the RIGHT, far on the LEFT.
    disp_map = np.full((H, W), d_near, dtype=np.float32)
    disp_map[:, :boundary_col] = d_far
    right_gray = _remap_by_disparity(canvas, disp_map, x0)

    strip_w = max(1, int(round(d_near - d_far)))
    occ_rng = np.random.default_rng(seed + 20_000)
    gt_invalid = np.zeros((H, W), dtype=bool)
    # Strip immediately RIGHT of the boundary, within the near (right-side)
    # plateau — the mirror-image placement of B's near-side strip.
    c1 = min(W, boundary_col + strip_w)
    right_gray = right_gray.copy()
    right_gray[:, boundary_col:c1] = occ_rng.integers(0, 255, (H, c1 - boundary_col), dtype=np.uint8)
    gt_invalid[:, boundary_col:c1] = True

    return Fixture(
        name=f"C_disocclusion_{near_m}m-{far_m}m_seed{seed}", scenario="C", depth_m=None,
        true_disparity_px=None, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
        gt_invalid_mask=gt_invalid, gt_disparity_map=disp_map,
    )


def make_D_mixed_valid_invalid_cell(seed, depth_m=2.0):
    """D. Mixed valid/invalid cell — left half of the frame is genuine,
    correlated texture at depth_m (like scenario A); right half is FLAT
    uniform grey (i.i.d.-noise-free but textureless) in BOTH eyes, which
    real StereoSGBM cannot match at all (no gradient anywhere -> rejected
    by uniquenessRatio/cost-ambiguity, not by construction — ground truth:
    the flat half should read invalid, not because we mark it so, but
    because SGBM genuinely cannot correlate a uniform region). This
    specifically tests a grid cell straddling a genuinely-valid region and
    a genuinely-can't-be-matched (not occluded, not noise) region."""
    d = disparity_for_depth(depth_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, _TEXTURE_SCALE["A"], seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W].copy()
    disp_map = np.full((H, W), d, dtype=np.float32)
    right_gray = _remap_by_disparity(canvas, disp_map, x0).copy()

    half = W // 2
    left_gray[:, half:] = 128.0
    right_gray[:, half:] = 128.0
    gt_invalid = np.zeros((H, W), dtype=bool)
    gt_invalid[:, half:] = True  # ground truth: no genuine texture to match here

    return Fixture(
        name=f"D_mixed_cell_{depth_m}m_seed{seed}", scenario="D", depth_m=None,
        true_disparity_px=None, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
        gt_invalid_mask=gt_invalid, gt_disparity_map=disp_map,
    )


def make_F_genuine_obstacle_boundary(seed, near_m=2.0, far_m=5.0, near_col_lo=100, near_col_hi=220):
    """F. Genuine, both-sides-supported obstacle boundary — a compact near
    "box" spanning columns [near_col_lo, near_col_hi) at near_m, against a
    far background at far_m everywhere else. TWO real transitions (left
    edge and right edge of the box), each occlusion-free (a fronto-parallel
    box directly facing the camera has no dis-occlusion strip of its own
    at either edge in this simplified construction — the occlusion strip
    formation requires the SIGN of the near/far transition, i.e. only one
    of the two box edges would realistically have one; kept fully clean
    here since this fixture's purpose is a clean recall/precision positive
    control, not another occlusion test)."""
    d_near = disparity_for_depth(near_m)
    d_far = disparity_for_depth(far_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, _TEXTURE_SCALE["B"], seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]

    disp_map = np.full((H, W), d_far, dtype=np.float32)
    disp_map[:, near_col_lo:near_col_hi] = d_near
    right_gray = _remap_by_disparity(canvas, disp_map, x0)

    return Fixture(
        name=f"F_obstacle_box_{near_m}m-{far_m}m_seed{seed}", scenario="F", depth_m=None,
        true_disparity_px=None, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
        gt_invalid_mask=np.zeros((H, W), dtype=bool), gt_disparity_map=disp_map,
    )


def make_E_decorrelated(seed):
    """E. Pure decorrelated negative case (= I1's scenario G)."""
    return make_decorrelated_fixture(seed)
