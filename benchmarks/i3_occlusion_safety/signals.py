"""
Phase I3, Step 2 — candidate local-contamination-signal evaluation. Purely
data-driven (disparity map + valid mask only, no optical flow/neural/new
sensors). Zero src/ changes.
"""
import json
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine.stereo.disparity_engine import DisparityEngine
from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.calibration.loader import load_stereo_calibration

from benchmarks.i3_occlusion_safety.fixtures import (
    make_A_clean_discontinuity, make_B_occlusion_strip, make_C_disocclusion_far_side,
    make_D_mixed_valid_invalid_cell, make_F_genuine_obstacle_boundary, W, H,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
_DISP_ENGINE = DisparityEngine(min_disparity=0, num_disparities=128, block_size=9)
_ESTIMATOR = DepthEstimator(_CALIB.Q)


def _compute(fx):
    raw, _ = _DISP_ENGINE.compute_disparity(fx.left, fx.right, compute_visualization=False)
    valid = raw > 0.0
    return raw, valid


# --------------------------------------------------------------------
# Candidate (i): shadow-zone / gradient-proportional masking
# --------------------------------------------------------------------
def shadow_zone_mask(raw_disparity, valid, grad_threshold_px=3.0, width_scale=1.0, side="left_of_drop"):
    """Per-row: compute column-wise diff of raw_disparity where BOTH sides
    are valid. Wherever diff[:, c] = disp[c+1]-disp[c] is very negative
    (a drop from high/near to low/far moving rightward) with magnitude
    >= grad_threshold_px, flag a zone of width round(width_scale*|diff|)
    columns ending at column c (i.e. the columns [c-width+1, c], the
    NEAR/high-disparity side immediately before the drop) as "shadow".
    side="right_of_rise" flags the mirror case (a rise from low to high
    moving rightward -> shadow on the columns just AFTER the rise, the
    far/low side of that transition, which is the geometry for a
    near-on-the-right / far-on-the-left step, i.e. fixture C)."""
    h, w = raw_disparity.shape
    mask = np.zeros((h, w), dtype=bool)
    diff = np.diff(raw_disparity, axis=1)  # (h, w-1): diff[:,c] = disp[c+1]-disp[c]
    both_valid = valid[:, :-1] & valid[:, 1:]
    for r in range(h):
        for c in range(w - 1):
            if not both_valid[r, c]:
                continue
            d = diff[r, c]
            if side == "left_of_drop" and d <= -grad_threshold_px:
                width = max(1, int(round(width_scale * abs(d))))
                c0 = max(0, c - width + 1)
                mask[r, c0:c + 1] = True
            elif side == "right_of_rise" and d >= grad_threshold_px:
                width = max(1, int(round(width_scale * abs(d))))
                c1 = min(w, c + 1 + width)
                mask[r, c + 1:c1] = True
    return mask


def eval_shadow_zone(fx, strip_cols, side, grad_threshold_px=3.0, width_scale=1.0):
    raw, valid = _compute(fx)
    mask = shadow_zone_mask(raw, valid, grad_threshold_px, width_scale, side)
    gt_bad = fx.gt_invalid_mask if fx.gt_invalid_mask is not None else np.zeros((H, W), dtype=bool)
    flagged_and_bad = mask & gt_bad
    flagged_and_good = mask & valid & ~gt_bad
    tp = int(flagged_and_bad.sum())
    fn = int((gt_bad & valid & ~mask).sum())
    fp = int(flagged_and_good.sum())
    total_valid_good = int((valid & ~gt_bad).sum())
    return {
        "tp": tp, "fn": fn, "fp": fp,
        "recall_on_strip": tp / max(int((gt_bad & valid).sum()), 1),
        "fp_rate_on_clean_valid": fp / max(total_valid_good, 1),
        "mask_total_flagged": int(mask.sum()),
    }


# --------------------------------------------------------------------
# Candidate (ii): per-cell coherence / outlier rejection (3x3 grid, same
# boundaries BoundaryEvidence uses: np.linspace(0,W,4)/np.linspace(0,H,4))
# --------------------------------------------------------------------
def cell_bounds(grid_n, size):
    b = np.linspace(0, size, grid_n + 1).astype(int)
    return list(zip(b[:-1], b[1:]))


def eval_cell_coherence(fx, row_range, col_range, tolerance_mad_mult=3.0):
    raw, valid = _compute(fx)
    r0, r1 = row_range
    c0, c1 = col_range
    cell_disp = raw[r0:r1, c0:c1]
    cell_valid = valid[r0:r1, c0:c1]
    vals = cell_disp[cell_valid]
    if vals.size == 0:
        return {"n_valid": 0, "flagged_fraction": None}
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    incoherent = np.abs(vals - med) > tolerance_mad_mult * mad
    return {
        "n_valid": int(vals.size), "median_disp": float(med), "mad": float(mad),
        "flagged_fraction": float(incoherent.mean()),
    }


# --------------------------------------------------------------------
# Candidate (iii): local invalid-neighbor density
# --------------------------------------------------------------------
def invalid_neighbor_density(valid, win=5):
    h, w = valid.shape
    pad = win // 2
    padded = np.pad((~valid).astype(np.float32), pad, mode="constant", constant_values=1.0)
    # box filter via cumulative sum for speed
    cs = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    out = np.zeros((h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            r2, c2 = r + win, c + win
            total = cs[r2, c2] - cs[r, c2] - cs[r2, c] + cs[r, c]
            out[r, c] = total / (win * win)
    return out


def eval_invalid_neighbor_density(fx, strip_cols, win=5):
    raw, valid = _compute(fx)
    density = invalid_neighbor_density(valid, win)
    gt_bad = fx.gt_invalid_mask if fx.gt_invalid_mask is not None else np.zeros((H, W), dtype=bool)
    strip_mask = gt_bad & valid
    interior_mask = valid & ~gt_bad
    return {
        "mean_density_on_strip": float(density[strip_mask].mean()) if strip_mask.any() else None,
        "mean_density_on_clean_interior": float(density[interior_mask].mean()) if interior_mask.any() else None,
    }


def main():
    out = {}

    fx_a = make_A_clean_discontinuity(1)
    fx_b = make_B_occlusion_strip(1)
    fx_c = make_C_disocclusion_far_side(1)
    fx_f = make_F_genuine_obstacle_boundary(1)

    # --- (i) shadow zone: verify side empirically first ---
    out["i_shadow_zone_side_check"] = {
        "B_left_of_drop": eval_shadow_zone(fx_b, None, "left_of_drop"),
        "B_right_of_rise": eval_shadow_zone(fx_b, None, "right_of_rise"),
        "C_left_of_drop": eval_shadow_zone(fx_c, None, "left_of_drop"),
        "C_right_of_rise": eval_shadow_zone(fx_c, None, "right_of_rise"),
    }

    # apply the correct side per fixture based on the check above, plus
    # false-positive rate on CLEAN fixtures A and F (which have no true
    # occlusion strip at all -> any flag there is a false positive)
    out["i_shadow_zone_main"] = {
        "B_occlusion_correct_side": eval_shadow_zone(fx_b, None, "left_of_drop", grad_threshold_px=3.0, width_scale=1.0),
        "C_disocclusion_correct_side": eval_shadow_zone(fx_c, None, "right_of_rise", grad_threshold_px=3.0, width_scale=1.0),
        "A_clean_fp_check_left": eval_shadow_zone(fx_a, None, "left_of_drop", grad_threshold_px=3.0, width_scale=1.0),
        "A_clean_fp_check_right": eval_shadow_zone(fx_a, None, "right_of_rise", grad_threshold_px=3.0, width_scale=1.0),
        "F_clean_fp_check_left": eval_shadow_zone(fx_f, None, "left_of_drop", grad_threshold_px=3.0, width_scale=1.0),
        "F_clean_fp_check_right": eval_shadow_zone(fx_f, None, "right_of_rise", grad_threshold_px=3.0, width_scale=1.0),
    }
    # threshold sensitivity
    out["i_shadow_zone_threshold_sweep"] = {}
    for thr in (1.0, 2.0, 3.0, 5.0):
        out["i_shadow_zone_threshold_sweep"][f"thr_{thr}"] = {
            "B_recall": eval_shadow_zone(fx_b, None, "left_of_drop", grad_threshold_px=thr)["recall_on_strip"],
            "A_fp_rate": eval_shadow_zone(fx_a, None, "left_of_drop", grad_threshold_px=thr)["fp_rate_on_clean_valid"],
            "F_fp_rate": eval_shadow_zone(fx_f, None, "left_of_drop", grad_threshold_px=thr)["fp_rate_on_clean_valid"],
        }

    # --- (ii) per-cell coherence, 3x3 grid ---
    rb = cell_bounds(3, H)
    cb = cell_bounds(3, W)
    # cell (0,1) = row block 0, col block 1 -> [107,213)
    r0, r1 = rb[0]
    c0, c1 = cb[1]
    out["ii_cell_coherence"] = {
        "A_straddling_cell(no_occlusion)": eval_cell_coherence(fx_a, (r0, r1), (c0, c1)),
        "B_straddling_cell(occlusion)": eval_cell_coherence(fx_b, (r0, r1), (c0, c1)),
        "A_pure_near_cell(col0)": eval_cell_coherence(fx_a, (r0, r1), cb[0]),
        "A_pure_far_cell(col2)": eval_cell_coherence(fx_a, (r0, r1), cb[2]),
    }

    # --- (iii) invalid-neighbor density ---
    out["iii_invalid_neighbor_density"] = {
        "B_occlusion": eval_invalid_neighbor_density(fx_b, None),
        "A_clean": eval_invalid_neighbor_density(fx_a, None),
    }

    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i3_occlusion_safety/results/signals.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
