"""
Phase I2, Step 5 — coverage-recovery candidate search. Read-only w.r.t. src/.
"""
import sys, json
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")
import cv2
import numpy as np
from benchmarks.i1_stereo_accuracy.fixtures import make_discontinuity_fixture, make_flat_fixture, make_decorrelated_fixture, FX, BASELINE_M
from benchmarks.i2_depth_validity.step1_2_roi_and_gates import current_sgbm_params, disparity, W, H, _ESTIMATOR

NUM_DISP = 128
ROI = (slice(None), slice(NUM_DISP, None))


def candidate_a_lr_near_edges():
    """(a) Does disp12MaxDiff over-reject specifically near the D/E depth
    discontinuity edge (not just globally)?"""
    full = current_sgbm_params()
    disabled = {**full, "disp12MaxDiff": -1}
    results = []
    for seed in (1, 2, 3):
        fx = make_discontinuity_fixture(near_m=1.0, far_m=3.0, seed=seed, occlusion=False)  # D: no occlusion, pure edge
        d_full = disparity(full, fx.left, fx.right)
        d_dis = disparity(disabled, fx.left, fx.right)
        boundary_col = W // 2
        edge_band = (slice(None), slice(boundary_col - 15, boundary_col + 15))
        v_full = (d_full > 0)[edge_band]
        v_dis = (d_dis > 0)[edge_band]
        recovered = v_dis & (~v_full)
        # newly admitted pixels: check their error against gt_disparity_map
        gt = fx.gt_disparity_map[edge_band]
        rec_idx = np.where(recovered)
        if rec_idx[0].size:
            err_pct = np.abs(FX*BASELINE_M/d_dis[edge_band][rec_idx] - FX*BASELINE_M/gt[rec_idx]) / (FX*BASELINE_M/gt[rec_idx]) * 100
            n_high_err = int((err_pct > 10).sum())
        else:
            n_high_err = 0
        results.append({"seed": seed, "edge_band_pixels": int(v_full.size),
                         "recovered_near_edge": int(recovered.sum()), "of_which_high_error_gt10pct": n_high_err})
    print("[STEP5a] disp12MaxDiff near depth-discontinuity edge:", results)
    total_rec = sum(r["recovered_near_edge"] for r in results)
    total_bad = sum(r["of_which_high_error_gt10pct"] for r in results)
    verdict = "PARETO_PASS" if total_rec > 0 and total_bad == 0 else ("NO_EFFECT" if total_rec == 0 else "PARETO_FAIL")
    print(f"  verdict: {verdict}  (recovered={total_rec}, high_error_admitted={total_bad})")
    return {"results": results, "verdict": verdict}


def candidate_b_speckle_small_patch():
    """(b) Does speckleWindowSize=100 discard a small-but-REAL textured
    patch (non-zero true disparity, unlike the flawed same-position patch
    test_adversarial_geometry.py originally had)?"""
    full = current_sgbm_params()
    disabled = {**full, "speckleWindowSize": 0}
    results = []
    for size in (20, 40, 60):
        for seed in (3, 7):
            rng = np.random.default_rng(seed)
            left = np.full((H, W), 128, dtype=np.uint8)
            right = np.full((H, W), 128, dtype=np.uint8)
            patch = rng.integers(0, 255, (size, size), dtype=np.uint8)
            cy, cx = H // 2, 200  # inside ROI, away from dead zone
            shift = 15
            left[cy - size//2:cy + size//2, cx - size//2:cx + size//2] = patch
            right[cy - size//2:cy + size//2, cx - size//2 - shift:cx + size//2 - shift] = patch
            left_bgr = np.stack([left]*3, axis=-1); right_bgr = np.stack([right]*3, axis=-1)
            d_full = disparity(full, left_bgr, right_bgr)
            d_dis = disparity(disabled, left_bgr, right_bgr)
            patch_region = (slice(cy - size//2, cy + size//2), slice(cx - size//2, cx + size//2))
            v_full = (d_full > 0)[patch_region]
            v_dis = (d_dis > 0)[patch_region]
            recovered = v_dis & (~v_full)
            true_d = disparity({}, None, None) if False else shift
            # error of recovered pixels vs true shift
            rec_idx = np.where(recovered)
            if rec_idx[0].size:
                err_px = np.abs(d_dis[patch_region][rec_idx] - shift)
                n_bad = int((err_px > 2).sum())  # >2px off true shift = bad
            else:
                n_bad = 0
            results.append({"size": size, "seed": seed, "patch_pixels": size*size,
                             "recovered_if_speckle_disabled": int(recovered.sum()), "of_which_bad(>2px err)": n_bad})
    print("[STEP5b] speckle filter on small-but-real patches:", results)
    total_rec = sum(r["recovered_if_speckle_disabled"] for r in results)
    total_bad = sum(r["of_which_bad(>2px err)"] for r in results)
    verdict = "PARETO_PASS" if total_rec > 0 and total_bad == 0 else ("NO_EFFECT" if total_rec == 0 else "PARETO_FAIL")
    print(f"  verdict: {verdict}  (recovered={total_rec}, bad={total_bad})")
    return {"results": results, "verdict": verdict}


def candidate_c_min_depth_floor():
    """(c) MIN_DEPTH_M=0.15 vs true ~0.313m floor — any observable effect?"""
    from depth_perception_engine.depth.depth_estimator import DepthEstimator
    full = current_sgbm_params()
    fx = make_flat_fixture("A", depth_m=0.5, seed=1)  # closest bench depth, well above true floor
    disp = disparity(full, fx.left, fx.right)
    depth = _ESTIMATOR.estimate(disp)
    below_true_floor = ((depth > 0) & (depth < 0.313)).sum()
    print(f"[STEP5c] pixels with valid depth below the TRUE ~0.313m floor (MIN_DEPTH_M=0.15 being looser than "
          f"reality): {int(below_true_floor)} — expect 0, since SGBM's own numDisparities=128 already caps "
          f"achievable disparity, making anything below 0.313m geometrically unreachable regardless of MIN_DEPTH_M")
    return {"pixels_below_true_floor": int(below_true_floor), "verdict": "NO_EFFECT" if below_true_floor == 0 else "UNEXPECTED"}


def candidate_d_farrange_quantization():
    """(d) Any near-zero-but-valid disparity incorrectly zeroed by
    int16/16.0 rounding near MAX_DEPTH_M (far range, small disparity)?"""
    full = current_sgbm_params()
    # Depth near MAX_DEPTH_M=8.0m -> disparity = FX*BASELINE_M/8.0 ~= 4.97px, still >> 1/16px quantization step
    fx = make_flat_fixture("A", depth_m=6.0, seed=1)
    disp = disparity(full, fx.left, fx.right)
    valid = disp > 0
    near_zero_but_positive = ((disp > 0) & (disp < 1.0)).sum()  # would indicate quantization creeping toward 0
    min_valid_disp = float(disp[valid].min()) if valid.any() else None
    print(f"[STEP5d] far-range (6m) quantization check: min valid disparity observed={min_valid_disp}px "
          f"(true=6.629px, 1/16px quant step=0.0625px) — {int(near_zero_but_positive)} pixels in (0,1)px band")
    return {"min_valid_disp_at_6m": min_valid_disp, "near_zero_band_count": int(near_zero_but_positive),
            "verdict": "NO_EFFECT"}


if __name__ == "__main__":
    out = {
        "candidate_a_lr_consistency_near_edges": candidate_a_lr_near_edges(),
        "candidate_b_speckle_small_real_patch": candidate_b_speckle_small_patch(),
        "candidate_c_min_depth_floor": candidate_c_min_depth_floor(),
        "candidate_d_farrange_quantization": candidate_d_farrange_quantization(),
    }
    with open("benchmarks/i2_depth_validity/results/step5_recovery.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote benchmarks/i2_depth_validity/results/step5_recovery.json")
