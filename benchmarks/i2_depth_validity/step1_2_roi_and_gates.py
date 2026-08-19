"""
Phase I2, Steps 1-2 — observable ROI derivation + validity-gate rejection
accounting. Read-only w.r.t. src/. Reuses benchmarks/i1_stereo_accuracy/
fixtures.py's ground-truth generators.
"""
import sys, json
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import cv2
import numpy as np

from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.calibration.loader import load_stereo_calibration
from benchmarks.i1_stereo_accuracy.fixtures import make_flat_fixture, BENCH_DEPTHS_M, FX, BASELINE_M

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
_ESTIMATOR = DepthEstimator(_CALIB.Q)
W, H = _CALIB.image_size


def current_sgbm_params(block_size=9, num_disparities=128, disp12MaxDiff=1, uniquenessRatio=20,
                         speckleWindowSize=100, speckleRange=32, preFilterCap=63):
    return dict(
        minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
        P1=8 * 1 * block_size ** 2, P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=disp12MaxDiff, uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize, speckleRange=speckleRange,
        preFilterCap=preFilterCap, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def disparity(params, left_bgr, right_bgr):
    stereo = cv2.StereoSGBM_create(**params)
    raw = stereo.compute(to_gray(left_bgr), to_gray(right_bgr))
    return (raw / 16.0).astype(np.float32)


# ============================================================
# STEP 1 — observable ROI, empirically + theoretically
# ============================================================
def step1():
    num_disp = 128
    theoretical_frac = (W - num_disp) / W
    print(f"[STEP1] theoretical whole-frame observable fraction = {theoretical_frac:.4f}  "
          f"(W={W}, numDisparities={num_disp}, dead-zone={num_disp}px)")

    # Empirically find the leftmost column that EVER reads valid across many
    # clean high-texture fixtures/seeds, to check for any extra block-edge
    # shift beyond the naive numDisparities boundary.
    params = current_sgbm_params()
    leftmost_valid_col_seen = W  # start high, take min
    n_checked = 0
    for depth_m in [0.5, 2.0, 6.0]:
        for seed in range(1, 6):
            fx = make_flat_fixture("A", depth_m=depth_m, seed=seed)
            disp = disparity(params, fx.left, fx.right)
            valid_cols = np.where((disp > 0).any(axis=0))[0]
            if valid_cols.size:
                leftmost_valid_col_seen = min(leftmost_valid_col_seen, int(valid_cols.min()))
            n_checked += 1
    print(f"[STEP1] empirical leftmost-ever-valid column across {n_checked} clean fixtures: "
          f"{leftmost_valid_col_seen}  (theoretical dead-zone boundary: {num_disp})")

    roi_col_start = max(leftmost_valid_col_seen, 0)
    roi_area = H * (W - roi_col_start)
    print(f"[STEP1] adopted ROI: columns [{roi_col_start}:{W}), area={roi_area}px "
          f"({roi_area/(H*W):.4f} of whole frame)")

    # full A/B/C x 7-depth x 3-seed sweep: whole-frame vs ROI-valid
    rows = []
    for scenario in ("A", "B", "C"):
        for depth_m in BENCH_DEPTHS_M:
            whole_vals, roi_vals = [], []
            for seed in (1, 2, 3):
                fx = make_flat_fixture(scenario, depth_m=depth_m, seed=seed)
                disp = disparity(params, fx.left, fx.right)
                valid = disp > 0.0
                whole_vals.append(float(valid.mean()))
                roi_vals.append(float(valid[:, roi_col_start:].mean()))
            row = {
                "scenario": scenario, "depth_m": depth_m,
                "whole_frame_valid_mean": float(np.mean(whole_vals)),
                "roi_valid_mean": float(np.mean(roi_vals)),
            }
            rows.append(row)
            print(f"  {scenario} @ {depth_m:>4.1f}m: whole={row['whole_frame_valid_mean']:.4f}  "
                  f"ROI={row['roi_valid_mean']:.4f}")
    return {
        "theoretical_whole_frame_observable_fraction": theoretical_frac,
        "empirical_leftmost_valid_col": leftmost_valid_col_seen,
        "roi_col_start": roi_col_start,
        "roi_area_px": roi_area,
        "roi_area_fraction_of_whole_frame": roi_area / (H * W),
        "sweep": rows,
    }


# ============================================================
# STEP 2 — validity gate ablation
# ============================================================
def step2():
    full = current_sgbm_params()
    ablations = {
        "FULL_CONFIG": full,
        "disp12MaxDiff_DISABLED(-1)": {**full, "disp12MaxDiff": -1},
        "uniquenessRatio_DISABLED(0)": {**full, "uniquenessRatio": 0},
        "speckle_DISABLED(0)": {**full, "speckleWindowSize": 0},
        "preFilterCap_RELAXED(0)": {**full, "preFilterCap": 0},
    }

    results = {}
    for label, fx_kind, depth_m in [("high_texture_B_2m", "B", 2.0), ("weak_texture_C_2m", "C", 2.0)]:
        pass

    out = []
    for fx_label, scenario, depth_m in [("B_2m_moderate_texture", "B", 2.0), ("C_2m_weak_texture", "C", 2.0)]:
        fx = make_flat_fixture(scenario, depth_m=depth_m, seed=1)
        num_disp = full["numDisparities"]
        roi_slice = (slice(None), slice(num_disp, None))
        roi_area = H * (W - num_disp)

        masks = {}
        depths = {}
        for label, params in ablations.items():
            disp = disparity(params, fx.left, fx.right)
            depth = _ESTIMATOR.estimate(disp)
            masks[label] = disp > 0.0
            depths[label] = depth

        full_mask_roi = masks["FULL_CONFIG"][roi_slice]
        full_before = int(full_mask_roi.size)  # observable ROI pixel count as "before" baseline population
        print(f"\n[STEP2] fixture={fx_label}  observable-ROI pixel count={full_before}")
        print(f"  FULL_CONFIG valid-in-ROI = {int(full_mask_roi.sum())} "
              f"({100*full_mask_roi.mean():.2f}% of ROI)")

        row = {"fixture": fx_label, "roi_pixel_count": full_before,
               "full_config_valid_in_roi": int(full_mask_roi.sum()),
               "full_config_valid_pct_of_roi": float(100 * full_mask_roi.mean())}

        for label in ablations:
            if label == "FULL_CONFIG":
                continue
            ablated_mask_roi = masks[label][roi_slice]
            # pixels this gate ALONE additionally rejects: valid under
            # ablation (gate off) but invalid under full config (gate on) —
            # i.e. what THIS gate removes, holding all others at their real
            # setting.
            recovered_if_disabled = ablated_mask_roi & (~full_mask_roi)
            n_recovered = int(recovered_if_disabled.sum())
            pct_roi = 100 * n_recovered / full_before
            row[f"{label}__pixels_this_gate_rejects"] = n_recovered
            row[f"{label}__pct_of_roi"] = pct_roi
            print(f"  {label:32s} -> this gate alone rejects {n_recovered:>6d} px "
                  f"({pct_roi:5.2f}% of ROI)")
        out.append(row)

        # depth-stage gates: disp<=0 sentinel additional loss beyond SGBM's
        # own invalid marking, and MIN/MAX depth clamp binding-ness
        disp_full = disparity(full, fx.left, fx.right)
        depth_full = _ESTIMATOR.estimate(disp_full)
        disp_valid = disp_full > 0.0
        depth_valid = depth_full > 0.0
        extra_depth_rejections = disp_valid & (~depth_valid)
        n_extra = int(extra_depth_rejections[roi_slice].sum())
        print(f"  depth-stage extra rejections beyond disp<=0 (MIN/MAX_DEPTH_M clamp, "
              f"finite checks) in ROI: {n_extra} px "
              f"({100*n_extra/full_before:.4f}% of ROI)")
        row["depth_stage_extra_rejections_in_roi"] = n_extra
        row["depth_stage_extra_rejections_pct_of_roi"] = float(100 * n_extra / full_before)

    return out


if __name__ == "__main__":
    step1_result = step1()
    step2_result = step2()
    with open("benchmarks/i2_depth_validity/results/step1_2_roi_and_gates.json", "w") as f:
        json.dump({"step1_roi": step1_result, "step2_gates": step2_result}, f, indent=2)
    print("\nWrote benchmarks/i2_depth_validity/results/step1_2_roi_and_gates.json")
