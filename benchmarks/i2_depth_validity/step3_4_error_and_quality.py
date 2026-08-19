"""
Phase I2, Steps 3-4 — true-error-vs-validity binning + GeometryFrameQuality
calibration. Read-only w.r.t. src/.
"""
import sys, json
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import cv2
import numpy as np

from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.calibration.loader import load_stereo_calibration
from depth_perception_engine.geometry.geometry_metrics import classify_geometry_quality, GeometryMetrics
from benchmarks.i1_stereo_accuracy.fixtures import (
    make_flat_fixture, make_discontinuity_fixture, make_decorrelated_fixture, BENCH_DEPTHS_M, FX, BASELINE_M,
)
from benchmarks.i2_depth_validity.step1_2_roi_and_gates import current_sgbm_params, disparity, W, H, _ESTIMATOR

NUM_DISP = 128
ROI = (slice(None), slice(NUM_DISP, None))
HEALTHY_MIN = 0.5
DEGRADED_MIN = 0.05


def true_err_bins(disp_map, valid_mask, true_disp, true_depth_m, roi=ROI):
    """Per-pixel disparity/depth error, binned, within ROI only."""
    vm = valid_mask[roi]
    dm = disp_map[roi]
    d_err = np.abs(dm[vm] - true_disp)
    z_meas = FX * BASELINE_M / dm[vm]
    z_err_pct = np.abs(z_meas - true_depth_m) / true_depth_m * 100.0
    bins = {"<=1%": 0, "1-3%": 0, "3-5%": 0, "5-10%": 0, ">10%": 0}
    for e in z_err_pct:
        if e <= 1: bins["<=1%"] += 1
        elif e <= 3: bins["1-3%"] += 1
        elif e <= 5: bins["3-5%"] += 1
        elif e <= 10: bins["5-10%"] += 1
        else: bins[">10%"] += 1
    n = len(z_err_pct)
    return bins, n, float(np.mean(z_err_pct)) if n else None, float(np.median(z_err_pct)) if n else None


def step3():
    params = current_sgbm_params()
    print("[STEP3] true-error distribution, A/B/C, ROI pixels only\n")
    out = []
    for scenario in ("A", "B", "C"):
        agg_bins = {"<=1%": 0, "1-3%": 0, "3-5%": 0, "5-10%": 0, ">10%": 0}
        agg_n = 0
        for depth_m in BENCH_DEPTHS_M:
            for seed in (1, 2, 3):
                fx = make_flat_fixture(scenario, depth_m=depth_m, seed=seed)
                disp = disparity(params, fx.left, fx.right)
                valid = disp > 0.0
                bins, n, mean_e, med_e = true_err_bins(disp, valid, fx.true_disparity_px, depth_m)
                for k in agg_bins: agg_bins[k] += bins[k]
                agg_n += n
        pct = {k: round(100 * v / agg_n, 3) for k, v in agg_bins.items()} if agg_n else {}
        print(f"  scenario {scenario}: n={agg_n}  bins%={pct}")
        out.append({"scenario": scenario, "n": agg_n, "bin_counts": agg_bins, "bin_pct": pct})

    # G: false-valid rate (already known from I1, reconfirm) + E occlusion strip
    g_false = []
    for seed in (1, 2, 3, 4, 5):
        fx = make_decorrelated_fixture(seed)
        disp = disparity(params, fx.left, fx.right)
        valid = disp > 0.0
        g_false.append(float(valid[ROI].mean()))
    print(f"\n  scenario G (decorrelated) false-valid-in-ROI mean = {np.mean(g_false):.4f}")

    e_outlier_check = []
    for seed in (1, 2, 3):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=True)
        disp = disparity(params, fx.left, fx.right)
        valid = disp > 0.0
        # non-occluded region: gt_invalid_mask False
        gt_valid_region = ~fx.gt_invalid_mask
        cmp_mask = gt_valid_region & valid
        gt_disp = fx.gt_disparity_map
        err = np.abs(disp[cmp_mask] - gt_disp[cmp_mask])
        z_true = FX * BASELINE_M / gt_disp[cmp_mask]
        z_meas = FX * BASELINE_M / disp[cmp_mask]
        z_err_pct = np.abs(z_meas - z_true) / z_true * 100.0
        n_high_err = int((z_err_pct > 10).sum())
        e_outlier_check.append({"seed": seed, "n_valid_nonoccluded": int(cmp_mask.sum()),
                                 "n_high_error_gt10pct": n_high_err,
                                 "pct_high_error": float(100 * n_high_err / max(cmp_mask.sum(), 1))})
    print(f"  scenario E non-occluded-region high-error(>10%) outlier check: {e_outlier_check}")

    return {"AB_C_bins": out, "G_false_valid_roi_mean": float(np.mean(g_false)), "E_outlier_check": e_outlier_check}


def step4():
    params = current_sgbm_params()
    print("\n[STEP4] GeometryFrameQuality (geometry_validity_state) calibration\n")

    cases = []
    # clean A/B at several depths -> expect near-HEALTHY (whole-frame ~0.6 ceiling)
    for scenario in ("A", "B"):
        for depth_m in [1.0, 3.0, 6.0]:
            cases.append(("clean", scenario, depth_m, None))
    # occlusion (E) -> expect DEGRADED
    cases.append(("occlusion", "E", None, (1.5, 5.0)))
    # decorrelated noise (G) -> expect NO_USABLE/INSUFFICIENT
    cases.append(("noise", "G", None, None))
    # weak texture far (C@6m) -> expect DEGRADED or less
    cases.append(("weak_far", "C", 6.0, None))

    results = []
    for kind, scenario, depth_m, extra in cases:
        for seed in (1, 2, 3):
            if kind in ("clean", "weak_far"):
                fx = make_flat_fixture(scenario, depth_m=depth_m, seed=seed)
                true_disp, true_depth = fx.true_disparity_px, depth_m
            elif kind == "occlusion":
                near_m, far_m = extra
                fx = make_discontinuity_fixture(near_m=near_m, far_m=far_m, seed=seed, occlusion=True)
                true_disp, true_depth = None, None
            else:
                fx = make_decorrelated_fixture(seed)
                true_disp, true_depth = None, None

            disp = disparity(params, fx.left, fx.right)
            depth = _ESTIMATOR.estimate(disp)
            valid_disp = disp > 0.0
            valid_depth = depth > 0.0

            whole_frame_valid_fraction = float(valid_disp.mean())  # matches GeometryMetrics basis (whole H*W)
            false_valid_fraction = None
            mean_err = median_err = p95_err = None

            if fx.gt_invalid_mask is not None and fx.gt_disparity_map is None:
                false_valid_fraction = float((valid_disp & fx.gt_invalid_mask).mean())
            elif fx.gt_disparity_map is not None:
                gt_valid_region = ~fx.gt_invalid_mask if fx.gt_invalid_mask is not None else np.ones_like(valid_disp)
                cmp_mask = gt_valid_region & valid_disp
                if fx.gt_invalid_mask is not None:
                    false_valid_fraction = float((valid_disp & fx.gt_invalid_mask).mean())
                gt_disp = fx.gt_disparity_map
                if cmp_mask.any():
                    z_true = FX * BASELINE_M / gt_disp[cmp_mask]
                    z_meas = FX * BASELINE_M / disp[cmp_mask]
                    err_pct = np.abs(z_meas - z_true) / z_true * 100.0
                    mean_err, median_err, p95_err = float(err_pct.mean()), float(np.median(err_pct)), float(np.percentile(err_pct, 95))
            elif true_disp is not None:
                if valid_disp.any():
                    z_meas = FX * BASELINE_M / disp[valid_disp]
                    err_pct = np.abs(z_meas - true_depth) / true_depth * 100.0
                    mean_err, median_err, p95_err = float(err_pct.mean()), float(np.median(err_pct)), float(np.percentile(err_pct, 95))
                false_valid_fraction = 0.0

            metrics = GeometryMetrics(min_obstacle_distance_m=None, mean_free_space_m=None,
                                       point_count=int(valid_depth.sum()), valid_fraction=float(valid_depth.mean()))
            quality_state = classify_geometry_quality(metrics, HEALTHY_MIN, DEGRADED_MIN)

            results.append({
                "kind": kind, "scenario": scenario, "depth_m": depth_m, "seed": seed,
                "whole_frame_valid_fraction": whole_frame_valid_fraction,
                "quality_state": quality_state,
                "mean_err_pct": mean_err, "median_err_pct": median_err, "p95_err_pct": p95_err,
                "false_valid_fraction": false_valid_fraction,
            })

    # aggregate by quality_state
    by_state = {}
    for r in results:
        by_state.setdefault(r["quality_state"], []).append(r)
    print("  Aggregated by quality_state:")
    summary = {}
    for state, rows in by_state.items():
        errs = [r["median_err_pct"] for r in rows if r["median_err_pct"] is not None]
        fv = [r["false_valid_fraction"] for r in rows if r["false_valid_fraction"] is not None]
        vf = [r["whole_frame_valid_fraction"] for r in rows]
        s = {
            "n": len(rows),
            "median_err_pct_mean": float(np.mean(errs)) if errs else None,
            "median_err_pct_max": float(np.max(errs)) if errs else None,
            "false_valid_mean": float(np.mean(fv)) if fv else None,
            "whole_frame_valid_fraction_range": [float(np.min(vf)), float(np.max(vf))],
        }
        summary[state] = s
        print(f"    {state:20s} n={s['n']:3d}  median_err%(mean of medians)={s['median_err_pct_mean']}  "
              f"false_valid_mean={s['false_valid_mean']}  valid_frac_range={s['whole_frame_valid_fraction_range']}")

    # headroom: max achievable whole-frame valid_fraction on a PERFECT clean scene vs HEALTHY_MIN
    perfect_ceiling = max(r["whole_frame_valid_fraction"] for r in results if r["kind"] == "clean")
    headroom = perfect_ceiling - HEALTHY_MIN
    print(f"\n  HEALTHY margin: perfect-scene whole-frame valid_fraction ceiling={perfect_ceiling:.4f}, "
          f"HEALTHY_MIN={HEALTHY_MIN}, absolute headroom={headroom:.4f} "
          f"({100*headroom/perfect_ceiling:.1f}% relative margin)")

    return {"raw": results, "by_state_summary": summary,
            "perfect_scene_ceiling": perfect_ceiling, "healthy_min": HEALTHY_MIN,
            "headroom_absolute": headroom, "headroom_relative_pct": 100 * headroom / perfect_ceiling}


if __name__ == "__main__":
    s3 = step3()
    s4 = step4()
    with open("benchmarks/i2_depth_validity/results/step3_4_error_and_quality.json", "w") as f:
        json.dump({"step3": s3, "step4": s4}, f, indent=2)
    print("\nWrote benchmarks/i2_depth_validity/results/step3_4_error_and_quality.json")
