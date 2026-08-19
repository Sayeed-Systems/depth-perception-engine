"""
Phase I1 — measurement collection. Runs a given SGBM parameter set against
every fixtures.py fixture through the REAL, unmodified cv2.StereoSGBM (via a
thin local wrapper that mirrors stereo/disparity_engine.py's own
construction exactly, so a candidate config can be swapped in without
touching src/) plus the REAL, unmodified DepthEstimator for depth
conversion. Read-only w.r.t. src/depth_perception_engine/ — no source file
is imported in write mode or monkeypatched.
"""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.calibration.loader import load_stereo_calibration

from benchmarks.i1_stereo_accuracy.fixtures import build_all_fixtures, FX, BASELINE_M

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
_ESTIMATOR = DepthEstimator(_CALIB.Q)


# Exact current defaults, stereo/disparity_engine.py:52-66 (as of this
# session's IA0 audit) — the "CURRENT" baseline candidate.
def current_sgbm_params(block_size: int = 13, num_disparities: int = 128):
    return dict(
        minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
        P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def make_stereo(params: dict):
    return cv2.StereoSGBM_create(**params)


def _to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def compute_disparity(stereo, left_bgr, right_bgr):
    left_gray = _to_gray(left_bgr)
    right_gray = _to_gray(right_bgr)
    t0 = time.perf_counter()
    raw = stereo.compute(left_gray, right_gray)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    disp = (raw / 16.0).astype(np.float32)
    return disp, dt_ms


def measure_fixture(stereo, fx):
    disp, dt_ms = compute_disparity(stereo, fx.left, fx.right)
    depth = _ESTIMATOR.estimate(disp)

    valid_disp = disp > 0.0
    valid_depth = depth > 0.0

    result = {
        "name": fx.name, "scenario": fx.scenario, "depth_m": fx.depth_m,
        "true_disparity_px": fx.true_disparity_px,
        "latency_ms": dt_ms,
        "valid_disparity_fraction": float(valid_disp.mean()),
        "valid_depth_fraction": float(valid_depth.mean()),
    }

    if fx.gt_invalid_mask is not None and fx.gt_disparity_map is None:
        # Pure negative case (G) or a fixture with an explicit
        # ground-truth-invalid region and no per-pixel expected disparity.
        gt_valid_mask = ~fx.gt_invalid_mask
        false_valid = valid_disp & fx.gt_invalid_mask
        result["false_valid_fraction"] = float(false_valid.mean())
        if gt_valid_mask.any():
            d_true = fx.true_disparity_px
        result["disparity_abs_error"] = None
        result["depth_abs_error_m"] = None
        result["depth_rel_error_pct"] = None
        return result

    if fx.gt_disparity_map is not None:
        # D/E: per-pixel ground truth disparity; compare only where a
        # genuine correspondence should exist (gt_invalid_mask False).
        gt_valid = ~fx.gt_invalid_mask if fx.gt_invalid_mask is not None else np.ones_like(valid_disp)
        cmp_mask = gt_valid & valid_disp
        false_valid = valid_disp & (fx.gt_invalid_mask if fx.gt_invalid_mask is not None else np.zeros_like(valid_disp))
        result["false_valid_fraction"] = float(false_valid.mean()) if fx.gt_invalid_mask is not None else 0.0
        if cmp_mask.any():
            err = np.abs(disp[cmp_mask] - fx.gt_disparity_map[cmp_mask])
            result["disparity_abs_error"] = float(np.median(err))
            result["disparity_mean_err"] = float(err.mean())
            result["disparity_std_err"] = float(err.std())
        else:
            result["disparity_abs_error"] = None
        # depth relative error over the same comparison region
        gt_depth_map = np.where(fx.gt_disparity_map > 0, FX * BASELINE_M / np.maximum(fx.gt_disparity_map, 1e-6), 0.0)
        if cmp_mask.any() and valid_depth[cmp_mask].any():
            dm = cmp_mask & valid_depth
            derr = np.abs(depth[dm] - gt_depth_map[dm])
            rel = derr / gt_depth_map[dm]
            result["depth_abs_error_m"] = float(np.median(derr))
            result["depth_rel_error_pct"] = float(np.median(rel) * 100.0)
        else:
            result["depth_abs_error_m"] = None
            result["depth_rel_error_pct"] = None
        return result

    # A/B/C/F: uniform true disparity everywhere.
    d_true = fx.true_disparity_px
    z_true = fx.depth_m
    result["false_valid_fraction"] = 0.0  # not applicable — whole frame should be valid

    if valid_disp.any():
        dvals = disp[valid_disp]
        result["disparity_median"] = float(np.median(dvals))
        result["disparity_mean"] = float(dvals.mean())
        result["disparity_std"] = float(dvals.std())
        result["disparity_abs_error"] = float(abs(np.median(dvals) - d_true))
    else:
        result["disparity_median"] = None
        result["disparity_abs_error"] = None

    if valid_depth.any():
        zvals = depth[valid_depth]
        result["depth_median"] = float(np.median(zvals))
        result["depth_mean"] = float(zvals.mean())
        result["depth_std"] = float(zvals.std())
        result["depth_abs_error_m"] = float(abs(np.median(zvals) - z_true))
        result["depth_rel_error_pct"] = float(abs(np.median(zvals) - z_true) / z_true * 100.0)
    else:
        result["depth_median"] = None
        result["depth_abs_error_m"] = None
        result["depth_rel_error_pct"] = None

    return result


def run_candidate(label: str, params: dict, fixtures):
    stereo = make_stereo(params)
    rows = [measure_fixture(stereo, fx) for fx in fixtures]
    for r in rows:
        r["candidate"] = label
    return rows


if __name__ == "__main__":
    import json

    fixtures = build_all_fixtures()
    rows = run_candidate("CURRENT_baseline", current_sgbm_params(), fixtures)
    with open("benchmarks/i1_stereo_accuracy/results/baseline_current.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} measurement rows.")
