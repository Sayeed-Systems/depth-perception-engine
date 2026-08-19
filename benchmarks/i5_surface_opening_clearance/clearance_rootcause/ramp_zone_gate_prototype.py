"""
I6.3A — CLEARANCE WIDE-RAMP-GATE PROTOTYPE ONLY. No production-code
changes.

Targets the one remaining false-clear root cause after I6.2's shadow-zone
gate closed the narrow_obstacle case: multi_zone beam13 (3/3 seeds,
~53-54% error, unchanged). Confirmed directly (prior session) that this
is NOT classical occlusion-shadow geometry -- geometry.reliability.
compute_shadow_zone_mask (forward or mirrored) measures exactly 0.0%
overlap with beam13's IQR-kept population. Instead, the actual depth
profile across the transition (row 120, x=180..239) shows a smooth
~20-pixel RAMP (3.98m -> 2.49m spread continuously across x~216-235),
consistent with StereoSGBM's own smoothness-regularization radius
(tied to block_size=9), not a narrow, geometrically-predicted occlusion
strip. compute_shadow_zone_mask's width model (round(disparity_gap),
~6px here) is far narrower than the actual contamination -- a width
mismatch, not a direction mismatch (bidirectional mirroring already
ruled that out).

Candidate mechanism under test: a DIRECTION-AGNOSTIC, WIDE-WINDOW local
disparity range detector -- compute_ramp_zone_mask(disparity, valid,
window_px, gradient_threshold_px). For every pixel, take the rolling
max-min disparity RANGE within a window of width window_px centered on
it (vectorized via a bounded shift/min/max loop, same style as
compute_shadow_zone_mask's own bounded shift/OR loop -- no per-pixel
Python loop). Flag it if that range >= gradient_threshold_px. This
generalizes the existing shadow-zone mechanism's "detect proximity to a
large disparity change" idea to a WIDER window and REMOVES the
directionality restriction (a ramp can run either way), at the cost of
being agnostic to *why* the range is large (real occlusion, wide SGBM
smoothing, or a genuine two-object scene with two different true ranges
close together in bearing -- all three would trigger identically; only
the last is a false-positive-risk case, evaluated below via the D/C
buckets).

Reuses the exact fixtures already used by clearance/measure.py and
clearance_rootcause/{real_validate,shadow_zone_gate_prototype}.py.
"""
import sys
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import json
import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

from benchmarks.i1_stereo_accuracy.fixtures import (
    make_decorrelated_fixture, make_discontinuity_fixture, W as _W, H as _H,
)
from benchmarks.i4_boundary_precision.fixtures import make_narrow_obstacle_fixture
from benchmarks.i5_surface_opening_clearance.clearance.fixtures import (
    make_multi_zone_fixture, make_dis_occlusion_fixture,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")

N_BEAMS = 20
MIN_VALID = 5
PERCENTILE = 15
DEAD_ZONE_PX = 128
CLEAR_M = 2.0
CAUTION_M = 1.0


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _pipeline():
    cfg = PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        enable_geometry_frame=True,
    )
    return DepthPerceptionPipeline(cfg, _CALIB, rectify=False, body_T_camera_left=_transform())


def _sector_true_range(x1, x2, zones):
    best = None
    for c0f, c1f, depth_m in zones:
        c0, c1 = int(c0f * _W), int(c1f * _W)
        ov0, ov1 = max(c0, x1, DEAD_ZONE_PX), min(c1, x2)
        if ov1 > ov0:
            best = depth_m if best is None else min(best, depth_m)
    return best


def _status_for(d_m):
    if d_m <= 0.0:
        return "NO_DATA"
    if d_m < CAUTION_M:
        return "BLOCKED"
    if d_m < CLEAR_M:
        return "CAUTION"
    return "CLEAR"


def compute_ramp_zone_mask(disparity_map, valid_mask, window_px, gradient_threshold_px):
    """Prototype only -- not geometry.reliability. Direction-agnostic
    rolling max-min disparity range over a window_px-wide centered
    window, vectorized via a bounded shift loop (half = window_px//2
    iterations, each a whole-array elementwise min/max -- no per-pixel
    Python loop, matching compute_shadow_zone_mask's own style)."""
    h, w = disparity_map.shape[:2]
    big = np.float32(1.0e6)
    d_max = np.where(valid_mask, disparity_map, -big).astype(np.float32)
    d_min = np.where(valid_mask, disparity_map, big).astype(np.float32)
    roll_max = d_max.copy()
    roll_min = d_min.copy()
    half = max(1, window_px // 2)
    for k in range(1, half + 1):
        roll_max[:, k:] = np.maximum(roll_max[:, k:], d_max[:, : w - k])
        roll_min[:, k:] = np.minimum(roll_min[:, k:], d_min[:, : w - k])
        roll_max[:, : w - k] = np.maximum(roll_max[:, : w - k], d_max[:, k:])
        roll_min[:, : w - k] = np.minimum(roll_min[:, : w - k], d_min[:, k:])
    rng = roll_max - roll_min
    return (rng >= gradient_threshold_px) & valid_mask


def analyze_beam(depth_map, ramp_mask, x1, x2):
    col2d = depth_map[:, x1:x2]
    ramp_col = ramp_mask[:, x1:x2]
    valid_mask = (col2d > 0) & np.isfinite(col2d)
    valid_count = int(valid_mask.sum())
    total_pixels = int(col2d.size)
    if valid_count < MIN_VALID:
        return None

    vals = col2d[valid_mask].astype(np.float64)
    ramp_flags = ramp_col[valid_mask]

    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        kept_mask = (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
    else:
        kept_mask = np.ones_like(vals, dtype=bool)
    if kept_mask.sum() < MIN_VALID:
        return None

    current_d_m = float(np.percentile(vals[kept_mask], PERCENTILE))
    kept_ramp_frac = float(ramp_flags[kept_mask].mean()) if kept_mask.sum() else 0.0

    return dict(current_d_m=current_d_m, kept_support=int(kept_mask.sum()), kept_ramp_frac=kept_ramp_frac)


def collect_records(window_px):
    pipeline = _pipeline()
    records = []

    def scan(result, zones, scenario, seed, known_false_clear_beams=()):
        depth_map = result.depth_map
        disp = result.disparity_map
        valid = disp > 0.0
        ramp_mask = compute_ramp_zone_mask(disp, valid, window_px=window_px, gradient_threshold_px=RAMP_THRESH)
        h, w = depth_map.shape[:2]
        beam_w = w / N_BEAMS
        for i in range(N_BEAMS):
            x1, x2 = int(i * beam_w), int((i + 1) * beam_w)
            a = analyze_beam(depth_map, ramp_mask, x1, x2)
            if a is None:
                continue
            true_range = _sector_true_range(x1, x2, zones) if zones else None
            bucket = (
                "A_known_false_clear" if i in known_false_clear_beams
                else ("C_noise" if scenario in ("decorrelated_invalid", "pure_noise_30seeds") else "D_ordinary")
            )
            records.append(dict(scenario=scenario, seed=seed, beam=i, x1=x1, x2=x2,
                                 true_range_m=true_range, bucket=bucket, **a))

    zones_mz = [(0.0, 0.33, 1.5), (0.33, 0.66, 4.0), (0.66, 1.0, 2.5)]
    for seed in range(1, 4):
        fx = make_multi_zone_fixture(zones_mz, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result, zones_mz, "multi_zone", seed, known_false_clear_beams={13})

    zones_no = [(0.0, 0.4, 5.0), (0.4, 0.6, 2.0), (0.6, 1.0, 5.0)]
    for seed in range(1, 4):
        fx = make_narrow_obstacle_fixture(near_m=2.0, far_m=5.0, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result, zones_no, "narrow_obstacle", seed)

    zones_occ = [(0.0, 0.5, 1.5), (0.5, 1.0, 5.0)]
    for seed in range(1, 6):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=True)
        result = pipeline.process(fx.left, fx.right)
        scan(result, zones_occ, "occlusion_near_side", seed)

    for seed in range(1, 6):
        fx = make_dis_occlusion_fixture(near_m=1.5, far_m=5.0, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result, zones_occ, "dis_occlusion_far_side", seed)

    fx = make_multi_zone_fixture([(0.0, 1.0, 1.5)], seed=1)
    result = pipeline.process(fx.left, fx.right)
    scan(result, [(0.0, 1.0, 1.5)], "single_near_1.5m", 1)

    fx = make_multi_zone_fixture([(0.0, 1.0, 5.0)], seed=2)
    result = pipeline.process(fx.left, fx.right)
    scan(result, [(0.0, 1.0, 5.0)], "single_far_5.0m", 2)

    zones_open = [(0.0, 0.4, 6.0), (0.4, 0.6, 6.0), (0.6, 1.0, 6.0)]
    for seed in range(1, 4):
        fx = make_multi_zone_fixture(zones_open, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result, zones_open, "open_far_uniform", seed)

    for seed in range(1, 31):
        fx = make_decorrelated_fixture(seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result, [], "pure_noise_30seeds", seed)

    return records


RAMP_THRESH = 2.0  # px, held fixed while window_px is swept below


def is_accurate(r, tol_pct=10.0):
    if r["true_range_m"] is None:
        return False
    err_pct = 100.0 * abs(r["current_d_m"] - r["true_range_m"]) / r["true_range_m"]
    return err_pct <= tol_pct


def main():
    print(f"gradient_threshold_px fixed at {RAMP_THRESH}px; sweeping window_px only\n")
    print(f"{'window_px':>10} | {'A_flagged/3':>12} | {'A_kept_ramp_frac (all 3)':>28} | "
          f"{'C_flagged':>10} | {'D_accurate_flagged(cost)':>26}")
    for window_px in [8, 12, 16, 20, 24, 30, 40, 60]:
        records = collect_records(window_px)
        A = [r for r in records if r["bucket"] == "A_known_false_clear"]
        C = [r for r in records if r["bucket"] == "C_noise"]
        D = [r for r in records if r["bucket"] == "D_ordinary"]
        D_accurate = [r for r in D if is_accurate(r)]

        thresh = 0.30  # matching the I6.2 shadow-zone gate's own validated cost/margin threshold
        a_flagged = sum(1 for r in A if r["kept_ramp_frac"] >= thresh)
        c_flagged = sum(1 for r in C if r["kept_ramp_frac"] >= thresh)
        d_flagged = sum(1 for r in D_accurate if r["kept_ramp_frac"] >= thresh)
        a_fracs = [round(r["kept_ramp_frac"], 3) for r in A]

        print(f"{window_px:10d} | {a_flagged:5d}/{len(A):<6d} | {str(a_fracs):>28} | "
              f"{c_flagged:10d} | {d_flagged:5d}/{len(D_accurate):<19d}")

    # Full dump at the window_px we expect to work best (ramp measured ~20px wide)
    best_window = 24
    records = collect_records(best_window)
    with open(
        "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i5_surface_opening_clearance/"
        "clearance_rootcause/results/ramp_zone_gate_prototype.json", "w"
    ) as f:
        json.dump(records, f, indent=2, default=str)

    print(f"\n--- Full record dump at window_px={best_window} ---")
    for bucket in ("A_known_false_clear", "C_noise", "D_ordinary"):
        rows = [r for r in records if r["bucket"] == bucket]
        print(f"\n-- {bucket}: n={len(rows)} --")
        show = rows if bucket != "D_ordinary" else [r for r in rows if r["kept_ramp_frac"] >= 0.30]
        for r in show:
            print(f"  {r['scenario']} seed={r['seed']} beam={r['beam']} x=[{r['x1']},{r['x2']}] "
                  f"true={r['true_range_m']} cur={r['current_d_m']:.3f} kept_ramp_frac={r['kept_ramp_frac']:.3f}")


if __name__ == "__main__":
    main()
