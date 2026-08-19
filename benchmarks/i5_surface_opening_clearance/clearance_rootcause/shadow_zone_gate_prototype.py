"""
I6.2A — CLEARANCE SHADOW-ZONE-GATE PROTOTYPE ONLY. No production-code
changes.

Per the I6 final-closure directive: before introducing any new heuristic,
check whether the EXISTING, already-validated I3 shadow-zone reliability
signal (geometry.reliability.compute_shadow_zone_mask -- the same,
unmodified function already threaded into build_obstacle_cloud/
build_free_space_rays/build_surface_evidence/build_boundary_evidence, but
NEVER threaded into ThreatAssessor.assess()/build_clearance_evidence())
already discriminates the false-clear ThreatAssessor beam columns from
ordinary correctly-measured ones and from noise -- without any new
clustering/statistical heuristic at all.

Mechanism under test: for each beam column, what fraction of the pixels
that actually fed ThreatAssessor's IQR-kept population (the population
distance_m is computed from) fall inside the shadow-zone mask? If that
overlap is high specifically for the known false-clear/contaminated
columns and low for ordinary correct columns, this is a direct, existing,
already-safety-validated signal that the beam's own aggregation ran on
contaminated evidence -- independent of what the "correct" value would
have been (which I6.1 already proved is not recoverable).

Reuses the exact fixtures already used by clearance/measure.py and
clearance_rootcause/real_validate.py.
"""
import sys
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import json
import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.reliability import compute_shadow_zone_mask

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

# I3's own defaults (PipelineConfig), unchanged -- not retuned here.
SHADOW_LOOKAHEAD_PX = 8
SHADOW_GRADIENT_THRESHOLD_PX = 3.0
SHADOW_MAX_WIDTH_PX = 40


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


def analyze_beam(depth_map, shadow_mask, x1, x2):
    """Recompute ThreatAssessor's own IQR-kept population for this column
    (this IS what distance_m is derived from) and measure what fraction of
    it falls inside the I3 shadow-zone mask. Returns None if ThreatAssessor
    itself would produce NO_DATA here (nothing to contaminate)."""
    col2d = depth_map[:, x1:x2]
    shadow_col = shadow_mask[:, x1:x2]
    valid_mask = (col2d > 0) & np.isfinite(col2d)
    valid_count = int(valid_mask.sum())
    total_pixels = int(col2d.size)
    if valid_count < MIN_VALID:
        return None

    vals = col2d[valid_mask].astype(np.float64)
    shadow_flags = shadow_col[valid_mask]

    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        kept_mask = (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
    else:
        kept_mask = np.ones_like(vals, dtype=bool)
    if kept_mask.sum() < MIN_VALID:
        return None

    current_d_m = float(np.percentile(vals[kept_mask], PERCENTILE))
    coverage_fraction = valid_count / total_pixels if total_pixels else 0.0

    kept_shadow_frac = float(shadow_flags[kept_mask].mean()) if kept_mask.sum() else 0.0
    all_valid_shadow_frac = float(shadow_flags.mean()) if valid_count else 0.0

    return dict(
        current_d_m=current_d_m, coverage_fraction=coverage_fraction,
        kept_support=int(kept_mask.sum()), valid_count=valid_count,
        kept_shadow_frac=kept_shadow_frac, all_valid_shadow_frac=all_valid_shadow_frac,
    )


def collect_records():
    pipeline = _pipeline()
    records = []

    def scan(result, zones, scenario, seed, known_false_clear_beams=(), known_artifact_beams=()):
        depth_map = result.depth_map
        shadow_mask = compute_shadow_zone_mask(
            result.disparity_map, result.disparity_map > 0.0,
            lookahead_px=SHADOW_LOOKAHEAD_PX,
            gradient_threshold_px=SHADOW_GRADIENT_THRESHOLD_PX,
            max_width_px=SHADOW_MAX_WIDTH_PX,
        )
        h, w = depth_map.shape[:2]
        beam_w = w / N_BEAMS
        for i in range(N_BEAMS):
            x1, x2 = int(i * beam_w), int((i + 1) * beam_w)
            a = analyze_beam(depth_map, shadow_mask, x1, x2)
            if a is None:
                continue
            true_range = _sector_true_range(x1, x2, zones) if zones else None

            if i in known_artifact_beams:
                bucket = "B_artifact_correct_reading"
            elif i in known_false_clear_beams:
                bucket = "A_known_false_clear"
            elif scenario in ("decorrelated_invalid", "pure_noise_30seeds"):
                bucket = "C_noise"
            else:
                bucket = "D_ordinary"

            records.append(dict(
                scenario=scenario, seed=seed, beam=i, x1=x1, x2=x2,
                true_range_m=true_range, bucket=bucket, **a,
            ))

    zones_mz = [(0.0, 0.33, 1.5), (0.33, 0.66, 4.0), (0.66, 1.0, 2.5)]
    for seed in range(1, 4):
        fx = make_multi_zone_fixture(zones_mz, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result, zones_mz, "multi_zone", seed, known_false_clear_beams={13})

    zones_no = [(0.0, 0.4, 5.0), (0.4, 0.6, 2.0), (0.6, 1.0, 5.0)]
    for seed in range(1, 4):
        fx = make_narrow_obstacle_fixture(near_m=2.0, far_m=5.0, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        artifact_beams = {9} if seed == 1 else set()
        scan(result, zones_no, "narrow_obstacle", seed,
             known_false_clear_beams={8}, known_artifact_beams=artifact_beams)

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


def sweep(records):
    by_bucket = {}
    for r in records:
        by_bucket.setdefault(r["bucket"], []).append(r)

    print(f"Total ThreatAssessor beam readings analyzed: {len(records)}")
    for bucket in ("A_known_false_clear", "B_artifact_correct_reading", "C_noise", "D_ordinary"):
        rows = by_bucket.get(bucket, [])
        print(f"\n--- bucket {bucket}: n={len(rows)} ---")
        for r in rows:
            print(f"  {r['scenario']} seed={r['seed']} beam={r['beam']} x=[{r['x1']},{r['x2']}] "
                  f"true={r['true_range_m']} cur_d_m={r['current_d_m']:.3f} "
                  f"kept_shadow_frac={r['kept_shadow_frac']:.3f} all_valid_shadow_frac={r['all_valid_shadow_frac']:.3f} "
                  f"kept_support={r['kept_support']} coverage={r['coverage_fraction']:.3f}")

    print("\n=== Sweep: minimum kept_shadow_frac to flag a beam as contaminated (-> DEGRADED) ===")
    print(f"{'threshold':>9} | {'A_flagged':>18} | {'B_flagged(cost)':>16} | "
          f"{'C_flagged':>10} | {'D_flagged(cost)':>16}")
    a_rows = by_bucket.get("A_known_false_clear", [])
    b_rows = by_bucket.get("B_artifact_correct_reading", [])
    c_rows = by_bucket.get("C_noise", [])
    d_rows = by_bucket.get("D_ordinary", [])
    for thresh in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90]:
        a_flagged = sum(1 for r in a_rows if r["kept_shadow_frac"] >= thresh)
        b_flagged = sum(1 for r in b_rows if r["kept_shadow_frac"] >= thresh)
        c_flagged = sum(1 for r in c_rows if r["kept_shadow_frac"] >= thresh)
        d_flagged = sum(1 for r in d_rows if r["kept_shadow_frac"] >= thresh)
        print(f"{thresh:9.2f} | {a_flagged:5d}/{len(a_rows):<11d} | {b_flagged:5d}/{len(b_rows):<9d} | "
              f"{c_flagged:10d} | {d_flagged:5d}/{len(d_rows):<9d}")


def main():
    records = collect_records()
    with open(
        "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i5_surface_opening_clearance/"
        "clearance_rootcause/results/shadow_zone_gate_prototype.json", "w"
    ) as f:
        json.dump(records, f, indent=2, default=str)
    sweep(records)


if __name__ == "__main__":
    main()
