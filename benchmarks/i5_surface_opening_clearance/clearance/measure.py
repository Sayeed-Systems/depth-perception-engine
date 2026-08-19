"""
Phase I5 Part C — clearance qualification measurement. Real, unmodified
DepthPerceptionPipeline, full config, rectify=False, real calibration.
"""
import json
import sys
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

from benchmarks.i1_stereo_accuracy.fixtures import (
    make_discontinuity_fixture, make_decorrelated_fixture, W as _W, H as _H,
)
from benchmarks.i4_boundary_precision.fixtures import make_narrow_obstacle_fixture
from benchmarks.i5_surface_opening_clearance.clearance.fixtures import (
    make_dis_occlusion_fixture, make_multi_zone_fixture,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")


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
    """True nearest STEREO-OBSERVABLE range for a sector [x1,x2): the
    minimum true depth among zones whose column range overlaps [x1,x2)
    AND lies outside the structural SGBM dead zone (columns < 128 are
    unobservable regardless of scene content, per PipelineConfig's own
    num_disparities=128 default)."""
    dead_zone_px = 128
    best = None
    for c0, c1, depth_m in zones:
        c0px, c1px = int(c0 * _W), int(c1 * _W)
        ov0, ov1 = max(c0px, x1, dead_zone_px), min(c1px, x2)
        if ov1 > ov0:
            best = depth_m if best is None else min(best, depth_m)
    return best


def _record_sectors(result, zones, scenario_label, seed, out):
    gf = result.geometry_frame
    for ce in gf.clearance_evidence or []:
        true_range = _sector_true_range(ce.x1, ce.x2, zones)
        row = {
            "scenario": scenario_label, "seed": seed, "index": ce.index,
            "x1": ce.x1, "x2": ce.x2, "bearing_center_rad": ce.bearing_center_rad,
            "true_range_m": true_range, "measured_range_m": ce.nearest_distance_m,
            "support_state": ce.support_state, "coverage_fraction": ce.coverage_fraction,
            "has_evidence": ce.has_evidence,
        }
        if true_range is not None and ce.nearest_distance_m is not None:
            err = ce.nearest_distance_m - true_range
            row["signed_error_m"] = err
            row["abs_error_m"] = abs(err)
            row["rel_error_pct"] = 100.0 * abs(err) / true_range
            row["false_clear"] = (ce.support_state == "SUPPORTED") and (err > 0.05)  # measured farther than true, >5cm slack
        out.append(row)


def main():
    out = []
    # Methodology fix (I6.2): each fixture below is an independent,
    # unrelated synthetic scene, not a continuation of the previous
    # frame. ThreatAssessor.assess() holds per-beam EMA-smoothed distance
    # (alpha=0.30) and debounced status (debounce_frames=3) ACROSS calls
    # on the same pipeline instance by design -- correct for a real video
    # stream, but a single shared pipeline/ThreatAssessor reused across
    # these unrelated back-to-back fixtures let each fixture's reported
    # distance/status be partially contaminated by whatever unrelated
    # scene ran immediately before it in this loop. Verified directly
    # (benchmarks/i5_surface_opening_clearance/clearance_rootcause/
    # methodology_check.py): on this exact fixture set, the shared-
    # pipeline methodology reported 30 false-clear sectors vs. 4 with a
    # fresh pipeline per fixture -- i.e. most of the previously-reported
    # false-clear population was stale EMA/debounce carryover, not a
    # genuine single-frame defect. Each fixture now gets its own fresh
    # DepthPerceptionPipeline (via _pipeline()), matching true first-frame
    # (or freshly-reset) single-scene behavior.

    # 1. Single near obstacle (whole frame near)
    fx = make_multi_zone_fixture([(0.0, 1.0, 1.5)], seed=1)
    result = _pipeline().process(fx.left, fx.right)
    _record_sectors(result, [(0.0, 1.0, 1.5)], "single_near_1.5m", 1, out)

    # 2. Single far obstacle (whole frame far)
    fx = make_multi_zone_fixture([(0.0, 1.0, 5.0)], seed=2)
    result = _pipeline().process(fx.left, fx.right)
    _record_sectors(result, [(0.0, 1.0, 5.0)], "single_far_5.0m", 2, out)

    # 3. Multiple discrete obstacle sectors
    zones = [(0.0, 0.33, 1.5), (0.33, 0.66, 4.0), (0.66, 1.0, 2.5)]
    for seed in range(1, 4):
        fx = make_multi_zone_fixture(zones, seed=seed)
        result = _pipeline().process(fx.left, fx.right)
        _record_sectors(result, zones, "multi_zone", seed, out)

    # 4. Narrow obstacle
    for seed in range(1, 4):
        fx = make_narrow_obstacle_fixture(near_m=2.0, far_m=5.0, seed=seed)
        result = _pipeline().process(fx.left, fx.right)
        zones = [(0.0, 0.4, 5.0), (0.4, 0.6, 2.0), (0.6, 1.0, 5.0)]
        _record_sectors(result, zones, "narrow_obstacle", seed, out)

    # 5. Partial occlusion (near-side strip) — re-verify I4's finding directly
    zones = [(0.0, 0.5, 1.5), (0.5, 1.0, 5.0)]
    for seed in range(1, 6):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=True)
        result = _pipeline().process(fx.left, fx.right)
        _record_sectors(result, zones, "occlusion_near_side", seed, out)

    # 6. Dis-occlusion (far-side strip) — THE key new fixture
    for seed in range(1, 6):
        fx = make_dis_occlusion_fixture(near_m=1.5, far_m=5.0, seed=seed)
        result = _pipeline().process(fx.left, fx.right)
        _record_sectors(result, zones, "dis_occlusion_far_side", seed, out)

    # 7. Invalid/no-correspondence sector (decorrelated noise, whole frame)
    for seed in range(1, 6):
        fx = make_decorrelated_fixture(seed)
        result = _pipeline().process(fx.left, fx.right)
        # No genuine zone anywhere -> true_range always None -> only
        # support_state/false-SUPPORTED tracked, not error.
        _record_sectors(result, [], "decorrelated_invalid", seed, out)

    # 8. Opening/free-space sector (genuinely far/open direction)
    zones = [(0.0, 0.4, 6.0), (0.4, 0.6, 6.0), (0.6, 1.0, 6.0)]  # uniformly far == "open"
    for seed in range(1, 4):
        fx = make_multi_zone_fixture(zones, seed=seed)
        result = _pipeline().process(fx.left, fx.right)
        _record_sectors(result, zones, "open_far_uniform", seed, out)

    # 9. Pure noise negative case, larger sample (30 seeds)
    for seed in range(1, 31):
        fx = make_decorrelated_fixture(seed)
        result = _pipeline().process(fx.left, fx.right)
        _record_sectors(result, [], "pure_noise_30seeds", seed, out)

    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i5_surface_opening_clearance/clearance/results/measure.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {len(out)} sector records to {path}")

    # ---- Summary ----
    import statistics
    supported_errs = [r["rel_error_pct"] for r in out if r.get("support_state") == "SUPPORTED" and "rel_error_pct" in r]
    if supported_errs:
        print(f"\nSUPPORTED-sector relative error: n={len(supported_errs)} "
              f"mean={statistics.mean(supported_errs):.3f}% max={max(supported_errs):.3f}% "
              f"median={statistics.median(supported_errs):.3f}%")

    false_clear_rows = [r for r in out if r.get("false_clear")]
    print(f"\nFALSE-CLEAR sectors (SUPPORTED, measured > true+5cm): {len(false_clear_rows)}")
    for r in false_clear_rows:
        print(f"  {r}")

    print("\n--- dis_occlusion_far_side detail ---")
    for r in out:
        if r["scenario"] == "dis_occlusion_far_side":
            print(f"  seed={r['seed']} x=[{r['x1']},{r['x2']}] true={r['true_range_m']} "
                  f"measured={r['measured_range_m']} state={r['support_state']} cov={r['coverage_fraction']:.3f} "
                  f"err%={r.get('rel_error_pct')}")

    print("\n--- occlusion_near_side detail (re-verify I4 finding) ---")
    for r in out:
        if r["scenario"] == "occlusion_near_side":
            print(f"  seed={r['seed']} x=[{r['x1']},{r['x2']}] true={r['true_range_m']} "
                  f"measured={r['measured_range_m']} state={r['support_state']} cov={r['coverage_fraction']:.3f} "
                  f"err%={r.get('rel_error_pct')}")

    print("\n--- pure_noise_30seeds: SUPPORTED count ---")
    noise_rows = [r for r in out if r["scenario"] == "pure_noise_30seeds"]
    n_supported = sum(1 for r in noise_rows if r["support_state"] == "SUPPORTED")
    print(f"  total sectors={len(noise_rows)} SUPPORTED={n_supported}")

    print("\n--- open_far_uniform: any sector reads close/BLOCKED-like? ---")
    for r in out:
        if r["scenario"] == "open_far_uniform" and r.get("measured_range_m") is not None:
            if r["measured_range_m"] < 5.5:
                print(f"  UNEXPECTED CLOSE READING: {r}")

    print("\nPer-scenario support_state counts:")
    scenarios = sorted(set(r["scenario"] for r in out))
    for scen in scenarios:
        rows = [r for r in out if r["scenario"] == scen]
        counts = {}
        for r in rows:
            counts[r["support_state"]] = counts.get(r["support_state"], 0) + 1
        print(f"  {scen}: n={len(rows)} {counts}")


if __name__ == "__main__":
    main()
