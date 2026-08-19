"""
I6.2 methodology check (diagnostic only, no production changes).

clearance/measure.py constructs ONE DepthPerceptionPipeline/ThreatAssessor
at the top of main() and reuses it across ALL ~9 unrelated fixture
families in sequence (single_near -> single_far -> multi_zone x3 ->
narrow_obstacle x3 -> occlusion x5 -> dis_occlusion x5 -> decorrelated x5
-> open_far x3 -> pure_noise x30). ThreatAssessor.assess() holds per-beam
EMA-smoothed distance and debounced status STATE across calls
(threat_assessment.py's own docstring: "constructing a fresh
ThreatAssessor per frame throws that smoothing away"). A single
unrelated-scene call only partially updates a beam's EMA distance
(alpha=0.30) and doesn't commit a new status until it repeats for
debounce_frames=3 consecutive calls -- neither of which holds when
consecutive calls in the loop are DIFFERENT, uncorrelated synthetic
scenes.

This checks whether the 28/252 false-clear count is an artifact of that
cross-fixture state carryover, by rerunning the exact same fixture set
with the pipeline (and therefore ThreatAssessor state) reset before each
one -- true single-frame, uncontaminated ClearanceEvidence -- and
reporting the false-clear count and per-record differences.
"""
import sys
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

from benchmarks.i1_stereo_accuracy.fixtures import make_discontinuity_fixture, make_decorrelated_fixture, W as _W
from benchmarks.i4_boundary_precision.fixtures import make_narrow_obstacle_fixture
from benchmarks.i5_surface_opening_clearance.clearance.fixtures import (
    make_dis_occlusion_fixture, make_multi_zone_fixture,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
DEAD_ZONE_PX = 128


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


def _record_sectors(result, zones, scenario_label, seed, out, reset_mode):
    gf = result.geometry_frame
    for ce in gf.clearance_evidence or []:
        true_range = _sector_true_range(ce.x1, ce.x2, zones)
        row = {
            "reset_mode": reset_mode, "scenario": scenario_label, "seed": seed, "index": ce.index,
            "x1": ce.x1, "x2": ce.x2, "true_range_m": true_range,
            "measured_range_m": ce.nearest_distance_m, "support_state": ce.support_state,
        }
        if true_range is not None and ce.nearest_distance_m is not None:
            err = ce.nearest_distance_m - true_range
            row["false_clear"] = (ce.support_state == "SUPPORTED") and (err > 0.05)
        out.append(row)
    return out


def run(reset_mode):
    """reset_mode: 'shared' (bug-reproducing, matches clearance/measure.py
    exactly) or 'fresh' (new pipeline instance per fixture -- clean,
    uncontaminated single-frame ClearanceEvidence)."""
    out = []
    pipeline = _pipeline() if reset_mode == "shared" else None

    def get_pipeline():
        nonlocal pipeline
        if reset_mode == "fresh":
            pipeline = _pipeline()
        return pipeline

    zones_mz = [(0.0, 0.33, 1.5), (0.33, 0.66, 4.0), (0.66, 1.0, 2.5)]
    for seed in range(1, 4):
        fx = make_multi_zone_fixture(zones_mz, seed=seed)
        result = get_pipeline().process(fx.left, fx.right)
        _record_sectors(result, zones_mz, "multi_zone", seed, out, reset_mode)

    zones_no = [(0.0, 0.4, 5.0), (0.4, 0.6, 2.0), (0.6, 1.0, 5.0)]
    for seed in range(1, 4):
        fx = make_narrow_obstacle_fixture(near_m=2.0, far_m=5.0, seed=seed)
        result = get_pipeline().process(fx.left, fx.right)
        _record_sectors(result, zones_no, "narrow_obstacle", seed, out, reset_mode)

    zones_occ = [(0.0, 0.5, 1.5), (0.5, 1.0, 5.0)]
    for seed in range(1, 6):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=True)
        result = get_pipeline().process(fx.left, fx.right)
        _record_sectors(result, zones_occ, "occlusion_near_side", seed, out, reset_mode)

    for seed in range(1, 6):
        fx = make_dis_occlusion_fixture(near_m=1.5, far_m=5.0, seed=seed)
        result = get_pipeline().process(fx.left, fx.right)
        _record_sectors(result, zones_occ, "dis_occlusion_far_side", seed, out, reset_mode)

    for seed in range(1, 6):
        fx = make_decorrelated_fixture(seed)
        result = get_pipeline().process(fx.left, fx.right)
        _record_sectors(result, [], "decorrelated_invalid", seed, out, reset_mode)

    zones_open = [(0.0, 0.4, 6.0), (0.4, 0.6, 6.0), (0.6, 1.0, 6.0)]
    for seed in range(1, 4):
        fx = make_multi_zone_fixture(zones_open, seed=seed)
        result = get_pipeline().process(fx.left, fx.right)
        _record_sectors(result, zones_open, "open_far_uniform", seed, out, reset_mode)

    return out


def main():
    shared = run("shared")
    fresh = run("fresh")

    shared_fc = [r for r in shared if r.get("false_clear")]
    fresh_fc = [r for r in fresh if r.get("false_clear")]
    print(f"SHARED-pipeline (matches clearance/measure.py exactly): "
          f"n_sectors={len(shared)} false_clear={len(shared_fc)}")
    print(f"FRESH-pipeline-per-fixture (uncontaminated single-frame): "
          f"n_sectors={len(fresh)} false_clear={len(fresh_fc)}")

    print("\n--- SHARED false-clear records ---")
    for r in shared_fc:
        print(f"  {r}")

    print("\n--- FRESH false-clear records ---")
    for r in fresh_fc:
        print(f"  {r}")

    print("\n--- Direct per-sector comparison (shared vs fresh), same scenario/seed/index ---")
    fresh_by_key = {(r["scenario"], r["seed"], r["index"]): r for r in fresh}
    for r in shared:
        key = (r["scenario"], r["seed"], r["index"])
        f = fresh_by_key.get(key)
        if f is None:
            continue
        if r["measured_range_m"] != f["measured_range_m"] or r["support_state"] != f["support_state"]:
            print(f"  {key}: SHARED measured={r['measured_range_m']} state={r['support_state']} fc={r.get('false_clear')} "
                  f"| FRESH measured={f['measured_range_m']} state={f['support_state']} fc={f.get('false_clear')} "
                  f"| true={r['true_range_m']}")


if __name__ == "__main__":
    main()
