"""
Phase I4, Step 1 — collect BoundaryEvidence support/depth_step/orientation
distributions across fixtures A-I, classified TP/FP/TN/FN against known
ground truth. Read-only w.r.t. src/ — uses the real, unmodified (post-I3)
DepthPerceptionPipeline exactly as shipped.
"""
import json
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

from benchmarks.i1_stereo_accuracy.fixtures import (
    make_discontinuity_fixture, make_decorrelated_fixture, make_repetitive_fixture, W as _W,
)
from benchmarks.i4_boundary_precision.fixtures import (
    make_narrow_obstacle_fixture, make_weak_texture_discontinuity_fixture, make_textureless_fixture,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
_DEAD_ZONE_PX = 128  # numDisparities structural left-edge dead zone


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _cfg(grid_rc=3):
    return PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, surface_grid_rows=grid_rc, surface_grid_cols=grid_rc,
        enable_boundary_geometry=True, boundary_grid_rows=grid_rc, boundary_grid_cols=grid_rc,
        enable_opening_geometry=True, enable_geometry_frame=True,
        # shipped I3 defaults: geometry_shadow_zone_enabled=True
    )


def _pipeline(cfg):
    return DepthPerceptionPipeline(cfg, _CALIB, rectify=False, body_T_camera_left=_transform())


def _left_cell_in_dead_zone(b):
    left_cell_x2_approx = b.x1 + (b.x2 - b.x1) // 2
    return left_cell_x2_approx <= _DEAD_ZONE_PX


def _classify(gf, genuine_cols, records, scenario_label, seed):
    """genuine_cols: list of true transition columns (RIGHT-direction
    pairs whose bbox straddles any of these columns are 'should be
    positive'), or empty list for a no-transition-anywhere fixture."""
    for b in gf.boundary_evidence or []:
        if b.direction != "RIGHT":
            continue
        if _left_cell_in_dead_zone(b):
            continue
        crosses = any(b.x1 < gc < b.x2 for gc in genuine_cols)
        is_positive = b.state == "OBSERVED_DISCONTINUITY"
        label = None
        if crosses and is_positive:
            label = "TP"
        elif crosses and not is_positive:
            label = "FN"
        elif not crosses and is_positive:
            label = "FP"
        else:
            label = "TN"
        records.append({
            "scenario": scenario_label, "seed": seed, "row": b.row, "col": b.col,
            "x1": b.x1, "x2": b.x2, "state": b.state,
            "support_from": b.support_fraction_from, "support_to": b.support_fraction_to,
            "support_min": min(b.support_fraction_from, b.support_fraction_to),
            "depth_step_m": b.depth_step_m, "orientation_rad": b.orientation_change_rad,
            "label": label,
        })


def main():
    records = []
    pipeline3 = _pipeline(_cfg(3))

    # A. strong genuine depth step
    for seed in range(1, 8):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=False)
        result = pipeline3.process(fx.left, fx.right)
        _classify(result.geometry_frame, [_W // 2], records, "A_strong_step", seed)

    # B. weak genuine depth step
    for seed in range(1, 8):
        fx = make_discontinuity_fixture(near_m=2.0, far_m=2.5, seed=seed, occlusion=False)
        result = pipeline3.process(fx.left, fx.right)
        _classify(result.geometry_frame, [_W // 2], records, "B_weak_step", seed)

    # C. distant genuine boundary
    for seed in range(1, 8):
        fx = make_discontinuity_fixture(near_m=5.0, far_m=6.0, seed=seed, occlusion=False)
        result = pipeline3.process(fx.left, fx.right)
        _classify(result.geometry_frame, [_W // 2], records, "C_distant", seed)

    # D. partially occluded genuine boundary (I3's own fixture)
    for seed in range(1, 8):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=True)
        result = pipeline3.process(fx.left, fx.right)
        _classify(result.geometry_frame, [_W // 2], records, "D_occluded", seed)

    # E. narrow/small obstacle boundary (two real transitions)
    for seed in range(1, 8):
        fx = make_narrow_obstacle_fixture(near_m=2.0, far_m=5.0, seed=seed)
        result = pipeline3.process(fx.left, fx.right)
        genuine_cols = [int(0.4 * _W), int(0.6 * _W)]
        _classify(result.geometry_frame, genuine_cols, records, "E_narrow_obstacle", seed)

    # F. weak-texture genuine boundary
    for seed in range(1, 8):
        fx = make_weak_texture_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed)
        result = pipeline3.process(fx.left, fx.right)
        _classify(result.geometry_frame, [_W // 2], records, "F_weak_texture", seed)

    # G. pure decorrelated noise
    for seed in range(1, 11):
        fx = make_decorrelated_fixture(seed)
        result = pipeline3.process(fx.left, fx.right)
        _classify(result.geometry_frame, [], records, "G_decorrelated", seed)

    # H. repetitive texture negative case
    for seed in range(1, 8):
        fx = make_repetitive_fixture(depth_m=2.0, seed=seed)
        result = pipeline3.process(fx.left, fx.right)
        _classify(result.geometry_frame, [], records, "H_repetitive", seed)

    # I. invalid/no-correspondence (textureless)
    fx = make_textureless_fixture()
    result = pipeline3.process(fx.left, fx.right)
    _classify(result.geometry_frame, [], records, "I_textureless", 0)

    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i4_boundary_precision/results/collect.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {len(records)} records to {path}")

    # Summary
    import statistics
    for label in ("TP", "FP", "TN", "FN"):
        vals = [r["support_min"] for r in records if r["label"] == label]
        if vals:
            print(f"{label}: n={len(vals)} support_min: min={min(vals):.4f} max={max(vals):.4f} "
                  f"median={statistics.median(vals):.4f} mean={statistics.mean(vals):.4f}")
        else:
            print(f"{label}: n=0")

    print("\nPer-scenario TP/FP/FN/TN:")
    scenarios = sorted(set(r["scenario"] for r in records))
    for scen in scenarios:
        rows = [r for r in records if r["scenario"] == scen]
        counts = {lbl: sum(1 for r in rows if r["label"] == lbl) for lbl in ("TP", "FP", "FN", "TN")}
        print(f"  {scen}: {counts}")


if __name__ == "__main__":
    main()
