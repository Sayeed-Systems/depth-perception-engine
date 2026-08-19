"""
Phase I1.1 — false-valid propagation safety closure.

Read-only w.r.t. src/depth_perception_engine/ — traces the REAL, unmodified
GeometryFrame chain (disparity -> valid_disparity_mask -> depth ->
valid_depth_mask -> geometry -> obstacle_cloud -> free_space_rays ->
clearance_evidence -> boundary_evidence -> opening_evidence ->
GeometryFrameQuality) for three deterministic negative fixtures:

  A. fully decorrelated / no-correspondence stereo (i.i.d. noise)
  B. controlled occlusion boundary (reuses fixtures.py's scenario E)
  C. weak-texture / no-support region (reuses fixtures.py's scenario C,
     plus the isolated-small-patch case)

Reuses fixtures.py's own ground-truth generators for A/B/C where possible;
for A, both the i.i.d.-noise technique (matching the original IA0
observation) and multiple seeds/grid sizes are exercised, since the
original "4 openings/frame at 6x8" finding predates the I1 SGBM fix and
must be re-verified against CURRENT source, not assumed to still hold.

No PipelineConfig/threshold change. No src/ file touched by this script.
"""
import json
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np
import cv2

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

from benchmarks.i1_stereo_accuracy.fixtures import (
    make_flat_fixture, make_discontinuity_fixture, make_decorrelated_fixture, FX, BASELINE_M,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIB.image_size


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _full_v1_config(surface_rows=3, surface_cols=3, boundary_rows=3, boundary_cols=3, **overrides):
    defaults = dict(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, surface_grid_rows=surface_rows, surface_grid_cols=surface_cols,
        enable_boundary_geometry=True, boundary_grid_rows=boundary_rows, boundary_grid_cols=boundary_cols,
        enable_opening_geometry=True,
        enable_geometry_frame=True,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _pipeline(config):
    return DepthPerceptionPipeline(config, _CALIB, body_T_camera_left=_transform())


def _trace(result, label):
    """Extract every metric the I1.1 task asks for from one process() result."""
    gf = result.geometry_frame
    out = {"label": label}

    valid_disp = result.valid_disparity_mask
    valid_depth = result.valid_depth_mask
    out["valid_disparity_fraction"] = float(valid_disp.mean())
    out["valid_depth_fraction"] = float(valid_depth.mean())
    out["valid_disparity_count"] = int(valid_disp.sum())
    out["valid_depth_count"] = int(valid_depth.sum())

    # UNKNOWN-space invariant: obstacle/ray counts must equal body-frame
    # valid_mask.sum() EXACTLY (D5/D6/E5/E6's own proven structural rule)
    # — verify it directly, don't just assume it still holds.
    body_valid_count = int(result.geometry_body.valid_mask.sum()) if result.geometry_body is not None else None
    obstacle_count = int(result.obstacle_cloud.points.shape[0]) if result.obstacle_cloud is not None else None
    ray_count = int(result.free_space_rays.ranges_m.shape[0]) if result.free_space_rays is not None else None
    out["body_frame_valid_count"] = body_valid_count
    out["obstacle_point_count"] = obstacle_count
    out["free_space_ray_count"] = ray_count
    out["obstacle_matches_body_valid_exactly"] = (obstacle_count == body_valid_count)
    out["rays_match_body_valid_exactly"] = (ray_count == body_valid_count)

    # clearance
    if gf.clearance_evidence is not None:
        support_states = [c.support_state for c in gf.clearance_evidence]
        out["clearance_n_sectors"] = len(gf.clearance_evidence)
        out["clearance_n_supported"] = sum(1 for s in support_states if s == "SUPPORTED")
        out["clearance_n_partially_supported"] = sum(1 for s in support_states if s == "PARTIALLY_SUPPORTED")
        out["clearance_n_no_evidence"] = sum(1 for s in support_states if s == "NO_EVIDENCE")

    # boundary
    if gf.boundary_evidence is not None:
        states = [b.state for b in gf.boundary_evidence]
        out["boundary_n_total"] = len(gf.boundary_evidence)
        out["boundary_n_observed_discontinuity"] = sum(1 for s in states if s == "OBSERVED_DISCONTINUITY")
        out["boundary_n_no_discontinuity"] = sum(1 for s in states if s == "NO_DISCONTINUITY")
        out["boundary_n_insufficient"] = sum(1 for s in states if s == "INSUFFICIENT_EVIDENCE")
        out["boundary_observed_support_fractions"] = [
            (round(b.support_fraction_from, 3), round(b.support_fraction_to, 3))
            for b in gf.boundary_evidence if b.state == "OBSERVED_DISCONTINUITY"
        ]

    # opening — the specific finding under investigation
    if gf.opening_evidence is not None:
        out["opening_n_confirmed"] = len(gf.opening_evidence)
        out["opening_details"] = [
            {
                "row": o.row, "col_start": o.col_start, "col_end": o.col_end,
                "approx_range_m": round(o.approx_range_m, 3),
                "approx_width_m": round(o.approx_width_m, 3),
                "support_fraction": round(o.support_fraction, 4),
                "at_image_boundary": o.at_image_boundary,
            }
            for o in gf.opening_evidence
        ]

    # quality rollup
    if gf.quality is not None:
        q = gf.quality
        out["quality_overall_state"] = q.overall_state
        out["quality_geometry_validity_state"] = q.geometry_validity_state
        out["quality_temporal_consistency_state"] = q.temporal_consistency_state
        out["quality_motion_reliability_state"] = q.motion_reliability_state
        out["quality_persistence_state"] = q.persistence_state
        out["quality_degradation_reasons"] = q.degradation_reasons

    out["geometry_metrics_valid_fraction"] = round(result.geometry_metrics.valid_fraction, 6) if result.geometry_metrics else None

    return out


def run_scenario_A(n_seeds=10):
    """Fully decorrelated / no-correspondence stereo — i.i.d. noise,
    both left and right independently random. NO true correspondence
    exists ANYWHERE. Every seed tried at both 3x3 (default) and 6x8
    grids, to directly re-verify (not assume) whether the pre-I1-fix
    '4 openings/frame at 6x8 on noise' finding still reproduces on the
    CURRENT, fixed source."""
    results = []
    for grid_label, cfg in [
        ("3x3", _full_v1_config(3, 3, 3, 3)),
        ("6x8", _full_v1_config(6, 8, 6, 8)),
    ]:
        pipeline = _pipeline(cfg)
        for seed in range(1, n_seeds + 1):
            fx = make_decorrelated_fixture(seed)
            result = pipeline.process(fx.left, fx.right)
            r = _trace(result, f"A_decorrelated_{grid_label}_seed{seed}")
            r["scenario"] = "A"
            r["grid"] = grid_label
            r["seed"] = seed
            # ground truth: entire frame should be invalid (gt_invalid_mask all True)
            r["gt_false_valid_fraction"] = float(result.valid_disparity_mask.mean())
            results.append(r)
    return results


def run_scenario_B(n_seeds=5):
    """Controlled occlusion boundary — fixtures.py's make_discontinuity_fixture
    with occlusion=True: a near/far depth step with a genuine occlusion
    strip (visible in only one eye) where ground truth says NO valid
    correspondence should exist."""
    results = []
    for grid_label, cfg in [("3x3", _full_v1_config(3, 3, 3, 3)), ("6x8", _full_v1_config(6, 8, 6, 8))]:
        pipeline = _pipeline(cfg)
        for seed in range(1, n_seeds + 1):
            fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=True)
            result = pipeline.process(fx.left, fx.right)
            r = _trace(result, f"B_occlusion_{grid_label}_seed{seed}")
            r["scenario"] = "B"
            r["grid"] = grid_label
            r["seed"] = seed
            # ground-truth false-valid within the occlusion strip specifically
            if fx.gt_invalid_mask is not None:
                false_valid = result.valid_disparity_mask & fx.gt_invalid_mask
                r["gt_false_valid_fraction_in_occlusion_strip"] = float(
                    false_valid.sum() / max(int(fx.gt_invalid_mask.sum()), 1)
                )
            results.append(r)
    return results


def run_scenario_C(n_seeds=5):
    """Weak-texture / no-support region — fixtures.py's scenario C
    (smooth low-frequency-gradient texture, sparse high-frequency content)
    at long range (6m), where correspondence is genuinely marginal."""
    results = []
    for grid_label, cfg in [("3x3", _full_v1_config(3, 3, 3, 3)), ("6x8", _full_v1_config(6, 8, 6, 8))]:
        pipeline = _pipeline(cfg)
        for seed in range(1, n_seeds + 1):
            fx = make_flat_fixture("C", depth_m=6.0, seed=seed)
            result = pipeline.process(fx.left, fx.right)
            r = _trace(result, f"C_weak_texture_6m_{grid_label}_seed{seed}")
            r["scenario"] = "C"
            r["grid"] = grid_label
            r["seed"] = seed
            results.append(r)
    return results


def derive_observable_envelope():
    cfg = PipelineConfig()
    theoretical_dead_zone_px = cfg.num_disparities  # min_disparity=0
    theoretical_observable_fraction = max(0.0, (_W - theoretical_dead_zone_px) / _W)
    return {
        "frame_width_px": _W,
        "min_disparity": cfg.min_disparity,
        "num_disparities": cfg.num_disparities,
        "theoretical_dead_zone_px": theoretical_dead_zone_px,
        "theoretical_whole_frame_observable_fraction": theoretical_observable_fraction,
    }


def main():
    out = {
        "observable_envelope": derive_observable_envelope(),
        "scenario_A_decorrelated": run_scenario_A(),
        "scenario_B_occlusion": run_scenario_B(),
        "scenario_C_weak_texture": run_scenario_C(),
    }
    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i1_stereo_accuracy/results/safety_closure.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {path}")

    print("\n=== Observable envelope ===")
    print(json.dumps(out["observable_envelope"], indent=2))

    for scen_key in ["scenario_A_decorrelated", "scenario_B_occlusion", "scenario_C_weak_texture"]:
        print(f"\n=== {scen_key} ===")
        for r in out[scen_key]:
            openings = r.get("opening_n_confirmed", 0)
            print(
                f"  {r['label']:40s} valid_disp={r['valid_disparity_fraction']:.4f} "
                f"obstacle_pts={r['obstacle_point_count']:>6} rays={r['free_space_ray_count']:>6} "
                f"obstacle==body_valid:{r['obstacle_matches_body_valid_exactly']} "
                f"boundary_OBS={r.get('boundary_n_observed_discontinuity')} "
                f"openings={openings} quality={r.get('quality_overall_state')} "
                f"reasons={r.get('quality_degradation_reasons')}"
            )
            if openings:
                print(f"      OPENING DETAIL: {r['opening_details']}")


if __name__ == "__main__":
    main()
