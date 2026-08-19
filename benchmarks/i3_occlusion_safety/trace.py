"""
Phase I3, Step 1 — precise reproduction/tracing of the occlusion-boundary
contamination mechanism across 6 fixtures. Zero src/ changes — imports the
REAL DisparityEngine/DepthEstimator/DepthPerceptionPipeline classes directly
(not a mirrored/reimplemented SGBM construction, to avoid the staleness risk
benchmarks/i1_stereo_accuracy/measure.py's own now-stale current_sgbm_params()
mirror already shows relative to the actual current src/ config).

IMPORTANT: DepthPerceptionPipeline's `rectify` parameter defaults to True,
which corrupts synthetic (never-actually-captured) fixture images by
applying the real camera's lens-distortion-correction maps to them
(confirmed and documented earlier this session: benchmarks/i1_stereo_accuracy/
safety_closure.py's own _pipeline() had exactly this bug). Every
DepthPerceptionPipeline construction in this file uses rectify=False.
"""
import json
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.stereo.disparity_engine import DisparityEngine
from depth_perception_engine.depth.depth_estimator import DepthEstimator

from benchmarks.i3_occlusion_safety.fixtures import (
    make_A_clean_discontinuity, make_B_occlusion_strip, make_C_disocclusion_far_side,
    make_D_mixed_valid_invalid_cell, make_E_decorrelated, make_F_genuine_obstacle_boundary,
    W, H,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")

# Direct disparity/depth engines — real classes, real (current, post-I1)
# PipelineConfig defaults (block_size=9, and DisparityEngine's own P1/P2
# channel-correct multiplier + uniquenessRatio=20 baked into its __init__).
_CFG_DEFAULTS = PipelineConfig()
_DISP_ENGINE = DisparityEngine(
    min_disparity=_CFG_DEFAULTS.min_disparity,
    num_disparities=_CFG_DEFAULTS.num_disparities,
    block_size=_CFG_DEFAULTS.block_size,
)
_DEPTH_ESTIMATOR = DepthEstimator(_CALIB.Q)


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _full_v1_config(surface_rc=3, boundary_rc=3):
    return PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, surface_grid_rows=surface_rc, surface_grid_cols=surface_rc,
        enable_boundary_geometry=True, boundary_grid_rows=boundary_rc, boundary_grid_cols=boundary_rc,
        enable_opening_geometry=True, enable_geometry_frame=True,
    )


def _pipeline(cfg):
    return DepthPerceptionPipeline(cfg, _CALIB, rectify=False, body_T_camera_left=_transform())


def _direct_disparity_depth(fx):
    """Raw disparity/depth via the REAL DisparityEngine/DepthEstimator
    classes directly — no pipeline, no rectification (matches fixtures'
    own unrectified construction)."""
    raw_disparity, _ = _DISP_ENGINE.compute_disparity(fx.left, fx.right, compute_visualization=False)
    depth = _DEPTH_ESTIMATOR.estimate(raw_disparity)
    valid_disp = raw_disparity > 0.0
    valid_depth = depth > 0.0
    return raw_disparity, depth, valid_disp, valid_depth


def _column_profile(raw_disparity, valid_disp, col_lo, col_hi, margin=15):
    """Median disparity per column across [col_lo-margin, col_hi+margin),
    plus valid-fraction per column, to see the exact shape of a transition
    or strip (ramp / snap / noisy-scatter)."""
    lo = max(0, col_lo - margin)
    hi = min(W, col_hi + margin)
    profile = []
    for c in range(lo, hi):
        col_vals = raw_disparity[:, c]
        col_valid = valid_disp[:, c]
        med = float(np.median(col_vals[col_valid])) if col_valid.any() else None
        profile.append({"col": c, "valid_fraction": float(col_valid.mean()), "median_disparity": med})
    return profile


def _local_support_5x5(valid_disp, row_c, col_c):
    r0, r1 = max(0, row_c - 2), min(H, row_c + 3)
    c0, c1 = max(0, col_c - 2), min(W, col_c + 3)
    window = valid_disp[r0:r1, c0:c1]
    return {"valid_count": int(window.sum()), "total": int(window.size), "fraction": float(window.mean())}


def _boundary_evidence_summary(gf, col_lo, col_hi):
    out = []
    if gf.boundary_evidence is None:
        return out
    for b in gf.boundary_evidence:
        if b.x2 < col_lo or b.x1 > col_hi:
            continue
        out.append({
            "row": b.row, "col": b.col, "direction": b.direction,
            "x1": b.x1, "x2": b.x2, "state": b.state,
            "support_fraction_from": round(b.support_fraction_from, 4),
            "support_fraction_to": round(b.support_fraction_to, 4),
            "depth_step_m": round(b.depth_step_m, 4) if b.depth_step_m is not None else None,
        })
    return out


def _surface_evidence_summary(gf, col_lo, col_hi):
    out = []
    if gf.surface_evidence is None:
        return out
    for s in gf.surface_evidence:
        if s.x2 < col_lo or s.x1 > col_hi:
            continue
        out.append({
            "row": s.row, "col": s.col, "x1": s.x1, "x2": s.x2,
            "support_count": s.support_count, "support_fraction": round(s.support_fraction, 4),
            "planarity": round(s.planarity, 4) if s.planarity is not None else None,
        })
    return out


def _clearance_summary(gf, col_lo, col_hi):
    out = []
    if gf.clearance_evidence is None:
        return out
    for c in gf.clearance_evidence:
        if c.x2 < col_lo or c.x1 > col_hi:
            continue
        out.append({
            "index": c.index, "x1": c.x1, "x2": c.x2,
            "support_state": c.support_state, "coverage_fraction": round(c.coverage_fraction, 4),
            "nearest_distance_m": round(c.nearest_distance_m, 4) if c.nearest_distance_m is not None else None,
        })
    return out


def _opening_summary(gf, col_lo, col_hi):
    out = []
    if gf.opening_evidence is None:
        return out
    for o in gf.opening_evidence:
        if o.x2 < col_lo or o.x1 > col_hi:
            continue
        out.append({
            "row": o.row, "col_start": o.col_start, "col_end": o.col_end,
            "support_fraction": round(o.support_fraction, 4), "at_image_boundary": o.at_image_boundary,
        })
    return out


def trace_fixture(fx, strip_cols, label, grid_rc=3):
    """strip_cols: (lo, hi) column range of the region under scrutiny
    (occlusion strip / mixed-cell boundary / box edge)."""
    col_lo, col_hi = strip_cols
    raw_disp, depth, valid_disp, valid_depth = _direct_disparity_depth(fx)

    result_row = {
        "label": label, "scenario": fx.scenario,
        "strip_cols": [col_lo, col_hi],
        "column_profile": _column_profile(raw_disp, valid_disp, col_lo, col_hi),
        "local_support_center": _local_support_5x5(valid_disp, H // 2, (col_lo + col_hi) // 2),
    }

    if fx.gt_invalid_mask is not None:
        strip_mask = np.zeros((H, W), dtype=bool)
        strip_mask[:, col_lo:col_hi] = True
        gt_bad_in_strip = fx.gt_invalid_mask & strip_mask
        false_valid_in_strip = valid_disp & gt_bad_in_strip
        result_row["strip_total_px"] = int(strip_mask.sum())
        result_row["gt_invalid_in_strip_px"] = int(gt_bad_in_strip.sum())
        result_row["false_valid_in_strip_px"] = int(false_valid_in_strip.sum())
        result_row["false_valid_in_strip_fraction"] = (
            float(false_valid_in_strip.sum() / max(int(gt_bad_in_strip.sum()), 1))
        )

    # Full-pipeline GeometryFrame trace
    cfg = _full_v1_config(grid_rc, grid_rc)
    pipeline = _pipeline(cfg)
    pres = pipeline.process(fx.left, fx.right)
    gf = pres.geometry_frame

    result_row["quality_overall_state"] = gf.quality.overall_state if gf.quality else None
    result_row["quality_geometry_validity_state"] = gf.quality.geometry_validity_state if gf.quality else None
    result_row["whole_frame_valid_fraction"] = round(pres.geometry_metrics.valid_fraction, 6) if pres.geometry_metrics else None
    result_row["boundary_evidence_near_strip"] = _boundary_evidence_summary(gf, col_lo, col_hi)
    result_row["surface_evidence_near_strip"] = _surface_evidence_summary(gf, col_lo, col_hi)
    result_row["clearance_near_strip"] = _clearance_summary(gf, col_lo, col_hi)
    result_row["opening_near_strip"] = _opening_summary(gf, col_lo, col_hi)

    # Contaminated obstacle/free-space-ray point count attributable to the
    # strip: since obstacle_cloud/free_space_rays are gated 1:1 on
    # geometry_body.valid_mask (the proven UNKNOWN-invariant), the count of
    # valid pixels within the strip's column range in the ORGANIZED
    # body-frame cloud equals exactly how many obstacle points / rays that
    # strip contributes (no need to search the unorganized ObstacleCloud).
    if pres.geometry_body is not None:
        body_valid = pres.geometry_body.valid_mask
        strip_body_valid = body_valid[:, col_lo:col_hi]
        result_row["contaminated_obstacle_points_from_strip"] = int(strip_body_valid.sum())
        result_row["contaminated_free_space_rays_from_strip"] = int(strip_body_valid.sum())
        result_row["total_obstacle_points"] = int(pres.obstacle_cloud.points.shape[0]) if pres.obstacle_cloud is not None else None
        result_row["total_free_space_rays"] = int(pres.free_space_rays.ranges_m.shape[0]) if pres.free_space_rays is not None else None

    return result_row


def main():
    out = {}

    # A. Clean discontinuity — the transition itself is at W//2, no strip;
    # trace a window around the boundary column.
    out["A_clean_discontinuity"] = [
        trace_fixture(make_A_clean_discontinuity(seed), (W // 2 - 5, W // 2 + 5), f"A_seed{seed}")
        for seed in (1, 2, 3)
    ]

    # B. Occlusion strip — near side, immediately left of W//2.
    fx_b1 = make_B_occlusion_strip(1)
    strip_lo_b, strip_hi_b = np.where(fx_b1.gt_invalid_mask.any(axis=0))[0][[0, -1]]
    out["B_occlusion_strip"] = [
        trace_fixture(make_B_occlusion_strip(seed), (int(strip_lo_b), int(strip_hi_b) + 1), f"B_seed{seed}")
        for seed in (1, 2, 3)
    ]

    # C. Dis-occlusion strip — far side (mirrored), immediately right of W//2.
    fx_c1 = make_C_disocclusion_far_side(1)
    strip_lo_c, strip_hi_c = np.where(fx_c1.gt_invalid_mask.any(axis=0))[0][[0, -1]]
    out["C_disocclusion_strip"] = [
        trace_fixture(make_C_disocclusion_far_side(seed), (int(strip_lo_c), int(strip_hi_c) + 1), f"C_seed{seed}")
        for seed in (1, 2, 3)
    ]

    # D. Mixed valid/invalid cell — boundary at W//2, invalid on the right half.
    out["D_mixed_cell"] = [
        trace_fixture(make_D_mixed_valid_invalid_cell(seed), (W // 2 - 5, W // 2 + 5), f"D_seed{seed}")
        for seed in (1, 2, 3)
    ]

    # E. Pure decorrelated — no meaningful "strip", trace the whole-frame center.
    out["E_decorrelated"] = [
        trace_fixture(make_E_decorrelated(seed), (W // 2 - 20, W // 2 + 20), f"E_seed{seed}")
        for seed in (1, 2, 3)
    ]

    # F. Genuine two-sided obstacle boundary — box edges at col 100 and 220.
    out["F_obstacle_boundary_left_edge"] = [
        trace_fixture(make_F_genuine_obstacle_boundary(seed), (95, 105), f"F_left_seed{seed}")
        for seed in (1, 2, 3)
    ]
    out["F_obstacle_boundary_right_edge"] = [
        trace_fixture(make_F_genuine_obstacle_boundary(seed), (215, 225), f"F_right_seed{seed}")
        for seed in (1, 2, 3)
    ]

    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i3_occlusion_safety/results/trace.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {path}")

    for key, rows in out.items():
        print(f"\n=== {key} ===")
        for r in rows:
            print(
                f"  {r['label']:20s} strip={r['strip_cols']} quality={r.get('quality_overall_state')} "
                f"whole_valid={r.get('whole_frame_valid_fraction')} "
                f"false_valid_in_strip_frac={r.get('false_valid_in_strip_fraction')} "
                f"contaminated_obstacle_pts={r.get('contaminated_obstacle_points_from_strip')} "
                f"n_boundary_near_strip={len(r.get('boundary_evidence_near_strip', []))} "
                f"n_opening_near_strip={len(r.get('opening_near_strip', []))}"
            )
            for b in r.get("boundary_evidence_near_strip", []):
                print(f"      BOUNDARY: {b}")


if __name__ == "__main__":
    main()
