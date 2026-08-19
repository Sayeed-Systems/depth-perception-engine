"""Phase I5 Part A -- run slanted/fronto/mixed/partial-invalid fixtures
through the REAL, unmodified DepthPerceptionPipeline (rectify=False,
matching this repo's own established synthetic-fixture precedent) and
measure SurfaceEvidence against analytically-derived ground truth."""
import json
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

from benchmarks.i1_stereo_accuracy.fixtures import make_flat_fixture, make_discontinuity_fixture, W, H
from benchmarks.i5_surface_opening_clearance.surface.fixtures import make_slanted_plane_pair

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _cfg(grid_rc=3):
    return PipelineConfig(
        enable_geometry=True, enable_surface_geometry=True,
        surface_grid_rows=grid_rc, surface_grid_cols=grid_rc,
        enable_geometry_frame=False,  # only need geometry_body -> surface_evidence; skip the rest
    )


def _pipeline(grid_rc=3):
    return DepthPerceptionPipeline(_cfg(grid_rc), _CALIB, rectify=False, body_T_camera_left=_transform())


def _angular_error_deg(expected, measured):
    if measured is None:
        return None
    cos_angle = float(np.clip(np.dot(expected, measured), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(cos_angle))))  # sign already resolved by viewpoint convention; abs() as a defensive tie-break only


records = []


def _run_slanted(label, theta_x, theta_y, z0, texture_scale, seed, grid_rc=1):
    pipeline = _pipeline(grid_rc)
    left, right, expected_normal, _ = make_slanted_plane_pair(theta_x, theta_y, z0, texture_scale, seed)
    result = pipeline.process(left, right)
    se = result.surface_evidence or []
    for s in se:
        err = _angular_error_deg(expected_normal, s.normal)
        records.append({
            "label": label, "theta_x": theta_x, "theta_y": theta_y, "z0": z0,
            "texture_scale": texture_scale, "seed": seed, "grid_rc": grid_rc,
            "row": s.row, "col": s.col,
            "support_fraction": s.support_fraction,
            "planarity": s.planarity,
            "angular_error_deg": err,
        })


def _run_fronto(label, depth_m, texture_scale, seed):
    pipeline = _pipeline(1)
    fx = make_flat_fixture("A", depth_m=depth_m, seed=seed)
    # override texture scale by reconstructing with desired scale if needed;
    # make_flat_fixture already uses fixed per-scenario scale, acceptable
    # for the "does depth/range affect error" question.
    result = pipeline.process(fx.left, fx.right)
    expected_normal = np.array([0.0, 0.0, -1.0])  # fronto-parallel, identity BODY rotation
    se = result.surface_evidence or []
    for s in se:
        err = _angular_error_deg(expected_normal, s.normal)
        records.append({
            "label": label, "theta_x": 0.0, "theta_y": 0.0, "z0": depth_m,
            "texture_scale": texture_scale, "seed": seed, "grid_rc": 1,
            "row": s.row, "col": s.col,
            "support_fraction": s.support_fraction, "planarity": s.planarity,
            "angular_error_deg": err,
        })


def _run_mixed_and_partial():
    pipeline = _pipeline(3)
    # Mixed-surface cell: genuine near/far step (no occlusion) -- the
    # 3x3 grid's straddling cell mixes two real planes.
    for seed in range(1, 4):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=False)
        result = pipeline.process(fx.left, fx.right)
        for s in (result.surface_evidence or []):
            records.append({
                "label": "mixed_surface_cell", "theta_x": None, "theta_y": None, "z0": None,
                "texture_scale": None, "seed": seed, "grid_rc": 3,
                "row": s.row, "col": s.col,
                "support_fraction": s.support_fraction, "planarity": s.planarity,
                "angular_error_deg": None,  # no single "true" normal for a straddling cell
                "normal": s.normal.tolist() if s.normal is not None else None,
            })

    # Partial-invalid-support cell: a textured plane with a large
    # textureless (zero-disparity-signal) patch punched into part of it.
    from benchmarks.i1_stereo_accuracy.fixtures import _low_freq_canvas, _remap_by_disparity, _to_bgr, disparity_for_depth
    from benchmarks.i1_stereo_accuracy.fixtures import _MAX_SHIFT_MARGIN
    for seed in range(1, 4):
        d = disparity_for_depth(2.0)
        canvas_w = W + _MAX_SHIFT_MARGIN
        canvas = _low_freq_canvas(canvas_w, H, 2, seed)
        x0 = _MAX_SHIFT_MARGIN // 2
        left_gray = canvas[:, x0:x0 + W].copy()
        disp_map = np.full((H, W), d, dtype=np.float32)
        right_gray = _remap_by_disparity(canvas, disp_map, x0)
        # Punch a flat, textureless patch into BOTH eyes over a big chunk
        # of the right half -- kills real correspondence there without
        # faking a second depth.
        left_gray[:, 200:280] = 128
        right_gray[:, 200:280] = 128
        result = pipeline.process(_to_bgr(left_gray), _to_bgr(right_gray))
        for s in (result.surface_evidence or []):
            records.append({
                "label": "partial_invalid_cell", "theta_x": 0.0, "theta_y": 0.0, "z0": 2.0,
                "texture_scale": 2, "seed": seed, "grid_rc": 3,
                "row": s.row, "col": s.col,
                "support_fraction": s.support_fraction, "planarity": s.planarity,
                "angular_error_deg": _angular_error_deg(np.array([0.0, 0.0, -1.0]), s.normal),
            })


def main():
    # 1. Fronto-parallel at multiple ranges
    for depth_m in (1.0, 2.0, 4.0):
        for seed in (1, 2, 3):
            _run_fronto(f"fronto_{depth_m}m", depth_m, texture_scale=2, seed=seed)

    # 2. Yaw-slanted, pitch-slanted, combined -- multiple angles, texture levels
    for theta in (15.0, 30.0):
        for seed in (1, 2, 3):
            _run_slanted(f"yaw_{theta}deg", 0.0, theta, 2.0, texture_scale=2, seed=seed)
            _run_slanted(f"pitch_{theta}deg", theta, 0.0, 2.0, texture_scale=2, seed=seed)
        _run_slanted(f"combined_{theta}deg", theta * 0.6, theta * 0.6, 2.0, texture_scale=2, seed=1)

    # 3. Texture-level sweep at fixed slant
    for texture_scale, label in ((2, "high_texture"), (6, "moderate_texture"), (24, "weak_texture")):
        for seed in (1, 2, 3):
            _run_slanted(f"yaw15_{label}", 0.0, 15.0, 2.0, texture_scale=texture_scale, seed=seed)

    # 4. Mixed-surface / partial-invalid
    _run_mixed_and_partial()

    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i5_surface_opening_clearance/surface/results/measure.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {len(records)} records to {path}")

    # Summary
    import statistics
    high_plan = [r for r in records if r["planarity"] is not None and r["planarity"] >= 0.95
                 and r["angular_error_deg"] is not None]
    errs = [r["angular_error_deg"] for r in high_plan]
    print(f"\nHigh-planarity (>=0.95) cells with known ground truth: n={len(errs)}")
    if errs:
        print(f"  min={min(errs):.4f} median={statistics.median(errs):.4f} "
              f"p95={sorted(errs)[int(0.95*len(errs))-1]:.4f} max={max(errs):.4f}")

    low_plan = [r for r in records if r["planarity"] is not None and r["planarity"] < 0.95]
    print(f"\nLow-planarity (<0.95) cells: n={len(low_plan)}")
    for r in low_plan[:30]:
        print(f"  {r['label']} seed={r['seed']} row={r['row']} col={r['col']} "
              f"planarity={r['planarity']:.4f} support={r['support_fraction']:.4f} "
              f"angular_error={r.get('angular_error_deg')}")

    none_normal = [r for r in records if r["planarity"] is None]
    print(f"\nNone-normal (insufficient support) cells: n={len(none_normal)}")

    print("\nPer-label mean angular error (all cells, incl. low-planarity, for reference):")
    labels = sorted(set(r["label"] for r in records))
    for lbl in labels:
        vals = [r["angular_error_deg"] for r in records if r["label"] == lbl and r["angular_error_deg"] is not None]
        plans = [r["planarity"] for r in records if r["label"] == lbl and r["planarity"] is not None]
        if vals:
            print(f"  {lbl}: n={len(vals)} mean_err={statistics.mean(vals):.4f}deg "
                  f"mean_planarity={statistics.mean(plans):.4f}")


if __name__ == "__main__":
    main()
