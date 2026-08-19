import json
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

from benchmarks.i1_stereo_accuracy.fixtures import make_discontinuity_fixture, make_decorrelated_fixture, W as _W
from benchmarks.i5_surface_opening_clearance.opening.fixtures import make_gap_fixture

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
_GRID_COLS = 6
_GRID_ROWS = 3
# np.linspace(0, 320, 7).astype(int) -> [0, 53, 106, 160, 213, 266, 320]
_BOUNDS = np.linspace(0, _W, _GRID_COLS + 1).astype(int).tolist()


def _transform():
    return RigidTransform(rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
                           from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY)


def _pipeline():
    cfg = PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, surface_grid_rows=_GRID_ROWS, surface_grid_cols=_GRID_COLS,
        enable_boundary_geometry=True, boundary_grid_rows=_GRID_ROWS, boundary_grid_cols=_GRID_COLS,
        enable_opening_geometry=True, enable_geometry_frame=True,
    )
    return DepthPerceptionPipeline(cfg, _CALIB, rectify=False, body_T_camera_left=_transform())


def _openings_overlapping(gf, c0, c1):
    return [o for o in (gf.opening_evidence or []) if not (o.x2 <= c0 or o.x1 >= c1)]


def run():
    pipeline = _pipeline()
    records = []

    # Cell-aligned column plans, using _BOUNDS = [0,53,106,160,213,266,320].
    # Cell 0,1 are (mostly) inside the 128px structural dead zone -> never
    # a usable "real" left flank at this grid; cell 2 [106,160) is HALF
    # dead-zone (106-128) / half real (128-160) -> marginal; cells 3,4,5
    # are fully real. All plans below place the LEFT flank/edge-absence
    # starting no earlier than cell 2, and the gap/right-flank in cells
    # that are fully real.
    scenarios = []

    # 1. width sweep, gap centered around cell 3, range=4m, flanks=2m both sides
    scenarios.append(("width_narrow_1cell",
                       dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(160, 213))))
    scenarios.append(("width_medium_2cell",
                       dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(160, 266))))
    # wide: gap spans cells 3,4; near_right only cell 5
    scenarios.append(("width_wide_3cell",
                       dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(106, 266))))

    # 2. range sweep, near flanks=1.5m, single-cell gap (160,213), ratio always clears 1.5
    for gap_m, label in [(2.3, "near"), (4.0, "medium"), (6.0, "far")]:
        scenarios.append(("range_" + label,
                           dict(near_left_m=1.5, gap_m=gap_m, near_right_m=1.5, gap_cols=(160, 213))))

    # 3. straight-on vs asymmetric flanks
    scenarios.append(("straight_on", dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(160, 213))))
    scenarios.append(("asymmetric", dict(near_left_m=1.5, gap_m=4.0, near_right_m=2.5, gap_cols=(160, 213))))

    # 4. partial invalid support inside gap cell
    scenarios.append(("partial_invalid",
                       dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(160, 213),
                            invalid_gap_cols=(175, 195))))

    # 5. image-edge truncation: left flank ABSENT (gap starts at col 0 -> only right flank real)
    scenarios.append(("edge_left_absent",
                       dict(near_left_m=None, gap_m=4.0, near_right_m=1.5, gap_cols=(0, 213))))
    # right flank ABSENT (gap runs to image edge -> only left flank real)
    scenarios.append(("edge_right_absent",
                       dict(near_left_m=1.5, gap_m=4.0, near_right_m=None, gap_cols=(160, _W))))

    # 6. ratio-boundary negative: gap not far enough beyond flank (ratio<1.5) -> should NOT confirm
    scenarios.append(("ratio_fail",
                       dict(near_left_m=2.0, gap_m=2.5, near_right_m=2.0, gap_cols=(160, 213))))

    n_seeds = 5
    for name, kwargs in scenarios:
        for seed in range(1, n_seeds + 1):
            fx, gt = make_gap_fixture(seed=seed, texture_scale=6, **kwargs)
            result = pipeline.process(fx.left, fx.right)
            gf = result.geometry_frame
            c0px, c1px = gt["col_span"]
            found = _openings_overlapping(gf, c0px, c1px)
            gt_positive = kwargs.get("near_left_m") is not None or kwargs.get("near_right_m") is not None
            is_ratio_fail = name == "ratio_fail"
            rec = {"scenario": name, "seed": seed, "n_found": len(found), "gt": gt,
                   "gt_expect_confirm": (gt_positive and not is_ratio_fail)}
            if found:
                best = found[0]
                rec.update({
                    "approx_range_m": best.approx_range_m, "approx_width_m": best.approx_width_m,
                    "support_fraction": best.support_fraction, "at_image_boundary": best.at_image_boundary,
                    "range_abs_err": abs(best.approx_range_m - gt["range_m"]),
                    "range_rel_err_pct": 100.0 * abs(best.approx_range_m - gt["range_m"]) / gt["range_m"],
                    "width_abs_err": abs(best.approx_width_m - gt["width_m"]),
                    "width_rel_err_pct": 100.0 * abs(best.approx_width_m - gt["width_m"]) / max(gt["width_m"], 1e-6),
                })
            records.append(rec)

    for seed in range(1, 21):
        fx = make_decorrelated_fixture(seed)
        result = pipeline.process(fx.left, fx.right)
        n = len(result.geometry_frame.opening_evidence or [])
        records.append({"scenario": "negative_noise", "seed": seed, "n_found": n, "gt_expect_confirm": False})

    for seed in range(1, 6):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=False)
        result = pipeline.process(fx.left, fx.right)
        openings = result.geometry_frame.opening_evidence or []
        records.append({
            "scenario": "single_step_not_opening", "seed": seed, "n_found": len(openings),
            "gt_expect_confirm": None,
            "details": [{"col_start": o.col_start, "col_end": o.col_end, "range_m": o.approx_range_m,
                          "width_m": o.approx_width_m, "at_boundary": o.at_image_boundary,
                          "support": o.support_fraction} for o in openings],
        })

    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i5_surface_opening_clearance/opening/results/measure.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Wrote {len(records)} records to {path}")
    return records


if __name__ == "__main__":
    run()
