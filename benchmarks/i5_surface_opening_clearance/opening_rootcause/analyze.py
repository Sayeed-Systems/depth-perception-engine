"""
Phase I5.1 Parts A-C: opening FN root-cause categorization + safe
merge-based discriminator search, prototyped entirely in this script
(zero src/ changes). Reuses the exact scenario set from
benchmarks/i5_surface_opening_clearance/opening/measure.py verbatim.
"""
import json
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.boundary import BoundaryDirection, BoundaryState

from benchmarks.i1_stereo_accuracy.fixtures import make_discontinuity_fixture, make_decorrelated_fixture, W as _W
from benchmarks.i5_surface_opening_clearance.opening.fixtures import make_gap_fixture

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
_GRID_COLS = 6
_GRID_ROWS = 3
_BOUNDS = np.linspace(0, _W, _GRID_COLS + 1).astype(int).tolist()


def _transform():
    return RigidTransform(rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
                           from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY)


def _pipeline(grid_rows=_GRID_ROWS, grid_cols=_GRID_COLS):
    cfg = PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, surface_grid_rows=grid_rows, surface_grid_cols=grid_cols,
        enable_boundary_geometry=True, boundary_grid_rows=grid_rows, boundary_grid_cols=grid_cols,
        enable_opening_geometry=True, enable_geometry_frame=True,
    )
    return DepthPerceptionPipeline(cfg, _CALIB, rectify=False, body_T_camera_left=_transform())


def _scenarios():
    scenarios = []
    scenarios.append(("width_narrow_1cell",
                       dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(160, 213))))
    scenarios.append(("width_medium_2cell",
                       dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(160, 266))))
    scenarios.append(("width_wide_3cell",
                       dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(106, 266))))
    for gap_m, label in [(2.3, "near"), (4.0, "medium"), (6.0, "far")]:
        scenarios.append(("range_" + label,
                           dict(near_left_m=1.5, gap_m=gap_m, near_right_m=1.5, gap_cols=(160, 213))))
    scenarios.append(("straight_on", dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(160, 213))))
    scenarios.append(("asymmetric", dict(near_left_m=1.5, gap_m=4.0, near_right_m=2.5, gap_cols=(160, 213))))
    scenarios.append(("partial_invalid",
                       dict(near_left_m=2.0, gap_m=4.0, near_right_m=2.0, gap_cols=(160, 213),
                            invalid_gap_cols=(175, 195))))
    scenarios.append(("edge_left_absent",
                       dict(near_left_m=None, gap_m=4.0, near_right_m=1.5, gap_cols=(0, 213))))
    scenarios.append(("edge_right_absent",
                       dict(near_left_m=1.5, gap_m=4.0, near_right_m=None, gap_cols=(160, _W))))
    scenarios.append(("ratio_fail",
                       dict(near_left_m=2.0, gap_m=2.5, near_right_m=2.0, gap_cols=(160, 213))))
    return scenarios


# ===================================================================
# Reimplementation of build_opening_evidence's cell/grid construction,
# so we can inspect intermediate state (span fragments, internal split
# edges, reference depths) that the real function doesn't expose.
# ===================================================================
def _build_cells(depth_map, grid_rows, grid_cols):
    valid_depth_mask = depth_map > 0.0
    h, w = depth_map.shape[:2]
    row_bounds = np.linspace(0, h, grid_rows + 1).astype(int)
    col_bounds = np.linspace(0, w, grid_cols + 1).astype(int)
    cells = [[None] * grid_cols for _ in range(grid_rows)]
    for r in range(grid_rows):
        y1, y2 = int(row_bounds[r]), int(row_bounds[r + 1])
        for c in range(grid_cols):
            x1, x2 = int(col_bounds[c]), int(col_bounds[c + 1])
            cell_valid = valid_depth_mask[y1:y2, x1:x2]
            cell_depth = depth_map[y1:y2, x1:x2]
            total_pixels = int(cell_valid.size)
            support_count = int(cell_valid.sum())
            support_fraction = (support_count / total_pixels) if total_pixels else 0.0
            median_depth_m = (
                float(np.median(cell_depth[cell_valid])) if support_count >= 30 else None
            )
            cells[r][c] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "support_fraction": support_fraction, "median_depth_m": median_depth_m}
    return cells


def _build_openings_with_merge(boundary_evidence, depth_map, frame_id, grid_rows, grid_cols,
                                min_support_count, min_range_ratio, focal_length_px,
                                merge_tolerance_m=None, merge_relative_tol=None):
    """Reimplements build_opening_evidence's exact algorithm, with an
    OPTIONAL merge step: an internal flank_edge is excluded from
    splitting a span if the two cells straddling it have near-equal
    median depth (|diff| <= merge_tolerance_m, or relative <=
    merge_relative_tol of the larger depth) -- i.e. "this looks like the
    same gap on both sides, not two real features," using ONLY
    depth_map-derived cell medians (no orientation/planarity signal)."""
    valid_depth_mask = depth_map > 0.0
    h, w = depth_map.shape[:2]
    cells = _build_cells(depth_map, grid_rows, grid_cols)

    right_state = {
        (be.row, be.col): be.state
        for be in boundary_evidence
        if be.direction == BoundaryDirection.RIGHT
    }

    evidence = []
    trace = []  # diagnostic trace of span decisions
    for r in range(grid_rows):
        raw_flank_edges = [
            e for e in range(grid_cols - 1)
            if right_state.get((r, e)) == BoundaryState.OBSERVED_DISCONTINUITY
        ]

        flank_edges = []
        for e in raw_flank_edges:
            a = cells[r][e]["median_depth_m"]
            b = cells[r][e + 1]["median_depth_m"]
            merge = False
            if merge_tolerance_m is not None and a is not None and b is not None:
                if abs(a - b) <= merge_tolerance_m:
                    merge = True
            if merge_relative_tol is not None and a is not None and b is not None:
                denom = max(a, b, 1e-6)
                if abs(a - b) / denom <= merge_relative_tol:
                    merge = True
            if merge:
                trace.append({"row": r, "merged_edge": e, "depth_a": a, "depth_b": b})
                continue
            flank_edges.append(e)

        boundaries = [(-1, False)] + [(e, True) for e in flank_edges] + [(grid_cols - 1, False)]
        for i in range(len(boundaries) - 1):
            left_edge, left_is_real = boundaries[i]
            right_edge, right_is_real = boundaries[i + 1]
            c_start, c_end = left_edge + 1, right_edge
            if c_start > c_end or (not left_is_real and not right_is_real):
                continue
            span_cells = [cells[r][c] for c in range(c_start, c_end + 1)]
            if any(cell["median_depth_m"] is None for cell in span_cells):
                continue
            reference_depths = []
            if left_is_real:
                reference_depths.append(cells[r][c_start - 1]["median_depth_m"])
            if right_is_real:
                reference_depths.append(cells[r][c_end + 1]["median_depth_m"])
            if any(depth is None for depth in reference_depths):
                continue
            reference_depth_m = max(reference_depths)
            if reference_depth_m <= 0.0:
                continue
            if not all(cell["median_depth_m"] >= min_range_ratio * reference_depth_m for cell in span_cells):
                continue
            approx_range_m = float(np.mean([cell["median_depth_m"] for cell in span_cells]))
            support_fraction = float(min(cell["support_fraction"] for cell in span_cells))
            x1, x2 = span_cells[0]["x1"], span_cells[-1]["x2"]
            y1, y2 = span_cells[0]["y1"], span_cells[0]["y2"]
            evidence.append({
                "frame_id": frame_id, "row": r, "col_start": c_start, "col_end": c_end,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "approx_range_m": approx_range_m,
                "approx_width_m": (x2 - x1) * approx_range_m / focal_length_px,
                "approx_height_m": (y2 - y1) * approx_range_m / focal_length_px,
                "support_fraction": support_fraction,
                "at_image_boundary": (not left_is_real) or (not right_is_real),
            })
    return evidence, trace


def _openings_overlapping(openings, c0, c1):
    return [o for o in openings if not (o["x2"] <= c0 or o["x1"] >= c1)]


FX = _CALIB.Q[2, 3]


def run_candidate(label, merge_tolerance_m=None, merge_relative_tol=None, n_seeds=5):
    pipeline = _pipeline()
    records = []
    for name, kwargs in _scenarios():
        for seed in range(1, n_seeds + 1):
            fx, gt = make_gap_fixture(seed=seed, texture_scale=6, **kwargs)
            result = pipeline.process(fx.left, fx.right)
            gf = result.geometry_frame
            openings_default, trace = _build_openings_with_merge(
                gf.boundary_evidence, result.depth_map, "camera_optical_left",
                _GRID_ROWS, _GRID_COLS, 30, 1.5, FX,
                merge_tolerance_m=merge_tolerance_m, merge_relative_tol=merge_relative_tol,
            )
            c0px, c1px = gt["col_span"]
            found = _openings_overlapping(openings_default, c0px, c1px)
            gt_positive = kwargs.get("near_left_m") is not None or kwargs.get("near_right_m") is not None
            is_ratio_fail = name == "ratio_fail"
            gt_expect_confirm = gt_positive and not is_ratio_fail
            rec = {"scenario": name, "seed": seed, "n_found": len(found),
                   "gt_expect_confirm": gt_expect_confirm, "n_merges": len(trace), "trace": trace}
            if found:
                best = found[0]
                rec.update({
                    "approx_range_m": best["approx_range_m"], "approx_width_m": best["approx_width_m"],
                    "range_rel_err_pct": 100.0 * abs(best["approx_range_m"] - gt["range_m"]) / gt["range_m"],
                    "width_rel_err_pct": 100.0 * abs(best["approx_width_m"] - gt["width_m"]) / max(gt["width_m"], 1e-6),
                })
            records.append(rec)

    # negative fixtures
    neg_false = 0
    neg_total = 0
    for grid_rc in (_GRID_ROWS, 6), :
        pass
    pipeline_3x3 = _pipeline(3, 3)
    pipeline_6x8 = _pipeline(3, 6)
    for pl, glabel in [(pipeline_3x3, "3x3"), (pipeline_6x8, "3x6")]:
        for seed in range(1, 21):
            fxn = make_decorrelated_fixture(seed)
            resultn = pl.process(fxn.left, fxn.right)
            gr, gc = (3, 3) if glabel == "3x3" else (3, 6)
            op, _ = _build_openings_with_merge(
                resultn.geometry_frame.boundary_evidence, resultn.depth_map, "x", gr, gc, 30, 1.5, FX,
                merge_tolerance_m=merge_tolerance_m, merge_relative_tol=merge_relative_tol,
            )
            neg_total += 1
            if len(op) > 0:
                neg_false += 1
            records.append({"scenario": f"negative_noise_{glabel}", "seed": seed, "n_found": len(op),
                             "gt_expect_confirm": False})

    # single-step-not-opening
    for seed in range(1, 6):
        fxs = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=False)
        results = pipeline.process(fxs.left, fxs.right)
        op, _ = _build_openings_with_merge(
            results.geometry_frame.boundary_evidence, results.depth_map, "x", _GRID_ROWS, _GRID_COLS, 30, 1.5, FX,
            merge_tolerance_m=merge_tolerance_m, merge_relative_tol=merge_relative_tol,
        )
        records.append({"scenario": "single_step_not_opening", "seed": seed, "n_found": len(op),
                         "gt_expect_confirm": None})

    # ---- TP/FP/FN/TN scoring ----
    tp = fp = fn = tn = 0
    width_errs, range_errs = [], []
    for r in records:
        gt_c = r["gt_expect_confirm"]
        if gt_c is None:
            continue
        found = r["n_found"] > 0
        if gt_c and found:
            tp += 1
            if "width_rel_err_pct" in r:
                width_errs.append(r["width_rel_err_pct"])
                range_errs.append(r["range_rel_err_pct"])
        elif gt_c and not found:
            fn += 1
        elif not gt_c and found:
            fp += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"\n=== Candidate: {label} (tol_abs={merge_tolerance_m}, tol_rel={merge_relative_tol}) ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  precision={precision:.4f} recall={recall:.4f}")
    if width_errs:
        print(f"width_rel_err_pct: median={np.median(width_errs):.3f} max={np.max(width_errs):.3f}")
        print(f"range_rel_err_pct: median={np.median(range_errs):.3f} max={np.max(range_errs):.3f}")
    print(f"negative false-opening rate: {neg_false}/{neg_total}")

    # FN cause breakdown for THIS candidate
    fn_causes = {}
    for r in records:
        if r.get("gt_expect_confirm") and r["n_found"] == 0:
            fn_causes.setdefault(r["scenario"], 0)
            fn_causes[r["scenario"]] += 1
    print("FN by scenario:", fn_causes)

    return {"label": label, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "width_err_median": float(np.median(width_errs)) if width_errs else None,
            "range_err_median": float(np.median(range_errs)) if range_errs else None,
            "neg_false": neg_false, "neg_total": neg_total, "fn_causes": fn_causes, "records": records}


if __name__ == "__main__":
    results = {}
    results["baseline_no_merge"] = run_candidate("baseline (no merge, = current shipped opening.py)")
    results["merge_abs_0.01m"] = run_candidate("merge if |depth_a-depth_b| <= 0.01m", merge_tolerance_m=0.01)
    results["merge_abs_0.05m"] = run_candidate("merge if |depth_a-depth_b| <= 0.05m", merge_tolerance_m=0.05)
    results["merge_abs_0.10m"] = run_candidate("merge if |depth_a-depth_b| <= 0.10m", merge_tolerance_m=0.10)
    results["merge_rel_0.02"] = run_candidate("merge if rel diff <= 2%", merge_relative_tol=0.02)
    results["merge_rel_0.05"] = run_candidate("merge if rel diff <= 5%", merge_relative_tol=0.05)
    results["merge_rel_0.10"] = run_candidate("merge if rel diff <= 10%", merge_relative_tol=0.10)

    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i5_surface_opening_clearance/opening_rootcause/results/candidates.json"
    with open(path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nWrote summary to {path}")
