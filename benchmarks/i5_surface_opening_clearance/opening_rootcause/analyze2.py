"""Combined candidate: depth-equality merge for spurious internal splits,
PLUS explicit dead-zone (None-median) hard-boundary splitting, so a span
can never straddle an unsupported (INSUFFICIENT_EVIDENCE) cell in the
first place -- instead of only rejecting the whole span after the fact."""
import sys, json
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")
import numpy as np
from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.boundary import BoundaryDirection, BoundaryState
from benchmarks.i1_stereo_accuracy.fixtures import make_discontinuity_fixture, make_decorrelated_fixture, W as _W
from benchmarks.i5_surface_opening_clearance.opening.fixtures import make_gap_fixture
from benchmarks.i5_surface_opening_clearance.opening_rootcause.analyze import (
    _pipeline, _scenarios, _build_cells, _openings_overlapping, FX, _GRID_ROWS, _GRID_COLS,
)


def _build_openings_v2(boundary_evidence, depth_map, frame_id, grid_rows, grid_cols,
                        min_support_count, min_range_ratio, focal_length_px,
                        merge_tolerance_m=None, split_at_none=False):
    cells = _build_cells(depth_map, grid_rows, grid_cols)
    right_state = {(be.row, be.col): be.state for be in boundary_evidence if be.direction == BoundaryDirection.RIGHT}
    evidence = []
    for r in range(grid_rows):
        raw_flank_edges = [e for e in range(grid_cols - 1) if right_state.get((r, e)) == BoundaryState.OBSERVED_DISCONTINUITY]

        flank_edges = []
        for e in raw_flank_edges:
            a = cells[r][e]["median_depth_m"]
            b = cells[r][e + 1]["median_depth_m"]
            if merge_tolerance_m is not None and a is not None and b is not None and abs(a - b) <= merge_tolerance_m:
                continue  # merged away -- not a real split
            flank_edges.append(e)

        # Hard splits: wherever one side has median=None and the other doesn't
        # (or both None), that edge can NEVER be crossed by a span -- distinct
        # from flank_edges (which mark CONFIRMED discontinuities); this marks
        # "structurally cannot be evaluated together at all."
        hard_splits = set()
        if split_at_none:
            for e in range(grid_cols - 1):
                a_none = cells[r][e]["median_depth_m"] is None
                b_none = cells[r][e + 1]["median_depth_m"] is None
                if a_none != b_none or a_none:  # differing support, or both None (can't bridge either)
                    hard_splits.add(e)

        all_split_points = sorted(set(flank_edges) | hard_splits)
        is_real_flank = {e: (e in flank_edges) for e in all_split_points}

        boundaries = [(-1, False)] + [(e, is_real_flank[e]) for e in all_split_points] + [(grid_cols - 1, False)]
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
            evidence.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                              "approx_range_m": approx_range_m,
                              "approx_width_m": (x2 - x1) * approx_range_m / focal_length_px,
                              "support_fraction": support_fraction,
                              "at_image_boundary": (not left_is_real) or (not right_is_real)})
    return evidence


def run_candidate(label, merge_tolerance_m=None, split_at_none=False, n_seeds=5):
    pipeline = _pipeline()
    records = []
    for name, kwargs in _scenarios():
        for seed in range(1, n_seeds + 1):
            fx, gt = make_gap_fixture(seed=seed, texture_scale=6, **kwargs)
            result = pipeline.process(fx.left, fx.right)
            gf = result.geometry_frame
            openings = _build_openings_v2(gf.boundary_evidence, result.depth_map, "x", _GRID_ROWS, _GRID_COLS,
                                           30, 1.5, FX, merge_tolerance_m=merge_tolerance_m, split_at_none=split_at_none)
            c0px, c1px = gt["col_span"]
            found = _openings_overlapping(openings, c0px, c1px)
            gt_positive = kwargs.get("near_left_m") is not None or kwargs.get("near_right_m") is not None
            gt_expect_confirm = gt_positive and name != "ratio_fail"
            rec = {"scenario": name, "seed": seed, "n_found": len(found), "gt_expect_confirm": gt_expect_confirm}
            if found:
                best = found[0]
                rec["range_rel_err_pct"] = 100.0 * abs(best["approx_range_m"] - gt["range_m"]) / gt["range_m"]
                rec["width_rel_err_pct"] = 100.0 * abs(best["approx_width_m"] - gt["width_m"]) / max(gt["width_m"], 1e-6)
            records.append(rec)

    neg_false, neg_total = 0, 0
    for gr, gc, glabel in [(3, 3, "3x3"), (3, 6, "3x6")]:
        pl = _pipeline(gr, gc)
        for seed in range(1, 21):
            fxn = make_decorrelated_fixture(seed)
            resultn = pl.process(fxn.left, fxn.right)
            op = _build_openings_v2(resultn.geometry_frame.boundary_evidence, resultn.depth_map, "x", gr, gc,
                                     30, 1.5, FX, merge_tolerance_m=merge_tolerance_m, split_at_none=split_at_none)
            neg_total += 1
            if op:
                neg_false += 1

    tp = fp = fn = tn = 0
    width_errs, range_errs = [], []
    fn_causes = {}
    for r in records:
        gt_c = r["gt_expect_confirm"]
        found = r["n_found"] > 0
        if gt_c and found:
            tp += 1
            if "width_rel_err_pct" in r:
                width_errs.append(r["width_rel_err_pct"]); range_errs.append(r["range_rel_err_pct"])
        elif gt_c and not found:
            fn += 1
            fn_causes[r["scenario"]] = fn_causes.get(r["scenario"], 0) + 1
        elif not gt_c and found:
            fp += 1
        else:
            tn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"\n=== {label} ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  precision={precision:.4f} recall={recall:.4f}")
    if width_errs:
        print(f"width_err median={np.median(width_errs):.3f} max={np.max(width_errs):.3f}  "
              f"range_err median={np.median(range_errs):.3f} max={np.max(range_errs):.3f}")
    print(f"negative false-opening: {neg_false}/{neg_total}")
    print("FN causes:", fn_causes)
    return dict(label=label, tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                neg_false=neg_false, neg_total=neg_total, fn_causes=fn_causes)


if __name__ == "__main__":
    out = {}
    out["merge_only_0.05"] = run_candidate("merge(0.05) only, no dead-zone split", merge_tolerance_m=0.05, split_at_none=False)
    out["split_only"] = run_candidate("dead-zone split only, no merge", merge_tolerance_m=None, split_at_none=True)
    out["merge_0.05_and_split"] = run_candidate("merge(0.05) + dead-zone split", merge_tolerance_m=0.05, split_at_none=True)
    out["merge_0.01_and_split"] = run_candidate("merge(0.01) + dead-zone split", merge_tolerance_m=0.01, split_at_none=True)
    out["merge_0.10_and_split"] = run_candidate("merge(0.10) + dead-zone split", merge_tolerance_m=0.10, split_at_none=True)

    with open("/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i5_surface_opening_clearance/opening_rootcause/results/candidates_v2.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
