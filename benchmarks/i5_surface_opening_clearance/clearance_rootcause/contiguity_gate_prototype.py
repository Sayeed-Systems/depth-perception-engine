"""
I6.1A — CLEARANCE CONTIGUITY-GATE PROTOTYPE ONLY. No production-code
changes. Read-only diagnostic reusing the exact fixtures already used by
benchmarks/i5_surface_opening_clearance/clearance/measure.py (the frozen
I5 clearance dataset) and clearance_rootcause/real_validate.py (the two
specific known cases: the genuine false-clear recoveries and the 0.430m
narrow_obstacle seed=1 beam9 quantization artifact).

For every one of ThreatAssessor's 20 beam columns across the dataset, this
recomputes the SAME IQR fencing ThreatAssessor.assess() already does
(q1-1.5*IQR / q3+1.5*IQR on the raw depth column), but instead of
discarding the low-side-rejected population, captures it with row
position preserved (depth_map[:, x1:x2] is never flattened before the
row/col split) and characterizes it: support count, median depth, depth
spread, row indices, longest contiguous vertical run, contiguous-run
fraction, number of separate runs, and a ground-truth label derived from
each fixture's own known scene geometry (genuine near obstacle vs no real
surface at that depth -> false quantized cluster).

Only the spatial-contiguity gate is swept. support_count>=12 and
spread<=0.05m are held fixed at the values already validated (0% false
hits on synthetic i.i.d. noise, candidates2.py) — not retuned here.
"""
import sys
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import json
import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

from benchmarks.i1_stereo_accuracy.fixtures import make_decorrelated_fixture, W as _W, H as _H
from benchmarks.i4_boundary_precision.fixtures import make_narrow_obstacle_fixture
from benchmarks.i5_surface_opening_clearance.clearance.fixtures import (
    make_multi_zone_fixture, make_dis_occlusion_fixture,
)
from benchmarks.i1_stereo_accuracy.fixtures import make_discontinuity_fixture

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")

N_BEAMS = 20
MIN_VALID = 5
PERCENTILE = 15
DEAD_ZONE_PX = 128
CLEAR_M = 2.0
CAUTION_M = 1.0

# Fixed, already-validated (candidates2.py) gates -- NOT swept here.
MIN_SUPPORT_COUNT = 12
MAX_SPREAD_M = 0.05
GAP_TOLERANCE_M = 0.02

# Depth-match tolerance for labeling a near_rejected cluster's median
# against the fixture's own known true zone depths.
GT_MATCH_TOL_M = 0.15


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


def _any_true_depth_at_column(x1, x2, zones, depth_val, tol=GT_MATCH_TOL_M):
    """Does ANY zone overlapping [x1,x2) (ignoring the dead-zone
    restriction -- we want the raw scene geometry here, not the
    observable-only nearest) have a true depth within tol of depth_val?
    Used to label a near_rejected cluster as corresponding to a real
    surface in the fixture vs. having no real counterpart at all."""
    for c0f, c1f, true_depth in zones:
        c0, c1 = int(c0f * _W), int(c1f * _W)
        if min(c1, x2) > max(c0, x1) and abs(true_depth - depth_val) <= tol:
            return True
    return False


def analyze_beam(depth_map, x1, x2):
    """Recompute ThreatAssessor's own IQR fencing on this column, with row
    position preserved. The low-side-rejected population is then split
    into gap-based sub-clusters (gap_tolerance=0.02m, the already-validated
    candidates2.py value -- held fixed, not swept here) because the raw
    IQR-rejected mass is frequently a broad blob spanning several distinct
    real depths, not one coherent surface (confirmed empirically: without
    this step the whole-blob median for known genuine cases lands nowhere
    near the true obstacle depth). Returns a list of per-sub-cluster
    records (possibly empty)."""
    col2d = depth_map[:, x1:x2]
    valid_mask = (col2d > 0) & np.isfinite(col2d)
    if valid_mask.sum() < MIN_VALID:
        return []
    rows, cols = np.where(valid_mask)
    vals = col2d[valid_mask].astype(np.float64)

    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return []
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    kept_mask = (vals >= low_fence) & (vals <= high_fence)
    current_d_m = float(np.percentile(vals[kept_mask], PERCENTILE)) if kept_mask.sum() >= MIN_VALID else 0.0

    rej_mask = vals < low_fence
    if not np.any(rej_mask):
        return []

    rej_vals = vals[rej_mask]
    rej_rows = rows[rej_mask]
    order = np.argsort(rej_vals)
    s_vals = rej_vals[order]
    s_rows = rej_rows[order]

    gaps = np.diff(s_vals)
    split_idx = np.where(gaps > GAP_TOLERANCE_M)[0] + 1
    val_clusters = np.split(s_vals, split_idx)
    row_clusters = np.split(s_rows, split_idx)

    out = []
    for c_vals, c_rows in zip(val_clusters, row_clusters):
        support = int(c_vals.size)
        median_depth = float(np.median(c_vals))
        spread = float(c_vals.max() - c_vals.min())

        unique_rows = np.unique(c_rows)
        diffs = np.diff(unique_rows)
        breaks = np.where(diffs > 1)[0]
        s_idx = breaks + 1
        runs = np.split(unique_rows, s_idx)
        run_lengths = [len(r) for r in runs]
        longest_run = int(max(run_lengths)) if run_lengths else 0
        n_runs = int(len(runs))
        contig_fraction = longest_run / unique_rows.size if unique_rows.size else 0.0

        out.append(dict(
            support=support, median_depth=median_depth, spread=spread,
            n_distinct_rows=int(unique_rows.size), longest_run=longest_run,
            n_runs=n_runs, contig_fraction=contig_fraction,
            current_d_m=current_d_m,
        ))
    return out


def _status_for(d_m):
    if d_m <= 0.0:
        return "NO_DATA"
    if d_m < CAUTION_M:
        return "BLOCKED"
    if d_m < CLEAR_M:
        return "CAUTION"
    return "CLEAR"


def collect_records():
    """Scan every beam of every fixture in the frozen I5 clearance dataset
    plus the two known real_validate.py reference cases. Returns a list of
    per-beam dicts with the near_rejected analysis, ground truth label,
    and bucket (A/B/C/D)."""
    pipeline = _pipeline()
    records = []

    def scan(depth_map, zones, scenario, seed, known_false_clear_beams=(), known_artifact_beams=()):
        h, w = depth_map.shape[:2]
        beam_w = w / N_BEAMS
        for i in range(N_BEAMS):
            x1, x2 = int(i * beam_w), int((i + 1) * beam_w)
            clusters = analyze_beam(depth_map, x1, x2)
            true_range = _sector_true_range(x1, x2, zones) if zones else None
            for cluster_idx, a in enumerate(clusters):
                matches_real_surface = (
                    _any_true_depth_at_column(x1, x2, zones, a["median_depth"]) if zones else False
                )

                if i in known_artifact_beams:
                    bucket = "B_artifact"
                    gt_label = "false quantized cluster"
                elif i in known_false_clear_beams:
                    bucket = "A_genuine_recovery"
                    gt_label = "genuine near obstacle" if matches_real_surface else "false quantized cluster"
                elif scenario in ("decorrelated_invalid", "pure_noise_30seeds"):
                    bucket = "C_noise"
                    gt_label = "false quantized cluster"
                else:
                    bucket = "D_ordinary"
                    gt_label = "genuine near obstacle" if matches_real_surface else "false quantized cluster"

                records.append(dict(
                    scenario=scenario, seed=seed, beam=i, x1=x1, x2=x2, cluster_idx=cluster_idx,
                    true_range_m=true_range, bucket=bucket, gt_label=gt_label,
                    matches_real_surface=matches_real_surface,
                    **a,
                ))

    # --- A / known false-clear recoveries + the real_validate.py D_ordinary sweep ---
    zones_mz = [(0.0, 0.33, 1.5), (0.33, 0.66, 4.0), (0.66, 1.0, 2.5)]
    for seed in range(1, 4):
        fx = make_multi_zone_fixture(zones_mz, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result.depth_map, zones_mz, "multi_zone", seed, known_false_clear_beams={13})

    zones_no = [(0.0, 0.4, 5.0), (0.4, 0.6, 2.0), (0.6, 1.0, 5.0)]
    for seed in range(1, 4):
        fx = make_narrow_obstacle_fixture(near_m=2.0, far_m=5.0, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        # beam 8 = known false-clear (x=[128,144], dead-zone-adjacent);
        # beam 9 (seed=1 specifically) = the known 0.430m artifact case.
        artifact_beams = {9} if seed == 1 else set()
        scan(result.depth_map, zones_no, "narrow_obstacle", seed,
             known_false_clear_beams={8}, known_artifact_beams=artifact_beams)

    # --- D: occlusion / dis-occlusion (already-qualified, should stay clean) ---
    zones_occ = [(0.0, 0.5, 1.5), (0.5, 1.0, 5.0)]
    for seed in range(1, 6):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=True)
        result = pipeline.process(fx.left, fx.right)
        scan(result.depth_map, zones_occ, "occlusion_near_side", seed)

    for seed in range(1, 6):
        fx = make_dis_occlusion_fixture(near_m=1.5, far_m=5.0, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result.depth_map, zones_occ, "dis_occlusion_far_side", seed)

    # --- D: single near / single far / open far uniform ---
    fx = make_multi_zone_fixture([(0.0, 1.0, 1.5)], seed=1)
    result = pipeline.process(fx.left, fx.right)
    scan(result.depth_map, [(0.0, 1.0, 1.5)], "single_near_1.5m", 1)

    fx = make_multi_zone_fixture([(0.0, 1.0, 5.0)], seed=2)
    result = pipeline.process(fx.left, fx.right)
    scan(result.depth_map, [(0.0, 1.0, 5.0)], "single_far_5.0m", 2)

    zones_open = [(0.0, 0.4, 6.0), (0.4, 0.6, 6.0), (0.6, 1.0, 6.0)]
    for seed in range(1, 4):
        fx = make_multi_zone_fixture(zones_open, seed=seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result.depth_map, zones_open, "open_far_uniform", seed)

    # --- C: pure/decorrelated noise (30 seeds, matches clearance/measure.py) ---
    for seed in range(1, 31):
        fx = make_decorrelated_fixture(seed)
        result = pipeline.process(fx.left, fx.right)
        scan(result.depth_map, [], "pure_noise_30seeds", seed)

    return records


def sweep(records):
    """Sweep ONLY the spatial-contiguity gate (min longest contiguous
    vertical run, in rows). support_count>=12 and spread<=0.05m are held
    fixed throughout, matching the already-validated candidates2.py
    values -- not retuned here."""
    # Candidates that already clear the fixed, non-contiguity gates:
    base_eligible = [r for r in records if r["support"] >= MIN_SUPPORT_COUNT and r["spread"] <= MAX_SPREAD_M]

    print(f"Total near_rejected populations found: {len(records)}")
    print(f"Populations clearing fixed gates (support>=12, spread<=0.05m): {len(base_eligible)}")
    print()
    by_bucket = {}
    for r in base_eligible:
        by_bucket.setdefault(r["bucket"], []).append(r)
    for bucket, rows in sorted(by_bucket.items()):
        print(f"--- bucket {bucket}: n={len(rows)} ---")
        for r in rows:
            print(f"  {r['scenario']} seed={r['seed']} beam={r['beam']} x=[{r['x1']},{r['x2']}] "
                  f"true={r['true_range_m']} median={r['median_depth']:.3f} support={r['support']} "
                  f"spread={r['spread']:.4f} rows={r['n_distinct_rows']} longest_run={r['longest_run']} "
                  f"n_runs={r['n_runs']} contig_frac={r['contig_fraction']:.3f} "
                  f"cur_d_m={r['current_d_m']:.3f} gt={r['gt_label']}")
    print()

    # Per-beam (not per-cluster) "known" sets so multi-cluster beams don't
    # double count -- a beam with >=1 gated cluster is one activation.
    n_known_A_beams = len({(r["scenario"], r["seed"], r["beam"]) for r in records if r["bucket"] == "A_genuine_recovery"})
    n_known_B_beams = len({(r["scenario"], r["seed"], r["beam"]) for r in records if r["bucket"] == "B_artifact"})

    print("=== Sweep: minimum longest contiguous vertical run (rows) ===")
    print(f"{'min_run':>8} | {'A_recovered_beams':>18} | {'B_activated_beams':>18} | "
          f"{'C_activated_beams':>18} | {'D_activated_beams':>18} | {'D_new_false_blocked_beams':>26}")
    for min_run in [0, 2, 4, 6, 8, 10, 12, 16, 20, 30, 40]:
        gated = [r for r in base_eligible if r["longest_run"] >= min_run]

        def beams_of(bucket):
            return {(r["scenario"], r["seed"], r["beam"]) for r in gated if r["bucket"] == bucket}

        a_beams = beams_of("A_genuine_recovery")
        b_beams = beams_of("B_artifact")
        c_beams = beams_of("C_noise")
        d_beams = beams_of("D_ordinary")

        # A beam counts as recovered if ANY of its gated clusters lands
        # within tolerance of the true range (mirrors "take nearest
        # cluster meeting gates" -- the nearest one that matches wins).
        a_recovered = 0
        for key in a_beams:
            cluster_rows = [r for r in gated if r["bucket"] == "A_genuine_recovery"
                             and (r["scenario"], r["seed"], r["beam"]) == key]
            if any(r["true_range_m"] is not None and abs(r["median_depth"] - r["true_range_m"]) <= GT_MATCH_TOL_M
                   for r in cluster_rows):
                a_recovered += 1

        d_new_false_blocked = set()
        for key in d_beams:
            cluster_rows = [r for r in gated if r["bucket"] == "D_ordinary"
                             and (r["scenario"], r["seed"], r["beam"]) == key]
            for r in cluster_rows:
                if (_status_for(r["median_depth"]) != _status_for(r["current_d_m"])
                        and _status_for(r["median_depth"]) in ("BLOCKED", "CAUTION")):
                    d_new_false_blocked.add(key)

        print(f"{min_run:8d} | {a_recovered:5d}/{n_known_A_beams:<11d} | {len(b_beams):5d}/{n_known_B_beams:<11d} | "
              f"{len(c_beams):18d} | {len(d_beams):18d} | {len(d_new_false_blocked):26d}")

    return base_eligible


def main():
    records = collect_records()
    with open(
        "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i5_surface_opening_clearance/"
        "clearance_rootcause/results/contiguity_gate_prototype.json", "w"
    ) as f:
        json.dump(records, f, indent=2, default=str)
    sweep(records)


if __name__ == "__main__":
    main()
