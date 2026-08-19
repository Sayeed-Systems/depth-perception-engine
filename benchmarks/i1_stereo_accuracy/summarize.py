"""Aggregate raw per-fixture JSON rows (measure.py output) into per-scenario
per-depth summary tables, plus overall coverage/safety/performance rollups
per candidate. Pure aggregation, no re-computation of disparity/depth."""
import json
import sys
from collections import defaultdict

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def summarize(rows, label=None):
    by_key = defaultdict(list)
    for r in rows:
        if label and r.get("candidate") != label:
            continue
        key = (r["scenario"], r.get("depth_m"))
        by_key[key].append(r)

    print(f"\n{'scenario':<10}{'depth_m':>8}{'n':>4}{'disp_err_med':>14}{'depth_err_%_med':>17}"
          f"{'valid_disp':>11}{'valid_depth':>12}{'false_valid':>13}")
    for (scenario, depth), items in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        n = len(items)
        de = [it["disparity_abs_error"] for it in items if it.get("disparity_abs_error") is not None]
        drp = [it["depth_rel_error_pct"] for it in items if it.get("depth_rel_error_pct") is not None]
        vdisp = np.mean([it["valid_disparity_fraction"] for it in items])
        vdepth = np.mean([it["valid_depth_fraction"] for it in items])
        fv = np.mean([it.get("false_valid_fraction", 0.0) for it in items])
        de_s = f"{np.median(de):.3f}" if de else "n/a"
        drp_s = f"{np.median(drp):.2f}" if drp else "n/a"
        print(f"{scenario:<10}{str(depth):>8}{n:>4}{de_s:>14}{drp_s:>17}{vdisp:>11.3f}{vdepth:>12.3f}{fv:>13.4f}")

    # overall rollups
    all_lat = [r["latency_ms"] for r in rows if (not label or r.get("candidate") == label)]
    print(f"\nLatency (disparity-compute only): mean={np.mean(all_lat):.3f}ms  "
          f"p95={np.percentile(all_lat,95):.3f}ms  p99={np.percentile(all_lat,99):.3f}ms  "
          f"max={np.max(all_lat):.3f}ms  n={len(all_lat)}")

    g_rows = [r for r in rows if r["scenario"] == "G" and (not label or r.get("candidate") == label)]
    if g_rows:
        fv_g = np.mean([r["false_valid_fraction"] for r in g_rows])
        print(f"Scenario G (decorrelated) mean false-valid fraction: {fv_g:.4f}")

    e_rows = [r for r in rows if r["scenario"] == "E" and (not label or r.get("candidate") == label)]
    if e_rows:
        fv_e = np.mean([r["false_valid_fraction"] for r in e_rows])
        print(f"Scenario E (occlusion strip) mean false-valid fraction: {fv_e:.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/i1_stereo_accuracy/results/baseline_current.json"
    rows = load(path)
    summarize(rows)
