import json
from collections import defaultdict

import numpy as np

with open("benchmarks/i1_stereo_accuracy/results/sweep.json") as f:
    rows = json.load(f)

by_cand = defaultdict(list)
for r in rows:
    by_cand[r["candidate"]].append(r)

print(f"{'candidate':<26}{'ABC_disp_err_med':>17}{'ABC_depth_err%_med':>19}{'ABC_valid_frac':>15}"
      f"{'G_false_valid':>14}{'E_false_valid':>14}{'F_valid_frac':>13}{'lat_mean_ms':>12}{'lat_p95_ms':>11}")

for cand, items in by_cand.items():
    abc = [r for r in items if r["scenario"] in ("A", "B", "C")]
    g = [r for r in items if r["scenario"] == "G"]
    e = [r for r in items if r["scenario"] == "E"]
    f_ = [r for r in items if r["scenario"] == "F"]
    de = [r["disparity_abs_error"] for r in abc if r.get("disparity_abs_error") is not None]
    drp = [r["depth_rel_error_pct"] for r in abc if r.get("depth_rel_error_pct") is not None]
    vf = np.mean([r["valid_disparity_fraction"] for r in abc])
    gfv = np.mean([r["false_valid_fraction"] for r in g]) if g else float("nan")
    efv = np.mean([r["false_valid_fraction"] for r in e]) if e else float("nan")
    ffv = np.mean([r["valid_disparity_fraction"] for r in f_]) if f_ else float("nan")
    lat = [r["latency_ms"] for r in items]
    print(f"{cand:<26}{np.median(de) if de else float('nan'):>17.4f}"
          f"{np.median(drp) if drp else float('nan'):>19.3f}{vf:>15.4f}"
          f"{gfv:>14.4f}{efv:>14.4f}{ffv:>13.4f}{np.mean(lat):>12.4f}{np.percentile(lat,95):>11.4f}")

# per-depth breakdown for the two leading candidates
for cand in ("0_CURRENT", "1_P1P2_FIXED", "8_P1P2fixed_modeFULL"):
    print(f"\n--- {cand}: A/B/C depth error % by depth ---")
    items = [r for r in by_cand[cand] if r["scenario"] in ("A", "B", "C") and r.get("depth_rel_error_pct") is not None]
    by_depth = defaultdict(list)
    for r in items:
        by_depth[r["depth_m"]].append(r["depth_rel_error_pct"])
    for d in sorted(by_depth):
        print(f"  {d:>4.1f}m: median={np.median(by_depth[d]):.3f}%  n={len(by_depth[d])}")
