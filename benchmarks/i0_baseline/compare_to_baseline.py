"""
Phase I0 — compare a candidate run against the frozen V1.0.1 baseline.

Re-runs measure.collect_all() (read-only; touches no DPE algorithm code)
as the "candidate" and diffs it, leaf by leaf, against a previously
frozen baseline JSON (as produced by record_baseline.py). For every
numeric leaf this reports baseline value, candidate value, and delta.
For every non-numeric leaf it reports whether the two values are equal.

This script does NOT assert pass/fail and introduces NO acceptance
threshold — it reports deltas honestly, exactly as measured, matching
this repo's own existing benchmark precedent (examples/benchmark_d14_
provider_validation.py) of observing/reporting rather than gating.
Deciding what delta is acceptable is a judgment call for whoever reads
the report, not something this script encodes.

Run:
    python -m benchmarks.i0_baseline.compare_to_baseline
    python -m benchmarks.i0_baseline.compare_to_baseline \\
        --baseline benchmarks/i0_baseline/baseline_v1.0.1.json \\
        --out /tmp/i0_candidate_vs_baseline.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from numbers import Number

import cv2
import numpy as np

import depth_perception_engine
from benchmarks.i0_baseline import measure

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_BASELINE = os.path.join(_REPO_ROOT, "benchmarks", "i0_baseline", "baseline_v1.0.1.json")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _walk_diff(path: str, baseline_val, candidate_val, rows: list) -> None:
    if isinstance(baseline_val, dict) or isinstance(candidate_val, dict):
        # Dict keys may be int in a freshly-computed candidate dict but str
        # once round-tripped through JSON (the baseline file) — normalize
        # both sides to str keys so the same logical key always matches.
        baseline_val = {str(k): v for k, v in (baseline_val or {}).items()}
        candidate_val = {str(k): v for k, v in (candidate_val or {}).items()}
        keys = sorted(set(baseline_val.keys()) | set(candidate_val.keys()))
        for k in keys:
            _walk_diff(f"{path}.{k}" if path else k, baseline_val.get(k), candidate_val.get(k), rows)
        return

    if isinstance(baseline_val, bool) or isinstance(candidate_val, bool):
        rows.append({"path": path, "baseline": baseline_val, "candidate": candidate_val,
                      "delta": None, "equal": baseline_val == candidate_val})
        return

    if isinstance(baseline_val, Number) and isinstance(candidate_val, Number):
        delta = float(candidate_val) - float(baseline_val)
        rows.append({"path": path, "baseline": baseline_val, "candidate": candidate_val,
                      "delta": delta, "equal": delta == 0.0})
        return

    if isinstance(baseline_val, list) and isinstance(candidate_val, list):
        rows.append({"path": path, "baseline": baseline_val, "candidate": candidate_val,
                      "delta": None, "equal": baseline_val == candidate_val})
        return

    rows.append({"path": path, "baseline": baseline_val, "candidate": candidate_val,
                  "delta": None, "equal": baseline_val == candidate_val})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=_DEFAULT_BASELINE, help="Path to the frozen baseline JSON.")
    parser.add_argument("--out", default=None, help="Optional path to write the full diff report as JSON.")
    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        print(f"Frozen baseline not found at {args.baseline} — run record_baseline.py first.", file=sys.stderr)
        sys.exit(1)

    with open(args.baseline, "r", encoding="utf-8") as f:
        baseline_doc = json.load(f)

    print(f"Baseline: DPE {baseline_doc.get('dpe_version')} @ {baseline_doc.get('git_commit')} "
          f"(recorded {baseline_doc.get('generated_at_utc')})")
    print("Collecting candidate measurements (read-only, no DPE code modified)...")
    candidate_metrics = measure.collect_all()

    rows = []
    _walk_diff("", baseline_doc.get("metrics", {}), candidate_metrics, rows)

    n_changed = sum(1 for r in rows if not r["equal"])
    print(f"\n{'metric':<70} {'baseline':>14} {'candidate':>14} {'delta':>12}")
    print("-" * 112)
    for r in rows:
        if r["equal"]:
            continue
        b = f"{r['baseline']:.6g}" if isinstance(r["baseline"], Number) and not isinstance(r["baseline"], bool) else str(r["baseline"])
        c = f"{r['candidate']:.6g}" if isinstance(r["candidate"], Number) and not isinstance(r["candidate"], bool) else str(r["candidate"])
        d = f"{r['delta']:+.6g}" if r["delta"] is not None else "n/a"
        print(f"{r['path']:<70} {b:>14} {c:>14} {d:>12}")

    print(f"\n{n_changed} / {len(rows)} leaf metrics differ from baseline (no pass/fail threshold applied).")

    candidate_doc = {
        "phase": "I0_CANDIDATE_VS_BASELINE_COMPARISON",
        "baseline_dpe_version": baseline_doc.get("dpe_version"),
        "baseline_git_commit": baseline_doc.get("git_commit"),
        "baseline_generated_at_utc": baseline_doc.get("generated_at_utc"),
        "candidate_dpe_version": depth_perception_engine.__version__,
        "candidate_git_commit": _git_commit(),
        "candidate_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "platform": sys.platform,
        },
        "n_leaf_metrics": len(rows),
        "n_changed": n_changed,
        "diff": rows,
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(candidate_doc, f, indent=2, default=str)
            f.write("\n")
        print(f"\nFull diff report written to: {args.out}")


if __name__ == "__main__":
    main()
