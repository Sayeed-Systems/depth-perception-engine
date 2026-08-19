"""
Phase I0 — record the frozen V1.0.1 benchmark baseline.

Runs measure.collect_all() (read-only; touches no DPE algorithm code)
and writes the result, plus provenance metadata (DPE version, git
commit, environment, timestamp), to a baseline JSON file.

This is meant to be run ONCE to freeze v1.0.1 and be committed to the
repo. By default it refuses to overwrite an existing baseline file —
the whole point of a freeze is that it does not silently move — pass
--force if a re-freeze is genuinely intended.

Run:
    python -m benchmarks.i0_baseline.record_baseline
    python -m benchmarks.i0_baseline.record_baseline --force
    python -m benchmarks.i0_baseline.record_baseline --out benchmarks/i0_baseline/baseline_v1.0.1.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

import depth_perception_engine
from benchmarks.i0_baseline import measure

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_OUT = os.path.join(_REPO_ROOT, "benchmarks", "i0_baseline", "baseline_v1.0.1.json")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=_DEFAULT_OUT, help="Output path for the frozen baseline JSON.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing baseline file.")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"Refusing to overwrite existing frozen baseline at {args.out} (pass --force to re-freeze).",
              file=sys.stderr)
        sys.exit(1)

    print("Collecting I0 baseline measurements (read-only, no DPE code modified)...")
    metrics = measure.collect_all()

    summary = {
        "phase": "I0_BENCHMARK_BASELINE_FREEZE",
        "dpe_version": depth_perception_engine.__version__,
        "git_commit": _git_commit(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "platform": sys.platform,
        },
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")

    print(f"Frozen baseline written to: {args.out}")
    print(f"DPE version: {summary['dpe_version']}  git commit: {summary['git_commit']}")


if __name__ == "__main__":
    main()
