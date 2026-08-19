"""Phase I1 Step 4 — one-at-a-time parameter sensitivity sweep (not a full
grid search, per the task's own "do not explode the search space"
instruction). Starts from the current production defaults, applies the
near-certain P1/P2 channel-multiplier fix first, then varies one parameter
at a time on top of that fixed baseline."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

from benchmarks.i1_stereo_accuracy.fixtures import build_all_fixtures
from benchmarks.i1_stereo_accuracy.measure import current_sgbm_params, run_candidate


def base_params(block_size=13, **overrides):
    p = current_sgbm_params(block_size=block_size)
    p.update(overrides)
    return p


def p1p2_fixed(block_size=13):
    # channel-correct P1/P2 for single-channel (grayscale) input: 8*1*bs^2 / 32*1*bs^2
    return dict(P1=8 * 1 * block_size ** 2, P2=32 * 1 * block_size ** 2)


CANDIDATES = {}

# 0: current production defaults (already measured in baseline_current.json,
#    re-included here so every candidate is compared under IDENTICAL fixture
#    generation/measurement code in one pass).
CANDIDATES["0_CURRENT"] = base_params()

# 1: P1/P2 channel-multiplier fix alone.
CANDIDATES["1_P1P2_FIXED"] = base_params(**p1p2_fixed())

# From here on, every candidate is P1/P2-fixed + one changed parameter.
_FIXED = p1p2_fixed()

# 2: mode sweep
CANDIDATES["2a_mode_full_SGBM"] = base_params(**_FIXED, mode=cv2.STEREO_SGBM_MODE_SGBM)
CANDIDATES["2b_mode_HH"] = base_params(**_FIXED, mode=cv2.STEREO_SGBM_MODE_HH)

# 3: uniquenessRatio sweep
for ur in (5, 15, 20):
    CANDIDATES[f"3_uniquenessRatio_{ur}"] = base_params(**_FIXED, uniquenessRatio=ur)

# 4: disp12MaxDiff / LR-consistency strength sweep
for d12 in (-1, 0, 2, 5):
    CANDIDATES[f"4_disp12MaxDiff_{d12}"] = base_params(**_FIXED, disp12MaxDiff=d12)

# 5: preFilterCap sweep
for pfc in (31, 90):
    CANDIDATES[f"5_preFilterCap_{pfc}"] = base_params(**_FIXED, preFilterCap=pfc)

# 6: speckle sweep (small)
for sw, sr in ((50, 16), (150, 48)):
    CANDIDATES[f"6_speckle_{sw}_{sr}"] = base_params(**_FIXED, speckleWindowSize=sw, speckleRange=sr)

# 7: blockSize sweep (recompute P1/P2 fixed-form at each block size)
for bs in (5, 9, 17, 21):
    CANDIDATES[f"7_blockSize_{bs}"] = base_params(block_size=bs, **p1p2_fixed(bs))

# Best-guess combined candidate (informed by expected sweep outcome — P1/P2
# fixed + full SGBM mode, since 3WAY is a speed/accuracy tradeoff and this
# repo already has latency headroom to spend).
CANDIDATES["8_P1P2fixed_modeFULL"] = base_params(**_FIXED, mode=cv2.STEREO_SGBM_MODE_SGBM)


if __name__ == "__main__":
    fixtures = build_all_fixtures()
    all_rows = []
    for label, params in CANDIDATES.items():
        rows = run_candidate(label, params, fixtures)
        all_rows.extend(rows)
        print(f"done: {label} ({len(rows)} rows)")
    with open("benchmarks/i1_stereo_accuracy/results/sweep.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Total rows: {len(all_rows)}")
