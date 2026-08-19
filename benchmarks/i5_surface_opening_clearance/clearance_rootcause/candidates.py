"""Candidate aggregation algorithms tested against the same synthetic
two-cluster sweep, plus a noise-robustness stress test."""
import numpy as np

PERCENTILE = 15
MIN_VALID = 5


def current_algorithm(col):
    valid = col[(col > 0) & np.isfinite(col)]
    if valid.size < MIN_VALID:
        return 0.0
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        valid = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    return float(np.percentile(valid, PERCENTILE)) if valid.size >= MIN_VALID else 0.0


def candidate_low_percentile(col, pct):
    valid = col[(col > 0) & np.isfinite(col)]
    if valid.size < MIN_VALID:
        return 0.0
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        valid = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    return float(np.percentile(valid, pct)) if valid.size >= MIN_VALID else 0.0


def candidate_nearest_cluster(col, min_support_count=5, gap_tolerance_m=0.5):
    """No IQR step at all (that's precisely what discards the minority
    near cluster). Sort valid depths ascending, split into clusters at
    any gap > gap_tolerance_m, report the NEAREST (lowest-depth) cluster
    whose own pixel count clears min_support_count. Falls back to 0.0
    (NO_DATA) if no cluster clears the floor."""
    valid = col[(col > 0) & np.isfinite(col)]
    if valid.size < MIN_VALID:
        return 0.0
    s = np.sort(valid)
    gaps = np.diff(s)
    split_idx = np.where(gaps > gap_tolerance_m)[0] + 1
    clusters = np.split(s, split_idx)
    for c in clusters:  # already ascending by construction (sorted input)
        if c.size >= min_support_count:
            return float(np.median(c))
    return 0.0  # no cluster meets the support floor -> honest NO_DATA, not a fabricated distance


def make_column(near_frac, near_depth, far_depth, n=240):
    n_near = int(round(near_frac * n))
    n_far = n - n_near
    return np.concatenate([np.full(n_near, near_depth), np.full(n_far, far_depth)]).astype(np.float32)


print("=== Two-cluster sweep: current vs candidates ===")
print(f"{'near_frac':>9} | {'current':>8} | {'pct5':>8} | {'pct10':>8} | {'cluster(n=5)':>13}")
for near_frac in [0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.50]:
    col = make_column(near_frac, 2.0, 5.0)
    cur = current_algorithm(col)
    p5 = candidate_low_percentile(col, 5)
    p10 = candidate_low_percentile(col, 10)
    clus = candidate_nearest_cluster(col)
    print(f"{near_frac:9.3f} | {cur:8.3f} | {p5:8.3f} | {p10:8.3f} | {clus:13.3f}")

print()
print("=== Noise robustness: i.i.d.-like scattered depths, various valid counts, seeded ===")
print(f"{'seed':>4} | {'n_valid':>7} | {'current':>8} | {'pct5':>8} | {'pct10':>8} | {'cluster(n=5)':>13}")
for seed in range(1, 11):
    rng = np.random.default_rng(seed)
    n_valid = rng.integers(5, 60)
    # simulate decorrelated-noise-driven "valid" depths: scattered across
    # the whole plausible depth range 0.3-8.0m, uniform random (worst case
    # for spurious clustering -- real noise from I1's own measurements is
    # sparser/more scattered than this synthetic upper-bound stress test)
    depths = rng.uniform(0.3, 8.0, size=n_valid).astype(np.float32)
    col = np.zeros(240, dtype=np.float32)
    col[:n_valid] = depths
    cur = current_algorithm(col)
    p5 = candidate_low_percentile(col, 5)
    p10 = candidate_low_percentile(col, 10)
    clus = candidate_nearest_cluster(col)
    print(f"{seed:4d} | {n_valid:7d} | {cur:8.3f} | {p5:8.3f} | {p10:8.3f} | {clus:13.3f}")

print()
print("=== Noise robustness, WORST-CASE clustered-by-chance stress: force a tight cluster in noise ===")
# Deliberately adversarial: is it possible for i.i.d. noise to form a
# tight false cluster with n>=5 by chance? Test with denser valid counts.
false_cluster_hits = 0
trials = 2000
rng = np.random.default_rng(42)
for t in range(trials):
    n_valid = rng.integers(20, 120)
    depths = rng.uniform(0.3, 8.0, size=n_valid).astype(np.float32)
    col = np.zeros(240, dtype=np.float32)
    col[:n_valid] = depths
    clus = candidate_nearest_cluster(col)
    cur = current_algorithm(col)
    # "false-clear-relevant" concern here is really: does the cluster
    # method report SOME confident near value that current would not?
    if clus > 0 and abs(clus - cur) > 0.05:
        false_cluster_hits += 1
print(f"trials={trials}, cluster-vs-current disagreement (>5cm) count={false_cluster_hits} "
      f"({100.0*false_cluster_hits/trials:.1f}%)")
