import numpy as np

MIN_VALID = 5


def candidate_nearest_cluster(col, min_support_count, gap_tolerance_m):
    valid = col[(col > 0) & np.isfinite(col)]
    if valid.size < MIN_VALID:
        return 0.0
    s = np.sort(valid)
    gaps = np.diff(s)
    split_idx = np.where(gaps > gap_tolerance_m)[0] + 1
    clusters = np.split(s, split_idx)
    for c in clusters:
        if c.size >= min_support_count:
            return float(np.median(c))
    return 0.0


def make_column(near_frac, near_depth, far_depth, n=240):
    n_near = int(round(near_frac * n))
    n_far = n - n_near
    return np.concatenate([np.full(n_near, near_depth), np.full(n_far, far_depth)]).astype(np.float32)


for gap_tol in [0.5, 0.10, 0.05, 0.02]:
    for min_supp in [5, 8, 12, 20]:
        # two-cluster sweep: does it correctly recover near=2.0 down to what near_frac?
        min_working_frac = None
        for near_frac in [0.005,0.01,0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30]:
            col = make_column(near_frac, 2.0, 5.0)
            d = candidate_nearest_cluster(col, min_supp, gap_tol)
            if abs(d - 2.0) < 0.01:
                min_working_frac = near_frac
                break
        # also check pathological 0.25 still works and nothing regresses at high frac
        col25 = make_column(0.25, 2.0, 5.0)
        d25 = candidate_nearest_cluster(col25, min_supp, gap_tol)

        # noise stress: realistic depth-noise simulation. Real per-pixel
        # stereo depth noise for a GENUINE surface has small std (~cm-level
        # at close range per prior IA0 measurements); i.i.d. DECORRELATED
        # noise (no real surface) scatters across the whole plausible range.
        rng = np.random.default_rng(123)
        trials = 3000
        false_hits = 0
        for t in range(trials):
            n_valid = rng.integers(20, 150)
            depths = rng.uniform(0.3, 8.0, size=n_valid).astype(np.float32)
            col = np.zeros(240, dtype=np.float32)
            col[:n_valid] = depths
            d = candidate_nearest_cluster(col, min_supp, gap_tol)
            if d > 0:
                false_hits += 1
        print(f"gap_tol={gap_tol:.2f} min_supp={min_supp:2d} | min_working_near_frac={min_working_frac} "
              f"| d25={d25:.3f} (want 2.0) | noise_false_support={false_hits}/{trials} ({100*false_hits/trials:.2f}%)")
