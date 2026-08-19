"""
Synthetic sweep: exactly reproduce ThreatAssessor.assess()'s per-beam
percentile+IQR arithmetic on a two-value (near_depth, far_depth) column
mixture, varying near_frac, to precisely locate where/why it fails, and
test candidate replacements — all without re-running the full stereo
pipeline (validated against real pipeline runs separately).
"""
import numpy as np

PERCENTILE = 15
MIN_VALID = 5


def current_algorithm(col):
    valid = col[(col > 0) & np.isfinite(col)]
    if valid.size < MIN_VALID:
        return 0.0, valid, valid
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    pre_iqr = valid.copy()
    if iqr > 0:
        valid = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    d_m = float(np.percentile(valid, PERCENTILE)) if valid.size >= MIN_VALID else 0.0
    return d_m, pre_iqr, valid


def make_column(near_frac, near_depth, far_depth, n=240):
    n_near = int(round(near_frac * n))
    n_far = n - n_near
    return np.concatenate([np.full(n_near, near_depth), np.full(n_far, far_depth)]).astype(np.float32)


print("near_frac | near_survives_IQR | d_m (current) | true_near | error_pct | direction")
for near_frac in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]:
    col = make_column(near_frac, near_depth=2.0, far_depth=5.0)
    d_m, pre_iqr, post_iqr = current_algorithm(col)
    near_survives = int((post_iqr < 3.0).sum())
    near_total = int((pre_iqr < 3.0).sum())
    err_pct = 100.0 * (d_m - 2.0) / 2.0 if d_m > 0 else float('nan')
    direction = "OVER(false-clear)" if d_m > 2.0 * 1.05 else ("under" if d_m < 2.0*0.95 else "OK")
    print(f"{near_frac:.2f}      | {near_survives}/{near_total}            | {d_m:.4f}       | 2.0000    | {err_pct:7.2f}   | {direction}")
