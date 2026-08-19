"""
DPE v1.1.1 release charts — generated from real benchmark data only.

Reads exclusively from `benchmarks.reporting.release_metrics_manifest`,
itself a thin, cited pass-through over committed benchmark scripts'
outputs (see that module's own docstring for why a manifest exists rather
than each chart re-parsing heterogeneous per-phase JSON directly). No value
in any chart below is invented, estimated, or manually typed outside that
manifest.

Deliberately several small, single-unit charts rather than one chart
mixing percentages/milliseconds/counts/FPS on one axis (README's own
documentation-pass task explicitly warned against that).

Run:
    python -m benchmarks.reporting.generate_release_charts

Output: docs/assets/metrics/*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.reporting import release_metrics_manifest as m

_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "docs", "assets", "metrics")

_BEFORE_COLOR = "#a83232"
_AFTER_COLOR = "#2b5c8a"


def _save(fig, name):
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def chart_depth_accuracy_before_after():
    data = m.depth_accuracy_by_distance()
    distances = [1.0, 2.0, 3.0, 5.0, 6.0]
    before = [data["before"][d] for d in distances]
    after = [data["after"][d] for d in distances]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(distances))
    w = 0.35
    ax.bar([i - w / 2 for i in x], before, w, label="Before I1 fix", color=_BEFORE_COLOR)
    ax.bar([i + w / 2 for i in x], after, w, label="v1.1.1 (after I1 fix)", color=_AFTER_COLOR)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - w / 2, b + 0.05, f"{b:.2f}%", ha="center", fontsize=8)
        ax.text(i + w / 2, a + 0.05, f"{a:.2f}%", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{d:.0f} m" for d in distances])
    ax.set_ylabel("Median relative depth error (%)")
    ax.set_title("Depth accuracy vs. target distance — synthetic fixtures\n(benchmarks/i1_stereo_accuracy, median rel. error, n=9-17/point)")
    ax.legend()
    _save(fig, "depth_accuracy_before_after.png")


def chart_robustness_before_after():
    dfv = m.decorrelated_false_valid_disparity()
    opening = m.opening_precision_recall_current()

    labels = ["Decorrelated\nfalse-valid\ndisparity", "Boundary\nprecision", "Boundary\nrecall",
              "Opening\nprecision", "Opening\nrecall"]
    before = [dfv["before"], m.BOUNDARY_BEFORE_PRECISION_PCT, m.BOUNDARY_BEFORE_RECALL_PCT,
              m.OPENING_BEFORE_PRECISION_PCT, m.OPENING_BEFORE_RECALL_PCT]
    tp, fp, fn, tn = m.BOUNDARY_AFTER_TP, m.BOUNDARY_AFTER_FP, m.BOUNDARY_AFTER_FN, m.BOUNDARY_AFTER_TN
    boundary_after_p = 100.0 * tp / (tp + fp) if (tp + fp) else 100.0
    boundary_after_r = 100.0 * tp / (tp + fn) if (tp + fn) else 100.0
    after = [dfv["after"], boundary_after_p, boundary_after_r,
             opening["precision_pct"], opening["recall_pct"]]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], before, w, label="Before", color=_BEFORE_COLOR)
    ax.bar([i + w / 2 for i in x], after, w, label="v1.1.1", color=_AFTER_COLOR)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - w / 2, b + 1, f"{b:.1f}%", ha="center", fontsize=8)
        ax.text(i + w / 2, a + 1, f"{a:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Robustness / false-evidence — before vs. v1.1.1\n(all metrics on a 0-100% scale; see docs/VALIDATION_MATRIX.md for counts)")
    ax.legend()
    _save(fig, "robustness_before_after.png")


def chart_clearance_false_clear_history():
    hist = m.CLEARANCE_FALSE_CLEAR_HISTORY
    stages = ["Initial\nmeasurement", "After benchmark-\nmethodology fix", "After reliability\ngating (final)"]
    pcts = [hist["initial_measurement"]["pct"], hist["after_benchmark_methodology_fix"]["pct"],
            hist["final_after_reliability_gating"]["pct"]]
    counts = [hist["initial_measurement"]["count"], hist["after_benchmark_methodology_fix"]["count"],
              hist["final_after_reliability_gating"]["count"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [_BEFORE_COLOR, "#c9962c", _AFTER_COLOR]
    bars = ax.bar(stages, pcts, color=colors)
    for b, pct, cnt in zip(bars, pcts, counts):
        ax.text(b.get_x() + b.get_width() / 2, pct + 0.3, f"{cnt}/252\n({pct}%)", ha="center", fontsize=9)
    ax.set_ylabel("False-clear ClearanceEvidence sectors (%)")
    ax.set_title("I6 clearance false-clear correction history\n(benchmarks/results/i6_final_qualification.json)")
    ax.set_ylim(0, 13)
    _save(fig, "clearance_false_clear_history.png")


def chart_latency_percentiles():
    perf = m.PERFORMANCE_V1_1_1_FRESH
    labels = ["mean", "median", "p95", "p99"]
    vals = [perf["mean_ms"], perf["median_ms"], perf["p95_ms"], perf["p99_ms"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color="#3e8e7e")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f} ms", ha="center", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(
        f"Standalone DPE latency — {perf['resolution']}, full V1 evidence config\n"
        f"FPS (mean-based): {perf['fps_mean_based']:.1f}",
        fontsize=11,
    )
    fig.text(0.5, 0.01, perf["environment"], ha="center", fontsize=7.5, style="italic")
    fig.subplots_adjust(bottom=0.16)
    _save(fig, "latency_percentiles.png")


def main():
    chart_depth_accuracy_before_after()
    chart_robustness_before_after()
    chart_clearance_false_clear_history()
    chart_latency_percentiles()


if __name__ == "__main__":
    main()
