"""Focused A/B: observation_id=None vs observation_id="benchmark-observation-X".

Same warmed-pipeline methodology as benchmarks/d1_execution: one constructed
pipeline instance per arm, warmup discarded, identical stereo input and
identical timestamps/motion in both arms. Arms are INTERLEAVED across trials
so drift in machine load cannot masquerade as a per-arm difference.
"""
import gc
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from depth_perception_engine.core import DepthPerceptionPipeline, StereoObservation  # noqa: E402

from benchmarks.d1_execution import fixtures as F  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def stats(samples):
    a = np.asarray(samples, float)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "stddev_ms": float(a.std(ddof=1)),
    }


def run_arm(with_identity, frames, cal, cfg, tf, n_warmup, n_iters):
    p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
    samples, last = [], None
    for i in range(n_warmup + n_iters):
        left, right = frames[i % len(frames)]
        ts = i * 0.1
        hints = F.motion_hint_window(ts - 0.1, ts, wz=0.05) if i else None
        obs = StereoObservation(
            left_image=left, right_image=right, left_timestamp=ts, motion_hints=hints,
            observation_id=(f"benchmark-observation-{i:06d}" if with_identity else None),
        )
        t0 = time.perf_counter()
        last = p.process_geometry_frame(obs)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= n_warmup:
            samples.append(dt)
    return samples, last


def main(n_warmup=25, n_iters=250, n_trials=4):
    cal, cfg, tf = F.calibration(), F.qualified_config(), F.body_transform()
    frames = F.scene_sequence(20)

    load_before = os.getloadavg()
    without, with_id = [], []
    trial_medians = {"without_identity": [], "with_identity": []}
    for trial in range(n_trials):
        # ABBA: alternate which arm runs first. Running A always-first would
        # let monotonic machine-load drift accumulate into arm B and show up
        # as a fake per-arm cost.
        order = [False, True] if trial % 2 == 0 else [True, False]
        collected = {}
        for with_identity in order:
            gc.collect()
            collected[with_identity] = run_arm(
                with_identity, frames, cal, cfg, tf, n_warmup, n_iters,
            )
        a, gf_a = collected[False]
        b, gf_b = collected[True]
        without += a
        with_id += b
        trial_medians["without_identity"].append(float(np.median(a)))
        trial_medians["with_identity"].append(float(np.median(b)))
    load_after = os.getloadavg()

    sa, sb = stats(without), stats(with_id)
    out = {
        "method": (
            "process_geometry_frame(), one warmed pipeline instance per arm, arms "
            "interleaved in ABBA order across %d trials so load drift cannot bias one "
            "arm; identical images/timestamps/motion in both arms." % n_trials
        ),
        "n_warmup_per_arm_per_trial": n_warmup,
        "n_iters_per_arm_per_trial": n_iters,
        "n_trials": n_trials,
        "load_average_before": list(load_before),
        "load_average_after": list(load_after),
        "A_without_identity": sa,
        "B_with_identity": sb,
        "delta_median_ms": sb["median_ms"] - sa["median_ms"],
        "delta_p95_ms": sb["p95_ms"] - sa["p95_ms"],
        "delta_median_pct": 100.0 * (sb["median_ms"] - sa["median_ms"]) / sa["median_ms"],
        "per_trial_medians_ms": trial_medians,
        "within_arm_trial_spread_ms": {
            k: float(max(v) - min(v)) for k, v in trial_medians.items()
        },
        "identity_propagated": {
            "without": gf_a.observation_id,
            "with": gf_b.observation_id,
        },
    }
    # Direct micro-measurement of the ONLY work D2 actually added per frame,
    # so the answer does not depend on resolving a ~1 ms full-pipeline noise
    # floor: one StereoObservation.__post_init__ (new in D2) plus one
    # resolved_observation_id property read.
    import timeit

    left, right = frames[0]
    micro_setup = {
        "StereoObservation": StereoObservation,
        "left": left, "right": right,
    }
    n_micro = 200000
    build_without = timeit.timeit(
        "StereoObservation(left_image=left, right_image=right, left_timestamp=1.0)",
        globals=micro_setup, number=n_micro) / n_micro * 1e6
    build_with = timeit.timeit(
        "StereoObservation(left_image=left, right_image=right, left_timestamp=1.0,"
        " observation_id='benchmark-observation-000001')",
        globals=micro_setup, number=n_micro) / n_micro * 1e6
    obs_with = StereoObservation(
        left_image=left, right_image=right, left_timestamp=1.0,
        observation_id="benchmark-observation-000001",
    )
    resolve = timeit.timeit(
        "obs.resolved_observation_id",
        globals={"obs": obs_with}, number=n_micro) / n_micro * 1e6

    out["micro_us_per_call"] = {
        "n": n_micro,
        "StereoObservation_construct_without_identity_us": build_without,
        "StereoObservation_construct_with_identity_us": build_with,
        "construct_delta_us": build_with - build_without,
        "resolved_observation_id_property_us": resolve,
        "total_added_per_frame_us": (build_with - build_without) + resolve,
        "as_fraction_of_frame_pct": 100.0
        * ((build_with - build_without) + resolve) / 1000.0 / sa["median_ms"],
    }

    path = os.path.join(RESULTS, "d2_identity_ab.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
