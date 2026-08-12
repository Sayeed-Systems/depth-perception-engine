"""
Performance / bounded-resource validation — Phase D14 (see
docs/DPE_V1_PROVIDER_CONTRACT.md's D14 record).

Measures the EXISTING implementation honestly. This script does not
optimize, tune, or change any algorithm — it is read-only instrumentation
around the real, public DepthPerceptionPipeline, following the exact
methodology/conventions examples/benchmark_e6_full_pipeline.py and
examples/benchmark_e6_memory_stability.py already established:
deterministic synthetic input, an explicit discarded warm-up phase before
any timed sample, mean/median/std/percentile reporting, and RSS/object-
count long-run tracking that OBSERVES and REPORTS rather than asserting
a pass/fail threshold (no documented real-time rate requirement exists
for this library — see docs/VALIDATION_REPORT.md's own "no specific
real-time rate requirement has been defined" precedent).

Four configurations (PipelineConfig progression, additive):
    A. core geometry only        — enable_geometry
    B. GeometryFrame + V1 evidence — A + obstacle/free-space/surface/
                                     boundary/opening geometry + GeometryFrame
    C. Level-4 temporal only     — A + full temporal stack (no D-phase
                                     evidence, no GeometryFrame) — isolates
                                     Level 4's own added cost
    D. full DPE V1 candidate     — B + C combined (everything on)

Two resolutions, matching this repository's own established benchmark
precedent (examples/benchmark_e6_full_pipeline.py): the real hardware
calibration (320x240, rectify=True) and a synthetic 640x480 calibration
(rectify=False) — no larger resolution is invented.

Also runs ONE sustained long-run (500 frames, matching
benchmark_e6_memory_stability.py's own precedent) at the full V1
candidate configuration, tracking RSS memory, TemporalHistory length,
TemporalPersistenceTracker's fixed-shape internal arrays, and
GeometryFrame's zero-duplication guarantee — then calls reset() and
confirms behavior returns to the documented first-frame baseline.

Finally, a lightweight cProfile pass over the full V1 candidate
configuration reports the top cumulative-time contributors — read-only
profiling, no code change.

Writes a machine-readable JSON summary to results/d14_performance_validation.json
(results/ is gitignored, matching runs/'s own existing precedent for
generated, non-committed output).

Run:
    pip install -e ..
    python examples/benchmark_d14_provider_validation.py
"""

import cProfile
import gc
import json
import os
import pstats
import sys
import time
from datetime import datetime, timezone
from io import StringIO

import cv2
import numpy as np

from depth_perception_engine import (
    DepthPerceptionPipeline,
    MotionHint,
    PipelineConfig,
    load_stereo_calibration,
)
from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.frames import FrameId, RigidTransform

N_WARMUP = 15
N_ITERS = 100
N_SUSTAINED_FRAMES = 500
SUSTAINED_CHECKPOINT_EVERY = 50

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
_RESULTS_PATH = os.path.join(_RESULTS_DIR, "d14_performance_validation.json")


# ===================================================================
# Fixtures — identical construction to examples/benchmark_e6_full_pipeline.py
# ===================================================================
def _synthetic_calibration(width: int, height: int) -> StereoCalibration:
    fx = fy = 600.0
    cx, cy = width / 2.0, height / 2.0
    baseline_m = 0.065
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((1, 5), dtype=np.float64)
    tx = 1.0 / (baseline_m * 1000.0)
    Q = np.array(
        [[1.0, 0.0, 0.0, -cx], [0.0, 1.0, 0.0, -cy], [0.0, 0.0, 0.0, fx], [0.0, 0.0, tx, 0.0]],
        dtype=np.float64,
    )
    return StereoCalibration(
        image_size=(width, height),
        camera_matrix_left=camera_matrix, dist_coeffs_left=dist_coeffs,
        camera_matrix_right=camera_matrix, dist_coeffs_right=dist_coeffs,
        R1=np.eye(3), R2=np.eye(3),
        P1=np.hstack([camera_matrix, np.zeros((3, 1))]), P2=np.hstack([camera_matrix, np.zeros((3, 1))]),
        Q=Q,
    )


def _transform() -> RigidTransform:
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _noise_stereo_pair(width: int, height: int, seed: int = 0):
    """Deterministic i.i.d. noise pair — identical methodology to
    examples/benchmark_e6_full_pipeline.py's own fixture. Purely a timing
    fixture (this script measures cost, not correctness) — D11 already
    established real StereoSGBM's smoothness prior still produces a
    dense, plausible-looking (if not metrically correct) disparity field
    from this kind of input, which is what exercises realistic per-cell
    D4-D8 evidence cost."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return left, right


def _zero_motion_hint(ts: float) -> MotionHint:
    """A real, valid, zero-angular-velocity MotionHint — exercises E5's
    actual reprojection code path (RotationCompensationStatus.APPLIED)
    deterministically, without introducing any scene inconsistency
    (identity rotation reproduces the input exactly, per
    tests/test_rotation_compensation.py::TestCompensatePriorGeometry::
    test_zero_rotation_reproduces_input) — the correct way to measure
    E5's real cost without confounding the temporal-consistency outcome."""
    return MotionHint(timestamp=ts, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY, valid=True)


# ===================================================================
# Configs A-D
# ===================================================================
def _config_a_core_geometry(**overrides):
    return PipelineConfig(enable_geometry=True, **overrides)


def _config_b_geometry_frame_v1_evidence(**overrides):
    return PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        enable_geometry_frame=True,
        **overrides,
    )


def _config_c_level4_temporal_only(**overrides):
    return PipelineConfig(
        enable_geometry=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=30,
        **overrides,
    )


def _config_d_full_v1_candidate(**overrides):
    return PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        enable_geometry_frame=True,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=30,
        **overrides,
    )


CONFIGS = {
    "A_core_geometry": _config_a_core_geometry,
    "B_geometry_frame_v1_evidence": _config_b_geometry_frame_v1_evidence,
    "C_level4_temporal_only": _config_c_level4_temporal_only,
    "D_full_v1_candidate": _config_d_full_v1_candidate,
}

# Configs that enable temporal need MotionHints to exercise E5's real
# reprojection cost, and a distinct per-frame timestamp to be admitted
# by TemporalHistory at all.
_TEMPORAL_CONFIGS = {"C_level4_temporal_only", "D_full_v1_candidate"}


# ===================================================================
# Stats helpers
# ===================================================================
def _stats(samples_ms):
    samples_ms = np.asarray(samples_ms)
    return {
        "mean_ms": float(samples_ms.mean()),
        "median_ms": float(np.median(samples_ms)),
        "std_ms": float(samples_ms.std()),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
        "max_ms": float(samples_ms.max()),
        "min_ms": float(samples_ms.min()),
        "fps_mean_based": float(1000.0 / samples_ms.mean()) if samples_ms.mean() > 0 else float("inf"),
        "n": int(samples_ms.size),
    }


def _print_stats(label, s):
    print(f"  {label:<32s} mean={s['mean_ms']:7.3f}ms  median={s['median_ms']:7.3f}ms  "
          f"std={s['std_ms']:6.3f}ms  p95={s['p95_ms']:7.3f}ms  p99={s['p99_ms']:7.3f}ms  "
          f"max={s['max_ms']:7.3f}ms  FPS={s['fps_mean_based']:6.1f}  (n={s['n']})")


# ===================================================================
# Part 1: latency benchmark across configs A-D, two resolutions
# ===================================================================
def _run_latency_benchmark(label, calibration, rectify, config_name, config_factory):
    w, h = calibration.image_size
    left, right = _noise_stereo_pair(w, h)
    config = config_factory()
    pipeline = DepthPerceptionPipeline(config, calibration, rectify=rectify, body_T_camera_left=_transform())

    use_motion = config_name in _TEMPORAL_CONFIGS

    # Warm-up: N_WARMUP discarded iterations, recorded separately (raw,
    # not discarded from the report) to show warm-up behavior explicitly.
    warmup_samples = np.empty(N_WARMUP, dtype=np.float64)
    for i in range(N_WARMUP):
        t0 = time.perf_counter()
        if use_motion:
            pipeline.process(left, right, left_timestamp=float(i), motion_hints=[_zero_motion_hint(float(i))])
        else:
            pipeline.process(left, right)
        warmup_samples[i] = (time.perf_counter() - t0) * 1000.0

    total_samples = np.empty(N_ITERS, dtype=np.float64)
    for i in range(N_ITERS):
        t0 = time.perf_counter()
        if use_motion:
            ts = float(N_WARMUP + i)
            pipeline.process(left, right, left_timestamp=ts, motion_hints=[_zero_motion_hint(ts)])
        else:
            pipeline.process(left, right)
        total_samples[i] = (time.perf_counter() - t0) * 1000.0

    s = _stats(total_samples)
    ws = _stats(warmup_samples)
    print(f"  warm-up (n={N_WARMUP}, discarded from steady-state stats below): "
          f"first={warmup_samples[0]:.3f}ms  mean={ws['mean_ms']:.3f}ms  last={warmup_samples[-1]:.3f}ms")
    _print_stats(config_name, s)

    return {
        "resolution": f"{w}x{h}", "rectify": rectify, "config": config_name,
        "warmup": {"n": N_WARMUP, "first_ms": float(warmup_samples[0]),
                   "last_ms": float(warmup_samples[-1]), "mean_ms": ws["mean_ms"]},
        "steady_state": s,
    }


# ===================================================================
# Part 2: sustained run — memory, temporal-history/persistence
# boundedness, GeometryFrame zero-duplication, reset() behavior
# ===================================================================
def _current_rss_kib() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
    except FileNotFoundError:
        pass
    return float("nan")


def _run_sustained(calibration, rectify):
    w, h = calibration.image_size
    left, right = _noise_stereo_pair(w, h)
    config = _config_d_full_v1_candidate()
    pipeline = DepthPerceptionPipeline(config, calibration, rectify=rectify, body_T_camera_left=_transform())

    print(f"\nWarm-up: {N_WARMUP} discarded frames before tracking begins.")
    for i in range(N_WARMUP):
        pipeline.process(left, right, left_timestamp=float(i), motion_hints=[_zero_motion_hint(float(i))])
    gc.collect()

    print(f"Sustained run: {N_SUSTAINED_FRAMES} frames, checkpoint every {SUSTAINED_CHECKPOINT_EVERY}.\n")
    print(f"{'frame':>6} {'RSS (KiB)':>11} {'dRSS':>9} {'hist_len':>9} "
          f"{'persist_shape':>15} {'latency_ms':>11}")

    checkpoints = []
    rss_start = None
    result = None
    persistence_shape_samples = set()
    history_len_samples = []
    latency_samples = np.empty(N_SUSTAINED_FRAMES, dtype=np.float64)

    for frame_idx in range(1, N_SUSTAINED_FRAMES + 1):
        ts = float(N_WARMUP + frame_idx)
        t0 = time.perf_counter()
        result = pipeline.process(left, right, left_timestamp=ts, motion_hints=[_zero_motion_hint(ts)])
        latency_samples[frame_idx - 1] = (time.perf_counter() - t0) * 1000.0

        hist_len = len(pipeline._temporal_history) if pipeline._temporal_history is not None else None
        history_len_samples.append(hist_len)
        tracker = pipeline._temporal_persistence_tracker
        if tracker is not None and tracker._support_count is not None:
            persistence_shape_samples.add(tuple(tracker._support_count.shape))

        if frame_idx % SUSTAINED_CHECKPOINT_EVERY == 0 or frame_idx == 1:
            gc.collect()
            rss = _current_rss_kib()
            if rss_start is None:
                rss_start = rss
            shape = tuple(tracker._support_count.shape) if (tracker is not None and tracker._support_count is not None) else None
            checkpoints.append({
                "frame": frame_idx, "rss_kib": rss, "d_rss_kib": rss - rss_start,
                "temporal_history_len": hist_len, "persistence_grid_shape": shape,
            })
            print(f"{frame_idx:>6} {rss:>11.1f} {rss - rss_start:>+9.1f} {str(hist_len):>9} "
                  f"{str(shape):>15} {latency_samples[frame_idx - 1]:>11.3f}")

    # --- GeometryFrame zero-duplication check (last frame) ---
    gf = result.geometry_frame
    zero_dup_checks = {
        "depth_map_is_result_depth_map": gf.depth_map is result.depth_map,
        "disparity_map_is_result_disparity_map": gf.disparity_map is result.disparity_map,
        "geometry_is_result_geometry": gf.geometry is result.geometry,
        "geometry_body_is_result_geometry_body": gf.geometry_body is result.geometry_body,
        "obstacle_cloud_is_result_obstacle_cloud": gf.obstacle_cloud is result.obstacle_cloud,
        "free_space_rays_is_result_free_space_rays": gf.free_space_rays is result.free_space_rays,
    }
    print(f"\nGeometryFrame zero-duplication (same object as DepthPerceptionResult's own field): {zero_dup_checks}")
    assert all(zero_dup_checks.values()), "GeometryFrame is duplicating an array/object instead of referencing it!"

    # --- output-size characteristics (last frame) ---
    output_sizes = {
        "depth_map_bytes": int(gf.depth_map.nbytes),
        "disparity_map_bytes": int(gf.disparity_map.nbytes),
        "geometry_body_points_bytes": int(gf.geometry_body.points.nbytes) if gf.geometry_body is not None else None,
        "obstacle_cloud_point_count": int(gf.obstacle_cloud.points.shape[0]) if gf.obstacle_cloud is not None else None,
        "free_space_rays_count": int(gf.free_space_rays.ranges_m.shape[0]) if gf.free_space_rays is not None else None,
        "region_evidence_count": len(gf.region_evidence) if gf.region_evidence is not None else None,
        "clearance_evidence_count": len(gf.clearance_evidence) if gf.clearance_evidence is not None else None,
        "surface_evidence_count": len(gf.surface_evidence) if gf.surface_evidence is not None else None,
        "boundary_evidence_count": len(gf.boundary_evidence) if gf.boundary_evidence is not None else None,
        "opening_evidence_count": len(gf.opening_evidence) if gf.opening_evidence is not None else None,
    }
    print(f"Output-size characteristics (last frame): {output_sizes}")

    # --- boundedness summary ---
    max_hist_len = max(x for x in history_len_samples if x is not None)
    print(f"\nTemporalHistory length over the run: max={max_hist_len}, "
          f"configured temporal_max_records={config.temporal_max_records} "
          f"-> bounded: {max_hist_len <= config.temporal_max_records}")
    print(f"TemporalPersistenceTracker internal grid shape(s) observed over the run: "
          f"{persistence_shape_samples} (must be a SINGLE fixed shape, never resized)")

    first_rss, last_rss = checkpoints[0]["rss_kib"], checkpoints[-1]["rss_kib"]
    total_delta = last_rss - first_rss
    print(f"\nRSS at frame {checkpoints[0]['frame']}: {first_rss:.1f} KiB; "
          f"at frame {checkpoints[-1]['frame']}: {last_rss:.1f} KiB; "
          f"total delta {total_delta:+.1f} KiB over {N_SUSTAINED_FRAMES} frames "
          f"({total_delta / N_SUSTAINED_FRAMES:+.2f} KiB/frame average).")
    deltas = [c["rss_kib"] - first_rss for c in checkpoints]
    monotonic = all(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1))
    if monotonic and total_delta > 2048:
        print("OBSERVATION: sustained upward RSS trend (> 2 MiB) — warrants further investigation, "
              "NOT claimed leak-free from this one run.")
    else:
        print(f"OBSERVATION: no monotonic RSS growth over {N_SUSTAINED_FRAMES} frames on this platform/run — "
              "evidence from one run of this length, not a formal leak-freedom proof.")

    # --- reset() behavior ---
    del result
    gc.collect()
    print("\nCalling pipeline.reset() ...")
    pipeline.reset()
    health_after_reset = pipeline.health()
    hist_len_after_reset = len(pipeline._temporal_history) if pipeline._temporal_history is not None else None
    ts = float(N_WARMUP + N_SUSTAINED_FRAMES + 1)
    t0 = time.perf_counter()
    result_after_reset = pipeline.process(left, right, left_timestamp=ts, motion_hints=[_zero_motion_hint(ts)])
    first_post_reset_latency_ms = (time.perf_counter() - t0) * 1000.0

    reset_report = {
        "frames_processed_after_reset_call": health_after_reset.frames_processed,
        "temporal_history_len_after_reset_call": hist_len_after_reset,
        "temporal_consistency_state_after_reset_and_one_frame": (
            result_after_reset.geometry_frame.temporal_consistency.state
            if result_after_reset.geometry_frame and result_after_reset.geometry_frame.temporal_consistency
            else None
        ),
        "frames_processed_after_one_post_reset_frame": pipeline.health().frames_processed,
        "first_post_reset_frame_latency_ms": first_post_reset_latency_ms,
        "first_pre_warmup_frame_latency_ms": None,  # filled in by caller for comparison
    }
    print(f"reset() report: {reset_report}")
    print("Expected baseline: frames_processed==0 immediately after reset(), TemporalHistory length==0, "
          "temporal_consistency.state=='INSUFFICIENT_EVIDENCE' on the very next frame (no prior record survives "
          "reset — same as a genuine first frame), frames_processed==1 after that one frame.")
    baseline_matches = (
        reset_report["frames_processed_after_reset_call"] == 0
        and reset_report["temporal_history_len_after_reset_call"] == 0
        and reset_report["temporal_consistency_state_after_reset_and_one_frame"] == "INSUFFICIENT_EVIDENCE"
        and reset_report["frames_processed_after_one_post_reset_frame"] == 1
    )
    print(f"reset() returns pipeline to documented baseline: {baseline_matches}")

    return {
        "checkpoints": checkpoints,
        "zero_duplication_checks": zero_dup_checks,
        "output_sizes": output_sizes,
        "temporal_history_max_len": max_hist_len,
        "temporal_history_configured_max": config.temporal_max_records,
        "temporal_history_bounded": bool(max_hist_len <= config.temporal_max_records),
        "persistence_grid_shapes_observed": [list(s) for s in persistence_shape_samples],
        "persistence_grid_shape_stable": len(persistence_shape_samples) == 1,
        "rss_first_kib": first_rss, "rss_last_kib": last_rss, "rss_total_delta_kib": total_delta,
        "rss_monotonic_non_decreasing": monotonic,
        "reset_behavior": reset_report,
        "reset_returns_to_documented_baseline": baseline_matches,
        "sustained_latency_stats": _stats(latency_samples),
    }


# ===================================================================
# Part 3: lightweight profiling (cProfile), full V1 candidate config
# ===================================================================
def _run_profile(calibration, rectify, n_frames=30):
    w, h = calibration.image_size
    left, right = _noise_stereo_pair(w, h)
    config = _config_d_full_v1_candidate()
    pipeline = DepthPerceptionPipeline(config, calibration, rectify=rectify, body_T_camera_left=_transform())

    for i in range(N_WARMUP):
        pipeline.process(left, right, left_timestamp=float(i), motion_hints=[_zero_motion_hint(float(i))])

    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(n_frames):
        ts = float(N_WARMUP + i)
        pipeline.process(left, right, left_timestamp=ts, motion_hints=[_zero_motion_hint(ts)])
    profiler.disable()

    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    report_text = buf.getvalue()
    print(report_text)

    # Machine-readable top contributors (own-code, i.e. filtered to this
    # package + directly-called cv2/numpy entry points), by cumulative time.
    top = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, lineno, funcname = func
        top.append({
            "function": f"{os.path.basename(filename)}:{lineno}({funcname})",
            "cumulative_s": ct, "total_s": tt, "calls": nc,
        })
    top.sort(key=lambda d: d["cumulative_s"], reverse=True)
    return {"n_frames_profiled": n_frames, "top_20_by_cumulative_time": top[:20]}


# ===================================================================
# main
# ===================================================================
def main() -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    real_calibration = load_stereo_calibration("examples/config/stereo_calibration.xml")
    synthetic_calibration = _synthetic_calibration(640, 480)

    resolutions = [
        ("real_hardware_320x240", real_calibration, True),
        ("synthetic_640x480", synthetic_calibration, False),
    ]

    print("=" * 100)
    print("PHASE D14 — PERFORMANCE / BOUNDED-RESOURCE VALIDATION")
    print("=" * 100)

    latency_results = []
    for res_label, calibration, rectify in resolutions:
        print(f"\n\n### Resolution: {res_label} ({calibration.image_size[0]}x{calibration.image_size[1]}, rectify={rectify}) ###")
        for config_name, factory in CONFIGS.items():
            print(f"\n-- Config {config_name} --")
            r = _run_latency_benchmark(res_label, calibration, rectify, config_name, factory)
            r["resolution_label"] = res_label
            latency_results.append(r)

    print("\n\n### Sustained run (full V1 candidate config, real hardware 320x240) ###")
    sustained_result = _run_sustained(real_calibration, True)

    print("\n\n### Lightweight profiling (full V1 candidate config, real hardware 320x240, "
          f"{30} frames) ###")
    profile_result = _run_profile(real_calibration, True, n_frames=30)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "platform": sys.platform,
        },
        "latency_benchmarks": latency_results,
        "sustained_run": sustained_result,
        "profile": profile_result,
    }
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n\nMachine-readable results written to: {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
