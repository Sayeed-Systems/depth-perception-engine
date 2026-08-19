"""
Measurement collection — Phase I0 benchmark freeze.

This module takes the scenarios defined in scenarios.py and runs them
through DPE's real, public, UNMODIFIED functions/pipeline (read-only —
identical discipline to tests/test_d10_integrated_ground_truth.py and
examples/benchmark_d14_provider_validation.py: no algorithm code is
touched, no output is patched, nothing is mocked). It only OBSERVES and
RECORDS. No pass/fail threshold is asserted anywhere in this file — see
scenarios.py's own docstring for why (task requirement: do not introduce
acceptance thresholds that did not already exist).

collect_all() returns one flat, JSON-serializable dict covering exactly
the eight categories required by the I0 baseline summary:
    depth_errors_by_target, valid_fractions, boundary_findings,
    opening_result, surface_normal, clearance_error, degradation_state,
    latency_fps
"""

import time

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, MotionHint
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.fusion.result_builder import build_clearance_evidence, build_geometry_frame_quality
from depth_perception_engine.geometry import (
    PointCloudBuilder,
    build_boundary_evidence,
    build_free_space_rays,
    build_geometry_metrics,
    build_obstacle_cloud,
    build_opening_evidence,
    build_surface_evidence,
    classify_geometry_quality,
)
from depth_perception_engine.models.result import BeamReading, ObstacleAssessment

from benchmarks.i0_baseline import scenarios as sc
from benchmarks.i0_baseline.calibration_geometry import derive_geometry, load_calibration

_ORIGIN = np.zeros(3)


def _calibration_and_geom():
    calibration = load_calibration()
    return calibration, derive_geometry(calibration)


# ===================================================================
# depth_errors_by_target
# ===================================================================
def measure_depth_errors_by_target() -> dict:
    calibration, _ = _calibration_and_geom()
    builder = PointCloudBuilder.from_calibration(calibration)
    results = {}
    for scenario in sc.depth_by_target_scenarios():
        cloud = builder.build(scenario.arrays["disparity"])
        expected_m = scenario.ground_truth["expected_depth_m"]
        measured_depths = cloud.points[..., 2][cloud.valid_mask]
        measured_m = float(np.mean(measured_depths)) if measured_depths.size else None
        results[scenario.ground_truth["target"]] = {
            "expected_depth_m": expected_m,
            "measured_depth_m": measured_m,
            "abs_error_m": (abs(measured_m - expected_m) if measured_m is not None else None),
            "valid_fraction": float(cloud.valid_mask.mean()),
        }
    return results


# ===================================================================
# surface_normal (angular error + planarity)
# ===================================================================
def measure_surface_normal() -> dict:
    calibration, _ = _calibration_and_geom()
    scenario = sc.slanted_plane_scenario()
    cloud = PointCloudBuilder.from_calibration(calibration).build(scenario.arrays["disparity"])
    evidence = build_surface_evidence(cloud, _ORIGIN, **scenario.params)
    if not evidence or evidence[0].normal is None:
        return {"measured_normal": None, "angular_error_rad": None, "planarity": None}

    cell = evidence[0]
    expected_normal = np.asarray(scenario.ground_truth["expected_normal"])
    cos_angle = float(np.clip(np.dot(cell.normal, expected_normal), -1.0, 1.0))
    angular_error_rad = float(np.arccos(cos_angle))
    return {
        "expected_normal": scenario.ground_truth["expected_normal"],
        "measured_normal": [float(x) for x in cell.normal],
        "angular_error_rad": angular_error_rad,
        "planarity": float(cell.planarity) if cell.planarity is not None else None,
        "support_count": int(cell.support_count),
    }


# ===================================================================
# boundary_findings + opening_result
# ===================================================================
def measure_boundary_and_opening() -> dict:
    scenario = sc.boundary_opening_scenario()
    depth = scenario.arrays["depth_map"]
    params = scenario.params

    boundary_evidence = build_boundary_evidence(
        depth, FrameId.CAMERA_OPTICAL_LEFT,
        grid_rows=params["grid_rows"], grid_cols=params["grid_cols"],
        min_support_count=params["min_support_count"],
        depth_step_threshold_m=params["depth_step_threshold_m"],
        orientation_change_threshold_rad=params["orientation_change_threshold_rad"],
    )
    opening_evidence = build_opening_evidence(
        boundary_evidence, depth, FrameId.CAMERA_OPTICAL_LEFT,
        grid_rows=params["grid_rows"], grid_cols=params["grid_cols"],
        min_support_count=params["min_support_count"],
        min_range_ratio=params["min_range_ratio"],
        focal_length_px=derive_geometry(load_calibration())["fx_px"],
    )

    states = {int(e.col): e.state for e in boundary_evidence}
    steps = {int(e.col): (float(e.depth_step_m) if e.depth_step_m is not None else None) for e in boundary_evidence}

    boundary_findings = {
        "states_by_col": states,
        "depth_step_m_by_col": steps,
        "expected_discontinuity_cols": scenario.ground_truth["expected_discontinuity_cols"],
        "expected_no_discontinuity_cols": scenario.ground_truth["expected_no_discontinuity_cols"],
        "matches_expected": (
            all(states.get(c) == "OBSERVED_DISCONTINUITY" for c in scenario.ground_truth["expected_discontinuity_cols"])
            and all(states.get(c) == "NO_DISCONTINUITY" for c in scenario.ground_truth["expected_no_discontinuity_cols"])
        ),
    }

    if opening_evidence:
        opening = opening_evidence[0]
        opening_result = {
            "n_openings": len(opening_evidence),
            "col_span": [int(opening.col_start), int(opening.col_end)],
            "approx_range_m": float(opening.approx_range_m),
            "approx_width_m": float(opening.approx_width_m),
            "expected_range_m": scenario.ground_truth["expected_opening_range_m"],
            "expected_col_span": scenario.ground_truth["expected_opening_col_span"],
            "range_abs_error_m": abs(float(opening.approx_range_m) - scenario.ground_truth["expected_opening_range_m"]),
        }
    else:
        opening_result = {"n_openings": 0, "expected_range_m": scenario.ground_truth["expected_opening_range_m"]}

    return {"boundary_findings": boundary_findings, "opening_result": opening_result}


# ===================================================================
# valid_fractions + degradation_state
# ===================================================================
def measure_valid_fractions_and_degradation() -> dict:
    calibration, _ = _calibration_and_geom()
    builder = PointCloudBuilder.from_calibration(calibration)
    valid_fractions = {}
    degradation_state = {}

    for scenario in sc.valid_fraction_scenarios():
        cloud = builder.build(scenario.arrays["disparity"])
        obstacle_cloud = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)
        rays = build_free_space_rays(cloud, _ORIGIN)
        metrics = build_geometry_metrics(cloud, obstacle_cloud, rays)
        quality = classify_geometry_quality(
            metrics,
            healthy_min_valid_fraction=sc._DEFAULT_CONFIG.geometry_healthy_min_valid_fraction,
            degraded_min_valid_fraction=sc._DEFAULT_CONFIG.geometry_degraded_min_valid_fraction,
        )
        frame_quality = build_geometry_frame_quality(
            metrics, None, None, None,
            geometry_healthy_min_valid_fraction=sc._DEFAULT_CONFIG.geometry_healthy_min_valid_fraction,
            geometry_degraded_min_valid_fraction=sc._DEFAULT_CONFIG.geometry_degraded_min_valid_fraction,
        )

        name = scenario.name
        valid_fractions[name] = {
            "measured_valid_fraction": float(metrics.valid_fraction),
            "expected_valid_fraction": scenario.ground_truth["expected_valid_fraction"],
            "abs_error": abs(float(metrics.valid_fraction) - scenario.ground_truth["expected_valid_fraction"]),
        }
        degradation_state[name] = {
            "measured_geometry_quality": str(quality),
            "expected_geometry_quality": scenario.ground_truth["expected_geometry_quality"],
            "measured_overall_state": frame_quality.overall_state,
            "expected_overall_state": scenario.ground_truth["expected_overall_state"],
            "degradation_reasons": list(frame_quality.degradation_reasons),
        }

    return {"valid_fractions": valid_fractions, "degradation_state": degradation_state}


# ===================================================================
# clearance_error
# ===================================================================
def measure_clearance_error() -> dict:
    scenario = sc.clearance_bearing_scenario()
    p = scenario.params
    beam = BeamReading(
        index=0, x1=p["x1"], x2=p["x2"], distance_m=p["distance_m"],
        status="CLEAR", valid_count=p["x2"] - p["x1"], total_pixels=p["x2"] - p["x1"],
    )
    obstacles = ObstacleAssessment(beams=[beam], safest_beam=None)
    clearance = build_clearance_evidence(
        obstacles, FrameId.CAMERA_OPTICAL_LEFT,
        p["focal_length_px"], p["principal_point_x_px"], p["min_coverage_fraction"],
    )[0]

    expected_bearing = scenario.ground_truth["expected_bearing_center_rad"]
    expected_distance = scenario.ground_truth["expected_distance_m"]
    return {
        "measured_distance_m": float(clearance.nearest_distance_m),
        "expected_distance_m": expected_distance,
        "distance_abs_error_m": abs(float(clearance.nearest_distance_m) - expected_distance),
        "measured_bearing_center_rad": float(clearance.bearing_center_rad),
        "expected_bearing_center_rad": expected_bearing,
        "bearing_abs_error_rad": abs(float(clearance.bearing_center_rad) - expected_bearing),
    }


# ===================================================================
# latency_fps
# ===================================================================
def _stats_ms(samples_ms: np.ndarray) -> dict:
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


def measure_latency_fps() -> dict:
    scenario = sc.latency_scenario()
    p = scenario.params
    left, right = scenario.arrays["left"], scenario.arrays["right"]

    transform = RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )
    pipeline = DepthPerceptionPipeline(p["config"], p["calibration"], rectify=p["rectify"], body_T_camera_left=transform)

    def _hint(ts):
        return MotionHint(timestamp=ts, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY, valid=True)

    n_warmup, n_iters = p["n_warmup"], p["n_iters"]
    warmup_samples = np.empty(n_warmup, dtype=np.float64)
    for i in range(n_warmup):
        t0 = time.perf_counter()
        pipeline.process(left, right, left_timestamp=float(i), motion_hints=[_hint(float(i))])
        warmup_samples[i] = (time.perf_counter() - t0) * 1000.0

    steady_samples = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        ts = float(n_warmup + i)
        t0 = time.perf_counter()
        pipeline.process(left, right, left_timestamp=ts, motion_hints=[_hint(ts)])
        steady_samples[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "resolution": f"{p['calibration'].image_size[0]}x{p['calibration'].image_size[1]}",
        "rectify": p["rectify"],
        "warmup": {"n": n_warmup, "first_ms": float(warmup_samples[0]), "last_ms": float(warmup_samples[-1])},
        "steady_state": _stats_ms(steady_samples),
    }


# ===================================================================
# top-level
# ===================================================================
def collect_all() -> dict:
    boundary_and_opening = measure_boundary_and_opening()
    valid_and_degradation = measure_valid_fractions_and_degradation()
    return {
        "depth_errors_by_target": measure_depth_errors_by_target(),
        "valid_fractions": valid_and_degradation["valid_fractions"],
        "boundary_findings": boundary_and_opening["boundary_findings"],
        "opening_result": boundary_and_opening["opening_result"],
        "surface_normal": measure_surface_normal(),
        "clearance_error": measure_clearance_error(),
        "degradation_state": valid_and_degradation["degradation_state"],
        "latency_fps": measure_latency_fps(),
    }
