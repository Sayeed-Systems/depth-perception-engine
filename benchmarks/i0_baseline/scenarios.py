"""
Scenario definitions — Phase I0 benchmark freeze.

This module ONLY constructs scenes and their analytically-known ground
truth. It never calls into depth_perception_engine's geometry/pipeline
code and never measures anything — that is measure.py's job (task 3's
required separation of "scenario definition" from "measurement
collection").

Every scenario's ground truth is derived from exactly two sources:
  1. This repo's checked-in calibration config
     (examples/config/stereo_calibration.xml), via calibration_geometry.py.
  2. depth_perception_engine.config.PipelineConfig's own documented
     defaults (grid sizes, support/threshold values) — read directly off
     a real PipelineConfig() instance, never re-typed as a magic number.

No new acceptance thresholds are introduced anywhere in this file.
Scenario geometry (target distances, tilt angle, valid-fraction split,
noise seeds) matches this repo's own existing D7/D10/D14 ground-truth
test precedent (tests/test_d10_integrated_ground_truth.py,
tests/test_clearance_geometry.py, examples/benchmark_d14_provider_validation.py)
so the "frozen" baseline is built on already-reviewed scenario choices,
not new ones invented for this freeze.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from depth_perception_engine import PipelineConfig
from depth_perception_engine.frames import FrameId

from benchmarks.i0_baseline.calibration_geometry import derive_geometry, load_calibration

_DEFAULT_CONFIG = PipelineConfig()


@dataclass
class Scenario:
    name: str
    description: str
    ground_truth: dict
    arrays: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


def _geom():
    calibration = load_calibration()
    return calibration, derive_geometry(calibration)


# ===================================================================
# 1 — depth-by-target: known fronto-parallel planes at three distances
# ===================================================================
TARGET_DISTANCES_M = {"near": 1.0, "mid": 3.0, "far": 8.0}


def depth_by_target_scenarios() -> list:
    _, geom = _geom()
    fx, baseline, w, h = geom["fx_px"], geom["baseline_m"], geom["width"], geom["height"]
    scenarios = []
    for target_name, z_m in TARGET_DISTANCES_M.items():
        disparity = np.full((h, w), fx * baseline / z_m, dtype=np.float32)
        scenarios.append(Scenario(
            name=f"depth_target_{target_name}",
            description=f"Fronto-parallel plane at exactly {z_m} m, uniform disparity "
                        "from the closed-form pinhole formula disparity = fx*baseline/Z.",
            ground_truth={"target": target_name, "expected_depth_m": z_m},
            arrays={"disparity": disparity},
        ))
    return scenarios


# ===================================================================
# 2 — slanted plane: surface-normal angular error / planarity
# (identical scenario to tests/test_d10_integrated_ground_truth.py's
# TestScenario3SlantedPlaneSurfaceNormal)
# ===================================================================
SLANTED_PLANE_THETA_DEG = 15.0
SLANTED_PLANE_Z0_M = 2.0


def slanted_plane_scenario() -> Scenario:
    _, geom = _geom()
    fx, cx, baseline, w, h = geom["fx_px"], geom["cx_px"], geom["baseline_m"], geom["width"], geom["height"]
    theta = np.deg2rad(SLANTED_PLANE_THETA_DEG)
    nx, nz = np.sin(theta), -np.cos(theta)
    d = nz * SLANTED_PLANE_Z0_M
    u = np.arange(w)
    z_u = d / (nx * (u - cx) / fx + nz)
    disparity_u = fx * baseline / z_u
    disparity = np.tile(disparity_u.astype(np.float32), (h, 1))
    return Scenario(
        name="slanted_plane",
        description=f"Single plane tilted {SLANTED_PLANE_THETA_DEG} deg about Y, filling the "
                    "whole frame; disparity from the closed-form pinhole/plane intersection.",
        ground_truth={"expected_normal": [float(nx), 0.0, float(nz)]},
        arrays={"disparity": disparity},
        params={"grid_rows": 1, "grid_cols": 1, "min_support_count": 3},
    )


# ===================================================================
# 3 — boundary + opening: known near/far/near row
# (identical scenario to tests/test_d10_integrated_ground_truth.py's
# TestScenario245KnownDepthsBoundaryAndOpening)
# ===================================================================
BOUNDARY_NEAR_M = 1.0
BOUNDARY_FAR_M = 5.0
BOUNDARY_GRID_COLS = 5


def boundary_opening_scenario() -> Scenario:
    _, geom = _geom()
    w, h = geom["width"], geom["height"]
    col_bounds = np.linspace(0, w, BOUNDARY_GRID_COLS + 1).astype(int)
    depth = np.zeros((h, w), dtype=np.float32)
    for c, (lo, hi) in enumerate(zip(col_bounds[:-1], col_bounds[1:])):
        depth[:, lo:hi] = BOUNDARY_NEAR_M if c in (0, 1, 4) else BOUNDARY_FAR_M
    return Scenario(
        name="boundary_opening_row",
        description="One row, 5 columns: [near][near][far][far][near] — two real depth "
                    "discontinuities flanking one confirmed opening (the FAR pair).",
        ground_truth={
            "near_m": BOUNDARY_NEAR_M, "far_m": BOUNDARY_FAR_M,
            "expected_discontinuity_cols": [1, 3],
            "expected_no_discontinuity_cols": [0, 2],
            "expected_opening_range_m": BOUNDARY_FAR_M,
            "expected_opening_col_span": [2, 3],
        },
        arrays={"depth_map": depth},
        params={
            "grid_rows": 1, "grid_cols": BOUNDARY_GRID_COLS,
            "min_support_count": 5,
            "depth_step_threshold_m": _DEFAULT_CONFIG.boundary_depth_step_threshold_m,
            "orientation_change_threshold_rad": _DEFAULT_CONFIG.boundary_orientation_change_threshold_rad,
            "min_range_ratio": _DEFAULT_CONFIG.opening_min_range_ratio,
        },
    )


# ===================================================================
# 4 — valid-fraction / degradation-state ladder: NO_USABLE_GEOMETRY /
# DEGRADED / HEALTHY, all hand-picked relative to PipelineConfig's own
# real thresholds (identical scenario to test_d10's Scenario 6)
# ===================================================================
DEGRADED_INVALID_FRACTION = 0.7  # -> valid_fraction=0.3, strictly between the two thresholds


def valid_fraction_scenarios() -> list:
    _, geom = _geom()
    fx, baseline, w, h = geom["fx_px"], geom["baseline_m"], geom["width"], geom["height"]
    mid_disparity = fx * baseline / 2.0

    healthy = np.full((h, w), mid_disparity, dtype=np.float32)

    degraded = np.full((h, w), mid_disparity, dtype=np.float32)
    n_invalid_cols = int(w * DEGRADED_INVALID_FRACTION)
    degraded[:, :n_invalid_cols] = 0.0
    expected_degraded_valid_fraction = (w - n_invalid_cols) / w

    insufficient = np.zeros((h, w), dtype=np.float32)

    return [
        Scenario(
            name="valid_fraction_healthy",
            description="Disparity valid everywhere -> valid_fraction=1.0.",
            ground_truth={
                "expected_valid_fraction": 1.0,
                "expected_geometry_quality": "HEALTHY",
                "expected_overall_state": "VALID",
            },
            arrays={"disparity": healthy},
        ),
        Scenario(
            name="valid_fraction_degraded",
            description=f"Disparity valid on exactly the right {(1 - DEGRADED_INVALID_FRACTION) * 100:.0f}% "
                        "of columns -> valid_fraction strictly between the configured "
                        "degraded/healthy thresholds.",
            ground_truth={
                "expected_valid_fraction": expected_degraded_valid_fraction,
                "expected_geometry_quality": "DEGRADED",
                "expected_overall_state": "DEGRADED",
            },
            arrays={"disparity": degraded},
        ),
        Scenario(
            name="valid_fraction_insufficient",
            description="Disparity invalid everywhere -> valid_fraction=0.0.",
            ground_truth={
                "expected_valid_fraction": 0.0,
                "expected_geometry_quality": "NO_USABLE_GEOMETRY",
                "expected_overall_state": "INSUFFICIENT",
            },
            arrays={"disparity": insufficient},
        ),
    ]


# ===================================================================
# 5 — clearance / bearing: known beam column range at a known distance
# (same bearing formula as tests/test_clearance_geometry.py's
# TestBearingDerivation, but using this repo's REAL calibration
# fx/cx instead of that unit test's arbitrary 200.0/160.0 constants —
# ground truth comes only from the checked-in calibration config)
# ===================================================================
def clearance_bearing_scenario() -> Scenario:
    _, geom = _geom()
    fx, cx, w = geom["fx_px"], geom["cx_px"], geom["width"]
    x1, x2 = int(w * 0.4), int(w * 0.65)
    distance_m = 4.0
    expected_bearing_center_rad = float(np.arctan2(((x1 + x2) / 2.0) - cx, fx))
    return Scenario(
        name="clearance_bearing",
        description="One beam spanning a known pixel column range at a known distance; "
                    "expected bearing from the closed-form atan2((u-cx)/fx) formula, "
                    "using this repo's real calibrated fx/cx.",
        ground_truth={
            "expected_distance_m": distance_m,
            "expected_bearing_center_rad": expected_bearing_center_rad,
        },
        params={
            "x1": x1, "x2": x2, "distance_m": distance_m,
            "focal_length_px": fx, "principal_point_x_px": cx,
            "min_coverage_fraction": 0.5,
        },
    )


# ===================================================================
# 6 — latency / FPS: deterministic noise stereo pair, full V1
# candidate configuration, real hardware calibration
# (identical methodology to examples/benchmark_d14_provider_validation.py's
# config D / real_hardware_320x240 case)
# ===================================================================
LATENCY_N_WARMUP = 15
LATENCY_N_ITERS = 100
LATENCY_SEED = 0


def latency_scenario() -> Scenario:
    calibration, geom = _geom()
    w, h = geom["width"], geom["height"]
    rng = np.random.default_rng(LATENCY_SEED)
    left = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    config = PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        enable_geometry_frame=True,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=30,
    )
    return Scenario(
        name="latency_full_v1_candidate",
        description="Full V1 candidate PipelineConfig, deterministic i.i.d. noise stereo "
                    "pair, real hardware calibration (rectify=True) — same methodology as "
                    "examples/benchmark_d14_provider_validation.py's config D.",
        ground_truth={},  # no acceptance threshold exists for latency; observed only
        arrays={"left": left, "right": right},
        params={
            "calibration": calibration, "rectify": True, "config": config,
            "n_warmup": LATENCY_N_WARMUP, "n_iters": LATENCY_N_ITERS,
        },
    )
