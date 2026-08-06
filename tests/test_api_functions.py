"""The five stateless top-level pipeline.api functions, called individually."""

import cv2
import numpy as np

from depth_perception_engine.models import DepthPerceptionResult, ObstacleAssessment, TraversabilityResult
from depth_perception_engine.pipeline import (
    classify_traversability,
    compute_disparity,
    detect_obstacles,
    estimate_depth,
    process_stereo_pair,
)


def test_compute_disparity(config, stereo_pair):
    left, right = stereo_pair
    raw_disparity, visualization = compute_disparity(left, right, config)
    assert raw_disparity.shape == left.shape[:2]
    assert raw_disparity.dtype == np.float32
    assert visualization.shape == left.shape[:2]
    assert visualization.dtype == np.uint8


def test_estimate_depth(config, calibration, stereo_pair):
    left, right = stereo_pair
    raw_disparity, _ = compute_disparity(left, right, config)
    depth_map = estimate_depth(raw_disparity, calibration)
    assert depth_map.shape == raw_disparity.shape
    assert depth_map.dtype == np.float32


def test_classify_traversability(config, calibration, stereo_pair):
    left, right = stereo_pair
    raw_disparity, _ = compute_disparity(left, right, config)
    depth_map = estimate_depth(raw_disparity, calibration)
    gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    result = classify_traversability(gray, raw_disparity, depth_map, config)
    assert isinstance(result, TraversabilityResult)
    assert len(result.regions) == config.traversability_grid_rows * config.traversability_grid_cols


def test_detect_obstacles(config, calibration, stereo_pair):
    left, right = stereo_pair
    raw_disparity, _ = compute_disparity(left, right, config)
    depth_map = estimate_depth(raw_disparity, calibration)

    result = detect_obstacles(depth_map, config, raw_disparity=raw_disparity)
    assert isinstance(result, ObstacleAssessment)
    assert len(result.beams) == config.n_beams


def test_process_stereo_pair_one_shot(config, calibration, stereo_pair):
    left, right = stereo_pair
    result = process_stereo_pair(left, right, config, calibration)
    assert isinstance(result, DepthPerceptionResult)


def test_process_stereo_pair_raises_on_rectification_failure(config, calibration):
    """Regression coverage for Issue C (remediation) in the stateless api.py
    entry point — mirrors test_pipeline.py's equivalent for
    DepthPerceptionPipeline. A frame-size mismatch against calibration must
    raise, not silently process the unrectified pair."""
    wrong_w, wrong_h = calibration.image_size[0] // 2, calibration.image_size[1] // 2
    rng = np.random.default_rng(7)
    left = rng.integers(0, 255, (wrong_h, wrong_w, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (wrong_h, wrong_w, 3), dtype=np.uint8)

    try:
        process_stereo_pair(left, right, config, calibration, rectify=True)
        assert False, "expected rectification failure to raise"
    except ValueError as exc:
        assert "does not match calibrated size" in str(exc)


def test_detect_obstacles_is_stateless_across_calls(config, calibration, stereo_pair):
    """Unlike DepthPerceptionPipeline, this functional form builds a fresh
    ThreatAssessor every call — so repeated calls on identical input must
    produce identical output (no carried-over EMA/debounce state)."""
    left, right = stereo_pair
    raw_disparity, _ = compute_disparity(left, right, config)
    depth_map = estimate_depth(raw_disparity, calibration)

    first = detect_obstacles(depth_map, config, raw_disparity=raw_disparity)
    second = detect_obstacles(depth_map, config, raw_disparity=raw_disparity)

    first_statuses = [b.status for b in first.beams]
    second_statuses = [b.status for b in second.beams]
    assert first_statuses == second_statuses
