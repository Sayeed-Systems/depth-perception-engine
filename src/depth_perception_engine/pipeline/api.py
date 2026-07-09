"""
Stateless, function-level public API.

Each function here builds whatever engine it needs fresh, on every call.
That's fine for compute_disparity/estimate_depth/classify_traversability,
which are pure per-frame computations — but detect_obstacles and
process_stereo_pair both go through obstacles.ThreatAssessor, which holds
per-beam EMA-smoothing and status-debouncing state ACROSS calls. Calling
these functions fresh every frame throws that smoothing away and reproduces
raw, unsmoothed SGBM noise frame-to-frame.

For real-time / repeated-frame use (i.e. what mp01_perception will do),
use pipeline.pipeline.DepthPerceptionPipeline instead — it builds every
engine once and reuses it, exactly like the original main.py's
build_pipeline() did. These functions exist for one-shot use, scripting,
and tests.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.config.pipeline_config import PipelineConfig
from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.fusion.result_builder import build_result, to_obstacle_assessment
from depth_perception_engine.models.result import (
    DepthPerceptionResult,
    ObstacleAssessment,
    TraversabilityResult,
)
from depth_perception_engine.obstacles.threat_assessment import ThreatAssessor
from depth_perception_engine.stereo.disparity_engine import DisparityEngine
from depth_perception_engine.stereo.rectification import RectificationEngine
from depth_perception_engine.traversability.scene_interpreter import SceneInterpreter
from depth_perception_engine.utils.validation import require_matching_stereo_pair


def compute_disparity(
    left_image: np.ndarray,
    right_image: np.ndarray,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a disparity map from a rectified stereo pair.

    Returns:
        ``(raw_disparity, visualization_disparity)`` — see
        stereo.disparity_engine.DisparityEngine.compute_disparity.
    """
    engine = DisparityEngine(
        min_disparity=config.min_disparity,
        num_disparities=config.num_disparities,
        block_size=config.block_size,
    )
    return engine.compute_disparity(left_image, right_image)


def estimate_depth(disparity: np.ndarray, calibration: StereoCalibration) -> np.ndarray:
    """Convert a raw disparity map to a metric depth map (metres).

    See depth.depth_estimator.DepthEstimator.estimate.
    """
    estimator = DepthEstimator.from_calibration(calibration)
    return estimator.estimate(disparity)


def classify_traversability(
    gray_image: np.ndarray,
    disparity_map: np.ndarray,
    depth_map: np.ndarray,
    config: PipelineConfig,
) -> TraversabilityResult:
    """Classify the scene into a region grid plus one global navigation decision.

    Note this needs the rectified grayscale image and the raw disparity map
    in addition to the depth map — the underlying algorithm
    (traversability.SceneInterpreter) judges each region using texture and
    disparity-validity signals the depth map alone doesn't carry, not depth
    values only. See traversability/region_analyzer.py's module docstring
    for why disparity validity can't be inferred from depth alone.
    """
    interpreter = SceneInterpreter(
        rows=config.traversability_grid_rows,
        cols=config.traversability_grid_cols,
        caution_m=config.caution_distance_m,
        clear_m=config.clear_distance_m,
        ambiguous_fraction_thresh=config.traversability_ambiguous_fraction_thresh,
    )
    regions = interpreter.analyze(gray_image, disparity_map, depth_map)
    decision = interpreter.decide_navigation(regions)
    return TraversabilityResult(regions=regions, decision=decision)


def detect_obstacles(
    depth_map: np.ndarray,
    config: PipelineConfig,
    raw_disparity: Optional[np.ndarray] = None,
) -> ObstacleAssessment:
    """Scan the depth map for obstacles in config.n_beams vertical beams.

    Stateless: builds a fresh ThreatAssessor every call, so EMA smoothing
    and status debouncing do NOT persist across calls made through this
    function — see this module's docstring. Use DepthPerceptionPipeline for
    frame-to-frame smoothing.
    """
    assessor = ThreatAssessor(
        n_beams=config.n_beams,
        clear_m=config.clear_distance_m,
        caution_m=config.caution_distance_m,
        percentile=config.obstacle_percentile,
        min_valid=config.obstacle_min_valid_px,
        blocked_invalid_ratio=config.obstacle_blocked_invalid_ratio,
        ema_alpha=config.obstacle_ema_alpha,
        debounce_frames=config.obstacle_debounce_frames,
        dead_zone_px=config.resolved_obstacle_dead_zone_px(),
    )
    return to_obstacle_assessment(assessor.assess(depth_map, raw_disparity))


def process_stereo_pair(
    left_image: np.ndarray,
    right_image: np.ndarray,
    config: PipelineConfig,
    calibration: StereoCalibration,
    rectify: bool = True,
) -> DepthPerceptionResult:
    """One-shot: run the full pipeline over a single stereo pair.

    Stateless — see this module's docstring. For repeated frames (i.e. a
    live or ROS2 feed), construct a DepthPerceptionPipeline once instead and
    call .process() on it; that preserves obstacle-detection temporal
    smoothing across frames the way this function cannot.

    Args:
        left_image, right_image: NumPy stereo pair (BGR or grayscale),
            already split — this function does not split a combined frame.
        config: Tunable thresholds.
        calibration: Loaded StereoCalibration (required — depth estimation
            has no meaning without it).
        rectify: If True (default), rectify left/right using *calibration*
            before computing disparity. Set False if the images are already
            rectified upstream. On a rectification failure, falls back to
            the unrectified images, matching the original pipeline's
            behavior.
    """
    require_matching_stereo_pair(left_image, right_image)
    t0 = time.perf_counter()

    left, right = left_image, right_image
    if rectify:
        rectifier = RectificationEngine(calibration)
        rectifier.initialize_rectification()
        try:
            left, right = rectifier.rectify(left, right)
        except (ValueError, RuntimeError):
            left, right = left_image, right_image

    raw_disparity, _ = compute_disparity(left, right, config)
    depth_map = estimate_depth(raw_disparity, calibration)

    gray = left if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    traversability = classify_traversability(gray, raw_disparity, depth_map, config)
    obstacles = detect_obstacles(depth_map, config, raw_disparity=raw_disparity)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return build_result(raw_disparity, depth_map, traversability, obstacles, elapsed_ms)
