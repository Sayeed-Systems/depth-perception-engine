"""
DepthPerceptionPipeline — the primary, stateful entry point.

Usage:

    from depth_perception_engine.pipeline import DepthPerceptionPipeline
    from depth_perception_engine.calibration import load_stereo_calibration
    from depth_perception_engine.config import PipelineConfig

    calibration = load_stereo_calibration("/path/to/stereo_calibration.xml")
    pipeline = DepthPerceptionPipeline(PipelineConfig(), calibration)

    result = pipeline.process(left_image, right_image)   # per frame

Every engine (DisparityEngine, RectificationEngine, ThreatAssessor) is
built ONCE in __init__ and reused across process() calls — mirroring the
original standalone main.py's build_pipeline(), and critically preserving
obstacles.ThreatAssessor's per-beam EMA smoothing and status debouncing
across frames. This is the entry point mp01_perception's
perception_processor.py should hold one instance of (constructed once,
e.g. in its own __init__) and call .process() on every frame — see
docs/INTEGRATION_READINESS.md.
"""

import time
from typing import Optional

import cv2
import numpy as np

from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.config.pipeline_config import PipelineConfig
from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.fusion.result_builder import build_result, to_obstacle_assessment
from depth_perception_engine.models.result import DepthPerceptionResult, TraversabilityResult
from depth_perception_engine.obstacles.threat_assessment import ThreatAssessor
from depth_perception_engine.stereo.disparity_engine import DisparityEngine
from depth_perception_engine.stereo.rectification import RectificationEngine
from depth_perception_engine.traversability.scene_interpreter import SceneInterpreter
from depth_perception_engine.utils.validation import require_matching_stereo_pair


class DepthPerceptionPipeline:
    """Stateful, reusable depth-perception pipeline for repeated frames."""

    def __init__(
        self,
        config: PipelineConfig,
        calibration: StereoCalibration,
        rectify: bool = True,
    ) -> None:
        """
        Args:
            config: Tunable thresholds (see config.PipelineConfig).
            calibration: Loaded StereoCalibration — required; depth
                estimation and (if rectify=True) rectification both need it.
                There is no default and no file path is read here: load it
                explicitly first with calibration.load_stereo_calibration().
            rectify: If True (default), rectify each incoming pair before
                computing disparity. Set False if the caller already
                supplies rectified images.
        """
        self._config = config
        self._calibration = calibration
        self._rectify = rectify

        self._rectifier: Optional[RectificationEngine] = None
        if rectify:
            self._rectifier = RectificationEngine(calibration)
            self._rectifier.initialize_rectification()

        self._disparity_engine = DisparityEngine(
            min_disparity=config.min_disparity,
            num_disparities=config.num_disparities,
            block_size=config.block_size,
        )
        self._depth_estimator = DepthEstimator.from_calibration(calibration)
        self._scene_interpreter = SceneInterpreter(
            rows=config.traversability_grid_rows,
            cols=config.traversability_grid_cols,
            caution_m=config.caution_distance_m,
            clear_m=config.clear_distance_m,
            ambiguous_fraction_thresh=config.traversability_ambiguous_fraction_thresh,
        )
        # Holds per-beam EMA/debounce state across process() calls — this is
        # the whole reason DepthPerceptionPipeline exists as a persistent
        # object instead of a function.
        self._threat_assessor = ThreatAssessor(
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

    # ------------------------------------------------------------------
    def process(self, left_image: np.ndarray, right_image: np.ndarray) -> DepthPerceptionResult:
        """Run one stereo pair through the full pipeline.

        Args:
            left_image, right_image: NumPy stereo pair (BGR or grayscale),
                already split — this does not split a combined frame (see
                stereo.FrameSplitter if the caller still needs that).

        Returns:
            A DepthPerceptionResult.

        Raises:
            ValueError, RuntimeError: Propagated unchanged from
                RectificationEngine.rectify() (when rectify=True) on a
                rectification failure — see the comment at that call site
                for why this is not caught and silently degraded here.
        """
        require_matching_stereo_pair(left_image, right_image)
        t0 = time.perf_counter()

        left, right = left_image, right_image
        if self._rectifier is not None:
            # Deliberately NOT caught here. Rectification failure (e.g. a
            # frame size that no longer matches the loaded calibration —
            # ValueError from RectificationEngine._validate_frame; or
            # uninitialised maps — RuntimeError) used to be swallowed and
            # fall back to running SGBM/depth/traversability on the
            # UNRECTIFIED pair instead, producing plausible-looking but
            # systematically wrong depth with no signal anything was
            # wrong. A caller has no way to trust depth computed from
            # unrectified images against calibration-derived rectification
            # maps, so this must invalidate the whole frame, not
            # degrade silently. Letting this propagate means the caller
            # (e.g. mp01_perception's PerceptionNode, whose broad
            # `except Exception` around processor.process() already drops
            # the frame, records the exact error in its diagnostics'
            # last_error field, and logs a warning) treats it exactly
            # like any other genuine processing failure — one bad frame
            # is dropped, not silently trusted.
            left, right = self._rectifier.rectify(left, right)

        # Computed once and reused for both SGBM matching (below) and
        # traversability texture analysis — DisparityEngine used to
        # convert left to grayscale internally, duplicating this exact
        # conversion on the same input every frame.
        gray = left if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        raw_disparity, _ = self._disparity_engine.compute_disparity(
            left, right, left_gray=gray, compute_visualization=False,
        )
        depth_map = self._depth_estimator.estimate(raw_disparity)

        regions = self._scene_interpreter.analyze(gray, raw_disparity, depth_map)
        decision = self._scene_interpreter.decide_navigation(regions)
        traversability = TraversabilityResult(regions=regions, decision=decision)

        obstacles = to_obstacle_assessment(
            self._threat_assessor.assess(depth_map, raw_disparity)
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return build_result(raw_disparity, depth_map, traversability, obstacles, elapsed_ms)

    # ------------------------------------------------------------------
    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def calibration(self) -> StereoCalibration:
        return self._calibration

    def __repr__(self) -> str:
        return (
            f"DepthPerceptionPipeline(image_size={self._calibration.image_size}, "
            f"rectify={self._rectify}, n_beams={self._config.n_beams}, "
            f"grid={self._config.traversability_grid_rows}x"
            f"{self._config.traversability_grid_cols})"
        )
