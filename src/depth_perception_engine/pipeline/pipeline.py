"""
DepthPerceptionPipeline — the primary, stateful entry point.

Usage:

    from depth_perception_engine.pipeline import DepthPerceptionPipeline
    from depth_perception_engine.calibration import load_stereo_calibration
    from depth_perception_engine.config import PipelineConfig

    calibration = load_stereo_calibration("/path/to/stereo_calibration.xml")
    pipeline = DepthPerceptionPipeline(PipelineConfig(), calibration)

    result = pipeline.process(left_image, right_image)   # per frame

Every engine (DisparityEngine, RectificationEngine, ThreatAssessor, and —
Level 3, Phase E3, when PipelineConfig.enable_geometry is True —
PointCloudBuilder) is built ONCE in __init__ and reused across process()
calls — mirroring the
original standalone main.py's build_pipeline(), and critically preserving
obstacles.ThreatAssessor's per-beam EMA smoothing and status debouncing
across frames. This is the entry point mp01_perception's
perception_processor.py should hold one instance of (constructed once,
e.g. in its own __init__) and call .process() on every frame — see
docs/INTEGRATION_READINESS.md.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from depth_perception_engine.calibration.contracts import RigCalibration
from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.config.pipeline_config import PipelineConfig
from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.fusion.result_builder import build_result, to_obstacle_assessment
from depth_perception_engine.geometry.free_space import build_free_space_rays
from depth_perception_engine.geometry.geometry_metrics import build_geometry_metrics
from depth_perception_engine.geometry.obstacle_extractor import build_obstacle_cloud
from depth_perception_engine.geometry.point_cloud_builder import PointCloudBuilder
from depth_perception_engine.geometry.rigid_transform import transform_point_cloud
from depth_perception_engine.models.result import (
    DepthPerceptionResult,
    PipelineHealth,
    StereoObservation,
    TraversabilityResult,
)
from depth_perception_engine.obstacles.threat_assessment import ThreatAssessor
from depth_perception_engine.stereo.disparity_engine import DisparityEngine
from depth_perception_engine.stereo.rectification import RectificationEngine
from depth_perception_engine.traversability.scene_interpreter import SceneInterpreter
from depth_perception_engine.utils.validation import require_matching_stereo_pair

logger = logging.getLogger(__name__)


class DepthPerceptionPipeline:
    """Stateful, reusable depth-perception pipeline for repeated frames."""

    def __init__(
        self,
        config: PipelineConfig,
        calibration: StereoCalibration,
        rectify: bool = True,
        body_T_camera_left: Optional[RigidTransform] = None,
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
            body_T_camera_left: Level 3, Phase E4. Optional camera-to-body
                extrinsic — a frames.RigidTransform with
                from_frame=FrameId.CAMERA_OPTICAL_LEFT,
                to_frame=FrameId.BODY, following the exact naming/apply
                convention frozen in frames.py/docs/COORDINATE_FRAMES.md
                (p_body = rotation @ p_camera + translation). Defaults to
                None — meaning "not calibrated," never "identity"; see
                calibration.contracts.RigCalibration's own docstring for
                why None must not be assumed to mean zero offset. New
                (additive) parameter — every existing call site
                (positional or keyword, up to and including `rectify`)
                continues to work unmodified. Validated the same way
                RigCalibration.__post_init__ already does (reused here,
                not re-implemented) — see _validate_body_transform below.
                Has no effect unless config.enable_geometry is also True
                (there is no camera-frame PointCloud to transform
                otherwise) — see process()'s geometry stage.
        """
        self._config = config
        self._calibration = calibration
        self._rectify = rectify
        self._body_T_camera_left = self._validate_body_transform(calibration, body_T_camera_left)

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
        # Level 3, Phase E3: built once (same discipline as every other
        # engine here), but only if geometry is actually enabled — the
        # disabled default must not pay even construction cost.
        self._point_cloud_builder: Optional[PointCloudBuilder] = (
            PointCloudBuilder.from_calibration(calibration)
            if config.enable_geometry
            else None
        )
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
        self._threat_assessor = self._build_threat_assessor()

        self._closed = False
        self._frames_processed = 0
        self._last_confidence: Optional[float] = None
        self._last_processing_time_ms: Optional[float] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_body_transform(
        calibration: StereoCalibration,
        body_T_camera_left: Optional[RigidTransform],
    ) -> Optional[RigidTransform]:
        """Validate body_T_camera_left via RigCalibration's own, already-
        frozen __post_init__ rules (to_frame == BODY, from_frame ==
        camera_frame_id) rather than re-implementing that check — the
        RigCalibration instance itself is discarded afterward; only the
        validated transform is kept, since process() needs nothing else
        from it. Returns None unchanged (RigCalibration itself already
        treats that as the explicit "not yet calibrated" case)."""
        RigCalibration(
            stereo=calibration,
            body_T_camera_left=body_T_camera_left,
            camera_frame_id=FrameId.CAMERA_OPTICAL_LEFT,
        )
        return body_T_camera_left

    # ------------------------------------------------------------------
    def _build_threat_assessor(self) -> ThreatAssessor:
        config = self._config
        return ThreatAssessor(
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
    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        calibration: StereoCalibration,
        rectify: bool = True,
        body_T_camera_left: Optional[RigidTransform] = None,
    ) -> "DepthPerceptionPipeline":
        """Alternate constructor, identical to DepthPerceptionPipeline(config,
        calibration, rectify, body_T_camera_left) — provided for symmetry
        with other engine-style APIs; the plain constructor remains
        equally valid."""
        return cls(config, calibration, rectify=rectify, body_T_camera_left=body_T_camera_left)

    # ------------------------------------------------------------------
    def process(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_timestamp: Optional[float] = None,
        right_timestamp: Optional[float] = None,
    ) -> DepthPerceptionResult:
        """Run one stereo pair through the full pipeline.

        Args:
            left_image, right_image: NumPy stereo pair (BGR or grayscale),
                already split — this does not split a combined frame (see
                stereo.FrameSplitter if the caller still needs that).
            left_timestamp, right_timestamp: Optional, opaque caller-defined
                floats, carried through unmodified onto the returned
                result's `timestamp` field (left_timestamp wins if both are
                given). This library performs no synchronization or skew
                checking on them — purely a pass-through convenience so a
                caller doesn't have to track timestamps out-of-band.

        Returns:
            A DepthPerceptionResult.

        Raises:
            RuntimeError: If called after close().
            ValueError, RuntimeError: Propagated unchanged from
                RectificationEngine.rectify() (when rectify=True) on a
                rectification failure — see the comment at that call site
                for why this is not caught and silently degraded here.
        """
        if self._closed:
            raise RuntimeError(
                "DepthPerceptionPipeline.process() called after close() — "
                "construct a new pipeline instead of reusing a closed one."
            )
        require_matching_stereo_pair(left_image, right_image)
        t0 = time.perf_counter()
        # Derived early (pure function of the two args above, no side
        # effects) so the geometry stage below can pass it straight
        # through to PointCloud.timestamp — previously only derived right
        # before build_result(), further down.
        result_timestamp = left_timestamp if left_timestamp is not None else right_timestamp

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

        # Level 3, Phase E3: camera-optical-frame 3D geometry, gated by
        # PipelineConfig.enable_geometry (self._point_cloud_builder is
        # None when disabled — see __init__). Fed from raw_disparity, the
        # exact same array depth_map above was just computed from — not
        # a second disparity/depth computation, and not a second
        # reprojection implementation: PointCloudBuilder.build() calls
        # the same E2-verified DepthEstimator.estimate_point_cloud() this
        # pipeline's own DepthEstimator instance's math was validated
        # against (see tests/test_depth_estimator.py::TestEstimatePointCloud).
        # Deliberately NOT wrapped in try/except: a genuine failure here
        # must invalidate the whole frame exactly like a rectification
        # failure does above, not be silently swallowed into a fake empty
        # PointCloud — see docs/LEVEL3_ARCHITECTURE.md's failure-semantics
        # note and tests/test_pipeline_geometry.py::TestFailureSemantics.
        geometry = None
        geometry_body = None
        if self._point_cloud_builder is not None:
            geom_t0 = time.perf_counter()
            geometry = self._point_cloud_builder.build(raw_disparity, timestamp=result_timestamp)
            logger.debug(
                "Geometry stage: %.2f ms, valid points: %d / %d",
                (time.perf_counter() - geom_t0) * 1000.0,
                int(geometry.valid_mask.sum()),
                geometry.valid_mask.size,
            )

            # Level 3, Phase E4: body-frame transform, additive on top of
            # the E3 camera-frame cloud above (never replaces it). Only
            # runs when a camera-to-body extrinsic was actually supplied
            # at construction — self._body_T_camera_left is None both
            # when enable_geometry=False (irrelevant, no camera cloud to
            # transform) and when it's True but no extrinsic is
            # configured yet; in the latter case geometry_body correctly
            # stays None rather than silently assuming identity — see
            # calibration.contracts.RigCalibration's docstring and
            # __init__'s body_T_camera_left docstring above. Deliberately
            # NOT wrapped in try/except, same reasoning as the geometry
            # stage immediately above: a genuine failure here must
            # invalidate the whole frame, not be silently swallowed.
            if self._body_T_camera_left is not None:
                body_t0 = time.perf_counter()
                geometry_body = transform_point_cloud(geometry, self._body_T_camera_left)
                logger.debug(
                    "Body-frame transform stage: %.2f ms", (time.perf_counter() - body_t0) * 1000.0,
                )

        # Level 3, Phase E5: structured spatial evidence (ObstacleCloud /
        # FreeSpaceRays / GeometryMetrics), derived only from geometry_body
        # above — no disparity/depth recomputation, no reprojection, no
        # second camera->body transform (origin below is read directly off
        # the already-validated self._body_T_camera_left.translation, the
        # camera's own position in body coordinates — not recomputed).
        # Only runs when geometry_body actually exists; there is nothing
        # for either builder to derive evidence from otherwise, and this
        # library never fabricates body-frame evidence — see
        # docs/COORDINATE_FRAMES.md. Each of the two spatial-evidence
        # builders is independently gated by its own config flag;
        # geometry_metrics itself is populated whenever geometry_body
        # exists, regardless of those two flags (cheap aggregation, no
        # separate gate needed — see build_geometry_metrics's docstring).
        # Deliberately NOT wrapped in try/except, same reasoning as the
        # E3/E4 stages above.
        obstacle_cloud = None
        free_space_rays = None
        geometry_metrics = None
        if geometry_body is not None:
            origin = self._body_T_camera_left.translation

            if self._config.enable_obstacle_geometry:
                obstacle_t0 = time.perf_counter()
                obstacle_cloud = build_obstacle_cloud(
                    geometry_body, origin,
                    min_range_m=self._config.obstacle_min_range_m,
                    max_range_m=self._config.obstacle_max_range_m,
                    stride=self._config.geometry_sampling_stride,
                )
                logger.debug(
                    "Obstacle-cloud stage: %.2f ms, %d points",
                    (time.perf_counter() - obstacle_t0) * 1000.0, obstacle_cloud.points.shape[0],
                )

            if self._config.enable_free_space_rays:
                rays_t0 = time.perf_counter()
                free_space_rays = build_free_space_rays(
                    geometry_body, origin, stride=self._config.geometry_sampling_stride,
                )
                logger.debug(
                    "Free-space-rays stage: %.2f ms, %d rays",
                    (time.perf_counter() - rays_t0) * 1000.0, free_space_rays.ranges_m.shape[0],
                )

            metrics_t0 = time.perf_counter()
            geometry_metrics = build_geometry_metrics(geometry_body, obstacle_cloud, free_space_rays)
            logger.debug(
                "Geometry-metrics stage: %.2f ms", (time.perf_counter() - metrics_t0) * 1000.0,
            )

        regions = self._scene_interpreter.analyze(gray, raw_disparity, depth_map)
        decision = self._scene_interpreter.decide_navigation(regions)
        traversability = TraversabilityResult(regions=regions, decision=decision)

        obstacles = to_obstacle_assessment(
            self._threat_assessor.assess(depth_map, raw_disparity)
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result = build_result(
            raw_disparity, depth_map, traversability, obstacles, elapsed_ms,
            timestamp=result_timestamp,
            geometry=geometry,
            geometry_body=geometry_body,
            obstacle_cloud=obstacle_cloud,
            free_space_rays=free_space_rays,
            geometry_metrics=geometry_metrics,
        )

        self._frames_processed += 1
        self._last_confidence = result.confidence
        self._last_processing_time_ms = result.processing_time_ms

        return result

    # ------------------------------------------------------------------
    def process_observation(self, observation: StereoObservation) -> DepthPerceptionResult:
        """Convenience wrapper: unpack a StereoObservation and call process().

        Equivalent to
        ``process(observation.left_image, observation.right_image,
        observation.left_timestamp, observation.right_timestamp)`` — provided
        for callers that prefer passing one value instead of four.
        """
        return self.process(
            observation.left_image,
            observation.right_image,
            left_timestamp=observation.left_timestamp,
            right_timestamp=observation.right_timestamp,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all cross-frame state and start over.

        Only ThreatAssessor carries cross-frame state (per-beam EMA/debounce
        — see __init__) — this rebuilds it fresh from the same config, so a
        long stereo dropout or a scene cut doesn't leave stale smoothed
        readings influencing the next several frames. Does not affect
        calibration, config, or rectification maps. Raises RuntimeError if
        called after close(), matching process()'s own post-close behavior.
        """
        if self._closed:
            raise RuntimeError("DepthPerceptionPipeline.reset() called after close().")
        self._threat_assessor = self._build_threat_assessor()
        self._frames_processed = 0
        self._last_confidence = None
        self._last_processing_time_ms = None

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Mark this pipeline as no longer usable.

        No hardware/file handles are held by this pipeline today (rectification
        maps and the SGBM matcher are plain in-memory OpenCV objects), so this
        is currently a pure state transition — but it establishes a real
        lifecycle contract: process()/reset() raise RuntimeError afterward,
        rather than that being undefined behavior. Idempotent — closing an
        already-closed pipeline is a no-op.
        """
        self._closed = True

    # ------------------------------------------------------------------
    def health(self) -> PipelineHealth:
        """Return a snapshot of this pipeline's own lifecycle state.

        Not a per-frame diagnosis — see DepthPerceptionResult for that.
        last_confidence/last_processing_time_ms are None until process() has
        been called at least once (or again after reset()).
        """
        return PipelineHealth(
            is_closed=self._closed,
            frames_processed=self._frames_processed,
            last_confidence=self._last_confidence,
            last_processing_time_ms=self._last_processing_time_ms,
        )

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
