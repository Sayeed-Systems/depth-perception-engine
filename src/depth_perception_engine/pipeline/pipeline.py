"""
DepthPerceptionPipeline — the primary, stateful entry point.

Usage:

    from depth_perception_engine.pipeline import DepthPerceptionPipeline
    from depth_perception_engine.calibration import load_stereo_calibration
    from depth_perception_engine.config import PipelineConfig

    calibration = load_stereo_calibration("/path/to/stereo_calibration.xml")
    pipeline = DepthPerceptionPipeline(PipelineConfig(), calibration)

    result = pipeline.process(left_image, right_image)   # per frame

DUAL-INTERFACE ARCHITECTURE (docs/DUAL_INTERFACE_ARCHITECTURE.md). This
class IS the DPE core engine, and process_observation() is its single
geometry implementation — every supported path into DPE funnels into that
one method:

    process(left, right, ...)          adapts loose arguments -> observation
    process_geometry_frame(obs)        core/embedded entry -> GeometryFrame
    standalone.StandaloneStereoInterface   adapts raw/convenient sensor input

There is no second geometry pipeline and no standalone/embedded mode flag
in this class. A consumer embedding DPE constructs this class (or imports
depth_perception_engine.core) and calls process_geometry_frame(); it never
imports depth_perception_engine.standalone, so that layer is absent from
its execution path rather than switched off inside it.

Every engine (DisparityEngine, RectificationEngine, ThreatAssessor, and —
Level 3, Phase E3, when PipelineConfig.enable_geometry is True —
PointCloudBuilder; Level 4, Phase E2, when PipelineConfig.enable_temporal
is True — temporal.TemporalHistory) is built ONCE in __init__ and reused
across process() calls — mirroring the
original standalone main.py's build_pipeline(), and critically preserving
obstacles.ThreatAssessor's per-beam EMA smoothing and status debouncing
across frames. This is the entry point mp01_perception's
perception_processor.py should hold one instance of (constructed once,
e.g. in its own __init__) and call .process() on every frame — see
docs/INTEGRATION_READINESS.md.
"""

import dataclasses
import logging
import time
from typing import Optional, Sequence

import cv2
import numpy as np

from depth_perception_engine.calibration.contracts import RigCalibration
from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.config.pipeline_config import PipelineConfig
from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.fusion.result_builder import (
    aggregate_confidence,
    build_geometry_frame,
    build_result,
    to_obstacle_assessment,
)
from depth_perception_engine.geometry.boundary import build_boundary_evidence
from depth_perception_engine.geometry.free_space import build_free_space_rays
from depth_perception_engine.geometry.geometry_metrics import build_geometry_metrics, classify_geometry_quality
from depth_perception_engine.geometry.obstacle_extractor import build_obstacle_cloud
from depth_perception_engine.geometry.opening import build_opening_evidence
from depth_perception_engine.geometry.point_cloud_builder import PointCloudBuilder
from depth_perception_engine.geometry.provider import GeometryFrame
from depth_perception_engine.geometry.reliability import compute_shadow_zone_mask, compute_ramp_zone_mask
from depth_perception_engine.geometry.rigid_transform import transform_point_cloud
from depth_perception_engine.geometry.surface import build_surface_evidence
from depth_perception_engine.models.result import (
    DepthPerceptionResult,
    PipelineHealth,
    StereoObservation,
    TraversabilityResult,
)
from depth_perception_engine.obstacles.threat_assessment import ThreatAssessor
from depth_perception_engine.stereo.disparity_engine import DisparityEngine
from depth_perception_engine.stereo.rectification import RectificationEngine
from depth_perception_engine.temporal.consistency import compute_temporal_consistency
from depth_perception_engine.temporal.history import TemporalAdmissionStatus, TemporalHistory
from depth_perception_engine.temporal.persistence import TemporalPersistenceTracker
from depth_perception_engine.temporal.reliability import compute_motion_aware_reliability
from depth_perception_engine.temporal.rotation_compensation import compute_rotation_compensation
from depth_perception_engine.temporal.stabilization import compute_temporal_stabilization
from depth_perception_engine.temporal.types import MotionHint, TemporalRecord
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
        # Level 4, Phase E2: built once (same discipline as every other
        # engine here), but only if temporal history is actually enabled —
        # the disabled default must not pay even construction cost, and
        # must accumulate zero temporal state. See
        # docs/E2_TEMPORAL_HISTORY_PLAN.md.
        self._temporal_history: Optional[TemporalHistory] = (
            TemporalHistory(
                max_records=config.temporal_max_records,
                max_age_s=config.temporal_max_age_s,
                gap_limit_s=config.temporal_gap_limit_s,
            )
            if config.enable_temporal
            else None
        )
        # Level 4, Phase E7: bounded per-cell persistence tracker, built
        # once (same discipline as every other engine here), but only
        # when persistence classification is actually enabled — nested
        # underneath enable_temporal AND enable_motion_aware_reliability
        # (E7 requires E6's own reliability verdict to gate whether a
        # frame may create/reinforce persistence — see
        # docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md's Rule 7). Fixed, bounded
        # per-cell state (four arrays, shape fixed on first update() call,
        # never growing with frame count) — not a second, unbounded
        # tracker and not a store of DepthPerceptionResult objects.
        self._temporal_persistence_tracker: Optional[TemporalPersistenceTracker] = (
            TemporalPersistenceTracker(
                min_support_count=config.persistence_min_support_count,
                max_dropout_frames=config.persistence_max_dropout_frames,
                expiration_absence_frames=config.persistence_expiration_absence_frames,
                agreement_tolerance_m=config.temporal_consistency_agreement_tolerance_m,
            )
            if (config.enable_temporal and config.enable_motion_aware_reliability and config.enable_temporal_persistence)
            else None
        )
        # Level 4, Phase E5: rectified intrinsics for prior-geometry
        # reprojection, derived once from the pipeline's own Q matrix —
        # the exact same f/cx/cy derivation depth.DepthEstimator/
        # calibration.contracts.StereoExtrinsics already use, not a new
        # intrinsics representation. Cheap; computed unconditionally
        # (not gated by enable_rotation_compensation) since it's just a
        # few scalar reads, no meaningful construction cost — mirrors
        # config/calibration properties already being unconditional.
        Q = calibration.Q
        self._rectified_focal_length_px: float = abs(float(Q[2, 3]))
        self._rectified_principal_point_px = (-float(Q[0, 3]), -float(Q[1, 3]))

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
            contamination_threshold=config.clearance_shadow_zone_contamination_threshold,
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
        motion_hint: Optional[MotionHint] = None,
        motion_hints: Optional[Sequence[MotionHint]] = None,
        observation_id: Optional[str] = None,
    ) -> DepthPerceptionResult:
        """Run one stereo pair through the full pipeline.

        STANDALONE-CONVENIENCE SIGNATURE. This method owns no geometry
        processing of its own: it adapts these loose arguments into the
        canonical core input contract (models.StereoObservation) and
        delegates to process_observation(), which is the single
        implementation both DPE interfaces converge on — see
        docs/DUAL_INTERFACE_ARCHITECTURE.md. Every argument below is
        forwarded onto the equivalent StereoObservation field BY
        REFERENCE; no image is copied, reshaped, or reconstructed here,
        so this remains exactly as cheap as it was before the
        dual-interface separation. Kept as a fully supported entry point
        (mp01_perception, this repository's own examples/benchmarks and
        the whole existing test suite call it) — an embedded consumer
        that already holds a StereoObservation should call
        process_observation()/process_geometry_frame() directly instead.

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
            observation_id: Phase D2. Optional opaque, caller-owned
                observation/transaction identity for this capture. New,
                additive, trailing, defaulted parameter — every existing
                positional or keyword call site is unaffected. Forwarded
                verbatim onto StereoObservation.observation_id and copied
                unchanged onto the resulting GeometryFrame; DPE never
                interprets it and no algorithm branches on it. NOT a
                coordinate frame — see models.StereoObservation and
                docs/D2_OBSERVATION_IDENTITY_CONTRACT.md.
            motion_hint: Level 4, Phase E2. Optional temporal.MotionHint
                associated with this frame — new, additive, defaulted
                parameter; every existing call site is unaffected. Has no
                effect on any Level 3 output whatsoever. Only consumed
                when config.enable_temporal is True, in which case it is
                attached unread to this frame's temporal.TemporalRecord
                (no integration, alignment, or validation of its contents
                beyond MotionHint.__post_init__'s own checks) — see
                docs/E2_TEMPORAL_HISTORY_PLAN.md. A missing motion hint
                (None, the default) is always legal and never blocks
                temporal-history admission or Level 3 perception.
            motion_hints: Level 4, Phase E5. Optional, bounded sequence of
                temporal.MotionHint samples spanning the interval leading
                up to this frame — distinct from the singular `motion_hint`
                above, which remains for TemporalRecord association only.
                Only consumed when config.enable_temporal and
                config.enable_rotation_compensation are both True, in
                which case admissible samples (see
                docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md's Decision 3) are
                integrated into a short-window relative rotation used to
                re-express the previous comparable frame's geometry before
                temporal_consistency/temporal_stabilization are computed.
                A missing/empty/insufficient sequence (None, the default)
                is always legal and never blocks temporal-history
                admission, consistency, stabilization, or Level 3
                perception — E5 falls back to exactly pre-E5 behavior.

        Returns:
            A DepthPerceptionResult.

        Raises:
            RuntimeError: If called after close().
            ValueError, RuntimeError: Propagated unchanged from
                RectificationEngine.rectify() (when rectify=True) on a
                rectification failure — see the comment at that call site
                for why this is not caught and silently degraded here.
        """
        return self.process_observation(
            StereoObservation(
                left_image=left_image,
                right_image=right_image,
                left_timestamp=left_timestamp,
                right_timestamp=right_timestamp,
                motion_hint=motion_hint,
                motion_hints=motion_hints,
                observation_id=observation_id,
            )
        )

    # ------------------------------------------------------------------
    def process_observation(self, observation: StereoObservation) -> DepthPerceptionResult:
        """CORE ENTRY POINT — run one canonical StereoObservation through
        the full geometry pipeline.

        This is DPE's single geometry-processing implementation. Every
        supported path into DPE ends here:

            standalone.StandaloneStereoInterface.process*()
                -> process()/process_observation()  -> THIS METHOD
            an embedded consumer (e.g. a future hybrid_perception_engine)
                -> process_observation()/process_geometry_frame()
                                                    -> THIS METHOD

        There is deliberately no second, interface-specific geometry
        implementation and no `standalone`/`embedded` mode flag anywhere
        in this class — the two interfaces differ only in how a valid
        StereoObservation is produced, never in what happens to it
        afterwards. See docs/DUAL_INTERFACE_ARCHITECTURE.md.

        `observation.left_image`/`.right_image` are used by reference —
        never copied here. `observation.calibration` remains reserved and
        unread (see models.StereoObservation's own docstring): the
        calibration this pipeline was constructed with is always the one
        used. `observation.observation_id` (Phase D2, and its deprecated
        `frame_id` alias) IS now read — but only to be copied verbatim
        onto the resulting GeometryFrame as opaque provenance; no
        geometry, temporal, or admission decision anywhere in this method
        depends on its value. Every other field maps one-to-one onto
        process()'s own identically named parameter, with identical
        semantics — see process()'s docstring for what each one does.

        Returns:
            A DepthPerceptionResult (whose `geometry_frame` field is
            populated only when PipelineConfig.enable_geometry_frame is
            True). An embedded consumer that wants the authoritative
            GeometryFrame contract directly should call
            process_geometry_frame() instead.

        Raises:
            RuntimeError: If called after close().
            ValueError, RuntimeError: Propagated unchanged from
                RectificationEngine.rectify() (when rectify=True) on a
                rectification failure — see the comment at that call site
                for why this is not caught and silently degraded here.
        """
        left_image = observation.left_image
        right_image = observation.right_image
        left_timestamp = observation.left_timestamp
        right_timestamp = observation.right_timestamp
        motion_hint = observation.motion_hint
        motion_hints = observation.motion_hints
        # Phase D2: THE one place DPE reads observation identity. Read
        # verbatim through StereoObservation's own resolver (which already
        # settled observation_id vs the deprecated frame_id alias) and
        # carried untouched to build_result() below. Nothing between here
        # and there inspects, parses, or branches on it — in particular no
        # temporal stage sees it, so chronology keeps keying on timestamp
        # exactly as before D2.
        observation_id = observation.resolved_observation_id
        if self._closed:
            raise RuntimeError(
                "DepthPerceptionPipeline.process_observation() called after close() — "
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

        # Phase I3: local occlusion/dis-occlusion reliability mask — see
        # geometry.reliability's module docstring. Computed once, from
        # raw_disparity/valid_disparity_mask alone (before any Level 3/4
        # stage runs), and threaded into the specific builders below
        # whose own per-cell "support"/"confirmed" semantics a genuine
        # occlusion shadow was found (benchmarks/i3_occlusion_safety/) to
        # defeat — never used to redefine valid_disparity_mask/
        # valid_depth_mask/PointCloud.valid_mask themselves, so Level 0-2
        # and the E2-E5 chain's own existing tested invariants are
        # unaffected. All-False (zero cost, zero effect) on any frame
        # with no large disparity discontinuity anywhere.
        shadow_zone_mask = None
        if self._config.geometry_shadow_zone_enabled:
            shadow_zone_mask = compute_shadow_zone_mask(
                raw_disparity, raw_disparity > 0.0,
                lookahead_px=self._config.geometry_shadow_zone_lookahead_px,
                gradient_threshold_px=self._config.geometry_shadow_zone_gradient_threshold_px,
                max_width_px=self._config.geometry_shadow_zone_max_width_px,
            )

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
        surface_evidence = None
        if geometry_body is not None:
            origin = self._body_T_camera_left.translation

            if self._config.enable_obstacle_geometry:
                obstacle_t0 = time.perf_counter()
                obstacle_cloud = build_obstacle_cloud(
                    geometry_body, origin,
                    min_range_m=self._config.obstacle_min_range_m,
                    max_range_m=self._config.obstacle_max_range_m,
                    stride=self._config.geometry_sampling_stride,
                    reliability_mask=shadow_zone_mask,
                )
                logger.debug(
                    "Obstacle-cloud stage: %.2f ms, %d points",
                    (time.perf_counter() - obstacle_t0) * 1000.0, obstacle_cloud.points.shape[0],
                )

            if self._config.enable_free_space_rays:
                rays_t0 = time.perf_counter()
                free_space_rays = build_free_space_rays(
                    geometry_body, origin, stride=self._config.geometry_sampling_stride,
                    reliability_mask=shadow_zone_mask,
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

            # Phase D4: deterministic, bounded local surface-normal
            # estimation — reuses this SAME geometry_body cloud and origin
            # (no second body-frame transform) — see
            # docs/DPE_V1_PROVIDER_CONTRACT.md's D4 record and
            # geometry.surface's own module docstring for the full
            # frame/normalization/orientation/invalid-support contract.
            if self._config.enable_surface_geometry:
                surface_t0 = time.perf_counter()
                surface_evidence = build_surface_evidence(
                    geometry_body, origin,
                    grid_rows=self._config.surface_grid_rows,
                    grid_cols=self._config.surface_grid_cols,
                    min_support_count=self._config.surface_min_support_count,
                    reliability_mask=shadow_zone_mask,
                )
                logger.debug(
                    "Surface-geometry stage: %.2f ms, %d cells",
                    (time.perf_counter() - surface_t0) * 1000.0, len(surface_evidence),
                )

        # Phase D5: deterministic, bounded depth/surface-orientation
        # discontinuity detection — deliberately OUTSIDE the
        # "geometry_body is not None" block above: its baseline signal
        # (depth discontinuity) needs only depth_map, a Level 0 output
        # present regardless of enable_geometry. surface_evidence (from
        # the block above — None if disabled or geometry_body absent) is
        # passed straight through and consulted only when its own grid
        # happens to match boundary_grid_rows/cols — see
        # docs/DPE_V1_PROVIDER_CONTRACT.md's D5 record and
        # geometry.boundary's own module docstring for the full
        # discontinuity-definition/invalid-vs-boundary/coordinate
        # contract.
        boundary_evidence = None
        if self._config.enable_boundary_geometry:
            boundary_t0 = time.perf_counter()
            boundary_evidence = build_boundary_evidence(
                depth_map, FrameId.CAMERA_OPTICAL_LEFT,
                grid_rows=self._config.boundary_grid_rows,
                grid_cols=self._config.boundary_grid_cols,
                min_support_count=self._config.boundary_min_support_count,
                depth_step_threshold_m=self._config.boundary_depth_step_threshold_m,
                orientation_change_threshold_rad=self._config.boundary_orientation_change_threshold_rad,
                surface_evidence=surface_evidence,
                surface_grid_rows=self._config.surface_grid_rows,
                surface_grid_cols=self._config.surface_grid_cols,
                reliability_mask=shadow_zone_mask,
                min_confirmation_support_fraction=self._config.boundary_min_confirmation_support_fraction,
            )
            logger.debug(
                "Boundary-geometry stage: %.2f ms, %d adjacencies",
                (time.perf_counter() - boundary_t0) * 1000.0, len(boundary_evidence),
            )

        # Phase D6: deterministic opening/passage-structure inference —
        # built entirely from this SAME frame's already-computed
        # boundary_evidence (never reclassified) plus depth_map (only to
        # recover per-cell absolute depth, which BoundaryEvidence's own
        # unsigned depth_step_m cannot give). Always uses boundary's own
        # grid/support-threshold configuration — no independent
        # opening_grid_rows/cols exists. Nested underneath
        # enable_boundary_geometry: nothing to build from otherwise. See
        # docs/DPE_V1_PROVIDER_CONTRACT.md's D6 record and
        # geometry.opening's own module docstring for the full
        # admission-rule/invalid-vs-opening/coordinate contract.
        opening_evidence = None
        if self._config.enable_boundary_geometry and self._config.enable_opening_geometry:
            opening_t0 = time.perf_counter()
            opening_evidence = build_opening_evidence(
                boundary_evidence, depth_map, FrameId.CAMERA_OPTICAL_LEFT,
                grid_rows=self._config.boundary_grid_rows,
                grid_cols=self._config.boundary_grid_cols,
                min_support_count=self._config.boundary_min_support_count,
                min_range_ratio=self._config.opening_min_range_ratio,
                focal_length_px=self._rectified_focal_length_px,
                min_merge_depth_diff_m=self._config.opening_min_merge_depth_diff_m,
            )
            logger.debug(
                "Opening-geometry stage: %.2f ms, %d openings",
                (time.perf_counter() - opening_t0) * 1000.0, len(opening_evidence),
            )

        regions = self._scene_interpreter.analyze(gray, raw_disparity, depth_map)
        decision = self._scene_interpreter.decide_navigation(regions)
        traversability = TraversabilityResult(regions=regions, decision=decision)

        # Phase I6.3: clearance contamination gating combines TWO
        # independent reliability signals — I6.2's shadow_zone_mask
        # (narrow, geometrically-predicted occlusion strips) and the
        # wide-ramp mask below (StereoSGBM's own smoothness-
        # regularization radius, a distinct, wider, direction-agnostic
        # contamination mechanism; see geometry.reliability.
        # compute_ramp_zone_mask's own docstring for the full root-cause
        # trace). Unioned into one mask, not two separate checks, so
        # ThreatAssessor.assess()'s contamination test stays a single
        # fraction-of-kept-population comparison. Neither signal is
        # threaded into obstacle_cloud/free_space_rays/surface/boundary —
        # only shadow_zone_mask is, unchanged from I3.
        clearance_reliability_mask = (
            shadow_zone_mask if self._config.clearance_shadow_zone_gating_enabled else None
        )
        if self._config.clearance_ramp_zone_gating_enabled:
            ramp_zone_mask = compute_ramp_zone_mask(
                raw_disparity, raw_disparity > 0.0,
                window_px=self._config.clearance_ramp_zone_window_px,
                gradient_threshold_px=self._config.clearance_ramp_zone_gradient_threshold_px,
            )
            clearance_reliability_mask = (
                ramp_zone_mask if clearance_reliability_mask is None
                else (clearance_reliability_mask | ramp_zone_mask)
            )

        obstacles = to_obstacle_assessment(
            self._threat_assessor.assess(
                depth_map, raw_disparity,
                reliability_mask=clearance_reliability_mask,
            )
        )

        # Level 4, Phase E2: bounded temporal-history admission, gated by
        # PipelineConfig.enable_temporal (self._temporal_history is None
        # when disabled — see __init__). Deliberately the LAST thing
        # computed before build_result(): every value it reads
        # (traversability, geometry_metrics, result_timestamp,
        # motion_hint) is already fully computed above, and admission
        # itself never feeds back into or alters any of Level 3's own
        # outputs — see docs/LEVEL4_ARCHITECTURE.md section 9's raw-vs-
        # temporal authority rule. classify_geometry_quality() here is
        # pure reuse of the already-frozen Level 3, Phase E6 classifier —
        # not a new algorithm, and not itself a temporal computation (it
        # never looks at history). temporal.TemporalHistory.admit() is
        # the ONLY place any timestamp-chronology decision is made; no
        # comparison logic is duplicated here.
        temporal_admission_status = None
        temporal_consistency = None
        temporal_stabilization = None
        rotation_compensation_status = None
        motion_aware_reliability = None
        temporal_persistence = None
        if self._temporal_history is not None:
            temporal_t0 = time.perf_counter()
            geometry_quality = None
            if geometry_metrics is not None:
                geometry_quality = classify_geometry_quality(
                    geometry_metrics,
                    healthy_min_valid_fraction=self._config.geometry_healthy_min_valid_fraction,
                    degraded_min_valid_fraction=self._config.geometry_degraded_min_valid_fraction,
                )
            # Level 4, Phase E3: decimated depth snapshot, the one piece
            # of data compute_temporal_consistency() needs and nothing
            # more (never the full-resolution depth_map, never a
            # PointCloud) — see docs/E3_IMPLEMENTATION_PLAN.md's
            # Decision 5. Reuses depth_map's own existing 0.0-is-invalid
            # convention, so no second validity array is needed.
            stride = self._config.temporal_consistency_sampling_stride
            depth_snapshot_m = depth_map[::stride, ::stride]
            # Captured BEFORE admit() — admission mutates the buffer
            # (appends this frame's own record, and may clear everything
            # on a gap restart), so this is genuinely "the previous
            # frame's record," not a reference to something that might
            # include the record we're about to add.
            previous_latest = self._temporal_history.latest
            record = TemporalRecord(
                timestamp=result_timestamp,
                confidence=aggregate_confidence(traversability),
                geometry_quality=geometry_quality,
                motion_hint=motion_hint,
                depth_snapshot_m=depth_snapshot_m,
            )
            temporal_admission_status = self._temporal_history.admit(record)
            logger.debug(
                "Temporal admission stage: %.2f ms, status=%s, history size=%d",
                (time.perf_counter() - temporal_t0) * 1000.0,
                temporal_admission_status,
                len(self._temporal_history),
            )
            # Level 4, Phase E7: a gap-triggered new sequence wipes
            # TemporalHistory's own chronology (see admit() above) — the
            # persistence tracker's per-cell state must start equally
            # fresh, matching TemporalHistory's own "complete amnesia"
            # behavior (docs/E2_TEMPORAL_HISTORY_PLAN.md's Decision 3/7)
            # rather than incorrectly carrying pre-gap support counts
            # into an unrelated new sequence.
            if (
                self._temporal_persistence_tracker is not None
                and temporal_admission_status == TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE
            ):
                self._temporal_persistence_tracker.clear()

            # Level 4, Phase E3: read-only consistency check against the
            # single most recent comparable prior record — see
            # docs/E3_IMPLEMENTATION_PLAN.md's Decision 2 trigger table.
            # Evaluated only when THIS frame's own timestamp was itself
            # admitted (temporal_admission_status is ACCEPTED or
            # ACCEPTED_NEW_SEQUENCE) — a chronology-rejected frame gets no
            # comparison at all (temporal_consistency stays None;
            # temporal_admission_status already explains why). On a
            # gap-triggered new sequence, `previous_latest` (captured
            # above, before admit() cleared the old history) is
            # deliberately NOT passed through — the pre-gap record must
            # never influence the new sequence's first consistency
            # judgement (docs/E2_TEMPORAL_HISTORY_PLAN.md's Decision 7).
            previous_for_comparison = (
                previous_latest if temporal_admission_status == TemporalAdmissionStatus.ACCEPTED else None
            )

            if temporal_admission_status in (
                TemporalAdmissionStatus.ACCEPTED, TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE,
            ):
                # Level 4, Phase E5: short-window rotational motion
                # compensation, strictly upstream of E3/E4
                # (docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md's Decision 5) —
                # substitutes previous_for_comparison with a rotation-
                # compensated copy (or leaves it unchanged on any
                # fallback condition) before compute_temporal_consistency
                # runs; neither that function nor
                # compute_temporal_stabilization is modified by this
                # stage. Gated independently of enable_temporal_
                # stabilization — E5 can usefully feed an E3-only
                # (consistency-only) consumer too.
                if self._config.enable_rotation_compensation:
                    rotation_t0 = time.perf_counter()
                    previous_timestamp_for_compensation = (
                        previous_for_comparison.timestamp if previous_for_comparison is not None else None
                    )
                    previous_for_comparison, rotation_compensation_status = compute_rotation_compensation(
                        previous_for_comparison, motion_hints,
                        previous_timestamp=previous_timestamp_for_compensation,
                        current_timestamp=result_timestamp,
                        stride=stride,
                        focal_length_px=self._rectified_focal_length_px,
                        principal_point_px=self._rectified_principal_point_px,
                    )
                    logger.debug(
                        "Rotation compensation stage: %.2f ms, status=%s",
                        (time.perf_counter() - rotation_t0) * 1000.0,
                        rotation_compensation_status,
                    )

                consistency_t0 = time.perf_counter()
                temporal_consistency = compute_temporal_consistency(
                    depth_snapshot_m, previous_for_comparison,
                    agreement_tolerance_m=self._config.temporal_consistency_agreement_tolerance_m,
                    min_agreement_fraction=self._config.temporal_consistency_min_agreement_fraction,
                )
                logger.debug(
                    "Temporal consistency stage: %.2f ms, state=%s",
                    (time.perf_counter() - consistency_t0) * 1000.0,
                    temporal_consistency.state,
                )

                # Level 4, Phase E4: deterministic, confidence-aware
                # stabilization, gated underneath enable_temporal_
                # stabilization (mirrors enable_obstacle_geometry/
                # enable_free_space_rays's own precedent of an
                # opt-in flag nested under a broader one — see
                # docs/E4_IMPLEMENTATION_PLAN.md's Decision 2/6).
                # Reuses depth_snapshot_m and previous_for_comparison
                # exactly as E3 computed them above — no second
                # decimation, no second "which prior record" decision.
                # Read-only of every Level 3 field: the function below
                # receives only two small arrays, one scalar
                # (previous_record.confidence), and this frame's own
                # already-computed TemporalConsistency — never a
                # mutable reference to depth_map/geometry/etc. — see
                # docs/E4_IMPLEMENTATION_PLAN.md's Decision 1/3.
                if self._config.enable_temporal_stabilization:
                    stabilization_t0 = time.perf_counter()
                    temporal_stabilization = compute_temporal_stabilization(
                        depth_snapshot_m, previous_for_comparison, temporal_consistency,
                        agreement_tolerance_m=self._config.temporal_consistency_agreement_tolerance_m,
                        min_history_confidence=self._config.temporal_stabilization_min_history_confidence,
                        min_comparable_fraction=self._config.temporal_stabilization_min_comparable_fraction,
                    )
                    logger.debug(
                        "Temporal stabilization stage: %.2f ms, state=%s",
                        (time.perf_counter() - stabilization_t0) * 1000.0,
                        temporal_stabilization.state,
                    )

            # Level 4, Phase E6: deterministic, explicit motion-aware
            # reliability assessment — deliberately OUTSIDE the narrower
            # "admitted this frame" block above (unlike E3/E4/E5), so a
            # chronology-rejected frame is still correctly classified
            # INSUFFICIENT_EVIDENCE rather than silently skipped — see
            # docs/LEVEL4_E6_IMPLEMENTATION_PLAN.md's "Where E6 runs"
            # section. Read-only of every value it reads: it never
            # mutates temporal_consistency/temporal_stabilization/
            # rotation_compensation_status, never calls
            # compensate_prior_geometry() (no additional motion
            # compensation), and never touches disparity_map/depth_map/
            # any geometry.* field. previous_for_comparison's own
            # timestamp (None on a rejected/first/gap-restart frame) is
            # exactly the "no comparison interval" signal
            # select_motion_hint_samples() already knows how to handle.
            if self._config.enable_motion_aware_reliability:
                reliability_t0 = time.perf_counter()
                previous_timestamp_for_reliability = (
                    previous_for_comparison.timestamp if previous_for_comparison is not None else None
                )
                temporal_consistency_state = temporal_consistency.state if temporal_consistency is not None else None
                temporal_stabilization_state = (
                    temporal_stabilization.state if temporal_stabilization is not None else None
                )
                motion_aware_reliability = compute_motion_aware_reliability(
                    temporal_consistency_state, temporal_stabilization_state, rotation_compensation_status,
                    motion_hints,
                    previous_timestamp=previous_timestamp_for_reliability,
                    current_timestamp=result_timestamp,
                    max_angular_motion_rad=self._config.reliability_max_angular_motion_rad,
                    min_motion_coverage_fraction=self._config.reliability_min_motion_coverage_fraction,
                )
                logger.debug(
                    "Motion-aware reliability stage: %.2f ms, state=%s",
                    (time.perf_counter() - reliability_t0) * 1000.0,
                    motion_aware_reliability.state,
                )

            # Level 4, Phase E7: deterministic, per-cell temporal-
            # persistence classification — the LAST temporal stage,
            # deliberately downstream of E6: it reads
            # motion_aware_reliability.state (an UNRELIABLE frame must
            # never create or reinforce persistence, Rule 7) and reuses
            # this frame's own already-computed depth_snapshot_m/
            # motion_hints exactly as E3/E5 already do — no second
            # decimation, no re-derivation of reliability. Requires ALL
            # of enable_temporal/enable_motion_aware_reliability/
            # enable_temporal_persistence (self._temporal_persistence_tracker
            # is None otherwise — see __init__), so this block is a pure
            # no-op (zero added cost) for every caller that has not
            # explicitly opted into all three. result_timestamp must be
            # a real value: with none, no chronology can be established
            # for this frame at all (mirrors TemporalHistory.admit()'s
            # own REJECTED_INVALID_TIMESTAMP outcome), so persistence is
            # simply not evaluated rather than guessing.
            if self._temporal_persistence_tracker is not None and result_timestamp is not None:
                persistence_t0 = time.perf_counter()
                motion_aware_reliability_state = (
                    motion_aware_reliability.state if motion_aware_reliability is not None else None
                )
                temporal_persistence = self._temporal_persistence_tracker.update(
                    depth_snapshot_m, result_timestamp,
                    motion_aware_reliability_state, motion_hints,
                    enable_rotation_compensation=self._config.enable_rotation_compensation,
                    stride=stride,
                    focal_length_px=self._rectified_focal_length_px,
                    principal_point_px=self._rectified_principal_point_px,
                )
                logger.debug(
                    "Temporal persistence stage: %.2f ms, state=%s, new=%d, persistent=%d, disappearing=%d",
                    (time.perf_counter() - persistence_t0) * 1000.0,
                    temporal_persistence.state,
                    temporal_persistence.new_count,
                    temporal_persistence.persistent_count,
                    temporal_persistence.disappearing_count,
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
            temporal_admission_status=temporal_admission_status,
            temporal_consistency=temporal_consistency,
            temporal_stabilization=temporal_stabilization,
            rotation_compensation_status=rotation_compensation_status,
            motion_aware_reliability=motion_aware_reliability,
            temporal_persistence=temporal_persistence,
            surface_evidence=surface_evidence,
            boundary_evidence=boundary_evidence,
            opening_evidence=opening_evidence,
            observation_id=observation_id,
        )

        # Phase D2: GeometryFrame provider contract, gated by
        # enable_geometry_frame (default False — every existing caller is
        # unaffected). Deliberately the LAST thing done before returning:
        # build_geometry_frame() only reads fields off `result`, the
        # DepthPerceptionResult just built above — no recomputation, no
        # new algorithm — see docs/DPE_V1_PROVIDER_CONTRACT.md. Phase D7:
        # focal_length_px/principal_point_px[0] are threaded through for
        # ClearanceEvidence's own bearing computation — the same
        # calibration-derived values already used for rotation
        # compensation/surface/boundary/opening geometry, not recomputed.
        # Phase D8: geometry_healthy_min_valid_fraction/
        # geometry_degraded_min_valid_fraction are the SAME existing
        # thresholds classify_geometry_quality() already used before D8 —
        # no new threshold — for GeometryFrame.quality's own
        # geometry_validity_state dimension.
        if self._config.enable_geometry_frame:
            result = dataclasses.replace(result, geometry_frame=self._build_geometry_frame(result))

        self._frames_processed += 1
        self._last_confidence = result.confidence
        self._last_processing_time_ms = result.processing_time_ms

        return result

    # ------------------------------------------------------------------
    def process_geometry_frame(self, observation: StereoObservation) -> GeometryFrame:
        """CORE ENTRY POINT — run one canonical StereoObservation and
        return the authoritative GeometryFrame contract directly.

        This is the method an embedded consumer (a future
        hybrid_perception_engine) is expected to call:

            geometry = pipeline.process_geometry_frame(observation)

        It is a thin composition of process_observation() (the single
        geometry implementation — called exactly once here, nothing is
        recomputed) and fusion.result_builder.build_geometry_frame() (the
        same builder, called with the same arguments, that
        process_observation() itself uses when
        PipelineConfig.enable_geometry_frame is True — see
        _build_geometry_frame below). No geometry algorithm, threshold, or
        calibration value is duplicated, re-derived, or reinterpreted for
        this path.

        Unlike DepthPerceptionResult.geometry_frame, the returned frame is
        never None: calling this method IS the caller's explicit request
        for a GeometryFrame, so it is built whether or not
        PipelineConfig.enable_geometry_frame happens to be set. That flag
        keeps its exact pre-existing meaning — it gates only whether
        DepthPerceptionResult.geometry_frame is populated on the legacy
        result object — and setting it merely means the frame this method
        returns was already built during process_observation() rather than
        immediately afterwards. Both branches call the identical builder
        with identical arguments and are therefore field-for-field
        identical; see tests/test_dual_interface_architecture.py.

        Raises:
            RuntimeError: If called after close() (propagated unchanged
                from process_observation()).
        """
        result = self.process_observation(observation)
        if result.geometry_frame is not None:
            return result.geometry_frame
        return self._build_geometry_frame(result)

    # ------------------------------------------------------------------
    def _build_geometry_frame(self, result: DepthPerceptionResult) -> GeometryFrame:
        """The one place GeometryFrame is constructed from a completed
        DepthPerceptionResult — shared verbatim by process_observation()'s
        own enable_geometry_frame branch and by process_geometry_frame()
        above, so the two can never drift apart.

        Phase D2/D7/D8 semantics are unchanged by the dual-interface
        refactor: build_geometry_frame() only reads fields off `result` —
        no recomputation, no new algorithm — and focal_length_px/
        principal_point_x_px/the two geometry_*_min_valid_fraction
        thresholds are the same already-computed, calibration-derived
        values this pipeline has passed since D7/D8.
        """
        return build_geometry_frame(
            result,
            focal_length_px=self._rectified_focal_length_px,
            principal_point_x_px=self._rectified_principal_point_px[0],
            clearance_min_coverage_fraction=self._config.clearance_min_coverage_fraction,
            geometry_healthy_min_valid_fraction=self._config.geometry_healthy_min_valid_fraction,
            geometry_degraded_min_valid_fraction=self._config.geometry_degraded_min_valid_fraction,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all cross-frame state and start over.

        ThreatAssessor carries cross-frame state (per-beam EMA/debounce —
        see __init__) — this rebuilds it fresh from the same config, so a
        long stereo dropout or a scene cut doesn't leave stale smoothed
        readings influencing the next several frames. Level 4, Phase E2:
        if temporal history is enabled, this also clears it completely
        (temporal.TemporalHistory.clear()) — reset() provides complete
        temporal amnesia, per docs/E2_TEMPORAL_HISTORY_PLAN.md's Decision
        3/11: the next accepted frame after reset() behaves as the first
        frame of a brand-new temporal sequence, with zero historical
        influence. Level 4, Phase E7: if persistence classification is
        enabled, this also clears the persistence tracker completely
        (temporal.persistence.TemporalPersistenceTracker.clear()) — every
        cell's support count/absence streak/first-observed timestamp is
        discarded, so the next frame after reset() starts as a brand-new
        persistence sequence too (Rule: reset clears persistence state).
        Does not affect calibration, config, or rectification maps.
        Raises RuntimeError if called after close(), matching process()'s
        own post-close behavior.
        """
        if self._closed:
            raise RuntimeError("DepthPerceptionPipeline.reset() called after close().")
        self._threat_assessor = self._build_threat_assessor()
        if self._temporal_history is not None:
            self._temporal_history.clear()
        if self._temporal_persistence_tracker is not None:
            self._temporal_persistence_tracker.clear()
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

    @property
    def temporal_history(self) -> Optional[TemporalHistory]:
        """Level 4, Phase E2. The pipeline's own temporal.TemporalHistory
        instance, or None if PipelineConfig.enable_temporal is False (the
        default) — mirrors .config/.calibration's read-only exposure
        pattern. Query its .records/.latest/__len__ to inspect current
        chronology; admission/clearing happen only via process()/reset()."""
        return self._temporal_history

    def __repr__(self) -> str:
        return (
            f"DepthPerceptionPipeline(image_size={self._calibration.image_size}, "
            f"rectify={self._rectify}, n_beams={self._config.n_beams}, "
            f"grid={self._config.traversability_grid_rows}x"
            f"{self._config.traversability_grid_cols})"
        )
