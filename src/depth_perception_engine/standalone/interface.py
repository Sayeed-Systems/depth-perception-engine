"""
StandaloneStereoInterface — DPE's STANDALONE / SENSOR-FACING convenience
interface.

Purpose (and the whole of it): make `depth_perception_engine` independently
runnable — for development, unit tests, benchmarks, datasets, physical
stereo/motion qualification, and debugging — by accepting the raw and
convenient inputs a development caller actually has, and ADAPTING them into
DPE's one canonical core input contract (`models.StereoObservation`).

    raw / convenient stereo + calibration + optional motion samples
                              |
                              v
                  StandaloneStereoInterface       <-- input adaptation only
                              |
                              v
        DepthPerceptionPipeline.process_observation()   <-- the ONE core
                              |
                              v
                        GeometryFrame

This class owns NO geometry. It contains no disparity, depth, rectification,
point-cloud, surface, boundary, opening, clearance, temporal, reliability,
or quality logic, and no calibration mathematics — every one of those stays
in `DepthPerceptionPipeline` and the algorithm modules underneath it. What
this class owns is exactly the four sensor-facing conveniences the core
must not own:

  1. Calibration FILE loading (`from_calibration_file`) — the core takes a
     `StereoCalibration` object and never touches a path.
  2. Combined side-by-side frame splitting (`split_combined_frame`, via the
     existing `stereo.FrameSplitter`) — many development cameras deliver one
     joined frame; the core is given an already-split pair.
  3. Raw motion-sample normalization (`build_motion_hint`/`build_motion_hints`)
     — plain `(timestamp, wx, wy, wz)` tuples become validated
     `temporal.MotionHint` values. No motion MATHEMATICS happens here: the
     integration/compensation of those hints remains entirely inside the
     core's own E5/E6/E7 stages, unchanged.
  4. Assembling all of the above into a `StereoObservation`
     (`build_observation`).

NOT a sensor driver. This class never opens a camera or device, never
subscribes to a topic, never synchronizes streams, never knows physical
versus simulated sources, and never imports ROS — it adapts values a caller
already holds. `examples/live_demo.py` remains where camera I/O lives.

STRUCTURAL SEPARATION, NOT A MODE FLAG. There is no `standalone_mode`,
`hpe_mode`, or `sensor_interface_enabled` boolean anywhere in DPE. When DPE
is embedded in a larger perception system, that system constructs
`DepthPerceptionPipeline` (or imports `depth_perception_engine.core`)
directly and simply never imports this module — the standalone layer is
absent from that execution path rather than switched off inside it.

ZERO EXTRA COPIES. Images are passed to the core BY REFERENCE. Splitting a
combined frame returns NumPy views (plain slicing), not copies. Nothing here
calls `.copy()`, reshapes, or reconstructs an array.
"""

import logging
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from depth_perception_engine.calibration.loader import load_stereo_calibration
from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.config.pipeline_config import PipelineConfig
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.provider import GeometryFrame
from depth_perception_engine.models.result import DepthPerceptionResult, PipelineHealth, StereoObservation
from depth_perception_engine.pipeline.pipeline import DepthPerceptionPipeline
from depth_perception_engine.stereo.frame_splitter import FrameSplitter
from depth_perception_engine.temporal.types import MotionHint

logger = logging.getLogger(__name__)

#: One raw angular-rate sample as a development caller usually has it:
#: either an already-built MotionHint (passed straight through), a
#: ``(timestamp, (wx, wy, wz))`` pair, or a flat ``(timestamp, wx, wy, wz)``.
RawMotionSample = Union[MotionHint, Sequence[Any]]


class StandaloneStereoInterface:
    """The supported public entry point for running DPE standalone.

    Construction mirrors DepthPerceptionPipeline's own signature exactly —
    same argument names, same defaults, same meanings — so moving between
    the two interfaces requires no re-learning:

        from depth_perception_engine.standalone import StandaloneStereoInterface

        dpe = StandaloneStereoInterface.from_calibration_file(
            "examples/config/stereo_calibration.xml",
            PipelineConfig(enable_geometry=True),
        )
        geometry = dpe.process_geometry_frame(left_image, right_image, timestamp=t)

    Args:
        config: The same config.PipelineConfig the core takes. Every
            geometry/temporal capability flag keeps its exact meaning —
            this class adds no flag of its own.
        calibration: A loaded calibration.StereoCalibration. Use
            from_calibration_file() to load one from a path instead.
        rectify: Forwarded unchanged to DepthPerceptionPipeline.
        body_T_camera_left: Forwarded unchanged to DepthPerceptionPipeline.
        engine: Dependency-injection seam, for tests and for a caller that
            already holds a configured pipeline. When supplied, that exact
            object is used and `config`/`calibration`/`rectify`/
            `body_T_camera_left` are ignored — this class never builds a
            second engine behind a caller's back, and never processes
            anything itself. Defaults to None (build one from `config` and
            `calibration`).
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        calibration: Optional[StereoCalibration] = None,
        rectify: bool = True,
        body_T_camera_left: Optional[RigidTransform] = None,
        engine: Optional[DepthPerceptionPipeline] = None,
    ) -> None:
        if engine is None:
            if calibration is None:
                raise ValueError(
                    "StandaloneStereoInterface requires a StereoCalibration "
                    "(or an already-constructed `engine`). Load one first with "
                    "StandaloneStereoInterface.from_calibration_file(path, config) "
                    "or calibration.load_stereo_calibration(path) — this library "
                    "never assumes a default calibration file."
                )
            engine = DepthPerceptionPipeline(
                config if config is not None else PipelineConfig(),
                calibration,
                rectify=rectify,
                body_T_camera_left=body_T_camera_left,
            )
        self._engine = engine
        self._splitter = FrameSplitter()

    # ------------------------------------------------------------------
    @classmethod
    def from_calibration_file(
        cls,
        calibration_path: str,
        config: Optional[PipelineConfig] = None,
        rectify: bool = True,
        body_T_camera_left: Optional[RigidTransform] = None,
    ) -> "StandaloneStereoInterface":
        """Load a calibration from disk and build a standalone interface.

        The ONE file-path-accepting entry point in either DPE interface.
        Delegates to the existing, unchanged
        calibration.load_stereo_calibration() — no parsing, unit, baseline,
        Q-matrix, or intrinsics logic is reimplemented here.
        """
        calibration = load_stereo_calibration(calibration_path)
        return cls(
            config if config is not None else PipelineConfig(),
            calibration,
            rectify=rectify,
            body_T_camera_left=body_T_camera_left,
        )

    # ------------------------------------------------------------------
    # Input adaptation — the entire reason this class exists
    # ------------------------------------------------------------------
    @staticmethod
    def build_motion_hint(
        timestamp: float,
        angular_velocity_rad_s: Sequence[float],
        frame_id: str = FrameId.BODY,
        valid: bool = True,
    ) -> MotionHint:
        """Normalize one raw angular-rate reading into a temporal.MotionHint.

        Pure adaptation: the (3,) float64 array required by MotionHint's own
        already-frozen __post_init__ validation is built from whatever
        array-like the caller has. No filtering, integration, axis
        conversion, or frame inference happens here — a MotionHint produced
        this way is indistinguishable from one a caller constructs directly,
        which is exactly the contract temporal.MotionHint requires.
        """
        vector = np.asarray(angular_velocity_rad_s, dtype=np.float64).reshape(-1)
        return MotionHint(
            timestamp=float(timestamp),
            angular_velocity_rad_s=vector,
            frame_id=frame_id,
            valid=valid,
        )

    @classmethod
    def build_motion_hints(
        cls,
        samples: Optional[Iterable[RawMotionSample]],
        frame_id: str = FrameId.BODY,
    ) -> Optional[List[MotionHint]]:
        """Normalize a bounded sequence of raw motion samples.

        Accepts, per element: an already-built MotionHint (returned
        unchanged — never rebuilt or re-validated), a
        ``(timestamp, (wx, wy, wz))`` pair, or a flat
        ``(timestamp, wx, wy, wz)``. Returns None for None (the core's own
        "no motion samples for this interval" case), preserving the
        existing degradation semantics exactly.
        """
        if samples is None:
            return None
        hints: List[MotionHint] = []
        for sample in samples:
            if isinstance(sample, MotionHint):
                hints.append(sample)
                continue
            values = list(sample)
            if len(values) == 2:
                timestamp, angular_velocity = values
            elif len(values) == 4:
                timestamp, angular_velocity = values[0], values[1:]
            else:
                raise ValueError(
                    "A raw motion sample must be a MotionHint, a "
                    "(timestamp, (wx, wy, wz)) pair, or a flat "
                    f"(timestamp, wx, wy, wz) 4-tuple — got {len(values)} value(s)."
                )
            hints.append(cls.build_motion_hint(timestamp, angular_velocity, frame_id=frame_id))
        return hints

    def split_combined_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split one side-by-side stereo frame into (left, right) views.

        Delegates to the existing, unchanged stereo.FrameSplitter. Returns
        NumPy VIEWS onto `frame` (plain slicing) — no copy is made here, and
        none is made downstream either.
        """
        return self._splitter.split(frame)

    def build_observation(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        timestamp: Optional[float] = None,
        right_timestamp: Optional[float] = None,
        motion_hint: Optional[MotionHint] = None,
        motion_hints: Optional[Sequence[MotionHint]] = None,
        motion_samples: Optional[Iterable[RawMotionSample]] = None,
        motion_frame_id: str = FrameId.BODY,
        observation_id: Optional[str] = None,
    ) -> StereoObservation:
        """Assemble the canonical core input contract from loose arguments.

        `timestamp` is the convenience spelling of the core's own
        `left_timestamp` (the field the core already treats as
        authoritative when both are present) — `right_timestamp` remains
        available for a caller that genuinely has two.

        `motion_samples` is the raw/convenient alternative to
        `motion_hints`: it is normalized through build_motion_hints() above.
        Supplying both is a caller error and raises, rather than silently
        preferring one.

        `observation_id` (Phase D2) is passed straight through onto the
        core contract's own StereoObservation.observation_id — this
        adapter neither generates nor interprets identity, exactly as it
        neither generates timestamps nor interprets motion. Note
        `motion_frame_id` above is an unrelated COORDINATE frame for the
        motion samples; the two must not be confused.

        Images are stored on the returned StereoObservation BY REFERENCE.
        """
        if motion_samples is not None and motion_hints is not None:
            raise ValueError(
                "Pass either motion_hints (already-built MotionHint values) or "
                "motion_samples (raw readings to normalize), not both."
            )
        if motion_samples is not None:
            motion_hints = self.build_motion_hints(motion_samples, frame_id=motion_frame_id)
        return StereoObservation(
            left_image=left_image,
            right_image=right_image,
            left_timestamp=timestamp,
            right_timestamp=right_timestamp,
            motion_hint=motion_hint,
            motion_hints=motion_hints,
            observation_id=observation_id,
        )

    # ------------------------------------------------------------------
    # Execution — every method below delegates to the ONE core engine
    # ------------------------------------------------------------------
    def process(self, left_image: np.ndarray, right_image: np.ndarray, **kwargs: Any) -> DepthPerceptionResult:
        """Adapt loose stereo arguments and run them through the core.

        Exactly equivalent to
        ``engine.process_observation(self.build_observation(...))`` — see
        build_observation() for the accepted keyword arguments.
        """
        return self._engine.process_observation(self.build_observation(left_image, right_image, **kwargs))

    def process_geometry_frame(
        self, left_image: np.ndarray, right_image: np.ndarray, **kwargs: Any
    ) -> GeometryFrame:
        """Adapt loose stereo arguments and return the authoritative
        GeometryFrame from the core.

        Exactly equivalent to
        ``engine.process_geometry_frame(self.build_observation(...))`` —
        the same single core implementation, the same single output
        contract an embedded consumer receives. There is no
        standalone-specific frame type.
        """
        return self._engine.process_geometry_frame(self.build_observation(left_image, right_image, **kwargs))

    def process_combined_frame(self, frame: np.ndarray, **kwargs: Any) -> DepthPerceptionResult:
        """Split one side-by-side frame, then run it through the core."""
        left_image, right_image = self.split_combined_frame(frame)
        return self.process(left_image, right_image, **kwargs)

    def process_combined_frame_geometry(self, frame: np.ndarray, **kwargs: Any) -> GeometryFrame:
        """Split one side-by-side frame, then return the core's GeometryFrame."""
        left_image, right_image = self.split_combined_frame(frame)
        return self.process_geometry_frame(left_image, right_image, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle / introspection — pure pass-through to the core
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the core engine's cross-frame state (see
        DepthPerceptionPipeline.reset()). This class holds no cross-frame
        state of its own to clear."""
        self._engine.reset()

    def close(self) -> None:
        """Close the core engine (see DepthPerceptionPipeline.close())."""
        self._engine.close()

    def health(self) -> PipelineHealth:
        """The core engine's own PipelineHealth snapshot, unmodified."""
        return self._engine.health()

    @property
    def engine(self) -> DepthPerceptionPipeline:
        """The single core engine this interface delegates to.

        Exposed so a standalone caller can hand the SAME engine to an
        embedded-style call site (or to a test) without rebuilding it —
        further evidence there is only ever one geometry implementation in
        play.
        """
        return self._engine

    @property
    def config(self) -> PipelineConfig:
        return self._engine.config

    @property
    def calibration(self) -> StereoCalibration:
        return self._engine.calibration

    def __repr__(self) -> str:
        return f"StandaloneStereoInterface(engine={self._engine!r})"
