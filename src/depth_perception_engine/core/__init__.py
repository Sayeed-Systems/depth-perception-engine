"""
CORE / EMBEDDED API — the DPE geometry engine boundary.

This is the module an EMBEDDED consumer (a future
`hybrid_perception_engine`) imports. It names, in one place, the complete
set of symbols required to construct DPE, hand it an already-prepared
observation, and receive DPE's authoritative output:

    from depth_perception_engine.core import (
        DepthPerceptionPipeline,   # the engine
        PipelineConfig,            # configuration
        StereoCalibration,         # calibration value object
        StereoObservation,         # THE canonical core input contract
        MotionHint,                # optional normalized motion input
        GeometryFrame,             # THE authoritative output contract
    )

    pipeline = DepthPerceptionPipeline(config, calibration)      # once
    geometry = pipeline.process_geometry_frame(observation)      # per frame

Every symbol re-exported here is the exact same object the package root
already exports (identity-checked in tests/test_public_api.py and
tests/test_dual_interface_architecture.py) — this module introduces no new
class, no new contract, and no wrapper around the engine. It exists purely
to make the embedded boundary explicit and structurally verifiable:

  * It imports NOTHING from `depth_perception_engine.standalone`. Importing
    this module therefore never constructs, initializes, or even loads the
    standalone sensor-facing adapter — proven in
    tests/test_dual_interface_architecture.py::TestStandaloneOptionality,
    which asserts the standalone module is absent from a fresh
    interpreter's `sys.modules` after importing this one.

  * It exposes no sensor acquisition of any kind: no camera/device
    handling, no combined-frame splitting, no calibration FILE loading
    (`load_stereo_calibration` is deliberately NOT re-exported here — an
    embedded consumer hands DPE a `StereoCalibration` object it already
    owns; reading one off disk is a standalone/development convenience).

  * `RigidTransform` and `FrameId` are included because
    `body_T_camera_left` (a documented constructor input) and every
    `frame_id` string value on the output type graph are meaningless
    without them.

Both DPE interfaces converge on ONE implementation —
`DepthPerceptionPipeline.process_observation()`. See
docs/DUAL_INTERFACE_ARCHITECTURE.md.
"""

from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.config.pipeline_config import PipelineConfig
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.provider import GeometryFrame
from depth_perception_engine.models.result import DepthPerceptionResult, PipelineHealth, StereoObservation
from depth_perception_engine.pipeline.pipeline import DepthPerceptionPipeline
from depth_perception_engine.temporal.types import MotionHint

__all__ = [
    # construction / configuration
    "DepthPerceptionPipeline",
    "PipelineConfig",
    "StereoCalibration",
    "RigidTransform",
    # canonical input contract
    "StereoObservation",
    "MotionHint",
    # authoritative output contract
    "GeometryFrame",
    "FrameId",
    # legacy/compatibility result shape (still returned by
    # process_observation(); an embedded consumer needs only GeometryFrame)
    "DepthPerceptionResult",
    "PipelineHealth",
]
