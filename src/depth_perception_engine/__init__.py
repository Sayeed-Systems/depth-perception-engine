"""
Depth Perception Engine — a standalone, ROS-free stereo depth/traversability/
obstacle library.

Repository/library name: depth_perception_engine. Canonical processing
object: DepthPerceptionPipeline — the repository is not named after a
class, and the class is not named after the repository; there is no
`DepthPerceptionEngine` symbol and none is planned (see docs/PUBLIC_API.md).

Canonical usage — every symbol below imports directly from this package
root; nothing external should need to import from a submodule:

    from depth_perception_engine import (
        DepthPerceptionPipeline,
        PipelineConfig,
        StereoObservation,
        load_stereo_calibration,
    )

    calibration = load_stereo_calibration("/path/to/stereo_calibration.xml")
    pipeline = DepthPerceptionPipeline(PipelineConfig(), calibration)   # build once
    result = pipeline.process(left_image, right_image)                 # per frame

Three API tiers — full reference in docs/PUBLIC_API.md:

    Tier 1 (primary):  DepthPerceptionPipeline, PipelineConfig,
                        StereoCalibration, StereoObservation,
                        DepthPerceptionResult, PipelineHealth,
                        TraversabilityResult, ObstacleAssessment,
                        BeamReading, NavigationDecision, RegionClass,
                        RegionStats, TextureClass, load_stereo_calibration
    Tier 2 (advanced):  process_stereo_pair, compute_disparity,
                        estimate_depth, detect_obstacles,
                        classify_traversability
    Tier 3 (internal):  everything else — submodules remain importable
                        (e.g. depth_perception_engine.pipeline) but are not
                        part of the documented, stable contract.

Existing subpackage imports (`from depth_perception_engine.pipeline import
DepthPerceptionPipeline`, etc.) continue to work and resolve to the exact
same objects as the top-level import — see
tests/test_public_api.py::TestImportIdentity. They remain valid
compatibility paths, not the canonical style for new code.

No module in this package imports rclpy, sensor_msgs, cv_bridge, opens a
camera device, or calls any cv2.imshow/waitKey/GUI function — see
tests/test_no_ros_dependency.py. Importing this package performs no I/O
and has no runtime side effects.
"""

from depth_perception_engine.calibration import StereoCalibration, load_stereo_calibration
from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.models import (
    BeamReading,
    DepthPerceptionResult,
    ObstacleAssessment,
    PipelineHealth,
    StereoObservation,
    TraversabilityResult,
)
from depth_perception_engine.pipeline import (
    DepthPerceptionPipeline,
    classify_traversability,
    compute_disparity,
    detect_obstacles,
    estimate_depth,
    process_stereo_pair,
)
from depth_perception_engine.traversability import (
    NavigationDecision,
    RegionClass,
    RegionStats,
    TextureClass,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # --- Tier 1: primary stable API ---
    # calibration
    "StereoCalibration",
    "load_stereo_calibration",
    # config
    "PipelineConfig",
    # results
    "DepthPerceptionResult",
    "TraversabilityResult",
    "ObstacleAssessment",
    "BeamReading",
    "StereoObservation",
    "PipelineHealth",
    # traversability result shape (embedded in DepthPerceptionResult;
    # promoted to Tier 1 because a caller cannot meaningfully interpret
    # result.traversability_mask without these — see docs/PUBLIC_API.md)
    "NavigationDecision",
    "RegionClass",
    "RegionStats",
    "TextureClass",
    # pipeline
    "DepthPerceptionPipeline",
    # --- Tier 2: advanced functional API ---
    "process_stereo_pair",
    "compute_disparity",
    "estimate_depth",
    "classify_traversability",
    "detect_obstacles",
]
