"""
Depth Perception Engine — a standalone, ROS-free stereo depth/traversability/
obstacle library.

Public surface (see docs/INTEGRATION_READINESS.md for how ROS2's
mp01_perception will call this):

    from depth_perception_engine.calibration import load_stereo_calibration
    from depth_perception_engine.config import PipelineConfig
    from depth_perception_engine.pipeline import DepthPerceptionPipeline

    calibration = load_stereo_calibration("/path/to/stereo_calibration.xml")
    pipeline = DepthPerceptionPipeline(PipelineConfig(), calibration)
    result = pipeline.process(left_image, right_image)

No module in this package imports rclpy, sensor_msgs, cv_bridge, opens a
camera device, or calls any cv2.imshow/waitKey/GUI function — see
tests/test_no_ros_dependency.py.
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
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
    # pipeline
    "DepthPerceptionPipeline",
    "process_stereo_pair",
    "compute_disparity",
    "estimate_depth",
    "classify_traversability",
    "detect_obstacles",
]
