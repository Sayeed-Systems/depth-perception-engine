"""Every module imports correctly, with no camera/hardware/GUI side effects at import time."""

import importlib


_MODULES = [
    "depth_perception_engine",
    "depth_perception_engine.calibration",
    "depth_perception_engine.calibration.models",
    "depth_perception_engine.calibration.loader",
    "depth_perception_engine.stereo",
    "depth_perception_engine.stereo.frame_splitter",
    "depth_perception_engine.stereo.rectification",
    "depth_perception_engine.stereo.disparity_engine",
    "depth_perception_engine.depth",
    "depth_perception_engine.depth.depth_estimator",
    "depth_perception_engine.depth.distance_reader",
    "depth_perception_engine.traversability",
    "depth_perception_engine.traversability.types",
    "depth_perception_engine.traversability.region_analyzer",
    "depth_perception_engine.traversability.scene_interpreter",
    "depth_perception_engine.obstacles",
    "depth_perception_engine.obstacles.threat_assessment",
    "depth_perception_engine.fusion",
    "depth_perception_engine.fusion.result_builder",
    "depth_perception_engine.config",
    "depth_perception_engine.config.pipeline_config",
    "depth_perception_engine.models",
    "depth_perception_engine.models.result",
    "depth_perception_engine.utils",
    "depth_perception_engine.utils.validation",
    "depth_perception_engine.utils.timing",
    "depth_perception_engine.pipeline",
    "depth_perception_engine.pipeline.api",
    "depth_perception_engine.pipeline.pipeline",
]


def test_every_module_imports():
    for name in _MODULES:
        importlib.import_module(name)


def test_top_level_public_api_is_exposed():
    import depth_perception_engine as dpe

    expected = {
        "StereoCalibration",
        "load_stereo_calibration",
        "PipelineConfig",
        "DepthPerceptionResult",
        "TraversabilityResult",
        "ObstacleAssessment",
        "BeamReading",
        "DepthPerceptionPipeline",
        "process_stereo_pair",
        "compute_disparity",
        "estimate_depth",
        "classify_traversability",
        "detect_obstacles",
    }
    missing = expected - set(dpe.__all__)
    assert not missing, f"Missing from depth_perception_engine.__all__: {missing}"
    for name in expected:
        assert hasattr(dpe, name), f"depth_perception_engine.{name} is not actually importable"
