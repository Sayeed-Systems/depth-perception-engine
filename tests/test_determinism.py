"""
Determinism — Level 3, Phase E6 (Task 7).

For identical input images, calibration, configuration, and extrinsics,
repeated processing must produce equivalent Level 3 geometry: validity
masks, camera points, body points, obstacle points, free-space rays,
geometry metrics, and (where applicable) the geometry-quality
classification. processing_time_ms is explicitly NOT required to be
identical — it is wall-clock and legitimately varies run to run.

Two independent proofs, matching this codebase's established "same
process() call repeated" (test_pipeline_body_frame.py) and "two separate
pipeline instances built from identical config" (test_pipeline_geometry.py)
patterns:
  1. The same pipeline instance, called twice with the same input.
  2. Two freshly-constructed pipeline instances (identical config,
     calibration, extrinsics), each called once — proves determinism does
     not depend on any hidden state left over from a first call.
"""

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import classify_geometry_quality

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_TOL = dict(rtol=1e-6, atol=1e-6)


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _full_config():
    return PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)


def _random_pair(seed=99):
    w, h = _CALIBRATION.image_size
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return left, right


def _assert_geometry_equivalent(a, b):
    np.testing.assert_array_equal(a.disparity_map, b.disparity_map)
    np.testing.assert_array_equal(a.depth_map, b.depth_map)
    np.testing.assert_array_equal(a.valid_disparity_mask, b.valid_disparity_mask)
    np.testing.assert_array_equal(a.valid_depth_mask, b.valid_depth_mask)
    assert a.confidence == b.confidence

    np.testing.assert_array_equal(a.geometry.valid_mask, b.geometry.valid_mask)
    np.testing.assert_allclose(a.geometry.points, b.geometry.points, **_TOL, equal_nan=True)

    np.testing.assert_array_equal(a.geometry_body.valid_mask, b.geometry_body.valid_mask)
    np.testing.assert_allclose(a.geometry_body.points, b.geometry_body.points, **_TOL, equal_nan=True)

    np.testing.assert_allclose(a.obstacle_cloud.points, b.obstacle_cloud.points, **_TOL)
    np.testing.assert_allclose(a.obstacle_cloud.distances_m, b.obstacle_cloud.distances_m, **_TOL)

    np.testing.assert_allclose(a.free_space_rays.ranges_m, b.free_space_rays.ranges_m, **_TOL)
    np.testing.assert_allclose(a.free_space_rays.directions, b.free_space_rays.directions, **_TOL)
    np.testing.assert_allclose(a.free_space_rays.origins, b.free_space_rays.origins, **_TOL)

    assert a.geometry_metrics == b.geometry_metrics

    config = _full_config()
    quality_a = classify_geometry_quality(
        a.geometry_metrics, config.geometry_healthy_min_valid_fraction, config.geometry_degraded_min_valid_fraction,
    )
    quality_b = classify_geometry_quality(
        b.geometry_metrics, config.geometry_healthy_min_valid_fraction, config.geometry_degraded_min_valid_fraction,
    )
    assert quality_a == quality_b


class TestSamePipelineInstanceRepeatedCalls:
    def test_normal_scene(self):
        pipeline = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())
        left, right = _random_pair()

        r1 = pipeline.process(left, right)
        r2 = pipeline.process(left, right)
        r3 = pipeline.process(left, right)

        _assert_geometry_equivalent(r1, r2)
        _assert_geometry_equivalent(r1, r3)

    def test_textureless_scene(self):
        """Determinism must hold in the degenerate (all-invalid) case
        too, not just the healthy case."""
        pipeline = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())
        w, h = _CALIBRATION.image_size
        left = np.full((h, w, 3), 128, dtype=np.uint8)
        right = left.copy()

        r1 = pipeline.process(left, right)
        r2 = pipeline.process(left, right)

        _assert_geometry_equivalent(r1, r2)
        assert r1.geometry_metrics.valid_fraction == 0.0


class TestIndependentPipelineInstances:
    """Determinism does not depend on any state carried over from a
    first call — two freshly-constructed pipelines, same everything."""

    def test_two_fresh_pipelines_identical_config_produce_equivalent_output(self):
        left, right = _random_pair()

        pipeline_a = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())
        pipeline_b = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())

        result_a = pipeline_a.process(left, right)
        result_b = pipeline_b.process(left, right)

        _assert_geometry_equivalent(result_a, result_b)

    def test_processing_time_is_not_required_to_match(self):
        """Explicit negative-space check: processing_time_ms is wall-clock
        and legitimately varies — this codebase must not (and does not)
        assert equality on it anywhere in this file."""
        left, right = _random_pair()
        pipeline_a = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())
        pipeline_b = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())

        result_a = pipeline_a.process(left, right)
        result_b = pipeline_b.process(left, right)

        assert isinstance(result_a.processing_time_ms, float)
        assert isinstance(result_b.processing_time_ms, float)
        # deliberately no equality assertion between the two
