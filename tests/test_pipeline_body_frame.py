"""
DepthPerceptionPipeline body-frame integration — Level 3, Phase E4.

Covers exactly what E4 adds on top of the already-tested E3 pipeline
integration (tests/test_pipeline_geometry.py) and the already-tested
transform math (tests/test_rigid_transform.py): the body_T_camera_left
constructor parameter, the process() integration point, the
geometry_body result field, absent-extrinsic semantics, and
zero-regression on every pre-E4 output (including the E3 camera-frame
cloud itself).
"""

import dataclasses

import numpy as np
import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import PointCloud, transform_point_cloud
from depth_perception_engine.pipeline import DepthPerceptionPipeline


def _illustrative_transform():
    """A synthetic, illustrative camera-to-body extrinsic for tests only —
    not a real measured rig calibration. See docs/COORDINATE_FRAMES.md's
    E4 section: real hardware deployment must supply measured/calibrated
    values through configuration, not this test fixture."""
    angle = np.deg2rad(15.0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    translation = np.array([0.08, 0.0, 0.05])  # metres — illustrative only
    return RigidTransform(
        rotation=rotation, translation=translation,
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


class TestConstructorParameter:
    def test_defaults_to_none(self, config, calibration):
        pipeline = DepthPerceptionPipeline(config, calibration)
        assert pipeline._body_T_camera_left is None  # internal, but this is the whole gate

    def test_accepts_a_valid_transform(self, calibration):
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        assert pipeline._body_T_camera_left is not None

    def test_rejects_transform_with_wrong_from_frame(self, calibration):
        config = PipelineConfig(enable_geometry=True)
        bad = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.BODY, to_frame=FrameId.BODY,
        )
        with pytest.raises(ValueError, match="from_frame"):
            DepthPerceptionPipeline(config, calibration, body_T_camera_left=bad)

    def test_rejects_transform_with_wrong_to_frame(self, calibration):
        config = PipelineConfig(enable_geometry=True)
        bad = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame="not_body",
        )
        with pytest.raises(ValueError, match="to_frame"):
            DepthPerceptionPipeline(config, calibration, body_T_camera_left=bad)

    def test_from_config_forwards_the_transform(self, calibration):
        config = PipelineConfig(enable_geometry=True)
        transform = _illustrative_transform()
        pipeline = DepthPerceptionPipeline.from_config(config, calibration, body_T_camera_left=transform)
        assert pipeline._body_T_camera_left is transform


class TestGeometryBodyOutput:
    def test_present_when_geometry_enabled_and_transform_supplied(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry_body is not None
        assert isinstance(result.geometry_body, PointCloud)
        assert result.geometry_body.frame_id == FrameId.BODY

    def test_absent_when_no_transform_supplied_not_assumed_identity(self, calibration, stereo_pair):
        """Absence of a body extrinsic must mean 'not calibrated,' never a
        silent identity assumption — see
        calibration.contracts.RigCalibration's docstring."""
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration)  # no body_T_camera_left
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry is not None  # camera-frame still produced
        assert result.geometry_body is None  # body-frame not fabricated

    def test_absent_when_geometry_disabled_even_with_transform_supplied(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=False)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry is None
        assert result.geometry_body is None

    def test_organized_shape_matches_camera_cloud(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry_body.points.shape == result.geometry.points.shape
        assert result.geometry_body.points.dtype == np.float32

    def test_valid_mask_identical_to_camera_cloud(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        np.testing.assert_array_equal(result.geometry_body.valid_mask, result.geometry.valid_mask)

    def test_invalid_pixels_stay_nan_in_body_frame(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        invalid = ~result.geometry_body.valid_mask
        assert np.any(invalid), "fixture must contain at least one invalid pixel"
        assert np.all(np.isnan(result.geometry_body.points[invalid]))
        assert np.all(np.isfinite(result.geometry_body.points[result.geometry_body.valid_mask]))

    def test_matches_e4_verified_transform_called_directly(self, calibration, stereo_pair):
        """Strongest correctness check, mirroring E3's own equivalent test:
        feed the pipeline's own camera-frame result.geometry into a
        freshly, independently constructed transform_point_cloud() call
        and confirm result.geometry_body is identical. Proves the
        pipeline uses the one canonical transform implementation, not a
        second divergent one."""
        transform = _illustrative_transform()
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)
        left, right = stereo_pair

        result = pipeline.process(left, right)
        reference = transform_point_cloud(result.geometry, transform)

        np.testing.assert_array_equal(result.geometry_body.points, reference.points)
        np.testing.assert_array_equal(result.geometry_body.valid_mask, reference.valid_mask)

    def test_timestamp_preserved_through_body_transform(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right, left_timestamp=99.0)

        assert result.geometry.timestamp == 99.0
        assert result.geometry_body.timestamp == 99.0


class TestDeterminism:
    def test_repeated_process_calls_produce_identical_body_geometry(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        r1 = pipeline.process(left, right)
        r2 = pipeline.process(left, right)

        np.testing.assert_array_equal(r1.geometry_body.points, r2.geometry_body.points)
        np.testing.assert_array_equal(r1.geometry_body.valid_mask, r2.geometry_body.valid_mask)


class TestZeroRegression:
    """Task 8: E4 must not alter any pre-E4 output — Level 0-2, E3's
    camera-frame cloud, or anything else — for a fixed input, whether or
    not a body transform is supplied."""

    def test_all_pre_e4_outputs_identical_with_or_without_body_transform(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=True)
        left, right = stereo_pair

        pipeline_no_body = DepthPerceptionPipeline(config, calibration)
        pipeline_with_body = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())

        result_no_body = pipeline_no_body.process(left, right)
        result_with_body = pipeline_with_body.process(left, right)

        # Level 0-2
        np.testing.assert_array_equal(result_no_body.disparity_map, result_with_body.disparity_map)
        np.testing.assert_array_equal(result_no_body.depth_map, result_with_body.depth_map)
        np.testing.assert_array_equal(result_no_body.valid_disparity_mask, result_with_body.valid_disparity_mask)
        np.testing.assert_array_equal(result_no_body.valid_depth_mask, result_with_body.valid_depth_mask)
        assert result_no_body.confidence == result_with_body.confidence
        assert result_no_body.traversability_mask.decision == result_with_body.traversability_mask.decision
        assert [b.status for b in result_no_body.obstacles.beams] == [b.status for b in result_with_body.obstacles.beams]

        # E3 camera-frame cloud — must be byte-identical regardless of
        # whether a body transform is configured
        np.testing.assert_array_equal(result_no_body.geometry.points, result_with_body.geometry.points)
        np.testing.assert_array_equal(result_no_body.geometry.valid_mask, result_with_body.geometry.valid_mask)
        assert result_no_body.geometry.frame_id == result_with_body.geometry.frame_id == FrameId.CAMERA_OPTICAL_LEFT

        # Only geometry_body differs
        assert result_no_body.geometry_body is None
        assert result_with_body.geometry_body is not None

    def test_config_disabled_geometry_unaffected_by_body_transform_presence(self, calibration, stereo_pair):
        config_off = PipelineConfig(enable_geometry=False)
        left, right = stereo_pair

        pipeline_a = DepthPerceptionPipeline(config_off, calibration)
        pipeline_b = DepthPerceptionPipeline(config_off, calibration, body_T_camera_left=_illustrative_transform())

        result_a = pipeline_a.process(left, right)
        result_b = pipeline_b.process(left, right)

        np.testing.assert_array_equal(result_a.disparity_map, result_b.disparity_map)
        np.testing.assert_array_equal(result_a.depth_map, result_b.depth_map)
        assert result_a.geometry is None and result_b.geometry is None
        assert result_a.geometry_body is None and result_b.geometry_body is None

    def test_full_e1_through_e3_field_set_matches_pre_e4_snapshot_shape(self, calibration, stereo_pair):
        """dataclasses.fields() order proves geometry_body was appended
        after E3's geometry, and nothing before it moved (Task 8 +
        E3's own equivalent test, restated at the E4 boundary)."""
        from depth_perception_engine.models import DepthPerceptionResult

        names = [f.name for f in dataclasses.fields(DepthPerceptionResult)]
        assert names[:9] == [
            "disparity_map", "depth_map", "traversability_mask", "obstacles",
            "confidence", "processing_time_ms", "valid_disparity_mask",
            "valid_depth_mask", "timestamp",
        ]
        assert names[9] == "geometry"
        assert names[10] == "geometry_body"


class TestLifecycle:
    def test_reset_then_process_still_produces_body_geometry(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair
        pipeline.process(left, right)

        pipeline.reset()
        result = pipeline.process(left, right)

        assert result.geometry_body is not None

    def test_close_then_process_raises_even_with_body_transform(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=True)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair
        pipeline.close()

        with pytest.raises(RuntimeError):
            pipeline.process(left, right)
