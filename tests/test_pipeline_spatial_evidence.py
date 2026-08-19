"""
DepthPerceptionPipeline spatial-evidence integration — Level 3, Phase E5.

Covers what E5 adds on top of the already-tested E4 pipeline integration
(tests/test_pipeline_body_frame.py) and the already-tested builder math
(tests/test_obstacle_extractor.py, tests/test_free_space.py,
tests/test_geometry_metrics.py): the two new config flags, the process()
integration point, the three new result fields, the mandatory
UNKNOWN-space safety rule enforced end-to-end through the real pipeline,
and zero-regression on every pre-E5 output.
"""

import numpy as np
import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import FreeSpaceRays, GeometryMetrics, ObstacleCloud
from depth_perception_engine.pipeline import DepthPerceptionPipeline


def _illustrative_transform():
    """Same synthetic, illustrative-only extrinsic as
    tests/test_pipeline_body_frame.py — not a measurement of any real
    airframe. See docs/COORDINATE_FRAMES.md's E4 section."""
    angle = np.deg2rad(15.0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    translation = np.array([0.08, 0.0, 0.05])
    return RigidTransform(
        rotation=rotation, translation=translation,
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _flat_textureless_pair(calibration):
    """Same fixture as tests/test_pipeline_geometry.py — reliably forces
    zero valid disparity/depth anywhere in the frame (verified there;
    reused here to force zero valid body-frame geometry too)."""
    width, height = calibration.image_size
    flat = np.full((height, width, 3), 128, dtype=np.uint8)
    return flat, flat.copy()


def _full_e5_config(**overrides):
    defaults = dict(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestOutputsPresentWhenEnabled:
    def test_all_three_e5_fields_present(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert isinstance(result.obstacle_cloud, ObstacleCloud)
        assert isinstance(result.free_space_rays, FreeSpaceRays)
        assert isinstance(result.geometry_metrics, GeometryMetrics)
        assert result.obstacle_cloud.frame_id == FrameId.BODY
        assert result.free_space_rays.frame_id == FrameId.BODY


class TestOutputsAbsentWhenDisabled:
    def test_absent_when_enable_geometry_false(self, calibration, stereo_pair):
        config = PipelineConfig(enable_geometry=False)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.obstacle_cloud is None
        assert result.free_space_rays is None
        assert result.geometry_metrics is None

    def test_absent_when_no_body_transform_even_with_flags_enabled(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_full_e5_config(), calibration)  # no body_T_camera_left
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry_body is None
        assert result.obstacle_cloud is None
        assert result.free_space_rays is None
        assert result.geometry_metrics is None

    def test_obstacle_cloud_absent_when_its_own_flag_disabled(self, calibration, stereo_pair):
        config = _full_e5_config(enable_obstacle_geometry=False)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.obstacle_cloud is None
        assert result.free_space_rays is not None  # independent flag, still on

    def test_free_space_rays_absent_when_its_own_flag_disabled(self, calibration, stereo_pair):
        config = _full_e5_config(enable_free_space_rays=False)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.free_space_rays is None
        assert result.obstacle_cloud is not None

    def test_geometry_metrics_present_even_with_both_sub_flags_disabled(self, calibration, stereo_pair):
        """geometry_metrics has no separate gate — populated whenever
        geometry_body exists, per its own documented design (its two
        Optional fields simply stay None when nothing was computed)."""
        config = _full_e5_config(enable_obstacle_geometry=False, enable_free_space_rays=False)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry_metrics is not None
        assert result.geometry_metrics.min_obstacle_distance_m is None
        assert result.geometry_metrics.mean_free_space_m is None
        assert result.geometry_metrics.point_count == int(result.geometry_body.valid_mask.sum())


class TestDerivedFromBodyPointCloudOnly:
    def test_obstacle_cloud_matches_canonical_builder_called_directly(self, calibration, stereo_pair):
        from depth_perception_engine.geometry import build_obstacle_cloud
        from depth_perception_engine.geometry.reliability import compute_shadow_zone_mask

        transform = _illustrative_transform()
        config = _full_e5_config()
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)
        left, right = stereo_pair

        result = pipeline.process(left, right)
        origin = transform.translation
        # Phase I3: the pipeline threads a shadow-zone reliability mask
        # (computed from disparity_map alone) into the canonical builder
        # — reproduce that exact same input here, or this "single
        # canonical producer" proof would spuriously fail on any frame
        # where the mask is non-trivial, not because a second, divergent
        # implementation exists.
        reliability_mask = None
        if config.geometry_shadow_zone_enabled:
            reliability_mask = compute_shadow_zone_mask(
                result.disparity_map, result.disparity_map > 0.0,
                lookahead_px=config.geometry_shadow_zone_lookahead_px,
                gradient_threshold_px=config.geometry_shadow_zone_gradient_threshold_px,
                max_width_px=config.geometry_shadow_zone_max_width_px,
            )
        reference = build_obstacle_cloud(
            result.geometry_body, origin,
            min_range_m=config.obstacle_min_range_m, max_range_m=config.obstacle_max_range_m,
            stride=config.geometry_sampling_stride,
            reliability_mask=reliability_mask,
        )

        np.testing.assert_array_equal(result.obstacle_cloud.points, reference.points)
        np.testing.assert_array_equal(result.obstacle_cloud.distances_m, reference.distances_m)

    def test_free_space_rays_matches_canonical_builder_called_directly(self, calibration, stereo_pair):
        from depth_perception_engine.geometry import build_free_space_rays
        from depth_perception_engine.geometry.reliability import compute_shadow_zone_mask

        transform = _illustrative_transform()
        config = _full_e5_config()
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)
        left, right = stereo_pair

        result = pipeline.process(left, right)
        origin = transform.translation
        # Phase I3: see the identical note in
        # test_obstacle_cloud_matches_canonical_builder_called_directly.
        reliability_mask = None
        if config.geometry_shadow_zone_enabled:
            reliability_mask = compute_shadow_zone_mask(
                result.disparity_map, result.disparity_map > 0.0,
                lookahead_px=config.geometry_shadow_zone_lookahead_px,
                gradient_threshold_px=config.geometry_shadow_zone_gradient_threshold_px,
                max_width_px=config.geometry_shadow_zone_max_width_px,
            )
        reference = build_free_space_rays(
            result.geometry_body, origin, stride=config.geometry_sampling_stride,
            reliability_mask=reliability_mask,
        )

        np.testing.assert_array_equal(result.free_space_rays.ranges_m, reference.ranges_m)
        np.testing.assert_array_equal(result.free_space_rays.directions, reference.directions)

    def test_free_space_ray_endpoints_correspond_to_body_cloud_surface_points(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)
        rays = result.free_space_rays
        endpoints = rays.origins + rays.directions * rays.ranges_m[:, np.newaxis]

        # Every ray endpoint must be one of the valid body-frame surface points.
        valid_points = result.geometry_body.points[result.geometry_body.valid_mask]
        # Spot-check a sample (full N^2 membership check would be slow at
        # full resolution) — sorted-lexicographic match on a subsample.
        sample_idx = np.linspace(0, endpoints.shape[0] - 1, min(50, endpoints.shape[0])).astype(int)
        for i in sample_idx:
            distances_to_valid = np.linalg.norm(valid_points - endpoints[i], axis=-1)
            assert np.min(distances_to_valid) < 1e-3


class TestUnknownSpaceSafetyRule:
    """
    Task 5 / the mission's Core Safety Rule, enforced end-to-end through
    the real DepthPerceptionPipeline (not just at the unit level already
    covered in tests/test_obstacle_extractor.py and tests/test_free_space.py):
    INVALID DEPTH MUST NEVER BECOME FREE SPACE, and must never become an
    obstacle point either.
    """

    def test_no_valid_geometry_yields_zero_obstacle_points_and_zero_rays(self, calibration):
        """A flat, textureless scene: SGBM finds no correspondence
        anywhere, so valid_disparity_mask/valid_depth_mask are entirely
        False (verified reliable in test_pipeline_geometry.py). This must
        propagate all the way through: zero obstacle points, zero rays —
        never a crash, and never a fabricated free-space/obstacle claim."""
        pipeline = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=_illustrative_transform())
        left, right = _flat_textureless_pair(calibration)

        result = pipeline.process(left, right)

        assert result.valid_depth_mask.sum() == 0
        assert result.obstacle_cloud.points.shape[0] == 0
        assert result.free_space_rays.ranges_m.shape[0] == 0
        assert result.geometry_metrics.point_count == 0
        assert result.geometry_metrics.min_obstacle_distance_m is None
        assert result.geometry_metrics.mean_free_space_m is None

    def test_invalid_pixel_count_equals_pixels_excluded_from_both_outputs(self, calibration, stereo_pair):
        """Every invalid pixel in geometry_body must be excluded from
        BOTH obstacle_cloud and free_space_rays — an invalid pixel can
        contribute to neither.

        Phase I3 update: this no longer proves an exact equality against
        valid_count. A geometrically-predicted occlusion/dis-occlusion
        shadow zone (geometry.reliability.compute_shadow_zone_mask, see
        its own module docstring and benchmarks/i3_occlusion_safety/) can
        ADDITIONALLY exclude some nominally-valid-but-unreliable pixels
        from both outputs — obstacle/ray counts can now be LESS than
        valid_count, never more (that remains the absolute, safety-
        critical direction: no invalid pixel can ever become an obstacle
        or a ray). The two outputs must still shrink by exactly the SAME
        amount, since both are built from the SAME reliability mask —
        that symmetry (never "obstacle-excluded but still free" or vice
        versa) is asserted explicitly below."""
        pipeline = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        valid_count = int(result.geometry_body.valid_mask.sum())
        assert valid_count < result.geometry_body.valid_mask.size, (
            "fixture must contain at least one invalid pixel for this to be meaningful"
        )
        assert result.obstacle_cloud.points.shape[0] <= valid_count
        assert result.free_space_rays.ranges_m.shape[0] <= valid_count
        assert result.obstacle_cloud.points.shape[0] == result.free_space_rays.ranges_m.shape[0], (
            "obstacle_cloud and free_space_rays must be excluded symmetrically by any "
            "reliability mask, never asymmetrically"
        )

    def test_range_excluded_points_are_not_silently_reinterpreted_as_free(self, calibration, stereo_pair):
        """A point excluded from ObstacleCloud by a tight range window is
        still UNKNOWN-with-respect-to-obstacle-reporting, not free — it
        still produces a free-space ray (its surface evidence is real),
        proving obstacle-range-filtering and free-space-ray generation
        are independent, not one silently implying the other."""
        config = _full_e5_config(obstacle_max_range_m=0.01)  # excludes virtually everything
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.obstacle_cloud.points.shape[0] == 0  # tight range excludes all obstacle points
        assert result.free_space_rays.ranges_m.shape[0] > 0  # but real surface evidence still produces rays


class TestCameraAndBodyGeometryUnchanged:
    def test_e3_camera_geometry_unchanged_by_e5(self, calibration, stereo_pair):
        transform = _illustrative_transform()
        left, right = stereo_pair

        pipeline_no_e5 = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration, body_T_camera_left=transform)
        pipeline_full_e5 = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=transform)

        result_no_e5 = pipeline_no_e5.process(left, right)
        result_full_e5 = pipeline_full_e5.process(left, right)

        np.testing.assert_array_equal(result_no_e5.geometry.points, result_full_e5.geometry.points)
        np.testing.assert_array_equal(result_no_e5.geometry.valid_mask, result_full_e5.geometry.valid_mask)

    def test_e4_body_geometry_unchanged_by_e5(self, calibration, stereo_pair):
        transform = _illustrative_transform()
        left, right = stereo_pair

        pipeline_no_e5 = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration, body_T_camera_left=transform)
        pipeline_full_e5 = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=transform)

        result_no_e5 = pipeline_no_e5.process(left, right)
        result_full_e5 = pipeline_full_e5.process(left, right)

        np.testing.assert_array_equal(result_no_e5.geometry_body.points, result_full_e5.geometry_body.points)
        np.testing.assert_array_equal(result_no_e5.geometry_body.valid_mask, result_full_e5.geometry_body.valid_mask)


class TestZeroRegression:
    def test_level_0_2_outputs_unchanged_with_e5_enabled(self, calibration, stereo_pair):
        left, right = stereo_pair

        pipeline_off = DepthPerceptionPipeline(PipelineConfig(), calibration)
        pipeline_on = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=_illustrative_transform())

        result_off = pipeline_off.process(left, right)
        result_on = pipeline_on.process(left, right)

        np.testing.assert_array_equal(result_off.disparity_map, result_on.disparity_map)
        np.testing.assert_array_equal(result_off.depth_map, result_on.depth_map)
        np.testing.assert_array_equal(result_off.valid_disparity_mask, result_on.valid_disparity_mask)
        np.testing.assert_array_equal(result_off.valid_depth_mask, result_on.valid_depth_mask)
        assert result_off.confidence == result_on.confidence
        assert result_off.traversability_mask.decision == result_on.traversability_mask.decision
        assert [b.status for b in result_off.obstacles.beams] == [b.status for b in result_on.obstacles.beams]


class TestLifecycle:
    def test_reset_then_process_still_produces_e5_outputs(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair
        pipeline.process(left, right)

        pipeline.reset()
        result = pipeline.process(left, right)

        assert result.obstacle_cloud is not None
        assert result.free_space_rays is not None
        assert result.geometry_metrics is not None

    def test_close_then_process_raises_with_e5_enabled(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair
        pipeline.close()

        with pytest.raises(RuntimeError):
            pipeline.process(left, right)


class TestConfigValidation:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.enable_obstacle_geometry is False
        assert config.enable_free_space_rays is False
        assert config.obstacle_min_range_m == 0.0
        assert config.obstacle_max_range_m == float("inf")
        assert config.geometry_sampling_stride == 1

    def test_rejects_negative_min_range(self):
        with pytest.raises(ValueError, match="obstacle_min_range_m"):
            PipelineConfig(obstacle_min_range_m=-1.0)

    def test_rejects_min_greater_than_max(self):
        with pytest.raises(ValueError, match="obstacle_min_range_m"):
            PipelineConfig(obstacle_min_range_m=5.0, obstacle_max_range_m=1.0)

    def test_rejects_stride_below_one(self):
        with pytest.raises(ValueError, match="geometry_sampling_stride"):
            PipelineConfig(geometry_sampling_stride=0)

    def test_accepts_equal_min_and_max_range(self):
        PipelineConfig(obstacle_min_range_m=2.0, obstacle_max_range_m=2.0)  # must not raise


class TestDeterminism:
    def test_repeated_process_calls_produce_identical_e5_outputs(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_full_e5_config(), calibration, body_T_camera_left=_illustrative_transform())
        left, right = stereo_pair

        r1 = pipeline.process(left, right)
        r2 = pipeline.process(left, right)

        np.testing.assert_array_equal(r1.obstacle_cloud.points, r2.obstacle_cloud.points)
        np.testing.assert_array_equal(r1.free_space_rays.ranges_m, r2.free_space_rays.ranges_m)
        assert r1.geometry_metrics == r2.geometry_metrics
