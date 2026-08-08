"""
State / recovery — Level 3, Phase E6 (Task 8).

E6 remains single-frame geometry — there must be no accidental
persistence of scene geometry between frames. The audit (Task 1)
confirmed geometry/geometry_body/obstacle_cloud/free_space_rays/
geometry_metrics are all plain local variables inside process(), freshly
computed every call, with zero caching anywhere in DepthPerceptionPipeline
— the only cross-frame state anywhere in the pipeline is
obstacles.ThreatAssessor's per-beam EMA/debounce (Level 0-2, pre-dates
E2-E6, deliberately unaffected by any of this). This file proves that
structural claim behaviorally: healthy -> invalid -> healthy, confirming
the invalid frame does not inherit the healthy frame's geometry, and the
following healthy frame is unaffected by having processed an invalid one
in between.
"""

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _full_config():
    return PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)


def _healthy_pair(seed=11):
    w, h = _CALIBRATION.image_size
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return left, right


def _invalid_pair():
    w, h = _CALIBRATION.image_size
    flat = np.full((h, w, 3), 128, dtype=np.uint8)
    return flat, flat.copy()


class TestHealthyInvalidHealthySequence:
    def test_invalid_frame_does_not_inherit_previous_healthy_geometry(self):
        pipeline = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())
        healthy_left, healthy_right = _healthy_pair()
        invalid_left, invalid_right = _invalid_pair()

        healthy_result = pipeline.process(healthy_left, healthy_right)
        assert healthy_result.geometry_metrics.valid_fraction > 0.1, (
            "fixture must be genuinely healthy for this test to be meaningful"
        )

        invalid_result = pipeline.process(invalid_left, invalid_right)

        # The invalid frame must describe ONLY the invalid frame — not a
        # leftover mix with the previous healthy frame's evidence.
        assert invalid_result.geometry_metrics.valid_fraction == 0.0
        assert invalid_result.geometry_metrics.point_count == 0
        assert invalid_result.obstacle_cloud.points.shape[0] == 0
        assert invalid_result.free_space_rays.ranges_m.shape[0] == 0
        assert np.all(np.isnan(invalid_result.geometry_body.points[~invalid_result.geometry_body.valid_mask]))
        # Every single pixel invalid — none of the previous frame's valid points leaked through.
        assert not np.any(invalid_result.geometry_body.valid_mask)

    def test_following_healthy_frame_recovers_normally(self):
        pipeline = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())
        healthy_left, healthy_right = _healthy_pair()
        invalid_left, invalid_right = _invalid_pair()

        first_healthy = pipeline.process(healthy_left, healthy_right)
        pipeline.process(invalid_left, invalid_right)
        second_healthy = pipeline.process(healthy_left, healthy_right)

        # Recovery is exact: the same healthy input, processed again after
        # an invalid frame in between, produces the same geometry as
        # before the invalid frame — proving no residual contamination.
        np.testing.assert_array_equal(first_healthy.geometry_body.valid_mask, second_healthy.geometry_body.valid_mask)
        np.testing.assert_allclose(first_healthy.geometry_body.points, second_healthy.geometry_body.points, equal_nan=True)
        np.testing.assert_allclose(first_healthy.obstacle_cloud.points, second_healthy.obstacle_cloud.points)
        np.testing.assert_allclose(first_healthy.free_space_rays.ranges_m, second_healthy.free_space_rays.ranges_m)
        assert first_healthy.geometry_metrics == second_healthy.geometry_metrics

    def test_no_stale_geometry_object_identity_leaks_across_calls(self):
        """Structural proof, not just value equality: the invalid frame's
        result objects must not literally BE (same object identity as)
        the healthy frame's — ruling out any accidental caching/reuse of
        a previous PointCloud/ObstacleCloud/FreeSpaceRays/GeometryMetrics
        instance."""
        pipeline = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())
        healthy_left, healthy_right = _healthy_pair()
        invalid_left, invalid_right = _invalid_pair()

        healthy_result = pipeline.process(healthy_left, healthy_right)
        invalid_result = pipeline.process(invalid_left, invalid_right)

        assert invalid_result.geometry is not healthy_result.geometry
        assert invalid_result.geometry_body is not healthy_result.geometry_body
        assert invalid_result.obstacle_cloud is not healthy_result.obstacle_cloud
        assert invalid_result.free_space_rays is not healthy_result.free_space_rays
        assert invalid_result.geometry_metrics is not healthy_result.geometry_metrics
        assert invalid_result.geometry_body.points is not healthy_result.geometry_body.points

    def test_multiple_healthy_invalid_cycles_remain_stable(self):
        """Extended version: three full cycles, proving stability isn't a
        one-off — relevant since this is the direct prerequisite property
        temporal fusion (explicitly out of scope, E6+) would depend on."""
        pipeline = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())
        healthy_left, healthy_right = _healthy_pair()
        invalid_left, invalid_right = _invalid_pair()

        healthy_results = []
        for _ in range(3):
            pipeline.process(invalid_left, invalid_right)
            healthy_results.append(pipeline.process(healthy_left, healthy_right))
            pipeline.process(invalid_left, invalid_right)

        for r in healthy_results:
            assert r.geometry_metrics.valid_fraction > 0.1
        for i in range(1, len(healthy_results)):
            np.testing.assert_array_equal(
                healthy_results[0].geometry_body.valid_mask, healthy_results[i].geometry_body.valid_mask,
            )


class TestThreatAssessorStateIsTheOnlyIntentionalException:
    """ThreatAssessor's per-beam EMA/debounce is Level 0-2, pre-dates
    E2-E6, and is explicitly the one deliberate exception to "no
    cross-frame state" in this pipeline — its status debouncing means a
    single invalid frame does not necessarily flip beam status
    immediately (see obstacles.ThreatAssessor's own debounce_frames
    config), which is intended Level 0-2 behavior this E6 pass does not
    touch. Geometry (E2-E5) has no equivalent smoothing anywhere — this
    test confirms that distinction explicitly, so a future reader does
    not mistake ThreatAssessor's debounce for a Level 3 geometry leak."""

    def test_geometry_is_immediately_and_fully_replaced_even_though_obstacles_may_debounce(self):
        pipeline = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=_transform())
        healthy_left, healthy_right = _healthy_pair()
        invalid_left, invalid_right = _invalid_pair()

        pipeline.process(healthy_left, healthy_right)
        invalid_result = pipeline.process(invalid_left, invalid_right)

        # Geometry (E2-E5): instantaneous, zero smoothing, zero debounce.
        assert invalid_result.geometry_metrics.valid_fraction == 0.0
        # obstacles (Level 0-2): status MAY still reflect debounce state
        # from the previous healthy frame — not asserted either way here,
        # since that is correct, pre-existing, out-of-scope behavior; the
        # only claim this test makes is that geometry does NOT share it.
