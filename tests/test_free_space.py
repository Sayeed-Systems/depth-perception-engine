"""
Unit tests for geometry.free_space.build_free_space_rays — Level 3,
Phase E5.

Synthetic geometry only. The core safety rule under test throughout:
invalid depth must NEVER produce a ray, and a ray's endpoint
(origin + direction * range) must exactly reconstruct the measured
surface point — the space strictly before that endpoint is what "free"
means here, never the endpoint itself.
"""

import numpy as np
import pytest

from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry import FreeSpaceRays, PointCloud, build_free_space_rays

_ORIGIN = np.array([0.0, 0.0, 0.0])


def _make_cloud(points_hw3: np.ndarray, valid_mask=None) -> PointCloud:
    if valid_mask is None:
        valid_mask = ~np.isnan(points_hw3).any(axis=-1)
    return PointCloud(points=points_hw3.astype(np.float32), frame_id=FrameId.BODY, valid_mask=valid_mask)


def _endpoints(rays: FreeSpaceRays) -> np.ndarray:
    return rays.origins + rays.directions * rays.ranges_m[:, np.newaxis]


class TestSingleKnownPoint:
    def test_single_valid_point_produces_one_ray_with_correct_origin_and_endpoint(self):
        points = np.full((1, 1, 3), np.nan, dtype=np.float32)
        points[0, 0] = [3.0, 4.0, 0.0]  # range 5.0 from origin
        cloud = _make_cloud(points)

        rays = build_free_space_rays(cloud, _ORIGIN)

        assert isinstance(rays, FreeSpaceRays)
        assert rays.origins.shape == (1, 3)
        assert rays.directions.shape == (1, 3)
        assert rays.ranges_m.shape == (1,)
        np.testing.assert_allclose(rays.origins[0], _ORIGIN)
        np.testing.assert_allclose(rays.ranges_m[0], 5.0, atol=1e-5)
        np.testing.assert_allclose(rays.directions[0], [0.6, 0.8, 0.0], atol=1e-5)
        np.testing.assert_allclose(_endpoints(rays)[0], [3.0, 4.0, 0.0], atol=1e-4)
        assert np.isclose(np.linalg.norm(rays.directions[0]), 1.0, atol=1e-6)

    def test_nonzero_origin_produces_correct_direction_and_range(self):
        points = np.full((1, 1, 3), np.nan, dtype=np.float32)
        points[0, 0] = [0.0, 0.0, 5.0]
        cloud = _make_cloud(points)
        origin = np.array([0.0, 0.0, 2.0])  # camera offset forward in body frame

        rays = build_free_space_rays(cloud, origin)

        np.testing.assert_allclose(rays.origins[0], origin)
        np.testing.assert_allclose(rays.ranges_m[0], 3.0, atol=1e-5)
        np.testing.assert_allclose(rays.directions[0], [0.0, 0.0, 1.0], atol=1e-6)


class TestFlatPlane:
    def test_rays_terminate_exactly_at_the_plane(self):
        h, w = 3, 4
        points = np.zeros((h, w, 3), dtype=np.float32)
        u, v = np.meshgrid(np.arange(w) - w / 2.0, np.arange(h) - h / 2.0)
        points[:, :, 0] = u
        points[:, :, 1] = v
        points[:, :, 2] = 4.0
        cloud = _make_cloud(points)

        rays = build_free_space_rays(cloud, _ORIGIN)
        endpoints = _endpoints(rays)

        assert rays.ranges_m.shape[0] == h * w
        np.testing.assert_allclose(endpoints[:, 2], 4.0, atol=1e-4)
        np.testing.assert_allclose(endpoints[:, 0], points[:, :, 0].ravel(), atol=1e-4)
        np.testing.assert_allclose(endpoints[:, 1], points[:, :, 1].ravel(), atol=1e-4)


class TestInvalidPointsGenerateNoRay:
    def test_all_invalid_produces_zero_rays(self):
        points = np.full((2, 2, 3), np.nan, dtype=np.float32)
        valid_mask = np.zeros((2, 2), dtype=bool)
        cloud = _make_cloud(points, valid_mask=valid_mask)

        rays = build_free_space_rays(cloud, _ORIGIN)

        assert rays.origins.shape == (0, 3)
        assert rays.directions.shape == (0, 3)
        assert rays.ranges_m.shape == (0,)

    def test_mixed_valid_invalid_only_valid_rays_present(self):
        points = np.full((1, 3, 3), np.nan, dtype=np.float32)
        points[0, 0] = [1.0, 0.0, 0.0]
        points[0, 2] = [0.0, 1.0, 0.0]
        valid_mask = np.array([[True, False, True]])
        cloud = _make_cloud(points, valid_mask=valid_mask)

        rays = build_free_space_rays(cloud, _ORIGIN)

        assert rays.ranges_m.shape[0] == 2
        np.testing.assert_allclose(np.sort(rays.ranges_m), [1.0, 1.0], atol=1e-5)

    def test_inf_marked_valid_produces_no_finite_ray(self):
        """Pathological caller-contract-violation input (Inf marked
        valid) must not silently produce a garbage ray — Inf offsets
        produce Inf/NaN range and direction, both explicitly excluded."""
        points = np.zeros((1, 1, 3), dtype=np.float32)
        points[0, 0] = [np.inf, 0.0, 0.0]
        valid_mask = np.ones((1, 1), dtype=bool)
        cloud = _make_cloud(points, valid_mask=valid_mask)

        rays = build_free_space_rays(cloud, _ORIGIN)

        assert rays.ranges_m.shape[0] == 0 or np.all(np.isfinite(rays.ranges_m))
        assert not np.any(np.isnan(rays.directions))
        assert not np.any(np.isinf(rays.directions))


class TestDegenerateOriginCoincidentPoint:
    def test_point_exactly_at_origin_excluded_not_nan(self):
        points = np.zeros((1, 1, 3), dtype=np.float32)
        points[0, 0] = [0.0, 0.0, 0.0]  # coincides with origin -> zero-length ray
        cloud = _make_cloud(points)

        rays = build_free_space_rays(cloud, _ORIGIN)

        assert rays.ranges_m.shape[0] == 0
        assert not np.any(np.isnan(rays.directions))


class TestSamplingStride:
    def test_stride_two_keeps_deterministic_grid_subset(self):
        points = np.zeros((4, 4, 3), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                points[r, c] = [float(r) + 1.0, float(c) + 1.0, 1.0]
        cloud = _make_cloud(points)

        rays = build_free_space_rays(cloud, _ORIGIN, stride=2)

        assert rays.ranges_m.shape[0] == 4  # (0,0),(0,2),(2,0),(2,2)

    def test_stride_is_deterministic_across_calls(self):
        rng = np.random.default_rng(2)
        points = rng.uniform(1.0, 3.0, size=(6, 7, 3)).astype(np.float32)
        cloud = _make_cloud(points)

        rays1 = build_free_space_rays(cloud, _ORIGIN, stride=3)
        rays2 = build_free_space_rays(cloud, _ORIGIN, stride=3)

        np.testing.assert_array_equal(rays1.ranges_m, rays2.ranges_m)
        np.testing.assert_array_equal(rays1.directions, rays2.directions)

    def test_rejects_stride_below_one(self):
        cloud = _make_cloud(np.full((1, 1, 3), 1.0, dtype=np.float32))
        with pytest.raises(ValueError, match="stride"):
            build_free_space_rays(cloud, _ORIGIN, stride=0)


class TestFramePreservation:
    def test_output_frame_id_matches_input(self):
        cloud = _make_cloud(np.full((1, 1, 3), 1.0, dtype=np.float32))
        rays = build_free_space_rays(cloud, _ORIGIN)
        assert rays.frame_id == FrameId.BODY

    def test_generic_not_hardcoded_to_body(self):
        points = np.full((1, 1, 3), 1.0, dtype=np.float32)
        cloud = PointCloud(points=points, frame_id="some_other_frame", valid_mask=np.ones((1, 1), dtype=bool))
        rays = build_free_space_rays(cloud, _ORIGIN)
        assert rays.frame_id == "some_other_frame"

    def test_no_timestamp_or_confidence_field_on_free_space_rays(self):
        """Frozen contract, reported not silently resolved — see
        docs/DATA_CONTRACTS.md's E5 section."""
        assert "timestamp" not in FreeSpaceRays.__dataclass_fields__
        assert "confidence" not in FreeSpaceRays.__dataclass_fields__


class TestOutputDtype:
    def test_all_arrays_float32(self):
        cloud = _make_cloud(np.full((2, 2, 3), 1.0, dtype=np.float32))
        rays = build_free_space_rays(cloud, _ORIGIN)
        assert rays.origins.dtype == np.float32
        assert rays.directions.dtype == np.float32
        assert rays.ranges_m.dtype == np.float32


class TestInputImmutability:
    def test_source_cloud_not_mutated(self):
        points = np.zeros((2, 2, 3), dtype=np.float32)
        points[0, 0] = [1.0, 2.0, 3.0]
        original = points.copy()
        cloud = _make_cloud(points)

        build_free_space_rays(cloud, _ORIGIN, stride=2)

        np.testing.assert_array_equal(cloud.points, original)
