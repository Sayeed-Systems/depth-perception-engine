"""
Unit tests for geometry.geometry_metrics.build_geometry_metrics — Level 3,
Phase E5.
"""

import numpy as np
import pytest

from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry import (
    FreeSpaceRays,
    GeometryMetrics,
    ObstacleCloud,
    PointCloud,
    build_geometry_metrics,
)


def _cloud(valid_mask: np.ndarray) -> PointCloud:
    h, w = valid_mask.shape
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[~valid_mask] = np.nan
    return PointCloud(points=points, frame_id=FrameId.BODY, valid_mask=valid_mask)


class TestPointCountAndValidFraction:
    def test_precise_definitions(self):
        valid_mask = np.array([[True, True, False], [True, False, False]])
        cloud = _cloud(valid_mask)

        metrics = build_geometry_metrics(cloud, obstacle_cloud=None, free_space_rays=None)

        assert isinstance(metrics, GeometryMetrics)
        assert metrics.point_count == 3
        assert metrics.valid_fraction == 3 / 6

    def test_all_valid(self):
        cloud = _cloud(np.ones((2, 2), dtype=bool))
        metrics = build_geometry_metrics(cloud, None, None)
        assert metrics.point_count == 4
        assert metrics.valid_fraction == 1.0

    def test_all_invalid(self):
        cloud = _cloud(np.zeros((2, 2), dtype=bool))
        metrics = build_geometry_metrics(cloud, None, None)
        assert metrics.point_count == 0
        assert metrics.valid_fraction == 0.0


class TestMinObstacleDistance:
    def test_none_when_obstacle_cloud_is_none(self):
        cloud = _cloud(np.ones((2, 2), dtype=bool))
        metrics = build_geometry_metrics(cloud, obstacle_cloud=None, free_space_rays=None)
        assert metrics.min_obstacle_distance_m is None

    def test_none_when_obstacle_cloud_is_empty(self):
        cloud = _cloud(np.ones((2, 2), dtype=bool))
        empty_oc = ObstacleCloud(
            points=np.zeros((0, 3), dtype=np.float32), frame_id=FrameId.BODY,
            distances_m=np.zeros((0,), dtype=np.float32),
        )
        metrics = build_geometry_metrics(cloud, obstacle_cloud=empty_oc, free_space_rays=None)
        assert metrics.min_obstacle_distance_m is None

    def test_exact_minimum_of_distances(self):
        cloud = _cloud(np.ones((2, 2), dtype=bool))
        oc = ObstacleCloud(
            points=np.zeros((3, 3), dtype=np.float32), frame_id=FrameId.BODY,
            distances_m=np.array([2.5, 0.8, 4.1], dtype=np.float32),
        )
        metrics = build_geometry_metrics(cloud, obstacle_cloud=oc, free_space_rays=None)
        assert metrics.min_obstacle_distance_m == pytest.approx(0.8)


class TestMeanFreeSpace:
    def test_none_when_rays_is_none(self):
        cloud = _cloud(np.ones((2, 2), dtype=bool))
        metrics = build_geometry_metrics(cloud, obstacle_cloud=None, free_space_rays=None)
        assert metrics.mean_free_space_m is None

    def test_none_when_rays_is_empty(self):
        cloud = _cloud(np.ones((2, 2), dtype=bool))
        empty_rays = FreeSpaceRays(
            origins=np.zeros((0, 3), dtype=np.float32),
            directions=np.zeros((0, 3), dtype=np.float32),
            ranges_m=np.zeros((0,), dtype=np.float32),
            frame_id=FrameId.BODY,
        )
        metrics = build_geometry_metrics(cloud, obstacle_cloud=None, free_space_rays=empty_rays)
        assert metrics.mean_free_space_m is None

    def test_exact_mean_of_ranges(self):
        cloud = _cloud(np.ones((2, 2), dtype=bool))
        rays = FreeSpaceRays(
            origins=np.zeros((3, 3), dtype=np.float32),
            directions=np.zeros((3, 3), dtype=np.float32),
            ranges_m=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            frame_id=FrameId.BODY,
        )
        metrics = build_geometry_metrics(cloud, obstacle_cloud=None, free_space_rays=rays)
        assert metrics.mean_free_space_m == pytest.approx(2.0)
