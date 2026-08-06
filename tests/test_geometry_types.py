"""
Unit tests for geometry.types — PointCloud, ObstacleCloud, FreeSpaceRays,
GeometryMetrics.

These are interface-only contracts (Level 3, Phase E1) — no producer
exists yet, so these tests only prove the types construct correctly with
their documented shapes/defaults, not any geometric behavior.
"""

import numpy as np

from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry import (
    FreeSpaceRays,
    GeometryMetrics,
    ObstacleCloud,
    PointCloud,
)


class TestPointCloud:
    def test_construction_with_required_fields_only(self):
        points = np.full((10, 12, 3), np.nan, dtype=np.float32)
        valid_mask = np.zeros((10, 12), dtype=bool)

        pc = PointCloud(points=points, frame_id=FrameId.CAMERA_OPTICAL_LEFT, valid_mask=valid_mask)

        assert pc.points.shape == (10, 12, 3)
        assert pc.valid_mask.shape == (10, 12)
        assert pc.frame_id == FrameId.CAMERA_OPTICAL_LEFT
        assert pc.confidence is None
        assert pc.timestamp is None

    def test_construction_with_optional_fields(self):
        points = np.zeros((4, 4, 3), dtype=np.float32)
        valid_mask = np.ones((4, 4), dtype=bool)
        confidence = np.full((4, 4), 0.5, dtype=np.float32)

        pc = PointCloud(
            points=points, frame_id=FrameId.CAMERA_OPTICAL_LEFT,
            valid_mask=valid_mask, confidence=confidence, timestamp=123.0,
        )

        assert pc.confidence is confidence
        assert pc.timestamp == 123.0

    def test_is_frozen(self):
        pc = PointCloud(
            points=np.zeros((1, 1, 3), dtype=np.float32),
            frame_id=FrameId.CAMERA_OPTICAL_LEFT,
            valid_mask=np.zeros((1, 1), dtype=bool),
        )
        try:
            pc.frame_id = "other"
            assert False, "expected frozen dataclass to reject mutation"
        except AttributeError:
            pass


class TestObstacleCloud:
    def test_construction_minimal(self):
        oc = ObstacleCloud(points=np.zeros((7, 3), dtype=np.float32), frame_id=FrameId.CAMERA_OPTICAL_LEFT)
        assert oc.points.shape == (7, 3)
        assert oc.distances_m is None
        assert oc.confidence is None

    def test_construction_with_distances_and_confidence(self):
        oc = ObstacleCloud(
            points=np.zeros((3, 3), dtype=np.float32),
            frame_id=FrameId.CAMERA_OPTICAL_LEFT,
            distances_m=np.array([0.5, 1.0, 1.5], dtype=np.float32),
            confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        )
        assert oc.distances_m.shape == (3,)
        assert oc.confidence.shape == (3,)


class TestFreeSpaceRays:
    def test_construction(self):
        n = 20
        rays = FreeSpaceRays(
            origins=np.zeros((n, 3), dtype=np.float32),
            directions=np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
            ranges_m=np.full(n, 8.0, dtype=np.float32),
            frame_id=FrameId.CAMERA_OPTICAL_LEFT,
        )
        assert rays.origins.shape == (n, 3)
        assert rays.directions.shape == (n, 3)
        assert rays.ranges_m.shape == (n,)


class TestGeometryMetrics:
    def test_construction_with_values(self):
        gm = GeometryMetrics(
            min_obstacle_distance_m=0.6, mean_free_space_m=2.3,
            point_count=1000, valid_fraction=0.42,
        )
        assert gm.min_obstacle_distance_m == 0.6
        assert gm.point_count == 1000

    def test_construction_with_no_obstacles_or_rays(self):
        gm = GeometryMetrics(
            min_obstacle_distance_m=None, mean_free_space_m=None,
            point_count=0, valid_fraction=0.0,
        )
        assert gm.min_obstacle_distance_m is None
        assert gm.mean_free_space_m is None
