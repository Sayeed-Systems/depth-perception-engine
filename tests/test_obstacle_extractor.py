"""
Unit tests for geometry.obstacle_extractor.build_obstacle_cloud — Level 3,
Phase E5.

Synthetic geometry only — no stereo matching, no calibration file, no
disparity. Each test constructs a PointCloud directly, matching the style
already established in tests/test_rigid_transform.py.
"""

import numpy as np
import pytest

from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry import ObstacleCloud, PointCloud, build_obstacle_cloud

_ORIGIN = np.array([0.0, 0.0, 0.0])


def _make_cloud(points_hw3: np.ndarray, valid_mask=None, confidence=None) -> PointCloud:
    if valid_mask is None:
        valid_mask = ~np.isnan(points_hw3).any(axis=-1)
    return PointCloud(
        points=points_hw3.astype(np.float32),
        frame_id=FrameId.BODY,
        valid_mask=valid_mask,
        confidence=confidence,
    )


class TestSingleKnownPoint:
    def test_single_valid_point_produces_one_obstacle_point(self):
        points = np.full((1, 1, 3), np.nan, dtype=np.float32)
        points[0, 0] = [1.0, 2.0, 3.0]
        cloud = _make_cloud(points)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)

        assert isinstance(oc, ObstacleCloud)
        assert oc.points.shape == (1, 3)
        np.testing.assert_allclose(oc.points[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(oc.distances_m[0], np.sqrt(1.0 + 4.0 + 9.0), atol=1e-5)
        assert oc.frame_id == FrameId.BODY


class TestFlatPlane:
    def test_flat_plane_of_known_depth_all_points_included(self):
        h, w = 4, 5
        points = np.zeros((h, w, 3), dtype=np.float32)
        u, v = np.meshgrid(np.arange(w) - w / 2, np.arange(h) - h / 2)
        points[:, :, 0] = u
        points[:, :, 1] = v
        points[:, :, 2] = 2.0  # constant depth plane at Z = 2.0
        cloud = _make_cloud(points)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)

        assert oc.points.shape[0] == h * w
        np.testing.assert_allclose(oc.points[:, 2], 2.0)


class TestInvalidPoints:
    def test_all_invalid_produces_zero_obstacle_points(self):
        points = np.full((3, 3, 3), np.nan, dtype=np.float32)
        valid_mask = np.zeros((3, 3), dtype=bool)
        cloud = _make_cloud(points, valid_mask=valid_mask)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)

        assert oc.points.shape == (0, 3)
        assert oc.distances_m.shape == (0,)


class TestMixedValidInvalid:
    def test_only_valid_points_are_represented(self):
        points = np.full((2, 3, 3), np.nan, dtype=np.float32)
        points[0, 0] = [1.0, 0.0, 0.0]
        points[1, 2] = [0.0, 5.0, 0.0]
        valid_mask = np.zeros((2, 3), dtype=bool)
        valid_mask[0, 0] = True
        valid_mask[1, 2] = True
        cloud = _make_cloud(points, valid_mask=valid_mask)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)

        assert oc.points.shape[0] == 2
        found = {tuple(p) for p in oc.points.tolist()}
        assert found == {(1.0, 0.0, 0.0), (0.0, 5.0, 0.0)}


class TestNaNInfNeverBecomeObstacles:
    def test_nan_marked_invalid_excluded(self):
        points = np.zeros((2, 2, 3), dtype=np.float32)
        points[0, 0] = [np.nan, np.nan, np.nan]
        points[0, 1] = [1.0, 1.0, 1.0]
        points[1, 0] = [2.0, 2.0, 2.0]
        points[1, 1] = [3.0, 3.0, 3.0]
        # valid_mask deliberately does NOT match isnan automatically here —
        # constructed explicitly to prove build_obstacle_cloud trusts
        # valid_mask, not its own isnan re-derivation, matching
        # PointCloud's own documented contract (valid_mask IS the source
        # of truth, not something callers should re-derive).
        valid_mask = np.array([[False, True], [True, True]])
        cloud = _make_cloud(points, valid_mask=valid_mask)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)

        assert oc.points.shape[0] == 3
        assert not np.any(np.isnan(oc.points))

    def test_inf_point_marked_valid_is_excluded_by_range_not_by_crashing(self):
        """A pathological case (Inf marked valid, contract violation by the
        caller) must not crash or silently pass through — Inf necessarily
        fails any finite max_range_m bound."""
        points = np.zeros((1, 1, 3), dtype=np.float32)
        points[0, 0] = [np.inf, 0.0, 0.0]
        valid_mask = np.ones((1, 1), dtype=bool)
        cloud = _make_cloud(points, valid_mask=valid_mask)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)

        assert oc.points.shape[0] == 0


class TestRangeClipping:
    def test_points_outside_max_range_excluded(self):
        points = np.zeros((1, 3, 3), dtype=np.float32)
        points[0, 0] = [0.0, 0.0, 1.0]    # range 1.0 -> in
        points[0, 1] = [0.0, 0.0, 5.0]    # range 5.0 -> in
        points[0, 2] = [0.0, 0.0, 20.0]   # range 20.0 -> excluded
        cloud = _make_cloud(points)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=10.0)

        assert oc.points.shape[0] == 2
        assert 20.0 not in oc.points[:, 2].tolist()

    def test_points_below_min_range_excluded(self):
        points = np.zeros((1, 2, 3), dtype=np.float32)
        points[0, 0] = [0.0, 0.0, 0.1]   # range 0.1 -> excluded (below min)
        points[0, 1] = [0.0, 0.0, 1.0]   # range 1.0 -> included
        cloud = _make_cloud(points)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.5, max_range_m=10.0)

        assert oc.points.shape[0] == 1
        np.testing.assert_allclose(oc.points[0], [0.0, 0.0, 1.0])

    def test_boundary_range_values_are_inclusive(self):
        points = np.zeros((1, 2, 3), dtype=np.float32)
        points[0, 0] = [0.0, 0.0, 1.0]    # exactly min
        points[0, 1] = [0.0, 0.0, 10.0]   # exactly max
        cloud = _make_cloud(points)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=1.0, max_range_m=10.0)

        assert oc.points.shape[0] == 2

    def test_rejects_min_greater_than_max(self):
        cloud = _make_cloud(np.zeros((1, 1, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="min_range_m"):
            build_obstacle_cloud(cloud, _ORIGIN, min_range_m=5.0, max_range_m=1.0)


class TestSamplingStride:
    def test_stride_two_keeps_deterministic_grid_subset(self):
        points = np.zeros((4, 4, 3), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                points[r, c] = [float(r), float(c), 1.0]
        cloud = _make_cloud(points)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0, stride=2)

        expected_rc = {(0, 0), (0, 2), (2, 0), (2, 2)}
        found_rc = {(int(p[0]), int(p[1])) for p in oc.points}
        assert found_rc == expected_rc

    def test_stride_is_deterministic_across_calls(self):
        rng = np.random.default_rng(1)
        points = rng.uniform(-2.0, 2.0, size=(6, 7, 3)).astype(np.float32)
        cloud = _make_cloud(points)

        oc1 = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0, stride=3)
        oc2 = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0, stride=3)

        np.testing.assert_array_equal(oc1.points, oc2.points)

    def test_rejects_stride_below_one(self):
        cloud = _make_cloud(np.zeros((1, 1, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="stride"):
            build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=1.0, stride=0)


class TestFramePreservation:
    def test_output_frame_id_matches_input(self):
        cloud = _make_cloud(np.full((1, 1, 3), 1.0, dtype=np.float32))
        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)
        assert oc.frame_id == FrameId.BODY

    def test_generic_not_hardcoded_to_body(self):
        points = np.full((1, 1, 3), 1.0, dtype=np.float32)
        cloud = PointCloud(points=points, frame_id="some_other_frame", valid_mask=np.ones((1, 1), dtype=bool))
        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)
        assert oc.frame_id == "some_other_frame"

    def test_no_timestamp_field_on_obstacle_cloud(self):
        """Frozen contract mismatch, reported not silently resolved — see
        docs/DATA_CONTRACTS.md's E5 section: ObstacleCloud has no
        timestamp field."""
        assert not hasattr(ObstacleCloud, "timestamp")
        assert "timestamp" not in ObstacleCloud.__dataclass_fields__


class TestConfidencePreservation:
    def test_confidence_filtered_and_preserved_when_present(self):
        points = np.zeros((1, 2, 3), dtype=np.float32)
        points[0, 0] = [0.0, 0.0, 1.0]
        points[0, 1] = [0.0, 0.0, 20.0]  # excluded by range
        confidence = np.array([[0.9, 0.2]], dtype=np.float32)
        cloud = _make_cloud(points, confidence=confidence)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=10.0)

        assert oc.confidence.shape == (1,)
        np.testing.assert_allclose(oc.confidence, [0.9])

    def test_confidence_stays_none_when_absent(self):
        cloud = _make_cloud(np.full((1, 1, 3), 1.0, dtype=np.float32))
        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)
        assert oc.confidence is None


class TestInputImmutability:
    def test_source_cloud_not_mutated(self):
        points = np.zeros((2, 2, 3), dtype=np.float32)
        points[0, 0] = [1.0, 2.0, 3.0]
        original = points.copy()
        cloud = _make_cloud(points)

        build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0, stride=2)

        np.testing.assert_array_equal(cloud.points, original)

    def test_output_not_aliased_to_source(self):
        points = np.zeros((1, 1, 3), dtype=np.float32)
        points[0, 0] = [1.0, 2.0, 3.0]
        cloud = _make_cloud(points)

        oc = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)
        oc.points[0, 0] = -999.0

        np.testing.assert_allclose(cloud.points[0, 0], [1.0, 2.0, 3.0])
