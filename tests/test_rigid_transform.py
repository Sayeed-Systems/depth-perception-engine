"""
Unit tests for geometry.rigid_transform.transform_point_cloud — Level 3,
Phase E4.

Analytical, independent of OpenCV (nothing in rigid_transform.py calls
cv2) — every expected value below is hand-computed from the standard
rotation-matrix formulas, not derived from the function under test itself.
"""

import numpy as np
import pytest

from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import PointCloud, transform_point_cloud

_TOL = 1e-5


def _make_cloud(points_hw3: np.ndarray, valid_mask=None, timestamp=None, confidence=None) -> PointCloud:
    h, w, _ = points_hw3.shape
    if valid_mask is None:
        valid_mask = ~np.isnan(points_hw3).any(axis=-1)
    return PointCloud(
        points=points_hw3.astype(np.float32),
        frame_id=FrameId.CAMERA_OPTICAL_LEFT,
        valid_mask=valid_mask,
        confidence=confidence,
        timestamp=timestamp,
    )


def _single_point_cloud(xyz, **kwargs) -> PointCloud:
    points = np.zeros((1, 1, 3), dtype=np.float32)
    points[0, 0] = xyz
    return _make_cloud(points, **kwargs)


def _rx90():
    """Rotation matrix, +90 degrees about X: (x, y, z) -> (x, -z, y)."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _ry90():
    """Rotation matrix, +90 degrees about Y: (x, y, z) -> (z, y, -x)."""
    return np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def _rz90():
    """Rotation matrix, +90 degrees about Z: (x, y, z) -> (-y, x, z)."""
    return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class TestIdentityTransform:
    def test_input_xyz_equals_output_xyz(self):
        cloud = _single_point_cloud([1.0, 2.0, 3.0])
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        np.testing.assert_allclose(out.points[0, 0], [1.0, 2.0, 3.0], atol=_TOL)


class TestPureTranslation:
    def test_known_xyz_plus_known_translation_equals_exact_expected(self):
        cloud = _single_point_cloud([1.0, -2.0, 3.5])
        translation = np.array([0.1, 0.2, -0.3])
        transform = RigidTransform(
            rotation=np.eye(3), translation=translation,
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        expected = np.array([1.1, -1.8, 3.2])
        np.testing.assert_allclose(out.points[0, 0], expected, atol=_TOL)


class TestNinetyDegreeRotations:
    @pytest.mark.parametrize(
        "rotation, xyz, expected",
        [
            (_rx90(), [1.0, 2.0, 3.0], [1.0, -3.0, 2.0]),
            (_ry90(), [1.0, 2.0, 3.0], [3.0, 2.0, -1.0]),
            (_rz90(), [1.0, 2.0, 3.0], [-2.0, 1.0, 3.0]),
        ],
        ids=["about_X", "about_Y", "about_Z"],
    )
    def test_rotation_only_matches_hand_computed_expectation(self, rotation, xyz, expected):
        cloud = _single_point_cloud(xyz)
        transform = RigidTransform(
            rotation=rotation, translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        np.testing.assert_allclose(out.points[0, 0], expected, atol=_TOL)


class TestCombinedRotationAndTranslation:
    def test_rotation_then_translation_matches_hand_computed_expectation(self):
        xyz = [1.0, 2.0, 3.0]
        translation = np.array([10.0, -5.0, 0.5])
        transform = RigidTransform(
            rotation=_rz90(), translation=translation,
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )
        cloud = _single_point_cloud(xyz)

        out = transform_point_cloud(cloud, transform)

        # R @ [1,2,3] = [-2, 1, 3]; + translation = [8, -4, 3.5]
        expected = np.array([8.0, -4.0, 3.5])
        np.testing.assert_allclose(out.points[0, 0], expected, atol=_TOL)

    def test_multiple_points_in_one_organized_cloud(self):
        """Confirms the vectorized reshape+matmul is applied correctly
        per-pixel, not just for a trivial 1x1 cloud."""
        points = np.array(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
             [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]],
            dtype=np.float32,
        )
        cloud = _make_cloud(points)
        translation = np.array([1.0, 1.0, 1.0])
        transform = RigidTransform(
            rotation=_rz90(), translation=translation,
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        # Rz90 @ [x,y,z] = [-y, x, z]
        expected = np.array(
            [[[1.0, 2.0, 1.0], [0.0, 1.0, 1.0]],
             [[1.0, 1.0, 2.0], [0.0, 2.0, 2.0]]],
        )
        np.testing.assert_allclose(out.points, expected, atol=_TOL)


class TestInverseRoundTrip:
    def test_forward_then_inverse_recovers_original_points(self):
        """RigidTransform has no .inverse() method (not part of the frozen
        E1 contract — not added here either, since that would extend a
        frozen type beyond this task's scope). The mathematical inverse
        of a rigid transform (rotation R, translation t) is
        (R.T, -R.T @ t) — computed directly in the test, not via any new
        library method."""
        rng = np.random.default_rng(3)
        points = rng.uniform(-5.0, 5.0, size=(4, 5, 3)).astype(np.float32)
        cloud = _make_cloud(points)

        angle = np.deg2rad(37.0)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([0.3, -0.7, 1.2])

        forward = RigidTransform(
            rotation=rotation, translation=translation,
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )
        inverse = RigidTransform(
            rotation=rotation.T, translation=-rotation.T @ translation,
            from_frame=FrameId.BODY, to_frame=FrameId.CAMERA_OPTICAL_LEFT,
        )

        body_cloud = transform_point_cloud(cloud, forward)
        recovered_cloud = transform_point_cloud(body_cloud, inverse)

        np.testing.assert_allclose(recovered_cloud.points, cloud.points, atol=1e-4)
        assert recovered_cloud.frame_id == FrameId.CAMERA_OPTICAL_LEFT


class TestShapeAndMaskPreservation:
    def test_organized_shape_is_preserved(self):
        points = np.zeros((7, 9, 3), dtype=np.float32)
        cloud = _make_cloud(points)
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        assert out.points.shape == (7, 9, 3)
        assert out.points.dtype == np.float32

    def test_valid_mask_preserved_exactly_but_not_aliased(self):
        points = np.zeros((3, 3, 3), dtype=np.float32)
        valid_mask = np.array([[True, False, True]] * 3)
        cloud = _make_cloud(points, valid_mask=valid_mask)
        transform = RigidTransform(
            rotation=_rz90(), translation=np.array([1.0, 2.0, 3.0]),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        np.testing.assert_array_equal(out.valid_mask, valid_mask)
        assert out.valid_mask is not valid_mask, "must be a copy, not the same array object"

    def test_confidence_preserved_when_present(self):
        points = np.zeros((2, 2, 3), dtype=np.float32)
        confidence = np.full((2, 2), 0.7, dtype=np.float32)
        cloud = _make_cloud(points, confidence=confidence)
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        np.testing.assert_array_equal(out.confidence, confidence)
        assert out.confidence is not confidence

    def test_confidence_stays_none_when_absent(self):
        points = np.zeros((2, 2, 3), dtype=np.float32)
        cloud = _make_cloud(points, confidence=None)
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        assert out.confidence is None


class TestInvalidPointsStayInvalid:
    def test_nan_points_remain_nan_after_transform(self):
        points = np.full((3, 3, 3), np.nan, dtype=np.float32)
        points[1, 1] = [1.0, 2.0, 3.0]
        valid_mask = np.zeros((3, 3), dtype=bool)
        valid_mask[1, 1] = True
        cloud = _make_cloud(points, valid_mask=valid_mask)
        transform = RigidTransform(
            rotation=_rx90(), translation=np.array([5.0, 5.0, 5.0]),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        assert np.all(np.isnan(out.points[~out.valid_mask]))
        assert np.all(np.isfinite(out.points[out.valid_mask]))

    def test_finite_valid_input_produces_finite_valid_output(self):
        rng = np.random.default_rng(11)
        points = rng.uniform(-10.0, 10.0, size=(5, 6, 3)).astype(np.float32)
        cloud = _make_cloud(points)  # all-finite -> valid_mask all True
        transform = RigidTransform(
            rotation=_ry90(), translation=np.array([1.0, -1.0, 2.0]),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        assert np.all(out.valid_mask)
        assert np.all(np.isfinite(out.points))


class TestFrameIdAndTimestamp:
    def test_frame_id_set_to_transform_to_frame(self):
        cloud = _single_point_cloud([0.0, 0.0, 0.0])
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        assert out.frame_id == FrameId.BODY
        assert out.frame_id != cloud.frame_id

    def test_frame_id_is_generic_not_hardcoded_to_body(self):
        """transform_point_cloud makes no special assumption about BODY —
        any to_frame string is honored, proving the implementation is
        frame-agnostic (Task 3: no assumption about rig mounting)."""
        cloud = _single_point_cloud([0.0, 0.0, 0.0])
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame="some_other_frame",
        )

        out = transform_point_cloud(cloud, transform)

        assert out.frame_id == "some_other_frame"

    def test_timestamp_preserved_unchanged(self):
        cloud = _single_point_cloud([0.0, 0.0, 0.0], timestamp=123.456)
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        assert out.timestamp == 123.456

    def test_timestamp_none_stays_none(self):
        cloud = _single_point_cloud([0.0, 0.0, 0.0], timestamp=None)
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)

        assert out.timestamp is None


class TestDoesNotMutateOriginal:
    def test_source_cloud_points_unchanged(self):
        points = np.zeros((2, 2, 3), dtype=np.float32)
        points[0, 0] = [1.0, 2.0, 3.0]
        original = points.copy()
        cloud = _make_cloud(points)
        transform = RigidTransform(
            rotation=_rz90(), translation=np.array([100.0, 100.0, 100.0]),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        transform_point_cloud(cloud, transform)

        np.testing.assert_array_equal(cloud.points, original)

    def test_source_cloud_valid_mask_unchanged(self):
        points = np.zeros((2, 2, 3), dtype=np.float32)
        valid_mask = np.array([[True, False], [False, True]])
        original_mask = valid_mask.copy()
        cloud = _make_cloud(points, valid_mask=valid_mask)
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out = transform_point_cloud(cloud, transform)
        out.valid_mask[0, 0] = not out.valid_mask[0, 0]  # mutate the OUTPUT

        np.testing.assert_array_equal(cloud.valid_mask, original_mask)  # source untouched


class TestDeterminism:
    def test_same_input_produces_identical_output(self):
        rng = np.random.default_rng(5)
        points = rng.uniform(-3.0, 3.0, size=(4, 4, 3)).astype(np.float32)
        cloud = _make_cloud(points)
        transform = RigidTransform(
            rotation=_rx90(), translation=np.array([1.0, 2.0, 3.0]),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        out1 = transform_point_cloud(cloud, transform)
        out2 = transform_point_cloud(cloud, transform)

        np.testing.assert_array_equal(out1.points, out2.points)
        np.testing.assert_array_equal(out1.valid_mask, out2.valid_mask)


class TestInputValidation:
    def test_rejects_non_point_cloud(self):
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )
        with pytest.raises(TypeError):
            transform_point_cloud("not a cloud", transform)

    def test_rejects_non_rigid_transform(self):
        cloud = _single_point_cloud([0.0, 0.0, 0.0])
        with pytest.raises(TypeError):
            transform_point_cloud(cloud, "not a transform")

    def test_rejects_mismatched_from_frame(self):
        cloud = _single_point_cloud([0.0, 0.0, 0.0])
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.BODY, to_frame="somewhere_else",
        )
        with pytest.raises(ValueError, match="does not match"):
            transform_point_cloud(cloud, transform)

    def test_rejects_non_finite_rotation(self):
        cloud = _single_point_cloud([0.0, 0.0, 0.0])
        rotation = np.eye(3)
        rotation[0, 0] = np.nan
        transform = RigidTransform(
            rotation=rotation, translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )
        with pytest.raises(ValueError, match="non-finite"):
            transform_point_cloud(cloud, transform)

    def test_rejects_non_finite_translation(self):
        cloud = _single_point_cloud([0.0, 0.0, 0.0])
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.array([np.inf, 0.0, 0.0]),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )
        with pytest.raises(ValueError, match="non-finite"):
            transform_point_cloud(cloud, transform)
