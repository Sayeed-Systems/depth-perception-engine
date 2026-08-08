"""
Unit tests for geometry.point_cloud_builder.PointCloudBuilder — Level 3,
Phase E2.

The underlying reprojection math (and its own analytic/cv2 cross-checks,
invalid-disparity rejection rules, determinism) is already covered in
tests/test_depth_estimator.py::TestEstimatePointCloud — this file does not
re-derive that math. It covers exactly what PointCloudBuilder itself adds
on top: construction/calibration validation, the 0.0-to-NaN conversion,
frame_id/timestamp/confidence contract, and that PointCloud's own
`points`/`valid_mask` agree with what DepthEstimator.estimate_point_cloud
independently produces for the same input.
"""

import cv2
import numpy as np
import pytest

from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry import PointCloud, PointCloudBuilder


def _make_q(focal_length_px=600.0, baseline_m=0.065, cx=160.0, cy=120.0, q33=0.0):
    tx = 1.0 / (baseline_m * 1000.0)
    return np.array(
        [
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, focal_length_px],
            [0.0, 0.0, tx, q33],
        ],
        dtype=np.float64,
    )


def _mixed_disparity_map(height=24, width=32):
    rng = np.random.default_rng(5)
    disp = rng.uniform(1.0, 120.0, size=(height, width)).astype(np.float32)
    disp[0:3, :] = 0.0
    disp[3:6, :] = -2.0
    disp[6, 0] = np.nan
    disp[6, 1] = np.inf
    return disp


class TestConstruction:
    def test_from_Q_directly(self):
        builder = PointCloudBuilder(_make_q())
        assert builder.focal_length_px == pytest.approx(600.0)
        assert builder.baseline_m == pytest.approx(0.065)

    def test_from_calibration(self, calibration):
        builder = PointCloudBuilder.from_calibration(calibration)
        assert builder.Q is calibration.Q or np.array_equal(builder.Q, calibration.Q)
        assert builder.focal_length_px == pytest.approx(614.5, abs=1.0)

    def test_rejects_non_ndarray_Q(self):
        with pytest.raises(TypeError):
            PointCloudBuilder([[1, 0], [0, 1]])

    def test_rejects_wrong_shape_Q(self):
        with pytest.raises(ValueError):
            PointCloudBuilder(np.eye(3))

    def test_rejects_nan_in_Q(self):
        Q = _make_q()
        Q[0, 3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            PointCloudBuilder(Q)

    def test_rejects_inf_in_Q(self):
        Q = _make_q()
        Q[2, 3] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            PointCloudBuilder(Q)

    def test_repr_contains_focal_length_and_baseline(self):
        builder = PointCloudBuilder(_make_q(focal_length_px=614.5, baseline_m=0.0647))
        text = repr(builder)
        assert "614.5" in text
        assert "0.0647" in text


class TestBuildContract:
    def test_returns_point_cloud_with_camera_optical_left_frame(self):
        builder = PointCloudBuilder(_make_q())
        disp = np.full((10, 12), 50.0, dtype=np.float32)

        pc = builder.build(disp)

        assert isinstance(pc, PointCloud)
        assert pc.frame_id == FrameId.CAMERA_OPTICAL_LEFT

    def test_points_and_valid_mask_shapes_and_dtypes(self):
        builder = PointCloudBuilder(_make_q())
        disp = _mixed_disparity_map(height=15, width=21)

        pc = builder.build(disp)

        assert pc.points.shape == (15, 21, 3)
        assert pc.points.dtype == np.float32
        assert pc.valid_mask.shape == (15, 21)
        assert pc.valid_mask.dtype == np.bool_

    def test_confidence_is_none(self):
        builder = PointCloudBuilder(_make_q())
        pc = builder.build(np.full((4, 4), 50.0, dtype=np.float32))
        assert pc.confidence is None

    def test_timestamp_passthrough(self):
        builder = PointCloudBuilder(_make_q())
        pc = builder.build(np.full((4, 4), 50.0, dtype=np.float32), timestamp=123.456)
        assert pc.timestamp == 123.456

    def test_timestamp_defaults_to_none(self):
        builder = PointCloudBuilder(_make_q())
        pc = builder.build(np.full((4, 4), 50.0, dtype=np.float32))
        assert pc.timestamp is None

    def test_is_frozen_dataclass(self):
        builder = PointCloudBuilder(_make_q())
        pc = builder.build(np.full((4, 4), 50.0, dtype=np.float32))
        with pytest.raises(AttributeError):
            pc.frame_id = "other"

    def test_rejects_non_ndarray_disparity(self):
        builder = PointCloudBuilder(_make_q())
        with pytest.raises(TypeError):
            builder.build([[1, 2], [3, 4]])

    def test_rejects_wrong_ndim_disparity(self):
        builder = PointCloudBuilder(_make_q())
        with pytest.raises(ValueError):
            builder.build(np.zeros(10, dtype=np.float32))


class TestNaNInvalidConversion:
    """
    PointCloud uses NaN for "no data" (not 0.0 — see geometry/types.py's
    module docstring: (0, 0, 0) is a legitimate point on the optical
    axis). DepthEstimator.estimate_point_cloud uses 0.0. This is the one
    real conversion PointCloudBuilder performs.
    """

    def test_invalid_pixels_are_nan_not_zero(self):
        builder = PointCloudBuilder(_make_q())
        disp = _mixed_disparity_map()

        pc = builder.build(disp)

        invalid = ~pc.valid_mask
        assert np.any(invalid), "fixture must contain at least one invalid pixel"
        assert np.all(np.isnan(pc.points[invalid]))

    def test_valid_pixels_are_never_nan(self):
        builder = PointCloudBuilder(_make_q())
        disp = _mixed_disparity_map()

        pc = builder.build(disp)

        assert np.any(pc.valid_mask), "fixture must contain at least one valid pixel"
        assert not np.any(np.isnan(pc.points[pc.valid_mask]))

    def test_a_point_legitimately_at_the_origin_is_not_confused_with_invalid(self):
        """
        The exact motivation documented in geometry/types.py: (0, 0, 0) is
        a real point (directly on the optical axis) and must be
        distinguishable from "no data" — which it is, here, only because
        invalid pixels are NaN, not 0.0. This test constructs a disparity
        map whose only valid pixel projects to (X, Y, Z) = (0, 0, Z) (u=cx,
        v=cy) and confirms it reports valid=True with finite coordinates,
        not confused with the surrounding invalid NaN pixels.
        """
        cx, cy = 8.0, 6.0
        Q = _make_q(focal_length_px=600.0, baseline_m=0.065, cx=cx, cy=cy)
        builder = PointCloudBuilder(Q)
        disp = np.zeros((12, 16), dtype=np.float32)  # all invalid (zero disparity)
        disp[int(cy), int(cx)] = 50.0  # the one valid pixel, on the optical axis

        pc = builder.build(disp)

        assert pc.valid_mask[int(cy), int(cx)]
        x, y, _z = pc.points[int(cy), int(cx)]
        np.testing.assert_allclose([x, y], [0.0, 0.0], atol=1e-4)
        assert np.all(np.isnan(pc.points[~pc.valid_mask]))

    def test_points_array_is_a_fresh_copy_not_aliased_across_calls(self):
        builder = PointCloudBuilder(_make_q())
        disp = np.full((6, 6), 50.0, dtype=np.float32)

        pc1 = builder.build(disp)
        pc2 = builder.build(disp)
        pc1.points[0, 0, 0] = -999.0

        assert pc2.points[0, 0, 0] != -999.0


class TestMatchesDepthEstimatorAndOpenCV:
    """Cross-check: PointCloud must carry exactly what the underlying
    (already independently math-validated — see test_depth_estimator.py)
    DepthEstimator.estimate_point_cloud produces, just NaN- instead of
    0.0-converted."""

    def test_matches_estimate_point_cloud_up_to_nan_conversion(self, calibration):
        disp = _mixed_disparity_map()
        estimator = DepthEstimator(calibration.Q)
        builder = PointCloudBuilder(calibration.Q)

        raw_points, raw_valid = estimator.estimate_point_cloud(disp)
        pc = builder.build(disp)

        np.testing.assert_array_equal(pc.valid_mask, raw_valid)
        np.testing.assert_allclose(pc.points[raw_valid], raw_points[raw_valid])
        assert np.all(np.isnan(pc.points[~raw_valid]))

    def test_matches_raw_opencv_reprojection_on_random_valid_disparities(self, calibration):
        rng = np.random.default_rng(2024)
        disp = rng.uniform(2.0, 140.0, size=(30, 40)).astype(np.float32)

        builder = PointCloudBuilder(calibration.Q)
        pc = builder.build(disp)

        ref = cv2.reprojectImageTo3D(disp, calibration.Q, handleMissingValues=False) / 1000.0
        expected_valid = (
            (ref[:, :, 2] >= DepthEstimator.MIN_DEPTH_M)
            & (ref[:, :, 2] <= DepthEstimator.MAX_DEPTH_M)
            & np.all(np.isfinite(ref), axis=-1)
        )

        np.testing.assert_array_equal(pc.valid_mask, expected_valid)
        np.testing.assert_allclose(
            pc.points[expected_valid], ref[expected_valid].astype(np.float32), rtol=1e-4, atol=1e-5,
        )


class TestDeterminism:
    def test_same_input_produces_identical_output(self, calibration):
        builder = PointCloudBuilder(calibration.Q)
        disp = _mixed_disparity_map()

        pc1 = builder.build(disp)
        pc2 = builder.build(disp)

        np.testing.assert_array_equal(pc1.points, pc2.points)
        np.testing.assert_array_equal(pc1.valid_mask, pc2.valid_mask)
        assert pc1.frame_id == pc2.frame_id
