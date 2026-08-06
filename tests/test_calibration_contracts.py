"""
Unit tests for calibration.contracts — CameraModel, CameraIntrinsics,
StereoExtrinsics, RectificationParameters, RigCalibration.

These are read-only views over an existing StereoCalibration; the point of
these tests is proving the extraction is correct and that the derivation
doesn't silently diverge from DepthEstimator's own — not testing any new
algorithm, since none exists here.
"""

import numpy as np
import pytest

from depth_perception_engine.calibration.contracts import (
    CameraIntrinsics,
    CameraModel,
    RectificationParameters,
    RigCalibration,
    StereoExtrinsics,
)
from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.frames import FrameId, RigidTransform


class TestCameraIntrinsics:
    def test_left_side_extraction(self, calibration):
        intr = CameraIntrinsics.from_calibration(calibration, side="left")
        np.testing.assert_array_equal(intr.camera_matrix, calibration.camera_matrix_left)
        np.testing.assert_array_equal(intr.dist_coeffs, calibration.dist_coeffs_left)
        assert intr.image_size == calibration.image_size
        assert intr.camera_model == CameraModel.PINHOLE

    def test_right_side_extraction(self, calibration):
        intr = CameraIntrinsics.from_calibration(calibration, side="right")
        np.testing.assert_array_equal(intr.camera_matrix, calibration.camera_matrix_right)
        np.testing.assert_array_equal(intr.dist_coeffs, calibration.dist_coeffs_right)

    def test_invalid_side_raises(self, calibration):
        with pytest.raises(ValueError):
            CameraIntrinsics.from_calibration(calibration, side="center")


class TestStereoExtrinsics:
    def test_matches_depth_estimators_own_derivation(self, calibration):
        ext = StereoExtrinsics.from_calibration(calibration)
        estimator = DepthEstimator.from_calibration(calibration)

        assert ext.baseline_m == estimator.baseline_m
        assert ext.focal_length_px == estimator.focal_length_px

    def test_baseline_is_positive_for_real_calibration(self, calibration):
        ext = StereoExtrinsics.from_calibration(calibration)
        assert ext.baseline_m > 0.0
        assert ext.focal_length_px > 0.0

    def test_zero_tx_yields_zero_baseline(self):
        Q = np.eye(4, dtype=np.float64)
        Q[2, 3] = 600.0
        Q[3, 2] = 0.0
        from depth_perception_engine.calibration.models import StereoCalibration

        calibration = StereoCalibration(
            image_size=(320, 240),
            camera_matrix_left=np.eye(3), dist_coeffs_left=np.zeros((1, 5)),
            camera_matrix_right=np.eye(3), dist_coeffs_right=np.zeros((1, 5)),
            R1=np.eye(3), R2=np.eye(3),
            P1=np.zeros((3, 4)), P2=np.zeros((3, 4)),
            Q=Q,
        )
        ext = StereoExtrinsics.from_calibration(calibration)
        assert ext.baseline_m == 0.0
        assert ext.focal_length_px == 600.0


class TestRectificationParameters:
    def test_extraction_matches_source(self, calibration):
        rect = RectificationParameters.from_calibration(calibration)
        np.testing.assert_array_equal(rect.R1, calibration.R1)
        np.testing.assert_array_equal(rect.R2, calibration.R2)
        np.testing.assert_array_equal(rect.P1, calibration.P1)
        np.testing.assert_array_equal(rect.P2, calibration.P2)
        np.testing.assert_array_equal(rect.Q, calibration.Q)


class TestRigCalibration:
    def test_default_has_no_body_transform(self, calibration):
        rig = RigCalibration(stereo=calibration)
        assert rig.body_T_camera_left is None
        assert rig.camera_frame_id == FrameId.CAMERA_OPTICAL_LEFT

    def test_valid_body_transform_accepted(self, calibration):
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )
        rig = RigCalibration(stereo=calibration, body_T_camera_left=transform)
        assert rig.body_T_camera_left is transform

    def test_body_transform_with_wrong_to_frame_raises(self, calibration):
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame="not_body",
        )
        with pytest.raises(ValueError):
            RigCalibration(stereo=calibration, body_T_camera_left=transform)

    def test_body_transform_with_mismatched_from_frame_raises(self, calibration):
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame="some_other_camera", to_frame=FrameId.BODY,
        )
        with pytest.raises(ValueError):
            RigCalibration(stereo=calibration, body_T_camera_left=transform)

    def test_custom_camera_frame_id_is_respected(self, calibration):
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame="camera_optical_rear", to_frame=FrameId.BODY,
        )
        rig = RigCalibration(
            stereo=calibration,
            body_T_camera_left=transform,
            camera_frame_id="camera_optical_rear",
        )
        assert rig.camera_frame_id == "camera_optical_rear"
