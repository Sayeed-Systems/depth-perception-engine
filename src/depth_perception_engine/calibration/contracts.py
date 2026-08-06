"""
Calibration contract decomposition — Level 3 (Phase E1).

StereoCalibration (models.py) is not modified by this module — it stays
the one flat, validated bag RectificationEngine/DepthEstimator actually
consume today, unchanged. Everything here is a read-only *view*: each type
below has a `.from_calibration(calibration)` classmethod that extracts its
fields from an existing StereoCalibration — pure extraction, no new math,
no change to any existing consumer's behavior.

Why decompose at all, if StereoCalibration already works: the intrinsic,
extrinsic, and rectification-specific pieces of a calibration are
conceptually distinct (a future multi-camera rig might share intrinsics
across cameras but not extrinsics, for instance), and baseline/focal-length
today only exist as private, undocumented derived state inside
DepthEstimator.__init__ — StereoExtrinsics exposes that exact same
derivation as an independent, testable, reusable value instead.

Does NOT perform rectification, disparity, or depth computation — see
docs/LEVEL3_CONTRACTS.md.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.frames import FrameId, RigidTransform


class CameraModel(str, Enum):
    """Camera projection model.

    PINHOLE is the only value — the only model this library's cv2 usage
    (rectification, StereoSGBM, reprojectImageTo3D) has ever assumed.
    Stated explicitly here rather than left implicit; other models
    (fisheye, omnidirectional) are not supported and no stub is offered
    for them — see docs/IMPLEMENTATION_STATUS.md's "not implemented" list.
    """

    PINHOLE = "pinhole"


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """One camera's intrinsic calibration — a read-only view over StereoCalibration.

    Attributes:
        camera_matrix: (3, 3) intrinsic matrix.
        dist_coeffs: (1, N) distortion coefficients.
        image_size: (width, height) this calibration was computed for.
        camera_model: always CameraModel.PINHOLE today.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    camera_model: CameraModel = CameraModel.PINHOLE

    @classmethod
    def from_calibration(cls, calibration: StereoCalibration, side: str) -> "CameraIntrinsics":
        """Extract one side's intrinsics from an existing StereoCalibration.

        Args:
            calibration: an already-loaded, already-validated StereoCalibration.
            side: "left" or "right".
        """
        if side == "left":
            camera_matrix, dist_coeffs = calibration.camera_matrix_left, calibration.dist_coeffs_left
        elif side == "right":
            camera_matrix, dist_coeffs = calibration.camera_matrix_right, calibration.dist_coeffs_right
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
        return cls(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=calibration.image_size,
        )


@dataclass(frozen=True, slots=True)
class StereoExtrinsics:
    """Left-right stereo baseline/focal length — a read-only view over StereoCalibration.

    Attributes:
        baseline_m: metres between the left and right optical centers.
        focal_length_px: rectified focal length, pixels.
    """

    baseline_m: float
    focal_length_px: float

    @classmethod
    def from_calibration(cls, calibration: StereoCalibration) -> "StereoExtrinsics":
        """Derive baseline/focal length from Q — the exact formula
        DepthEstimator.__init__ already uses internally (Q[2,3] = focal
        length in px, Q[3,2] = -1/Tx with Tx the baseline in mm) — exposed
        here as an independent, testable value rather than private state
        buried in a different class. See tests/test_calibration_contracts.py
        for the cross-check against DepthEstimator's own derivation."""
        Q = calibration.Q
        focal_length_px = abs(float(Q[2, 3]))
        tx = float(Q[3, 2])
        baseline_m = abs(1.0 / tx) / 1000.0 if tx != 0.0 else 0.0
        return cls(baseline_m=baseline_m, focal_length_px=focal_length_px)


@dataclass(frozen=True, slots=True)
class RectificationParameters:
    """The rectification-specific subset of StereoCalibration — a named view.

    Attributes:
        R1, R2: (3, 3) rectification rotations.
        P1, P2: (3, 4) rectified projection matrices.
        Q: (4, 4) disparity-to-depth mapping matrix.
    """

    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray

    @classmethod
    def from_calibration(cls, calibration: StereoCalibration) -> "RectificationParameters":
        return cls(
            R1=calibration.R1, R2=calibration.R2,
            P1=calibration.P1, P2=calibration.P2,
            Q=calibration.Q,
        )


@dataclass(frozen=True, slots=True)
class RigCalibration:
    """A StereoCalibration plus (optionally) its mounting relationship to the body frame.

    Wraps an existing StereoCalibration without modifying it — nothing in
    the current pipeline constructs or consumes a RigCalibration; this
    exists so a future caller has somewhere to put a camera-to-body
    transform once one is actually calibrated, without requiring a change
    to StereoCalibration's own shape (which RectificationEngine/
    DepthEstimator consume directly and unchanged).

    Attributes:
        stereo: the underlying, already-validated StereoCalibration.
        body_T_camera_left: optional rigid transform from
            FrameId.CAMERA_OPTICAL_LEFT to FrameId.BODY — the camera's pose
            in the body frame. None means "not yet calibrated/unknown",
            not "identity" — do not assume zero offset when this is None.
        camera_frame_id: the frame this rig's depth/point-cloud output
            lives in. Defaults to FrameId.CAMERA_OPTICAL_LEFT, matching
            every result this library produces today.
    """

    stereo: StereoCalibration
    body_T_camera_left: Optional[RigidTransform] = None
    camera_frame_id: str = FrameId.CAMERA_OPTICAL_LEFT

    def __post_init__(self) -> None:
        if self.body_T_camera_left is not None:
            if self.body_T_camera_left.to_frame != FrameId.BODY:
                raise ValueError(
                    "RigCalibration.body_T_camera_left.to_frame must be "
                    f"FrameId.BODY ({FrameId.BODY!r}), got "
                    f"{self.body_T_camera_left.to_frame!r}."
                )
            if self.body_T_camera_left.from_frame != self.camera_frame_id:
                raise ValueError(
                    "RigCalibration.body_T_camera_left.from_frame must match "
                    f"camera_frame_id ({self.camera_frame_id!r}), got "
                    f"{self.body_T_camera_left.from_frame!r}."
                )
