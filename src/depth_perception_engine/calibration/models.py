"""
Stereo calibration data model.

A plain, immutable container for the matrices a stereo rig's calibration
produces. Carrying these as one object — instead of nine loose ndarrays —
is what lets RectificationEngine and DepthEstimator be constructed from
calibration that was loaded once by the caller, rather than each class
reaching into a file on disk itself.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class StereoCalibration:
    """Everything a rectification/depth pipeline needs from a stereo calibration.

    Attributes:
        image_size: ``(width, height)`` the calibration was computed for.
        camera_matrix_left / camera_matrix_right: 3x3 intrinsic matrices.
        dist_coeffs_left / dist_coeffs_right: distortion coefficients.
        R1 / R2: 3x3 rectification rotations.
        P1 / P2: 3x4 rectified projection matrices.
        Q: 4x4 disparity-to-depth mapping matrix.
    """

    image_size: Tuple[int, int]
    camera_matrix_left: np.ndarray
    dist_coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_right: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray

    def __post_init__(self) -> None:
        required = {
            "camera_matrix_left": (self.camera_matrix_left, (3, 3)),
            "camera_matrix_right": (self.camera_matrix_right, (3, 3)),
            "dist_coeffs_left": (self.dist_coeffs_left, None),
            "dist_coeffs_right": (self.dist_coeffs_right, None),
            "R1": (self.R1, (3, 3)),
            "R2": (self.R2, (3, 3)),
            "P1": (self.P1, (3, 4)),
            "P2": (self.P2, (3, 4)),
            "Q": (self.Q, (4, 4)),
        }
        for name, (mat, expected_shape) in required.items():
            if mat is None or not isinstance(mat, np.ndarray) or mat.size == 0:
                raise ValueError(
                    f"StereoCalibration.{name} is missing or empty."
                )
            if expected_shape is not None and mat.shape != expected_shape:
                raise ValueError(
                    f"StereoCalibration.{name} has shape {mat.shape}, "
                    f"expected {expected_shape}."
                )
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError(
                f"StereoCalibration.image_size must be positive, got {self.image_size}."
            )
