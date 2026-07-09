"""
Stereo rectification module.

Responsibilities: take stereo calibration data, initialise undistort-rectify
maps for both cameras, and remap raw stereo image pairs to a common
rectified plane.

Does NOT perform camera acquisition, frame splitting, disparity computation,
depth estimation, distance measurement, object detection, visualisation
dashboard logic, or neural network inference — and does NOT load calibration
from a file itself (see depth_perception_engine.calibration.load_stereo_calibration
for that); it only ever receives an already-loaded StereoCalibration object,
so this module carries no hardcoded or default file path.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from depth_perception_engine.calibration.models import StereoCalibration

logger = logging.getLogger(__name__)


class RectificationEngine:
    """Builds rectification maps from a StereoCalibration and rectifies stereo pairs."""

    def __init__(self, calibration: StereoCalibration) -> None:
        """
        Args:
            calibration: An already-loaded StereoCalibration (see
                         depth_perception_engine.calibration.load_stereo_calibration).
        """
        self._calibration = calibration

        # Rectification maps (CV_16SC2 for remap speed) — built lazily by
        # initialize_rectification().
        self._map1_left: Optional[np.ndarray] = None
        self._map2_left: Optional[np.ndarray] = None
        self._map1_right: Optional[np.ndarray] = None
        self._map2_right: Optional[np.ndarray] = None

        logger.info(
            "RectificationEngine created — image size: %s",
            calibration.image_size,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialize_rectification(self) -> None:
        """Build the undistort-rectify maps for both cameras.

        Computes two pairs of maps (one per camera) using
        :func:`cv2.initUndistortRectifyMap` with ``CV_16SC2`` precision,
        which is optimised for use with :func:`cv2.remap`.
        """
        c = self._calibration
        logger.info("Initializing rectification maps...")

        self._map1_left, self._map2_left = cv2.initUndistortRectifyMap(
            c.camera_matrix_left,
            c.dist_coeffs_left,
            c.R1,
            c.P1,
            c.image_size,
            cv2.CV_16SC2,
        )

        self._map1_right, self._map2_right = cv2.initUndistortRectifyMap(
            c.camera_matrix_right,
            c.dist_coeffs_right,
            c.R2,
            c.P2,
            c.image_size,
            cv2.CV_16SC2,
        )

        logger.info(
            "Rectification maps created — size: %dx%d",
            c.image_size[0],
            c.image_size[1],
        )

    def rectify(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify a stereo image pair using pre-computed maps.

        Applies :func:`cv2.remap` with bilinear interpolation to both frames.

        Args:
            left_frame: Raw left camera image (BGR or grayscale ndarray).
            right_frame: Raw right camera image (BGR or grayscale ndarray).

        Returns:
            ``(left_rectified, right_rectified)`` — rectified images with the
            same dtype and number of channels as the inputs.

        Raises:
            RuntimeError: If rectification maps have not been initialised.
            ValueError: If either frame is ``None``, not a numpy ndarray,
                        has fewer than 2 dimensions, or does not match the
                        calibrated image size.
        """
        if not self.is_initialized():
            raise RuntimeError(
                "Rectification maps not initialised. "
                "Call initialize_rectification() first."
            )

        self._validate_frame(left_frame, "left")
        self._validate_frame(right_frame, "right")

        assert self._map1_left is not None
        assert self._map2_left is not None
        assert self._map1_right is not None
        assert self._map2_right is not None

        left_rectified: np.ndarray = cv2.remap(
            left_frame,
            self._map1_left,
            self._map2_left,
            cv2.INTER_LINEAR,
        )
        right_rectified: np.ndarray = cv2.remap(
            right_frame,
            self._map1_right,
            self._map2_right,
            cv2.INTER_LINEAR,
        )

        logger.debug("Stereo images rectified.")
        return left_rectified, right_rectified

    def is_initialized(self) -> bool:
        """Return ``True`` if the rectification maps are ready for use."""
        return all(
            m is not None
            for m in (
                self._map1_left,
                self._map2_left,
                self._map1_right,
                self._map2_right,
            )
        )

    # ------------------------------------------------------------------
    # Properties — read-only access for downstream modules
    # ------------------------------------------------------------------

    @property
    def calibration(self) -> StereoCalibration:
        return self._calibration

    @property
    def image_size(self) -> Tuple[int, int]:
        """``(width, height)`` from the calibration."""
        return self._calibration.image_size

    @property
    def Q(self) -> np.ndarray:
        """Disparity-to-depth mapping matrix (4×4)."""
        return self._calibration.Q

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_frame(self, frame: np.ndarray, side: str) -> None:
        """Raise if *frame* is unsuitable for remapping."""
        if frame is None:
            raise ValueError(f"{side} frame is None.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(
                f"{side} frame must be a numpy.ndarray, "
                f"got {type(frame).__name__}."
            )
        if frame.ndim < 2:
            raise ValueError(
                f"{side} frame has fewer than 2 dimensions ({frame.ndim})."
            )
        exp_w, exp_h = self._calibration.image_size
        h, w = frame.shape[:2]
        if (w, h) != (exp_w, exp_h):
            raise ValueError(
                f"{side} frame size ({w}×{h}) does not match calibrated "
                f"size ({exp_w}×{exp_h})."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RectificationEngine("
            f"image_size={self._calibration.image_size}, "
            f"initialized={self.is_initialized()})"
        )
