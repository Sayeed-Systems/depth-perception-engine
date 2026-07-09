"""
Stereo frame splitting module.

Responsibilities: receive a combined stereo frame, validate it, split it into
equal left and right images, and return both halves.

Does NOT perform camera acquisition, calibration, rectification, disparity,
depth estimation, detection, visualization, or neural network inference.

Kept as an optional utility: ROS2's mp01_camera already publishes separately
split left/right topics, so mp01_perception will not need this class. It
remains here for any caller (e.g. this project's own examples/live_demo.py)
that still receives a single combined side-by-side frame.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FrameSplitter:
    """Splits a side-by-side stereo frame into separate left and right images."""

    def validate_frame(self, frame: np.ndarray) -> None:
        """Validate that *frame* is suitable for splitting.

        Args:
            frame: The combined stereo frame to validate.

        Raises:
            ValueError: If *frame* is None, has fewer than 2 dimensions, has
                        zero-size dimensions, or has an odd width.
            TypeError: If *frame* is not a numpy ndarray.
        """
        if frame is None:
            raise ValueError("Frame is None — cannot split a null frame.")

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(frame).__name__}."
            )

        if frame.ndim < 2:
            raise ValueError(
                f"Frame must have at least 2 dimensions, got {frame.ndim}."
            )

        height, width = frame.shape[:2]

        if height == 0 or width == 0:
            raise ValueError(
                f"Frame has zero-size dimension: height={height}, width={width}."
            )

        if width % 2 != 0:
            raise ValueError(
                f"Frame width ({width}) is not divisible by 2 — "
                "cannot split into equal left and right halves."
            )

    def split(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split a combined side-by-side stereo frame into left and right images.

        The frame is assumed to contain the left view in the left half and the
        right view in the right half, with both halves of equal width.

        Args:
            frame: A BGR (or grayscale) ndarray with shape
                   ``(H, W[, C])`` where *W* is even.

        Returns:
            A ``(left_frame, right_frame)`` tuple of ndarrays, each with shape
            ``(H, W//2[, C])``.

        Raises:
            ValueError: If the frame fails validation (see :meth:`validate_frame`).
            TypeError: If *frame* is not a numpy ndarray.
        """
        logger.debug("Frame received")

        self.validate_frame(frame)

        height, width = frame.shape[:2]
        logger.debug("Frame dimensions detected: height=%d, width=%d", height, width)

        midpoint = width // 2

        left_frame = frame[:, :midpoint]
        right_frame = frame[:, midpoint:]

        logger.debug("Stereo frame successfully split at midpoint=%d", midpoint)
        logger.debug("Left frame shape: %s", left_frame.shape)
        logger.debug("Right frame shape: %s", right_frame.shape)

        return left_frame, right_frame
