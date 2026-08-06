"""
Stereo disparity computation module.

Responsibilities: receive left/right frames, validate them, convert to
grayscale when needed, compute a disparity map with StereoSGBM, normalize it
for visualization, and return both the raw and visualization forms.

Does NOT perform camera acquisition, frame splitting, calibration,
rectification, depth estimation, object detection, visualization windows,
or neural network inference.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DisparityEngine:
    """Computes stereo disparity maps using OpenCV StereoSGBM."""

    def __init__(
        self,
        min_disparity: int = 0,
        num_disparities: int = 128,
        block_size: int = 11,
    ) -> None:
        """
        Args:
            min_disparity: Minimum possible disparity value (default 0).
            num_disparities: Range of disparity search; must be divisible by 16
                             (default 128).
            block_size: Matched block size; must be odd and ≥ 1 (default 11).
        """
        if num_disparities % 16 != 0:
            raise ValueError(
                f"num_disparities ({num_disparities}) must be divisible by 16."
            )
        if block_size % 2 == 0 or block_size < 1:
            raise ValueError(
                f"block_size ({block_size}) must be a positive odd integer."
            )

        self._min_disparity = min_disparity
        self._num_disparities = num_disparities
        self._block_size = block_size

        # P1 / P2 penalty terms — standard heuristic based on block size
        p1 = 8 * 3 * block_size ** 2
        p2 = 32 * 3 * block_size ** 2

        self._stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        logger.info(
            "DisparityEngine initialised — minDisparity=%d, "
            "numDisparities=%d, blockSize=%d",
            min_disparity,
            num_disparities,
            block_size,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate_inputs(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> None:
        """Validate that both frames are suitable for disparity computation.

        Args:
            left_frame: Left stereo image (BGR or grayscale ndarray).
            right_frame: Right stereo image (BGR or grayscale ndarray).

        Raises:
            ValueError: If either frame is None, has zero-size dimensions, or
                        the two frames have mismatched spatial dimensions.
            TypeError: If either frame is not a numpy ndarray.
        """
        for name, frame in (("left", left_frame), ("right", right_frame)):
            if frame is None:
                raise ValueError(f"{name} frame is None.")
            if not isinstance(frame, np.ndarray):
                raise TypeError(
                    f"{name} frame must be a numpy.ndarray, "
                    f"got {type(frame).__name__}."
                )
            if frame.ndim < 2:
                raise ValueError(
                    f"{name} frame has fewer than 2 dimensions ({frame.ndim})."
                )
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                raise ValueError(
                    f"{name} frame has zero-size dimension: "
                    f"height={h}, width={w}."
                )

        lh, lw = left_frame.shape[:2]
        rh, rw = right_frame.shape[:2]
        if (lh, lw) != (rh, rw):
            raise ValueError(
                f"Left frame shape {left_frame.shape[:2]} does not match "
                f"right frame shape {right_frame.shape[:2]}."
            )

    def normalize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """Normalize a raw disparity map to a uint8 image for visualization.

        Pixels with invalid disparity (value ≤ 0 after SGBM fixed-point
        conversion) are set to zero so they appear black.

        Args:
            disparity: Raw int16 or float32 disparity map from StereoSGBM.

        Returns:
            Normalized uint8 ndarray in the range [0, 255].
        """
        disp_float = disparity.astype(np.float32)

        # StereoSGBM returns values multiplied by 16; scale to actual pixels
        disp_float /= 16.0

        # Mask invalid regions before normalization
        valid_mask = disp_float > 0
        if valid_mask.any():
            disp_min = disp_float[valid_mask].min()
            disp_max = disp_float[valid_mask].max()
            if disp_max > disp_min:
                disp_float = np.where(
                    valid_mask,
                    (disp_float - disp_min) / (disp_max - disp_min) * 255.0,
                    0.0,
                )
            else:
                disp_float = np.zeros_like(disp_float)
        else:
            logger.warning("No valid disparity pixels found — returning blank map.")
            disp_float = np.zeros_like(disp_float)

        return disp_float.astype(np.uint8)

    def compute_disparity(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        left_gray: Optional[np.ndarray] = None,
        compute_visualization: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute a disparity map from a rectified stereo image pair.

        Args:
            left_frame: Left stereo image (BGR or grayscale ndarray).
            right_frame: Right stereo image (BGR or grayscale ndarray).
            left_gray: Precomputed grayscale conversion of *left_frame*,
                if the caller already needed one for another purpose
                (e.g. traversability texture analysis) — reused here
                instead of converting a second time. Must be the same
                conversion _to_grayscale would produce. Default None
                converts internally, as before.
            compute_visualization: Whether to compute and return the
                normalized uint8 visualization form. Default True
                preserves this method's existing return contract.
                Production/repeated-frame callers that never read
                *visualization_disparity* should pass False to skip
                that extra full-frame normalization work entirely.

        Returns:
            ``(raw_disparity, visualization_disparity)`` where

            * *raw_disparity* is a float32 ndarray of true disparity values
              (pixels) suitable for downstream depth computation.
            * *visualization_disparity* is a uint8 ndarray normalized to
              [0, 255] — brighter pixels indicate closer objects — or
              None if compute_visualization=False.

        Raises:
            ValueError: If input validation fails.
            TypeError: If either frame is not a numpy ndarray.
            RuntimeError: If StereoSGBM computation fails unexpectedly.
        """
        logger.debug("compute_disparity called")

        self.validate_inputs(left_frame, right_frame)

        left_gray = left_gray if left_gray is not None else self._to_grayscale(left_frame, "left")
        right_gray = self._to_grayscale(right_frame, "right")

        logger.debug(
            "Input dimensions — left: %s, right: %s",
            left_gray.shape,
            right_gray.shape,
        )

        try:
            raw = self._stereo.compute(left_gray, right_gray)
        except cv2.error as exc:
            raise RuntimeError(
                f"StereoSGBM computation failed: {exc}"
            ) from exc

        # Convert fixed-point int16 → true float32 disparity in pixels
        raw_disparity = (raw / 16.0).astype(np.float32)

        visualization_disparity = (
            self.normalize_disparity(raw) if compute_visualization else None
        )

        logger.debug(
            "Disparity computed — shape: %s, min: %.2f, max: %.2f",
            raw_disparity.shape,
            float(raw_disparity.min()),
            float(raw_disparity.max()),
        )

        return raw_disparity, visualization_disparity

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(frame: np.ndarray, label: str) -> np.ndarray:
        """Return a grayscale version of *frame*, converting if necessary."""
        if frame.ndim == 2:
            logger.debug("%s frame is already grayscale", label)
            return frame
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]
        if frame.ndim == 3 and frame.shape[2] == 3:
            logger.debug("Converting %s frame BGR → grayscale", label)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 and frame.shape[2] == 4:
            logger.debug("Converting %s frame BGRA → grayscale", label)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        raise ValueError(
            f"Unsupported {label} frame shape: {frame.shape}."
        )

    def __repr__(self) -> str:
        return (
            f"DisparityEngine(min_disparity={self._min_disparity}, "
            f"num_disparities={self._num_disparities}, "
            f"block_size={self._block_size})"
        )
