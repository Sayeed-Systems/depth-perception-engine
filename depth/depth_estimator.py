"""
Depth estimator module.

Responsibilities: load the Q matrix from calibration, use
cv2.reprojectImageTo3D to convert disparity maps into metric depth (metres),
and expose the Q-derived baseline and focal length for downstream use.

Does NOT perform camera acquisition, frame splitting, disparity computation,
distance region reading, object detection, or visualisation logic.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Converts disparity maps to metric depth using the stereo Q matrix."""

    # Depth range clamp — values outside [MIN, MAX] are zeroed (invalid).
    MIN_DEPTH_M: float = 0.15   # 15 cm — closer than this is below stereo baseline limit
    MAX_DEPTH_M: float = 8.0    # 8 m — beyond reliable range for 64mm baseline

    def __init__(self, Q: np.ndarray) -> None:
        """
        Args:
            Q: 4×4 disparity-to-depth mapping matrix from cv2.stereoRectify.

        Raises:
            TypeError: If Q is not a numpy ndarray.
            ValueError: If Q does not have shape (4, 4).
        """
        if not isinstance(Q, np.ndarray):
            raise TypeError(f"Q must be a numpy.ndarray, got {type(Q).__name__}.")
        if Q.shape != (4, 4):
            raise ValueError(f"Q must have shape (4, 4), got {Q.shape}.")

        self._Q = Q.astype(np.float64)

        # Extract focal length and baseline from Q for informational use.
        # Q[2,3] = f (focal length in px), Q[3,2] = -1/Tx (Tx = -baseline_px)
        self._focal_length_px: float = abs(float(Q[2, 3]))
        tx = float(Q[3, 2])
        self._baseline_m: float = abs(1.0 / tx) / 1000.0 if tx != 0.0 else 0.0

        logger.info(
            "DepthEstimator initialised — focal_length=%.1f px, baseline=%.1f mm",
            self._focal_length_px,
            self._baseline_m * 1000.0,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate(self, disparity: np.ndarray) -> np.ndarray:
        """Convert a disparity map to a metric depth map using the Q matrix.

        Uses cv2.reprojectImageTo3D and extracts the Z channel. Pixels with
        depth outside [MIN_DEPTH_M, MAX_DEPTH_M] are set to 0.0 (invalid).

        Args:
            disparity: float32 ndarray of disparity values (pixels), as
                       produced by DisparityEngine.compute_disparity.

        Returns:
            float32 ndarray of the same spatial shape with depth values in
            metres. Invalid / out-of-range pixels are 0.0.

        Raises:
            TypeError: If disparity is not a numpy ndarray.
            ValueError: If disparity has fewer than 2 dimensions.
            RuntimeError: If reprojection fails unexpectedly.
        """
        if not isinstance(disparity, np.ndarray):
            raise TypeError(
                f"disparity must be numpy.ndarray, got {type(disparity).__name__}."
            )
        if disparity.ndim < 2:
            raise ValueError(
                f"disparity must have at least 2 dimensions, got {disparity.ndim}."
            )

        try:
            points_3d = cv2.reprojectImageTo3D(
                disparity.astype(np.float32), self._Q, handleMissingValues=True
            )
        except cv2.error as exc:
            raise RuntimeError(f"reprojectImageTo3D failed: {exc}") from exc

        # Calibration object-points were in mm → Z comes out in mm; convert to m.
        depth = (points_3d[:, :, 2] / 1000.0).astype(np.float32)

        # Zero out invalid / out-of-range values
        valid = (
            np.isfinite(depth)
            & (depth >= self.MIN_DEPTH_M)
            & (depth <= self.MAX_DEPTH_M)
        )
        depth[~valid] = 0.0

        valid_count = int(valid.sum())
        total = int(depth.size)
        logger.debug(
            "Depth estimated — valid: %d / %d (%.1f%%)",
            valid_count, total,
            100.0 * valid_count / total if total > 0 else 0.0,
        )

        return depth

    def estimate_point_cloud(
        self, disparity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return XYZ point cloud and a boolean validity mask.

        Args:
            disparity: float32 disparity map (pixels).

        Returns:
            ``(points_3d, valid_mask)`` where *points_3d* is a (H, W, 3)
            float32 array of (X, Y, Z) in metres and *valid_mask* is a (H, W)
            bool array marking finite, in-range points.
        """
        if not isinstance(disparity, np.ndarray):
            raise TypeError(
                f"disparity must be numpy.ndarray, got {type(disparity).__name__}."
            )

        points_3d = cv2.reprojectImageTo3D(
            disparity.astype(np.float32), self._Q, handleMissingValues=True
        ).astype(np.float32)
        points_3d /= 1000.0  # mm → m

        z = points_3d[:, :, 2]
        valid_mask = (
            np.isfinite(z) & (z >= self.MIN_DEPTH_M) & (z <= self.MAX_DEPTH_M)
        )

        return points_3d, valid_mask

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def Q(self) -> np.ndarray:
        """The 4×4 Q matrix used for reprojection."""
        return self._Q

    @property
    def focal_length_px(self) -> float:
        """Focal length extracted from Q (pixels)."""
        return self._focal_length_px

    @property
    def baseline_m(self) -> float:
        """Camera baseline extracted from Q (metres)."""
        return self._baseline_m

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration_file(cls, path: str) -> "DepthEstimator":
        """Construct a DepthEstimator by loading Q from a calibration XML.

        Args:
            path: Path to the OpenCV FileStorage XML (e.g.
                  ``config/stereo_calibration.xml``).

        Raises:
            FileNotFoundError: If *path* does not exist.
            RuntimeError: If the file cannot be opened or Q is missing.
        """
        import os
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Calibration file not found: {path}")

        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"cv2.FileStorage could not open: {path}")

        try:
            Q = fs.getNode("Q").mat()
        finally:
            fs.release()

        if Q is None or Q.size == 0:
            raise RuntimeError(f"Q matrix missing or empty in: {path}")

        return cls(Q)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DepthEstimator(focal_length_px={self._focal_length_px:.1f}, "
            f"baseline_m={self._baseline_m:.4f})"
        )