"""
Depth estimator module.

Responsibilities: use the calibration's Q matrix with
cv2.reprojectImageTo3D to convert disparity maps into metric depth (metres),
and expose the Q-derived baseline and focal length for downstream use.

Does NOT perform camera acquisition, frame splitting, disparity computation,
distance region reading, object detection, or visualisation logic.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

from depth_perception_engine.calibration.loader import load_stereo_calibration
from depth_perception_engine.calibration.models import StereoCalibration

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

        Computes only the Z (depth) channel cv2.reprojectImageTo3D would
        produce, not the full X/Y/Z reprojection — nothing downstream of
        this method ever consumes X or Y (see estimate_point_cloud() for
        the rare caller that actually wants the full point cloud). For
        any Q produced by cv2.stereoRectify, Z depends only on Q's row 2
        and the homogeneous divisor W only on row 3 — rows 0/1 (which
        produce X/Y) never factor into disparity at all, by construction
        of the standard Q matrix — so for every pixel with a *valid*
        disparity, Z = (Q[2,2]*d + Q[2,3]) / (Q[3,2]*d + Q[3,3]) is
        mathematically identical to
        cv2.reprojectImageTo3D(...)[:, :, 2] — verified in
        tests/test_depth_estimator.py's TestZOnlyMatchesFullReprojection
        across both this project's real calibration and synthetic Q
        matrices with a non-zero principal-point-offset term.

        Deliberately does NOT use reprojectImageTo3D's own
        handleMissingValues sentinel (which special-cases whichever
        disparity value happens to be the *frame's minimum* — an
        OpenCV-internal heuristic that doesn't match this codebase's
        own invalid-disparity convention, used everywhere else in this
        package: disparity <= 0, see e.g. DisparityEngine's
        normalize_disparity). Instead, disparity <= 0 is masked
        explicitly below, before the division — a small but real
        correctness improvement, not just a perf-neutral swap: the old
        sentinel-based approach could (for some Q matrices) fail to
        flag a genuinely invalid pixel as invalid, or could incorrectly
        flag a legitimate small-but-positive disparity as invalid
        purely because it happened to be the frame's minimum.

        Pixels with a non-positive disparity, or a resulting depth
        outside [MIN_DEPTH_M, MAX_DEPTH_M], are set to 0.0 (invalid).

        Args:
            disparity: float32 ndarray of disparity values (pixels), as
                       produced by DisparityEngine.compute_disparity.

        Returns:
            float32 ndarray of the same spatial shape with depth values in
            metres. Invalid / out-of-range pixels are 0.0.

        Raises:
            TypeError: If disparity is not a numpy ndarray.
            ValueError: If disparity has fewer than 2 dimensions.
        """
        if not isinstance(disparity, np.ndarray):
            raise TypeError(
                f"disparity must be numpy.ndarray, got {type(disparity).__name__}."
            )
        if disparity.ndim < 2:
            raise ValueError(
                f"disparity must have at least 2 dimensions, got {disparity.ndim}."
            )

        disp_f = disparity.astype(np.float32)
        invalid_disp = disp_f <= 0.0

        q22, q23 = float(self._Q[2, 2]), float(self._Q[2, 3])
        q32, q33 = float(self._Q[3, 2]), float(self._Q[3, 3])

        with np.errstate(divide="ignore", invalid="ignore"):
            z_mm = q22 * disp_f + q23
            w = q32 * disp_f + q33
            depth_mm = z_mm / w

        depth_mm[invalid_disp] = np.nan

        # Calibration object-points were in mm → Z comes out in mm; convert to m.
        depth = (depth_mm / 1000.0).astype(np.float32)

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
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration(cls, calibration: StereoCalibration) -> "DepthEstimator":
        """Construct a DepthEstimator from an already-loaded StereoCalibration."""
        return cls(calibration.Q)

    @classmethod
    def from_calibration_file(cls, path: str) -> "DepthEstimator":
        """Construct a DepthEstimator by loading calibration from an XML file.

        Convenience wrapper around
        depth_perception_engine.calibration.load_stereo_calibration — useful
        for scripts/examples; the core pipeline API always takes an
        already-loaded StereoCalibration instead (see from_calibration).

        Args:
            path: Path to the OpenCV FileStorage XML (e.g.
                  ``config/stereo_calibration.xml``). Must be supplied by the
                  caller — there is no default.
        """
        return cls.from_calibration(load_stereo_calibration(path))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DepthEstimator(focal_length_px={self._focal_length_px:.1f}, "
            f"baseline_m={self._baseline_m:.4f})"
        )
