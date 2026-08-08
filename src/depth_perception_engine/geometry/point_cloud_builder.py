"""
Point-cloud builder — Level 3, Phase E2.

The one canonical point-cloud generator this repository offers. Wires the
existing (previously dead, and previously buggy — see
DepthEstimator.estimate_point_cloud's docstring) reprojection math into a
real, tested execution path, producing a geometry.PointCloud in the camera
optical frame.

No reprojection math lives in this module — that stays in
DepthEstimator.estimate_point_cloud (cv2.reprojectImageTo3D driven off the
calibration's Q matrix, unchanged in shape/meaning by this pass). This
class owns exactly one thing: converting DepthEstimator's (points, mask,
0.0-invalid) return shape into geometry.PointCloud's (points, valid_mask,
NaN-invalid) contract — see docs/LEVEL3_CONTRACTS.md for why the two
conventions differ (0.0 is a legitimate coordinate on the optical axis;
NaN is not, so PointCloud uses NaN for "no data" instead).

Does NOT perform obstacle extraction, free-space ray casting, occupancy
mapping, body-frame transformation, or DepthPerceptionResult/pipeline
wiring — see docs/E2_IMPLEMENTATION_PLAN.md for what those steps are and
why they stay separate.
"""

import logging
from typing import Optional

import numpy as np

from depth_perception_engine.calibration.models import StereoCalibration
from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry.types import PointCloud

logger = logging.getLogger(__name__)


class PointCloudBuilder:
    """Builds a camera-optical-frame PointCloud from a disparity map."""

    def __init__(self, Q: np.ndarray) -> None:
        """
        Args:
            Q: 4x4 disparity-to-depth mapping matrix from cv2.stereoRectify.

        Raises:
            TypeError: If Q is not a numpy ndarray.
            ValueError: If Q does not have shape (4, 4), or contains a
                        non-finite (NaN/Inf) entry — an invalid
                        calibration must never silently produce a point
                        cloud (see docs/CALIBRATION.md's documented gap:
                        StereoCalibration itself does not check matrix
                        entries are finite, so this builder checks Q
                        itself rather than assuming an upstream check
                        already ran).
        """
        if isinstance(Q, np.ndarray) and Q.shape == (4, 4) and not np.all(np.isfinite(Q)):
            raise ValueError(
                "Q contains non-finite (NaN/Inf) entries — refusing to build "
                "a PointCloudBuilder from an invalid calibration."
            )
        # DepthEstimator.__init__ validates type/shape; reused rather than
        # re-implemented here.
        self._depth_estimator = DepthEstimator(Q)

    @classmethod
    def from_calibration(cls, calibration: StereoCalibration) -> "PointCloudBuilder":
        """Construct a PointCloudBuilder from an already-loaded StereoCalibration."""
        return cls(calibration.Q)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(
        self,
        disparity: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> PointCloud:
        """Build a PointCloud from a disparity map.

        Fully vectorized (NumPy/OpenCV) — no Python-level pixel loop —
        and deterministic: the same disparity array always produces
        bit-identical output.

        Rejection rules (mirrors DepthEstimator.estimate_point_cloud
        exactly — see that method's docstring for the full rationale):
        a pixel is invalid, and its point is set to NaN in the output,
        when its disparity is non-finite (NaN/Inf), zero, or negative;
        or when the reprojected depth falls outside
        [DepthEstimator.MIN_DEPTH_M, DepthEstimator.MAX_DEPTH_M]; or when
        the reprojected X/Y/Z itself is non-finite for any other reason.
        No invalid pixel can ever produce a finite XYZ in the output.

        Args:
            disparity: float32 (or castable) ndarray of disparity values
                       (pixels), shape (H, W), as produced by
                       DisparityEngine.compute_disparity.
            timestamp: opaque, caller-defined — passed straight through
                       to PointCloud.timestamp.

        Returns:
            A PointCloud in FrameId.CAMERA_OPTICAL_LEFT: `points` shape
            (H, W, 3) float32 metres, NaN (all 3 channels) at every pixel
            where `valid_mask` is False. `valid_mask` is (H, W) bool.
            `confidence` is None — this builder does not compute a
            per-pixel confidence signal.

        Raises:
            TypeError: If disparity is not a numpy ndarray.
            ValueError: If disparity does not have exactly 2 dimensions.
        """
        points_m, valid_mask = self._depth_estimator.estimate_point_cloud(disparity)

        points_m = points_m.copy()
        points_m[~valid_mask] = np.nan

        return PointCloud(
            points=points_m,
            frame_id=FrameId.CAMERA_OPTICAL_LEFT,
            valid_mask=valid_mask,
            confidence=None,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def Q(self) -> np.ndarray:
        """The 4x4 Q matrix used for reprojection."""
        return self._depth_estimator.Q

    @property
    def focal_length_px(self) -> float:
        """Focal length extracted from Q (pixels)."""
        return self._depth_estimator.focal_length_px

    @property
    def baseline_m(self) -> float:
        """Camera baseline extracted from Q (metres)."""
        return self._depth_estimator.baseline_m

    def __repr__(self) -> str:
        return (
            f"PointCloudBuilder(focal_length_px={self.focal_length_px:.1f}, "
            f"baseline_m={self.baseline_m:.4f})"
        )
