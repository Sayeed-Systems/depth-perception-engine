"""
Distance reading module.

Responsibilities: receive a depth map, extract a configurable centre ROI,
filter out invalid depth values, compute a stable median distance estimate,
and return the result in both metres and centimetres.

Does NOT perform camera acquisition, frame splitting, disparity computation,
depth generation, zone classification, object detection, or visualisation.

Not part of DepthPerceptionPipeline's default output — it is a single-point
convenience reading (e.g. "what's straight ahead"), independent of the
per-beam obstacle scan (obstacles/threat_assessment.py) and the region grid
(traversability/scene_interpreter.py). Kept available for callers that want
it (e.g. examples/live_demo.py's on-screen distance badge).
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default centre ROI dimensions in pixels
_DEFAULT_ROI_WIDTH: int = 50
_DEFAULT_ROI_HEIGHT: int = 50


class DistanceReader:
    """Extracts a stable distance estimate from the centre ROI of a depth map."""

    def __init__(
        self,
        roi_width: int = _DEFAULT_ROI_WIDTH,
        roi_height: int = _DEFAULT_ROI_HEIGHT,
    ) -> None:
        """
        Args:
            roi_width: Width of the centre ROI in pixels (default 50).
            roi_height: Height of the centre ROI in pixels (default 50).

        Raises:
            ValueError: If either ROI dimension is less than 1.
        """
        if roi_width < 1:
            raise ValueError(f"roi_width must be >= 1, got {roi_width}.")
        if roi_height < 1:
            raise ValueError(f"roi_height must be >= 1, got {roi_height}.")

        self._roi_width = roi_width
        self._roi_height = roi_height

        logger.info(
            "DistanceReader initialised — ROI size: %dx%d px",
            roi_width,
            roi_height,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_roi(self, depth_map: np.ndarray) -> np.ndarray:
        """Extract the centre ROI patch from *depth_map*.

        The ROI is centred on the image. If the requested ROI is larger than
        the image in either dimension it is clamped to the image boundary.

        Args:
            depth_map: 2-D (H, W) float32 depth map in metres.

        Returns:
            A 1-D float32 ndarray of all depth values inside the ROI
            (row-major flattened), including invalid values — call
            :meth:`filter_depth_values` before computing a distance estimate.

        Raises:
            TypeError: If *depth_map* is not a numpy ndarray.
            ValueError: If *depth_map* is None or has fewer than 2 dimensions.
        """
        self._validate_depth_map(depth_map)

        x1, y1, x2, y2 = self._compute_roi_bounds(depth_map)
        roi_patch = depth_map[y1:y2, x1:x2]

        logger.debug(
            "ROI extracted — rows [%d:%d], cols [%d:%d], shape: %s",
            y1, y2, x1, x2, roi_patch.shape,
        )

        return roi_patch.flatten().astype(np.float32)

    def get_roi_bounds(self, depth_map: np.ndarray) -> Tuple[int, int, int, int]:
        """Return the pixel bounds of the centre ROI for overlay drawing.

        Args:
            depth_map: 2-D (H, W) depth map used to determine image size.

        Returns:
            ``(x1, y1, x2, y2)`` — suitable for use with ``cv2.rectangle``.
        """
        self._validate_depth_map(depth_map)
        return self._compute_roi_bounds(depth_map)

    @staticmethod
    def filter_depth_values(depth_values: np.ndarray) -> np.ndarray:
        """Remove invalid depth values from a 1-D array.

        Filtered out: zeros, NaN, infinite values, and negative values.

        Args:
            depth_values: 1-D float32 ndarray of raw depth values (metres).

        Returns:
            1-D float32 ndarray containing only finite, positive values.
            May be empty if no valid values exist.
        """
        valid_mask = np.isfinite(depth_values) & (depth_values > 0.0)
        filtered = depth_values[valid_mask]

        logger.debug(
            "filter_depth_values — %d / %d values retained",
            filtered.size,
            depth_values.size,
        )

        return filtered

    @staticmethod
    def compute_distance(depth_values: np.ndarray) -> Dict[str, float]:
        """Compute a stable distance estimate from a 1-D array of valid depths.

        Uses the median of *depth_values* as the distance estimate. If
        *depth_values* is empty, all outputs are 0.0.

        Args:
            depth_values: 1-D float32 ndarray of pre-filtered depth values
                          in metres (output of :meth:`filter_depth_values`).

        Returns:
            Dict with keys:

            * ``"distance_meters"``      — median depth in metres.
            * ``"distance_centimeters"`` — median depth in centimetres.
            * ``"min_depth"``            — minimum depth in metres.
            * ``"max_depth"``            — maximum depth in metres.
            * ``"median_depth"``         — median depth in metres.
            * ``"valid_pixel_count"``    — number of valid pixels used.
        """
        if depth_values.size == 0:
            logger.warning("compute_distance — no valid depth values; returning zeros.")
            return {
                "distance_meters": 0.0,
                "distance_centimeters": 0.0,
                "min_depth": 0.0,
                "max_depth": 0.0,
                "median_depth": 0.0,
                "valid_pixel_count": 0,
            }

        median_m = float(np.median(depth_values))
        stats: Dict[str, float] = {
            "distance_meters": median_m,
            "distance_centimeters": median_m * 100.0,
            "min_depth": float(depth_values.min()),
            "max_depth": float(depth_values.max()),
            "median_depth": median_m,
            "valid_pixel_count": float(depth_values.size),
        }

        logger.debug(
            "compute_distance — median: %.3f m (%.1f cm), "
            "min: %.3f m, max: %.3f m, valid: %d",
            stats["distance_meters"],
            stats["distance_centimeters"],
            stats["min_depth"],
            stats["max_depth"],
            stats["valid_pixel_count"],
        )

        return stats

    def read_distance(self, depth_map: np.ndarray) -> Dict[str, float]:
        """Extract a stable centre-ROI distance from a depth map.

        Runs the full pipeline:
            extract_roi → filter_depth_values → compute_distance

        Args:
            depth_map: 2-D float32 ndarray of depth values in metres, as
                       produced by DepthEstimator.

        Returns:
            Dict with keys ``"distance_meters"``, ``"distance_centimeters"``,
            ``"min_depth"``, ``"max_depth"``, ``"median_depth"``, and
            ``"valid_pixel_count"``.  All values are 0.0 / 0 when no valid
            pixels are found inside the ROI.

        Raises:
            TypeError: If *depth_map* is not a numpy ndarray.
            ValueError: If *depth_map* fails shape validation.
            RuntimeError: If distance computation fails unexpectedly.
        """
        logger.debug(
            "read_distance called — depth_map shape: %s",
            getattr(depth_map, "shape", "unknown"),
        )

        try:
            roi_values = self.extract_roi(depth_map)
            valid_values = self.filter_depth_values(roi_values)
            measurement = self.compute_distance(valid_values)
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Distance reading failed: {exc}") from exc

        logger.info(
            "Distance — %.3f m (%.1f cm), valid pixels: %d",
            measurement["distance_meters"],
            measurement["distance_centimeters"],
            measurement["valid_pixel_count"],
        )

        return measurement

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_depth_map(depth_map: np.ndarray) -> None:
        """Raise on None, wrong type, or fewer than 2 dimensions."""
        if depth_map is None:
            raise ValueError("depth_map is None.")
        if not isinstance(depth_map, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(depth_map).__name__}."
            )
        if depth_map.ndim < 2:
            raise ValueError(
                f"depth_map must have at least 2 dimensions, got {depth_map.ndim}."
            )

    def _compute_roi_bounds(
        self, depth_map: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """Compute ``(x1, y1, x2, y2)`` of the centre ROI, clamped to image."""
        h, w = depth_map.shape[:2]
        cy, cx = h // 2, w // 2
        half_h = self._roi_height // 2
        half_w = self._roi_width // 2

        y1 = max(0, cy - half_h)
        y2 = min(h, cy + half_h + (self._roi_height % 2))
        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w + (self._roi_width % 2))

        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DistanceReader(roi_width={self._roi_width}, "
            f"roi_height={self._roi_height})"
        )
