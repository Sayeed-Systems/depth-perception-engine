"""
Region analyzer — computes texture, disparity-validity, depth statistics,
and a confidence score for a single image region, then classifies it.

This is purely a traversability module built on the existing stereo
pipeline's outputs (raw disparity + metric depth + the rectified grayscale
frame). It does NOT run any detector or neural network — every signal here
is a classical image-statistics computation (gradients, entropy, percentiles).

Why this exists: StereoSGBM returns invalid disparity for textureless
surfaces (a blank wall) for the same reason it returns invalid disparity
for genuinely empty space beyond range — there's no correspondence to find
either way. Treating "invalid disparity" as "free space" is unsafe. This
module instead treats invalid disparity as LOW CONFIDENCE and combines it
with texture and gradient statistics from the raw image to judge whether a
low-confidence region is probably a near, textureless obstacle (a wall)
or genuinely ambiguous.
"""

import logging

import cv2
import numpy as np

from depth_perception_engine.traversability.types import (
    RegionClass,
    RegionStats,
    TextureClass,
)

logger = logging.getLogger(__name__)

_ENTROPY_HIST_BINS = 256
_MAX_8BIT_ENTROPY  = 8.0  # log2(256) — used to normalise entropy into [0, 1]


class RegionAnalyzer:
    """Computes statistics and a classification for one rectangular region."""

    def __init__(
        self,
        texture_low_thresh:          float = 8.0,
        texture_high_thresh:         float = 25.0,
        invalid_ratio_wall_thresh:   float = 0.85,
        invalid_ratio_unknown_thresh: float = 0.50,
        confidence_low_thresh:       float = 0.35,
        obstacle_caution_m:          float = 0.60,
        obstacle_clear_m:            float = 1.20,
        min_valid_pixels:            int   = 20,
    ) -> None:
        """
        Args:
            texture_low_thresh:           texture_score below this → LOW_TEXTURE
            texture_high_thresh:          texture_score above this → HIGH_TEXTURE
            invalid_ratio_wall_thresh:     invalid_ratio at/above this, combined with
                                           low texture + low confidence, → PROBABLE_WALL
            invalid_ratio_unknown_thresh: invalid_ratio at/above this (without meeting
                                           the wall criteria) → LOW_TEXTURE_UNKNOWN
            confidence_low_thresh:        confidence at/below this → LOW_CONFIDENCE
                                           (unless a more specific rule already applied)
            obstacle_caution_m:           depth_min_m below this (with good confidence)
                                           → OBSTACLE
            obstacle_clear_m:             unused here directly; kept for parity with the
                                           navigation layer's thresholds
            min_valid_pixels:             regions with fewer valid disparity pixels than
                                           this never get computed depth statistics

        Raises:
            ValueError: If any threshold is out of its valid range.
        """
        if not (0.0 <= invalid_ratio_unknown_thresh <= invalid_ratio_wall_thresh <= 1.0):
            raise ValueError(
                "Require 0 <= invalid_ratio_unknown_thresh <= invalid_ratio_wall_thresh <= 1."
            )
        if not (0.0 <= confidence_low_thresh <= 1.0):
            raise ValueError("confidence_low_thresh must be in [0, 1].")
        if texture_low_thresh < 0.0 or texture_high_thresh < texture_low_thresh:
            raise ValueError("Require 0 <= texture_low_thresh <= texture_high_thresh.")

        self._texture_low_thresh           = texture_low_thresh
        self._texture_high_thresh          = texture_high_thresh
        self._invalid_ratio_wall_thresh    = invalid_ratio_wall_thresh
        self._invalid_ratio_unknown_thresh = invalid_ratio_unknown_thresh
        self._confidence_low_thresh        = confidence_low_thresh
        self._obstacle_caution_m           = obstacle_caution_m
        self._obstacle_clear_m             = obstacle_clear_m
        self._min_valid_pixels             = min_valid_pixels

    # ------------------------------------------------------------------
    def analyze(
        self,
        name: str,
        row: int,
        col: int,
        x1: int, y1: int, x2: int, y2: int,
        gray_region:  np.ndarray,
        disp_region:  np.ndarray,
        depth_region: np.ndarray,
    ) -> RegionStats:
        """Compute every signal for one region and classify it.

        Args:
            name:         region label (e.g. "TL", "MC")
            row, col:     grid position of this region
            x1,y1,x2,y2:  pixel bounds of this region in the full frame
            gray_region:  grayscale crop of the rectified left frame
            disp_region:  raw disparity crop (same bounds); invalid <= 0
            depth_region: metric depth crop in metres (same bounds); invalid == 0

        Raises:
            ValueError: If the three input crops don't share the same shape.
        """
        if gray_region.shape[:2] != disp_region.shape[:2] or gray_region.shape[:2] != depth_region.shape[:2]:
            raise ValueError(
                f"Region crops must share shape: gray={gray_region.shape[:2]}, "
                f"disp={disp_region.shape[:2]}, depth={depth_region.shape[:2]}"
            )

        total_pixels = int(disp_region.size)
        valid_mask   = disp_region > 0
        valid_count  = int(valid_mask.sum())
        valid_pct    = 100.0 * valid_count / total_pixels if total_pixels else 0.0
        invalid_pct  = 100.0 - valid_pct
        invalid_ratio = invalid_pct / 100.0

        depth_avg_m, depth_median_m, depth_min_m, depth_max_m = self._depth_stats(
            depth_region, valid_count
        )

        texture_score      = self._texture_score(gray_region)
        entropy             = self._entropy(gray_region)
        gradient_magnitude  = self._gradient_magnitude(gray_region)
        texture_class       = self._texture_class(texture_score)

        confidence = self._confidence(valid_pct / 100.0, texture_score, entropy)

        classification = self._classify(
            invalid_ratio=invalid_ratio,
            texture_class=texture_class,
            confidence=confidence,
            depth_min_m=depth_min_m,
        )

        return RegionStats(
            name=name, row=row, col=col, x1=x1, y1=y1, x2=x2, y2=y2,
            valid_pct=valid_pct, invalid_pct=invalid_pct, invalid_ratio=invalid_ratio,
            depth_avg_m=depth_avg_m, depth_median_m=depth_median_m,
            depth_min_m=depth_min_m, depth_max_m=depth_max_m,
            texture_score=texture_score, entropy=entropy,
            gradient_magnitude=gradient_magnitude, confidence=confidence,
            texture_class=texture_class, classification=classification,
        )

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _depth_stats(self, depth_region: np.ndarray, valid_count: int):
        if valid_count < self._min_valid_pixels:
            return 0.0, 0.0, 0.0, 0.0

        valid_depths = depth_region[depth_region > 0]
        if valid_depths.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        return (
            float(np.mean(valid_depths)),
            float(np.median(valid_depths)),
            float(np.min(valid_depths)),
            float(np.max(valid_depths)),
        )

    @staticmethod
    def _texture_score(gray_region: np.ndarray) -> float:
        """Laplacian variance — classic, cheap measure of local texture/sharpness."""
        if gray_region.size == 0:
            return 0.0
        return float(cv2.Laplacian(gray_region, cv2.CV_64F).var())

    @staticmethod
    def _gradient_magnitude(gray_region: np.ndarray) -> float:
        """Mean Sobel gradient magnitude — raw average edge strength."""
        if gray_region.size == 0:
            return 0.0
        gx = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt(gx * gx + gy * gy)))

    @staticmethod
    def _entropy(gray_region: np.ndarray) -> float:
        """Shannon entropy (bits) of the region's grayscale histogram."""
        if gray_region.size == 0:
            return 0.0
        hist, _ = np.histogram(gray_region, bins=_ENTROPY_HIST_BINS, range=(0, 256))
        probs = hist.astype(np.float64) / max(1, hist.sum())
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def _texture_class(self, texture_score: float) -> TextureClass:
        if texture_score >= self._texture_high_thresh:
            return TextureClass.HIGH_TEXTURE
        if texture_score >= self._texture_low_thresh:
            return TextureClass.MEDIUM_TEXTURE
        return TextureClass.LOW_TEXTURE

    def _confidence(self, valid_frac: float, texture_score: float, entropy: float) -> float:
        """Blend disparity validity, texture, and entropy into one [0, 1] score.

        Validity is weighted highest because it directly reflects whether
        SGBM found correspondences at all; texture and entropy are
        corroborating signals for *why* validity might be low.
        """
        texture_norm = float(np.clip(texture_score / self._texture_high_thresh, 0.0, 1.0))
        entropy_norm = float(np.clip(entropy / _MAX_8BIT_ENTROPY, 0.0, 1.0))
        confidence = 0.5 * valid_frac + 0.3 * texture_norm + 0.2 * entropy_norm
        return float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def _classify(
        self,
        invalid_ratio: float,
        texture_class: TextureClass,
        confidence: float,
        depth_min_m: float,
    ) -> RegionClass:
        """Combine invalid_ratio + texture + confidence + depth — never a single threshold.

        Priority order (first match wins):
            1. PROBABLE_WALL       — very high invalid_ratio AND low texture AND low confidence
            2. LOW_TEXTURE_UNKNOWN — high invalid_ratio but doesn't meet the stricter wall criteria
            3. LOW_CONFIDENCE      — confidence too low to trust, regardless of the reason
            4. OBSTACLE            — confident reading with something close
            5. CLEAR               — confident reading, nothing close
        """
        if (
            invalid_ratio >= self._invalid_ratio_wall_thresh
            and texture_class == TextureClass.LOW_TEXTURE
            and confidence <= self._confidence_low_thresh
        ):
            return RegionClass.PROBABLE_WALL

        if invalid_ratio >= self._invalid_ratio_unknown_thresh and texture_class != TextureClass.HIGH_TEXTURE:
            return RegionClass.LOW_TEXTURE_UNKNOWN

        if confidence <= self._confidence_low_thresh:
            return RegionClass.LOW_CONFIDENCE

        if depth_min_m > 0.0 and depth_min_m < self._obstacle_caution_m:
            return RegionClass.OBSTACLE

        return RegionClass.CLEAR
