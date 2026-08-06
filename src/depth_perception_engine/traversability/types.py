"""
Shared types for the traversability layer.

These are pure data definitions (enums + a result dataclass) with no
behaviour, so RegionAnalyzer, SceneInterpreter, and any visualisation code
can all depend on them without circular imports.
"""

from dataclasses import dataclass
from enum import Enum


class TextureClass(str, Enum):
    """Local texture richness of a region, used to judge stereo-matching reliability."""
    HIGH_TEXTURE   = "HIGH_TEXTURE"
    MEDIUM_TEXTURE = "MEDIUM_TEXTURE"
    LOW_TEXTURE    = "LOW_TEXTURE"


class RegionClass(str, Enum):
    """Navigation-relevant classification of one grid region.

    UNKNOWN              — HARD FLOOR: fewer valid disparity pixels than
                            RegionAnalyzer's min_valid_pixels threshold.
                            Returned unconditionally before any other signal
                            (texture, entropy, confidence) is even
                            considered — see RegionAnalyzer._classify. A
                            region with insufficient stereo evidence has, by
                            definition, no basis for any of the classes
                            below and must never be reported as one of them.
                            SceneInterpreter treats UNKNOWN as ambiguous, so
                            it can never itself produce MOVE_FORWARD.
    CLEAR                — confident reading, no obstacle within range
    OBSTACLE             — confident reading, something close
    LOW_TEXTURE_UNKNOWN  — too little texture/disparity to judge either way
    PROBABLE_WALL        — very high invalid-disparity + very low texture +
                            very low confidence: a textureless near surface
                            (e.g. a blank wall) that stereo cannot range, but
                            which must still be treated as a likely obstacle
    LOW_CONFIDENCE       — generic catch-all for an unreliable reading that
                            doesn't fit the more specific categories above
    """
    UNKNOWN             = "UNKNOWN"
    CLEAR               = "CLEAR"
    OBSTACLE            = "OBSTACLE"
    LOW_TEXTURE_UNKNOWN = "LOW_TEXTURE_UNKNOWN"
    PROBABLE_WALL       = "PROBABLE_WALL"
    LOW_CONFIDENCE      = "LOW_CONFIDENCE"


class NavigationDecision(str, Enum):
    """Global, frame-level recommendation derived from the full region grid."""
    MOVE_FORWARD    = "MOVE_FORWARD"
    TURN_LEFT       = "TURN_LEFT"
    TURN_RIGHT      = "TURN_RIGHT"
    SLOW_DOWN       = "SLOW_DOWN"
    STOP            = "STOP"
    HOVER           = "HOVER"
    ROTATE_AND_SCAN = "ROTATE_AND_SCAN"


@dataclass(frozen=True)
class RegionStats:
    """All computed signals and the resulting classification for one grid cell."""
    name: str
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int

    valid_pct:   float   # % of pixels with usable disparity
    invalid_pct: float   # 100 - valid_pct
    invalid_ratio: float  # invalid_pct / 100, kept as its own field per spec

    # Absolute pixel counts backing valid_pct/invalid_pct above. Exposed
    # separately (not just derivable from valid_pct + region bounds)
    # because valid_pct is a lossy float division of these two integers —
    # a consumer that needs the exact count RegionAnalyzer itself gated
    # min_valid_pixels on (e.g. an independent, out-of-process validity
    # check) must not have to reconstruct it via a float round-trip.
    valid_count:  int
    total_pixels: int

    depth_avg_m:    float
    depth_median_m: float
    depth_min_m:    float
    depth_max_m:    float

    texture_score:      float
    entropy:            float
    gradient_magnitude: float
    confidence:         float

    texture_class:  TextureClass
    classification: RegionClass
