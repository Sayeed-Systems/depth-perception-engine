"""
Regression coverage for RegionAnalyzer's classification hard gate.

Before this test suite existed, no test in this library exercised
classification of a region with insufficient (or zero) valid disparity —
exactly the gap that let a live-confirmed defect through: a region with
zero valid disparity pixels could still classify as CLEAR (or OBSTACLE),
because texture/entropy alone could push its confidence score above the
low-confidence threshold regardless of whether SGBM found any
correspondence at all. See RegionAnalyzer._classify's docstring for the
full mechanism.
"""

import numpy as np
import pytest

from depth_perception_engine.traversability.region_analyzer import RegionAnalyzer
from depth_perception_engine.traversability.types import RegionClass, TextureClass

_H, _W = 40, 40


def _analyzer(**overrides):
    return RegionAnalyzer(**overrides)


def _zeros(h=_H, w=_W):
    return np.zeros((h, w), dtype=np.uint8)


def _textured_gray(h=_H, w=_W, seed=0):
    """A high-texture-looking grayscale crop with no real depth backing it."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    checker = ((x // 3 + y // 3) % 2) * 255
    noise = rng.integers(0, 40, size=(h, w))
    return np.clip(checker.astype(np.int32) + noise, 0, 255).astype(np.uint8)


class TestUnknownHardGate:
    """The exact live-confirmed defect and its fix."""

    def test_zero_valid_disparity_is_unknown_even_with_high_texture_image(self):
        """
        The headline regression: a region with a highly-textured-looking
        grayscale crop (which alone could previously push confidence above
        the low-confidence threshold) but ZERO valid disparity pixels must
        classify as UNKNOWN, not CLEAR/OBSTACLE — confirming the exact
        live-observed failure mode (all-black/all-white degenerate frames)
        cannot recur.
        """
        analyzer = _analyzer()
        gray = _textured_gray()
        disp = np.zeros((_H, _W), dtype=np.float32)  # every pixel invalid (<=0)
        depth = np.zeros((_H, _W), dtype=np.float32)

        stats = analyzer.analyze("MC", 1, 1, 0, 0, _W, _H, gray, disp, depth)

        assert stats.classification == RegionClass.UNKNOWN
        assert stats.classification != RegionClass.CLEAR
        assert stats.classification != RegionClass.OBSTACLE

    def test_valid_count_just_below_threshold_is_unknown(self):
        analyzer = _analyzer(min_valid_pixels=20)
        gray = _textured_gray()
        disp = np.zeros((_H, _W), dtype=np.float32)
        disp[0, :19] = 5.0  # 19 valid pixels, one short of the threshold
        depth = np.zeros((_H, _W), dtype=np.float32)

        stats = analyzer.analyze("MC", 1, 1, 0, 0, _W, _H, gray, disp, depth)

        assert stats.classification == RegionClass.UNKNOWN

    def test_valid_count_at_threshold_is_not_forced_unknown(self):
        """Crossing the threshold must actually change the outcome — proves
        the gate is a real boundary, not an always-true/always-false stub."""
        analyzer = _analyzer(min_valid_pixels=20)
        gray = _textured_gray()
        disp = np.full((_H, _W), 5.0, dtype=np.float32)  # every pixel valid
        depth = np.full((_H, _W), 2.0, dtype=np.float32)  # far away, clear

        stats = analyzer.analyze("MC", 1, 1, 0, 0, _W, _H, gray, disp, depth)

        assert stats.classification != RegionClass.UNKNOWN

    def test_unknown_takes_priority_over_probable_wall_criteria(self):
        """A region that would otherwise match PROBABLE_WALL's criteria
        (high invalid_ratio + low texture) must still report UNKNOWN, not
        PROBABLE_WALL, when valid_count is under the hard floor — the gate
        is checked first, unconditionally, per _classify's priority order."""
        analyzer = _analyzer(min_valid_pixels=20)
        gray = _zeros()  # flat -> LOW_TEXTURE, low confidence
        disp = np.zeros((_H, _W), dtype=np.float32)
        depth = np.zeros((_H, _W), dtype=np.float32)

        stats = analyzer.analyze("MC", 1, 1, 0, 0, _W, _H, gray, disp, depth)

        assert stats.classification == RegionClass.UNKNOWN

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_unknown_holds_across_varied_high_confidence_looking_textures(self, seed):
        """Sweeps several synthetic textures that would each independently
        drive texture/entropy-derived confidence up, to confirm the gate
        holds regardless of the specific image content — not just one
        lucky/unlucky synthetic pattern."""
        analyzer = _analyzer()
        gray = _textured_gray(seed=seed)
        disp = np.zeros((_H, _W), dtype=np.float32)
        depth = np.zeros((_H, _W), dtype=np.float32)

        stats = analyzer.analyze("MC", 1, 1, 0, 0, _W, _H, gray, disp, depth)

        assert stats.classification == RegionClass.UNKNOWN


class TestClassifyDirectly:
    """Lower-level coverage of _classify()'s priority order via the public analyze() path."""

    def test_sufficient_valid_disparity_and_near_depth_is_obstacle(self):
        analyzer = _analyzer(obstacle_caution_m=0.60)
        gray = _textured_gray()
        disp = np.full((_H, _W), 40.0, dtype=np.float32)
        depth = np.full((_H, _W), 0.30, dtype=np.float32)  # closer than caution_m

        stats = analyzer.analyze("MC", 1, 1, 0, 0, _W, _H, gray, disp, depth)

        assert stats.classification == RegionClass.OBSTACLE

    def test_sufficient_valid_disparity_and_far_depth_is_clear(self):
        analyzer = _analyzer(obstacle_caution_m=0.60)
        gray = _textured_gray()
        disp = np.full((_H, _W), 5.0, dtype=np.float32)
        depth = np.full((_H, _W), 2.0, dtype=np.float32)  # well beyond caution_m

        stats = analyzer.analyze("MC", 1, 1, 0, 0, _W, _H, gray, disp, depth)

        assert stats.classification == RegionClass.CLEAR
