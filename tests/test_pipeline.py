"""
DepthPerceptionPipeline — instantiation, process(), and output structure.

Uses synthetic (random-noise) NumPy images throughout — the point of these
tests is proving the pipeline runs end-to-end and returns a well-formed
DepthPerceptionResult, not validating depth accuracy against real geometry
(random noise has no real stereo correspondence to recover in the first
place).
"""

import numpy as np

from depth_perception_engine.models import (
    BeamReading,
    DepthPerceptionResult,
    ObstacleAssessment,
    TraversabilityResult,
)
from depth_perception_engine.pipeline import DepthPerceptionPipeline
from depth_perception_engine.traversability.types import NavigationDecision, RegionStats


def test_pipeline_can_be_instantiated(config, calibration):
    pipeline = DepthPerceptionPipeline(config, calibration)
    assert pipeline.config is config
    assert pipeline.calibration is calibration


def test_process_can_be_called(config, calibration, stereo_pair):
    pipeline = DepthPerceptionPipeline(config, calibration)
    left, right = stereo_pair
    result = pipeline.process(left, right)
    assert isinstance(result, DepthPerceptionResult)


def test_output_is_a_structured_object_not_a_dict(config, calibration, stereo_pair):
    pipeline = DepthPerceptionPipeline(config, calibration)
    left, right = stereo_pair
    result = pipeline.process(left, right)

    assert not isinstance(result, dict)
    assert isinstance(result.traversability_mask, TraversabilityResult)
    assert isinstance(result.obstacles, ObstacleAssessment)
    for beam in result.obstacles.beams:
        assert isinstance(beam, BeamReading)
    for region in result.traversability_mask.regions.values():
        assert isinstance(region, RegionStats)
    assert isinstance(result.traversability_mask.decision, NavigationDecision)


def test_output_field_shapes_and_types(config, calibration, stereo_pair):
    pipeline = DepthPerceptionPipeline(config, calibration)
    left, right = stereo_pair
    result = pipeline.process(left, right)

    width, height = calibration.image_size
    assert result.disparity_map.shape == (height, width)
    assert result.depth_map.shape == (height, width)
    assert isinstance(result.disparity_map, np.ndarray)
    assert isinstance(result.depth_map, np.ndarray)

    assert len(result.obstacles.beams) == config.n_beams
    assert len(result.traversability_mask.regions) == (
        config.traversability_grid_rows * config.traversability_grid_cols
    )

    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0

    assert isinstance(result.processing_time_ms, float)
    assert result.processing_time_ms >= 0.0


def test_process_can_be_called_repeatedly(config, calibration, stereo_pair):
    """Confirms the pipeline is reusable across frames — this is the whole
    point of it being a persistent object rather than a one-shot function
    (see obstacles.ThreatAssessor's EMA/debounce state)."""
    pipeline = DepthPerceptionPipeline(config, calibration)
    left, right = stereo_pair
    for _ in range(3):
        result = pipeline.process(left, right)
        assert isinstance(result, DepthPerceptionResult)


def test_mismatched_stereo_pair_is_rejected(config, calibration, stereo_pair):
    pipeline = DepthPerceptionPipeline(config, calibration)
    left, _right = stereo_pair
    wrong_size_right = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        pipeline.process(left, wrong_size_right)
        assert False, "expected a ValueError for mismatched stereo pair shapes"
    except ValueError:
        pass


class TestRectificationFailureInvalidatesTheFrame:
    """
    Regression coverage for Issue C (remediation): a rectification failure
    must propagate as an exception — invalidating the whole frame — instead
    of silently falling back to running SGBM/depth/traversability on the
    unrectified pair. Before this fix, the frame below would have returned
    a normal-looking DepthPerceptionResult computed from unrectified
    images, with no signal anything was wrong.
    """

    def test_frame_size_mismatched_against_calibration_raises_not_silently_continues(
        self, config, calibration,
    ):
        pipeline = DepthPerceptionPipeline(config, calibration, rectify=True)
        # Same shape as each other (passes require_matching_stereo_pair),
        # but NOT the calibration's image_size (fails rectification's own
        # frame-size check against the loaded calibration).
        wrong_w, wrong_h = calibration.image_size[0] // 2, calibration.image_size[1] // 2
        rng = np.random.default_rng(7)
        left = rng.integers(0, 255, (wrong_h, wrong_w, 3), dtype=np.uint8)
        right = rng.integers(0, 255, (wrong_h, wrong_w, 3), dtype=np.uint8)

        try:
            pipeline.process(left, right)
            assert False, (
                "expected rectification failure to raise, not silently "
                "process the unrectified pair"
            )
        except ValueError as exc:
            assert "does not match calibrated size" in str(exc)

    def test_rectify_false_bypasses_rectification_entirely_and_still_succeeds(
        self, config, calibration, stereo_pair,
    ):
        """Sanity check: rectify=False is a deliberate, documented opt-out
        (caller supplies pre-rectified images) — must still work normally,
        proving the fix only removed the silent in-process fallback, not
        the legitimate rectify=False path."""
        pipeline = DepthPerceptionPipeline(config, calibration, rectify=False)
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert isinstance(result, DepthPerceptionResult)
