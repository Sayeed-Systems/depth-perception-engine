"""
DepthPerceptionPipeline — instantiation, process(), and output structure.

Uses synthetic (random-noise) NumPy images throughout — the point of these
tests is proving the pipeline runs end-to-end and returns a well-formed
DepthPerceptionResult, not validating depth accuracy against real geometry
(random noise has no real stereo correspondence to recover in the first
place).
"""

import numpy as np
import pytest

from depth_perception_engine.models import (
    BeamReading,
    DepthPerceptionResult,
    ObstacleAssessment,
    PipelineHealth,
    StereoObservation,
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


class TestFromConfig:
    def test_from_config_is_equivalent_to_constructor(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline.from_config(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right)
        assert isinstance(result, DepthPerceptionResult)


class TestValidityMasks:
    def test_masks_match_disparity_and_depth_sign_convention(
        self, config, calibration, stereo_pair,
    ):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right)

        assert result.valid_disparity_mask.shape == result.disparity_map.shape
        assert result.valid_disparity_mask.dtype == np.bool_
        np.testing.assert_array_equal(
            result.valid_disparity_mask, result.disparity_map > 0
        )

        assert result.valid_depth_mask.shape == result.depth_map.shape
        assert result.valid_depth_mask.dtype == np.bool_
        np.testing.assert_array_equal(result.valid_depth_mask, result.depth_map > 0)


class TestTimestampPassthrough:
    def test_left_timestamp_wins_when_both_given(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0, right_timestamp=2.0)
        assert result.timestamp == 1.0

    def test_right_timestamp_used_when_left_missing(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, right_timestamp=2.0)
        assert result.timestamp == 2.0

    def test_timestamp_defaults_to_none(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right)
        assert result.timestamp is None


class TestProcessObservation:
    def test_process_observation_matches_direct_process_call(
        self, config, calibration, stereo_pair,
    ):
        left, right = stereo_pair
        obs = StereoObservation(
            left_image=left, right_image=right,
            left_timestamp=5.0, right_timestamp=5.1, frame_id="f0",
        )
        pipeline = DepthPerceptionPipeline(config, calibration)
        result = pipeline.process_observation(obs)

        assert isinstance(result, DepthPerceptionResult)
        assert result.timestamp == 5.0


class TestLifecycle:
    def test_health_before_any_process_call(self, config, calibration):
        pipeline = DepthPerceptionPipeline(config, calibration)
        health = pipeline.health()

        assert isinstance(health, PipelineHealth)
        assert health.is_closed is False
        assert health.frames_processed == 0
        assert health.last_confidence is None
        assert health.last_processing_time_ms is None

    def test_health_reflects_last_processed_frame(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair

        pipeline.process(left, right)
        pipeline.process(left, right)
        health = pipeline.health()

        assert health.frames_processed == 2
        assert health.last_confidence is not None
        assert health.last_processing_time_ms is not None

    def test_reset_clears_frame_count_and_last_metrics(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right)

        pipeline.reset()
        health = pipeline.health()

        assert health.frames_processed == 0
        assert health.last_confidence is None
        assert health.last_processing_time_ms is None

    def test_reset_does_not_affect_config_or_calibration(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right)

        pipeline.reset()

        assert pipeline.config is config
        assert pipeline.calibration is calibration
        # Still fully usable after reset.
        result = pipeline.process(left, right)
        assert isinstance(result, DepthPerceptionResult)

    def test_close_then_process_raises(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.close()

        with pytest.raises(RuntimeError):
            pipeline.process(left, right)

    def test_close_then_reset_raises(self, config, calibration):
        pipeline = DepthPerceptionPipeline(config, calibration)
        pipeline.close()

        with pytest.raises(RuntimeError):
            pipeline.reset()

    def test_close_is_idempotent(self, config, calibration):
        pipeline = DepthPerceptionPipeline(config, calibration)
        pipeline.close()
        pipeline.close()  # must not raise

        assert pipeline.health().is_closed is True

    def test_close_reflected_in_health(self, config, calibration):
        pipeline = DepthPerceptionPipeline(config, calibration)
        pipeline.close()

        assert pipeline.health().is_closed is True
