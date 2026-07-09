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
