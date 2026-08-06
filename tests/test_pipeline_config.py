"""Unit tests for PipelineConfig's __post_init__ validation."""

import pytest

from depth_perception_engine.config import PipelineConfig


def test_default_config_is_valid():
    PipelineConfig()  # must not raise


@pytest.mark.parametrize(
    "field, value",
    [
        ("num_disparities", 100),   # not divisible by 16
        ("block_size", 12),          # even
        ("block_size", 0),           # not positive
        ("caution_distance_m", 0.0),
        ("caution_distance_m", -1.0),
        ("n_beams", 0),
        ("obstacle_percentile", -1),
        ("obstacle_percentile", 101),
        ("obstacle_min_valid_px", -1),
        ("obstacle_blocked_invalid_ratio", -0.1),
        ("obstacle_blocked_invalid_ratio", 1.1),
        ("obstacle_ema_alpha", 0.0),
        ("obstacle_ema_alpha", 1.1),
        ("obstacle_debounce_frames", 0),
        ("obstacle_dead_zone_px", -1),
        ("traversability_grid_rows", 0),
        ("traversability_grid_cols", 0),
        ("traversability_ambiguous_fraction_thresh", -0.1),
        ("traversability_ambiguous_fraction_thresh", 1.1),
    ],
)
def test_invalid_field_raises_value_error(field, value):
    with pytest.raises(ValueError):
        PipelineConfig(**{field: value})


def test_caution_must_be_less_than_clear():
    with pytest.raises(ValueError):
        PipelineConfig(caution_distance_m=1.5, clear_distance_m=1.0)


def test_caution_equal_to_clear_is_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(caution_distance_m=1.0, clear_distance_m=1.0)


def test_obstacle_dead_zone_px_none_is_valid():
    PipelineConfig(obstacle_dead_zone_px=None)  # must not raise


def test_error_message_is_readable():
    with pytest.raises(ValueError, match="num_disparities"):
        PipelineConfig(num_disparities=100)
