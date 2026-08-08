"""
Unit tests for geometry.classify_geometry_quality / GeometryQuality —
Level 3, Phase E6.

Deliberately based on exactly one field (GeometryMetrics.valid_fraction),
not a blended score — see geometry/geometry_metrics.py's module docstring
for why. Boundary conditions get explicit coverage per Task 5's own
requirement.
"""

import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.geometry import GeometryMetrics, GeometryQuality, classify_geometry_quality


def _metrics(valid_fraction: float) -> GeometryMetrics:
    return GeometryMetrics(
        min_obstacle_distance_m=None, mean_free_space_m=None,
        point_count=0, valid_fraction=valid_fraction,
    )


class TestClassificationTiers:
    def test_full_coverage_is_healthy(self):
        assert classify_geometry_quality(_metrics(1.0), 0.5, 0.05) == GeometryQuality.HEALTHY

    def test_zero_coverage_is_no_usable_geometry(self):
        assert classify_geometry_quality(_metrics(0.0), 0.5, 0.05) == GeometryQuality.NO_USABLE_GEOMETRY

    def test_mid_coverage_is_degraded(self):
        assert classify_geometry_quality(_metrics(0.25), 0.5, 0.05) == GeometryQuality.DEGRADED


class TestBoundaryConditions:
    def test_exactly_at_healthy_threshold_is_healthy(self):
        """Lower bound of HEALTHY is inclusive."""
        assert classify_geometry_quality(_metrics(0.5), 0.5, 0.05) == GeometryQuality.HEALTHY

    def test_just_below_healthy_threshold_is_degraded(self):
        assert classify_geometry_quality(_metrics(0.499999), 0.5, 0.05) == GeometryQuality.DEGRADED

    def test_exactly_at_degraded_threshold_is_degraded(self):
        """Lower bound of DEGRADED is inclusive."""
        assert classify_geometry_quality(_metrics(0.05), 0.5, 0.05) == GeometryQuality.DEGRADED

    def test_just_below_degraded_threshold_is_no_usable_geometry(self):
        assert classify_geometry_quality(_metrics(0.049999), 0.5, 0.05) == GeometryQuality.NO_USABLE_GEOMETRY

    def test_equal_thresholds_collapse_degraded_tier_to_the_empty_set(self):
        """healthy == degraded is a legal (if unusual) configuration — the
        HEALTHY check runs first and is itself inclusive, so DEGRADED
        becomes unreachable at that exact value: >= threshold is always
        HEALTHY, < threshold is always NO_USABLE_GEOMETRY."""
        assert classify_geometry_quality(_metrics(0.3), 0.3, 0.3) == GeometryQuality.HEALTHY
        assert classify_geometry_quality(_metrics(0.3 + 1e-6), 0.3, 0.3) == GeometryQuality.HEALTHY
        assert classify_geometry_quality(_metrics(0.3 - 1e-6), 0.3, 0.3) == GeometryQuality.NO_USABLE_GEOMETRY

    def test_thresholds_at_the_extremes_0_and_1(self):
        assert classify_geometry_quality(_metrics(0.0), 1.0, 0.0) == GeometryQuality.DEGRADED
        assert classify_geometry_quality(_metrics(1.0), 1.0, 0.0) == GeometryQuality.HEALTHY


class TestThresholdValidation:
    def test_rejects_degraded_greater_than_healthy(self):
        with pytest.raises(ValueError, match="degraded_min_valid_fraction"):
            classify_geometry_quality(_metrics(0.5), healthy_min_valid_fraction=0.1, degraded_min_valid_fraction=0.5)

    def test_rejects_out_of_range_healthy_threshold(self):
        with pytest.raises(ValueError, match="healthy_min_valid_fraction"):
            classify_geometry_quality(_metrics(0.5), healthy_min_valid_fraction=1.5, degraded_min_valid_fraction=0.05)

    def test_rejects_out_of_range_degraded_threshold(self):
        with pytest.raises(ValueError, match="degraded_min_valid_fraction"):
            classify_geometry_quality(_metrics(0.5), healthy_min_valid_fraction=0.5, degraded_min_valid_fraction=-0.1)


class TestNotAutoWired:
    def test_not_a_field_on_geometry_metrics(self):
        """Deliberately opt-in — not stored anywhere, not a new frozen field."""
        assert "quality" not in GeometryMetrics.__dataclass_fields__
        assert "geometry_quality" not in GeometryMetrics.__dataclass_fields__


class TestConfigThresholds:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.geometry_healthy_min_valid_fraction == 0.5
        assert config.geometry_degraded_min_valid_fraction == 0.05

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError, match="geometry_degraded_min_valid_fraction"):
            PipelineConfig(geometry_healthy_min_valid_fraction=0.1, geometry_degraded_min_valid_fraction=0.5)

    def test_rejects_out_of_range_healthy(self):
        with pytest.raises(ValueError, match="geometry_healthy_min_valid_fraction"):
            PipelineConfig(geometry_healthy_min_valid_fraction=1.5)

    def test_rejects_out_of_range_degraded(self):
        with pytest.raises(ValueError, match="geometry_degraded_min_valid_fraction"):
            PipelineConfig(geometry_degraded_min_valid_fraction=-0.1)

    def test_equal_thresholds_are_accepted(self):
        PipelineConfig(geometry_healthy_min_valid_fraction=0.3, geometry_degraded_min_valid_fraction=0.3)  # must not raise

    def test_config_thresholds_feed_classify_geometry_quality_directly(self):
        """Integration sanity: the config fields are literally what a
        caller is expected to pass through unchanged."""
        config = PipelineConfig()
        metrics = _metrics(0.5)
        result = classify_geometry_quality(
            metrics, config.geometry_healthy_min_valid_fraction, config.geometry_degraded_min_valid_fraction,
        )
        assert result == GeometryQuality.HEALTHY
