"""
Unit tests for DistanceReader.

Uses hand-constructed depth maps with known values throughout — the point
is proving ROI extraction, invalid-value filtering, and the median-based
distance estimate are numerically correct, not just that the methods run.
"""

import numpy as np
import pytest

from depth_perception_engine.depth.distance_reader import DistanceReader


class TestConstruction:
    def test_default_roi_size(self):
        DistanceReader()  # must not raise

    def test_roi_width_below_one_raises(self):
        with pytest.raises(ValueError):
            DistanceReader(roi_width=0)

    def test_roi_height_below_one_raises(self):
        with pytest.raises(ValueError):
            DistanceReader(roi_height=0)


class TestExtractRoi:
    def test_extracts_centred_patch_of_known_values(self):
        # 10x10 depth map, value == column index (0..9) in every row, so the
        # centre ROI's expected content is fully known in advance.
        depth_map = np.tile(np.arange(10, dtype=np.float32), (10, 1))
        reader = DistanceReader(roi_width=4, roi_height=4)

        roi = reader.extract_roi(depth_map)

        x1, y1, x2, y2 = reader.get_roi_bounds(depth_map)
        expected = depth_map[y1:y2, x1:x2].flatten()
        np.testing.assert_array_equal(roi, expected)

    def test_roi_clamped_when_larger_than_image(self):
        depth_map = np.ones((5, 5), dtype=np.float32) * 2.0
        reader = DistanceReader(roi_width=100, roi_height=100)

        roi = reader.extract_roi(depth_map)

        assert roi.size == 25
        assert np.all(roi == 2.0)

    def test_none_depth_map_raises_value_error(self):
        reader = DistanceReader()
        with pytest.raises(ValueError):
            reader.extract_roi(None)

    def test_non_ndarray_raises_type_error(self):
        reader = DistanceReader()
        with pytest.raises(TypeError):
            reader.extract_roi([[1, 2], [3, 4]])

    def test_1d_array_raises_value_error(self):
        reader = DistanceReader()
        with pytest.raises(ValueError):
            reader.extract_roi(np.zeros(10, dtype=np.float32))


class TestGetRoiBounds:
    def test_bounds_are_within_image(self):
        depth_map = np.zeros((240, 320), dtype=np.float32)
        reader = DistanceReader(roi_width=80, roi_height=80)

        x1, y1, x2, y2 = reader.get_roi_bounds(depth_map)

        assert 0 <= x1 < x2 <= 320
        assert 0 <= y1 < y2 <= 240
        assert (x2 - x1) == 80
        assert (y2 - y1) == 80


class TestFilterDepthValues:
    def test_removes_zeros_nan_inf_and_negatives(self):
        values = np.array(
            [1.0, 0.0, -1.0, np.nan, np.inf, -np.inf, 2.5, 0.001],
            dtype=np.float32,
        )

        filtered = DistanceReader.filter_depth_values(values)

        np.testing.assert_array_equal(np.sort(filtered), np.sort(np.array([1.0, 2.5, 0.001], dtype=np.float32)))

    def test_all_invalid_returns_empty_array(self):
        values = np.array([0.0, -1.0, np.nan, np.inf], dtype=np.float32)

        filtered = DistanceReader.filter_depth_values(values)

        assert filtered.size == 0

    def test_all_valid_passes_through_unchanged(self):
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        filtered = DistanceReader.filter_depth_values(values)

        np.testing.assert_array_equal(filtered, values)


class TestComputeDistance:
    def test_known_values_produce_known_median(self):
        # Median of [1, 2, 3, 4, 5] is 3.0 exactly.
        values = np.array([5.0, 1.0, 3.0, 2.0, 4.0], dtype=np.float32)

        stats = DistanceReader.compute_distance(values)

        assert stats["distance_meters"] == pytest.approx(3.0)
        assert stats["distance_centimeters"] == pytest.approx(300.0)
        assert stats["median_depth"] == pytest.approx(3.0)
        assert stats["min_depth"] == pytest.approx(1.0)
        assert stats["max_depth"] == pytest.approx(5.0)
        assert stats["valid_pixel_count"] == 5

    def test_single_value(self):
        stats = DistanceReader.compute_distance(np.array([2.5], dtype=np.float32))

        assert stats["distance_meters"] == pytest.approx(2.5)
        assert stats["min_depth"] == pytest.approx(2.5)
        assert stats["max_depth"] == pytest.approx(2.5)
        assert stats["valid_pixel_count"] == 1

    def test_empty_array_returns_all_zeros(self):
        stats = DistanceReader.compute_distance(np.array([], dtype=np.float32))

        assert stats == {
            "distance_meters": 0.0,
            "distance_centimeters": 0.0,
            "min_depth": 0.0,
            "max_depth": 0.0,
            "median_depth": 0.0,
            "valid_pixel_count": 0,
        }

    def test_centimeters_is_meters_times_100(self):
        stats = DistanceReader.compute_distance(np.array([1.0, 1.0], dtype=np.float32))

        assert stats["distance_centimeters"] == pytest.approx(stats["distance_meters"] * 100.0)


class TestReadDistance:
    def test_full_pipeline_on_uniform_depth_map(self):
        depth_map = np.full((40, 40), 1.5, dtype=np.float32)
        reader = DistanceReader(roi_width=10, roi_height=10)

        result = reader.read_distance(depth_map)

        assert result["distance_meters"] == pytest.approx(1.5)
        assert result["valid_pixel_count"] == 100  # 10x10 ROI, all valid

    def test_roi_with_no_valid_depth_returns_zeros(self):
        depth_map = np.zeros((40, 40), dtype=np.float32)  # all invalid (== 0)
        reader = DistanceReader(roi_width=10, roi_height=10)

        result = reader.read_distance(depth_map)

        assert result["distance_meters"] == 0.0
        assert result["valid_pixel_count"] == 0

    def test_mixed_valid_invalid_only_counts_valid(self):
        depth_map = np.zeros((10, 10), dtype=np.float32)
        depth_map[4:6, 4:6] = 2.0  # 4 valid pixels inside a 4x4 centred ROI
        reader = DistanceReader(roi_width=4, roi_height=4)

        result = reader.read_distance(depth_map)

        assert result["distance_meters"] == pytest.approx(2.0)
        assert result["valid_pixel_count"] == 4

    def test_none_depth_map_raises_value_error(self):
        reader = DistanceReader()
        with pytest.raises(ValueError):
            reader.read_distance(None)


class TestRepr:
    def test_repr_contains_roi_dimensions(self):
        reader = DistanceReader(roi_width=64, roi_height=48)
        text = repr(reader)
        assert "64" in text
        assert "48" in text
