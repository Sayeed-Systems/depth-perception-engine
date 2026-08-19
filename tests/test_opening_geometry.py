"""
Phase D6 tests — geometric openings/passage structure (OpeningEvidence,
geometry.opening.build_opening_evidence).

Covers the 14 scenarios the D6 task named. Unit-level tests
(TestRealOpeningProducesEvidence, TestApproxWidth*, TestApproxHeight*,
TestContinuousWallProducesNoFalseOpening, TestInvalidDepth*,
TestInsufficientSupport*, TestPartialOpeningAtImageBoundary,
TestDeterministicOutput, TestCoordinateFrameContract) construct depth maps
directly and feed them through the REAL geometry.boundary.
build_boundary_evidence first (so BoundaryEvidence is genuine, not
hand-faked) before calling build_opening_evidence — controlled synthetic
geometry, expected evidence known in advance. Integration-level tests
(TestGeometryFrame*, TestFlagDisabled*, TestLegacyOutputsUnchanged,
TestNoBehavioralLeakage) run the real, unmodified pipeline. Point 14
("full existing suite has zero regressions") has no dedicated test here —
it's proven by the full `pytest tests/ -q` run this file is part of.
"""

import dataclasses
import inspect

import numpy as np
import pytest

import depth_perception_engine.geometry.opening as opening_module
from depth_perception_engine import (
    DepthPerceptionPipeline,
    OpeningEvidence,
    PipelineConfig,
    load_stereo_calibration,
)
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.boundary import build_boundary_evidence
from depth_perception_engine.geometry.opening import build_opening_evidence

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size
_FOCAL_LENGTH_PX = 100.0


# ===================================================================
# Helpers
# ===================================================================
def _flat_depth_map(depth=5.0, size=60):
    return np.full((size, size), depth, dtype=np.float32)


def _three_zone_depth_map(near=1.0, far=3.0, size=60):
    """Column bands: [0, size/3) near, [size/3, 2*size/3) far, [2*size/3,
    size) near — a classic "gap between two structures" pattern, exactly
    aligned to a 3-column grid partition of `size`."""
    depth_map = np.full((size, size), near, dtype=np.float32)
    third = size // 3
    depth_map[:, third:2 * third] = far
    return depth_map


def _two_zone_depth_map(near=1.0, far=3.0, size=60):
    """Left half near, right half far — aligned to a 2-column grid."""
    depth_map = np.full((size, size), near, dtype=np.float32)
    depth_map[:, size // 2:] = far
    return depth_map


def _boundary_and_opening(
    depth_map, grid_rows, grid_cols, min_support_count=5,
    depth_step_threshold_m=0.15, min_range_ratio=1.5, frame_id=FrameId.CAMERA_OPTICAL_LEFT,
):
    boundary_evidence = build_boundary_evidence(
        depth_map, frame_id, grid_rows=grid_rows, grid_cols=grid_cols,
        min_support_count=min_support_count, depth_step_threshold_m=depth_step_threshold_m,
        orientation_change_threshold_rad=0.5236,
    )
    opening_evidence = build_opening_evidence(
        boundary_evidence, depth_map, frame_id, grid_rows=grid_rows, grid_cols=grid_cols,
        min_support_count=min_support_count, min_range_ratio=min_range_ratio,
        focal_length_px=_FOCAL_LENGTH_PX,
    )
    return boundary_evidence, opening_evidence


# ===================================================================
# 1. Two supported structures with a real free-space gap produce
# opening evidence
# ===================================================================
class TestRealOpeningProducesEvidence:
    def test_gap_between_two_near_structures_is_confirmed(self):
        depth_map = _three_zone_depth_map(near=1.0, far=3.0)
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3)

        assert len(opening_evidence) == 1
        opening = opening_evidence[0]
        assert opening.row == 0
        assert opening.col_start == 1
        assert opening.col_end == 1
        assert opening.at_image_boundary is False
        assert opening.approx_range_m == pytest.approx(3.0, abs=1e-3)


# ===================================================================
# 2. Known opening width is approximately correct
# ===================================================================
class TestApproxWidthIsApproximatelyCorrect:
    def test_width_matches_pixel_extent_times_range_over_focal_length(self):
        size = 60
        depth_map = _three_zone_depth_map(near=1.0, far=3.0, size=size)
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3)
        opening = opening_evidence[0]

        expected_pixel_width = size // 3  # the middle column's own width
        expected_width_m = expected_pixel_width * opening.approx_range_m / _FOCAL_LENGTH_PX
        assert opening.x2 - opening.x1 == expected_pixel_width
        assert opening.approx_width_m == pytest.approx(expected_width_m, abs=1e-6)
        # Sanity: a real, physically reasonable positive value.
        assert opening.approx_width_m > 0.0


# ===================================================================
# 3. Known height is correct where observable/defined
# ===================================================================
class TestApproxHeightIsCorrectWhereDefined:
    def test_height_matches_row_pixel_span_times_range_over_focal_length(self):
        size = 60
        depth_map = _three_zone_depth_map(near=1.0, far=3.0, size=size)
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3)
        opening = opening_evidence[0]

        expected_height_m = size * opening.approx_range_m / _FOCAL_LENGTH_PX  # single row spans the full image height
        assert opening.y2 - opening.y1 == size
        assert opening.approx_height_m == pytest.approx(expected_height_m, abs=1e-6)


# ===================================================================
# 4. Continuous wall does not create a false opening
# ===================================================================
class TestContinuousWallProducesNoFalseOpening:
    def test_uniform_depth_produces_zero_openings(self):
        depth_map = _flat_depth_map()
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3)
        assert opening_evidence == []


# ===================================================================
# 5. Missing/invalid depth alone does not become an opening
# ===================================================================
class TestInvalidDepthDoesNotBecomeAnOpening:
    def test_zero_depth_gap_is_not_confirmed(self):
        size = 60
        depth_map = np.full((size, size), 1.0, dtype=np.float32)
        third = size // 3
        depth_map[:, third:2 * third] = 0.0  # no valid depth at all in the "gap"
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3)
        assert opening_evidence == []


# ===================================================================
# 6. Low support/unknown region remains invalid/unknown
# ===================================================================
class TestInsufficientSupportRemainsUnknown:
    def test_below_threshold_support_gap_is_not_confirmed(self):
        size = 60
        depth_map = np.full((size, size), 1.0, dtype=np.float32)
        third = size // 3
        # Middle band reads far, but only 2 valid pixels total — below
        # min_support_count=5.
        depth_map[:, third:2 * third] = 0.0
        depth_map[0, third] = 3.0
        depth_map[0, third + 1] = 3.0
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3, min_support_count=5)
        assert opening_evidence == []


# ===================================================================
# 7. Partial opening at image boundary is handled explicitly
# ===================================================================
class TestPartialOpeningAtImageBoundary:
    def test_gap_touching_last_column_is_marked_at_image_boundary(self):
        depth_map = _two_zone_depth_map(near=1.0, far=3.0)
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=2)

        assert len(opening_evidence) == 1
        opening = opening_evidence[0]
        assert opening.col_start == 1
        assert opening.col_end == 1
        assert opening.at_image_boundary is True

    def test_fully_bounded_opening_is_not_marked_at_image_boundary(self):
        depth_map = _three_zone_depth_map()
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3)
        assert opening_evidence[0].at_image_boundary is False

    def test_row_with_no_confirmed_flank_at_all_produces_no_partial_opening(self):
        # Whole row reads uniformly far relative to nothing — no confirmed
        # discontinuity anywhere -> must not be admitted even though it
        # touches both image edges.
        depth_map = _flat_depth_map(depth=5.0, size=60)
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=1)
        assert opening_evidence == []


# ===================================================================
# 8. Output is deterministic
# ===================================================================
class TestDeterministicOutput:
    def test_repeated_calls_produce_identical_evidence(self):
        depth_map = _three_zone_depth_map()
        boundary_evidence, first = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3)
        second = build_opening_evidence(
            boundary_evidence, depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=3, min_support_count=5, min_range_ratio=1.5,
            focal_length_px=_FOCAL_LENGTH_PX,
        )
        assert first == second


# ===================================================================
# 9. Coordinate/frame contract is correct
# ===================================================================
class TestCoordinateFrameContract:
    def test_frame_id_is_generic_not_hardcoded(self):
        depth_map = _three_zone_depth_map()
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3, frame_id=FrameId.BODY)
        assert opening_evidence[0].frame_id == FrameId.BODY

    def test_pixel_bounds_within_image(self):
        size = 60
        depth_map = _three_zone_depth_map(size=size)
        _, opening_evidence = _boundary_and_opening(depth_map, grid_rows=1, grid_cols=3)
        opening = opening_evidence[0]
        assert 0 <= opening.x1 < opening.x2 <= size
        assert 0 <= opening.y1 < opening.y2 <= size

    def test_invalid_construction_params_are_rejected(self):
        depth_map = _flat_depth_map()
        with pytest.raises(ValueError):
            build_opening_evidence(
                [], depth_map, FrameId.CAMERA_OPTICAL_LEFT,
                grid_rows=1, grid_cols=1, min_support_count=1,
                min_range_ratio=1.0,  # must be > 1.0
                focal_length_px=100.0,
            )
        with pytest.raises(ValueError):
            build_opening_evidence(
                [], depth_map, FrameId.CAMERA_OPTICAL_LEFT,
                grid_rows=1, grid_cols=1, min_support_count=1,
                min_range_ratio=1.5, focal_length_px=0.0,  # must be > 0
            )


# ===================================================================
# 10/11/12 — pipeline integration: GeometryFrame exposure, disabled
# path, legacy outputs
# ===================================================================
def _transform() -> RigidTransform:
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _random_pair(seed=42):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    return left, right


def _opening_config(**overrides):
    defaults = dict(
        enable_geometry_frame=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        boundary_grid_rows=2, boundary_grid_cols=2, boundary_min_support_count=5,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestGeometryFrameExposesOpeningEvidence:
    def test_present_as_a_list_type_correct(self):
        pipeline = DepthPerceptionPipeline(_opening_config(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.opening_evidence is not None
        assert isinstance(result.opening_evidence, list)
        for opening in result.opening_evidence:
            assert isinstance(opening, OpeningEvidence)

        assert result.geometry_frame is not None
        assert result.geometry_frame.opening_evidence is not None

    def test_zero_recomputation_same_object_as_result(self):
        pipeline = DepthPerceptionPipeline(_opening_config(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert result.geometry_frame.opening_evidence is result.opening_evidence


class TestFlagDisabledPreservesPreviousBehavior:
    def test_disabled_by_default(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert result.opening_evidence is None

    def test_disabled_when_boundary_geometry_off_even_if_opening_flag_on(self):
        config = PipelineConfig(
            enable_geometry_frame=True, enable_boundary_geometry=False, enable_opening_geometry=True,
        )
        pipeline = DepthPerceptionPipeline(config, _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.opening_evidence is None
        assert result.geometry_frame.opening_evidence is None

    def test_disabled_when_boundary_geometry_on_but_opening_flag_off(self):
        config = PipelineConfig(
            enable_geometry_frame=True, enable_boundary_geometry=True, enable_opening_geometry=False,
        )
        pipeline = DepthPerceptionPipeline(config, _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.opening_evidence is None
        assert result.boundary_evidence is not None  # boundary itself still ran


class TestLegacyOutputsUnchanged:
    def test_upstream_fields_identical_regardless_of_opening_geometry_flag(self):
        left, right = _random_pair()

        result_off = DepthPerceptionPipeline(
            _opening_config(enable_opening_geometry=False), _CALIBRATION,
        ).process(left, right)
        result_on = DepthPerceptionPipeline(
            _opening_config(enable_opening_geometry=True), _CALIBRATION,
        ).process(left, right)

        assert np.array_equal(result_off.disparity_map, result_on.disparity_map)
        assert np.array_equal(result_off.depth_map, result_on.depth_map)
        assert result_off.boundary_evidence == result_on.boundary_evidence
        assert result_off.confidence == result_on.confidence


# ===================================================================
# 13. No passability/traversability/vehicle semantics leak
# ===================================================================
class TestNoBehavioralLeakage:
    def test_opening_evidence_field_names_carry_no_behavioral_concept(self):
        fields = {f.name for f in dataclasses.fields(OpeningEvidence)}
        forbidden = {
            "passable", "traversable", "safe", "doorway", "window",
            "road_gap", "fly_through", "vehicle_width", "clearance",
        }
        assert not (fields & forbidden)

    def test_opening_module_does_not_import_behavioral_types(self):
        import_lines = [
            line for line in inspect.getsource(opening_module).splitlines()
            if line.startswith(("import ", "from "))
        ]
        source = "\n".join(import_lines)
        for forbidden in ("RegionClass", "NavigationDecision", "ThreatAssessor"):
            assert forbidden not in source, f"{forbidden} must not be imported by geometry/opening.py"
            assert not hasattr(opening_module, forbidden)

    def test_build_opening_evidence_signature_takes_no_platform_dimension_input(self):
        # Phase I5.1 added `min_merge_depth_diff_m` — a span-assembly
        # recalibration derived only from already-computed cell median
        # depths (see geometry.opening's own docstring), not a platform
        # dimension.
        params = set(inspect.signature(build_opening_evidence).parameters)
        assert params == {
            "boundary_evidence", "depth_map", "frame_id", "grid_rows", "grid_cols",
            "min_support_count", "min_range_ratio", "focal_length_px", "min_merge_depth_diff_m",
        }
