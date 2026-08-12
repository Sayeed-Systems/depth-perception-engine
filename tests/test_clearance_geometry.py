"""
Phase D7 tests — directional clearance refinement (ClearanceEvidence's
coverage/support and calibrated bearing fields, plus the minimal
ThreatAssessor/BeamReading compatibility extension that feeds them).

Covers the 13 scenarios the D7 task named. Unit-level tests
(TestBearing*, TestCoverage*, TestDistanceUnchanged,
TestThreatAssessorCompatibility) use analytically controlled calibration
values and hand-built depth maps/ObstacleAssessments — expected results
known in advance, not merely "a value exists". Integration-level tests
(TestGeometryFrameExposesRefinedClearanceEvidence, TestFlagDisabled*,
TestNoBehavioralLeakage) run the real, unmodified pipeline. Point 13
("full existing suite has zero regressions") has no dedicated test here —
it's proven by the full `pytest tests/ -q` run this file is part of.
"""

import dataclasses
import inspect
import math

import numpy as np
import pytest

import depth_perception_engine.geometry.provider as provider_module
from depth_perception_engine import (
    ClearanceEvidence,
    ClearanceSupportState,
    DepthPerceptionPipeline,
    PipelineConfig,
    load_stereo_calibration,
)
from depth_perception_engine.frames import FrameId
from depth_perception_engine.fusion.result_builder import _bearing_rad, build_clearance_evidence
from depth_perception_engine.models.result import BeamReading, ObstacleAssessment
from depth_perception_engine.obstacles.threat_assessment import ThreatAssessor

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size
_FOCAL_LENGTH_PX = 200.0
_PRINCIPAL_POINT_X_PX = 160.0


# ===================================================================
# Helpers
# ===================================================================
def _beam(index=0, x1=0, x2=10, distance_m=2.0, status="CLEAR", valid_count=10, total_pixels=10):
    return BeamReading(
        index=index, x1=x1, x2=x2, distance_m=distance_m, status=status,
        valid_count=valid_count, total_pixels=total_pixels,
    )


def _clearance_for(beam, min_coverage_fraction=0.5):
    obstacles = ObstacleAssessment(beams=[beam], safest_beam=None)
    return build_clearance_evidence(
        obstacles, FrameId.CAMERA_OPTICAL_LEFT,
        _FOCAL_LENGTH_PX, _PRINCIPAL_POINT_X_PX, min_coverage_fraction,
    )[0]


def _depth_map(fill_m, h=60, w=200):
    return np.full((h, w), fill_m, dtype=np.float32)


def _zeros(h=60, w=200):
    return np.zeros((h, w), dtype=np.float32)


# ===================================================================
# 1/2/3/4 — bearing derivation
# ===================================================================
class TestBearingDerivation:
    def test_known_pixel_produces_expected_bearing(self):
        # Analytically exact: atan2(200 - 160, 200) with these controlled values.
        expected = math.atan2(40.0, 200.0)
        assert _bearing_rad(200.0, _PRINCIPAL_POINT_X_PX, _FOCAL_LENGTH_PX) == pytest.approx(expected)

    def test_optical_axis_pixel_produces_zero_bearing(self):
        assert _bearing_rad(_PRINCIPAL_POINT_X_PX, _PRINCIPAL_POINT_X_PX, _FOCAL_LENGTH_PX) == pytest.approx(0.0, abs=1e-9)

    def test_left_of_principal_point_is_negative(self):
        bearing = _bearing_rad(_PRINCIPAL_POINT_X_PX - 50.0, _PRINCIPAL_POINT_X_PX, _FOCAL_LENGTH_PX)
        assert bearing < 0.0

    def test_right_of_principal_point_is_positive(self):
        bearing = _bearing_rad(_PRINCIPAL_POINT_X_PX + 50.0, _PRINCIPAL_POINT_X_PX, _FOCAL_LENGTH_PX)
        assert bearing > 0.0

    def test_longer_focal_length_produces_smaller_angle_for_the_same_offset(self):
        near_bearing = _bearing_rad(260.0, _PRINCIPAL_POINT_X_PX, focal_length_px=100.0)
        far_bearing = _bearing_rad(260.0, _PRINCIPAL_POINT_X_PX, focal_length_px=400.0)
        assert far_bearing < near_bearing  # narrower FOV per pixel -> smaller angle for the same pixel offset

    def test_shifting_principal_point_shifts_the_zero_crossing(self):
        # With cx moved to 200, pixel 200 must now read (approximately) zero.
        assert _bearing_rad(200.0, principal_point_x_px=200.0, focal_length_px=_FOCAL_LENGTH_PX) == pytest.approx(0.0, abs=1e-9)

    def test_clearance_evidence_bearing_bounds_are_monotonic_and_center_between(self):
        beam = _beam(x1=100, x2=200)
        clearance = _clearance_for(beam)
        assert clearance.bearing_min_rad <= clearance.bearing_center_rad <= clearance.bearing_max_rad

    def test_bearing_fields_computed_from_documented_pixel_columns(self):
        beam = _beam(x1=100, x2=200)
        clearance = _clearance_for(beam)
        assert clearance.bearing_center_rad == pytest.approx(
            _bearing_rad(150.0, _PRINCIPAL_POINT_X_PX, _FOCAL_LENGTH_PX)
        )
        assert clearance.bearing_min_rad == pytest.approx(
            _bearing_rad(100.0, _PRINCIPAL_POINT_X_PX, _FOCAL_LENGTH_PX)
        )
        assert clearance.bearing_max_rad == pytest.approx(
            _bearing_rad(200.0, _PRINCIPAL_POINT_X_PX, _FOCAL_LENGTH_PX)
        )


# ===================================================================
# 5/6/7 — coverage / support
# ===================================================================
class TestCoverageAndSupport:
    def test_full_coverage_is_numerically_correct(self):
        beam = _beam(distance_m=2.0, status="CLEAR", valid_count=10, total_pixels=10)
        clearance = _clearance_for(beam)
        assert clearance.valid_count == 10
        assert clearance.total_pixels == 10
        assert clearance.coverage_fraction == pytest.approx(1.0)
        assert clearance.support_state == ClearanceSupportState.SUPPORTED

    def test_zero_valid_data_is_explicitly_no_evidence(self):
        beam = _beam(distance_m=0.0, status="NO_DATA", valid_count=0, total_pixels=10)
        clearance = _clearance_for(beam)
        assert clearance.has_evidence is False
        assert clearance.nearest_distance_m is None
        assert clearance.valid_count == 0
        assert clearance.coverage_fraction == pytest.approx(0.0)
        assert clearance.support_state == ClearanceSupportState.NO_EVIDENCE
        # A missing measurement never becomes infinite/free clearance.
        assert clearance.nearest_distance_m != float("inf")

    def test_partial_coverage_below_threshold_is_partially_supported(self):
        # has_evidence True (a real distance was produced), but coverage
        # below the 0.5 default threshold.
        beam = _beam(distance_m=2.0, status="CLEAR", valid_count=3, total_pixels=10)
        clearance = _clearance_for(beam, min_coverage_fraction=0.5)
        assert clearance.has_evidence is True
        assert clearance.nearest_distance_m == pytest.approx(2.0)
        assert clearance.coverage_fraction == pytest.approx(0.3)
        assert clearance.support_state == ClearanceSupportState.PARTIALLY_SUPPORTED

    def test_coverage_at_exactly_the_threshold_is_supported(self):
        beam = _beam(distance_m=2.0, status="CLEAR", valid_count=5, total_pixels=10)
        clearance = _clearance_for(beam, min_coverage_fraction=0.5)
        assert clearance.coverage_fraction == pytest.approx(0.5)
        assert clearance.support_state == ClearanceSupportState.SUPPORTED

    def test_zero_total_pixels_does_not_divide_by_zero(self):
        beam = _beam(distance_m=0.0, status="NO_DATA", valid_count=0, total_pixels=0)
        clearance = _clearance_for(beam)
        assert clearance.coverage_fraction == pytest.approx(0.0)
        assert clearance.support_state == ClearanceSupportState.NO_EVIDENCE


# ===================================================================
# 8. Distance measurement remains unchanged from pre-D7 behavior
# ===================================================================
class TestDistanceUnchanged:
    def test_uniform_depth_beam_reports_the_exact_depth(self):
        assessor = ThreatAssessor(n_beams=4, min_valid=1, debounce_frames=1)
        result = assessor.assess(_depth_map(2.5, w=40))
        assert all(b["distance_m"] == pytest.approx(2.5) for b in result["beams"])

    def test_clearance_evidence_distance_matches_beam_distance_exactly(self):
        beam = _beam(distance_m=1.234, status="CLEAR", valid_count=10, total_pixels=10)
        clearance = _clearance_for(beam)
        assert clearance.nearest_distance_m == beam.distance_m


# ===================================================================
# 9. Compatibility ThreatAssessor outputs remain unchanged where
# previously defined
# ===================================================================
class TestThreatAssessorCompatibility:
    def test_existing_beam_keys_still_present_and_correct(self):
        assessor = ThreatAssessor(n_beams=4, min_valid=1, debounce_frames=1)
        result = assessor.assess(_depth_map(3.0, w=40))
        for beam in result["beams"]:
            assert set(("index", "x1", "x2", "distance_m", "status")) <= set(beam.keys())
        assert result["safest_beam"] is not None

    def test_new_keys_are_additive_not_replacing_anything(self):
        assessor = ThreatAssessor(n_beams=4, min_valid=1, debounce_frames=1)
        result = assessor.assess(_depth_map(3.0, w=40))
        for beam in result["beams"]:
            assert "valid_count" in beam
            assert "total_pixels" in beam
            assert beam["valid_count"] == beam["total_pixels"]  # fully valid uniform depth map

    def test_zero_depth_map_still_reports_no_data_and_zero_valid_count(self):
        assessor = ThreatAssessor(n_beams=4, min_valid=1, debounce_frames=1)
        result = assessor.assess(_zeros(w=40))
        assert all(b["status"] == ThreatAssessor.NO_DATA for b in result["beams"])
        assert all(b["valid_count"] == 0 for b in result["beams"])

    def test_beam_reading_direct_construction_without_new_fields_still_works(self):
        # Backward compatibility: a caller constructing a BeamReading
        # without valid_count/total_pixels (as any pre-D7 code would)
        # must not break.
        beam = BeamReading(index=0, x1=0, x2=10, distance_m=1.0, status="CLEAR")
        assert beam.valid_count == 0
        assert beam.total_pixels == 0


# ===================================================================
# 10. GeometryFrame exposes refined ClearanceEvidence
# ===================================================================
def _random_pair(seed=42):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    return left, right


class TestGeometryFrameExposesRefinedClearanceEvidence:
    def test_clearance_evidence_carries_all_new_fields(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry_frame=True), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.geometry_frame is not None
        clearance_list = result.geometry_frame.clearance_evidence
        assert clearance_list is not None
        assert len(clearance_list) > 0
        for clearance in clearance_list:
            assert isinstance(clearance, ClearanceEvidence)
            assert clearance.frame_id == FrameId.CAMERA_OPTICAL_LEFT
            assert clearance.support_state in (
                ClearanceSupportState.SUPPORTED,
                ClearanceSupportState.PARTIALLY_SUPPORTED,
                ClearanceSupportState.NO_EVIDENCE,
            )
            assert clearance.bearing_min_rad <= clearance.bearing_center_rad <= clearance.bearing_max_rad
            assert 0.0 <= clearance.coverage_fraction <= 1.0

    def test_beam_reading_coverage_available_even_without_geometry_frame(self):
        # ThreatAssessor always runs (Level 0-2, unconditional) — coverage
        # is populated on the legacy compatibility path too, not just when
        # GeometryFrame is enabled.
        pipeline = DepthPerceptionPipeline(PipelineConfig(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert all(b.total_pixels > 0 for b in result.obstacles.beams)


# ===================================================================
# 11. No behavioral semantics leak into the contract
# ===================================================================
class TestNoBehavioralLeakage:
    def test_clearance_evidence_field_names_carry_no_behavioral_concept(self):
        fields = {f.name for f in dataclasses.fields(ClearanceEvidence)}
        forbidden = {
            "safe", "unsafe", "passable", "traversable", "turn_direction",
            "vehicle_fit", "clearance_threshold", "navigation_command",
        }
        assert not (fields & forbidden)

    def test_support_state_values_carry_no_behavioral_label(self):
        forbidden_terms = ("SAFE", "PASSABLE", "TRAVERSABLE", "TURN", "FIT")
        for value in (ClearanceSupportState.SUPPORTED, ClearanceSupportState.PARTIALLY_SUPPORTED,
                      ClearanceSupportState.NO_EVIDENCE):
            for term in forbidden_terms:
                assert term not in value

    def test_provider_module_does_not_import_behavioral_types(self):
        import_lines = [
            line for line in inspect.getsource(provider_module).splitlines()
            if line.startswith(("import ", "from "))
        ]
        source = "\n".join(import_lines)
        for forbidden in ("RegionClass", "NavigationDecision", "ThreatAssessor"):
            assert forbidden not in source, f"{forbidden} must not be imported by geometry/provider.py"
            assert not hasattr(provider_module, forbidden)


# ===================================================================
# 12. Feature-disabled behavior remains unchanged
# ===================================================================
class TestFlagDisabledPreservesPreviousBehavior:
    def test_geometry_frame_disabled_means_no_clearance_evidence_object_at_all(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert result.geometry_frame is None

    def test_legacy_obstacle_assessment_shape_unaffected(self):
        left, right = _random_pair()
        result_off = DepthPerceptionPipeline(PipelineConfig(enable_geometry_frame=False), _CALIBRATION).process(left, right)
        result_on = DepthPerceptionPipeline(PipelineConfig(enable_geometry_frame=True), _CALIBRATION).process(left, right)

        for beam_off, beam_on in zip(result_off.obstacles.beams, result_on.obstacles.beams):
            assert beam_off.distance_m == beam_on.distance_m
            assert beam_off.status == beam_on.status
            assert beam_off.valid_count == beam_on.valid_count
            assert beam_off.total_pixels == beam_on.total_pixels
