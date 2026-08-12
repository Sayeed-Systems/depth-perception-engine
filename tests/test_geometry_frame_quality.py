"""
Phase D8 tests — geometric quality/uncertainty contract
(GeometryFrameQuality, GeometryFrameQualityState, RegionEvidence.frame_id).

Covers the 13 scenarios the D8 task named. Unit-level tests
(TestFullySupportedGeometry*, TestLowCoverage*, TestMissingGeometry*,
TestTemporalDegradation*, TestMotionReliabilityDegradation*,
TestAbsentVsBadEvidence, TestDeterminism, TestRegionEvidenceFrameId,
TestNoBehavioralLeakage) call build_geometry_frame_quality() directly
with controlled, hand-built GeometryMetrics/TemporalConsistency/
MotionAwareReliability/TemporalPersistence objects — expected quality
states known in advance, not merely "a value exists". Integration-level
tests (TestGeometryFrameExposesQuality, TestLegacyConfidenceUnchanged,
TestFlagDisabled) run the real, unmodified pipeline. Point 13 ("full
existing suite has zero regressions") has no dedicated test here — it's
proven by the full `pytest tests/ -q` run this file is part of.
"""

import dataclasses
import inspect

import numpy as np
import pytest

import depth_perception_engine.geometry.provider as provider_module
from depth_perception_engine import (
    DepthPerceptionPipeline,
    GeometryFrameQuality,
    GeometryFrameQualityState,
    PipelineConfig,
    RegionEvidence,
    load_stereo_calibration,
)
from depth_perception_engine.frames import FrameId
from depth_perception_engine.fusion.result_builder import build_geometry_frame_quality, build_region_evidence
from depth_perception_engine.geometry.types import GeometryMetrics
from depth_perception_engine.models.result import TraversabilityResult
from depth_perception_engine.temporal.consistency import TemporalConsistencyState
from depth_perception_engine.temporal.persistence import TemporalPersistenceState
from depth_perception_engine.temporal.reliability import MotionAwareReliabilityState
from depth_perception_engine.temporal.types import MotionAwareReliability, TemporalConsistency, TemporalPersistence
from depth_perception_engine.traversability.types import NavigationDecision, RegionClass, RegionStats, TextureClass

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size
_HEALTHY_MIN = 0.5
_DEGRADED_MIN = 0.05


# ===================================================================
# Helpers
# ===================================================================
def _geometry_metrics(valid_fraction):
    return GeometryMetrics(
        min_obstacle_distance_m=None, mean_free_space_m=None, point_count=100, valid_fraction=valid_fraction,
    )


def _temporal_consistency(state):
    return TemporalConsistency(
        state=state, comparable_count=10, agreeing_count=8, disagreement_count=2, agreement_fraction=0.8,
    )


def _motion_reliability(state):
    return MotionAwareReliability(
        state=state, motion_sample_count=3, motion_coverage_fraction=0.9,
        rotation_compensation_status="APPLIED", angular_motion_magnitude_rad=0.01,
        temporal_consistency_state=None, temporal_stabilization_state=None,
    )


def _temporal_persistence(state):
    return TemporalPersistence(
        state=state, state_grid=None, support_count_grid=None, age_s_grid=None,
        new_count=0, persistent_count=0, disappearing_count=0, expired_count=0,
        eligible_count=0, persistent_fraction=None,
    )


def _quality(**overrides):
    kwargs = dict(
        geometry_metrics=None, temporal_consistency=None,
        motion_aware_reliability=None, temporal_persistence=None,
        geometry_healthy_min_valid_fraction=_HEALTHY_MIN,
        geometry_degraded_min_valid_fraction=_DEGRADED_MIN,
    )
    kwargs.update(overrides)
    return build_geometry_frame_quality(**kwargs)


# ===================================================================
# 1. Fully supported controlled geometry produces the expected
# good/valid quality state
# ===================================================================
class TestFullySupportedGeometryIsValid:
    def test_all_four_dimensions_valid_yields_overall_valid(self):
        quality = _quality(
            geometry_metrics=_geometry_metrics(1.0),
            temporal_consistency=_temporal_consistency(TemporalConsistencyState.CONSISTENT),
            motion_aware_reliability=_motion_reliability(MotionAwareReliabilityState.RELIABLE),
            temporal_persistence=_temporal_persistence(TemporalPersistenceState.CLASSIFIED),
        )
        assert quality.geometry_validity_state == GeometryFrameQualityState.VALID
        assert quality.temporal_consistency_state == GeometryFrameQualityState.VALID
        assert quality.motion_reliability_state == GeometryFrameQualityState.VALID
        assert quality.persistence_state == GeometryFrameQualityState.VALID
        assert quality.overall_state == GeometryFrameQualityState.VALID
        assert quality.degradation_reasons == []


# ===================================================================
# 2. Low geometric coverage is represented as degraded/insufficient
# according to documented rules
# ===================================================================
class TestLowCoverageIsDegradedOrInsufficient:
    def test_valid_fraction_between_thresholds_is_degraded(self):
        # 0.05 <= 0.2 < 0.5
        quality = _quality(geometry_metrics=_geometry_metrics(0.2))
        assert quality.geometry_validity_state == GeometryFrameQualityState.DEGRADED
        assert quality.overall_state == GeometryFrameQualityState.DEGRADED
        assert "GEOMETRY_VALIDITY:DEGRADED" in quality.degradation_reasons

    def test_valid_fraction_below_degraded_threshold_is_insufficient(self):
        quality = _quality(geometry_metrics=_geometry_metrics(0.01))
        assert quality.geometry_validity_state == GeometryFrameQualityState.INSUFFICIENT
        assert quality.overall_state == GeometryFrameQualityState.INSUFFICIENT

    def test_valid_fraction_at_healthy_threshold_is_valid(self):
        quality = _quality(geometry_metrics=_geometry_metrics(_HEALTHY_MIN))
        assert quality.geometry_validity_state == GeometryFrameQualityState.VALID


# ===================================================================
# 3. Missing geometry is not reported as trustworthy
# ===================================================================
class TestMissingGeometryIsNotTrustworthy:
    def test_geometry_metrics_none_is_absent_not_valid(self):
        quality = _quality(geometry_metrics=None)
        assert quality.geometry_validity_state is None
        assert quality.geometry_validity_state != GeometryFrameQualityState.VALID

    def test_everything_absent_yields_insufficient_overall_never_valid(self):
        quality = _quality()  # all four None
        assert quality.overall_state == GeometryFrameQualityState.INSUFFICIENT
        assert quality.overall_state != GeometryFrameQualityState.VALID


# ===================================================================
# 4. Temporal degradation is represented correctly when temporal
# evidence exists
# ===================================================================
class TestTemporalDegradation:
    def test_contradictory_consistency_is_degraded(self):
        quality = _quality(temporal_consistency=_temporal_consistency(TemporalConsistencyState.CONTRADICTORY))
        assert quality.temporal_consistency_state == GeometryFrameQualityState.DEGRADED
        assert quality.overall_state == GeometryFrameQualityState.DEGRADED
        assert "TEMPORAL_CONSISTENCY:DEGRADED" in quality.degradation_reasons

    def test_consistent_is_valid(self):
        quality = _quality(temporal_consistency=_temporal_consistency(TemporalConsistencyState.CONSISTENT))
        assert quality.temporal_consistency_state == GeometryFrameQualityState.VALID


# ===================================================================
# 5. Motion reliability degradation is represented correctly
# ===================================================================
class TestMotionReliabilityDegradation:
    def test_unreliable_is_degraded(self):
        quality = _quality(motion_aware_reliability=_motion_reliability(MotionAwareReliabilityState.UNRELIABLE))
        assert quality.motion_reliability_state == GeometryFrameQualityState.DEGRADED
        assert quality.overall_state == GeometryFrameQualityState.DEGRADED

    def test_degraded_state_is_degraded(self):
        quality = _quality(motion_aware_reliability=_motion_reliability(MotionAwareReliabilityState.DEGRADED))
        assert quality.motion_reliability_state == GeometryFrameQualityState.DEGRADED

    def test_reliable_is_valid(self):
        quality = _quality(motion_aware_reliability=_motion_reliability(MotionAwareReliabilityState.RELIABLE))
        assert quality.motion_reliability_state == GeometryFrameQualityState.VALID


# ===================================================================
# 6. Absent optional temporal evidence is distinguished from
# contradictory/bad evidence
# ===================================================================
class TestAbsentVsBadEvidence:
    def test_absent_none_vs_insufficient_vs_degraded_are_three_distinct_values(self):
        absent = _quality(temporal_consistency=None)
        insufficient = _quality(
            temporal_consistency=_temporal_consistency(TemporalConsistencyState.INSUFFICIENT_EVIDENCE)
        )
        degraded = _quality(temporal_consistency=_temporal_consistency(TemporalConsistencyState.CONTRADICTORY))

        assert absent.temporal_consistency_state is None
        assert insufficient.temporal_consistency_state == GeometryFrameQualityState.INSUFFICIENT
        assert degraded.temporal_consistency_state == GeometryFrameQualityState.DEGRADED

        values = {absent.temporal_consistency_state, insufficient.temporal_consistency_state,
                  degraded.temporal_consistency_state}
        assert len(values) == 3  # genuinely three distinct values, not two conflated into one

    def test_not_comparable_is_also_insufficient_not_absent(self):
        quality = _quality(temporal_consistency=_temporal_consistency(TemporalConsistencyState.NOT_COMPARABLE))
        assert quality.temporal_consistency_state == GeometryFrameQualityState.INSUFFICIENT


# ===================================================================
# 7. Quality/degradation reasons are deterministic
# ===================================================================
class TestDeterminism:
    def test_repeated_calls_produce_identical_quality(self):
        kwargs = dict(
            geometry_metrics=_geometry_metrics(0.3),
            temporal_consistency=_temporal_consistency(TemporalConsistencyState.CONTRADICTORY),
            motion_aware_reliability=_motion_reliability(MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE),
            temporal_persistence=_temporal_persistence(TemporalPersistenceState.UNRELIABLE),
            geometry_healthy_min_valid_fraction=_HEALTHY_MIN,
            geometry_degraded_min_valid_fraction=_DEGRADED_MIN,
        )
        first = build_geometry_frame_quality(**kwargs)
        second = build_geometry_frame_quality(**kwargs)
        assert first == second

    def test_degradation_reasons_order_is_fixed(self):
        quality = _quality(
            geometry_metrics=_geometry_metrics(0.01),
            temporal_consistency=_temporal_consistency(TemporalConsistencyState.CONTRADICTORY),
            motion_aware_reliability=_motion_reliability(MotionAwareReliabilityState.UNRELIABLE),
            temporal_persistence=_temporal_persistence(TemporalPersistenceState.UNRELIABLE),
        )
        assert quality.degradation_reasons == [
            "GEOMETRY_VALIDITY:INSUFFICIENT",
            "TEMPORAL_CONSISTENCY:DEGRADED",
            "MOTION_RELIABILITY:DEGRADED",
            "PERSISTENCE:DEGRADED",
        ]


# ===================================================================
# 8. RegionEvidence frame_id is correct
# ===================================================================
class TestRegionEvidenceFrameId:
    def test_build_region_evidence_populates_frame_id(self):
        region_stats = RegionStats(
            name="TL", row=0, col=0, x1=0, y1=0, x2=10, y2=10,
            valid_pct=80.0, invalid_pct=20.0, invalid_ratio=0.2,
            valid_count=80, total_pixels=100,
            depth_avg_m=2.0, depth_median_m=2.0, depth_min_m=1.5, depth_max_m=2.5,
            texture_score=15.0, entropy=4.0, gradient_magnitude=10.0, confidence=0.7,
            texture_class=TextureClass.MEDIUM_TEXTURE, classification=RegionClass.CLEAR,
        )
        traversability = TraversabilityResult(regions={"TL": region_stats}, decision=NavigationDecision.HOVER)
        result = build_region_evidence(traversability, FrameId.CAMERA_OPTICAL_LEFT)
        assert result["TL"].frame_id == FrameId.CAMERA_OPTICAL_LEFT

    def test_frame_id_is_generic_not_hardcoded(self):
        region_stats = RegionStats(
            name="TL", row=0, col=0, x1=0, y1=0, x2=10, y2=10,
            valid_pct=80.0, invalid_pct=20.0, invalid_ratio=0.2,
            valid_count=80, total_pixels=100,
            depth_avg_m=2.0, depth_median_m=2.0, depth_min_m=1.5, depth_max_m=2.5,
            texture_score=15.0, entropy=4.0, gradient_magnitude=10.0, confidence=0.7,
            texture_class=TextureClass.MEDIUM_TEXTURE, classification=RegionClass.CLEAR,
        )
        traversability = TraversabilityResult(regions={"TL": region_stats}, decision=NavigationDecision.HOVER)
        result = build_region_evidence(traversability, FrameId.BODY)
        assert result["TL"].frame_id == FrameId.BODY


# ===================================================================
# 9/10 — pipeline integration: GeometryFrame exposure, legacy confidence
# ===================================================================
def _random_pair(seed=42):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    return left, right


class TestGeometryFrameExposesQuality:
    def test_quality_present_and_well_formed(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry_frame=True), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.geometry_frame is not None
        quality = result.geometry_frame.quality
        assert quality is not None
        assert isinstance(quality, GeometryFrameQuality)
        assert quality.overall_state in (
            GeometryFrameQualityState.VALID, GeometryFrameQualityState.DEGRADED,
            GeometryFrameQualityState.INSUFFICIENT,
        )
        # Nothing enabled beyond enable_geometry_frame -> no temporal
        # capabilities were ever computed -> all temporal dimensions
        # absent (None), never fabricated.
        assert quality.temporal_consistency_state is None
        assert quality.motion_reliability_state is None
        assert quality.persistence_state is None


class TestLegacyConfidenceUnchanged:
    def test_confidence_identical_regardless_of_geometry_frame_flag(self):
        left, right = _random_pair()
        result_off = DepthPerceptionPipeline(PipelineConfig(enable_geometry_frame=False), _CALIBRATION).process(left, right)
        result_on = DepthPerceptionPipeline(PipelineConfig(enable_geometry_frame=True), _CALIBRATION).process(left, right)
        assert result_off.confidence == result_on.confidence

    def test_confidence_is_not_geometry_frame_quality(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry_frame=True), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        # confidence remains a plain float, never reinterpreted as/coupled
        # to the structured quality contract.
        assert isinstance(result.confidence, float)
        assert not hasattr(result, "quality")


# ===================================================================
# 12. Feature-disabled behavior remains backward compatible
# ===================================================================
class TestFlagDisabledPreservesPreviousBehavior:
    def test_geometry_frame_disabled_by_default(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert result.geometry_frame is None


# ===================================================================
# 11. No behavioral/platform semantics leak into quality
# ===================================================================
class TestNoBehavioralLeakage:
    def test_quality_field_names_carry_no_behavioral_concept(self):
        fields = {f.name for f in dataclasses.fields(GeometryFrameQuality)}
        forbidden = {
            "safe", "unsafe", "traversable", "passable", "landing_suitable",
            "vehicle_fit", "navigation_score", "platform",
        }
        assert not (fields & forbidden)

    def test_state_values_carry_no_behavioral_label(self):
        forbidden_terms = ("SAFE", "TRAVERSABLE", "PASSABLE", "LANDING", "VEHICLE")
        for value in (GeometryFrameQualityState.VALID, GeometryFrameQualityState.DEGRADED,
                      GeometryFrameQualityState.INSUFFICIENT):
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

    def test_build_geometry_frame_quality_signature_takes_no_platform_input(self):
        params = set(inspect.signature(build_geometry_frame_quality).parameters)
        assert params == {
            "geometry_metrics", "temporal_consistency", "motion_aware_reliability", "temporal_persistence",
            "geometry_healthy_min_valid_fraction", "geometry_degraded_min_valid_fraction",
        }
