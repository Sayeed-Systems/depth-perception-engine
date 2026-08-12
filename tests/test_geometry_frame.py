"""
Phase D2/D3 tests — GeometryFrame, the final, authoritative DPE V1
provider contract (see docs/DPE_V1_PROVIDER_CONTRACT.md).

D2 section covers the 10 scenarios the D2 task named: public importability,
the enable_geometry_frame opt-in gate (disabled -> None + unchanged
behavior, enabled -> produced), zero recomputation (GeometryFrame reuses
the SAME already-computed evidence objects, never rebuilds them),
timestamp/frame_id correctness (including the nested body-frame
override), geometric fields matching existing source evidence, Level 4
result objects being exposed when available, absent optional temporal
evidence being handled cleanly, and temporal implementation internals
(TemporalHistory/TemporalRecord/TemporalAdmissionStatus/
TemporalPersistenceTracker/compute_* functions) staying non-public even
though their RESULT types were promoted.

D3 section (TestD3*) covers the D3 task's own 9 scenarios: PointCloud/
ObstacleCloud/FreeSpaceRays/GeometryMetrics being proper Tier 1 contracts;
RegionEvidence/ClearanceEvidence being publicly importable; GeometryFrame
exposing both; their values matching existing RegionAnalyzer/ThreatAssessor
source computation exactly (extraction, not recomputation — proven via
value equality, since RegionEvidence/ClearanceEvidence are distinct
objects from RegionStats/BeamReading, unlike D2's object-identity proof);
no duplicated computation path; no behavioral/navigation-decision leakage
(RegionClass/NavigationDecision/BeamReading.status-style labels absent
from both the field set and the module's own imports); legacy
TraversabilityResult/ObstacleAssessment/NavigationDecision/RegionClass
output remaining byte-identical; and the enable_geometry_frame-disabled
path remaining unaffected. Point 9 ("full existing suite has zero
regressions") has no dedicated test here — it's proven by the full
`pytest tests/ -q` run this file is part of.

No new algorithm is exercised here — every assertion reads already-public
fields produced by the real, unmodified pipeline, mirroring
tests/test_level4_integration_e8.py's own "full chain" configuration
pattern for the scenarios that need real multi-frame temporal evidence.
"""

import dataclasses
import inspect

import numpy as np
import pytest

import depth_perception_engine as dpe
import depth_perception_engine.geometry.provider as provider_module
from depth_perception_engine import (
    ClearanceEvidence,
    DepthPerceptionPipeline,
    FreeSpaceRays,
    GeometryFrame,
    GeometryMetrics,
    ObstacleCloud,
    PipelineConfig,
    PointCloud,
    RegionEvidence,
    load_stereo_calibration,
)
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.provider import GeometryFrame as GeometryFrameFromSubpackage
from depth_perception_engine.temporal import MotionHint

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size


def _transform() -> RigidTransform:
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _full_config(**overrides):
    """Every Level 3 geometry flag, every Level 4 flag, AND
    enable_geometry_frame — mirrors test_level4_integration_e8.py's own
    _full_config(), plus the D2 flag this file exists to validate."""
    defaults = dict(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        enable_geometry_frame=True,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=50,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _pipeline(**config_overrides):
    return DepthPerceptionPipeline(_full_config(**config_overrides), _CALIBRATION, body_T_camera_left=_transform())


def _random_pair(seed=42):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    return left, right


def _hint(ts, omega):
    return MotionHint(timestamp=ts, angular_velocity_rad_s=np.array(omega, dtype=np.float64), frame_id=FrameId.BODY)


def _run_sequence(pipeline, n=4, seed=42):
    """Feed n frames of the same scene (real geometry, real temporal
    agreement) with real timestamps and small motion hints, returning the
    LAST frame's result — by frame 2+ every Level 4 field this file needs
    (temporal_consistency/stabilization/motion_aware_reliability, and by
    frame 3+ temporal_persistence PERSISTENT cells) is populated."""
    left, right = _random_pair(seed)
    result = None
    for i in range(n):
        ts = float(i)
        hints = [_hint(ts - 0.5, [0.0, 0.001, 0.0])] if i > 0 else None
        result = pipeline.process(
            left, right, left_timestamp=ts, right_timestamp=ts, motion_hints=hints,
        )
    return result


# ===================================================================
# 1. Publicly importable
# ===================================================================
class TestPubliclyImportable:
    def test_importable_from_package_root(self):
        from depth_perception_engine import GeometryFrame as _GF

        assert _GF is not None

    def test_importable_from_geometry_subpackage(self):
        from depth_perception_engine.geometry import GeometryFrame as _GF

        assert _GF is not None

    def test_root_and_subpackage_import_are_the_same_object(self):
        assert dpe.GeometryFrame is GeometryFrameFromSubpackage

    def test_is_a_dataclass_with_the_approved_field_list(self):
        fields = {f.name for f in dataclasses.fields(GeometryFrame)}
        assert fields == {
            "timestamp", "frame_id",
            "disparity_map", "depth_map", "valid_disparity_mask", "valid_depth_mask",
            "geometry", "geometry_body", "obstacle_cloud", "free_space_rays", "geometry_metrics",
            "temporal_consistency", "temporal_stabilization", "rotation_compensation_status",
            "motion_aware_reliability", "temporal_persistence",
            # Phase D3
            "region_evidence", "clearance_evidence",
            # Phase D4
            "surface_evidence",
            # Phase D5
            "boundary_evidence",
            # Phase D6
            "opening_evidence",
            # Phase D8
            "quality",
        }


# ===================================================================
# 2. Flag disabled -> None, unchanged behavior
# ===================================================================
class TestFlagDisabled:
    def test_geometry_frame_is_none_by_default(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert result.geometry_frame is None

    def test_disabled_leaves_every_other_field_unaffected(self):
        left, right = _random_pair()

        result_off = _pipeline(enable_geometry_frame=False).process(left, right, left_timestamp=0.0)
        result_on = _pipeline(enable_geometry_frame=True).process(left, right, left_timestamp=0.0)

        assert result_off.geometry_frame is None
        assert result_on.geometry_frame is not None

        assert np.array_equal(result_off.disparity_map, result_on.disparity_map)
        assert np.array_equal(result_off.depth_map, result_on.depth_map)
        assert result_off.confidence == result_on.confidence
        assert result_off.timestamp == result_on.timestamp
        assert result_off.traversability_mask.decision == result_on.traversability_mask.decision
        assert np.array_equal(result_off.geometry.points, result_on.geometry.points, equal_nan=True)


# ===================================================================
# 3. Flag enabled -> produced
# ===================================================================
class TestFlagEnabled:
    def test_geometry_frame_is_produced_when_enabled(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0)
        assert isinstance(result.geometry_frame, GeometryFrame)


# ===================================================================
# 4. Zero recomputation — same objects, not copies/rebuilds
# ===================================================================
class TestZeroRecomputation:
    def test_level3_fields_are_the_same_objects_as_result(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0)
        gf = result.geometry_frame

        assert gf.disparity_map is result.disparity_map
        assert gf.depth_map is result.depth_map
        assert gf.valid_disparity_mask is result.valid_disparity_mask
        assert gf.valid_depth_mask is result.valid_depth_mask
        assert gf.geometry is result.geometry
        assert gf.geometry_body is result.geometry_body
        assert gf.obstacle_cloud is result.obstacle_cloud
        assert gf.free_space_rays is result.free_space_rays
        assert gf.geometry_metrics is result.geometry_metrics

    def test_level4_fields_are_the_same_objects_as_result_and_not_none(self):
        pipeline = _pipeline()
        result = _run_sequence(pipeline, n=4)
        gf = result.geometry_frame

        assert gf.temporal_consistency is not None
        assert gf.temporal_consistency is result.temporal_consistency
        assert gf.temporal_stabilization is not None
        assert gf.temporal_stabilization is result.temporal_stabilization
        assert gf.motion_aware_reliability is not None
        assert gf.motion_aware_reliability is result.motion_aware_reliability
        assert gf.temporal_persistence is not None
        assert gf.temporal_persistence is result.temporal_persistence
        assert gf.rotation_compensation_status == result.rotation_compensation_status


# ===================================================================
# 5. timestamp / frame_id correctness
# ===================================================================
class TestTimestampAndFrameId:
    def test_timestamp_matches_result(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=3.5)
        assert result.geometry_frame.timestamp == 3.5
        assert result.geometry_frame.timestamp == result.timestamp

    def test_top_level_frame_id_is_camera_optical_left(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0)
        assert result.geometry_frame.frame_id == FrameId.CAMERA_OPTICAL_LEFT

    def test_geometry_body_declares_its_own_differing_frame(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0)
        gf = result.geometry_frame
        assert gf.geometry.frame_id == FrameId.CAMERA_OPTICAL_LEFT == gf.frame_id
        assert gf.geometry_body.frame_id == FrameId.BODY
        assert gf.obstacle_cloud.frame_id == FrameId.BODY
        assert gf.free_space_rays.frame_id == FrameId.BODY


# ===================================================================
# 6. Geometric fields match existing source evidence (value-level)
# ===================================================================
class TestGeometricFieldsMatchSource:
    def test_values_equal_the_result_they_were_read_from(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0)
        gf = result.geometry_frame

        assert np.array_equal(gf.disparity_map, result.disparity_map)
        assert np.array_equal(gf.depth_map, result.depth_map)
        assert np.array_equal(gf.geometry.points, result.geometry.points, equal_nan=True)
        assert np.array_equal(gf.geometry_body.points, result.geometry_body.points, equal_nan=True)


# ===================================================================
# 7. Level 4 result objects exposed correctly when available
# ===================================================================
class TestLevel4ResultsExposedWhenAvailable:
    def test_full_chain_temporal_evidence_present_on_geometry_frame(self):
        pipeline = _pipeline()
        result = _run_sequence(pipeline, n=4)
        gf = result.geometry_frame

        from depth_perception_engine import (
            MotionAwareReliability,
            RotationCompensationStatus,
            TemporalConsistency,
            TemporalPersistence,
            TemporalStabilization,
        )

        assert isinstance(gf.temporal_consistency, TemporalConsistency)
        assert isinstance(gf.temporal_stabilization, TemporalStabilization)
        assert isinstance(gf.motion_aware_reliability, MotionAwareReliability)
        assert isinstance(gf.temporal_persistence, TemporalPersistence)
        assert gf.rotation_compensation_status in (
            RotationCompensationStatus.APPLIED, RotationCompensationStatus.NOT_APPLIED,
        )


# ===================================================================
# 8. Absent optional temporal evidence handled cleanly
# ===================================================================
class TestAbsentOptionalTemporalEvidenceHandledCleanly:
    def test_temporal_disabled_entirely_leaves_geometry_frame_temporal_fields_none(self):
        config = PipelineConfig(
            enable_geometry=True, enable_geometry_frame=True, enable_temporal=False,
        )
        pipeline = DepthPerceptionPipeline(config, _CALIBRATION, body_T_camera_left=_transform())
        left, right = _random_pair()
        result = pipeline.process(left, right, left_timestamp=0.0)

        assert result.geometry_frame is not None
        gf = result.geometry_frame
        assert gf.temporal_consistency is None
        assert gf.temporal_stabilization is None
        assert gf.rotation_compensation_status is None
        assert gf.motion_aware_reliability is None
        assert gf.temporal_persistence is None

    def test_first_frame_of_full_chain_has_no_comparable_prior_and_does_not_raise(self):
        # No comparable prior frame exists yet (this is the very first
        # admitted record) — represented by the well-defined
        # INSUFFICIENT_EVIDENCE state, never a crash and never a
        # fabricated comparison.
        pipeline = _pipeline()
        left, right = _random_pair()
        result = pipeline.process(left, right, left_timestamp=0.0)

        assert result.geometry_frame is not None
        assert result.geometry_frame.temporal_consistency.state == dpe.TemporalConsistencyState.INSUFFICIENT_EVIDENCE
        assert (
            result.geometry_frame.temporal_stabilization.state
            == dpe.TemporalStabilizationState.INSUFFICIENT_EVIDENCE
        )


# ===================================================================
# 9. Temporal implementation internals stay non-public
# ===================================================================
class TestTemporalInternalsNotPromoted:
    # MotionHint was promoted to Tier 1 at Phase D13 (an INPUT contract
    # required to construct StereoObservation.motion_hint/.motion_hints
    # and DepthPerceptionPipeline.process()'s own motion_hint/
    # motion_hints parameters without an internal
    # depth_perception_engine.temporal import) — it is no longer one of
    # the un-promoted internals this test guards. See
    # docs/DPE_V1_PROVIDER_CONTRACT.md's D13 record and
    # tests/test_public_api.py's TIER_1_SYMBOLS.
    NOT_PROMOTED = [
        "TemporalHistory", "TemporalRecord", "TemporalAdmissionStatus",
        "TemporalPersistenceTracker",
        "compute_temporal_consistency", "compute_temporal_stabilization",
        "compute_rotation_compensation", "compute_motion_aware_reliability",
        "compensate_prior_geometry_with_payload",
    ]

    def test_none_of_these_are_root_attributes(self):
        leaked = [name for name in self.NOT_PROMOTED if hasattr(dpe, name)]
        assert not leaked, f"Temporal internals leaked onto the package root: {leaked}"

    def test_none_of_these_are_in_all(self):
        leaked = [name for name in self.NOT_PROMOTED if name in dpe.__all__]
        assert not leaked, f"Temporal internals leaked into __all__: {leaked}"


# ===================================================================
# D3.1 — PointCloud/ObstacleCloud/FreeSpaceRays/GeometryMetrics are
# proper Tier 1 contracts
# ===================================================================
class TestD3TierOnePromotions:
    TYPES = [PointCloud, ObstacleCloud, FreeSpaceRays, GeometryMetrics]
    NAMES = ["PointCloud", "ObstacleCloud", "FreeSpaceRays", "GeometryMetrics"]

    def test_importable_from_package_root(self):
        for name in self.NAMES:
            assert hasattr(dpe, name)

    def test_root_and_subpackage_import_are_the_same_object(self):
        import depth_perception_engine.geometry as dpe_geometry

        for name in self.NAMES:
            assert getattr(dpe, name) is getattr(dpe_geometry, name)

    def test_in_all(self):
        for name in self.NAMES:
            assert name in dpe.__all__

    def test_builders_and_algorithms_remain_tier_3(self):
        # The architect's D3 rule is explicit: promoting a result type
        # does not mean promoting its producer.
        for name in (
            "PointCloudBuilder", "transform_point_cloud", "build_obstacle_cloud",
            "build_free_space_rays", "build_geometry_metrics", "GeometryQuality",
            "classify_geometry_quality",
        ):
            assert not hasattr(dpe, name), f"{name} should remain Tier 3"
            assert name not in dpe.__all__


# ===================================================================
# D3.2 — RegionEvidence / ClearanceEvidence are publicly importable
# ===================================================================
class TestD3EvidenceTypesPubliclyImportable:
    def test_importable_from_package_root(self):
        assert RegionEvidence is not None
        assert ClearanceEvidence is not None

    def test_importable_from_geometry_subpackage(self):
        from depth_perception_engine.geometry import ClearanceEvidence as _CE
        from depth_perception_engine.geometry import RegionEvidence as _RE

        assert _RE is RegionEvidence
        assert _CE is ClearanceEvidence

    def test_in_all(self):
        assert "RegionEvidence" in dpe.__all__
        assert "ClearanceEvidence" in dpe.__all__

    def test_region_evidence_field_shape(self):
        fields = {f.name for f in dataclasses.fields(RegionEvidence)}
        assert fields == {
            "frame_id", "name", "row", "col", "x1", "y1", "x2", "y2",
            "valid_count", "total_pixels", "valid_fraction",
            "depth_avg_m", "depth_median_m", "depth_min_m", "depth_max_m",
            "texture_score", "entropy", "gradient_magnitude", "texture_class", "confidence",
        }

    def test_clearance_evidence_field_shape(self):
        fields = {f.name for f in dataclasses.fields(ClearanceEvidence)}
        assert fields == {
            "frame_id", "index", "x1", "x2", "nearest_distance_m", "has_evidence",
            # Phase D7
            "valid_count", "total_pixels", "coverage_fraction", "support_state",
            "bearing_center_rad", "bearing_min_rad", "bearing_max_rad",
        }


# ===================================================================
# D3.3 — GeometryFrame exposes both
# ===================================================================
class TestD3GeometryFrameExposesEvidence:
    def test_region_evidence_present_and_keyed_like_traversability_regions(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0)
        gf = result.geometry_frame

        assert gf.region_evidence is not None
        assert isinstance(gf.region_evidence, dict)
        assert set(gf.region_evidence.keys()) == set(result.traversability_mask.regions.keys())
        for evidence in gf.region_evidence.values():
            assert isinstance(evidence, RegionEvidence)

    def test_clearance_evidence_present_and_shaped_like_obstacle_beams(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0)
        gf = result.geometry_frame

        assert gf.clearance_evidence is not None
        assert isinstance(gf.clearance_evidence, list)
        assert len(gf.clearance_evidence) == len(result.obstacles.beams)
        for evidence in gf.clearance_evidence:
            assert isinstance(evidence, ClearanceEvidence)


# ===================================================================
# D3.4 — evidence matches existing source computation (extraction, not
# recomputation — proven via value equality against the source RegionStats/
# BeamReading, since RegionEvidence/ClearanceEvidence are distinct objects)
# ===================================================================
class TestD3EvidenceMatchesSourceComputation:
    def test_region_evidence_values_match_region_stats(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0)
        gf = result.geometry_frame

        for name, stats in result.traversability_mask.regions.items():
            evidence = gf.region_evidence[name]
            assert evidence.name == stats.name
            assert evidence.row == stats.row
            assert evidence.col == stats.col
            assert (evidence.x1, evidence.y1, evidence.x2, evidence.y2) == (stats.x1, stats.y1, stats.x2, stats.y2)
            assert evidence.valid_count == stats.valid_count
            assert evidence.total_pixels == stats.total_pixels
            assert evidence.valid_fraction == pytest.approx(stats.valid_pct / 100.0)
            assert evidence.depth_avg_m == stats.depth_avg_m
            assert evidence.depth_median_m == stats.depth_median_m
            assert evidence.depth_min_m == stats.depth_min_m
            assert evidence.depth_max_m == stats.depth_max_m
            assert evidence.texture_score == stats.texture_score
            assert evidence.entropy == stats.entropy
            assert evidence.gradient_magnitude == stats.gradient_magnitude
            assert evidence.texture_class == stats.texture_class
            assert evidence.confidence == stats.confidence

    def test_clearance_evidence_values_match_beam_readings(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0)
        gf = result.geometry_frame

        for beam, evidence in zip(result.obstacles.beams, gf.clearance_evidence):
            assert evidence.index == beam.index
            assert evidence.x1 == beam.x1
            assert evidence.x2 == beam.x2
            if beam.distance_m > 0.0:
                assert evidence.has_evidence is True
                assert evidence.nearest_distance_m == beam.distance_m
            else:
                assert evidence.has_evidence is False
                assert evidence.nearest_distance_m is None


# ===================================================================
# D3.5 — no duplicated computation path: enabling GeometryFrame must not
# call RegionAnalyzer/ThreatAssessor's own algorithms an extra time
# ===================================================================
class TestD3NoDuplicatedComputationPath:
    def test_region_analyzer_and_threat_assessor_called_exactly_once_per_frame(self, mocker):
        from depth_perception_engine.traversability.region_analyzer import RegionAnalyzer
        from depth_perception_engine.obstacles.threat_assessment import ThreatAssessor

        analyze_spy = mocker.spy(RegionAnalyzer, "analyze")
        assess_spy = mocker.spy(ThreatAssessor, "assess")

        left, right = _random_pair()
        pipeline = _pipeline()
        pipeline.process(left, right, left_timestamp=0.0)

        # One call to assess() per frame; RegionAnalyzer.analyze() is
        # called once per grid cell (9 for the default 3x3 grid) — the
        # same count either way, proving GeometryFrame construction adds
        # no second traversability/obstacle pass.
        assert assess_spy.call_count == 1
        assert analyze_spy.call_count == 9


# ===================================================================
# D3.6 — no behavioral/navigation-decision leakage
# ===================================================================
class TestD3NoBehavioralLeakage:
    def test_region_evidence_has_no_classification_field(self):
        fields = {f.name for f in dataclasses.fields(RegionEvidence)}
        assert "classification" not in fields

    def test_clearance_evidence_has_no_status_field(self):
        fields = {f.name for f in dataclasses.fields(ClearanceEvidence)}
        assert "status" not in fields

    def test_provider_module_does_not_import_behavioral_types(self):
        # Checks the module's actual bound names (what it imported and can
        # use), not its docstring prose, which legitimately names these
        # types to explain what was deliberately excluded.
        import_lines = [
            line for line in inspect.getsource(provider_module).splitlines()
            if line.startswith(("import ", "from "))
        ]
        source = "\n".join(import_lines)
        for forbidden in ("RegionClass", "NavigationDecision", "ThreatAssessor"):
            assert forbidden not in source, f"{forbidden} must not be imported by geometry/provider.py"
            assert not hasattr(provider_module, forbidden)

    def test_no_behavioral_enum_values_appear_as_geometry_frame_field_defaults_or_names(self):
        behavioral_tokens = {
            "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "SLOW_DOWN",
            "STOP", "HOVER", "ROTATE_AND_SCAN",
            "CLEAR", "CAUTION", "BLOCKED", "PROBABLE_WALL",
            "LOW_CONFIDENCE", "LOW_TEXTURE_UNKNOWN",
        }
        all_field_names = {f.name for f in dataclasses.fields(RegionEvidence)} | {
            f.name for f in dataclasses.fields(ClearanceEvidence)
        }
        assert not (behavioral_tokens & all_field_names)


# ===================================================================
# D3.7 — legacy outputs remain unchanged
# ===================================================================
class TestD3LegacyOutputsUnchanged:
    def test_traversability_and_obstacles_identical_regardless_of_geometry_frame_flag(self):
        left, right = _random_pair()

        result_off = _pipeline(enable_geometry_frame=False).process(left, right, left_timestamp=0.0)
        result_on = _pipeline(enable_geometry_frame=True).process(left, right, left_timestamp=0.0)

        assert result_off.traversability_mask.decision == result_on.traversability_mask.decision
        for name in result_off.traversability_mask.regions:
            off_region = result_off.traversability_mask.regions[name]
            on_region = result_on.traversability_mask.regions[name]
            assert off_region.classification == on_region.classification
            assert off_region.confidence == on_region.confidence

        for off_beam, on_beam in zip(result_off.obstacles.beams, result_on.obstacles.beams):
            assert off_beam.status == on_beam.status
            assert off_beam.distance_m == on_beam.distance_m


# ===================================================================
# D3.8 — enable_geometry_frame disabled path remains unaffected
# ===================================================================
class TestD3FlagDisabledUnaffected:
    def test_geometry_frame_none_means_no_evidence_attributes_to_read(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert result.geometry_frame is None
        # traversability/obstacles are still computed as always — D3 added
        # no new gate around them.
        assert result.traversability_mask is not None
        assert result.obstacles is not None
