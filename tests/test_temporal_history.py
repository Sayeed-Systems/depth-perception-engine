"""
Level 4, Phase E2 tests — bounded temporal history + timestamp integrity.

Covers scenarios A-P from the E2 task (docs/E2_TEMPORAL_HISTORY_PLAN.md):
unit-level TemporalHistory/TemporalRecord chronology (A-J, N, O) plus
DepthPerceptionPipeline integration (K, L, M, disabled-by-default
zero-regression). Scenario P (the full existing regression suite) is
proven by `pytest tests/ -q` as a whole, not a single test here.

No E3+ algorithm is tested — E2 performs no temporal filtering, fusion,
IMU integration, rotation compensation, or persistence classification, so
none of that is exercised below (see TestScopeDiscipline).
"""

import ast
import dataclasses
import gc
import inspect
import sys

import numpy as np
import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import GeometryQuality
from depth_perception_engine.models.result import DepthPerceptionResult, StereoObservation
from depth_perception_engine.pipeline import DepthPerceptionPipeline
from depth_perception_engine.temporal import (
    MotionHint,
    TemporalAdmissionStatus,
    TemporalHistory,
    TemporalRecord,
)


def _record(ts, confidence=0.8, geometry_quality=None, motion_hint=None):
    return TemporalRecord(
        timestamp=ts, confidence=confidence,
        geometry_quality=geometry_quality, motion_hint=motion_hint,
    )


def _illustrative_transform():
    """Same synthetic, illustrative-only extrinsic as
    tests/test_pipeline_body_frame.py — not a measurement of any real
    airframe. See docs/COORDINATE_FRAMES.md's E4 section."""
    angle = np.deg2rad(15.0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    translation = np.array([0.08, 0.0, 0.05])
    return RigidTransform(
        rotation=rotation, translation=translation,
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _flat_textureless_pair(calibration):
    """Same fixture as tests/test_pipeline_spatial_evidence.py — reliably
    forces zero valid disparity/depth/geometry anywhere in the frame."""
    width, height = calibration.image_size
    flat = np.full((height, width, 3), 128, dtype=np.uint8)
    return flat, flat.copy()


# ===================================================================
# A. Initialization
# ===================================================================
class TestInitialization:
    def test_history_empty_after_construction(self):
        history = TemporalHistory(max_records=5, max_age_s=1.0, gap_limit_s=0.5)
        assert len(history) == 0
        assert history.records == ()

    def test_latest_is_none_when_empty(self):
        history = TemporalHistory(max_records=5, max_age_s=1.0, gap_limit_s=0.5)
        assert history.latest is None

    def test_construction_rejects_invalid_bounds(self):
        with pytest.raises(ValueError):
            TemporalHistory(max_records=0, max_age_s=1.0, gap_limit_s=0.5)
        with pytest.raises(ValueError):
            TemporalHistory(max_records=5, max_age_s=0.0, gap_limit_s=0.5)
        with pytest.raises(ValueError):
            TemporalHistory(max_records=5, max_age_s=1.0, gap_limit_s=0.0)


# ===================================================================
# B. Normal chronological insertion
# ===================================================================
class TestChronologicalInsertion:
    def test_increasing_timestamps_accepted(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        for ts in [0.0, 0.1, 0.2, 0.3]:
            assert history.admit(_record(ts)) == TemporalAdmissionStatus.ACCEPTED
        assert len(history) == 4

    def test_chronological_order_preserved(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        timestamps = [0.0, 0.1, 0.2, 0.3]
        for ts in timestamps:
            history.admit(_record(ts))
        assert [r.timestamp for r in history.records] == timestamps
        assert history.latest.timestamp == 0.3


# ===================================================================
# C. Count bound
# ===================================================================
class TestCountBound:
    def test_exceeding_max_records_evicts_oldest(self):
        history = TemporalHistory(max_records=3, max_age_s=100.0, gap_limit_s=50.0)
        for ts in [0.0, 1.0, 2.0, 3.0, 4.0]:
            history.admit(_record(ts))
        assert len(history) == 3
        assert [r.timestamp for r in history.records] == [2.0, 3.0, 4.0]

    def test_newest_always_retained(self):
        history = TemporalHistory(max_records=1, max_age_s=100.0, gap_limit_s=50.0)
        for ts in [0.0, 1.0, 2.0]:
            history.admit(_record(ts))
        assert len(history) == 1
        assert history.latest.timestamp == 2.0

    def test_eviction_deterministic_across_independent_instances(self):
        history_a = TemporalHistory(max_records=2, max_age_s=100.0, gap_limit_s=50.0)
        history_b = TemporalHistory(max_records=2, max_age_s=100.0, gap_limit_s=50.0)
        for ts in [0.0, 1.0, 2.0, 3.0]:
            history_a.admit(_record(ts))
            history_b.admit(_record(ts))
        assert [r.timestamp for r in history_a.records] == [r.timestamp for r in history_b.records]


# ===================================================================
# D. Time bound
# ===================================================================
class TestTimeBound:
    def test_entries_outside_window_evicted(self):
        history = TemporalHistory(max_records=100, max_age_s=1.0, gap_limit_s=100.0)
        for ts in [0.0, 0.5, 1.0, 1.5, 2.0]:
            history.admit(_record(ts))
        assert [r.timestamp for r in history.records] == [1.0, 1.5, 2.0]

    def test_history_module_never_imports_time(self):
        """Age-based eviction must be timestamp-based, never wall-clock —
        proven structurally: temporal/history.py must not import the
        `time` module at all."""
        import depth_perception_engine.temporal.history as history_module

        tree = ast.parse(inspect.getsource(history_module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert "time" not in {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "time"


# ===================================================================
# E. Combined bounds
# ===================================================================
class TestCombinedBounds:
    def test_count_bound_tighter_than_age_bound_wins(self):
        history = TemporalHistory(max_records=2, max_age_s=100.0, gap_limit_s=50.0)
        for ts in [0.0, 1.0, 2.0]:
            history.admit(_record(ts))
        assert [r.timestamp for r in history.records] == [1.0, 2.0]

    def test_age_bound_tighter_than_count_bound_wins(self):
        history = TemporalHistory(max_records=100, max_age_s=1.0, gap_limit_s=50.0)
        for ts in [0.0, 0.5, 1.0, 1.5]:
            history.admit(_record(ts))
        assert [r.timestamp for r in history.records] == [0.5, 1.0, 1.5]


# ===================================================================
# F. Older timestamp
# ===================================================================
class TestOlderTimestamp:
    def test_older_timestamp_rejected(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        history.admit(_record(1.0))
        assert history.admit(_record(0.5)) == TemporalAdmissionStatus.REJECTED_OLDER_TIMESTAMP

    def test_history_unchanged_after_rejection(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        history.admit(_record(1.0))
        before = history.records
        history.admit(_record(0.5))
        assert history.records == before


# ===================================================================
# G. Duplicate timestamp
# ===================================================================
class TestDuplicateTimestamp:
    def test_duplicate_rejected(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        history.admit(_record(1.0, confidence=0.9))
        status = history.admit(_record(1.0, confidence=0.1))
        assert status == TemporalAdmissionStatus.REJECTED_DUPLICATE_TIMESTAMP

    def test_original_record_retained_unchanged(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        history.admit(_record(1.0, confidence=0.9))
        history.admit(_record(1.0, confidence=0.1))
        assert len(history) == 1
        assert history.latest.confidence == 0.9


# ===================================================================
# H. Out-of-order timestamp
# ===================================================================
class TestOutOfOrderTimestamp:
    def test_offending_entry_rejected_history_preserved_no_reorder(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        for ts in [10.0, 10.1, 10.2]:
            history.admit(_record(ts))

        status = history.admit(_record(10.15))

        assert status == TemporalAdmissionStatus.REJECTED_OLDER_TIMESTAMP
        assert [round(r.timestamp, 2) for r in history.records] == [10.0, 10.1, 10.2]


# ===================================================================
# I. Large gap
# ===================================================================
class TestLargeGap:
    def test_gap_exceeding_limit_clears_old_history_and_starts_new_sequence(self):
        history = TemporalHistory(max_records=10, max_age_s=100.0, gap_limit_s=0.5)
        for ts in [0.00, 0.03, 0.06, 0.09]:
            history.admit(_record(ts))
        assert len(history) == 4

        status = history.admit(_record(2.70))

        assert status == TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE
        assert len(history) == 1
        assert history.latest.timestamp == 2.70


# ===================================================================
# J. Gap below threshold
# ===================================================================
class TestGapBelowThreshold:
    def test_history_retained_and_observation_appended_normally(self):
        history = TemporalHistory(max_records=10, max_age_s=100.0, gap_limit_s=0.5)
        history.admit(_record(0.0))
        status = history.admit(_record(0.3))
        assert status == TemporalAdmissionStatus.ACCEPTED
        assert [r.timestamp for r in history.records] == [0.0, 0.3]


# ===================================================================
# Invalid timestamps (missing/NaN/Inf) — a chronology failure distinct
# from older/duplicate, still "no observation" for history purposes.
# ===================================================================
class TestInvalidTimestamp:
    def test_none_timestamp_rejected(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        assert history.admit(_record(None)) == TemporalAdmissionStatus.REJECTED_INVALID_TIMESTAMP
        assert len(history) == 0

    def test_nan_timestamp_rejected(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        assert history.admit(_record(float("nan"))) == TemporalAdmissionStatus.REJECTED_INVALID_TIMESTAMP

    def test_inf_timestamp_rejected(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        assert history.admit(_record(float("inf"))) == TemporalAdmissionStatus.REJECTED_INVALID_TIMESTAMP


# ===================================================================
# N. Degradation and recovery
# ===================================================================
class TestDegradationRecovery:
    def test_good_invalid_good_sequence_all_admitted(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        records = [
            _record(0.0, geometry_quality=GeometryQuality.HEALTHY),
            _record(0.1, geometry_quality=GeometryQuality.HEALTHY),
            _record(0.2, geometry_quality=GeometryQuality.NO_USABLE_GEOMETRY),
            _record(0.3, geometry_quality=GeometryQuality.NO_USABLE_GEOMETRY),
            _record(0.4, geometry_quality=GeometryQuality.HEALTHY),
            _record(0.5, geometry_quality=GeometryQuality.HEALTHY),
        ]
        statuses = [history.admit(r) for r in records]

        assert all(s == TemporalAdmissionStatus.ACCEPTED for s in statuses)
        assert len(history) == 6

    def test_no_stuck_degraded_state_subsequent_good_frame_accepted(self):
        history = TemporalHistory(max_records=10, max_age_s=10.0, gap_limit_s=5.0)
        history.admit(_record(0.0, geometry_quality=GeometryQuality.NO_USABLE_GEOMETRY))
        history.admit(_record(0.1, geometry_quality=GeometryQuality.NO_USABLE_GEOMETRY))

        status = history.admit(_record(0.2, geometry_quality=GeometryQuality.HEALTHY))

        assert status == TemporalAdmissionStatus.ACCEPTED
        assert history.latest.geometry_quality == GeometryQuality.HEALTHY

    def test_admission_never_inspects_quality_or_confidence(self):
        """Structural proof, not just behavioral: degradation/recovery
        works because admit() is a pure function of timestamps — confirmed
        by construction, not coincidence. AST-based (attribute-access
        nodes only), not a raw text search, so this doesn't false-positive
        on the method's own docstring mentioning these field names while
        explaining the guarantee (same reasoning as
        tests/test_level4_architecture_guards.py)."""
        import textwrap

        source = textwrap.dedent(inspect.getsource(TemporalHistory.admit))
        tree = ast.parse(source)
        accessed_attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert "geometry_quality" not in accessed_attrs
        assert "confidence" not in accessed_attrs


# ===================================================================
# O. Memory / bounding
# ===================================================================
class TestMemoryBounding:
    def test_repeated_insertion_cannot_exceed_configured_bound(self):
        history = TemporalHistory(max_records=5, max_age_s=1000.0, gap_limit_s=500.0)
        for i in range(1000):
            history.admit(_record(float(i)))
        assert len(history) == 5

    def test_old_records_actually_released(self):
        """TemporalRecord uses slots=True (matching every other frozen
        contract in this codebase), so it carries no __weakref__ slot —
        weakref.ref() cannot target it. Proven instead via refcount: once
        evicted, the only remaining reference is this test's own local
        variable plus getrefcount()'s own transient argument reference."""
        history = TemporalHistory(max_records=2, max_age_s=1000.0, gap_limit_s=500.0)
        first = _record(0.0)
        history.admit(first)
        history.admit(_record(1.0))
        history.admit(_record(2.0))  # evicts the first record (count bound)
        gc.collect()

        assert first not in history.records
        assert sys.getrefcount(first) == 2


# ===================================================================
# Contract-level: TemporalRecord itself
# ===================================================================
class TestTemporalRecordContract:
    def test_frozen_immutable(self):
        record = _record(1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.timestamp = 2.0

    def test_field_set_is_exactly_the_frozen_e2_e3_contract(self):
        """depth_snapshot_m was added at E3 (docs/E3_IMPLEMENTATION_PLAN.md's
        Decision 5) — updated here alongside that addition, not widened
        silently. See tests/test_temporal_consistency.py for coverage of
        what it's actually used for."""
        names = {f.name for f in dataclasses.fields(TemporalRecord)}
        assert names == {"timestamp", "confidence", "geometry_quality", "motion_hint", "depth_snapshot_m"}


class TestScopeDiscipline:
    """E2 is infrastructure only — guards against exactly the algorithm
    scope creep this phase's own instructions forbid."""

    def test_temporal_history_has_no_filtering_fusion_or_algorithm_methods(self):
        forbidden = {
            "smooth", "fuse", "average", "filter", "stabilize", "align",
            "compensate", "integrate", "classify_persistence", "estimate_pose",
        }
        public_methods = {name for name in dir(TemporalHistory) if not name.startswith("_")}
        assert not (public_methods & forbidden)


# ===================================================================
# Pipeline integration
# ===================================================================
class TestPipelineDisabledByDefault:
    def test_temporal_history_property_is_none_by_default(self, config, calibration):
        pipeline = DepthPerceptionPipeline(config, calibration)
        assert pipeline.temporal_history is None

    def test_result_temporal_admission_status_is_none_by_default(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.temporal_admission_status is None


class TestZeroRegressionOnLevel3Output:
    def test_enabling_temporal_does_not_change_level3_numeric_output(self, calibration, stereo_pair):
        left, right = stereo_pair
        pipeline_off = DepthPerceptionPipeline(PipelineConfig(enable_temporal=False), calibration)
        pipeline_on = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)

        result_off = pipeline_off.process(left, right, left_timestamp=1.0)
        result_on = pipeline_on.process(left, right, left_timestamp=1.0)

        np.testing.assert_array_equal(result_off.disparity_map, result_on.disparity_map)
        np.testing.assert_array_equal(result_off.depth_map, result_on.depth_map)
        assert result_off.confidence == result_on.confidence
        assert result_off.timestamp == result_on.timestamp
        assert result_off.temporal_admission_status is None
        assert result_on.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED


class TestPipelineIntegration:
    def test_enabling_constructs_temporal_history(self, calibration):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        assert pipeline.temporal_history is not None
        assert len(pipeline.temporal_history) == 0

    def test_process_admits_frame_with_valid_timestamp(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right, left_timestamp=1.0)

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED
        assert len(pipeline.temporal_history) == 1
        assert pipeline.temporal_history.latest.timestamp == 1.0

    def test_missing_timestamp_rejected_but_level3_result_still_returned(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)  # no timestamp at all

        assert result.temporal_admission_status == TemporalAdmissionStatus.REJECTED_INVALID_TIMESTAMP
        assert len(pipeline.temporal_history) == 0
        assert isinstance(result, DepthPerceptionResult)
        assert result.disparity_map is not None

    def test_older_timestamp_does_not_destroy_level3_result(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=5.0)

        result = pipeline.process(left, right, left_timestamp=1.0)

        assert result.temporal_admission_status == TemporalAdmissionStatus.REJECTED_OLDER_TIMESTAMP
        assert isinstance(result, DepthPerceptionResult)
        assert result.confidence >= 0.0
        assert len(pipeline.temporal_history) == 1  # only the first frame's record

    def test_reset_clears_temporal_history(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)
        pipeline.process(left, right, left_timestamp=1.1)
        assert len(pipeline.temporal_history) == 2

        pipeline.reset()

        assert len(pipeline.temporal_history) == 0
        assert pipeline.temporal_history.latest is None

    def test_next_frame_after_reset_starts_fresh_sequence(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=5.0)
        pipeline.reset()

        # A timestamp "older" than the pre-reset newest is fine now — no
        # historical influence survives reset().
        result = pipeline.process(left, right, left_timestamp=1.0)

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED
        assert pipeline.temporal_history.latest.timestamp == 1.0

    def test_reset_does_not_affect_ordinary_level3_lifecycle(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        pipeline.reset()
        result = pipeline.process(left, right, left_timestamp=2.0)

        assert isinstance(result, DepthPerceptionResult)
        assert pipeline.health().frames_processed == 1

    def test_close_then_process_still_raises_with_temporal_enabled(self, calibration):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        pipeline.close()
        with pytest.raises(RuntimeError):
            pipeline.process(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))


# ===================================================================
# K. Missing motion hint / motion-hint association
# ===================================================================
class TestMotionHintAssociation:
    def test_missing_motion_hint_admitted_normally(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right, left_timestamp=1.0)  # no motion_hint

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED
        assert pipeline.temporal_history.latest.motion_hint is None

    def test_motion_hint_associated_with_record(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        hint = MotionHint(timestamp=1.0, angular_velocity_rad_s=np.array([0.01, 0.0, 0.0]), frame_id=FrameId.BODY)

        pipeline.process(left, right, left_timestamp=1.0, motion_hint=hint)

        assert pipeline.temporal_history.latest.motion_hint is hint

    def test_process_observation_forwards_motion_hint(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        hint = MotionHint(timestamp=1.0, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY)
        obs = StereoObservation(left_image=left, right_image=right, left_timestamp=1.0, motion_hint=hint)

        pipeline.process_observation(obs)

        assert pipeline.temporal_history.latest.motion_hint is hint

    def test_invalid_motion_hint_still_associated_not_specially_handled(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        hint = MotionHint(
            timestamp=1.0, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY, valid=False,
        )

        result = pipeline.process(left, right, left_timestamp=1.0, motion_hint=hint)

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED
        assert pipeline.temporal_history.latest.motion_hint.valid is False


# ===================================================================
# L. NO_USABLE_GEOMETRY observations represented in chronology
# ===================================================================
class TestNoUsableGeometryRepresentedInChronology:
    def test_no_usable_geometry_frame_still_admitted_with_honest_marker(self, calibration):
        config = PipelineConfig(enable_temporal=True, enable_geometry=True)
        transform = _illustrative_transform()
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)
        left, right = _flat_textureless_pair(calibration)

        result = pipeline.process(left, right, left_timestamp=1.0)

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED
        assert pipeline.temporal_history.latest.geometry_quality == GeometryQuality.NO_USABLE_GEOMETRY

    def test_geometry_quality_is_none_when_geometry_disabled(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair

        pipeline.process(left, right, left_timestamp=1.0)

        assert pipeline.temporal_history.latest.geometry_quality is None

    def test_no_observation_distinguishable_from_admitted_no_usable_geometry(self, calibration):
        """Decision 9's core distinction: (A) no observation occurred (a
        rejected admission leaves no record at all) vs. (B) an observation
        occurred but produced no usable geometry (a record exists, marked
        accordingly)."""
        config = PipelineConfig(enable_temporal=True, enable_geometry=True)
        transform = _illustrative_transform()
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)
        left, right = _flat_textureless_pair(calibration)

        pipeline.process(left, right, left_timestamp=5.0)
        rejected_result = pipeline.process(left, right, left_timestamp=1.0)  # older -> no record (A)

        assert rejected_result.temporal_admission_status == TemporalAdmissionStatus.REJECTED_OLDER_TIMESTAMP
        assert len(pipeline.temporal_history) == 1
        assert pipeline.temporal_history.latest.geometry_quality == GeometryQuality.NO_USABLE_GEOMETRY  # (B)

    def test_no_geometry_object_fabricated_only_a_decimated_depth_snapshot_retained(self, calibration):
        """geometry_quality is a category label only. Level 4, Phase E3
        (docs/E3_IMPLEMENTATION_PLAN.md's Decision 5) deliberately does
        retain ONE small array — TemporalRecord.depth_snapshot_m, a
        decimated depth_map — but never a geometry.PointCloud/
        ObstacleCloud/FreeSpaceRays object, and never the full-resolution
        depth_map itself."""
        from depth_perception_engine.geometry import FreeSpaceRays, ObstacleCloud, PointCloud

        config = PipelineConfig(enable_temporal=True, enable_geometry=True)
        transform = _illustrative_transform()
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)
        left, right = _flat_textureless_pair(calibration)

        result = pipeline.process(left, right, left_timestamp=1.0)

        record = pipeline.temporal_history.latest
        for field_value in dataclasses.astuple(record):
            assert not isinstance(field_value, (PointCloud, ObstacleCloud, FreeSpaceRays))

        stride = config.temporal_consistency_sampling_stride
        expected_shape = result.depth_map[::stride, ::stride].shape
        assert record.depth_snapshot_m.shape == expected_shape
        assert record.depth_snapshot_m.shape != result.depth_map.shape  # strictly smaller than full resolution


# ===================================================================
# PipelineConfig: new Level 4 fields
# ===================================================================
class TestPipelineConfigTemporalFields:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.enable_temporal is False
        assert config.temporal_max_records == 30
        assert config.temporal_max_age_s == 1.0
        assert config.temporal_gap_limit_s == 0.5

    def test_rejects_invalid_max_records(self):
        with pytest.raises(ValueError):
            PipelineConfig(temporal_max_records=0)

    def test_rejects_invalid_max_age(self):
        with pytest.raises(ValueError):
            PipelineConfig(temporal_max_age_s=0.0)

    def test_rejects_invalid_gap_limit(self):
        with pytest.raises(ValueError):
            PipelineConfig(temporal_gap_limit_s=-1.0)


# ===================================================================
# Determinism
# ===================================================================
class TestDeterminism:
    def test_repeated_processing_produces_identical_admission_statuses(self, calibration, stereo_pair):
        left, right = stereo_pair
        pipeline_a = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        pipeline_b = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)

        statuses_a = [pipeline_a.process(left, right, left_timestamp=t).temporal_admission_status for t in [0.0, 0.1, 0.2]]
        statuses_b = [pipeline_b.process(left, right, left_timestamp=t).temporal_admission_status for t in [0.0, 0.1, 0.2]]

        assert statuses_a == statuses_b == [TemporalAdmissionStatus.ACCEPTED] * 3
