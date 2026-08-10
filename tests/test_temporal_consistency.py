"""
Level 4, Phase E3 tests — read-only temporal consistency.

Covers: the pure compute_temporal_consistency() function (all four states,
the score=0 ambiguity resolution, pure/vectorized/non-mutating), the
TemporalConsistency/TemporalConsistencyState contracts, and
DepthPerceptionPipeline integration (the full Decision 2 trigger table,
proof that current Level 3 evidence is never rewritten under any outcome
including CONTRADICTORY, and memory bounding of the new
TemporalRecord.depth_snapshot_m field).

No E4 algorithm is tested — E3 performs no smoothing, fusion, IMU
integration, rotation compensation, or persistence classification, so none
of that is exercised here.
"""

import ast
import dataclasses
import inspect

import numpy as np
import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.geometry import FreeSpaceRays, ObstacleCloud, PointCloud
from depth_perception_engine.models.result import DepthPerceptionResult
from depth_perception_engine.pipeline import DepthPerceptionPipeline
from depth_perception_engine.temporal import (
    TemporalAdmissionStatus,
    TemporalConsistency,
    TemporalConsistencyState,
    TemporalRecord,
    compute_temporal_consistency,
)


def _snapshot(*rows):
    return np.array(rows, dtype=np.float32)


def _prior(depth_snapshot_m, timestamp=0.0):
    return TemporalRecord(timestamp=timestamp, confidence=0.8, depth_snapshot_m=depth_snapshot_m)


# ===================================================================
# compute_temporal_consistency() — pure function
# ===================================================================
class TestNoPriorRecord:
    def test_none_previous_record_is_insufficient_evidence(self):
        current = _snapshot([1.0, 2.0])
        result = compute_temporal_consistency(current, None, agreement_tolerance_m=0.05, min_agreement_fraction=0.7)
        assert result.state == TemporalConsistencyState.INSUFFICIENT_EVIDENCE
        assert result.comparable_count == 0
        assert result.agreeing_count == 0
        assert result.disagreement_count == 0
        assert result.agreement_fraction is None

    def test_previous_record_with_no_snapshot_is_insufficient_evidence(self):
        """Defensive: a TemporalRecord constructed without a snapshot
        (depth_snapshot_m=None, its own default) — not a real path through
        DepthPerceptionPipeline, but must not crash."""
        current = _snapshot([1.0, 2.0])
        previous = TemporalRecord(timestamp=0.0, confidence=0.8)  # depth_snapshot_m defaults None
        result = compute_temporal_consistency(current, previous, 0.05, 0.7)
        assert result.state == TemporalConsistencyState.INSUFFICIENT_EVIDENCE


class TestShapeMismatch:
    def test_different_shapes_are_not_comparable(self):
        current = _snapshot([1.0, 2.0, 3.0])
        previous = _prior(_snapshot([1.0, 2.0]))
        result = compute_temporal_consistency(current, previous, 0.05, 0.7)
        assert result.state == TemporalConsistencyState.NOT_COMPARABLE
        assert result.comparable_count == 0
        assert result.agreement_fraction is None


class TestZeroOverlapIsInsufficientNotContradictory:
    def test_no_pixel_valid_in_both_frames(self):
        """The score=0 ambiguity Decision 1 explicitly warns about:
        comparable_count == 0 must be INSUFFICIENT_EVIDENCE, never
        CONTRADICTORY and never a 0/0 agreement_fraction."""
        current = _snapshot([1.0, 0.0])       # only index 0 valid
        previous = _prior(_snapshot([0.0, 2.0]))  # only index 1 valid — no overlap
        result = compute_temporal_consistency(current, previous, 0.05, 0.7)
        assert result.state == TemporalConsistencyState.INSUFFICIENT_EVIDENCE
        assert result.comparable_count == 0
        assert result.agreement_fraction is None


class TestConsistentAndContradictory:
    def test_full_agreement_is_consistent_with_fraction_one(self):
        current = _snapshot([1.00, 2.00, 3.00])
        previous = _prior(_snapshot([1.01, 1.99, 3.02]))  # all within 0.05 tolerance
        result = compute_temporal_consistency(current, previous, agreement_tolerance_m=0.05, min_agreement_fraction=0.7)
        assert result.state == TemporalConsistencyState.CONSISTENT
        assert result.comparable_count == 3
        assert result.agreeing_count == 3
        assert result.disagreement_count == 0
        assert result.agreement_fraction == 1.0

    def test_full_disagreement_is_contradictory_with_fraction_zero(self):
        current = _snapshot([1.0, 2.0, 3.0])
        previous = _prior(_snapshot([5.0, 6.0, 7.0]))  # all far outside tolerance
        result = compute_temporal_consistency(current, previous, agreement_tolerance_m=0.05, min_agreement_fraction=0.7)
        assert result.state == TemporalConsistencyState.CONTRADICTORY
        assert result.comparable_count == 3
        assert result.agreeing_count == 0
        assert result.disagreement_count == 3
        assert result.agreement_fraction == 0.0
        # Distinguishable from INSUFFICIENT_EVIDENCE precisely because a
        # real number exists here, unlike the zero-overlap case above.
        assert result.agreement_fraction is not None

    def test_mixed_agreement_counts_are_consistent_with_each_other(self):
        current = _snapshot([1.0, 2.0, 3.0, 4.0])
        previous = _prior(_snapshot([1.0, 2.0, 9.0, 9.0]))  # 2 agree, 2 disagree
        result = compute_temporal_consistency(current, previous, agreement_tolerance_m=0.05, min_agreement_fraction=0.7)
        assert result.comparable_count == 4
        assert result.agreeing_count == 2
        assert result.disagreement_count == 2
        assert result.agreement_fraction == 0.5
        assert result.disagreement_count == result.comparable_count - result.agreeing_count

    def test_agreement_fraction_exactly_at_threshold_is_consistent(self):
        """>= is inclusive, matching PipelineConfig's own boundary
        convention elsewhere (e.g. GeometryQuality's HEALTHY check)."""
        current = _snapshot([1.0, 1.0, 1.0, 9.0])  # 3/4 = 0.75 agree
        previous = _prior(_snapshot([1.0, 1.0, 1.0, 1.0]))
        result = compute_temporal_consistency(current, previous, agreement_tolerance_m=0.05, min_agreement_fraction=0.75)
        assert result.agreement_fraction == 0.75
        assert result.state == TemporalConsistencyState.CONSISTENT

    def test_tolerance_boundary_is_inclusive(self):
        current = _snapshot([1.05])
        previous = _prior(_snapshot([1.00]))
        result = compute_temporal_consistency(current, previous, agreement_tolerance_m=0.05, min_agreement_fraction=0.7)
        assert result.agreeing_count == 1  # diff == tolerance counts as agreeing


class TestPureFunction:
    def test_does_not_mutate_inputs(self):
        current = _snapshot([1.0, 2.0, 3.0])
        previous_snapshot = _snapshot([9.0, 9.0, 9.0])
        current_copy = current.copy()
        previous_copy = previous_snapshot.copy()
        compute_temporal_consistency(current, _prior(previous_snapshot), 0.05, 0.7)
        np.testing.assert_array_equal(current, current_copy)
        np.testing.assert_array_equal(previous_snapshot, previous_copy)

    def test_no_for_or_while_loop_fully_vectorized(self):
        import depth_perception_engine.temporal.consistency as consistency_module

        tree = ast.parse(inspect.getsource(consistency_module))
        offenders = [node.lineno for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
        assert not offenders, f"Found Python for/while loop(s) at line(s): {offenders}"


# ===================================================================
# Contracts
# ===================================================================
class TestTemporalConsistencyContract:
    def test_frozen_immutable(self):
        result = TemporalConsistency(
            state=TemporalConsistencyState.CONSISTENT, comparable_count=1,
            agreeing_count=1, disagreement_count=0, agreement_fraction=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.state = TemporalConsistencyState.CONTRADICTORY

    def test_field_set_is_exactly_the_frozen_e3_contract(self):
        names = {f.name for f in dataclasses.fields(TemporalConsistency)}
        assert names == {"state", "comparable_count", "agreeing_count", "disagreement_count", "agreement_fraction"}


class TestTemporalConsistencyStateContract:
    def test_exactly_the_four_architect_specified_states(self):
        values = {
            TemporalConsistencyState.CONSISTENT,
            TemporalConsistencyState.CONTRADICTORY,
            TemporalConsistencyState.INSUFFICIENT_EVIDENCE,
            TemporalConsistencyState.NOT_COMPARABLE,
        }
        assert values == {"CONSISTENT", "CONTRADICTORY", "INSUFFICIENT_EVIDENCE", "NOT_COMPARABLE"}


# ===================================================================
# Pipeline integration — Decision 2's trigger table
# ===================================================================
class TestPipelineDisabledByDefault:
    def test_temporal_consistency_is_none_when_temporal_disabled(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.temporal_consistency is None


class TestTriggerTable:
    def test_first_frame_ever_is_insufficient_evidence(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right, left_timestamp=1.0)

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED
        assert result.temporal_consistency.state == TemporalConsistencyState.INSUFFICIENT_EVIDENCE
        assert result.temporal_consistency.agreement_fraction is None

    def test_first_frame_after_reset_is_insufficient_evidence(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)
        pipeline.process(left, right, left_timestamp=1.1)  # would be CONSISTENT if not reset
        pipeline.reset()

        result = pipeline.process(left, right, left_timestamp=5.0)

        assert result.temporal_consistency.state == TemporalConsistencyState.INSUFFICIENT_EVIDENCE

    def test_gap_triggered_new_sequence_is_insufficient_evidence_not_compared_to_pre_gap_record(
        self, calibration, stereo_pair,
    ):
        config = PipelineConfig(enable_temporal=True, temporal_gap_limit_s=0.5)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)  # establishes a prior record

        result = pipeline.process(left, right, left_timestamp=10.0)  # gap > 0.5s

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE
        assert result.temporal_consistency.state == TemporalConsistencyState.INSUFFICIENT_EVIDENCE

    def test_duplicate_timestamp_rejection_produces_no_comparison(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        result = pipeline.process(left, right, left_timestamp=1.0)  # duplicate

        assert result.temporal_admission_status == TemporalAdmissionStatus.REJECTED_DUPLICATE_TIMESTAMP
        assert result.temporal_consistency is None

    def test_older_timestamp_rejection_produces_no_comparison(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=5.0)

        result = pipeline.process(left, right, left_timestamp=1.0)  # older

        assert result.temporal_admission_status == TemporalAdmissionStatus.REJECTED_OLDER_TIMESTAMP
        assert result.temporal_consistency is None

    def test_invalid_timestamp_rejection_produces_no_comparison(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)  # no timestamp at all

        assert result.temporal_admission_status == TemporalAdmissionStatus.REJECTED_INVALID_TIMESTAMP
        assert result.temporal_consistency is None

    def test_identical_consecutive_frames_are_consistent(self, calibration, stereo_pair):
        """Processing the same stereo pair twice in a row must produce a
        byte-identical depth_map (deterministic SGBM), so every
        comparable pixel agrees exactly — the real-pipeline, non-mocked
        proof of CONSISTENT."""
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        result = pipeline.process(left, right, left_timestamp=1.1)

        assert result.temporal_consistency.state == TemporalConsistencyState.CONSISTENT
        assert result.temporal_consistency.agreement_fraction == 1.0
        assert result.temporal_consistency.disagreement_count == 0

    def test_missing_motion_hint_does_not_block_consistency_evaluation(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_temporal=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)  # no motion_hint

        result = pipeline.process(left, right, left_timestamp=1.1)  # no motion_hint

        assert result.temporal_consistency is not None
        assert result.temporal_consistency.state == TemporalConsistencyState.CONSISTENT


class TestContradictoryThroughRealPipeline:
    """Forces a real CONTRADICTORY verdict through DepthPerceptionPipeline
    itself (not just the pure function) by directly admitting a
    synthetic, deliberately-very-different prior record into the
    pipeline's own exposed temporal_history — a legitimate use of that
    read-write-capable property (mirrors .config/.calibration's own
    "exposed, not specially guarded" precedent), not a private/internal
    reach-in."""

    def test_real_pipeline_reports_contradictory_when_prior_evidence_differs_sharply(
        self, calibration, stereo_pair,
    ):
        config = PipelineConfig(enable_temporal=True)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        baseline_result = pipeline.process(left, right, left_timestamp=1.0)

        stride = config.temporal_consistency_sampling_stride
        real_snapshot = baseline_result.depth_map[::stride, ::stride]
        assert (real_snapshot > 0.0).any(), "fixture must produce at least some valid depth"
        far_off_snapshot = np.where(real_snapshot > 0.0, real_snapshot + 5.0, 0.0).astype(np.float32)
        # Timestamps kept within the default temporal_gap_limit_s (0.5s) of
        # each other so admission stays plain ACCEPTED, not
        # ACCEPTED_NEW_SEQUENCE (which would correctly wipe this synthetic
        # record before the real comparison — not what this test wants to
        # exercise).
        synthetic_prior = TemporalRecord(timestamp=1.1, confidence=0.5, depth_snapshot_m=far_off_snapshot)
        pipeline.temporal_history.admit(synthetic_prior)

        result = pipeline.process(left, right, left_timestamp=1.2)

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED
        assert result.temporal_consistency.state == TemporalConsistencyState.CONTRADICTORY
        assert result.temporal_consistency.comparable_count > 0
        assert result.temporal_consistency.agreement_fraction < 0.7


# ===================================================================
# Decision 4 — current evidence is never rewritten, proven not claimed
# ===================================================================
class TestCurrentEvidenceUnchanged:
    def test_contradictory_verdict_does_not_alter_any_level3_field(self, calibration, stereo_pair):
        left, right = stereo_pair

        config_on = PipelineConfig(enable_temporal=True)
        pipeline_on = DepthPerceptionPipeline(config_on, calibration)
        baseline = pipeline_on.process(left, right, left_timestamp=1.0)
        stride = config_on.temporal_consistency_sampling_stride
        real_snapshot = baseline.depth_map[::stride, ::stride]
        far_off_snapshot = np.where(real_snapshot > 0.0, real_snapshot + 5.0, 0.0).astype(np.float32)
        pipeline_on.temporal_history.admit(TemporalRecord(timestamp=1.1, confidence=0.5, depth_snapshot_m=far_off_snapshot))
        result_on = pipeline_on.process(left, right, left_timestamp=1.2)
        assert result_on.temporal_consistency.state == TemporalConsistencyState.CONTRADICTORY

        pipeline_off = DepthPerceptionPipeline(PipelineConfig(enable_temporal=False), calibration)
        pipeline_off.process(left, right, left_timestamp=1.0)
        pipeline_off.process(left, right, left_timestamp=1.1)
        result_off = pipeline_off.process(left, right, left_timestamp=1.2)

        np.testing.assert_array_equal(result_on.disparity_map, result_off.disparity_map)
        np.testing.assert_array_equal(result_on.depth_map, result_off.depth_map)
        np.testing.assert_array_equal(result_on.valid_disparity_mask, result_off.valid_disparity_mask)
        np.testing.assert_array_equal(result_on.valid_depth_mask, result_off.valid_depth_mask)
        assert result_on.confidence == result_off.confidence
        assert result_on.traversability_mask.decision == result_off.traversability_mask.decision

    def test_compute_temporal_consistency_receives_no_mutable_level3_reference(self):
        """Structural proof: the function's own signature never accepts a
        DepthPerceptionResult, geometry.PointCloud, or any Level 3 type —
        it cannot rewrite what it was never given."""
        sig = inspect.signature(compute_temporal_consistency)
        param_annotations = [str(p.annotation) for p in sig.parameters.values()]
        for forbidden in ("DepthPerceptionResult", "PointCloud", "ObstacleCloud", "FreeSpaceRays"):
            assert not any(forbidden in a for a in param_annotations)


# ===================================================================
# Memory bounding of the new TemporalRecord.depth_snapshot_m field
# ===================================================================
class TestMemoryBounding:
    def test_snapshot_is_decimated_not_full_resolution(self, calibration, stereo_pair):
        config = PipelineConfig(enable_temporal=True, temporal_consistency_sampling_stride=4)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right, left_timestamp=1.0)

        snapshot = pipeline.temporal_history.latest.depth_snapshot_m
        full_h, full_w = result.depth_map.shape
        assert snapshot.shape[0] < full_h
        assert snapshot.shape[1] < full_w
        assert snapshot.nbytes < result.depth_map.nbytes

    def test_stride_one_means_no_decimation(self, calibration, stereo_pair):
        config = PipelineConfig(enable_temporal=True, temporal_consistency_sampling_stride=1)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right, left_timestamp=1.0)

        snapshot = pipeline.temporal_history.latest.depth_snapshot_m
        assert snapshot.shape == result.depth_map.shape

    def test_history_buffer_still_bounded_by_max_records_with_larger_records(self, calibration, stereo_pair):
        config = PipelineConfig(enable_temporal=True, temporal_max_records=3, temporal_max_age_s=1000.0, temporal_gap_limit_s=500.0)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair

        for i in range(10):
            pipeline.process(left, right, left_timestamp=float(i))

        assert len(pipeline.temporal_history) == 3


# ===================================================================
# PipelineConfig: new Level 4, Phase E3 fields
# ===================================================================
class TestPipelineConfigTemporalConsistencyFields:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.temporal_consistency_sampling_stride == 4
        assert config.temporal_consistency_agreement_tolerance_m == 0.05
        assert config.temporal_consistency_min_agreement_fraction == 0.7

    def test_rejects_invalid_stride(self):
        with pytest.raises(ValueError):
            PipelineConfig(temporal_consistency_sampling_stride=0)

    def test_rejects_negative_tolerance(self):
        with pytest.raises(ValueError):
            PipelineConfig(temporal_consistency_agreement_tolerance_m=-0.01)

    def test_rejects_out_of_range_min_agreement_fraction(self):
        with pytest.raises(ValueError):
            PipelineConfig(temporal_consistency_min_agreement_fraction=1.5)
        with pytest.raises(ValueError):
            PipelineConfig(temporal_consistency_min_agreement_fraction=-0.1)


# ===================================================================
# Scope discipline
# ===================================================================
class TestScopeDiscipline:
    def test_temporal_consistency_state_has_no_source_or_provenance_leak(self):
        """Guards against exactly the branching this repository's Level 4
        instructions forbid — TemporalConsistencyState must never be used
        to encode where evidence came from, only whether it agrees."""
        names = {f.name for f in dataclasses.fields(TemporalConsistency)}
        forbidden = {"source", "is_simulated", "provenance"}
        assert not (names & forbidden)

    def test_no_result_type_is_a_pointcloud_wrapper(self):
        """TemporalConsistency must remain small scalars/state only —
        never grow a pixel-level payload."""
        for field in dataclasses.fields(TemporalConsistency):
            assert field.type in ("str", "int", "Optional[float]", "float") or "float" in str(field.type) or "int" in str(field.type) or "str" in str(field.type)
