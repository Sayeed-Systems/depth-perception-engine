"""
Level 4, Phase E4 tests — deterministic, confidence-aware temporal
stabilization.

Covers: the pure compute_temporal_stabilization() function (gate failures,
per-pixel blend/override math, the UNKNOWN-never-becomes-FREE safety
property), the TemporalStabilization/TemporalStabilizationState contracts,
and DepthPerceptionPipeline integration (every scenario the architect's
Decision 10 listed: first frame, consistent/noisy stabilization, strong
contradiction override, invalid history, non-comparable, large gap/reset,
missing MotionHint, raw Level 3 output unchanged, determinism, and memory
bounding).

No E5 algorithm is tested — E4 performs no IMU integration, motion
warping, ego-motion estimation, or rotation/translation compensation, so
none of that is exercised here.
"""

import ast
import dataclasses
import inspect

import numpy as np
import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.models.result import DepthPerceptionResult, StereoObservation
from depth_perception_engine.pipeline import DepthPerceptionPipeline
from depth_perception_engine.temporal import (
    TemporalAdmissionStatus,
    TemporalConsistency,
    TemporalConsistencyState,
    TemporalRecord,
    TemporalStabilization,
    TemporalStabilizationState,
    compute_temporal_stabilization,
)


def _snapshot(row):
    return np.array(row, dtype=np.float32)


def _prior(depth_snapshot_m, confidence=0.6, timestamp=0.0):
    return TemporalRecord(timestamp=timestamp, confidence=confidence, depth_snapshot_m=depth_snapshot_m)


def _consistency(state, comparable_count=1, agreeing_count=1):
    return TemporalConsistency(
        state=state, comparable_count=comparable_count, agreeing_count=agreeing_count,
        disagreement_count=comparable_count - agreeing_count,
        agreement_fraction=(agreeing_count / comparable_count) if comparable_count else None,
    )


_DEFAULT_KWARGS = dict(agreement_tolerance_m=0.05, min_history_confidence=0.3, min_comparable_fraction=0.1)


# ===================================================================
# compute_temporal_stabilization() — gate failures
# ===================================================================
class TestGateFailures:
    def test_no_previous_record_is_insufficient_evidence(self):
        current = _snapshot([1.0, 2.0])
        result = compute_temporal_stabilization(current, None, None, **_DEFAULT_KWARGS)
        assert result.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE
        assert result.stabilized_depth_m is None
        assert result.eligible_count == 0
        assert result.stabilized_count == 0
        assert result.contradiction_count == 0
        assert result.stabilized_fraction is None
        assert result.contradiction_fraction is None

    def test_not_comparable_consistency_is_insufficient_evidence(self):
        current = _snapshot([1.0, 2.0])
        previous = _prior(_snapshot([9.0, 9.0]))
        consistency = _consistency(TemporalConsistencyState.NOT_COMPARABLE, comparable_count=0, agreeing_count=0)
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)
        assert result.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE

    def test_insufficient_evidence_consistency_is_insufficient_evidence(self):
        current = _snapshot([1.0, 2.0])
        previous = _prior(_snapshot([9.0, 9.0]))
        consistency = _consistency(TemporalConsistencyState.INSUFFICIENT_EVIDENCE, comparable_count=0, agreeing_count=0)
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)
        assert result.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE

    def test_low_history_confidence_is_insufficient_evidence(self):
        """'Invalid history' — the architect's own phrasing for this case."""
        current = _snapshot([1.0, 2.0])
        previous = _prior(_snapshot([1.0, 2.0]), confidence=0.1)  # below default 0.3 threshold
        consistency = _consistency(TemporalConsistencyState.CONSISTENT, comparable_count=2, agreeing_count=2)
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)
        assert result.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE

    def test_insufficient_comparable_fraction_is_insufficient_evidence(self):
        current = _snapshot([1.0] * 100)
        previous = _prior(_snapshot([1.0] + [0.0] * 99))  # only 1/100 = 1% comparable, below 10% default
        consistency = _consistency(TemporalConsistencyState.CONSISTENT, comparable_count=1, agreeing_count=1)
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)
        assert result.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE

    def test_exactly_at_thresholds_passes_the_gate(self):
        current = _snapshot([1.0] * 10)
        previous = _prior(_snapshot([1.0] * 10), confidence=0.3)  # exactly the min
        consistency = _consistency(TemporalConsistencyState.CONSISTENT, comparable_count=1, agreeing_count=1)  # 1/10 = 0.1 exactly
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)
        assert result.state != TemporalStabilizationState.INSUFFICIENT_EVIDENCE


# ===================================================================
# Per-pixel math
# ===================================================================
class TestPerPixelMath:
    def test_agreeing_pixel_is_confidence_weighted_blended(self):
        current = _snapshot([1.00])
        previous = _prior(_snapshot([1.02]), confidence=0.5)
        consistency = _consistency(TemporalConsistencyState.CONSISTENT, comparable_count=1, agreeing_count=1)
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)
        expected = (1.00 + 0.5 * 1.02) / 1.5
        assert result.state == TemporalStabilizationState.STABILIZED
        np.testing.assert_allclose(result.stabilized_depth_m[0], expected, atol=1e-5)
        assert result.stabilized_count == 1
        assert result.contradiction_count == 0
        assert result.stabilized_fraction == 1.0

    def test_zero_confidence_history_degrades_to_current_value_exactly(self):
        current = _snapshot([1.0])
        previous = _prior(_snapshot([9.0]), confidence=0.0)  # below gate normally, but test the formula's own limit
        consistency = _consistency(TemporalConsistencyState.CONTRADICTORY, comparable_count=1, agreeing_count=0)
        # Force past the gate by using a permissive min_history_confidence for this specific check.
        result = compute_temporal_stabilization(
            current, previous, consistency,
            agreement_tolerance_m=20.0, min_history_confidence=0.0, min_comparable_fraction=0.1,
        )
        # tolerance=20 forces "agrees" so we can isolate the w=0 blend formula.
        assert result.stabilized_depth_m[0] == pytest.approx(1.0)

    def test_contradicting_pixel_keeps_current_value_exactly(self):
        current = _snapshot([1.0])
        previous = _prior(_snapshot([9.0]), confidence=0.9)  # high confidence, but far off
        consistency = _consistency(TemporalConsistencyState.CONTRADICTORY, comparable_count=1, agreeing_count=0)
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)
        assert result.stabilized_depth_m[0] == 1.0  # untouched, not a blend of 1.0 and 9.0
        assert result.contradiction_count == 1
        assert result.stabilized_count == 0
        assert result.state == TemporalStabilizationState.CURRENT_ONLY_FALLBACK

    def test_current_invalid_pixel_never_filled_from_history_unknown_stays_unknown(self):
        """The UNKNOWN-never-becomes-FREE safety property: a pixel current
        does not see is never filled in from a historical value, no
        matter how confident or "free" that historical value was."""
        current = _snapshot([0.0])  # current: no measurement (UNKNOWN)
        previous = _prior(_snapshot([8.0]), confidence=1.0)  # history: far/free
        consistency = _consistency(TemporalConsistencyState.CONSISTENT, comparable_count=1, agreeing_count=1)
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)
        # eligible_count excludes this pixel entirely (current_valid is False here)
        assert result.eligible_count == 0
        assert result.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE

    def test_current_valid_previous_invalid_pixel_keeps_current_unchanged(self):
        current = _snapshot([1.0, 3.0])
        previous = _prior(_snapshot([1.0, 0.0]))  # previous invalid at index 1
        consistency = _consistency(TemporalConsistencyState.CONSISTENT, comparable_count=1, agreeing_count=1)
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)
        assert result.stabilized_depth_m[1] == 3.0  # unchanged — nothing to blend with

    def test_mixed_pixel_scene(self):
        # idx0: agrees -> blended. idx1: contradicts -> current wins.
        # idx2: current invalid -> stays invalid. idx3: previous invalid -> current wins.
        current = _snapshot([1.0, 2.0, 0.0, 4.0])
        previous = _prior(_snapshot([1.01, 9.0, 5.0, 0.0]), confidence=0.5)
        consistency = _consistency(TemporalConsistencyState.CONTRADICTORY, comparable_count=2, agreeing_count=1)
        result = compute_temporal_stabilization(current, previous, consistency, **_DEFAULT_KWARGS)

        assert result.eligible_count == 3  # idx0, idx1, idx3 (idx2 excluded, current invalid)
        assert result.stabilized_count == 1
        assert result.contradiction_count == 1
        np.testing.assert_allclose(result.stabilized_depth_m[0], (1.0 + 0.5 * 1.01) / 1.5, atol=1e-5)
        assert result.stabilized_depth_m[1] == 2.0
        assert result.stabilized_depth_m[2] == 0.0
        assert result.stabilized_depth_m[3] == 4.0


class TestPureFunction:
    def test_does_not_mutate_inputs(self):
        current = _snapshot([1.0, 2.0, 3.0])
        previous_snapshot = _snapshot([1.0, 9.0, 3.0])
        current_copy = current.copy()
        previous_copy = previous_snapshot.copy()
        consistency = _consistency(TemporalConsistencyState.CONTRADICTORY, comparable_count=3, agreeing_count=2)
        compute_temporal_stabilization(current, _prior(previous_snapshot), consistency, **_DEFAULT_KWARGS)
        np.testing.assert_array_equal(current, current_copy)
        np.testing.assert_array_equal(previous_snapshot, previous_copy)

    def test_no_for_or_while_loop_fully_vectorized(self):
        import depth_perception_engine.temporal.stabilization as stabilization_module

        tree = ast.parse(inspect.getsource(stabilization_module))
        offenders = [node.lineno for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
        assert not offenders, f"Found Python for/while loop(s) at line(s): {offenders}"


# ===================================================================
# Scope discipline (Decisions 8-9)
# ===================================================================
class TestScopeDiscipline:
    def test_stabilization_never_reads_motion_hint(self):
        import depth_perception_engine.temporal.stabilization as stabilization_module

        tree = ast.parse(inspect.getsource(stabilization_module))
        accessed_attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert "motion_hint" not in accessed_attrs

    def test_function_signature_has_no_motion_hint_parameter(self):
        sig = inspect.signature(compute_temporal_stabilization)
        assert "motion_hint" not in sig.parameters

    def test_no_forbidden_learned_or_motion_identifiers_in_module(self):
        """AST-based (code identifiers only), not a raw text search — must
        not false-positive on the module's own docstring explaining that
        these concepts are absent (same reasoning as
        tests/test_level4_architecture_guards.py)."""
        import depth_perception_engine.temporal.stabilization as stabilization_module

        tree = ast.parse(inspect.getsource(stabilization_module))
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                identifiers.update(alias.name for alias in node.names)

        forbidden = {"kalman", "neural", "torch", "tensorflow", "ego_motion", "imu", "rotation_matrix"}
        lowered = {name.lower() for name in identifiers}
        offenders = forbidden & lowered
        assert not offenders, f"Forbidden identifier(s) found in code: {offenders}"

    def test_temporal_record_was_not_extended_for_e4(self):
        """Decision 5: mathematically confirmed unnecessary — the E2/E3
        fields already carry everything the estimator needs."""
        names = {f.name for f in dataclasses.fields(TemporalRecord)}
        assert names == {"timestamp", "confidence", "geometry_quality", "motion_hint", "depth_snapshot_m"}


# ===================================================================
# Contracts
# ===================================================================
class TestTemporalStabilizationContract:
    def test_frozen_immutable(self):
        result = TemporalStabilization(
            state=TemporalStabilizationState.STABILIZED, stabilized_depth_m=_snapshot([1.0]),
            eligible_count=1, stabilized_count=1, contradiction_count=0,
            stabilized_fraction=1.0, contradiction_fraction=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.state = TemporalStabilizationState.CURRENT_ONLY_FALLBACK

    def test_field_set_is_exactly_the_frozen_e4_contract(self):
        names = {f.name for f in dataclasses.fields(TemporalStabilization)}
        assert names == {
            "state", "stabilized_depth_m", "eligible_count",
            "stabilized_count", "contradiction_count",
            "stabilized_fraction", "contradiction_fraction",
        }


class TestTemporalStabilizationStateContract:
    def test_exactly_the_three_architect_specified_states(self):
        values = {
            TemporalStabilizationState.STABILIZED,
            TemporalStabilizationState.CURRENT_ONLY_FALLBACK,
            TemporalStabilizationState.INSUFFICIENT_EVIDENCE,
        }
        assert values == {"STABILIZED", "CURRENT_ONLY_FALLBACK", "INSUFFICIENT_EVIDENCE"}


# ===================================================================
# PipelineConfig fields
# ===================================================================
class TestPipelineConfigStabilizationFields:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.enable_temporal_stabilization is False
        assert config.temporal_stabilization_min_history_confidence == 0.3
        assert config.temporal_stabilization_min_comparable_fraction == 0.1

    def test_rejects_out_of_range_min_history_confidence(self):
        with pytest.raises(ValueError):
            PipelineConfig(temporal_stabilization_min_history_confidence=1.5)
        with pytest.raises(ValueError):
            PipelineConfig(temporal_stabilization_min_history_confidence=-0.1)

    def test_rejects_out_of_range_min_comparable_fraction(self):
        with pytest.raises(ValueError):
            PipelineConfig(temporal_stabilization_min_comparable_fraction=1.5)
        with pytest.raises(ValueError):
            PipelineConfig(temporal_stabilization_min_comparable_fraction=-0.1)


# ===================================================================
# Pipeline integration — Decision 10's scenario list
# ===================================================================
def _enabled_config(**overrides):
    return PipelineConfig(enable_temporal=True, enable_temporal_stabilization=True, **overrides)


class TestPipelineDisabledByDefault:
    def test_disabled_when_enable_temporal_stabilization_false(self, calibration, stereo_pair):
        config = PipelineConfig(enable_temporal=True, enable_temporal_stabilization=False)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.temporal_stabilization is None

    def test_disabled_when_enable_temporal_also_false(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)  # both default False
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.temporal_stabilization is None

    def test_flag_alone_without_enable_temporal_has_no_effect(self, calibration, stereo_pair):
        """enable_temporal_stabilization=True with enable_temporal=False
        (unset) must not error and must not activate anything — mirrors
        enable_obstacle_geometry's own precedent when enable_geometry is
        False."""
        config = PipelineConfig(enable_temporal=False, enable_temporal_stabilization=True)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.temporal_stabilization is None
        assert pipeline.temporal_history is None


class TestScenario1FirstFrame:
    def test_first_frame_is_current_only_insufficient_evidence(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.temporal_stabilization.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE
        assert result.temporal_stabilization.stabilized_depth_m is None


class TestScenario2ConsistentFramesStabilize:
    def test_identical_consecutive_frames_produce_stabilized_state(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        result = pipeline.process(left, right, left_timestamp=1.1)

        assert result.temporal_consistency.state == TemporalConsistencyState.CONSISTENT
        assert result.temporal_stabilization.state == TemporalStabilizationState.STABILIZED
        assert result.temporal_stabilization.stabilized_fraction > 0.0
        # Identical inputs -> identical depth -> blended value equals the
        # (identical) current/previous value exactly.
        valid = result.depth_map[::pipeline.config.temporal_consistency_sampling_stride,
                                  ::pipeline.config.temporal_consistency_sampling_stride] > 0.0
        np.testing.assert_allclose(
            result.temporal_stabilization.stabilized_depth_m[valid],
            result.depth_map[::pipeline.config.temporal_consistency_sampling_stride,
                              ::pipeline.config.temporal_consistency_sampling_stride][valid],
            atol=1e-4,
        )


class TestScenario3StrongContradictionCurrentWins:
    def test_real_pipeline_current_wins_over_sharply_different_history(self, calibration, stereo_pair):
        config = _enabled_config()
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        baseline = pipeline.process(left, right, left_timestamp=1.0)

        stride = config.temporal_consistency_sampling_stride
        real_snapshot = baseline.depth_map[::stride, ::stride]
        far_off_snapshot = np.where(real_snapshot > 0.0, real_snapshot + 5.0, 0.0).astype(np.float32)
        pipeline.temporal_history.admit(
            TemporalRecord(timestamp=1.1, confidence=0.9, depth_snapshot_m=far_off_snapshot)
        )

        result = pipeline.process(left, right, left_timestamp=1.2)

        assert result.temporal_consistency.state == TemporalConsistencyState.CONTRADICTORY
        stab = result.temporal_stabilization
        assert stab.contradiction_count > 0
        current_snapshot = result.depth_map[::stride, ::stride]
        contradicted_mask = (current_snapshot > 0.0) & (far_off_snapshot > 0.0)
        np.testing.assert_array_equal(
            stab.stabilized_depth_m[contradicted_mask], current_snapshot[contradicted_mask],
        )


class TestScenario4UnknownNeverBecomesFree:
    def test_pipeline_level_unknown_stays_unknown(self, calibration):
        """Real-pipeline version of the pure-function proof above: force a
        textureless (all-invalid) current frame against a confident,
        far-range prior — no pixel may be filled in."""
        config = _enabled_config()
        pipeline = DepthPerceptionPipeline(config, calibration)
        stride = config.temporal_consistency_sampling_stride
        width, height = calibration.image_size
        flat = np.full((height, width, 3), 128, dtype=np.uint8)

        pipeline.process(flat, flat.copy(), left_timestamp=1.0)  # establishes a record (likely all-invalid too)
        real_shape = flat[::stride, ::stride, 0].shape
        fake_confident_prior = TemporalRecord(
            timestamp=1.1, confidence=1.0,
            depth_snapshot_m=np.full(real_shape, 5.0, dtype=np.float32),
        )
        pipeline.temporal_history.admit(fake_confident_prior)

        result = pipeline.process(flat, flat.copy(), left_timestamp=1.2)

        if result.temporal_stabilization.state != TemporalStabilizationState.INSUFFICIENT_EVIDENCE:
            stab = result.temporal_stabilization
            current_snapshot = result.depth_map[::stride, ::stride]
            still_invalid = current_snapshot <= 0.0
            assert np.all(stab.stabilized_depth_m[still_invalid] <= 0.0)


class TestScenario5InvalidHistoryCurrentOnly:
    def test_low_confidence_prior_falls_back_to_insufficient_evidence(self, calibration, stereo_pair):
        config = _enabled_config(temporal_stabilization_min_history_confidence=0.9)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)  # real confidence unlikely to reach 0.9

        result = pipeline.process(left, right, left_timestamp=1.1)

        # Either genuinely insufficient (confidence gate failed) or, if the
        # fixture happens to produce very high confidence, still a
        # legitimate non-error outcome — assert the gate is respected.
        if pipeline.temporal_history.records[0].confidence < 0.9:
            assert result.temporal_stabilization.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE


class TestScenario6NonComparableCurrentOnly:
    def test_shape_mismatch_not_comparable_falls_back(self, calibration, stereo_pair):
        config = _enabled_config()
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)
        # Overwrite history with a record whose snapshot has the wrong shape.
        pipeline.temporal_history.admit(
            TemporalRecord(timestamp=1.1, confidence=0.9, depth_snapshot_m=np.zeros((2, 2), dtype=np.float32))
        )

        result = pipeline.process(left, right, left_timestamp=1.2)

        assert result.temporal_consistency.state == TemporalConsistencyState.NOT_COMPARABLE
        assert result.temporal_stabilization.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE


class TestScenario7LargeGapOrResetNoStaleStabilization:
    def test_gap_triggered_new_sequence_yields_insufficient_evidence(self, calibration, stereo_pair):
        config = _enabled_config(temporal_gap_limit_s=0.5)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        result = pipeline.process(left, right, left_timestamp=100.0)  # far beyond gap limit

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE
        assert result.temporal_stabilization.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE

    def test_reset_yields_insufficient_evidence_on_next_frame(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)
        pipeline.process(left, right, left_timestamp=1.1)  # would stabilize normally
        pipeline.reset()

        result = pipeline.process(left, right, left_timestamp=5.0)

        assert result.temporal_stabilization.state == TemporalStabilizationState.INSUFFICIENT_EVIDENCE


class TestScenario8MissingMotionHintLegal:
    def test_stabilization_runs_normally_without_any_motion_hint(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)  # no motion_hint

        result = pipeline.process(left, right, left_timestamp=1.1)  # no motion_hint

        assert result.temporal_stabilization is not None
        assert result.temporal_stabilization.state == TemporalStabilizationState.STABILIZED

    def test_process_observation_without_motion_hint_also_stabilizes(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process_observation(StereoObservation(left_image=left, right_image=right, left_timestamp=1.0))

        result = pipeline.process_observation(
            StereoObservation(left_image=left, right_image=right, left_timestamp=1.1)
        )

        assert result.temporal_stabilization.state == TemporalStabilizationState.STABILIZED


class TestScenario9RawLevel3OutputUnchanged:
    def test_stabilization_never_alters_any_level3_field(self, calibration, stereo_pair):
        left, right = stereo_pair

        config_on = _enabled_config()
        pipeline_on = DepthPerceptionPipeline(config_on, calibration)
        pipeline_on.process(left, right, left_timestamp=1.0)
        result_on = pipeline_on.process(left, right, left_timestamp=1.1)
        assert result_on.temporal_stabilization.state == TemporalStabilizationState.STABILIZED

        pipeline_off = DepthPerceptionPipeline(PipelineConfig(), calibration)  # entirely disabled
        pipeline_off.process(left, right, left_timestamp=1.0)
        result_off = pipeline_off.process(left, right, left_timestamp=1.1)

        np.testing.assert_array_equal(result_on.disparity_map, result_off.disparity_map)
        np.testing.assert_array_equal(result_on.depth_map, result_off.depth_map)
        np.testing.assert_array_equal(result_on.valid_disparity_mask, result_off.valid_disparity_mask)
        np.testing.assert_array_equal(result_on.valid_depth_mask, result_off.valid_depth_mask)
        assert result_on.confidence == result_off.confidence
        assert result_on.traversability_mask.decision == result_off.traversability_mask.decision

    def test_signature_accepts_no_mutable_level3_type(self):
        sig = inspect.signature(compute_temporal_stabilization)
        param_annotations = [str(p.annotation) for p in sig.parameters.values()]
        for forbidden in ("DepthPerceptionResult", "PointCloud", "ObstacleCloud", "FreeSpaceRays"):
            assert not any(forbidden in a for a in param_annotations)


class TestScenario10Determinism:
    def test_repeated_identical_runs_produce_identical_stabilization(self, calibration, stereo_pair):
        left, right = stereo_pair

        def _run():
            pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
            pipeline.process(left, right, left_timestamp=1.0)
            return pipeline.process(left, right, left_timestamp=1.1)

        result_a = _run()
        result_b = _run()

        assert result_a.temporal_stabilization.state == result_b.temporal_stabilization.state
        assert result_a.temporal_stabilization.stabilized_fraction == result_b.temporal_stabilization.stabilized_fraction
        np.testing.assert_array_equal(
            result_a.temporal_stabilization.stabilized_depth_m,
            result_b.temporal_stabilization.stabilized_depth_m,
        )


class TestScenario11MemoryBounded:
    def test_temporal_record_gained_no_new_field(self):
        """Decision 5: confirmed mathematically unnecessary — memory cost
        of this phase is zero additional retained bytes in history."""
        names = {f.name for f in dataclasses.fields(TemporalRecord)}
        assert "stabilized_depth_m" not in names
        assert len(names) == 5  # exactly the E2/E3 fields, nothing added

    def test_stabilized_array_matches_decimated_snapshot_shape_not_full_resolution(self, calibration, stereo_pair):
        config = _enabled_config()
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        result = pipeline.process(left, right, left_timestamp=1.1)

        stride = config.temporal_consistency_sampling_stride
        expected_shape = result.depth_map[::stride, ::stride].shape
        assert result.temporal_stabilization.stabilized_depth_m.shape == expected_shape
        assert result.temporal_stabilization.stabilized_depth_m.shape != result.depth_map.shape

    def test_history_buffer_still_bounded_with_stabilization_enabled(self, calibration, stereo_pair):
        config = _enabled_config(temporal_max_records=3, temporal_max_age_s=1000.0, temporal_gap_limit_s=500.0)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair

        for i in range(10):
            pipeline.process(left, right, left_timestamp=float(i))

        assert len(pipeline.temporal_history) == 3


class TestScenario12FullRegressionSuite:
    """A placeholder marker — the actual proof is `pytest tests/ -q` as a
    whole, run separately, not a single test here."""

    def test_marker(self):
        assert True
