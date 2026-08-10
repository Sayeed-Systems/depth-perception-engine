"""
Level 4, Phase E6 tests — deterministic, motion-aware reliability
assessment.

Covers: the pure compute_motion_aware_reliability() function (every
decision-tree branch, threshold boundaries), the MotionAwareReliability/
MotionAwareReliabilityState contracts, and DepthPerceptionPipeline
integration (every scenario the architect's test list named).

No E7 algorithm is tested — E6 performs no geometry modification and no
additional motion compensation, so none of that is exercised here.
"""

import dataclasses
import inspect

import numpy as np
import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.frames import FrameId
from depth_perception_engine.pipeline import DepthPerceptionPipeline
from depth_perception_engine.temporal import (
    MotionAwareReliability,
    MotionAwareReliabilityState,
    MotionHint,
    RotationCompensationStatus,
    TemporalAdmissionStatus,
    TemporalConsistencyState,
    TemporalRecord,
    TemporalStabilizationState,
    compute_motion_aware_reliability,
)

_MAX_ANGLE = 0.0873  # ~5 degrees, PipelineConfig's own default
_MIN_COVERAGE = 0.5


def _hint(ts, omega):
    return MotionHint(timestamp=ts, angular_velocity_rad_s=np.array(omega, dtype=np.float64), frame_id=FrameId.BODY)


def _reliability(**overrides):
    kwargs = dict(
        temporal_consistency_state=None,
        temporal_stabilization_state=None,
        rotation_compensation_status=None,
        motion_hints=None,
        previous_timestamp=None,
        current_timestamp=None,
        max_angular_motion_rad=_MAX_ANGLE,
        min_motion_coverage_fraction=_MIN_COVERAGE,
    )
    kwargs.update(overrides)
    return compute_motion_aware_reliability(**kwargs)


# ===================================================================
# Decision tree — unit level
# ===================================================================
class TestNoE3Comparison:
    def test_temporal_consistency_none_is_insufficient_evidence(self):
        result = _reliability(temporal_consistency_state=None)
        assert result.state == MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE
        assert result.motion_sample_count == 0
        assert result.motion_coverage_fraction is None
        assert result.angular_motion_magnitude_rad is None


class TestE3InsufficientOrNotComparable:
    def test_insufficient_evidence_propagates(self):
        result = _reliability(temporal_consistency_state=TemporalConsistencyState.INSUFFICIENT_EVIDENCE)
        assert result.state == MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE

    def test_not_comparable_propagates(self):
        result = _reliability(temporal_consistency_state=TemporalConsistencyState.NOT_COMPARABLE)
        assert result.state == MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE


class TestPoorE3Comparability:
    def test_contradictory_is_unreliable_regardless_of_motion(self):
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONTRADICTORY,
            rotation_compensation_status=RotationCompensationStatus.APPLIED,
        )
        assert result.state == MotionAwareReliabilityState.UNRELIABLE


class TestE5NotEnabled:
    def test_consistent_with_no_rotation_compensation_status_is_reliable(self):
        result = _reliability(temporal_consistency_state=TemporalConsistencyState.CONSISTENT, rotation_compensation_status=None)
        assert result.state == MotionAwareReliabilityState.RELIABLE


class TestCompensationUnavailable:
    def test_not_applied_is_degraded(self):
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            rotation_compensation_status=RotationCompensationStatus.NOT_APPLIED,
        )
        assert result.state == MotionAwareReliabilityState.DEGRADED


class TestStationaryOrSmallRotation:
    def test_near_zero_rotation_is_reliable(self):
        hints = [_hint(0.9, [0.0, 0.0001, 0.0])]
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            rotation_compensation_status=RotationCompensationStatus.APPLIED,
            motion_hints=hints, previous_timestamp=0.0, current_timestamp=1.0,
        )
        assert result.state == MotionAwareReliabilityState.RELIABLE
        assert result.angular_motion_magnitude_rad < 0.001

    def test_valid_small_rotation_within_threshold_is_reliable(self):
        hints = [_hint(0.9, [0.0, 0.05, 0.0])]  # well under 5 degrees
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            rotation_compensation_status=RotationCompensationStatus.APPLIED,
            motion_hints=hints, previous_timestamp=0.0, current_timestamp=1.0,
        )
        assert result.state == MotionAwareReliabilityState.RELIABLE
        assert result.angular_motion_magnitude_rad < _MAX_ANGLE


class TestIncreasingAngularMotion:
    def test_state_transitions_reliable_to_unreliable_as_angle_grows(self):
        omegas = [0.01, 0.05, 0.09, 0.5, 1.5]  # radians, at dt=0.9 (near full interval)
        states = []
        for omega in omegas:
            hints = [_hint(0.9, [0.0, omega, 0.0])]
            result = _reliability(
                temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
                rotation_compensation_status=RotationCompensationStatus.APPLIED,
                motion_hints=hints, previous_timestamp=0.0, current_timestamp=1.0,
            )
            states.append(result.state)
        assert states[0] == MotionAwareReliabilityState.RELIABLE
        assert states[-1] == MotionAwareReliabilityState.UNRELIABLE
        # Monotonic: once UNRELIABLE is reached, it never reverts to RELIABLE for a larger angle.
        first_unreliable = states.index(MotionAwareReliabilityState.UNRELIABLE)
        assert all(s == MotionAwareReliabilityState.UNRELIABLE for s in states[first_unreliable:])

    def test_boundary_is_strict_greater_than_not_greater_or_equal(self):
        """The comparison in _classify() is `> max_angular_motion_rad`
        (breach), not `>=` — verified directly against the classifier,
        avoiding any floating-point round-trip fragility from deriving
        the angle through cv2.Rodrigues/arccos (a value constructed to
        equal the threshold exactly via that path can legitimately land
        a few ULPs to either side)."""
        from depth_perception_engine.temporal.reliability import _classify

        state_at_threshold = _classify(
            TemporalConsistencyState.CONSISTENT, RotationCompensationStatus.APPLIED,
            angular_motion_magnitude_rad=_MAX_ANGLE, motion_coverage_fraction=1.0,
            max_angular_motion_rad=_MAX_ANGLE, min_motion_coverage_fraction=_MIN_COVERAGE,
        )
        state_just_above = _classify(
            TemporalConsistencyState.CONSISTENT, RotationCompensationStatus.APPLIED,
            angular_motion_magnitude_rad=_MAX_ANGLE + 1e-9, motion_coverage_fraction=1.0,
            max_angular_motion_rad=_MAX_ANGLE, min_motion_coverage_fraction=_MIN_COVERAGE,
        )
        assert state_at_threshold == MotionAwareReliabilityState.RELIABLE
        assert state_just_above == MotionAwareReliabilityState.UNRELIABLE


class TestIncompleteMotionHintCoverage:
    def test_low_coverage_is_degraded(self):
        hints = [_hint(0.2, [0.0, 0.01, 0.0])]  # covers only 0.0-0.2 of a (0,1] interval
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            rotation_compensation_status=RotationCompensationStatus.APPLIED,
            motion_hints=hints, previous_timestamp=0.0, current_timestamp=1.0,
        )
        assert result.motion_coverage_fraction == pytest.approx(0.2)
        assert result.state == MotionAwareReliabilityState.DEGRADED

    def test_full_coverage_is_reliable(self):
        hints = [_hint(1.0, [0.0, 0.01, 0.0])]  # covers the entire interval
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            rotation_compensation_status=RotationCompensationStatus.APPLIED,
            motion_hints=hints, previous_timestamp=0.0, current_timestamp=1.0,
        )
        assert result.motion_coverage_fraction == pytest.approx(1.0)
        assert result.state == MotionAwareReliabilityState.RELIABLE


class TestInvalidOrStaleMotion:
    def test_invalid_hint_excluded_yields_zero_samples(self):
        hints = [MotionHint(timestamp=0.5, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY, valid=False)]
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            rotation_compensation_status=RotationCompensationStatus.NOT_APPLIED,
            motion_hints=hints, previous_timestamp=0.0, current_timestamp=1.0,
        )
        assert result.motion_sample_count == 0
        assert result.state == MotionAwareReliabilityState.DEGRADED

    def test_stale_hint_outside_interval_excluded(self):
        hints = [_hint(-0.5, [0.0, 0.01, 0.0])]  # before previous_timestamp
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            rotation_compensation_status=RotationCompensationStatus.NOT_APPLIED,
            motion_hints=hints, previous_timestamp=0.0, current_timestamp=1.0,
        )
        assert result.motion_sample_count == 0


class TestE4Unavailable:
    def test_temporal_stabilization_state_none_does_not_block_classification(self):
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            temporal_stabilization_state=None,
            rotation_compensation_status=RotationCompensationStatus.APPLIED,
            motion_hints=[_hint(0.9, [0.0, 0.01, 0.0])], previous_timestamp=0.0, current_timestamp=1.0,
        )
        assert result.state == MotionAwareReliabilityState.RELIABLE
        assert result.temporal_stabilization_state is None

    def test_temporal_stabilization_state_reported_verbatim_when_present(self):
        result = _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            temporal_stabilization_state=TemporalStabilizationState.CURRENT_ONLY_FALLBACK,
        )
        assert result.temporal_stabilization_state == TemporalStabilizationState.CURRENT_ONLY_FALLBACK


class TestPureFunction:
    def test_never_raises_on_degenerate_inputs(self):
        _reliability()  # everything None/default
        _reliability(motion_hints=[], previous_timestamp=1.0, current_timestamp=1.0)  # zero-length interval
        _reliability(previous_timestamp=float("nan"), current_timestamp=1.0)

    def test_does_not_mutate_motion_hints_list(self):
        hints = [_hint(0.9, [0.0, 0.01, 0.0])]
        hints_copy = list(hints)
        _reliability(
            temporal_consistency_state=TemporalConsistencyState.CONSISTENT,
            rotation_compensation_status=RotationCompensationStatus.APPLIED,
            motion_hints=hints, previous_timestamp=0.0, current_timestamp=1.0,
        )
        assert hints == hints_copy

    def test_signature_accepts_no_mutable_upstream_type(self):
        sig = inspect.signature(compute_motion_aware_reliability)
        param_annotations = [str(p.annotation) for p in sig.parameters.values()]
        forbidden = ("DepthPerceptionResult", "PointCloud", "ObstacleCloud", "FreeSpaceRays", "TemporalConsistency", "TemporalStabilization")
        for name in forbidden:
            assert not any(name in a for a in param_annotations)

    def test_never_calls_compensate_prior_geometry(self):
        import ast

        import depth_perception_engine.temporal.reliability as module

        tree = ast.parse(inspect.getsource(module))
        called_names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert "compensate_prior_geometry" not in called_names


# ===================================================================
# Contracts
# ===================================================================
class TestMotionAwareReliabilityContract:
    def test_frozen_immutable(self):
        result = _reliability()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.state = MotionAwareReliabilityState.RELIABLE

    def test_field_set_is_exactly_the_frozen_e6_contract(self):
        names = {f.name for f in dataclasses.fields(MotionAwareReliability)}
        assert names == {
            "state", "motion_sample_count", "motion_coverage_fraction",
            "rotation_compensation_status", "angular_motion_magnitude_rad",
            "temporal_consistency_state", "temporal_stabilization_state",
        }


class TestMotionAwareReliabilityStateContract:
    def test_exactly_the_four_architect_specified_states(self):
        values = {
            MotionAwareReliabilityState.RELIABLE,
            MotionAwareReliabilityState.DEGRADED,
            MotionAwareReliabilityState.UNRELIABLE,
            MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE,
        }
        assert values == {"RELIABLE", "DEGRADED", "UNRELIABLE", "INSUFFICIENT_EVIDENCE"}


class TestPipelineConfigFields:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.enable_motion_aware_reliability is False
        assert config.reliability_max_angular_motion_rad == pytest.approx(0.0873)
        assert config.reliability_min_motion_coverage_fraction == 0.5

    def test_rejects_non_positive_max_angular_motion(self):
        with pytest.raises(ValueError):
            PipelineConfig(reliability_max_angular_motion_rad=0.0)
        with pytest.raises(ValueError):
            PipelineConfig(reliability_max_angular_motion_rad=-0.1)

    def test_rejects_out_of_range_min_coverage(self):
        with pytest.raises(ValueError):
            PipelineConfig(reliability_min_motion_coverage_fraction=1.5)
        with pytest.raises(ValueError):
            PipelineConfig(reliability_min_motion_coverage_fraction=-0.1)


# ===================================================================
# Pipeline integration
# ===================================================================
def _enabled_config(**overrides):
    return PipelineConfig(enable_temporal=True, enable_motion_aware_reliability=True, **overrides)


class TestPipelineDisabledByDefault:
    def test_disabled_by_default(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.motion_aware_reliability is None

    def test_flag_alone_without_enable_temporal_has_no_effect(self, calibration, stereo_pair):
        config = PipelineConfig(enable_temporal=False, enable_motion_aware_reliability=True)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.motion_aware_reliability is None


class TestFirstFrameAndChronologyRejection:
    def test_first_frame_is_insufficient_evidence(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.motion_aware_reliability.state == MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE

    def test_rejected_frame_still_classified_insufficient_evidence(self, calibration, stereo_pair):
        """The key architectural claim: E6 runs even when E3 didn't."""
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=5.0)

        result = pipeline.process(left, right, left_timestamp=1.0)  # older -> rejected

        assert result.temporal_admission_status == TemporalAdmissionStatus.REJECTED_OLDER_TIMESTAMP
        assert result.temporal_consistency is None
        assert result.motion_aware_reliability is not None
        assert result.motion_aware_reliability.state == MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE


class TestPoorE3ComparabilityThroughRealPipeline:
    def test_contradictory_prior_yields_unreliable(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        baseline = pipeline.process(left, right, left_timestamp=1.0)

        stride = pipeline.config.temporal_consistency_sampling_stride
        real_snapshot = baseline.depth_map[::stride, ::stride]
        far_off = np.where(real_snapshot > 0.0, real_snapshot + 5.0, 0.0).astype(np.float32)
        pipeline.temporal_history.admit(TemporalRecord(timestamp=1.05, confidence=0.9, depth_snapshot_m=far_off))

        result = pipeline.process(left, right, left_timestamp=1.1)

        assert result.temporal_consistency.state == TemporalConsistencyState.CONTRADICTORY
        assert result.motion_aware_reliability.state == MotionAwareReliabilityState.UNRELIABLE


class TestCompensationUnavailableThroughRealPipeline:
    def test_no_motion_hints_supplied_is_degraded_when_e5_enabled(self, calibration, stereo_pair):
        config = _enabled_config(enable_rotation_compensation=True)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        result = pipeline.process(left, right, left_timestamp=1.1)  # no motion_hints

        assert result.rotation_compensation_status == RotationCompensationStatus.NOT_APPLIED
        assert result.motion_aware_reliability.state == MotionAwareReliabilityState.DEGRADED

    def test_e5_disabled_entirely_is_reliable_on_consistent_frames(self, calibration, stereo_pair):
        config = _enabled_config(enable_rotation_compensation=False)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        result = pipeline.process(left, right, left_timestamp=1.1)

        assert result.rotation_compensation_status is None
        assert result.motion_aware_reliability.rotation_compensation_status is None
        assert result.motion_aware_reliability.state == MotionAwareReliabilityState.RELIABLE


class TestResetAndGap:
    def test_reset_yields_insufficient_evidence_on_next_frame(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)
        pipeline.process(left, right, left_timestamp=1.1)
        pipeline.reset()

        result = pipeline.process(left, right, left_timestamp=5.0)

        assert result.motion_aware_reliability.state == MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE

    def test_gap_restart_yields_insufficient_evidence(self, calibration, stereo_pair):
        config = _enabled_config(temporal_gap_limit_s=0.5)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        result = pipeline.process(left, right, left_timestamp=100.0)

        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE
        assert result.motion_aware_reliability.state == MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE


class TestDeterminism:
    def test_repeated_identical_runs_produce_identical_reliability(self, calibration, stereo_pair):
        left, right = stereo_pair

        def _run():
            pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
            pipeline.process(left, right, left_timestamp=1.0)
            return pipeline.process(left, right, left_timestamp=1.1)

        result_a = _run()
        result_b = _run()

        assert result_a.motion_aware_reliability.state == result_b.motion_aware_reliability.state
        assert result_a.motion_aware_reliability.motion_sample_count == result_b.motion_aware_reliability.motion_sample_count


class TestRawAndUpstreamOutputsUnchanged:
    def test_enabling_e6_never_alters_raw_or_e3_e4_e5_outputs(self, calibration, stereo_pair):
        left, right = stereo_pair

        config_e6_on = PipelineConfig(
            enable_temporal=True, enable_temporal_stabilization=True,
            enable_rotation_compensation=True, enable_motion_aware_reliability=True,
        )
        pipeline_on = DepthPerceptionPipeline(config_e6_on, calibration)
        pipeline_on.process(left, right, left_timestamp=1.0)
        hint = _hint(1.05, [0.0, 0.01, 0.0])
        result_on = pipeline_on.process(left, right, left_timestamp=1.1, motion_hints=[hint])

        config_e6_off = PipelineConfig(
            enable_temporal=True, enable_temporal_stabilization=True,
            enable_rotation_compensation=True, enable_motion_aware_reliability=False,
        )
        pipeline_off = DepthPerceptionPipeline(config_e6_off, calibration)
        pipeline_off.process(left, right, left_timestamp=1.0)
        result_off = pipeline_off.process(left, right, left_timestamp=1.1, motion_hints=[hint])

        assert result_on.motion_aware_reliability is not None
        assert result_off.motion_aware_reliability is None

        np.testing.assert_array_equal(result_on.disparity_map, result_off.disparity_map)
        np.testing.assert_array_equal(result_on.depth_map, result_off.depth_map)
        assert result_on.confidence == result_off.confidence
        assert result_on.temporal_consistency.state == result_off.temporal_consistency.state
        assert result_on.temporal_consistency.agreement_fraction == result_off.temporal_consistency.agreement_fraction
        assert result_on.temporal_stabilization.state == result_off.temporal_stabilization.state
        assert result_on.rotation_compensation_status == result_off.rotation_compensation_status


class TestNoPlatformAssumptions:
    def test_no_forbidden_platform_terms_in_reliability_module(self):
        import ast

        import depth_perception_engine.temporal.reliability as module

        tree = ast.parse(inspect.getsource(module))
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
        forbidden = {"aerial", "drone", "ground", "marine", "rover", "boat"}
        offenders = forbidden & {name.lower() for name in identifiers}
        assert not offenders
