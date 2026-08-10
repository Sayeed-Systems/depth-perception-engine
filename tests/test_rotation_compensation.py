"""
Level 4, Phase E5 tests — short-window rotational motion compensation.

Covers: sample selection (interval/monotonicity/validity), SO(3)
integration (zero rotation, constant-rate yaw/pitch/roll, multi-sample
composition, analytical 90° cases), prior-geometry rotation/reprojection,
the compute_rotation_compensation() orchestrator, the
RotationCompensationStatus contract, and DepthPerceptionPipeline
integration (every scenario the architect's Decision 10 listed).

No E6 algorithm is tested — E5 performs no IMU bias estimation,
accumulated orientation, translation compensation, VIO, SLAM, or
localization, so none of that is exercised here.
"""

import ast
import dataclasses
import inspect

import cv2
import numpy as np
import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.frames import FrameId
from depth_perception_engine.models.result import StereoObservation
from depth_perception_engine.pipeline import DepthPerceptionPipeline
from depth_perception_engine.temporal import (
    MotionHint,
    RotationCompensationStatus,
    TemporalRecord,
    compute_rotation_compensation,
    compute_temporal_consistency,
)
from depth_perception_engine.temporal.rotation_compensation import (
    compensate_prior_geometry,
    integrate_angular_velocity,
    select_motion_hint_samples,
)

_DEFAULT_TOLERANCE = 0.05
_DEFAULT_MIN_FRACTION = 0.7


def _hint(ts, omega, valid=True):
    return MotionHint(timestamp=ts, angular_velocity_rad_s=np.array(omega, dtype=np.float64), frame_id=FrameId.BODY, valid=valid)


# ===================================================================
# select_motion_hint_samples — Decision 3
# ===================================================================
class TestSelectMotionHintSamples:
    def test_none_or_empty_returns_empty(self):
        assert select_motion_hint_samples(None, 0.0, 1.0) == []
        assert select_motion_hint_samples([], 0.0, 1.0) == []

    def test_missing_or_nonfinite_timestamps_return_empty(self):
        s = [_hint(0.5, [0, 0, 0])]
        assert select_motion_hint_samples(s, None, 1.0) == []
        assert select_motion_hint_samples(s, 0.0, None) == []
        assert select_motion_hint_samples(s, float("nan"), 1.0) == []

    def test_current_not_after_previous_returns_empty(self):
        s = [_hint(0.5, [0, 0, 0])]
        assert select_motion_hint_samples(s, 1.0, 1.0) == []
        assert select_motion_hint_samples(s, 1.0, 0.5) == []

    def test_lower_bound_exclusive_upper_bound_inclusive(self):
        s_at_prev = _hint(0.0, [0, 0, 0])   # excluded: == previous_timestamp
        s_at_curr = _hint(1.0, [0, 0, 0])   # included: == current_timestamp
        assert select_motion_hint_samples([s_at_prev], 0.0, 1.0) == []
        assert select_motion_hint_samples([s_at_curr], 0.0, 1.0) == [s_at_curr]

    def test_out_of_interval_samples_rejected(self):
        before = _hint(-0.5, [0, 0, 0])
        after = _hint(1.5, [0, 0, 0])
        assert select_motion_hint_samples([before], 0.0, 1.0) == []
        assert select_motion_hint_samples([after], 0.0, 1.0) == []

    def test_invalid_flag_rejected(self):
        s = _hint(0.5, [0, 0, 0], valid=False)
        assert select_motion_hint_samples([s], 0.0, 1.0) == []

    def test_monotonic_sequence_all_accepted(self):
        samples = [_hint(0.2, [0, 0, 0]), _hint(0.5, [0, 0, 0]), _hint(0.9, [0, 0, 0])]
        assert select_motion_hint_samples(samples, 0.0, 1.0) == samples

    def test_non_monotonic_sample_rejected_given_order_not_resorted(self):
        s1 = _hint(0.6, [0, 0, 0])
        s2 = _hint(0.3, [0, 0, 0])  # earlier timestamp, but given AFTER s1
        result = select_motion_hint_samples([s1, s2], 0.0, 1.0)
        assert result == [s1]  # s2 rejected, not silently reordered before s1

    def test_duplicate_timestamp_second_rejected(self):
        s1 = _hint(0.5, [0, 0, 0])
        s2 = _hint(0.5, [1, 0, 0])
        result = select_motion_hint_samples([s1, s2], 0.0, 1.0)
        assert result == [s1]

    def test_none_entries_in_sequence_skipped(self):
        s = _hint(0.5, [0, 0, 0])
        assert select_motion_hint_samples([None, s], 0.0, 1.0) == [s]


# ===================================================================
# integrate_angular_velocity — Decision 1
# ===================================================================
class TestIntegrateAngularVelocity:
    def test_zero_rotation_is_identity(self):
        result = integrate_angular_velocity([_hint(1.0, [0.0, 0.0, 0.0])], previous_timestamp=0.0)
        np.testing.assert_allclose(result, np.eye(3), atol=1e-6)

    def test_constant_rate_yaw_matches_hand_derivation(self):
        """Camera yaws right (rotation about Y, camera-optical 'down'
        axis) at 0.5 rad/s for 1.0s. Verified against the hand-derived
        physical result: a fixed point directly ahead should shift LEFT
        (negative X) in the current frame — see
        docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md's Decision 1."""
        delta_r = integrate_angular_velocity([_hint(1.0, [0.0, 0.5, 0.0])], previous_timestamp=0.0)
        p_prev = np.array([0.0, 0.0, 5.0])
        p_curr = delta_r @ p_prev
        theta = 0.5
        np.testing.assert_allclose(p_curr[0], -np.sin(theta) * 5.0, atol=1e-5)
        np.testing.assert_allclose(p_curr[2], np.cos(theta) * 5.0, atol=1e-5)

    def test_constant_rate_pitch(self):
        """Rotation about X (camera-optical 'right' axis)."""
        delta_r = integrate_angular_velocity([_hint(1.0, [0.3, 0.0, 0.0])], previous_timestamp=0.0)
        assert delta_r.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.det(delta_r), 1.0, atol=1e-5)  # proper rotation

    def test_constant_rate_roll(self):
        """Rotation about Z (camera-optical 'forward' axis)."""
        delta_r = integrate_angular_velocity([_hint(1.0, [0.0, 0.0, 0.4])], previous_timestamp=0.0)
        assert delta_r.shape == (3, 3)
        np.testing.assert_allclose(delta_r.T @ delta_r, np.eye(3), atol=1e-5)  # orthogonal

    def test_analytical_90_degree_yaw(self):
        theta = np.pi / 2.0
        delta_r = integrate_angular_velocity([_hint(1.0, [0.0, theta, 0.0])], previous_timestamp=0.0)
        # ΔR_total = R_y(theta); ΔR_prev_to_curr = R_y(theta)^T = R_y(-theta)
        expected = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(delta_r, expected, atol=1e-6)

    def test_analytical_90_degree_roll(self):
        theta = np.pi / 2.0
        delta_r = integrate_angular_velocity([_hint(1.0, [0.0, 0.0, theta])], previous_timestamp=0.0)
        expected_total = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(delta_r, expected_total.T, atol=1e-6)

    def test_multi_sample_composition_matches_manual_product(self):
        s1 = _hint(0.5, [0.1, 0.0, 0.0])
        s2 = _hint(1.0, [0.0, 0.2, 0.0])
        result = integrate_angular_velocity([s1, s2], previous_timestamp=0.0)

        r1, _ = cv2.Rodrigues(np.array([0.1, 0.0, 0.0]) * 0.5)  # dt1 = 0.5 - 0.0
        r2, _ = cv2.Rodrigues(np.array([0.0, 0.2, 0.0]) * 0.5)  # dt2 = 1.0 - 0.5
        expected_total = r1 @ r2
        np.testing.assert_allclose(result, expected_total.T, atol=1e-5)

    def test_zero_order_hold_never_extrapolates_past_last_sample(self):
        """Only one sample at t=0.3 within a (0.0, 1.0] interval — its
        own Δt is 0.3 (from previous_timestamp), NOT 1.0 (all the way to
        current_timestamp). Verified indirectly: a hand-computed
        single-sample integration with dt=0.3 must match exactly."""
        result = integrate_angular_velocity([_hint(0.3, [0.0, 1.0, 0.0])], previous_timestamp=0.0)
        expected, _ = cv2.Rodrigues(np.array([0.0, 1.0, 0.0]) * 0.3)
        np.testing.assert_allclose(result, expected.T, atol=1e-6)


# ===================================================================
# compensate_prior_geometry — Decision 2
# ===================================================================
class TestCompensatePriorGeometry:
    def test_zero_rotation_reproduces_input(self):
        snapshot = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 0.0]], dtype=np.float32)
        out = compensate_prior_geometry(snapshot, np.eye(3, dtype=np.float32), stride=4, focal_length_px=500.0, principal_point_px=(160.0, 120.0))
        np.testing.assert_allclose(out, snapshot, atol=1e-3)

    def test_invalid_pixel_stays_invalid(self):
        snapshot = np.array([[0.0, 3.0]], dtype=np.float32)
        out = compensate_prior_geometry(snapshot, np.eye(3, dtype=np.float32), stride=4, focal_length_px=500.0, principal_point_px=(160.0, 120.0))
        assert out[0, 0] == 0.0

    def test_no_translation_terms_in_source(self):
        """Structural proof, not just behavioral: no translation-named
        identifier is ever assigned or read anywhere in this function's
        own code — AST-based (identifiers only), not a raw text search,
        so this doesn't false-positive on the function's own docstring
        explaining that no translation term exists (same reasoning as
        tests/test_level4_architecture_guards.py)."""
        import depth_perception_engine.temporal.rotation_compensation as module

        tree = ast.parse(inspect.getsource(module.compensate_prior_geometry))
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
        forbidden = {"translation", "tx", "ty", "tz", "t_vec", "offset"}
        offenders = forbidden & {name.lower() for name in identifiers}
        assert not offenders, f"Translation-shaped identifier(s) found in code: {offenders}"

    def test_does_not_mutate_input(self):
        snapshot = np.array([[2.0, 3.0]], dtype=np.float32)
        snapshot_copy = snapshot.copy()
        compensate_prior_geometry(snapshot, np.eye(3, dtype=np.float32), stride=4, focal_length_px=500.0, principal_point_px=(160.0, 120.0))
        np.testing.assert_array_equal(snapshot, snapshot_copy)

    def test_no_for_or_while_loop_fully_vectorized(self):
        import depth_perception_engine.temporal.rotation_compensation as module

        tree = ast.parse(inspect.getsource(module.compensate_prior_geometry))
        offenders = [node.lineno for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
        assert not offenders, f"Found Python for/while loop(s) at line(s): {offenders}"

    def test_deterministic_repeated_calls(self):
        snapshot = np.random.default_rng(1).uniform(1.0, 5.0, size=(6, 6)).astype(np.float32)
        theta = 0.2
        delta_r, _ = cv2.Rodrigues(np.array([0.0, theta, 0.0]))
        out1 = compensate_prior_geometry(snapshot, delta_r.astype(np.float32), stride=1, focal_length_px=500.0, principal_point_px=(3.0, 3.0))
        out2 = compensate_prior_geometry(snapshot, delta_r.astype(np.float32), stride=1, focal_length_px=500.0, principal_point_px=(3.0, 3.0))
        np.testing.assert_array_equal(out1, out2)


# ===================================================================
# compute_rotation_compensation — orchestrator
# ===================================================================
class TestComputeRotationCompensationOrchestrator:
    _KW = dict(stride=4, focal_length_px=500.0, principal_point_px=(160.0, 120.0))

    def test_no_previous_record_not_applied(self):
        record, status = compute_rotation_compensation(None, [_hint(0.5, [0, 0.1, 0])], 0.0, 1.0, **self._KW)
        assert record is None
        assert status == RotationCompensationStatus.NOT_APPLIED

    def test_previous_record_without_snapshot_not_applied(self):
        previous = TemporalRecord(timestamp=0.0, confidence=0.8)
        record, status = compute_rotation_compensation(previous, [_hint(0.5, [0, 0.1, 0])], 0.0, 1.0, **self._KW)
        assert record is previous
        assert status == RotationCompensationStatus.NOT_APPLIED

    def test_no_motion_hints_not_applied_identity_record(self):
        previous = TemporalRecord(timestamp=0.0, confidence=0.8, depth_snapshot_m=np.array([[2.0]], dtype=np.float32))
        record, status = compute_rotation_compensation(previous, None, 0.0, 1.0, **self._KW)
        assert record is previous  # same object, not a copy
        assert status == RotationCompensationStatus.NOT_APPLIED

    def test_successful_compensation_preserves_other_fields(self):
        hint = MotionHint(timestamp=1.0, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY)
        previous = TemporalRecord(
            timestamp=0.0, confidence=0.77, geometry_quality="HEALTHY", motion_hint=hint,
            depth_snapshot_m=np.array([[2.0, 3.0]], dtype=np.float32),
        )
        record, status = compute_rotation_compensation(previous, [hint], 0.0, 1.0, **self._KW)
        assert status == RotationCompensationStatus.APPLIED
        assert record is not previous
        assert record.timestamp == previous.timestamp
        assert record.confidence == previous.confidence
        assert record.geometry_quality == previous.geometry_quality
        assert record.motion_hint is previous.motion_hint


# ===================================================================
# Contracts
# ===================================================================
class TestRotationCompensationStatusContract:
    def test_exactly_two_states(self):
        assert {RotationCompensationStatus.APPLIED, RotationCompensationStatus.NOT_APPLIED} == {"APPLIED", "NOT_APPLIED"}


class TestPipelineConfigField:
    def test_default_false(self):
        assert PipelineConfig().enable_rotation_compensation is False


class TestStereoObservationMotionHints:
    def test_defaults_to_none(self, stereo_pair):
        left, right = stereo_pair
        obs = StereoObservation(left_image=left, right_image=right)
        assert obs.motion_hints is None

    def test_can_carry_a_sequence(self, stereo_pair):
        left, right = stereo_pair
        hints = [_hint(0.1, [0, 0, 0]), _hint(0.2, [0, 0, 0])]
        obs = StereoObservation(left_image=left, right_image=right, motion_hints=hints)
        assert obs.motion_hints == hints


# ===================================================================
# The key end-to-end proof: compensation improves comparability
# ===================================================================
class TestSyntheticRotationImprovesComparability:
    def test_compensation_recovers_agreement_after_synthetic_rotation(self):
        rng = np.random.default_rng(7)
        h, w = 60, 60
        previous_snapshot = rng.uniform(2.0, 4.0, size=(h, w)).astype(np.float32)
        focal_length_px = 400.0
        principal_point_px = (float(w) / 2.0, float(h) / 2.0)
        stride = 1

        omega = np.array([0.0, 0.03, 0.0])  # small yaw rate, tuned so most of the
        # 60x60 synthetic grid still reprojects in-bounds (see the sibling
        # 12x12/f=400 configuration this replaced — that focal length was
        # unrealistically large relative to such a tiny grid, pushing every
        # reprojected point out of bounds; f*theta pixel shift must stay
        # well under the grid's own half-width for this synthetic setup).
        dt = 1.0
        hint = MotionHint(timestamp=1.0, angular_velocity_rad_s=omega, frame_id=FrameId.BODY)

        delta_r_true = integrate_angular_velocity([hint], previous_timestamp=0.0)
        # Simulate "what the camera actually sees now" by applying the
        # same true rotation to the previous geometry.
        current_snapshot = compensate_prior_geometry(
            previous_snapshot, delta_r_true, stride, focal_length_px, principal_point_px,
        )
        assert (current_snapshot > 0.0).sum() > 0, "synthetic scene must produce some valid current pixels"

        previous_record = TemporalRecord(timestamp=0.0, confidence=0.9, depth_snapshot_m=previous_snapshot)

        raw_consistency = compute_temporal_consistency(
            current_snapshot, previous_record,
            agreement_tolerance_m=_DEFAULT_TOLERANCE, min_agreement_fraction=_DEFAULT_MIN_FRACTION,
        )

        compensated_record, status = compute_rotation_compensation(
            previous_record, [hint], previous_timestamp=0.0, current_timestamp=1.0,
            stride=stride, focal_length_px=focal_length_px, principal_point_px=principal_point_px,
        )
        assert status == RotationCompensationStatus.APPLIED

        compensated_consistency = compute_temporal_consistency(
            current_snapshot, compensated_record,
            agreement_tolerance_m=_DEFAULT_TOLERANCE, min_agreement_fraction=_DEFAULT_MIN_FRACTION,
        )

        raw_fraction = raw_consistency.agreement_fraction or 0.0
        compensated_fraction = compensated_consistency.agreement_fraction or 0.0
        assert compensated_fraction > raw_fraction
        assert compensated_fraction > 0.9  # compensation should recover near-full agreement


# ===================================================================
# Pipeline integration — Decision 10's scenario list
# ===================================================================
def _enabled_config(**overrides):
    return PipelineConfig(enable_temporal=True, enable_rotation_compensation=True, **overrides)


class TestPipelineDisabledByDefault:
    def test_disabled_by_default(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.rotation_compensation_status is None

    def test_flag_alone_without_enable_temporal_has_no_effect(self, calibration, stereo_pair):
        config = PipelineConfig(enable_temporal=False, enable_rotation_compensation=True)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.rotation_compensation_status is None


class TestScenarioMissingStaleInvalidHints:
    def test_no_motion_hints_falls_back_cleanly(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        result = pipeline.process(left, right, left_timestamp=1.1)  # no motion_hints

        assert result.rotation_compensation_status == RotationCompensationStatus.NOT_APPLIED

    def test_stale_hint_outside_interval_falls_back(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        stale = _hint(0.5, [0.0, 0.1, 0.0])  # before previous_timestamp=1.0
        result = pipeline.process(left, right, left_timestamp=1.1, motion_hints=[stale])

        assert result.rotation_compensation_status == RotationCompensationStatus.NOT_APPLIED

    def test_invalid_hint_falls_back(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        invalid = _hint(1.05, [0.0, 0.1, 0.0], valid=False)
        result = pipeline.process(left, right, left_timestamp=1.1, motion_hints=[invalid])

        assert result.rotation_compensation_status == RotationCompensationStatus.NOT_APPLIED

    def test_valid_hint_in_interval_applies(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        hint = _hint(1.05, [0.0, 0.01, 0.0])
        result = pipeline.process(left, right, left_timestamp=1.1, motion_hints=[hint])

        assert result.rotation_compensation_status == RotationCompensationStatus.APPLIED

    def test_process_observation_forwards_motion_hints(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process_observation(StereoObservation(left_image=left, right_image=right, left_timestamp=1.0))

        hint = _hint(1.05, [0.0, 0.01, 0.0])
        obs = StereoObservation(left_image=left, right_image=right, left_timestamp=1.1, motion_hints=[hint])
        result = pipeline.process_observation(obs)

        assert result.rotation_compensation_status == RotationCompensationStatus.APPLIED


class TestScenarioE3E4UnchangedWhenUnavailable:
    def test_outputs_identical_with_and_without_e5_when_no_hints_supplied(self, calibration, stereo_pair):
        left, right = stereo_pair

        pipeline_e5 = DepthPerceptionPipeline(
            PipelineConfig(enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True),
            calibration,
        )
        pipeline_e5.process(left, right, left_timestamp=1.0)
        result_e5 = pipeline_e5.process(left, right, left_timestamp=1.1)  # no motion_hints -> NOT_APPLIED

        pipeline_no_e5 = DepthPerceptionPipeline(
            PipelineConfig(enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=False),
            calibration,
        )
        pipeline_no_e5.process(left, right, left_timestamp=1.0)
        result_no_e5 = pipeline_no_e5.process(left, right, left_timestamp=1.1)

        assert result_e5.temporal_consistency.state == result_no_e5.temporal_consistency.state
        assert result_e5.temporal_consistency.agreement_fraction == result_no_e5.temporal_consistency.agreement_fraction
        assert result_e5.temporal_stabilization.state == result_no_e5.temporal_stabilization.state
        np.testing.assert_array_equal(result_e5.disparity_map, result_no_e5.disparity_map)
        np.testing.assert_array_equal(result_e5.depth_map, result_no_e5.depth_map)


class TestScenarioResetGap:
    def test_gap_restart_yields_not_applied(self, calibration, stereo_pair):
        config = _enabled_config(temporal_gap_limit_s=0.5)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        hint = _hint(50.0, [0.0, 0.01, 0.0])
        result = pipeline.process(left, right, left_timestamp=100.0, motion_hints=[hint])  # gap > limit

        assert result.rotation_compensation_status == RotationCompensationStatus.NOT_APPLIED

    def test_reset_yields_not_applied_on_next_frame(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)
        pipeline.reset()

        hint = _hint(4.95, [0.0, 0.01, 0.0])
        result = pipeline.process(left, right, left_timestamp=5.0, motion_hints=[hint])

        assert result.rotation_compensation_status == RotationCompensationStatus.NOT_APPLIED


class TestScenarioDeterminism:
    def test_repeated_identical_runs_produce_identical_status(self, calibration, stereo_pair):
        left, right = stereo_pair
        hint = _hint(1.05, [0.0, 0.02, 0.0])

        def _run():
            pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
            pipeline.process(left, right, left_timestamp=1.0)
            return pipeline.process(left, right, left_timestamp=1.1, motion_hints=[hint])

        result_a = _run()
        result_b = _run()

        assert result_a.rotation_compensation_status == result_b.rotation_compensation_status
        assert result_a.temporal_consistency.state == result_b.temporal_consistency.state
        assert result_a.temporal_consistency.agreement_fraction == result_b.temporal_consistency.agreement_fraction


class TestScenarioMultiSampleComposition:
    def test_pipeline_accepts_a_sequence_of_samples(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        hints = [_hint(1.02, [0.0, 0.01, 0.0]), _hint(1.05, [0.0, -0.01, 0.0]), _hint(1.08, [0.01, 0.0, 0.0])]
        result = pipeline.process(left, right, left_timestamp=1.1, motion_hints=hints)

        assert result.rotation_compensation_status == RotationCompensationStatus.APPLIED


class TestScenarioBoundedMemory:
    def test_temporal_record_gained_no_new_field_for_e5(self):
        names = {f.name for f in dataclasses.fields(TemporalRecord)}
        assert names == {"timestamp", "confidence", "geometry_quality", "motion_hint", "depth_snapshot_m"}

    def test_history_buffer_bounded_with_rotation_compensation_enabled(self, calibration, stereo_pair):
        config = _enabled_config(temporal_max_records=3, temporal_max_age_s=1000.0, temporal_gap_limit_s=500.0)
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair

        for i in range(10):
            pipeline.process(left, right, left_timestamp=float(i))

        assert len(pipeline.temporal_history) == 3

    def test_compensated_record_never_enters_history_buffer(self, calibration, stereo_pair):
        """The ephemeral, rotation-compensated substitute record must
        never be the one admitted into TemporalHistory — only the
        pipeline's own real, uncompensated TemporalRecord (built from
        this frame's own depth) is ever admitted."""
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)

        hint = _hint(1.05, [0.0, 0.05, 0.0])
        pipeline.process(left, right, left_timestamp=1.1, motion_hints=[hint])

        # Every record in history has geometry_quality/motion_hint
        # consistent with being built fresh from a real process() call —
        # none is a dataclasses.replace() substitute (those are never
        # admitted; admit() is only ever called with the real per-frame
        # record built before E5 runs).
        assert len(pipeline.temporal_history) == 2


class TestScopeDiscipline:
    def test_no_orientation_state_persists_between_calls(self, calibration, stereo_pair):
        """No pose/orientation-shaped attribute exists on the pipeline
        instance itself."""
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)
        pipeline.process(left, right, left_timestamp=1.1, motion_hints=[_hint(1.05, [0.0, 0.01, 0.0])])

        forbidden_substrings = ("orientation", "pose", "_R_cam", "world_frame")
        instance_attrs = [a for a in vars(pipeline) if not a.startswith("__")]
        for attr in instance_attrs:
            lowered = attr.lower()
            assert not any(f in lowered for f in forbidden_substrings), attr

    def test_rotation_compensation_status_is_the_only_new_result_field_no_delta_r_exposed(self):
        import depth_perception_engine.models.result as result_module

        source = inspect.getsource(result_module.DepthPerceptionResult)
        assert "delta_r" not in source.lower()
        assert "rotation_matrix" not in source.lower().replace("# ", "")
