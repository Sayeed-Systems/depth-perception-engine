"""
Level 4 (Phase E1) contract tests.

Covers exactly what E1 froze: temporal.MotionHint (construction, validation,
immutability, frame-genericity, deliberate absence of pose/localization
fields) and StereoObservation.motion_hint (default, backward compatibility,
non-consumption by process_observation()). No algorithm exists yet, so no
algorithm is tested here — mirrors this phase's own instruction not to
write "fake algorithm tests for algorithms that do not exist."
"""

import dataclasses

import numpy as np
import pytest

from depth_perception_engine.frames import FrameId
from depth_perception_engine.models.result import DepthPerceptionResult, StereoObservation
from depth_perception_engine.pipeline.pipeline import DepthPerceptionPipeline
from depth_perception_engine.temporal import MotionHint


class TestMotionHintConstruction:
    def test_minimal_valid_construction(self):
        hint = MotionHint(
            timestamp=1.0,
            angular_velocity_rad_s=np.array([0.0, 0.0, 0.0]),
            frame_id=FrameId.BODY,
        )
        assert hint.timestamp == 1.0
        assert hint.frame_id == FrameId.BODY
        assert np.array_equal(hint.angular_velocity_rad_s, np.zeros(3))

    def test_valid_defaults_to_true(self):
        hint = MotionHint(
            timestamp=1.0,
            angular_velocity_rad_s=np.array([0.1, 0.2, 0.3]),
            frame_id=FrameId.BODY,
        )
        assert hint.valid is True

    def test_valid_can_be_explicitly_false(self):
        hint = MotionHint(
            timestamp=1.0,
            angular_velocity_rad_s=np.array([0.1, 0.2, 0.3]),
            frame_id=FrameId.BODY,
            valid=False,
        )
        assert hint.valid is False

    def test_frame_id_is_generic_not_hardcoded_to_body(self):
        """Mirrors tests/test_rigid_transform.py's own frame-genericity
        proof for RigidTransform — MotionHint must accept any caller-
        defined frame string, not just frames.FrameId.BODY."""
        hint = MotionHint(
            timestamp=1.0,
            angular_velocity_rad_s=np.zeros(3),
            frame_id="a_future_second_imu_frame",
        )
        assert hint.frame_id == "a_future_second_imu_frame"

        hint_camera = MotionHint(
            timestamp=1.0,
            angular_velocity_rad_s=np.zeros(3),
            frame_id=FrameId.CAMERA_OPTICAL_LEFT,
        )
        assert hint_camera.frame_id == FrameId.CAMERA_OPTICAL_LEFT


class TestMotionHintValidation:
    def test_rejects_wrong_shape_angular_velocity(self):
        with pytest.raises(ValueError):
            MotionHint(
                timestamp=1.0,
                angular_velocity_rad_s=np.array([0.0, 0.0]),
                frame_id=FrameId.BODY,
            )

    def test_rejects_non_ndarray_angular_velocity(self):
        with pytest.raises(ValueError):
            MotionHint(
                timestamp=1.0,
                angular_velocity_rad_s=[0.0, 0.0, 0.0],
                frame_id=FrameId.BODY,
            )

    def test_rejects_2d_angular_velocity(self):
        with pytest.raises(ValueError):
            MotionHint(
                timestamp=1.0,
                angular_velocity_rad_s=np.zeros((3, 1)),
                frame_id=FrameId.BODY,
            )

    def test_rejects_empty_frame_id(self):
        with pytest.raises(ValueError):
            MotionHint(
                timestamp=1.0,
                angular_velocity_rad_s=np.zeros(3),
                frame_id="",
            )


class TestMotionHintImmutability:
    def test_frozen_dataclass_rejects_attribute_assignment(self):
        hint = MotionHint(
            timestamp=1.0,
            angular_velocity_rad_s=np.zeros(3),
            frame_id=FrameId.BODY,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            hint.timestamp = 2.0


class TestMotionHintScopeDiscipline:
    """Guards against exactly the scope creep this phase's own instructions
    forbid: no linear velocity/acceleration/position field, and no
    source/is_simulated field that would invite simulation-specific
    branching in a future consumer. See docs/LEVEL4_CONTRACTS.md."""

    EXPECTED_FIELDS = {"timestamp", "angular_velocity_rad_s", "frame_id", "valid"}

    def test_field_set_is_exactly_the_frozen_e1_contract(self):
        actual_fields = {f.name for f in dataclasses.fields(MotionHint)}
        assert actual_fields == self.EXPECTED_FIELDS, (
            "MotionHint's field set changed — if this is a deliberate "
            "extension, update docs/LEVEL4_CONTRACTS.md's rationale "
            "alongside this test, don't just widen the assertion."
        )

    def test_no_source_or_provenance_field(self):
        names = {f.name for f in dataclasses.fields(MotionHint)}
        forbidden = {"source", "is_simulated", "provenance", "producer"}
        assert not (names & forbidden), (
            "MotionHint must never carry a source/provenance field — see "
            "docs/LEVEL4_SIMULATED_IMU.md."
        )

    def test_no_translation_or_pose_field(self):
        names = {f.name for f in dataclasses.fields(MotionHint)}
        forbidden = {
            "linear_velocity_m_s", "position", "translation", "pose",
            "acceleration_m_s2", "velocity",
        }
        assert not (names & forbidden), (
            "MotionHint must stay rotation-only — a translation-capable "
            "field would invite localization creep, see "
            "docs/LEVEL4_ARCHITECTURE.md section 3."
        )


class TestTemporalPackagePublicSurface:
    def test_temporal_all_is_exactly_the_frozen_e1_e6_contract(self):
        """Was ["MotionHint"] only at Phase E1; Phase E2 added
        TemporalRecord/TemporalHistory/TemporalAdmissionStatus; Phase E3
        added TemporalConsistency/TemporalConsistencyState/
        compute_temporal_consistency; Phase E4 added
        TemporalStabilization/TemporalStabilizationState/
        compute_temporal_stabilization; Phase E5 added
        RotationCompensationStatus/compute_rotation_compensation/
        compensate_prior_geometry_with_payload (the last of these added at
        Phase E7, alongside E5's own module, since it shares E5's exact
        reprojection implementation — see
        temporal/rotation_compensation.py's own module docstring); Phase E6
        added MotionAwareReliability/MotionAwareReliabilityState/
        compute_motion_aware_reliability; Phase E7 added
        TemporalPersistence/TemporalPersistenceState/
        TemporalPersistenceCellState/TemporalPersistenceTracker (see
        tests/test_temporal_history.py, tests/test_temporal_consistency.py,
        tests/test_temporal_stabilization.py,
        tests/test_rotation_compensation.py,
        tests/test_motion_aware_reliability.py,
        tests/test_temporal_persistence.py) — updated here alongside each
        addition, not widened silently. Still a boundary marker: any
        further addition must update this assertion deliberately, not by
        accident."""
        import depth_perception_engine.temporal as temporal_pkg

        assert temporal_pkg.__all__ == [
            "MotionHint",
            "TemporalRecord",
            "TemporalHistory",
            "TemporalAdmissionStatus",
            "TemporalConsistency",
            "TemporalConsistencyState",
            "compute_temporal_consistency",
            "TemporalStabilization",
            "TemporalStabilizationState",
            "compute_temporal_stabilization",
            "RotationCompensationStatus",
            "compute_rotation_compensation",
            "compensate_prior_geometry_with_payload",
            "MotionAwareReliability",
            "MotionAwareReliabilityState",
            "compute_motion_aware_reliability",
            "TemporalPersistence",
            "TemporalPersistenceState",
            "TemporalPersistenceCellState",
            "TemporalPersistenceTracker",
        ]


class TestStereoObservationMotionHint:
    def test_defaults_to_none(self, stereo_pair):
        left, right = stereo_pair
        obs = StereoObservation(left_image=left, right_image=right)
        assert obs.motion_hint is None

    def test_can_be_constructed_without_motion_hint_backward_compatible(self, stereo_pair):
        """Every pre-Level-4 construction of StereoObservation must keep
        working unmodified — the whole point of an additive, defaulted
        field."""
        left, right = stereo_pair
        obs = StereoObservation(
            left_image=left, right_image=right,
            left_timestamp=1.0, right_timestamp=1.1, frame_id="f0",
        )
        assert obs.motion_hint is None

    def test_can_carry_a_motion_hint(self, stereo_pair):
        left, right = stereo_pair
        hint = MotionHint(
            timestamp=1.0, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY,
        )
        obs = StereoObservation(left_image=left, right_image=right, motion_hint=hint)
        assert obs.motion_hint is hint


class TestStereoObservationMotionHintNotConsumed:
    """Level 4, Phase E1: motion_hint is reserved but inert — mirrors
    test_pipeline.py::TestProcessObservation's own proof that `calibration`
    is inert today."""

    def test_process_observation_output_identical_with_or_without_motion_hint(
        self, config, calibration, stereo_pair,
    ):
        left, right = stereo_pair
        hint = MotionHint(
            timestamp=1.0, angular_velocity_rad_s=np.array([1.0, 2.0, 3.0]), frame_id=FrameId.BODY,
        )
        obs_without = StereoObservation(left_image=left, right_image=right, left_timestamp=1.0)
        obs_with = StereoObservation(
            left_image=left, right_image=right, left_timestamp=1.0, motion_hint=hint,
        )

        pipeline_without = DepthPerceptionPipeline(config, calibration)
        pipeline_with = DepthPerceptionPipeline(config, calibration)

        result_without = pipeline_without.process_observation(obs_without)
        result_with = pipeline_with.process_observation(obs_with)

        assert isinstance(result_without, DepthPerceptionResult)
        assert isinstance(result_with, DepthPerceptionResult)
        np.testing.assert_array_equal(result_without.disparity_map, result_with.disparity_map)
        np.testing.assert_array_equal(result_without.depth_map, result_with.depth_map)
        assert result_without.confidence == result_with.confidence
        assert result_without.timestamp == result_with.timestamp

    def test_an_invalid_motion_hint_is_still_not_consumed(self, config, calibration, stereo_pair):
        """valid=False must not trip any special handling either — nothing
        reads this field yet, valid or not."""
        left, right = stereo_pair
        invalid_hint = MotionHint(
            timestamp=1.0, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY, valid=False,
        )
        obs = StereoObservation(left_image=left, right_image=right, motion_hint=invalid_hint)
        pipeline = DepthPerceptionPipeline(config, calibration)

        result = pipeline.process_observation(obs)

        assert isinstance(result, DepthPerceptionResult)
