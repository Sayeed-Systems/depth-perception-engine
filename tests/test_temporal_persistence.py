"""
Level 4, Phase E7 tests — deterministic, per-cell temporal-persistence
classification.

Covers: the stateful TemporalPersistenceTracker (unit level, direct
control over synthetic depth grids and reliability states) and
DepthPerceptionPipeline integration (config gating, result-field
population, reset/gap clearing, byte-identical Level 0-6 outputs).

No semantics, neural/object-recognition, planning, VIO, SLAM, or
localization is exercised here — E7 performs none of that.
"""

import numpy as np
import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.frames import FrameId
from depth_perception_engine.pipeline import DepthPerceptionPipeline
from depth_perception_engine.temporal import MotionHint
from depth_perception_engine.temporal.persistence import (
    TemporalPersistenceCellState,
    TemporalPersistenceState,
    TemporalPersistenceTracker,
)
from depth_perception_engine.temporal.reliability import MotionAwareReliabilityState
from depth_perception_engine.temporal.rotation_compensation import compensate_prior_geometry, integrate_angular_velocity

_TOLERANCE = 0.05
_STRIDE = 1
_FOCAL = 400.0
_PP = (2.0, 2.0)


def _tracker(min_support=2, max_dropout=1, expiration=5, tolerance=_TOLERANCE):
    return TemporalPersistenceTracker(
        min_support_count=min_support,
        max_dropout_frames=max_dropout,
        expiration_absence_frames=expiration,
        agreement_tolerance_m=tolerance,
    )


def _update(tracker, depth, ts, reliability=MotionAwareReliabilityState.RELIABLE, hints=None, rotate=False):
    return tracker.update(
        depth, ts, reliability, hints,
        enable_rotation_compensation=rotate,
        stride=_STRIDE, focal_length_px=_FOCAL, principal_point_px=_PP,
    )


def _hint(ts, omega):
    return MotionHint(timestamp=ts, angular_velocity_rad_s=np.array(omega, dtype=np.float64), frame_id=FrameId.BODY)


# ===================================================================
# Basic classification
# ===================================================================
class TestBasicClassification:
    def test_one_frame_surface_is_new(self):
        tracker = _tracker()
        depth = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        result = _update(tracker, depth, 0.0)

        assert result.state == TemporalPersistenceState.CLASSIFIED
        assert result.state_grid[0, 1] == TemporalPersistenceCellState.NEW
        assert result.support_count_grid[0, 1] == 1
        assert result.new_count == 1
        assert result.persistent_count == 0
        assert result.eligible_count == 1

    def test_single_observation_can_never_be_persistent(self):
        """Rule 2, structurally: PipelineConfig validates
        persistence_min_support_count >= 2, so a first observation
        (support_count == 1) can never read PERSISTENT."""
        tracker = _tracker(min_support=2)
        depth = np.array([[2.0]], dtype=np.float32)
        result = _update(tracker, depth, 0.0)
        assert result.state_grid[0, 0] != TemporalPersistenceCellState.PERSISTENT

    def test_repeated_stable_surface_becomes_persistent(self):
        tracker = _tracker(min_support=2)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        result = _update(tracker, depth, 1.0)

        assert result.state_grid[0, 0] == TemporalPersistenceCellState.PERSISTENT
        assert result.support_count_grid[0, 0] == 2
        assert result.persistent_count == 1
        assert result.persistent_fraction == 1.0

    def test_repeated_agreement_keeps_incrementing_support(self):
        tracker = _tracker(min_support=2)
        depth = np.array([[2.0]], dtype=np.float32)
        for i, ts in enumerate([0.0, 1.0, 2.0, 3.0], start=1):
            result = _update(tracker, depth, ts)
        assert result.support_count_grid[0, 0] == 4

    def test_strong_new_obstacle_appears_immediately_as_new_alongside_persistent(self):
        """History (an already-PERSISTENT cell elsewhere) must never
        suppress or delay a brand-new occupied cell appearing this same
        frame — Rule 3."""
        tracker = _tracker(min_support=2)
        depth_a = np.array([[2.0, 0.0]], dtype=np.float32)
        _update(tracker, depth_a, 0.0)
        _update(tracker, depth_a, 1.0)  # cell (0,0) now PERSISTENT

        depth_b = np.array([[2.0, 3.0]], dtype=np.float32)  # (0,1) newly occupied
        result = _update(tracker, depth_b, 2.0)

        assert result.state_grid[0, 0] == TemporalPersistenceCellState.PERSISTENT
        assert result.state_grid[0, 1] == TemporalPersistenceCellState.NEW

    def test_contradiction_resets_to_new_not_suppressed(self):
        """Strong current geometry that disagrees with tracked history at
        the same cell must reclassify as NEW immediately, never blocked
        or held back by the stale prior value — Rule 3."""
        tracker = _tracker(min_support=2)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        _update(tracker, depth, 1.0)  # PERSISTENT, support_count=2

        contradicting = np.array([[9.0]], dtype=np.float32)  # far beyond tolerance
        result = _update(tracker, contradicting, 2.0)

        assert result.state_grid[0, 0] == TemporalPersistenceCellState.NEW
        assert result.support_count_grid[0, 0] == 1


# ===================================================================
# Dropout / disappearance / expiration
# ===================================================================
class TestDropoutAndExpiration:
    def test_one_dropout_does_not_instantly_erase_persistence(self):
        tracker = _tracker(min_support=2, max_dropout=1, expiration=5)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        _update(tracker, depth, 1.0)  # PERSISTENT

        empty = np.array([[0.0]], dtype=np.float32)
        result = _update(tracker, empty, 2.0)  # one dropout frame

        assert result.state_grid[0, 0] == TemporalPersistenceCellState.PERSISTENT
        assert result.support_count_grid[0, 0] == 2  # unchanged, not reset

    def test_repeated_absence_becomes_disappearing(self):
        tracker = _tracker(min_support=2, max_dropout=1, expiration=5)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        _update(tracker, depth, 1.0)  # PERSISTENT

        empty = np.array([[0.0]], dtype=np.float32)
        _update(tracker, empty, 2.0)  # dropout 1 -> still PERSISTENT (grace)
        result = _update(tracker, empty, 3.0)  # dropout 2 -> beyond grace

        assert result.state_grid[0, 0] == TemporalPersistenceCellState.DISAPPEARING

    def test_expiration_is_deterministic_and_reverts_to_no_evidence(self):
        tracker = _tracker(min_support=2, max_dropout=1, expiration=3)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        _update(tracker, depth, 1.0)  # PERSISTENT, support=2

        empty = np.array([[0.0]], dtype=np.float32)
        _update(tracker, empty, 2.0)   # absence 1 (<=max_dropout) -> PERSISTENT
        _update(tracker, empty, 3.0)   # absence 2 (>max_dropout) -> DISAPPEARING
        _update(tracker, empty, 4.0)   # absence 3 (>expiration=3?) boundary
        result = _update(tracker, empty, 5.0)  # absence 4 > expiration=3 -> EXPIRED

        assert result.state_grid[0, 0] == TemporalPersistenceCellState.NO_EVIDENCE
        assert result.support_count_grid[0, 0] == 0
        assert result.expired_count == 1

    def test_reappearance_after_expiration_is_new_again(self):
        tracker = _tracker(min_support=2, max_dropout=0, expiration=1)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        _update(tracker, depth, 1.0)  # PERSISTENT

        empty = np.array([[0.0]], dtype=np.float32)
        _update(tracker, empty, 2.0)  # absence 1 > expiration(1)? equal, not expired yet
        _update(tracker, empty, 3.0)  # absence 2 > expiration(1) -> expired

        result = _update(tracker, depth, 4.0)  # reappears
        assert result.state_grid[0, 0] == TemporalPersistenceCellState.NEW
        assert result.support_count_grid[0, 0] == 1


# ===================================================================
# UNKNOWN != FREE
# ===================================================================
class TestUnknownNeverFree:
    def test_no_evidence_cells_never_encode_free(self):
        """Only four codes exist at all (NO_EVIDENCE/NEW/PERSISTENT/
        DISAPPEARING) — none of them means or implies FREE space."""
        tracker = _tracker()
        depth = np.zeros((3, 3), dtype=np.float32)  # nothing occupied, ever
        result = _update(tracker, depth, 0.0)

        assert set(np.unique(result.state_grid).tolist()) <= {
            TemporalPersistenceCellState.NO_EVIDENCE,
        }
        assert result.eligible_count == 0
        assert result.persistent_fraction is None

    def test_dropout_never_reports_a_free_state(self):
        tracker = _tracker(min_support=2, max_dropout=5, expiration=10)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        _update(tracker, depth, 1.0)

        empty = np.array([[0.0]], dtype=np.float32)
        result = _update(tracker, empty, 2.0)
        assert result.state_grid[0, 0] in (
            TemporalPersistenceCellState.PERSISTENT, TemporalPersistenceCellState.DISAPPEARING,
        )


# ===================================================================
# E6 reliability gating (Rule 7)
# ===================================================================
class TestReliabilityGating:
    def test_unreliable_frame_does_not_create_persistence(self):
        tracker = _tracker()
        depth = np.array([[2.0]], dtype=np.float32)
        result = _update(tracker, depth, 0.0, reliability=MotionAwareReliabilityState.UNRELIABLE)

        assert result.state == TemporalPersistenceState.UNRELIABLE
        assert result.state_grid is None  # nothing classified yet at all
        assert result.support_count_grid is None

    def test_unreliable_frame_does_not_reinforce_existing_persistence(self):
        tracker = _tracker(min_support=2)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        _update(tracker, depth, 1.0)  # PERSISTENT, support=2

        result = _update(tracker, depth, 2.0, reliability=MotionAwareReliabilityState.UNRELIABLE)
        assert result.state == TemporalPersistenceState.UNRELIABLE
        assert result.support_count_grid[0, 0] == 2  # unchanged, not incremented to 3

        # Grid was NOT erased either (Rule 4 spirit: degradation must not
        # immediately erase previously persistent evidence).
        assert result.state_grid[0, 0] == TemporalPersistenceCellState.PERSISTENT

    def test_reliable_frame_after_unreliable_continues_building_support(self):
        tracker = _tracker(min_support=3)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        _update(tracker, depth, 1.0)  # support=2

        _update(tracker, depth, 2.0, reliability=MotionAwareReliabilityState.UNRELIABLE)  # skipped
        result = _update(tracker, depth, 3.0)  # support=3

        assert result.support_count_grid[0, 0] == 3
        assert result.state_grid[0, 0] == TemporalPersistenceCellState.PERSISTENT

    def test_degraded_and_insufficient_evidence_states_still_permit_updates(self):
        """Only UNRELIABLE gates persistence — E6's own DEGRADED and
        INSUFFICIENT_EVIDENCE states are ordinary, permitted inputs (a
        first frame naturally reports E6 INSUFFICIENT_EVIDENCE too, which
        must not block E7's own bookkeeping)."""
        for state in (MotionAwareReliabilityState.DEGRADED, MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE, None):
            tracker = _tracker()
            depth = np.array([[2.0]], dtype=np.float32)
            result = _update(tracker, depth, 0.0, reliability=state)
            assert result.state == TemporalPersistenceState.CLASSIFIED
            assert result.state_grid[0, 0] == TemporalPersistenceCellState.NEW


# ===================================================================
# Reset / clear
# ===================================================================
class TestResetClearsState:
    def test_clear_resets_persistence_state(self):
        tracker = _tracker(min_support=2)
        depth = np.array([[2.0]], dtype=np.float32)
        _update(tracker, depth, 0.0)
        _update(tracker, depth, 1.0)  # PERSISTENT

        tracker.clear()
        result = _update(tracker, depth, 2.0)

        assert result.state_grid[0, 0] == TemporalPersistenceCellState.NEW
        assert result.support_count_grid[0, 0] == 1


# ===================================================================
# Determinism / bounded memory
# ===================================================================
class TestDeterminism:
    def test_identical_sequences_produce_identical_results(self):
        depth_sequence = [
            np.array([[2.0, 0.0]], dtype=np.float32),
            np.array([[2.0, 3.0]], dtype=np.float32),
            np.array([[0.0, 3.0]], dtype=np.float32),
        ]

        def _run():
            tracker = _tracker(min_support=2)
            results = []
            for i, depth in enumerate(depth_sequence):
                results.append(_update(tracker, depth, float(i)))
            return results

        a = _run()
        b = _run()
        for ra, rb in zip(a, b):
            np.testing.assert_array_equal(ra.state_grid, rb.state_grid)
            np.testing.assert_array_equal(ra.support_count_grid, rb.support_count_grid)
            assert ra.state == rb.state


class TestBoundedMemory:
    def test_grid_shape_never_grows_with_frame_count(self):
        tracker = _tracker()
        depth = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        shapes = set()
        for i in range(50):
            result = _update(tracker, depth, float(i))
            shapes.add(result.state_grid.shape)
            shapes.add(result.support_count_grid.shape)
        assert shapes == {(2, 2)}


# ===================================================================
# Rotation-compensated persistence (reuses E5)
# ===================================================================
class TestRotationCompensatedPersistence:
    def test_rotation_compensated_observation_can_become_persistent(self):
        rng = np.random.default_rng(3)
        h, w = 40, 40
        previous_snapshot = rng.uniform(2.0, 4.0, size=(h, w)).astype(np.float32)
        focal_length_px = 400.0
        principal_point_px = (float(w) / 2.0, float(h) / 2.0)
        stride = 1
        omega = np.array([0.0, 0.02, 0.0])
        hint = MotionHint(timestamp=1.0, angular_velocity_rad_s=omega, frame_id=FrameId.BODY)
        delta_r = integrate_angular_velocity([hint], previous_timestamp=0.0)

        # "What the camera actually sees now" after this exact rotation —
        # same synthetic construction test_rotation_compensation.py's own
        # TestSyntheticRotationImprovesComparability uses.
        current_snapshot = compensate_prior_geometry(
            previous_snapshot, delta_r, stride, focal_length_px, principal_point_px,
        )
        assert (current_snapshot > 0.0).sum() > 0

        tracker_with_compensation = TemporalPersistenceTracker(
            min_support_count=2, max_dropout_frames=1, expiration_absence_frames=5, agreement_tolerance_m=_TOLERANCE,
        )
        tracker_with_compensation.update(
            previous_snapshot, 0.0, MotionAwareReliabilityState.RELIABLE, None,
            enable_rotation_compensation=True, stride=stride,
            focal_length_px=focal_length_px, principal_point_px=principal_point_px,
        )
        result_with = tracker_with_compensation.update(
            current_snapshot, 1.0, MotionAwareReliabilityState.RELIABLE, [hint],
            enable_rotation_compensation=True, stride=stride,
            focal_length_px=focal_length_px, principal_point_px=principal_point_px,
        )

        tracker_without_compensation = TemporalPersistenceTracker(
            min_support_count=2, max_dropout_frames=1, expiration_absence_frames=5, agreement_tolerance_m=_TOLERANCE,
        )
        tracker_without_compensation.update(
            previous_snapshot, 0.0, MotionAwareReliabilityState.RELIABLE, None,
            enable_rotation_compensation=False, stride=stride,
            focal_length_px=focal_length_px, principal_point_px=principal_point_px,
        )
        result_without = tracker_without_compensation.update(
            current_snapshot, 1.0, MotionAwareReliabilityState.RELIABLE, [hint],
            enable_rotation_compensation=False, stride=stride,
            focal_length_px=focal_length_px, principal_point_px=principal_point_px,
        )

        # WITH rotation compensation: the (physically stationary) surface
        # is recognized as the same evidence, seen twice -> PERSISTENT.
        assert result_with.persistent_count > 0
        # WITHOUT: raw grid-aligned comparison sees a totally different
        # depth field at each pixel (the surface visibly moved in image
        # space) -> far fewer (typically zero) cells read PERSISTENT.
        assert result_without.persistent_count < result_with.persistent_count


# ===================================================================
# Structural-incompatibility fallback (rare, defensive)
# ===================================================================
class TestStructuralIncompatibility:
    def test_shape_change_yields_insufficient_evidence(self):
        tracker = _tracker()
        _update(tracker, np.array([[2.0]], dtype=np.float32), 0.0)

        result = _update(tracker, np.zeros((2, 2), dtype=np.float32), 1.0)
        assert result.state == TemporalPersistenceState.INSUFFICIENT_EVIDENCE
        assert result.state_grid is None


# ===================================================================
# PipelineConfig contract
# ===================================================================
class TestPipelineConfigField:
    def test_default_false(self):
        assert PipelineConfig().enable_temporal_persistence is False

    def test_min_support_count_must_be_at_least_two(self):
        with pytest.raises(ValueError):
            PipelineConfig(persistence_min_support_count=1)

    def test_expiration_must_exceed_dropout_grace(self):
        with pytest.raises(ValueError):
            PipelineConfig(persistence_max_dropout_frames=3, persistence_expiration_absence_frames=3)

    def test_negative_dropout_frames_rejected(self):
        with pytest.raises(ValueError):
            PipelineConfig(persistence_max_dropout_frames=-1)


# ===================================================================
# Pipeline integration
# ===================================================================
def _enabled_config(**overrides):
    return PipelineConfig(
        enable_temporal=True,
        enable_motion_aware_reliability=True,
        enable_temporal_persistence=True,
        **overrides,
    )


class TestPipelineDisabledByDefault:
    def test_disabled_by_default(self, config, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.temporal_persistence is None

    def test_flag_alone_without_prerequisites_has_no_effect(self, calibration, stereo_pair):
        config = PipelineConfig(enable_temporal=True, enable_temporal_persistence=True)  # E6 missing
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair
        result = pipeline.process(left, right, left_timestamp=1.0)
        assert result.temporal_persistence is None


class TestPipelineClassifiesRepeatedEvidence:
    def test_repeated_identical_frame_builds_persistence(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(persistence_min_support_count=2), calibration)
        left, right = stereo_pair

        first = pipeline.process(left, right, left_timestamp=1.0)
        second = pipeline.process(left, right, left_timestamp=1.1)  # deterministic SGBM -> identical depth

        assert first.temporal_persistence.state == "CLASSIFIED"
        assert first.temporal_persistence.new_count > 0
        assert first.temporal_persistence.persistent_count == 0

        assert second.temporal_persistence.persistent_count > 0

    def test_reset_clears_pipeline_level_persistence(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(persistence_min_support_count=2), calibration)
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)
        pipeline.process(left, right, left_timestamp=1.1)  # some cells now PERSISTENT

        pipeline.reset()
        result = pipeline.process(left, right, left_timestamp=5.0)

        assert result.temporal_persistence.persistent_count == 0
        assert result.temporal_persistence.new_count > 0

    def test_large_timestamp_gap_starts_fresh_sequence(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(
            _enabled_config(persistence_min_support_count=2, temporal_gap_limit_s=0.5), calibration,
        )
        left, right = stereo_pair
        pipeline.process(left, right, left_timestamp=1.0)
        pipeline.process(left, right, left_timestamp=1.1)  # PERSISTENT built up

        result = pipeline.process(left, right, left_timestamp=100.0)  # gap > limit
        assert result.temporal_persistence.persistent_count == 0
        assert result.temporal_persistence.new_count > 0


class TestPipelineOutputsUnchanged:
    def test_level0_through_6_outputs_byte_identical_with_and_without_e7(self, calibration, stereo_pair):
        left, right = stereo_pair

        pipeline_e7 = DepthPerceptionPipeline(_enabled_config(), calibration)
        result_e7 = pipeline_e7.process(left, right, left_timestamp=1.0)
        result_e7_b = pipeline_e7.process(left, right, left_timestamp=1.1)

        pipeline_no_e7 = DepthPerceptionPipeline(
            PipelineConfig(enable_temporal=True, enable_motion_aware_reliability=True, enable_temporal_persistence=False),
            calibration,
        )
        result_no_e7 = pipeline_no_e7.process(left, right, left_timestamp=1.0)
        result_no_e7_b = pipeline_no_e7.process(left, right, left_timestamp=1.1)

        np.testing.assert_array_equal(result_e7.disparity_map, result_no_e7.disparity_map)
        np.testing.assert_array_equal(result_e7.depth_map, result_no_e7.depth_map)
        assert result_e7.confidence == result_no_e7.confidence
        assert result_e7_b.motion_aware_reliability.state == result_no_e7_b.motion_aware_reliability.state
        assert result_no_e7.temporal_persistence is None
        assert result_no_e7_b.temporal_persistence is None


class TestPipelineBoundedMemory:
    def test_grid_size_bounded_across_many_frames(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
        left, right = stereo_pair
        shapes = set()
        for i in range(20):
            result = pipeline.process(left, right, left_timestamp=float(i))
            if result.temporal_persistence.state_grid is not None:
                shapes.add(result.temporal_persistence.state_grid.shape)
        assert len(shapes) == 1


class TestPipelineDeterminism:
    def test_repeated_identical_runs_produce_identical_persistence(self, calibration, stereo_pair):
        left, right = stereo_pair

        def _run():
            pipeline = DepthPerceptionPipeline(_enabled_config(), calibration)
            pipeline.process(left, right, left_timestamp=1.0)
            return pipeline.process(left, right, left_timestamp=1.1)

        result_a = _run()
        result_b = _run()

        assert result_a.temporal_persistence.state == result_b.temporal_persistence.state
        assert result_a.temporal_persistence.new_count == result_b.temporal_persistence.new_count
        assert result_a.temporal_persistence.persistent_count == result_b.temporal_persistence.persistent_count
        np.testing.assert_array_equal(
            result_a.temporal_persistence.state_grid, result_b.temporal_persistence.state_grid,
        )
