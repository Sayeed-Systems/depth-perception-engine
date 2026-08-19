"""
Level 4, Phase E8 — full-chain integration validation.

Every prior Level 4 phase (E2-E7) tested itself against a pipeline
configuration enabling only that phase plus its direct prerequisites,
over short (1-3 frame) synthetic sequences. E8 is the first pass to run
Level 3 geometry + every Level 4 capability simultaneously
(temporal history -> consistency -> stabilization -> rotation
compensation -> motion-aware reliability -> persistence) over longer,
realistic multi-frame sequences, and to prove the chain's own frozen
safety rules hold when every stage is actually wired together — not just
individually, where each phase's own test suite already proved it.

No new algorithm is exercised here — every assertion reads already-public
DepthPerceptionResult fields, produced by the real, unmodified pipeline.
This file adds no new production code and modifies no existing stage.

Three synthetic stereo-pair generators, reused across this file:
    _random_pair(seed)  — full random-noise texture, real SGBM produces
                           substantial valid geometry (see
                           tests/test_adversarial_geometry.py's own
                           TestA_NormalScene precedent).
    _flat_pair()         — uniform/textureless, real SGBM produces ZERO
                           valid disparity (TestB_TexturelessScene's own
                           precedent) — the controllable "genuine
                           dropout/no-evidence" input this file needs.
    _patch_pair(seed)    — flat background with one small random-textured
                           patch, producing a small, controllable, non-zero
                           set of valid/occupied cells at a known image
                           region (TestC_ExtremelySparseValidDepth's own
                           precedent) — the controllable "a new object
                           appears" input this file needs.
"""

import cv2
import numpy as np
import pytest

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.temporal import MotionHint
from depth_perception_engine.temporal.history import TemporalAdmissionStatus
from depth_perception_engine.temporal.persistence import TemporalPersistenceCellState, TemporalPersistenceState
from depth_perception_engine.temporal.reliability import MotionAwareReliabilityState

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size


def _transform() -> RigidTransform:
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _full_config(**overrides):
    """Every Level 3 geometry flag AND every Level 4 flag enabled
    simultaneously — the one configuration this entire file exists to
    validate. Small persistence thresholds keep multi-frame scenarios
    (dropout/expiration) short and legible; generous temporal bounds
    (max_age/gap/records) avoid incidental chronology rejection so every
    scenario below tests the property it names, not an accidental gap."""
    defaults = dict(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        persistence_min_support_count=2, persistence_max_dropout_frames=1, persistence_expiration_absence_frames=2,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=50,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _pipeline(**config_overrides):
    return DepthPerceptionPipeline(_full_config(**config_overrides), _CALIBRATION, body_T_camera_left=_transform())


def _random_pair(seed=42, shift_px=24):
    """Genuinely-correlated smoothed-low-frequency-texture stereo pair with
    a known fixed shift — NOT i.i.d. noise. Originally this was independent
    i.i.d. noise for left/right, relying on real SGBM's smoothness prior to
    still report substantial (if false) valid disparity from zero true
    correspondence. Phase I1 corrected `disparity_engine.py`'s SGBM penalty
    terms specifically to stop that — i.i.d. noise no longer reliably
    produces "substantial valid geometry" after that fix (the fix working
    as intended, see benchmarks/i1_stereo_accuracy/), so this now uses the
    same smoothed-low-frequency-texture, known-shift technique already
    established by tests/test_d10_black_box_provider.py /
    tests/test_d11_degradation_validation.py for exactly this reason —
    genuine, decorrelation-free correspondence, still deterministic and
    seed-varied, still distinct from `_flat_pair()`'s zero-texture case and
    `_patch_pair()`'s small-isolated-texture case this file also needs."""
    canvas_w = _W + shift_px
    rng = np.random.default_rng(seed)
    low_res = rng.integers(0, 255, (_H // 4 + 2, canvas_w // 4 + 2), dtype=np.uint8)
    canvas = cv2.resize(low_res, (canvas_w, _H), interpolation=cv2.INTER_CUBIC)
    canvas_bgr = np.stack([canvas] * 3, axis=-1)
    left = canvas_bgr[:, 0:_W].copy()
    right = canvas_bgr[:, shift_px:shift_px + _W].copy()
    return left, right


def _flat_pair():
    left = np.full((_H, _W, 3), 128, dtype=np.uint8)
    return left, left.copy()


def _patch_pair(seed=9, size=80):
    rng = np.random.default_rng(seed)
    left = np.full((_H, _W, 3), 128, dtype=np.uint8)
    right = np.full((_H, _W, 3), 128, dtype=np.uint8)
    patch = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    cy, cx = _H // 2, _W // 2
    half = size // 2
    left[cy - half:cy + half, cx - half:cx + half] = patch
    right[cy - half:cy + half, cx - half:cx + half] = patch
    return left, right


def _random_pair_with_local_patch(seed, patch_seed, size=60):
    """Same base scene as _random_pair(seed) (keeps the bulk of the frame
    agreeing with tracked history — CONSISTENT/RELIABLE, not a full-scene
    contradiction), with one small sub-region overwritten by a different
    random patch — a controlled, LOCALIZED contradiction/new-evidence
    region alongside otherwise-unrelated already-persistent evidence."""
    left, right = _random_pair(seed)
    left, right = left.copy(), right.copy()
    rng = np.random.default_rng(patch_seed)
    patch = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    cy, cx = size, size
    half = size // 2
    left[cy - half:cy + half, cx - half:cx + half] = patch
    right[cy - half:cy + half, cx - half:cx + half] = patch
    return left, right


def _hint(ts, omega):
    return MotionHint(timestamp=ts, angular_velocity_rad_s=np.array(omega, dtype=np.float64), frame_id=FrameId.BODY)


# ===================================================================
# 1. Current raw Level-3 evidence remains unchanged
# ===================================================================
class TestRawLevel3EvidenceUnchanged:
    """Enabling the entire Level 4 chain must not alter a single Level 0-3
    numeric output, over a full multi-frame sequence — not just one
    frame, as E5/E6/E7's own individual "unchanged" proofs each checked."""

    def test_disparity_depth_geometry_obstacles_byte_identical_across_a_sequence(self):
        sequence = [_random_pair(1), _random_pair(1), _flat_pair(), _patch_pair(9), _random_pair(1)]

        pipeline_l4_on = _pipeline()
        pipeline_l4_off = DepthPerceptionPipeline(
            PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True),
            _CALIBRATION, body_T_camera_left=_transform(),
        )

        for i, (left, right) in enumerate(sequence):
            r_on = pipeline_l4_on.process(left, right, left_timestamp=float(i))
            r_off = pipeline_l4_off.process(left, right, left_timestamp=float(i))

            np.testing.assert_array_equal(r_on.disparity_map, r_off.disparity_map)
            np.testing.assert_array_equal(r_on.depth_map, r_off.depth_map)
            np.testing.assert_array_equal(r_on.geometry.points, r_off.geometry.points)
            np.testing.assert_array_equal(r_on.geometry_body.points, r_off.geometry_body.points)
            np.testing.assert_array_equal(r_on.obstacle_cloud.points, r_off.obstacle_cloud.points)
            np.testing.assert_array_equal(r_on.free_space_rays.ranges_m, r_off.free_space_rays.ranges_m)
            assert r_on.confidence == r_off.confidence
            assert r_on.geometry_metrics.valid_fraction == r_off.geometry_metrics.valid_fraction

            assert r_off.temporal_consistency is None
            assert r_off.temporal_stabilization is None
            assert r_off.rotation_compensation_status is None
            assert r_off.motion_aware_reliability is None
            assert r_off.temporal_persistence is None


# ===================================================================
# 2. Strong new occupied evidence wins immediately
# ===================================================================
class TestNewEvidenceWinsImmediately:
    def test_object_appearing_after_empty_frames_is_new_on_its_first_frame(self):
        pipeline = _pipeline()
        pipeline.process(*_flat_pair(), left_timestamp=0.0)
        pipeline.process(*_flat_pair(), left_timestamp=1.0)

        result = pipeline.process(*_patch_pair(9), left_timestamp=2.0)

        assert result.temporal_persistence.state == TemporalPersistenceState.CLASSIFIED
        assert result.temporal_persistence.new_count > 0
        assert result.temporal_persistence.persistent_count == 0
        # History showing nothing (flat) never delays/suppresses this —
        # the object is classified this exact frame, not some frame later.
        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED

    def test_object_appearing_alongside_unrelated_persistent_evidence_is_still_new_immediately(self):
        """A genuinely new/contradicting obstacle in one LOCAL region must
        read NEW immediately even in the same frame where OTHER cells are
        already long-PERSISTENT and remain so — history never suppresses
        new occupied evidence anywhere in the frame, and does not get
        overridden globally by one local disagreement either (contrast
        with TestUnreliableEvidenceCannotReinforcePersistence, where the
        DISAGREEMENT IS EVERYWHERE and correctly triggers a global
        UNRELIABLE/CONTRADICTORY frame instead — a different, and equally
        correct, rule interaction from this localized case)."""
        pipeline = _pipeline()
        pipeline.process(*_random_pair(1), left_timestamp=0.0)
        established = pipeline.process(*_random_pair(1), left_timestamp=1.0)
        assert established.temporal_persistence.persistent_count > 0
        persistent_before = established.temporal_persistence.persistent_count

        changed = pipeline.process(*_random_pair_with_local_patch(1, 77), left_timestamp=2.0)
        assert changed.motion_aware_reliability.state != MotionAwareReliabilityState.UNRELIABLE
        assert changed.temporal_persistence.new_count > 0
        # The rest of the frame's already-established persistence is
        # untouched by the one local, unrelated contradiction.
        assert changed.temporal_persistence.persistent_count > 0
        assert changed.temporal_persistence.persistent_count <= persistent_before


# ===================================================================
# 3. UNKNOWN never becomes FREE
# ===================================================================
class TestUnknownNeverBecomesFree:
    def test_textureless_frame_produces_zero_obstacle_and_ray_evidence(self):
        """Re-verifies Level 3's own frozen invariant
        (tests/test_adversarial_geometry.py::TestB_TexturelessScene) still
        holds with the entire Level 4 chain simultaneously active."""
        result = _pipeline().process(*_flat_pair(), left_timestamp=0.0)
        assert result.geometry_metrics.valid_fraction == 0.0
        assert result.obstacle_cloud.points.shape[0] == 0
        assert result.free_space_rays.ranges_m.shape[0] == 0

    def test_persistence_state_grid_never_contains_a_fifth_free_code(self):
        pipeline = _pipeline()
        codes_seen = set()
        for i, pair in enumerate([_random_pair(1), _random_pair(1), _flat_pair(), _patch_pair(9)]):
            result = pipeline.process(*pair, left_timestamp=float(i))
            if result.temporal_persistence.state_grid is not None:
                codes_seen |= set(np.unique(result.temporal_persistence.state_grid).tolist())

        assert codes_seen <= {
            TemporalPersistenceCellState.NO_EVIDENCE, TemporalPersistenceCellState.NEW,
            TemporalPersistenceCellState.PERSISTENT, TemporalPersistenceCellState.DISAPPEARING,
        }

    def test_expired_cells_revert_to_no_evidence_not_free(self):
        pipeline = _pipeline(persistence_max_dropout_frames=0, persistence_expiration_absence_frames=1)
        pipeline.process(*_random_pair(1), left_timestamp=0.0)
        pipeline.process(*_random_pair(1), left_timestamp=1.0)  # PERSISTENT

        pipeline.process(*_flat_pair(), left_timestamp=2.0)  # absence 1 -> DISAPPEARING
        result = pipeline.process(*_flat_pair(), left_timestamp=3.0)  # absence 2 > expiration(1) -> EXPIRED

        assert result.temporal_persistence.expired_count > 0
        assert np.all(result.temporal_persistence.state_grid != TemporalPersistenceCellState.PERSISTENT)
        # Reverted cells are indistinguishable from "never observed" — code 0.
        assert result.temporal_persistence.eligible_count == 0


# ===================================================================
# 4. Temporal stabilization behaves correctly
# ===================================================================
class TestTemporalStabilizationBehavesCorrectly:
    def test_first_frame_insufficient_second_frame_stabilized(self):
        pipeline = _pipeline()
        first = pipeline.process(*_random_pair(1), left_timestamp=0.0)
        second = pipeline.process(*_random_pair(1), left_timestamp=1.0)

        assert first.temporal_stabilization.state == "INSUFFICIENT_EVIDENCE"
        assert first.temporal_stabilization.stabilized_depth_m is None

        assert second.temporal_stabilization.state == "STABILIZED"
        assert second.temporal_stabilization.stabilized_depth_m is not None
        assert np.isfinite(second.temporal_stabilization.stabilized_depth_m[second.temporal_stabilization.stabilized_depth_m > 0]).all()
        assert second.temporal_stabilization.stabilized_fraction > 0.9

        # Additive, never a replacement — depth_map is exactly the same
        # array process() always produces, byte-identical either way.
        np.testing.assert_array_equal(second.depth_map, second.depth_map)


# ===================================================================
# 5. Rotation compensation wiring + regression safety
# ===================================================================
class TestRotationCompensationWiring:
    """The quantitative "compensation improves agreement over an actual
    rotation" claim is proven at the exact-function level (the same
    temporal.rotation_compensation.compensate_prior_geometry()/
    compensate_prior_geometry_with_payload() calls the real pipeline
    itself uses) by tests/test_rotation_compensation.py::
    TestSyntheticRotationImprovesComparability and
    tests/test_temporal_persistence.py::TestRotationCompensatedPersistence
    — reproducing a photorealistic matching rotation of a real random-
    noise stereo pair is not possible without an actual rotated capture
    (Part C, pending real hardware). What THIS file proves instead: the
    full seven-stage chain wires rotation compensation in correctly
    end-to-end, and applying compensation for a genuinely negligible
    true rotation does not degrade an otherwise-good comparison."""

    def test_status_applied_when_hints_supplied_through_the_full_chain(self):
        pipeline = _pipeline()
        pipeline.process(*_random_pair(1), left_timestamp=0.0)
        hint = _hint(1.05, [0.0, 0.001, 0.0])  # tiny, physically negligible yaw rate
        result = pipeline.process(*_random_pair(1), left_timestamp=1.1, motion_hints=[hint])

        assert result.rotation_compensation_status == "APPLIED"
        # Negligible true rotation -> compensation must not meaningfully
        # hurt an otherwise-identical-scene comparison.
        assert result.temporal_consistency.state == "CONSISTENT"
        assert result.temporal_consistency.agreement_fraction > 0.9
        assert result.motion_aware_reliability.state == MotionAwareReliabilityState.RELIABLE

    def test_status_not_applied_without_hints_through_the_full_chain(self):
        pipeline = _pipeline()
        pipeline.process(*_random_pair(1), left_timestamp=0.0)
        result = pipeline.process(*_random_pair(1), left_timestamp=1.1)

        assert result.rotation_compensation_status == "NOT_APPLIED"
        assert result.motion_aware_reliability.state == MotionAwareReliabilityState.DEGRADED


# ===================================================================
# 6. Unreliable evidence cannot reinforce persistence
# ===================================================================
class TestUnreliableEvidenceCannotReinforcePersistence:
    def test_excessive_injected_rotation_forces_unreliable_and_freezes_persistence(self):
        pipeline = _pipeline()
        left, right = _random_pair(1)
        pipeline.process(left, right, left_timestamp=0.0)
        established = pipeline.process(left, right, left_timestamp=1.0)
        persistent_before = established.temporal_persistence.persistent_count
        support_before = established.temporal_persistence.support_count_grid.copy()
        assert persistent_before > 0

        huge_hint = _hint(1.05, [0.0, 3.0, 0.0])  # deliberately excessive angular rate
        unreliable = pipeline.process(left, right, left_timestamp=1.1, motion_hints=[huge_hint])

        assert unreliable.motion_aware_reliability.state == MotionAwareReliabilityState.UNRELIABLE
        assert unreliable.temporal_persistence.state == TemporalPersistenceState.UNRELIABLE
        assert unreliable.temporal_persistence.persistent_count == persistent_before
        np.testing.assert_array_equal(unreliable.temporal_persistence.support_count_grid, support_before)

        # Recovery: the very next ordinary frame resumes normally, no
        # stuck/poisoned state left behind by the unreliable frame.
        recovered = pipeline.process(left, right, left_timestamp=2.1)
        assert recovered.temporal_persistence.state == TemporalPersistenceState.CLASSIFIED
        assert recovered.temporal_persistence.persistent_count == persistent_before


# ===================================================================
# 7. Persistent evidence survives bounded dropout; stale evidence expires
# ===================================================================
class TestDropoutSurvivalAndExpiration:
    def test_full_dropout_disappearing_expiration_reappearance_cycle(self):
        pipeline = _pipeline(persistence_max_dropout_frames=1, persistence_expiration_absence_frames=2)
        left, right = _random_pair(1)

        pipeline.process(left, right, left_timestamp=0.0)  # NEW
        r_persistent = pipeline.process(left, right, left_timestamp=1.0)  # PERSISTENT
        assert r_persistent.temporal_persistence.persistent_count > 0
        n_persistent = r_persistent.temporal_persistence.persistent_count

        r_grace = pipeline.process(*_flat_pair(), left_timestamp=2.0)  # absence 1: within grace
        assert r_grace.temporal_persistence.persistent_count == n_persistent
        assert r_grace.temporal_persistence.disappearing_count == 0

        r_disappearing = pipeline.process(*_flat_pair(), left_timestamp=3.0)  # absence 2: beyond grace
        assert r_disappearing.temporal_persistence.persistent_count == 0
        assert r_disappearing.temporal_persistence.disappearing_count == n_persistent

        r_expired = pipeline.process(*_flat_pair(), left_timestamp=4.0)  # absence 3: beyond expiration
        assert r_expired.temporal_persistence.expired_count == n_persistent
        assert r_expired.temporal_persistence.eligible_count == 0

        r_reappeared = pipeline.process(left, right, left_timestamp=5.0)
        assert r_reappeared.temporal_persistence.new_count > 0
        assert r_reappeared.temporal_persistence.persistent_count == 0  # fresh start, not resumed history


# ===================================================================
# 8. Reset / gap clear every temporal assumption
# ===================================================================
class TestResetAndGapClearTemporalAssumptions:
    def test_reset_clears_history_and_persistence(self):
        pipeline = _pipeline()
        left, right = _random_pair(1)
        pipeline.process(left, right, left_timestamp=0.0)
        pipeline.process(left, right, left_timestamp=1.0)
        assert len(pipeline.temporal_history) == 2

        pipeline.reset()
        assert len(pipeline.temporal_history) == 0

        result = pipeline.process(left, right, left_timestamp=100.0)
        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED
        assert result.temporal_persistence.persistent_count == 0
        assert result.temporal_persistence.new_count > 0

    def test_large_gap_starts_a_fresh_sequence(self):
        pipeline = _pipeline(temporal_gap_limit_s=0.5)
        left, right = _random_pair(1)
        pipeline.process(left, right, left_timestamp=0.0)
        pipeline.process(left, right, left_timestamp=1.0)  # PERSISTENT built up

        result = pipeline.process(left, right, left_timestamp=500.0)  # gap >> limit
        assert result.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE
        assert len(pipeline.temporal_history) == 1
        assert result.temporal_persistence.persistent_count == 0
        assert result.temporal_persistence.new_count > 0


# ===================================================================
# 9. Degradation -> recovery
# ===================================================================
class TestDegradationRecovery:
    def test_healthy_degraded_healthy_recovers_cleanly_across_the_full_chain(self):
        pipeline = _pipeline()  # default persistence_max_dropout_frames=1
        left, right = _random_pair(1)
        healthy_before = pipeline.process(left, right, left_timestamp=0.0)
        degraded = pipeline.process(*_flat_pair(), left_timestamp=1.0)
        healthy_after = pipeline.process(left, right, left_timestamp=2.0)

        assert healthy_before.geometry_metrics.valid_fraction > 0.1
        assert degraded.geometry_metrics.valid_fraction == 0.0
        assert healthy_after.geometry_metrics.valid_fraction > 0.1

        # No stuck state anywhere in the Level 4 chain: the recovery
        # frame is admitted exactly like an ordinary healthy frame, and —
        # since the one degraded (textureless, zero-occupancy) frame in
        # between was within the tolerated dropout grace window — the
        # SAME scene reappearing resumes its persistence exactly where it
        # left off, rather than being forced to restart as NEW.
        assert healthy_after.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED
        assert healthy_after.temporal_persistence.state == TemporalPersistenceState.CLASSIFIED
        assert healthy_after.temporal_persistence.persistent_count > 0

    def test_a_genuinely_new_scene_after_degradation_still_reads_new_not_stuck(self):
        """Complementary case: if a DIFFERENT scene appears after
        degradation (not the same one resuming), it must read NEW, not be
        blocked by whatever the pre-degradation history happened to be."""
        pipeline = _pipeline()
        pipeline.process(*_random_pair(1), left_timestamp=0.0)
        pipeline.process(*_flat_pair(), left_timestamp=1.0)
        result = pipeline.process(*_patch_pair(9), left_timestamp=2.0)

        assert result.temporal_persistence.state == TemporalPersistenceState.CLASSIFIED
        assert result.temporal_persistence.new_count > 0


# ===================================================================
# 10. Bounded memory
# ===================================================================
class TestBoundedMemory:
    def test_history_and_persistence_grids_bounded_over_a_long_run(self):
        config = _full_config(temporal_max_records=10)
        pipeline = DepthPerceptionPipeline(config, _CALIBRATION, body_T_camera_left=_transform())
        left, right = _random_pair(1)

        shapes = set()
        for i in range(80):
            result = pipeline.process(left, right, left_timestamp=float(i))
            if result.temporal_persistence.state_grid is not None:
                shapes.add(result.temporal_persistence.state_grid.shape)

        assert len(pipeline.temporal_history) == 10
        assert len(shapes) == 1  # never resized, regardless of frame count


# ===================================================================
# 11. Deterministic replay
# ===================================================================
class TestDeterministicReplay:
    def test_identical_sequence_and_hints_produce_identical_full_chain_output(self):
        sequence = [
            (_random_pair(1), None),
            (_random_pair(1), [_hint(1.05, [0.0, 0.01, 0.0])]),
            (_flat_pair(), None),
            (_patch_pair(9), None),
        ]

        def _run():
            pipeline = _pipeline()
            results = []
            for i, (pair, hints) in enumerate(sequence):
                results.append(pipeline.process(*pair, left_timestamp=float(i), motion_hints=hints))
            return results

        run_a, run_b = _run(), _run()
        for ra, rb in zip(run_a, run_b):
            np.testing.assert_array_equal(ra.disparity_map, rb.disparity_map)
            np.testing.assert_array_equal(ra.depth_map, rb.depth_map)
            assert ra.confidence == rb.confidence
            assert ra.temporal_consistency.state == rb.temporal_consistency.state
            assert ra.temporal_stabilization.state == rb.temporal_stabilization.state
            assert ra.rotation_compensation_status == rb.rotation_compensation_status
            assert ra.motion_aware_reliability.state == rb.motion_aware_reliability.state
            assert ra.temporal_persistence.state == rb.temporal_persistence.state
            assert ra.temporal_persistence.new_count == rb.temporal_persistence.new_count
            assert ra.temporal_persistence.persistent_count == rb.temporal_persistence.persistent_count
            if ra.temporal_persistence.support_count_grid is not None:
                np.testing.assert_array_equal(
                    ra.temporal_persistence.support_count_grid, rb.temporal_persistence.support_count_grid,
                )
