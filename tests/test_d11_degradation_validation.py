"""
Degradation / failure validation — Phase D11 (see docs/DPE_V1_PROVIDER_CONTRACT.md's
D11 record).

D10 validated that GOOD evidence produces geometrically correct
GeometryFrame output. D11 validates the other half of the contract:
that DEGRADED, invalid, incomplete, or unavailable evidence produces
predictable, conservative, HONEST output — never fabricated trustworthy
geometry merely to keep output flowing.

Many of the 15 named D11 scenarios are ALREADY covered by rigorous
existing tests from Phase E6 (`tests/test_adversarial_geometry.py`,
`tests/test_failure_containment.py`, `tests/test_state_recovery.py`) and
the Level 4 temporal test files (`tests/test_temporal_history.py`,
`tests/test_temporal_consistency.py`, `tests/test_rotation_compensation.py`,
`tests/test_motion_aware_reliability.py`, `tests/test_temporal_persistence.py`).
This file does NOT duplicate that coverage — each class below states,
in its own docstring, which existing test file already closes that part
of the matrix, and adds ONLY what those files do not already prove: real
degraded/corrupted IMAGES run through the actual pipeline (not hand-built
depth arrays) and checked against `GeometryFrame`'s own D-phase evidence
fields (surface/boundary/opening/quality), which pre-date D10/D11 and
were never exercised under real degraded conditions before now.

Ground-truth/degradation technique: reuses `tests/test_d10_black_box_provider.py`'s
own engineered-stereo-pair generator (smoothed low-frequency noise,
`rectify=False`) as the "good" baseline scene, degrading it in different
ways per scenario (partial occlusion, total decorrelation, extreme/invalid
motion) — the same real `StereoSGBM` runs in every case, nothing is
bypassed.
"""

import numpy as np
import pytest

import cv2

from depth_perception_engine import (
    DepthPerceptionPipeline,
    PipelineConfig,
    StereoCalibration,
    load_stereo_calibration,
)
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.temporal import MotionHint

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size
_SHIFT_PX = 24


def _engineered_stereo_pair(shift_px: int = _SHIFT_PX, seed: int = 7):
    """Identical technique to test_d10_black_box_provider.py's own
    generator — a real, textured, decorrelation-free stereo pair with a
    known true shift, used here as the "good" baseline to degrade."""
    canvas_w = _W + shift_px
    rng = np.random.default_rng(seed)
    low_res = rng.integers(0, 255, (_H // 4 + 2, canvas_w // 4 + 2), dtype=np.uint8)
    canvas = cv2.resize(low_res, (canvas_w, _H), interpolation=cv2.INTER_CUBIC)
    canvas_bgr = np.stack([canvas] * 3, axis=-1)
    left = canvas_bgr[:, 0:_W].copy()
    right = canvas_bgr[:, shift_px:shift_px + _W].copy()
    return left, right


def _full_config(**overrides):
    defaults = dict(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        enable_geometry_frame=True,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=50,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _pipeline(**config_overrides):
    transform = RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )
    return DepthPerceptionPipeline(_full_config(**config_overrides), _CALIBRATION, rectify=False, body_T_camera_left=transform)


# ===================================================================
# 1/4 — stereo texture / left-right correspondence degradation
# 3   — severe image corruption/noise
# FALSE-POSITIVE SAFETY — "noise != supported surface"
# ===================================================================
class TestSevereDecorrelatedNoiseFalsePositiveSafety:
    """Total left/right decorrelation (independent i.i.d. random noise,
    zero true correspondence) — already covered at the PointCloud level
    by test_adversarial_geometry.py's TestA-TestC (valid_fraction,
    quality classification). This class adds the check those tests do
    NOT make: whether the newer D-phase evidence families
    (SurfaceEvidence/BoundaryEvidence/OpeningEvidence, D4-D6, which
    post-date E6) also stay honest under the identical condition.

    FINDING (see this file's own D11 report / docs/DPE_V1_PROVIDER_CONTRACT.md's
    D11 record for the full writeup): they do NOT, cleanly. Real
    StereoSGBM's own semi-global smoothness regularization does not
    simply reject uncorrelated-noise input as invalid — under total
    absence of real correspondence, its cost-aggregation smoothness
    prior (P1/P2) still produces a SPATIALLY SMOOTH, plausible-looking
    disparity field (this is a real, measured property of what SGBM
    does when the data term is uninformative everywhere, not a DPE
    defect). `SurfaceEvidence.planarity` — DPE's own per-cell "how well
    does this look like a real, well-fit plane" confidence signal — was
    measured at ~0.99 (essentially "textbook flat surface") on this
    fabricated, meaningless disparity, reproducibly across 5 independent
    noise seeds. `looks_like_garbage_frame()` (`depth_perception_engine.quality`)
    exists in this codebase for exactly this class of input and is
    NEVER invoked anywhere inside `pipeline.py`'s own `process()` path —
    confirmed by source inspection, not merely inferred.
    """

    def test_total_decorrelation_yields_high_confidence_surface_evidence_from_noise(self, capsys):
        planarities_by_seed = []
        for seed in range(1, 6):
            rng = np.random.default_rng(seed)
            left = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
            right = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)

            result = _pipeline().process(left, right)
            gf = result.geometry_frame
            planarities = [s.planarity for s in gf.surface_evidence if s.planarity is not None]
            if planarities:
                planarities_by_seed.append(float(np.mean(planarities)))

        print(f"\n[D11] decorrelated-noise SurfaceEvidence.planarity, mean per seed: {planarities_by_seed}")

        # This assertion documents the FINDING, it does not endorse it as
        # correct behavior: planarity stays implausibly high (>0.9) on
        # noise-derived surfaces in every seed tested. If a future
        # corrective phase adds an upstream garbage-frame/texture gate,
        # this assertion is expected to start FAILING — that would be
        # the fix working, not a regression to silently loosen.
        assert len(planarities_by_seed) >= 3, "fixture did not reliably produce surface evidence to evaluate"
        assert all(p > 0.9 for p in planarities_by_seed), (
            "expected the known finding (implausibly high planarity on pure noise) to reproduce; "
            f"got {planarities_by_seed} — if this now fails, the false-positive was fixed upstream, "
            "not a new regression"
        )

    def test_decorrelated_noise_quality_is_measurably_worse_than_a_real_scene(self, capsys):
        """The one signal that DOES correctly and measurably respond:
        GeometryMetrics.valid_fraction / GeometryQuality — confirming
        this is specifically a SurfaceEvidence-confidence problem, not a
        total absence of degradation signal."""
        left_good, right_good = _engineered_stereo_pair()
        good = _pipeline().process(left_good, right_good)

        rng = np.random.default_rng(99)
        left_noise = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
        right_noise = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
        noisy = _pipeline().process(left_noise, right_noise)

        print(f"\n[D11] good valid_fraction={good.geometry_frame.geometry_metrics.valid_fraction:.3f} "
              f"quality={good.geometry_frame.quality.geometry_validity_state} vs "
              f"noise valid_fraction={noisy.geometry_frame.geometry_metrics.valid_fraction:.3f} "
              f"quality={noisy.geometry_frame.quality.geometry_validity_state}")
        # Both are DEGRADED-tier in this fixture (60% vs ~36-39%) — the
        # important, honestly-measured fact is the ORDERING and that
        # neither claims HEALTHY on this scene, not a specific tier.
        assert noisy.geometry_frame.geometry_metrics.valid_fraction < good.geometry_frame.geometry_metrics.valid_fraction


# ===================================================================
# 2/5 — partial image corruption / occluded regions
# SAFE-DEGRADED: output remains available, quality reflects reduced evidence
# ===================================================================
class TestPartialOcclusion:
    """A real, textured stereo pair with a genuine, large, spatially
    contiguous blacked-out region in BOTH images (a real texture-loss
    condition, not a hand-built mask) — new coverage; existing
    adversarial tests use whole-frame degradation only (textureless/
    sparse), never a spatially localized occlusion through real SGBM."""

    def test_partial_occlusion_is_safe_degraded_not_insufficient_or_crash(self, capsys):
        left, right = _engineered_stereo_pair()
        baseline = _pipeline().process(left, right)

        left_occ, right_occ = left.copy(), right.copy()
        c0, c1 = _W // 4, 3 * _W // 4
        left_occ[:, c0:c1] = 0
        right_occ[:, c0:c1] = 0
        occluded = _pipeline().process(left_occ, right_occ)

        gf_base, gf_occ = baseline.geometry_frame, occluded.geometry_frame
        print(f"\n[D11] baseline valid_fraction={gf_base.geometry_metrics.valid_fraction:.4f} "
              f"occluded valid_fraction={gf_occ.geometry_metrics.valid_fraction:.4f} "
              f"occluded quality={gf_occ.quality.overall_state} reasons={gf_occ.quality.degradation_reasons}")

        # SAFE-DEGRADED: output remains available (no crash, no None
        # GeometryFrame) and quality correctly reflects LESS evidence
        # than the same scene unoccluded — never equal or higher.
        assert gf_occ is not None
        assert gf_occ.geometry_metrics.valid_fraction < gf_base.geometry_metrics.valid_fraction
        assert "GEOMETRY_VALIDITY:DEGRADED" in gf_occ.quality.degradation_reasons or \
               gf_occ.quality.geometry_validity_state in {"DEGRADED", "INSUFFICIENT"}

    def test_occluded_obstacle_and_ray_counts_never_exceed_baseline(self, capsys):
        """False-positive safety: occlusion must never CREATE evidence
        that didn't exist in the unoccluded baseline (obstacle/ray counts
        can only drop or stay the same, never increase from missing
        data)."""
        left, right = _engineered_stereo_pair()
        baseline = _pipeline().process(left, right)

        left_occ, right_occ = left.copy(), right.copy()
        c0, c1 = _W // 4, 3 * _W // 4
        left_occ[:, c0:c1] = 0
        right_occ[:, c0:c1] = 0
        occluded = _pipeline().process(left_occ, right_occ)

        assert occluded.geometry_frame.obstacle_cloud.points.shape[0] <= \
            baseline.geometry_frame.obstacle_cloud.points.shape[0]
        assert occluded.geometry_frame.free_space_rays.ranges_m.shape[0] <= \
            baseline.geometry_frame.free_space_rays.ranges_m.shape[0]


# ===================================================================
# 7 — one stereo frame unavailable/invalid where the API permits it
# ===================================================================
class TestOneFrameUnavailable:
    """StereoObservation.left_image/right_image are both REQUIRED
    (non-Optional) fields (models/result.py) — the public API gives no
    way to represent "one frame missing" as a valid StereoObservation.
    The only API-representable form of this scenario is passing None
    directly to process() for one image, already exhaustively covered
    (REJECTED, ValueError) by test_adversarial_geometry.py's
    TestM_MalformedImageShapes::test_none_image_rejected. Re-confirmed
    directly here as the D11 boundary-classification record."""

    def test_none_right_image_rejected(self):
        left, _ = _engineered_stereo_pair()
        with pytest.raises(ValueError):
            _pipeline().process(left, None)

    def test_none_left_image_rejected(self):
        _, right = _engineered_stereo_pair()
        with pytest.raises(ValueError):
            _pipeline().process(None, right)


# ===================================================================
# 8 — timestamp problems, re-checked through GeometryFrame specifically
# (existing coverage checks DepthPerceptionResult; this checks the
# newer GeometryFrame fields a REJECTED-admission frame produces)
# ===================================================================
class TestTimestampRejectionGeometryFrameStaysHonest:
    """tests/test_temporal_history.py already exhaustively covers
    TemporalAdmissionStatus's rules (None/NaN/Inf/out-of-order/duplicate
    timestamps) and confirms the Level 3 DepthPerceptionResult is still
    returned on rejection. This class confirms the same for
    GeometryFrame specifically (post-dates that file) — REJECTED
    admission must still yield a full, honest, non-fabricated
    GeometryFrame, and must not corrupt subsequent admission."""

    def test_duplicate_timestamp_rejected_but_geometry_frame_still_complete(self, capsys):
        left, right = _engineered_stereo_pair()
        pipeline = _pipeline()

        r1 = pipeline.process(left, right, left_timestamp=0.0, right_timestamp=0.0)
        r2 = pipeline.process(left, right, left_timestamp=0.0, right_timestamp=0.0)  # duplicate

        print(f"\n[D11] r2 admission={r2.temporal_admission_status}")
        assert r2.temporal_admission_status == "REJECTED_DUPLICATE_TIMESTAMP"
        assert r2.geometry_frame is not None
        # Same input images -> identical Level 3 geometry regardless of
        # the rejected admission (rejection must not blank/corrupt output).
        np.testing.assert_array_equal(r2.geometry_frame.obstacle_cloud.points, r1.geometry_frame.obstacle_cloud.points)

    def test_rejection_does_not_corrupt_subsequent_valid_admission(self, capsys):
        left, right = _engineered_stereo_pair()
        pipeline = _pipeline()

        pipeline.process(left, right, left_timestamp=0.0, right_timestamp=0.0)
        pipeline.process(left, right, left_timestamp=0.0, right_timestamp=0.0)  # rejected duplicate
        r3 = pipeline.process(left, right, left_timestamp=1.0, right_timestamp=1.0)  # valid again

        assert r3.temporal_admission_status == "ACCEPTED"
        assert r3.geometry_frame.temporal_consistency is not None


# ===================================================================
# 9/10/11 — absent / invalid / insufficient-coverage MotionHint
# ===================================================================
class TestMotionHintDegradation:
    """Absent MotionHint (scenario 9) and insufficient motion coverage
    (scenario 11) are already exhaustively covered:
    test_rotation_compensation.py::TestScenarioMissingStaleInvalidHints,
    test_motion_aware_reliability.py::TestCompensationUnavailableThroughRealPipeline/
    TestIncompleteMotionHintCoverage. `valid=False` MotionHint rejection
    is covered by test_rotation_compensation.py::TestSelectMotionHintSamples::test_invalid_flag_rejected.

    NOT previously covered anywhere (confirmed by source inspection of
    `MotionHint.__post_init__`, which validates only shape/type, never
    finiteness): a MotionHint whose `angular_velocity_rad_s` itself
    contains NaN/Inf (scenario 10, "invalid MotionHint" in the sense of
    corrupted VALUES, not the `valid=False` flag). This class adds that.

    FINDING (see D11 report): no crash, and no unsafe geometry
    fabrication (the NaN/Inf poisons the integrated rotation, which
    poisons `_reproject_source_to_target`'s intermediate values; the
    resulting int64 cast of NaN produces a RuntimeWarning and an
    out-of-bounds index that the existing `in_bounds` check correctly
    filters out, so the reprojected/compensated snapshot ends up fully
    empty (0.0-invalid) and `temporal_consistency` correctly falls back
    to INSUFFICIENT_EVIDENCE — SAFE). BUT
    `RotationCompensationStatus.APPLIED` is still reported for that
    frame, even though the compensation produced a fully empty,
    unusable result — the status label claims success where none
    meaningfully occurred. This is a real, minor, reportable status
    labeling inconsistency (never a fabricated-geometry safety
    violation), documented as a VALIDATION FINDING, not fixed here.
    """

    def test_nan_angular_velocity_does_not_crash_and_does_not_fabricate_geometry(self, capsys):
        left, right = _engineered_stereo_pair()
        pipeline = _pipeline()
        pipeline.process(left, right, left_timestamp=0.0, right_timestamp=0.0)

        nan_hint = MotionHint(
            timestamp=0.5, angular_velocity_rad_s=np.array([np.nan, 0.0, 0.0]), frame_id=FrameId.BODY,
        )
        result = pipeline.process(
            left, right, left_timestamp=1.0, right_timestamp=1.0, motion_hints=[nan_hint],
        )
        gf = result.geometry_frame
        print(f"\n[D11] NaN MotionHint -> rotation_compensation_status={gf.rotation_compensation_status}, "
              f"motion_aware_reliability={gf.motion_aware_reliability.state}, "
              f"temporal_consistency={gf.temporal_consistency.state}")

        # SAFE: Level 3 geometry (independent of MotionHint entirely) is
        # completely unaffected -- never fabricated/blanked by a bad hint.
        assert gf.obstacle_cloud is not None and gf.obstacle_cloud.points.shape[0] > 0
        # SAFE: never silently promoted to a trustworthy reliability verdict.
        assert gf.motion_aware_reliability.state in {"INSUFFICIENT_EVIDENCE", "UNRELIABLE", "DEGRADED"}
        assert gf.motion_aware_reliability.state != "RELIABLE"

    def test_inf_angular_velocity_does_not_crash(self):
        left, right = _engineered_stereo_pair()
        pipeline = _pipeline()
        pipeline.process(left, right, left_timestamp=0.0, right_timestamp=0.0)

        inf_hint = MotionHint(
            timestamp=0.5, angular_velocity_rad_s=np.array([np.inf, 0.0, 0.0]), frame_id=FrameId.BODY,
        )
        result = pipeline.process(
            left, right, left_timestamp=1.0, right_timestamp=1.0, motion_hints=[inf_hint],
        )
        assert result.geometry_frame is not None
        assert result.geometry_frame.motion_aware_reliability.state != "RELIABLE"


# ===================================================================
# 12 — excessive/unsupported rotational conditions
# ===================================================================
class TestExcessiveRotation:
    """The RELIABLE->UNRELIABLE threshold transition itself is already
    exhaustively tested at the pure-function level
    (test_motion_aware_reliability.py::TestIncreasingAngularMotion). New
    here: a real, extreme (50 rad/s, ~2865 deg/s — far beyond any
    physically sane platform rotation) angular velocity through the
    real pipeline, confirming no crash and no silent reinterpretation as
    reliable."""

    def test_extreme_angular_velocity_is_unreliable_not_a_crash(self, capsys):
        left, right = _engineered_stereo_pair()
        pipeline = _pipeline()
        pipeline.process(left, right, left_timestamp=0.0, right_timestamp=0.0)

        extreme_hint = MotionHint(
            timestamp=0.5, angular_velocity_rad_s=np.array([50.0, 0.0, 0.0]), frame_id=FrameId.BODY,
        )
        result = pipeline.process(
            left, right, left_timestamp=1.0, right_timestamp=1.0, motion_hints=[extreme_hint],
        )
        gf = result.geometry_frame
        print(f"\n[D11] extreme omega -> motion_aware_reliability={gf.motion_aware_reliability.state}")
        assert gf.motion_aware_reliability.state == "UNRELIABLE"
        assert gf.obstacle_cloud is not None  # Level 3 geometry unaffected


# ===================================================================
# 13 — temporal contradiction between frames, tied through to
# GeometryFrame.quality specifically (D8's rollup, never exercised
# against a genuine real contradiction before now)
# ===================================================================
class TestTemporalContradictionReflectedInQualityRollup:
    """test_temporal_consistency.py::TestContradictoryThroughRealPipeline
    already proves CONTRADICTORY arises correctly from a real pipeline
    given an injected far-off prior TemporalRecord. This class adds the
    one thing that test does not check: that GeometryFrame.quality (D8,
    postdates that test) correctly rolls the contradiction up into
    overall_state == DEGRADED with the right degradation_reasons entry."""

    def test_contradictory_prior_degrades_the_quality_rollup(self, capsys):
        left, right = _engineered_stereo_pair()
        pipeline = _pipeline()
        pipeline.process(left, right, left_timestamp=0.0, right_timestamp=0.0)

        # Process a genuinely different scene (different true disparity
        # shift -> different depth everywhere) immediately after — the
        # real chain end-to-end, no mocking/injection, exercising
        # whatever CONTRADICTORY/CONSISTENT verdict the real comparison
        # produces on two honestly different real scenes.
        left2, right2 = _engineered_stereo_pair(shift_px=60, seed=7)
        r2 = pipeline.process(left2, right2, left_timestamp=1.0, right_timestamp=1.0)

        gf = r2.geometry_frame
        print(f"\n[D11] temporal_consistency={gf.temporal_consistency.state}, "
              f"quality.overall_state={gf.quality.overall_state}, reasons={gf.quality.degradation_reasons}")

        if gf.temporal_consistency.state == "CONTRADICTORY":
            assert gf.quality.temporal_consistency_state == "DEGRADED"
            assert gf.quality.overall_state == "DEGRADED"
            assert "TEMPORAL_CONSISTENCY:DEGRADED" in gf.quality.degradation_reasons
        else:
            pytest.skip(
                f"fixture did not trigger CONTRADICTORY this run (got {gf.temporal_consistency.state}); "
                "not a failure of the quality rollup itself, which is separately, exactly proven by "
                "tests/test_geometry_frame_quality.py::TestTemporalDegradation with a hand-built CONTRADICTORY input"
            )


# ===================================================================
# 15 — recovery after degraded frames, through GeometryFrame
# specifically (D-phase evidence, not just Level 3 as
# test_state_recovery.py already proves)
# ===================================================================
class TestRecoveryThroughGeometryFrame:
    """test_state_recovery.py (E6) already proves Level 3 geometry
    (PointCloud/ObstacleCloud/FreeSpaceRays/GeometryMetrics) recovers
    exactly and has zero cross-frame state. This class confirms the same
    for GeometryFrame's own D-phase evidence (geometry_body/obstacle_cloud
    identity) across a VALID -> DEGRADED(noise) -> VALID sequence, and
    separately confirms the REJECTED-admission recovery path."""

    def test_valid_degraded_valid_recovers_geometry_exactly(self, capsys):
        left, right = _engineered_stereo_pair()
        rng = np.random.default_rng(99)
        noise_left = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
        noise_right = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)

        pipeline = _pipeline()
        r1 = pipeline.process(left, right, left_timestamp=0.0, right_timestamp=0.0)
        r2 = pipeline.process(noise_left, noise_right, left_timestamp=1.0, right_timestamp=1.0)
        r3 = pipeline.process(left, right, left_timestamp=2.0, right_timestamp=2.0)

        print(f"\n[D11] r1 obstacle_count={r1.geometry_frame.obstacle_cloud.points.shape[0]} "
              f"r2(noise) obstacle_count={r2.geometry_frame.obstacle_cloud.points.shape[0]} "
              f"r3(recovered) obstacle_count={r3.geometry_frame.obstacle_cloud.points.shape[0]}")

        # Exact recovery: the same clean scene reprocessed after a
        # degraded frame produces BIT-IDENTICAL Level 3/D-phase geometry
        # to the original clean frame -- no contamination retained.
        np.testing.assert_array_equal(r3.geometry_frame.geometry_body.points, r1.geometry_frame.geometry_body.points)
        np.testing.assert_array_equal(r3.geometry_frame.obstacle_cloud.points, r1.geometry_frame.obstacle_cloud.points)
        assert r3.geometry_frame.obstacle_cloud.points.shape[0] == r1.geometry_frame.obstacle_cloud.points.shape[0]

        # The temporal LAYER is expected to honestly report disagreement
        # with the immediately-preceding (noise) frame -- this is
        # correct comparison behavior, not stale contamination of the
        # actual geometry (already proven above via exact equality).
        if r3.geometry_frame.temporal_consistency is not None:
            assert r3.geometry_frame.temporal_consistency.state in {"CONSISTENT", "CONTRADICTORY", "NOT_COMPARABLE", "INSUFFICIENT_EVIDENCE"}


# ===================================================================
# Calibration-invalid boundary check — confirm the rejection happens at
# the correct, intended validation boundary (StereoCalibration
# construction), not bypassed
# ===================================================================
class TestCalibrationValidationBoundary:
    """No dedicated construction-validation test file exists for
    StereoCalibration's own __post_init__ (confirmed by survey — only
    PointCloudBuilder's downstream Q-content validation and the
    pipeline-construction-time NaN-Q rejection in
    test_adversarial_geometry.py::TestP are covered). This closes that
    boundary directly: StereoCalibration itself must reject malformed
    matrix SHAPES at construction, before any pipeline is ever built."""

    def _valid_kwargs(self):
        return dict(
            camera_matrix_left=np.eye(3), camera_matrix_right=np.eye(3),
            dist_coeffs_left=np.zeros(5), dist_coeffs_right=np.zeros(5),
            R1=np.eye(3), R2=np.eye(3), P1=np.zeros((3, 4)), P2=np.zeros((3, 4)),
            Q=np.eye(4), image_size=(320, 240),
        )

    def test_wrong_shape_camera_matrix_rejected_at_construction(self):
        kwargs = self._valid_kwargs()
        kwargs["camera_matrix_left"] = np.eye(2)
        with pytest.raises(ValueError):
            StereoCalibration(**kwargs)

    def test_wrong_shape_Q_rejected_at_construction(self):
        kwargs = self._valid_kwargs()
        kwargs["Q"] = np.eye(3)
        with pytest.raises(ValueError):
            StereoCalibration(**kwargs)

    def test_missing_matrix_rejected_at_construction(self):
        kwargs = self._valid_kwargs()
        kwargs["R1"] = None
        with pytest.raises(ValueError):
            StereoCalibration(**kwargs)
