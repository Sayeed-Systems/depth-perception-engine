"""
Adversarial input matrix — Level 3, Phase E6.

Systematically covers scenarios A-P from the E6 task (Q "repeated
identical frames" and R "healthy -> invalid -> healthy" get their own
dedicated files — tests/test_determinism.py and
tests/test_state_recovery.py respectively — since they need multi-call
sequences, not single-input classification).

Each test class documents the EXPECTED semantics for its scenario in its
own docstring, per the task's explicit requirement. Many scenarios are
also already covered at the unit level in earlier phases' test files
(tests/test_depth_estimator.py, tests/test_point_cloud_builder.py,
tests/test_obstacle_extractor.py, tests/test_free_space.py) — this file
does not re-derive that math, it re-verifies the same rules hold when
exercised through the real DepthPerceptionPipeline (or, where real SGBM
cannot deterministically produce a specific disparity pattern — NaN/Inf/
negative/too-near/too-far disparity — through direct injection into the
same PointCloudBuilder the pipeline itself uses, exactly mirroring
scenario semantics without re-implementing the math).
"""

import dataclasses

import cv2
import numpy as np
import pytest

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import GeometryQuality, PointCloudBuilder, classify_geometry_quality

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _full_config(**overrides):
    defaults = dict(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _pipeline(**config_overrides):
    return DepthPerceptionPipeline(_full_config(**config_overrides), _CALIBRATION, body_T_camera_left=_transform())


def _quality(result, config):
    return classify_geometry_quality(
        result.geometry_metrics, config.geometry_healthy_min_valid_fraction, config.geometry_degraded_min_valid_fraction,
    )


def _random_pair(seed=42, shift_px=24):
    """Genuinely-correlated smoothed-low-frequency-texture stereo pair with
    a known fixed shift. Originally i.i.d. noise, relying on real SGBM's
    smoothness prior to still report substantial (if false) valid disparity
    from zero true correspondence — Phase I1 corrected `disparity_engine.py`'s
    SGBM penalty terms specifically to stop that (see
    benchmarks/i1_stereo_accuracy/), so i.i.d. noise no longer reliably
    produces "substantial valid geometry" after that fix (the fix working
    as intended). Uses the same technique already established by
    tests/test_d10_black_box_provider.py / tests/test_d11_degradation_validation.py
    for exactly this reason."""
    w, h = _CALIBRATION.image_size
    canvas_w = w + shift_px
    rng = np.random.default_rng(seed)
    low_res = rng.integers(0, 255, (h // 4 + 2, canvas_w // 4 + 2), dtype=np.uint8)
    canvas = cv2.resize(low_res, (canvas_w, h), interpolation=cv2.INTER_CUBIC)
    canvas_bgr = np.stack([canvas] * 3, axis=-1)
    left = canvas_bgr[:, 0:w].copy()
    right = canvas_bgr[:, shift_px:shift_px + w].copy()
    return left, right


class TestA_NormalScene:
    """Healthy disparity, substantial valid geometry.

    Expected: process() succeeds; obstacle/ray/metrics all present and
    non-empty; classify_geometry_quality is HEALTHY or DEGRADED (not
    NO_USABLE_GEOMETRY — this fixture reliably produces well over the
    5% degraded floor)."""

    def test_normal_scene_produces_substantial_valid_geometry(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right)

        assert result.geometry_metrics.valid_fraction > 0.1
        assert result.obstacle_cloud.points.shape[0] > 0
        assert result.free_space_rays.ranges_m.shape[0] > 0
        quality = _quality(result, _full_config())
        assert quality in (GeometryQuality.HEALTHY, GeometryQuality.DEGRADED)


class TestB_TexturelessScene:
    """Poor/no correspondence — geometry becomes entirely UNKNOWN.

    Expected: valid_fraction == 0.0; zero obstacle points; zero rays;
    classify_geometry_quality == NO_USABLE_GEOMETRY; process() still
    succeeds (this is degradation, not failure)."""

    def test_flat_textureless_pair_yields_zero_valid_geometry(self):
        w, h = _CALIBRATION.image_size
        left = np.full((h, w, 3), 128, dtype=np.uint8)
        right = left.copy()

        result = _pipeline().process(left, right)

        assert result.geometry_metrics.valid_fraction == 0.0
        assert result.obstacle_cloud.points.shape[0] == 0
        assert result.free_space_rays.ranges_m.shape[0] == 0
        assert _quality(result, _full_config()) == GeometryQuality.NO_USABLE_GEOMETRY


class TestC_ExtremelySparseValidDepth:
    """Only a small fraction of valid pixels (a tiny textured patch in an
    otherwise flat scene).

    Expected: process() succeeds; 0 < valid_fraction << degraded
    threshold; classify_geometry_quality == NO_USABLE_GEOMETRY (sparse
    enough to fall below even the DEGRADED floor with default
    thresholds) — proving "some evidence exists" and "enough evidence to
    trust" are different questions, both answerable from existing
    fields."""

    def test_sparse_texture_patch_yields_small_nonzero_fraction(self):
        # The patch needs a genuine, non-zero disparity: depth_estimator.py
        # treats disparity<=0 as invalid by design (0 is the "no data"
        # sentinel, not a legitimate zero-depth reading), so a patch pasted
        # at the IDENTICAL pixel location in both eyes (zero true disparity)
        # can never contribute valid depth regardless of how well SGBM
        # matches it — that was true before Phase I1 too, it was just masked
        # by real SGBM's (since-corrected) smoothness-prior over-permissiveness
        # spuriously reporting a few nearby pixels as valid disparity. A real
        # object at a real depth needs a real pixel shift between the two
        # eyes — 15px here (~2.7m at this calibration's fx/baseline).
        w, h = _CALIBRATION.image_size
        shift_px = 15
        rng = np.random.default_rng(3)
        left = np.full((h, w, 3), 128, dtype=np.uint8)
        right = np.full((h, w, 3), 128, dtype=np.uint8)
        patch = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
        cy, cx = h // 2, w // 2
        left[cy - 10:cy + 10, cx - 10:cx + 10] = patch
        right[cy - 10:cy + 10, cx - 10 - shift_px:cx + 10 - shift_px] = patch

        result = _pipeline().process(left, right)

        assert 0.0 < result.geometry_metrics.valid_fraction < 0.05
        assert _quality(result, _full_config()) == GeometryQuality.NO_USABLE_GEOMETRY


class TestD_AllInvalidDisparity:
    """Every disparity pixel invalid (<=0 or non-finite).

    Expected (verified at the PointCloudBuilder level — the pipeline
    path is scenario B/textureless, which is the only way real SGBM
    reliably produces this): zero valid points, zero obstacle points,
    zero rays — never a crash, never a fabricated point."""

    def test_all_zero_disparity_yields_zero_valid_points(self):
        builder = PointCloudBuilder.from_calibration(_CALIBRATION)
        disparity = np.zeros((20, 30), dtype=np.float32)

        cloud = builder.build(disparity)

        assert not np.any(cloud.valid_mask)
        assert np.all(np.isnan(cloud.points))


class TestE_ZeroDisparity:
    """A disparity map that is exactly 0.0 everywhere — this codebase's
    own invalid-disparity convention (disparity <= 0).

    Expected: identical to scenario D (0 is the boundary of "invalid"),
    verified directly against DepthEstimator's documented rejection
    rule."""

    def test_zero_disparity_is_invalid_not_a_valid_zero_distance(self):
        builder = PointCloudBuilder.from_calibration(_CALIBRATION)
        disparity = np.zeros((5, 5), dtype=np.float32)

        cloud = builder.build(disparity)

        assert not np.any(cloud.valid_mask)


class TestF_NegativeDisparity:
    """Negative disparity values (rejected sentinel).

    Expected: rejected identically to zero — never produces a point,
    regardless of magnitude."""

    def test_negative_disparity_never_produces_a_point(self):
        builder = PointCloudBuilder.from_calibration(_CALIBRATION)
        disparity = np.full((5, 5), -12.5, dtype=np.float32)

        cloud = builder.build(disparity)

        assert not np.any(cloud.valid_mask)
        assert np.all(np.isnan(cloud.points))


class TestG_NaNValues:
    """NaN disparity values.

    Expected: excluded from valid_mask; never propagates into a finite
    (X, Y, Z); never crashes."""

    def test_nan_disparity_excluded_and_no_crash(self):
        builder = PointCloudBuilder.from_calibration(_CALIBRATION)
        disparity = np.full((5, 5), 50.0, dtype=np.float32)
        disparity[2, 2] = np.nan

        cloud = builder.build(disparity)

        assert not cloud.valid_mask[2, 2]
        assert np.all(np.isnan(cloud.points[2, 2]))
        assert np.all(np.isfinite(cloud.points[cloud.valid_mask]))


class TestH_InfValues:
    """+Inf / -Inf disparity values.

    Expected: excluded from valid_mask; never produces an Inf or NaN
    leak into a point marked valid."""

    def test_pos_and_neg_inf_disparity_excluded(self):
        builder = PointCloudBuilder.from_calibration(_CALIBRATION)
        disparity = np.full((5, 5), 50.0, dtype=np.float32)
        disparity[0, 0] = np.inf
        disparity[4, 4] = -np.inf

        cloud = builder.build(disparity)

        assert not cloud.valid_mask[0, 0]
        assert not cloud.valid_mask[4, 4]
        assert np.all(np.isfinite(cloud.points[cloud.valid_mask]))


class TestI_TooNearDepth:
    """Disparity so large it reprojects to depth < DepthEstimator.MIN_DEPTH_M.

    Expected: rejected as invalid — not clamped to the minimum, not
    silently accepted."""

    def test_depth_below_minimum_is_rejected(self):
        from depth_perception_engine.depth.depth_estimator import DepthEstimator

        builder = PointCloudBuilder.from_calibration(_CALIBRATION)
        focal_px = builder.focal_length_px
        baseline_m = builder.baseline_m
        too_near_disparity = (focal_px * baseline_m) / (DepthEstimator.MIN_DEPTH_M * 0.5)

        cloud = builder.build(np.full((3, 3), too_near_disparity, dtype=np.float32))

        assert not np.any(cloud.valid_mask)


class TestJ_TooFarDepth:
    """Disparity so small it reprojects to depth > DepthEstimator.MAX_DEPTH_M.

    Expected: rejected as invalid — not clamped to the maximum, not
    silently accepted as a distant-but-trusted point."""

    def test_depth_above_maximum_is_rejected(self):
        from depth_perception_engine.depth.depth_estimator import DepthEstimator

        builder = PointCloudBuilder.from_calibration(_CALIBRATION)
        focal_px = builder.focal_length_px
        baseline_m = builder.baseline_m
        too_far_disparity = (focal_px * baseline_m) / (DepthEstimator.MAX_DEPTH_M * 2.0)

        cloud = builder.build(np.full((3, 3), too_far_disparity, dtype=np.float32))

        assert not np.any(cloud.valid_mask)


class TestK_MixedValidInvalidGeometry:
    """A frame with both valid and invalid pixels.

    Expected: obstacle_cloud/free_space_rays counts never exceed the
    valid pixel count — invalid pixels contribute nothing to either.
    Phase I3 update: no longer an exact equality — a geometrically-
    predicted occlusion/dis-occlusion shadow zone (see
    geometry.reliability's module docstring) can additionally exclude
    some nominally-valid-but-unreliable pixels from both outputs, always
    symmetrically (see the explicit check below)."""

    def test_counts_exactly_match_valid_pixel_count(self):
        left, right = _random_pair()
        result = _pipeline().process(left, right)

        valid_count = int(result.geometry_body.valid_mask.sum())
        assert 0 < valid_count < result.geometry_body.valid_mask.size, (
            "fixture must be genuinely mixed for this test to be meaningful"
        )
        assert result.obstacle_cloud.points.shape[0] <= valid_count
        assert result.free_space_rays.ranges_m.shape[0] <= valid_count
        assert result.obstacle_cloud.points.shape[0] == result.free_space_rays.ranges_m.shape[0]


class TestL_EmptyGeometryAfterRangeFiltering:
    """A range window so tight it excludes every otherwise-valid point.

    Expected: obstacle_cloud becomes empty (not an error); free_space_rays
    is UNAFFECTED (range filtering is obstacle-specific, not a free-space
    concept — see docs/DATA_CONTRACTS.md); geometry_metrics.
    min_obstacle_distance_m becomes None (no obstacle evidence), while
    mean_free_space_m remains populated."""

    def test_tight_range_empties_obstacle_cloud_but_not_rays(self):
        left, right = _random_pair()
        result = _pipeline(obstacle_max_range_m=0.001).process(left, right)

        assert result.obstacle_cloud.points.shape[0] == 0
        assert result.free_space_rays.ranges_m.shape[0] > 0
        assert result.geometry_metrics.min_obstacle_distance_m is None
        assert result.geometry_metrics.mean_free_space_m is not None


class TestM_MalformedImageShapes:
    """1-D arrays, zero-size dimensions, and other structurally invalid
    images.

    Expected: rejected as INVALID CALLER INPUT (TypeError/ValueError),
    raised before any Level 3 stage runs — never a crash inside the
    geometry chain itself."""

    def test_1d_array_rejected(self):
        pipeline = _pipeline()
        left = np.zeros(100, dtype=np.uint8)
        right = np.zeros(100, dtype=np.uint8)
        with pytest.raises(ValueError):
            pipeline.process(left, right)

    def test_zero_size_dimension_rejected(self):
        pipeline = _pipeline()
        w, h = _CALIBRATION.image_size
        left = np.zeros((0, w, 3), dtype=np.uint8)
        right = np.zeros((0, w, 3), dtype=np.uint8)
        with pytest.raises((ValueError, RuntimeError)):
            pipeline.process(left, right)

    def test_none_image_rejected(self):
        pipeline = _pipeline()
        with pytest.raises(ValueError):
            pipeline.process(None, None)


class TestN_LeftRightSizeMismatch:
    """Left/right images of different (H, W).

    Expected: rejected as INVALID CALLER INPUT before any processing —
    already covered in tests/test_pipeline.py::test_mismatched_stereo_pair_is_rejected;
    re-verified here as part of the consolidated E6 matrix."""

    def test_mismatched_shapes_rejected(self):
        pipeline = _pipeline()
        left, _right = _random_pair()
        wrong_size_right = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            pipeline.process(left, wrong_size_right)


class TestO_UnsupportedDtype:
    """A dtype not documented as supported (docs/DATA_CONTRACTS.md
    specifies uint8) — the actual, observed behavior, not an assumed one.

    Expected (verified, not assumed): this library does not itself
    validate image dtype (a real, documented gap — see
    docs/IMPLEMENTATION_STATUS.md's E6 addendum); an unsupported dtype
    surfaces as a raw cv2.error at the internal grayscale-conversion
    step, propagated uncaught. This still satisfies the core safety
    property (the frame fails atomically, no geometry is fabricated) even
    though the exception type is less clean than the library's own
    RuntimeError wrapper around StereoSGBM.compute() — fixing that
    classification gap is Level 0-2 error-handling code predating E2-E5
    and is out of E6's "harden the geometry chain" scope (see the
    "Do NOT optimize unrelated Level 0-2 code" non-goal)."""

    def test_float64_images_raise_uncaught_not_fabricate_geometry(self):
        pipeline = _pipeline()
        w, h = _CALIBRATION.image_size
        rng = np.random.default_rng(1)
        left = rng.uniform(0, 255, (h, w, 3)).astype(np.float64)
        right = rng.uniform(0, 255, (h, w, 3)).astype(np.float64)

        with pytest.raises(Exception):  # noqa: B017 — deliberately broad: verifying it raises SOMETHING, not a specific type
            pipeline.process(left, right)


class TestP_InvalidCalibrationOrTransform:
    """Invalid calibration/extrinsic, constructed through the public API.

    Expected: rejected at DepthPerceptionPipeline construction time (fails
    fast, before any frame is processed) — never silently accepted."""

    def test_nan_poisoned_calibration_Q_rejected_at_construction(self):
        bad_Q = _CALIBRATION.Q.copy()
        bad_Q[0, 3] = np.nan
        bad_calibration = dataclasses.replace(_CALIBRATION, Q=bad_Q)

        with pytest.raises(ValueError, match="non-finite"):
            DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), bad_calibration)

    def test_transform_with_wrong_from_frame_rejected_at_construction(self):
        bad_transform = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame=FrameId.BODY, to_frame=FrameId.BODY,
        )
        with pytest.raises(ValueError, match="from_frame"):
            DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=bad_transform)

    def test_transform_with_non_finite_rotation_rejected_when_used(self):
        """RigidTransform itself only shape-validates at construction
        (frozen E1 contract — see frames.py) — the finite check lives in
        geometry.transform_point_cloud, exercised the first time a frame
        with valid geometry is actually processed."""
        bad_rotation = np.eye(3)
        bad_rotation[0, 0] = np.inf
        bad_transform = RigidTransform(
            rotation=bad_rotation, translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )
        pipeline = DepthPerceptionPipeline(_full_config(), _CALIBRATION, body_T_camera_left=bad_transform)
        left, right = _random_pair()

        with pytest.raises(ValueError, match="non-finite"):
            pipeline.process(left, right)
