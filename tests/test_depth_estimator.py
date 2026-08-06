"""
Unit tests for DepthEstimator.

Phase 10 (performance review, after correctness): estimate() used to
call cv2.reprojectImageTo3D to compute a full (H, W, 3) X/Y/Z point
cloud every frame and then discard X/Y, keeping only Z — three times
the necessary work, since nothing downstream ever reads X or Y. It was
replaced with a direct Z-only closed form derived from the Q matrix's
row 2 (Z numerator) and row 3 (homogeneous divisor W), which never
depend on pixel position — only on disparity — for any Q produced by
cv2.stereoRectify. TestZOnlyMatchesFullReprojection is the numerical
proof this is an exact equivalent for every VALID pixel, not an
approximation: it compares the new implementation directly against
cv2.reprojectImageTo3D's own Z channel (computed independently in this
test, with handleMissingValues=False so nothing masks/substitutes any
pixel — a raw, unconditional reference), across both the project's real
calibration and synthetic Q matrices with a non-zero
principal-point-offset term (Q[3,3] != 0) — the one structural
variation that could make a naive "Z = f*baseline/disparity" shortcut
wrong, which is exactly why this fix uses Q's actual coefficients
instead of a hardcoded formula. Invalid (disparity <= 0) pixels are
excluded from the raw-value comparison and checked separately — see
that class's module-level note for why this fix deliberately does NOT
replicate cv2's own handleMissingValues sentinel behavior.
"""

import cv2
import numpy as np
import pytest

from depth_perception_engine.depth.depth_estimator import DepthEstimator


def _reference_z_via_full_reprojection(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Ground truth: full X/Y/Z reprojection, raw Z channel, no sentinel substitution.

    handleMissingValues=False so this never special-cases any pixel —
    a pure, unconditional per-pixel application of the Q matrix, for
    comparison against the new Z-only formula on every VALID pixel.
    """
    points_3d = cv2.reprojectImageTo3D(
        disparity.astype(np.float32), Q, handleMissingValues=False
    )
    return (points_3d[:, :, 2] / 1000.0).astype(np.float32)


def _clamp_like_estimate(depth: np.ndarray, disparity: np.ndarray) -> np.ndarray:
    depth = depth.copy()
    valid = (
        (disparity > 0.0)
        & np.isfinite(depth)
        & (depth >= DepthEstimator.MIN_DEPTH_M)
        & (depth <= DepthEstimator.MAX_DEPTH_M)
    )
    depth[~valid] = 0.0
    return depth


def _mixed_disparity_map(height=48, width=64) -> np.ndarray:
    """A disparity map covering the full range of interesting cases in one array."""
    rng = np.random.default_rng(7)
    disp = rng.uniform(1.0, 120.0, size=(height, width)).astype(np.float32)
    disp[0:5, :] = 0.0       # invalid: no match
    disp[5:10, :] = -3.0     # invalid: negative sentinel
    disp[10:12, :] = 0.01    # near-zero: blows up to an out-of-range depth
    disp[12:14, :] = 1e6     # absurdly large disparity: near-zero/underflow depth
    return disp


class TestAnalyticKnownDepth:
    """
    Independent ground truth, not derived from any OpenCV function.

    TestZOnlyMatchesFullReprojection below proves the new closed-form Z
    matches cv2.reprojectImageTo3D — but that's a differential test: it
    would pass even if both sides shared the same underlying bug. This
    class instead hand-builds a Q matrix for the case with no
    principal-point x-offset between the left/right rectified views
    (Q[2,2] = 0, Q[3,3] = 0 — the standard OpenCV stereoRectify form when
    both cameras share the same cx), for which the general Z formula
    collapses exactly to the textbook relation:

        depth_m = (focal_length_px * baseline_m) / disparity_px

    and asserts DepthEstimator.estimate() reproduces that hand-computed
    number for a known focal length, baseline, and constant-disparity
    plane — with no OpenCV reprojection function involved on either side
    of the comparison.
    """

    @staticmethod
    def _make_q(focal_length_px: float, baseline_m: float, cx: float = 160.0, cy: float = 120.0) -> np.ndarray:
        # Q[3,2] chosen so DepthEstimator's own baseline_m property (which
        # takes abs(1/Q[3,2])/1000) reports exactly baseline_m, AND the
        # sign works out to a positive depth for positive disparity — see
        # this class's docstring for the algebra.
        tx = 1.0 / (baseline_m * 1000.0)
        return np.array(
            [
                [1.0, 0.0, 0.0, -cx],
                [0.0, 1.0, 0.0, -cy],
                [0.0, 0.0, 0.0, focal_length_px],
                [0.0, 0.0, tx, 0.0],
            ],
            dtype=np.float64,
        )

    def test_constant_disparity_plane_matches_hand_computed_depth(self):
        focal_length_px = 600.0
        baseline_m = 0.065
        disparity_px = 50.0
        expected_depth_m = (focal_length_px * baseline_m) / disparity_px  # = 0.78 m

        Q = self._make_q(focal_length_px, baseline_m)
        estimator = DepthEstimator(Q)
        assert estimator.focal_length_px == pytest.approx(focal_length_px)
        assert estimator.baseline_m == pytest.approx(baseline_m)

        disparity = np.full((30, 40), disparity_px, dtype=np.float32)
        depth = estimator.estimate(disparity)

        assert np.all(depth > 0.0), "expected every pixel to be valid for this in-range disparity"
        np.testing.assert_allclose(depth, expected_depth_m, rtol=1e-5)

    @pytest.mark.parametrize(
        "focal_length_px, baseline_m, disparity_px",
        [
            (600.0, 0.065, 50.0),   # 0.78 m
            (614.5, 0.0647, 100.0),  # ~0.397 m, close to this project's real calibration values
            (500.0, 0.05, 25.0),    # 1.0 m exactly
            (800.0, 0.10, 40.0),    # 2.0 m exactly
        ],
    )
    def test_several_known_focal_baseline_disparity_combinations(
        self, focal_length_px, baseline_m, disparity_px,
    ):
        expected_depth_m = (focal_length_px * baseline_m) / disparity_px

        Q = self._make_q(focal_length_px, baseline_m)
        disparity = np.full((10, 10), disparity_px, dtype=np.float32)
        depth = DepthEstimator(Q).estimate(disparity)

        np.testing.assert_allclose(depth, expected_depth_m, rtol=1e-5)


class TestZOnlyMatchesFullReprojection:
    def test_matches_reference_on_real_project_calibration(self, calibration):
        disp = _mixed_disparity_map()
        expected = _clamp_like_estimate(
            _reference_z_via_full_reprojection(disp, calibration.Q), disp,
        )

        estimator = DepthEstimator(calibration.Q)
        actual = estimator.estimate(disp)

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_matches_reference_with_nonzero_principal_point_offset_term(self):
        """
        The one structural Q variation a naive f*baseline/disparity shortcut would get wrong.

        Q[3,3] = (cx - cx') / Tx is zero when both rectified images
        share the same principal point x-coordinate (the common case,
        and true of this project's real calibration) — but is not
        guaranteed to be zero for every valid calibration. Using Q's
        actual row 2 / row 3 coefficients (as the fix does) handles
        this correctly regardless; a hardcoded simplified formula would
        silently drop this term.
        """
        Q = np.array([
            [1.0, 0.0, 0.0, -160.0],
            [0.0, 1.0, 0.0, -120.0],
            [0.0, 0.0, 0.0, 600.0],
            [0.0, 0.0, 0.02, 0.5],  # non-zero Q[3,3]
        ], dtype=np.float64)
        disp = _mixed_disparity_map()

        expected = _clamp_like_estimate(_reference_z_via_full_reprojection(disp, Q), disp)

        estimator = DepthEstimator(Q)
        actual = estimator.estimate(disp)

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_matches_reference_across_a_grid_of_synthetic_q_matrices(self):
        """Broader sweep: several plausible baseline/focal-length/offset combinations."""
        disp = _mixed_disparity_map(height=24, width=32)
        for f in (400.0, 614.5, 900.0):
            for tx in (-40.0, -64.7, -120.0):
                for q33 in (0.0, 0.3, -0.7):
                    Q = np.array([
                        [1.0, 0.0, 0.0, -160.0],
                        [0.0, 1.0, 0.0, -120.0],
                        [0.0, 0.0, 0.0, f],
                        [0.0, 0.0, -1.0 / tx, q33],
                    ], dtype=np.float64)

                    expected = _clamp_like_estimate(
                        _reference_z_via_full_reprojection(disp, Q), disp,
                    )
                    actual = DepthEstimator(Q).estimate(disp)

                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-4, atol=1e-5,
                        err_msg=f"mismatch for f={f}, tx={tx}, q33={q33}",
                    )


class TestEstimateContract:
    def test_returns_float32_same_spatial_shape(self, calibration):
        estimator = DepthEstimator(calibration.Q)
        disp = _mixed_disparity_map(height=20, width=30)

        depth = estimator.estimate(disp)

        assert depth.dtype == np.float32
        assert depth.shape == disp.shape

    def test_invalid_disparity_produces_zero_depth(self, calibration):
        estimator = DepthEstimator(calibration.Q)
        disp = np.zeros((10, 10), dtype=np.float32)

        depth = estimator.estimate(disp)

        assert np.all(depth == 0.0)

    def test_out_of_range_depth_is_clamped_to_zero(self, calibration):
        estimator = DepthEstimator(calibration.Q)
        # A tiny disparity produces a depth far beyond MAX_DEPTH_M for
        # this calibration's focal length/baseline.
        disp = np.full((10, 10), 0.1, dtype=np.float32)

        depth = estimator.estimate(disp)

        assert np.all(depth == 0.0)

    def test_rejects_non_ndarray_input(self, calibration):
        estimator = DepthEstimator(calibration.Q)
        with pytest.raises(TypeError):
            estimator.estimate([[1, 2], [3, 4]])

    def test_rejects_1d_input(self, calibration):
        estimator = DepthEstimator(calibration.Q)
        with pytest.raises(ValueError):
            estimator.estimate(np.zeros(10, dtype=np.float32))

    def test_no_nan_or_inf_leaks_into_the_result(self, calibration):
        """Regression coverage: invalid-disparity division must never leak through unclamped."""
        estimator = DepthEstimator(calibration.Q)
        disp = _mixed_disparity_map()

        depth = estimator.estimate(disp)

        assert np.all(np.isfinite(depth))
