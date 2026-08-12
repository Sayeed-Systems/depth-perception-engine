"""
Black-box GeometryFrame provider-consumption validation — Phase D10 (see
docs/DPE_V1_PROVIDER_CONTRACT.md's D10 record).

Phase D9's completeness audit found exactly one real test-coverage gap:
no test consumed GeometryFrame through ONLY the supported public API,
the way a future hybrid_perception_engine consumer actually would. This
file closes that gap directly.

STRUCTURAL RULE — enforced by TestPublicApiOnlyImportSurface below, not
merely by convention: this file imports depth_perception_engine only via
its root package and the two documented public subpackage paths already
used elsewhere in this suite's own black-box-style tests
(`depth_perception_engine.frames`, `depth_perception_engine.temporal`) —
never `.traversability`, `.obstacles`, `.geometry`, `.fusion`, or any
other internal module. It therefore cannot depend on RegionAnalyzer,
ThreatAssessor, TemporalHistory, TemporalRecord, or any private algorithm
module by construction, not merely by omission.

Ground-truth technique: a real stereo pair with a KNOWN, engineered
integer pixel shift (smoothed random low-frequency texture, generated
once on a canvas W+shift wide, then cropped into left/right views) run
through the REAL, unmodified StereoSGBM matcher (`rectify=False` — see
"rectify note" below) — this is the one metric E7's own synthetic
ground-truth suite deliberately never measured, since
tests/test_e7_synthetic_ground_truth.py injects disparity directly and
never exercises SGBM itself. Measured disparity/depth error is reported
honestly (median/mean/std against the known true shift) — no invented
pass threshold is silently relied upon; the one sanity bound asserted
below is explicitly labeled as a
PROPOSED candidate, derived from what was actually measured during
development (median error ~0.0 px, std ~0.02-0.03 px on this fixture),
not invented independently of any measurement.

Rectify note: `rectify=False` is used deliberately, matching this
repository's own established precedent for synthetic (non-real-camera)
stereo fixtures (see docs/VALIDATION_REPORT.md's E3/E6 addenda, "synthetic
calibration, rectify=False") — the real calibration's own undistort-
rectify maps are fit to actual lens distortion and do not apply to a
flat, undistorted synthetic image; enabling rectify=True on a synthetic
fixture silently corrupts the engineered pixel shift instead of testing
anything real (confirmed empirically during this file's development:
rectify=True produced disparity with no measurable relationship to the
true shift at all).
"""

import numpy as np
import pytest

# ===================================================================
# Public-API-only imports — see the module docstring's structural rule.
# ===================================================================
import cv2  # noqa: F401 -- test fixture generation only, not part of the assertion surface

import depth_perception_engine as dpe
from depth_perception_engine import (
    DepthPerceptionPipeline,
    GeometryFrame,
    PipelineConfig,
    load_stereo_calibration,
)
from depth_perception_engine.frames import FrameId, RigidTransform

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size
_FX = 614.5223992233675
_BASELINE_M = 0.0647261287661154
_SHIFT_PX = 24  # -> expected_depth ~= 1.657 m, well inside DepthEstimator's [0.15, 8.0] m range


def _engineered_stereo_pair(shift_px: int, seed: int = 7):
    """A real stereo pair with a KNOWN true disparity of exactly
    `shift_px`, everywhere the two crops overlap. Built from smoothed
    low-frequency noise (real local gradient structure, unlike i.i.d.
    per-pixel noise, which was tried during development and found to
    defeat StereoSGBM's smoothness-regularized cost aggregation
    entirely — median disparity came back unrelated to the true shift).
    Not periodic (unlike block-replicated noise, also tried and rejected
    during development for producing aliased matches at multiples of the
    block size) — one continuous canvas, cropped, never tiled.
    """
    canvas_w = _W + shift_px
    rng = np.random.default_rng(seed)
    low_res = rng.integers(0, 255, (_H // 4 + 2, canvas_w // 4 + 2), dtype=np.uint8)
    canvas = cv2.resize(low_res, (canvas_w, _H), interpolation=cv2.INTER_CUBIC)
    canvas_bgr = np.stack([canvas] * 3, axis=-1)
    # World point at canvas column c appears at column c in the left crop
    # and c - shift_px in the right crop -> x_left - x_right = +shift_px,
    # the standard rectified-stereo sign convention (nearer -> larger
    # disparity; this is a FRONTO-PARALLEL synthetic plane, uniform
    # shift everywhere).
    left = canvas_bgr[:, 0:_W]
    right = canvas_bgr[:, shift_px:shift_px + _W]
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
# Structural: this file itself never reaches into DPE internals
# ===================================================================
class TestPublicApiOnlyImportSurface:
    def test_this_test_file_imports_no_internal_dpe_module(self):
        import ast
        import pathlib

        source = pathlib.Path(__file__).read_text()
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        allowed_prefixes = (
            "depth_perception_engine.frames",
            "depth_perception_engine.temporal",
        )
        for module in imported_modules:
            if module == "depth_perception_engine":
                continue
            if not module.startswith("depth_perception_engine"):
                continue
            assert module.startswith(allowed_prefixes), (
                f"black-box provider test imported {module!r} — only the root "
                "package and the frames/temporal public subpackages are "
                "permitted; it exists to prove GeometryFrame is consumable "
                "through the public API alone, without RegionAnalyzer/"
                "ThreatAssessor/TemporalHistory/TemporalRecord/private modules"
            )


# ===================================================================
# Scenario: real SGBM disparity/depth error on a known engineered shift
# ===================================================================
class TestRealDisparityAndDepthErrorAgainstKnownShift:
    def test_measured_disparity_and_depth_error_reported_honestly(self, capsys):
        left, right = _engineered_stereo_pair(_SHIFT_PX)
        expected_depth_m = (_FX * _BASELINE_M) / _SHIFT_PX

        pipeline = _pipeline()
        result = pipeline.process(left, right)

        valid = result.valid_disparity_mask
        assert valid.sum() > 0, "fixture produced zero valid disparity — cannot measure error"
        disparity_error = result.disparity_map[valid] - _SHIFT_PX
        depth_error = result.depth_map[result.valid_depth_mask] - expected_depth_m

        median_disp_err = float(np.median(disparity_error))
        mean_disp_err = float(np.mean(disparity_error))
        std_disp_err = float(np.std(disparity_error))
        median_depth_err = float(np.median(depth_error))
        std_depth_err = float(np.std(depth_error))

        print(f"\n[D10 black-box] valid_fraction={valid.mean():.3f} "
              f"disparity_error: median={median_disp_err:.4f}px mean={mean_disp_err:.4f}px std={std_disp_err:.4f}px "
              f"depth_error: median={median_depth_err:.6f}m std={std_depth_err:.6f}m "
              f"(expected_depth={expected_depth_m:.4f}m)")

        # PROPOSED candidate thresholds (not a pre-existing spec — none
        # was found in docs/VALIDATION_REPORT.md or elsewhere for real
        # SGBM-vs-truth disparity error): 1.0 px median disparity error,
        # a >30x margin above the ~0.0 px / ~0.02-0.03 px std actually
        # measured on this fixture during development. A real deployment
        # should replace this with a value derived from its own
        # target-hardware measurements, not this synthetic fixture's.
        assert abs(median_disp_err) < 1.0, (
            f"median disparity error {median_disp_err:.4f}px exceeds the proposed "
            "1.0px candidate bound"
        )

    def test_repeated_calls_on_identical_input_are_bit_identical(self):
        """Determinism / repeatability (D10 scenario 8, single-call form)
        — through the FULL black-box path, not just at the function
        level (that form is already covered by test_determinism.py)."""
        left, right = _engineered_stereo_pair(_SHIFT_PX)

        result_a = _pipeline().process(left, right)
        result_b = _pipeline().process(left, right)

        np.testing.assert_array_equal(result_a.disparity_map, result_b.disparity_map)
        np.testing.assert_array_equal(result_a.depth_map, result_b.depth_map)
        gf_a, gf_b = result_a.geometry_frame, result_b.geometry_frame
        np.testing.assert_array_equal(gf_a.obstacle_cloud.points, gf_b.obstacle_cloud.points)
        np.testing.assert_array_equal(gf_a.free_space_rays.ranges_m, gf_b.free_space_rays.ranges_m)
        assert gf_a.geometry_metrics == gf_b.geometry_metrics
        assert [s.normal.tolist() if s.normal is not None else None for s in gf_a.surface_evidence] == \
               [s.normal.tolist() if s.normal is not None else None for s in gf_b.surface_evidence]
        assert gf_a.boundary_evidence == gf_b.boundary_evidence
        assert gf_a.opening_evidence == gf_b.opening_evidence


# ===================================================================
# Cross-field consistency, checked through GeometryFrame's own public
# fields only
# ===================================================================
class TestGeometryFrameCrossFieldConsistency:
    def _frame(self):
        left, right = _engineered_stereo_pair(_SHIFT_PX)
        result = _pipeline().process(left, right)
        assert isinstance(result.geometry_frame, GeometryFrame)
        return result.geometry_frame

    def test_frame_ids_use_the_documented_public_vocabulary(self):
        gf = self._frame()
        assert gf.frame_id == dpe.FrameId.CAMERA_OPTICAL_LEFT
        assert gf.geometry.frame_id == dpe.FrameId.CAMERA_OPTICAL_LEFT
        assert gf.geometry_body.frame_id == dpe.FrameId.BODY
        assert gf.obstacle_cloud.frame_id == dpe.FrameId.BODY
        for region in gf.region_evidence.values():
            assert region.frame_id == dpe.FrameId.CAMERA_OPTICAL_LEFT
        for clearance in gf.clearance_evidence:
            assert clearance.frame_id == dpe.FrameId.CAMERA_OPTICAL_LEFT
        for surface in gf.surface_evidence:
            assert surface.frame_id == dpe.FrameId.BODY

    def test_obstacle_and_ray_counts_match_valid_body_geometry_exactly(self):
        gf = self._frame()
        expected_count = int(gf.geometry_body.valid_mask.sum())
        assert gf.obstacle_cloud.points.shape[0] == expected_count
        assert gf.free_space_rays.ranges_m.shape[0] == expected_count
        assert gf.geometry_metrics.point_count == expected_count

    def test_clearance_bearing_bounds_are_monotonic_and_coverage_in_unit_range(self):
        gf = self._frame()
        assert len(gf.clearance_evidence) > 0
        for sector in gf.clearance_evidence:
            assert sector.bearing_min_rad <= sector.bearing_center_rad <= sector.bearing_max_rad
            assert 0.0 <= sector.coverage_fraction <= 1.0
            if sector.support_state == "NO_EVIDENCE":
                assert sector.nearest_distance_m is None
                assert not sector.has_evidence
            else:
                assert sector.nearest_distance_m is not None
                assert sector.has_evidence

    def test_boundary_and_opening_evidence_are_internally_consistent(self):
        gf = self._frame()
        assert len(gf.boundary_evidence) > 0
        for edge in gf.boundary_evidence:
            if edge.state == "INSUFFICIENT_EVIDENCE":
                assert edge.depth_step_m is None
            else:
                assert edge.depth_step_m is not None
                assert edge.depth_step_m >= 0.0
        for opening in gf.opening_evidence:
            assert opening.approx_range_m > 0.0
            assert opening.approx_width_m >= 0.0
            assert 0.0 <= opening.support_fraction <= 1.0

    def test_quality_rollup_agrees_with_its_own_documented_dimension_map(self):
        """Cross-checks GeometryFrame.quality against GeometryMetrics.
        valid_fraction using only the documented default PipelineConfig
        thresholds (geometry_healthy_min_valid_fraction=0.5,
        geometry_degraded_min_valid_fraction=0.05) — these are public,
        documented config VALUES, reproduced inline rather than by
        importing the internal classify_geometry_quality() algorithm
        itself, to keep this test genuinely black-box."""
        gf = self._frame()
        q = gf.quality
        assert q is not None
        valid_fraction = gf.geometry_metrics.valid_fraction
        if valid_fraction >= 0.5:
            expected = "VALID"
        elif valid_fraction >= 0.05:
            expected = "DEGRADED"
        else:
            expected = "INSUFFICIENT"
        assert q.geometry_validity_state == expected
        assert q.overall_state in {"VALID", "DEGRADED", "INSUFFICIENT"}
        if q.overall_state == "DEGRADED":
            assert any(":DEGRADED" in reason for reason in q.degradation_reasons)


# ===================================================================
# Scenario 8 — repeated (near-)static frames: real temporal state
# transitions through GeometryFrame, public API only
# ===================================================================
class TestScenario8RepeatedFramesTemporalStateTransitions:
    def test_state_transitions_match_the_documented_first_frame_vs_steady_state_pattern(self, capsys):
        left, right = _engineered_stereo_pair(_SHIFT_PX)
        pipeline = _pipeline()

        observed = []
        for i in range(4):
            result = pipeline.process(left, right, left_timestamp=float(i), right_timestamp=float(i))
            gf = result.geometry_frame
            observed.append((
                gf.temporal_consistency.state,
                gf.temporal_persistence.state,
                gf.quality.overall_state,
            ))
        print(f"\n[D10 Scenario 8] (temporal_consistency, temporal_persistence, quality) per frame: {observed}")

        # Frame 0: no prior frame exists yet -> INSUFFICIENT_EVIDENCE,
        # never fabricated as CONSISTENT.
        assert observed[0][0] == "INSUFFICIENT_EVIDENCE"
        # Frames 1-3: the identical repeated scene is genuinely
        # self-consistent -> CONSISTENT every time, not merely "not
        # contradictory by omission".
        for tc_state, _, _ in observed[1:]:
            assert tc_state == "CONSISTENT"
        # temporal_persistence reaches CLASSIFIED (a real, computed
        # verdict, never stuck at INSUFFICIENT_EVIDENCE) by the last frame.
        assert observed[-1][1] == "CLASSIFIED"

    def test_repeated_frames_produce_stable_not_growing_evidence_counts(self):
        """A static scene's obstacle/ray/opening counts must not drift
        frame to frame purely from repetition (no hidden accumulation)."""
        left, right = _engineered_stereo_pair(_SHIFT_PX)
        pipeline = _pipeline()

        counts = []
        for i in range(4):
            result = pipeline.process(left, right, left_timestamp=float(i), right_timestamp=float(i))
            gf = result.geometry_frame
            counts.append((gf.obstacle_cloud.points.shape[0], gf.free_space_rays.ranges_m.shape[0]))

        assert len(set(counts)) == 1, f"evidence counts drifted across identical repeated frames: {counts}"
