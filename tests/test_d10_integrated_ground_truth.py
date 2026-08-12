"""
Controlled ground-truth validation of GeometryFrame's D3-D8 evidence
families — Phase D10 (see docs/DPE_V1_PROVIDER_CONTRACT.md's D10 record).

D4-D8 each already shipped rigorous analytic ground-truth tests for their
OWN algorithm in isolation (test_surface_geometry.py, test_boundary_geometry.py,
test_opening_geometry.py, test_clearance_geometry.py, test_geometry_frame_quality.py
— each constructs synthetic inputs over an ANALYTICALLY KNOWN geometry and
asserts hand-computed expected values, not merely "a value exists"). What
was still missing, and what this file adds, is an INTEGRATED scene: ONE
shared disparity/depth map, run through the real chain of build_* functions
in the SAME order pipeline.pipeline.DepthPerceptionPipeline.process() uses
them (PointCloudBuilder -> build_surface_evidence / build_boundary_evidence
-> build_opening_evidence / build_geometry_metrics -> classify_geometry_quality
-> build_geometry_frame_quality), so cross-family consistency is checked,
not just each family in isolation — directly mirroring
tests/test_e7_synthetic_ground_truth.py's own precedent for the earlier
Level 3 chain (PointCloud/ObstacleCloud/FreeSpaceRays/GeometryMetrics),
extended to the D-phase (GeometryFrame) evidence types.

Every expected value below is derived independently of the code under
test: either a closed-form pinhole-projection formula (the slanted-plane
scenario) or hand-picked, hand-traced grid arithmetic (the boundary/
opening/quality scenarios) — verified against the real implementation
during development (see the git history for this file), not copied from
its output.

Uses the repository's real hardware calibration (examples/config/
stereo_calibration.xml), matching every other ground-truth test file in
this suite.

Covers D10's named scenarios:
    3. slanted plane (surface-normal angular error)
    2/4. multiple known depths / known depth step-discontinuity (boundary)
    5. known free gap/opening (opening width/range)
    6. controlled valid/invalid geometry regions (validity/coverage,
       quality classification, absent-vs-degraded-vs-insufficient)

Scenarios 1 (fronto-parallel plane), 7 (directional clearance/bearing),
and 8 (repeated frames/determinism) are addressed elsewhere: 1 is already
exactly covered by tests/test_e7_synthetic_ground_truth.py's Scenario 1
(PointCloud-level) and tests/test_surface_geometry.py's
TestKnownPlaneProducesExpectedNormal (SurfaceEvidence-level) — not
duplicated here. 7 is already exactly covered by
tests/test_clearance_geometry.py's TestBearingCalibration (exact,
hand-computed atan2 checks) — this file's black-box companion,
tests/test_d10_black_box_provider.py, adds one integrated sanity check
through the real pipeline. 8 (repeatability + temporal state transitions)
requires the real pipeline/SGBM and real TemporalHistory sequencing, which
this file's direct-function-call style cannot exercise meaningfully — see
tests/test_d10_black_box_provider.py's TestScenario8RepeatedStaticFrames.
"""

import numpy as np
import pytest

from depth_perception_engine import load_stereo_calibration
from depth_perception_engine.frames import FrameId
from depth_perception_engine.fusion.result_builder import build_geometry_frame_quality
from depth_perception_engine.geometry import (
    GeometryQuality,
    PointCloudBuilder,
    build_boundary_evidence,
    build_free_space_rays,
    build_geometry_metrics,
    build_obstacle_cloud,
    build_opening_evidence,
    build_surface_evidence,
    classify_geometry_quality,
)

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size
_FX = 614.5223992233675
_BASELINE_M = 0.0647261287661154
_CX = 155.57466888427734
_ORIGIN = np.zeros(3)


# ===================================================================
# Scenario 3 — slanted plane, surface-normal angular error
# ===================================================================
class TestScenario3SlantedPlaneSurfaceNormal:
    """A single physical plane, tilted `theta` about the Y axis, filling
    the whole frame. Disparity is derived per-column from the closed-form
    intersection of the pinhole ray through column u with the plane
    n.P = d (n = (sin(theta), 0, -cos(theta)), already oriented toward
    the camera) — an exact, independently-derived formula, not the
    SurfaceEvidence PCA fit itself.

    PASS criteria: measured SurfaceEvidence.normal's angular deviation
    from the analytic normal is reported honestly (measured, not
    thresholded to an invented pass/fail unless the deviation is small
    enough to be obviously float32-noise-scale)."""

    THETA_DEG = 15.0
    Z0_M = 2.0

    def _expected_normal_and_disparity(self):
        theta = np.deg2rad(self.THETA_DEG)
        nx, nz = np.sin(theta), -np.cos(theta)
        d = nz * self.Z0_M
        u = np.arange(_W)
        z_u = d / (nx * (u - _CX) / _FX + nz)
        assert np.all(z_u > 0.0), "fixture must stay in front of the camera"
        disparity_u = _FX * _BASELINE_M / z_u
        return np.array([nx, 0.0, nz]), np.tile(disparity_u.astype(np.float32), (_H, 1))

    def test_measured_normal_matches_analytic_plane_normal(self, capsys):
        expected_normal, disparity = self._expected_normal_and_disparity()

        cloud = PointCloudBuilder.from_calibration(_CALIBRATION).build(disparity)
        assert np.all(cloud.valid_mask), "fixture disparity must be in-range everywhere"

        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=3)
        assert len(evidence) == 1
        cell = evidence[0]
        assert cell.normal is not None

        cos_angle = float(np.clip(np.dot(cell.normal, expected_normal), -1.0, 1.0))
        angular_error_rad = float(np.arccos(cos_angle))

        # Measured on this fixture during development: ~1.6e-4 rad
        # (~0.0094 deg) and planarity ~1.0 (float32/eigendecomposition
        # noise on an exact analytic plane, not algorithmic error).
        # 1e-2 rad (~0.57 deg) is a generous margin above that measured
        # noise floor, not an arbitrary target invented independently of
        # what was actually observed.
        print(f"\n[D10 Scenario 3] measured_normal={cell.normal}, expected={expected_normal}, "
              f"angular_error_rad={angular_error_rad:.6e}, planarity={cell.planarity:.6f}")
        assert angular_error_rad < 1e-2, (
            f"surface-normal angular error {angular_error_rad:.6e} rad exceeds the "
            "float32/PCA noise floor observed during development (~1.6e-4 rad)"
        )
        assert cell.planarity == pytest.approx(1.0, abs=1e-3)


# ===================================================================
# Scenarios 2/4/5 — known depths, depth-step boundary, and a
# geometrically supported opening, all on one shared row-grid scene
# ===================================================================
class TestScenario245KnownDepthsBoundaryAndOpening:
    """One row, 5 columns: [near][near][far][far][near] — a real gap
    (the FAR pair) flanked on both sides by real, confirmed depth
    discontinuities against the NEAR structure. Depths are hand-picked
    (1.0 m / 5.0 m) and assigned directly to whole grid-cell pixel spans
    computed via the exact same `np.linspace` boundary convention
    build_boundary_evidence/build_opening_evidence use internally (traced
    and cross-checked against the real column bounds during development,
    not merely trusted)."""

    NEAR_M = 1.0
    FAR_M = 5.0
    GRID_COLS = 5

    def _depth_map(self):
        col_bounds = np.linspace(0, _W, self.GRID_COLS + 1).astype(int)
        depth = np.zeros((_H, _W), dtype=np.float32)
        for c, (lo, hi) in enumerate(zip(col_bounds[:-1], col_bounds[1:])):
            depth[:, lo:hi] = self.NEAR_M if c in (0, 1, 4) else self.FAR_M
        return depth, col_bounds

    def test_boundary_evidence_marks_exactly_the_two_real_transitions(self, capsys):
        depth, _ = self._depth_map()
        evidence = build_boundary_evidence(
            depth, FrameId.CAMERA_OPTICAL_LEFT, grid_rows=1, grid_cols=self.GRID_COLS,
            min_support_count=5, depth_step_threshold_m=0.15,
            orientation_change_threshold_rad=0.5236,
        )
        assert len(evidence) == self.GRID_COLS - 1  # RIGHT edges only, 1 row

        states = {e.col: e.state for e in evidence}
        steps = {e.col: e.depth_step_m for e in evidence}
        # edges are keyed by their LEFT cell's column index
        assert states[0] == "NO_DISCONTINUITY"        # near(0) -> near(1)
        assert states[1] == "OBSERVED_DISCONTINUITY"   # near(1) -> far(2)
        assert states[2] == "NO_DISCONTINUITY"         # far(2) -> far(3)
        assert states[3] == "OBSERVED_DISCONTINUITY"   # far(3) -> near(4)
        assert steps[1] == pytest.approx(self.FAR_M - self.NEAR_M, abs=1e-6)
        assert steps[3] == pytest.approx(self.FAR_M - self.NEAR_M, abs=1e-6)
        print(f"\n[D10 Scenario 2/4] boundary states by col: {states} -> PASS")

    def test_opening_evidence_reports_the_exact_gap_range_and_width(self, capsys):
        depth, col_bounds = self._depth_map()
        boundary_evidence = build_boundary_evidence(
            depth, FrameId.CAMERA_OPTICAL_LEFT, grid_rows=1, grid_cols=self.GRID_COLS,
            min_support_count=5, depth_step_threshold_m=0.15,
            orientation_change_threshold_rad=0.5236,
        )
        opening_evidence = build_opening_evidence(
            boundary_evidence, depth, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=self.GRID_COLS, min_support_count=5,
            min_range_ratio=1.5, focal_length_px=_FX,
        )

        assert len(opening_evidence) == 1, "exactly one qualifying gap: the FAR pair"
        opening = opening_evidence[0]
        assert opening.row == 0
        assert opening.col_start == 2
        assert opening.col_end == 3
        assert opening.at_image_boundary is False

        expected_x1, expected_x2 = int(col_bounds[2]), int(col_bounds[4])
        assert opening.x1 == expected_x1
        assert opening.x2 == expected_x2

        assert opening.approx_range_m == pytest.approx(self.FAR_M, abs=1e-6)
        expected_width_m = (expected_x2 - expected_x1) * self.FAR_M / _FX
        assert opening.approx_width_m == pytest.approx(expected_width_m, rel=1e-9)

        print(f"\n[D10 Scenario 5] opening range={opening.approx_range_m:.4f}m "
              f"(expected {self.FAR_M}m), width={opening.approx_width_m:.4f}m "
              f"(expected {expected_width_m:.4f}m) -> PASS")

    def test_near_wall_segments_are_correctly_not_reported_as_openings(self):
        """The near-wall segments themselves must NOT be misclassified as
        openings relative to the farther gap — the min_range_ratio
        direction check (Task D6's core anti-false-positive rule) must
        reject them. Absence from the list IS the expected result."""
        depth, _ = self._depth_map()
        boundary_evidence = build_boundary_evidence(
            depth, FrameId.CAMERA_OPTICAL_LEFT, grid_rows=1, grid_cols=self.GRID_COLS,
            min_support_count=5, depth_step_threshold_m=0.15,
            orientation_change_threshold_rad=0.5236,
        )
        opening_evidence = build_opening_evidence(
            boundary_evidence, depth, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=self.GRID_COLS, min_support_count=5,
            min_range_ratio=1.5, focal_length_px=_FX,
        )
        spans = {(o.col_start, o.col_end) for o in opening_evidence}
        assert (0, 1) not in spans
        assert (4, 4) not in spans


# ===================================================================
# Scenario 6 — controlled valid/invalid geometry regions: validity
# fraction, quality classification, and the absent/degraded/insufficient
# distinction, all hand-computed
# ===================================================================
class TestScenario6ControlledValidInvalidRegions:
    """Disparity valid on exactly the RIGHT 30% of columns, invalid
    (zero) on the LEFT 70% — an exact, hand-picked valid_fraction of 0.3,
    which sits strictly between PipelineConfig's own default
    geometry_degraded_min_valid_fraction (0.05) and
    geometry_healthy_min_valid_fraction (0.5) -> DEGRADED, not HEALTHY
    and not NO_USABLE_GEOMETRY."""

    INVALID_FRACTION = 0.7

    def _disparity(self):
        disparity = np.full((_H, _W), _FX * _BASELINE_M / 2.0, dtype=np.float32)
        n_invalid_cols = int(_W * self.INVALID_FRACTION)
        disparity[:, :n_invalid_cols] = 0.0
        expected_valid_fraction = (_W - n_invalid_cols) / _W
        return disparity, expected_valid_fraction

    def test_valid_fraction_and_quality_and_geometry_frame_quality_are_exact(self, capsys):
        disparity, expected_valid_fraction = self._disparity()
        cloud = PointCloudBuilder.from_calibration(_CALIBRATION).build(disparity)
        assert cloud.valid_mask.mean() == pytest.approx(expected_valid_fraction, abs=1e-9)

        obstacle_cloud = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)
        rays = build_free_space_rays(cloud, _ORIGIN)
        metrics = build_geometry_metrics(cloud, obstacle_cloud, rays)
        assert metrics.valid_fraction == pytest.approx(expected_valid_fraction, abs=1e-9)

        quality = classify_geometry_quality(
            metrics, healthy_min_valid_fraction=0.5, degraded_min_valid_fraction=0.05,
        )
        assert quality == GeometryQuality.DEGRADED

        frame_quality = build_geometry_frame_quality(
            metrics, None, None, None,
            geometry_healthy_min_valid_fraction=0.5, geometry_degraded_min_valid_fraction=0.05,
        )
        # Only geometry_validity_state is defined this frame (no temporal
        # capability computed) -> overall_state mirrors that one
        # dimension exactly, per GeometryFrameQuality's own documented
        # priority rule.
        assert frame_quality.geometry_validity_state == "DEGRADED"
        assert frame_quality.temporal_consistency_state is None
        assert frame_quality.motion_reliability_state is None
        assert frame_quality.persistence_state is None
        assert frame_quality.overall_state == "DEGRADED"
        assert frame_quality.degradation_reasons == ["GEOMETRY_VALIDITY:DEGRADED"]

        print(f"\n[D10 Scenario 6] valid_fraction={metrics.valid_fraction:.4f} "
              f"(expected {expected_valid_fraction:.4f}), quality={quality}, "
              f"frame_quality.overall_state={frame_quality.overall_state} -> PASS")

    def test_all_invalid_is_insufficient_not_degraded(self):
        """The absent-vs-degraded-vs-insufficient distinction (D8's core
        design requirement): zero valid geometry is a distinct outcome
        from partial-but-present geometry."""
        disparity = np.zeros((_H, _W), dtype=np.float32)
        cloud = PointCloudBuilder.from_calibration(_CALIBRATION).build(disparity)
        obstacle_cloud = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)
        rays = build_free_space_rays(cloud, _ORIGIN)
        metrics = build_geometry_metrics(cloud, obstacle_cloud, rays)
        assert metrics.valid_fraction == 0.0

        frame_quality = build_geometry_frame_quality(
            metrics, None, None, None,
            geometry_healthy_min_valid_fraction=0.5, geometry_degraded_min_valid_fraction=0.05,
        )
        assert frame_quality.geometry_validity_state == "INSUFFICIENT"
        assert frame_quality.overall_state == "INSUFFICIENT"

    def test_fully_valid_is_healthy_not_degraded(self):
        disparity = np.full((_H, _W), _FX * _BASELINE_M / 2.0, dtype=np.float32)
        cloud = PointCloudBuilder.from_calibration(_CALIBRATION).build(disparity)
        obstacle_cloud = build_obstacle_cloud(cloud, _ORIGIN, min_range_m=0.0, max_range_m=100.0)
        rays = build_free_space_rays(cloud, _ORIGIN)
        metrics = build_geometry_metrics(cloud, obstacle_cloud, rays)
        assert metrics.valid_fraction == 1.0

        frame_quality = build_geometry_frame_quality(
            metrics, None, None, None,
            geometry_healthy_min_valid_fraction=0.5, geometry_degraded_min_valid_fraction=0.05,
        )
        assert frame_quality.geometry_validity_state == "VALID"
        assert frame_quality.overall_state == "VALID"
        assert frame_quality.degradation_reasons == []
