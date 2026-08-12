"""
Phase D4 tests — local surface geometry (SurfaceEvidence,
geometry.surface.build_surface_evidence).

Covers the 12 scenarios the D4 task named. Unit-level tests (TestKnownPlane*,
TestNormalization, TestOrientation*, TestInsufficientSupport*, TestMixed*,
TestCoordinateFrame) construct PointCloud objects directly, over an
ANALYTICALLY KNOWN plane wherever possible, so expected geometry is known —
not merely that a value exists. Integration-level tests (TestGeometryFrame*,
TestFlagDisabled*, TestLegacyOutputsUnchanged, TestNoBehavioralLeakage) run
the real, unmodified pipeline, mirroring test_geometry_frame.py's own
"full chain" configuration pattern. Point 12 ("full existing suite has zero
regressions") has no dedicated test here — it's proven by the full
`pytest tests/ -q` run this file is part of.
"""

import dataclasses
import inspect

import numpy as np
import pytest

import depth_perception_engine.geometry.surface as surface_module
from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, SurfaceEvidence, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.surface import build_surface_evidence
from depth_perception_engine.geometry.types import PointCloud

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size


# ===================================================================
# Helpers — synthetic PointClouds over analytically known geometry
# ===================================================================
def _flat_plane_cloud(z=5.0, size=40, frame_id=FrameId.BODY):
    """Every point lies exactly on the plane Z = z (normal (0, 0, ±1))."""
    xs, ys = np.meshgrid(
        np.linspace(-1.0, 1.0, size), np.linspace(-1.0, 1.0, size), indexing="xy",
    )
    points = np.zeros((size, size, 3), dtype=np.float64)
    points[..., 0] = xs
    points[..., 1] = ys
    points[..., 2] = z
    valid_mask = np.ones((size, size), dtype=bool)
    return PointCloud(points=points, frame_id=frame_id, valid_mask=valid_mask)


def _tilted_plane_cloud(size=40, frame_id=FrameId.BODY):
    """Every point lies exactly on the plane Z = X + 5 (tilted 45 degrees
    about the Y axis) — analytic normal is +-(1, 0, -1)/sqrt(2)."""
    xs, ys = np.meshgrid(
        np.linspace(-1.0, 1.0, size), np.linspace(-1.0, 1.0, size), indexing="xy",
    )
    points = np.zeros((size, size, 3), dtype=np.float64)
    points[..., 0] = xs
    points[..., 1] = ys
    points[..., 2] = xs + 5.0
    valid_mask = np.ones((size, size), dtype=bool)
    return PointCloud(points=points, frame_id=frame_id, valid_mask=valid_mask)


def _all_invalid_cloud(size=40, frame_id=FrameId.BODY):
    points = np.full((size, size, 3), np.nan, dtype=np.float64)
    valid_mask = np.zeros((size, size), dtype=bool)
    return PointCloud(points=points, frame_id=frame_id, valid_mask=valid_mask)


_ORIGIN = np.zeros(3)  # viewpoint at (0, 0, 0) — behind every plane above (z < plane's z)


# ===================================================================
# 1. Known planar geometry produces expected normals
# ===================================================================
class TestKnownPlaneProducesExpectedNormal:
    def test_flat_plane_normal_is_exact(self):
        cloud = _flat_plane_cloud(z=5.0)
        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=3)
        assert len(evidence) == 1
        cell = evidence[0]

        assert cell.normal is not None
        # Viewpoint (0,0,0) is behind the plane (z=5) along +Z, so the
        # camera-facing normal must point back toward -Z.
        np.testing.assert_allclose(cell.normal, [0.0, 0.0, -1.0], atol=1e-5)
        np.testing.assert_allclose(cell.centroid_m, [0.0, 0.0, 5.0], atol=1e-5)
        assert cell.planarity == pytest.approx(1.0, abs=1e-6)

    def test_tilted_plane_normal_is_perpendicular_to_the_plane(self):
        cloud = _tilted_plane_cloud()
        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=3)
        cell = evidence[0]

        assert cell.normal is not None
        # In-plane directions: (1, 0, 1)/sqrt(2) (along increasing X) and
        # (0, 1, 0) (along Y) — the normal must be perpendicular to both,
        # for ANY analytically constructed plane, not just this one.
        in_plane_x = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        in_plane_y = np.array([0.0, 1.0, 0.0])
        assert np.dot(cell.normal, in_plane_x) == pytest.approx(0.0, abs=1e-5)
        assert np.dot(cell.normal, in_plane_y) == pytest.approx(0.0, abs=1e-5)
        assert cell.planarity == pytest.approx(1.0, abs=1e-6)

        # And matches the exact analytic normal up to sign (resolved by
        # the viewpoint convention, checked separately in TestOrientation).
        analytic_normal = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
        assert abs(np.dot(cell.normal, analytic_normal)) == pytest.approx(1.0, abs=1e-5)


# ===================================================================
# 2. Normal vectors obey the documented normalization convention
# ===================================================================
class TestNormalization:
    def test_normals_are_unit_vectors(self):
        for cloud in (_flat_plane_cloud(), _tilted_plane_cloud()):
            evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=2, grid_cols=2, min_support_count=3)
            for cell in evidence:
                if cell.normal is not None:
                    assert np.linalg.norm(cell.normal) == pytest.approx(1.0, abs=1e-5)


# ===================================================================
# 3. Orientation/sign convention is deterministic
# ===================================================================
class TestOrientationSignConvention:
    def test_repeated_calls_produce_identical_normals(self):
        cloud = _tilted_plane_cloud()
        first = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=3)
        second = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=3)
        np.testing.assert_array_equal(first[0].normal, second[0].normal)

    def test_normal_oriented_toward_viewpoint(self):
        cloud = _flat_plane_cloud(z=5.0)
        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=3)
        cell = evidence[0]
        view_vector = _ORIGIN - cell.centroid_m
        assert np.dot(cell.normal, view_vector) >= 0.0

    def test_flipping_viewpoint_to_the_other_side_flips_the_normal(self):
        cloud = _flat_plane_cloud(z=5.0)
        viewpoint_behind = np.array([0.0, 0.0, -10.0])   # z < 5: "in front" side
        viewpoint_ahead = np.array([0.0, 0.0, 20.0])      # z > 5: opposite side

        behind = build_surface_evidence(cloud, viewpoint_behind, grid_rows=1, grid_cols=1, min_support_count=3)
        ahead = build_surface_evidence(cloud, viewpoint_ahead, grid_rows=1, grid_cols=1, min_support_count=3)

        np.testing.assert_allclose(behind[0].normal, -np.array(ahead[0].normal), atol=1e-5)


# ===================================================================
# 4. Invalid depth does not fabricate surfaces
# ===================================================================
class TestInvalidDepthDoesNotFabricateSurfaces:
    def test_all_invalid_cloud_produces_no_planes(self):
        cloud = _all_invalid_cloud()
        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=2, grid_cols=2, min_support_count=3)

        assert len(evidence) == 4
        for cell in evidence:
            assert cell.support_count == 0
            assert cell.support_fraction == 0.0
            assert cell.centroid_m is None
            assert cell.normal is None
            assert cell.planarity is None


# ===================================================================
# 5. Insufficient support is represented explicitly
# ===================================================================
class TestInsufficientSupportRepresentedExplicitly:
    def test_below_threshold_reports_none_but_keeps_real_support_count(self):
        cloud = _flat_plane_cloud(size=10)
        # min_support_count deliberately higher than the cell can ever
        # contain (10x10 = 100 total points, one cell).
        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=1000)
        cell = evidence[0]

        assert cell.support_count == 100
        assert cell.support_fraction == pytest.approx(1.0)
        assert cell.centroid_m is None
        assert cell.normal is None
        assert cell.planarity is None

    def test_exactly_at_threshold_fits_a_plane(self):
        cloud = _flat_plane_cloud(size=10)  # 100 points, all valid
        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=100)
        cell = evidence[0]

        assert cell.support_count == 100
        assert cell.normal is not None

    def test_min_support_count_below_three_is_rejected(self):
        cloud = _flat_plane_cloud(size=10)
        with pytest.raises(ValueError):
            build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=2)


# ===================================================================
# 6. Mixed valid/invalid geometry behaves safely
# ===================================================================
class TestMixedValidInvalidGeometry:
    def test_half_valid_half_invalid_grid_handles_each_cell_independently(self):
        cloud = _flat_plane_cloud(size=40)
        # Invalidate the entire right half of the grid.
        points = cloud.points.copy()
        valid_mask = cloud.valid_mask.copy()
        valid_mask[:, 20:] = False
        points[~valid_mask] = np.nan
        mixed_cloud = PointCloud(points=points, frame_id=cloud.frame_id, valid_mask=valid_mask)

        evidence = build_surface_evidence(mixed_cloud, _ORIGIN, grid_rows=1, grid_cols=2, min_support_count=3)
        assert len(evidence) == 2
        left_cell, right_cell = evidence[0], evidence[1]

        assert left_cell.support_count > 0
        assert left_cell.normal is not None
        assert right_cell.support_count == 0
        assert right_cell.normal is None

    def test_output_length_is_always_grid_rows_times_grid_cols(self):
        for cloud in (_flat_plane_cloud(), _all_invalid_cloud()):
            for rows, cols in ((1, 1), (2, 2), (3, 3), (4, 5)):
                evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=rows, grid_cols=cols, min_support_count=3)
                assert len(evidence) == rows * cols


# ===================================================================
# 7. Coordinate-frame contract is correct
# ===================================================================
class TestCoordinateFrameContract:
    def test_frame_id_matches_source_cloud(self):
        cloud = _flat_plane_cloud(frame_id=FrameId.BODY)
        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=2, grid_cols=2, min_support_count=3)
        for cell in evidence:
            assert cell.frame_id == FrameId.BODY

    def test_frame_id_is_generic_not_hardcoded(self):
        cloud = _flat_plane_cloud(frame_id=FrameId.CAMERA_OPTICAL_LEFT)
        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=3)
        assert evidence[0].frame_id == FrameId.CAMERA_OPTICAL_LEFT

    def test_centroid_and_normal_expressed_in_cloud_frame_no_second_transform(self):
        # centroid_m must equal the mean of the SAME points passed in,
        # untransformed — proof there is no hidden second frame conversion.
        cloud = _flat_plane_cloud(z=5.0, frame_id=FrameId.BODY)
        evidence = build_surface_evidence(cloud, _ORIGIN, grid_rows=1, grid_cols=1, min_support_count=3)
        expected_centroid = cloud.points[cloud.valid_mask].mean(axis=0)
        np.testing.assert_allclose(evidence[0].centroid_m, expected_centroid, atol=1e-5)


# ===================================================================
# 8/9/10/11 — pipeline integration: GeometryFrame exposure, disabled
# path, legacy outputs
# ===================================================================
def _transform() -> RigidTransform:
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _random_pair(seed=42):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    return left, right


def _surface_config(**overrides):
    defaults = dict(
        enable_geometry=True, enable_geometry_frame=True, enable_surface_geometry=True,
        surface_grid_rows=2, surface_grid_cols=2, surface_min_support_count=10,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestGeometryFrameExposesSurfaceEvidence:
    def test_present_and_shaped_like_the_configured_grid(self):
        pipeline = DepthPerceptionPipeline(_surface_config(), _CALIBRATION, body_T_camera_left=_transform())
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.surface_evidence is not None
        assert len(result.surface_evidence) == 4  # 2x2 grid
        for cell in result.surface_evidence:
            assert isinstance(cell, SurfaceEvidence)

        assert result.geometry_frame is not None
        assert result.geometry_frame.surface_evidence is not None

    def test_zero_recomputation_same_object_as_result(self):
        pipeline = DepthPerceptionPipeline(_surface_config(), _CALIBRATION, body_T_camera_left=_transform())
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.geometry_frame.surface_evidence is result.surface_evidence


class TestFlagDisabledPreservesPreviousBehavior:
    def test_disabled_by_default(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert result.surface_evidence is None

    def test_disabled_even_with_geometry_frame_and_geometry_body_enabled(self):
        config = PipelineConfig(enable_geometry=True, enable_geometry_frame=True, enable_surface_geometry=False)
        pipeline = DepthPerceptionPipeline(config, _CALIBRATION, body_T_camera_left=_transform())
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.surface_evidence is None
        assert result.geometry_frame is not None
        assert result.geometry_frame.surface_evidence is None


class TestLegacyOutputsUnchanged:
    def test_upstream_fields_identical_regardless_of_surface_geometry_flag(self):
        left, right = _random_pair()

        result_off = DepthPerceptionPipeline(
            _surface_config(enable_surface_geometry=False), _CALIBRATION, body_T_camera_left=_transform(),
        ).process(left, right)
        result_on = DepthPerceptionPipeline(
            _surface_config(enable_surface_geometry=True), _CALIBRATION, body_T_camera_left=_transform(),
        ).process(left, right)

        assert np.array_equal(result_off.disparity_map, result_on.disparity_map)
        assert np.array_equal(result_off.depth_map, result_on.depth_map)
        np.testing.assert_array_equal(result_off.geometry.points, result_on.geometry.points)
        np.testing.assert_array_equal(result_off.geometry_body.points, result_on.geometry_body.points)
        assert result_off.traversability_mask.decision == result_on.traversability_mask.decision
        assert result_off.confidence == result_on.confidence


# ===================================================================
# 12. No behavioral semantics leak into surface evidence
# ===================================================================
class TestNoBehavioralLeakage:
    def test_surface_evidence_field_names_carry_no_behavioral_concept(self):
        fields = {f.name for f in dataclasses.fields(SurfaceEvidence)}
        forbidden = {"classification", "status", "decision", "traversable", "landing"}
        assert not (fields & forbidden)

    def test_surface_module_does_not_import_behavioral_types(self):
        import_lines = [
            line for line in inspect.getsource(surface_module).splitlines()
            if line.startswith(("import ", "from "))
        ]
        source = "\n".join(import_lines)
        for forbidden in ("RegionClass", "NavigationDecision", "ThreatAssessor"):
            assert forbidden not in source, f"{forbidden} must not be imported by geometry/surface.py"
            assert not hasattr(surface_module, forbidden)

    def test_build_surface_evidence_signature_takes_no_traversability_or_obstacle_input(self):
        # SurfaceEvidence must be fully derivable from a PointCloud +
        # viewpoint + grid config alone — no traversability.RegionStats/
        # RegionClass or obstacles.BeamReading/ThreatAssessor input path
        # exists for it to read from.
        params = set(inspect.signature(build_surface_evidence).parameters)
        assert params == {"cloud", "viewpoint", "grid_rows", "grid_cols", "min_support_count"}
