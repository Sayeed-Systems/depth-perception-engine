"""
Phase D5 tests — geometric boundaries/discontinuities (BoundaryEvidence,
geometry.boundary.build_boundary_evidence).

Covers the 12 scenarios the D5 task named. Unit-level tests
(TestFlatPlane*, TestKnownDepthStep*, TestSurfaceOrientationDiscontinuity*,
TestInvalidDepth*, TestInsufficientSupport*, TestDeterministicOutput,
TestSpatialFrameContract) construct depth maps / SurfaceEvidence lists
directly, over ANALYTICALLY CONTROLLED geometry, so expected evidence is
known — not merely that a value exists. Integration-level tests
(TestGeometryFrame*, TestFlagDisabled*, TestLegacyOutputsUnchanged,
TestNoBehavioralLeakage) run the real, unmodified pipeline, mirroring
test_surface_geometry.py's own pattern. Point 12 ("full existing suite has
zero regressions") has no dedicated test here — it's proven by the full
`pytest tests/ -q` run this file is part of.
"""

import dataclasses
import inspect

import numpy as np
import pytest

import depth_perception_engine.geometry.boundary as boundary_module
from depth_perception_engine import (
    BoundaryEvidence,
    BoundaryState,
    DepthPerceptionPipeline,
    PipelineConfig,
    load_stereo_calibration,
)
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry.boundary import build_boundary_evidence
from depth_perception_engine.geometry.surface import SurfaceEvidence

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size


# ===================================================================
# Helpers
# ===================================================================
def _flat_depth_map(depth=5.0, size=40):
    return np.full((size, size), depth, dtype=np.float32)


def _step_depth_map(near=2.0, far=5.0, size=40):
    """Left half at `near`, right half at `far` — a clean step exactly at
    the midline."""
    depth_map = np.full((size, size), near, dtype=np.float32)
    depth_map[:, size // 2:] = far
    return depth_map


def _surface_evidence_pair(normal_a, normal_b, frame_id=FrameId.CAMERA_OPTICAL_LEFT):
    """Two SurfaceEvidence cells (row-major, 1x2 grid) with the given
    normals and identical, well-supported centroids/planarity — isolates
    the orientation signal from the depth signal."""
    common = dict(
        frame_id=frame_id, y1=0, y2=10, support_count=50, support_fraction=1.0, planarity=1.0,
    )
    return [
        SurfaceEvidence(row=0, col=0, x1=0, x2=5, centroid_m=np.array([0.0, 0.0, 5.0]), normal=normal_a, **common),
        SurfaceEvidence(row=0, col=1, x1=5, x2=10, centroid_m=np.array([1.0, 0.0, 5.0]), normal=normal_b, **common),
    ]


# ===================================================================
# 1. Flat continuous plane does not produce false discontinuities
# ===================================================================
class TestFlatPlaneProducesNoFalseDiscontinuities:
    def test_uniform_depth_grid_is_all_no_discontinuity(self):
        depth_map = _flat_depth_map()
        evidence = build_boundary_evidence(
            depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=3, grid_cols=3, min_support_count=5,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
        )
        assert len(evidence) == 12  # 3*2 RIGHT + 2*3 DOWN
        for cell in evidence:
            assert cell.state == BoundaryState.NO_DISCONTINUITY
            assert cell.depth_step_m == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# 2. Known depth step produces expected boundary evidence
# ===================================================================
class TestKnownDepthStepProducesExpectedEvidence:
    def test_step_exactly_at_grid_boundary_is_observed(self):
        depth_map = _step_depth_map(near=2.0, far=5.0, size=40)
        # grid_cols=2 -> the column boundary falls exactly at x=20,
        # matching the step's own location.
        evidence = build_boundary_evidence(
            depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=2, min_support_count=5,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
        )
        assert len(evidence) == 1  # 1 RIGHT edge, no DOWN edges (1 row)
        cell = evidence[0]
        assert cell.state == BoundaryState.OBSERVED_DISCONTINUITY
        assert cell.depth_step_m == pytest.approx(3.0, abs=1e-6)

    def test_step_smaller_than_threshold_is_not_observed(self):
        depth_map = _step_depth_map(near=5.0, far=5.05, size=40)  # 5cm step
        evidence = build_boundary_evidence(
            depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=2, min_support_count=5,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
        )
        assert evidence[0].state == BoundaryState.NO_DISCONTINUITY
        assert evidence[0].depth_step_m == pytest.approx(0.05, abs=1e-6)


# ===================================================================
# 3. Known surface-orientation change can produce supported discontinuity
# ===================================================================
class TestSurfaceOrientationDiscontinuity:
    def test_perpendicular_normals_with_zero_depth_step_is_observed(self):
        depth_map = _flat_depth_map(depth=5.0, size=10)  # no depth step at all
        surface_evidence = _surface_evidence_pair(
            normal_a=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            normal_b=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # 90 degrees apart
        )
        evidence = build_boundary_evidence(
            depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=2, min_support_count=5,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,  # ~30 deg
            surface_evidence=surface_evidence, surface_grid_rows=1, surface_grid_cols=2,
        )
        cell = evidence[0]
        assert cell.depth_step_m == pytest.approx(0.0, abs=1e-6)
        assert cell.orientation_change_rad == pytest.approx(np.pi / 2, abs=1e-5)
        assert cell.state == BoundaryState.OBSERVED_DISCONTINUITY

    def test_small_orientation_change_below_threshold_is_not_observed(self):
        depth_map = _flat_depth_map(depth=5.0, size=10)
        surface_evidence = _surface_evidence_pair(
            normal_a=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            normal_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),  # identical
        )
        evidence = build_boundary_evidence(
            depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=2, min_support_count=5,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
            surface_evidence=surface_evidence, surface_grid_rows=1, surface_grid_cols=2,
        )
        assert evidence[0].orientation_change_rad == pytest.approx(0.0, abs=1e-6)
        assert evidence[0].state == BoundaryState.NO_DISCONTINUITY

    def test_mismatched_surface_grid_falls_back_to_depth_only(self):
        depth_map = _flat_depth_map(depth=5.0, size=10)
        surface_evidence = _surface_evidence_pair(
            normal_a=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            normal_b=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        # surface_grid_rows/cols (1x2) does NOT match boundary's own grid
        # (2x2) -> orientation must never be consulted.
        evidence = build_boundary_evidence(
            depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=2, grid_cols=2, min_support_count=1,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
            surface_evidence=surface_evidence, surface_grid_rows=1, surface_grid_cols=2,
        )
        for cell in evidence:
            assert cell.orientation_change_rad is None


# ===================================================================
# 4. Invalid/missing depth alone is not fabricated into a boundary
# ===================================================================
class TestInvalidDepthNotFabricatedIntoBoundary:
    def test_one_side_all_invalid_is_insufficient_not_discontinuous(self):
        depth_map = _flat_depth_map(depth=5.0, size=40)
        depth_map[:, 20:] = 0.0  # right half has NO valid depth at all
        evidence = build_boundary_evidence(
            depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=2, min_support_count=5,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
        )
        cell = evidence[0]
        assert cell.state == BoundaryState.INSUFFICIENT_EVIDENCE
        assert cell.depth_step_m is None
        assert cell.orientation_change_rad is None
        # support info is still real and informative, not omitted.
        assert cell.support_fraction_from == pytest.approx(1.0)
        assert cell.support_fraction_to == pytest.approx(0.0)


# ===================================================================
# 5. Insufficient support remains unknown/invalid
# ===================================================================
class TestInsufficientSupportRemainsUnknown:
    def test_below_min_support_count_is_insufficient(self):
        depth_map = _flat_depth_map(depth=5.0, size=10)
        depth_map[1:, :] = 0.0  # only the first row (10 px) stays valid per cell-half
        evidence = build_boundary_evidence(
            depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=2, min_support_count=1000,  # unreachable
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
        )
        assert evidence[0].state == BoundaryState.INSUFFICIENT_EVIDENCE

    def test_min_support_count_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            build_boundary_evidence(
                _flat_depth_map(), FrameId.CAMERA_OPTICAL_LEFT,
                grid_rows=1, grid_cols=1, min_support_count=0,
                depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
            )


# ===================================================================
# 6. Deterministic output
# ===================================================================
class TestDeterministicOutput:
    def test_repeated_calls_produce_identical_evidence(self):
        depth_map = _step_depth_map()
        surface_evidence = _surface_evidence_pair(
            normal_a=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            normal_b=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        kwargs = dict(
            depth_map=depth_map, frame_id=FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=2, min_support_count=5,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
            surface_evidence=surface_evidence, surface_grid_rows=1, surface_grid_cols=2,
        )
        first = build_boundary_evidence(**kwargs)
        second = build_boundary_evidence(**kwargs)
        assert first == second


# ===================================================================
# 7. Spatial/frame contract is correct
# ===================================================================
class TestSpatialFrameContract:
    def test_frame_id_is_generic_not_hardcoded(self):
        evidence = build_boundary_evidence(
            _flat_depth_map(), FrameId.BODY,
            grid_rows=2, grid_cols=2, min_support_count=5,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
        )
        for cell in evidence:
            assert cell.frame_id == FrameId.BODY

    def test_pixel_bounds_span_both_adjacent_cells_within_image(self):
        size = 40
        evidence = build_boundary_evidence(
            _flat_depth_map(size=size), FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=2, grid_cols=2, min_support_count=5,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
        )
        for cell in evidence:
            assert 0 <= cell.x1 < cell.x2 <= size
            assert 0 <= cell.y1 < cell.y2 <= size

    def test_output_length_matches_grid_edge_count(self):
        for rows, cols in ((1, 1), (2, 2), (3, 4)):
            evidence = build_boundary_evidence(
                _flat_depth_map(), FrameId.CAMERA_OPTICAL_LEFT,
                grid_rows=rows, grid_cols=cols, min_support_count=5,
                depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
            )
            expected = rows * (cols - 1) + cols * (rows - 1)
            assert len(evidence) == expected


# ===================================================================
# 8/9/10 — pipeline integration: GeometryFrame exposure, disabled path,
# legacy outputs
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


def _boundary_config(**overrides):
    defaults = dict(
        enable_geometry_frame=True, enable_boundary_geometry=True,
        boundary_grid_rows=2, boundary_grid_cols=2, boundary_min_support_count=5,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestGeometryFrameExposesBoundaryEvidence:
    def test_present_and_shaped_like_the_configured_grid(self):
        pipeline = DepthPerceptionPipeline(_boundary_config(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.boundary_evidence is not None
        assert len(result.boundary_evidence) == 4  # 2*(2-1) + 2*(2-1)
        for cell in result.boundary_evidence:
            assert isinstance(cell, BoundaryEvidence)

        assert result.geometry_frame is not None
        assert result.geometry_frame.boundary_evidence is not None

    def test_zero_recomputation_same_object_as_result(self):
        pipeline = DepthPerceptionPipeline(_boundary_config(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert result.geometry_frame.boundary_evidence is result.boundary_evidence

    def test_works_without_enable_geometry_or_body_extrinsic(self):
        # Baseline depth-discontinuity signal needs no PointCloud at all.
        config = PipelineConfig(
            enable_geometry_frame=True, enable_boundary_geometry=True,
            enable_geometry=False,
        )
        pipeline = DepthPerceptionPipeline(config, _CALIBRATION)  # no body_T_camera_left
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.geometry is None  # confirms enable_geometry really was off
        assert result.boundary_evidence is not None


class TestFlagDisabledPreservesPreviousBehavior:
    def test_disabled_by_default(self):
        pipeline = DepthPerceptionPipeline(PipelineConfig(), _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)
        assert result.boundary_evidence is None

    def test_disabled_even_with_geometry_frame_enabled(self):
        config = PipelineConfig(enable_geometry_frame=True, enable_boundary_geometry=False)
        pipeline = DepthPerceptionPipeline(config, _CALIBRATION)
        left, right = _random_pair()
        result = pipeline.process(left, right)

        assert result.boundary_evidence is None
        assert result.geometry_frame is not None
        assert result.geometry_frame.boundary_evidence is None


class TestLegacyOutputsUnchanged:
    def test_upstream_fields_identical_regardless_of_boundary_geometry_flag(self):
        left, right = _random_pair()

        result_off = DepthPerceptionPipeline(
            _boundary_config(enable_boundary_geometry=False), _CALIBRATION,
        ).process(left, right)
        result_on = DepthPerceptionPipeline(
            _boundary_config(enable_boundary_geometry=True), _CALIBRATION,
        ).process(left, right)

        assert np.array_equal(result_off.disparity_map, result_on.disparity_map)
        assert np.array_equal(result_off.depth_map, result_on.depth_map)
        assert result_off.traversability_mask.decision == result_on.traversability_mask.decision
        assert result_off.confidence == result_on.confidence
        assert result_off.surface_evidence == result_on.surface_evidence  # both None here


# ===================================================================
# 11. No behavioral/semantic labels leak into the contract
# ===================================================================
class TestNoBehavioralLeakage:
    def test_boundary_evidence_field_names_carry_no_behavioral_concept(self):
        fields = {f.name for f in dataclasses.fields(BoundaryEvidence)}
        forbidden = {
            "classification", "status", "decision", "curb", "shoreline",
            "road_edge", "wall_edge", "safe", "traversable", "opening",
        }
        assert not (fields & forbidden)

    def test_boundary_state_values_carry_no_semantic_or_behavioral_label(self):
        state_values = {BoundaryState.OBSERVED_DISCONTINUITY, BoundaryState.NO_DISCONTINUITY,
                         BoundaryState.INSUFFICIENT_EVIDENCE}
        forbidden_terms = ("CURB", "SHORELINE", "ROAD", "WALL_EDGE", "SAFE", "TRAVERSABLE", "OPENING")
        for value in state_values:
            for term in forbidden_terms:
                assert term not in value

    def test_boundary_module_does_not_import_behavioral_or_opening_types(self):
        import_lines = [
            line for line in inspect.getsource(boundary_module).splitlines()
            if line.startswith(("import ", "from "))
        ]
        source = "\n".join(import_lines)
        for forbidden in ("RegionClass", "NavigationDecision", "ThreatAssessor"):
            assert forbidden not in source, f"{forbidden} must not be imported by geometry/boundary.py"
            assert not hasattr(boundary_module, forbidden)

    def test_build_boundary_evidence_never_infers_an_opening(self):
        # Two adjacent OBSERVED_DISCONTINUITY cells must not, by
        # themselves, produce any "opening"/"passage" field or type —
        # the function's own return type never varies by adjacency count.
        depth_map = _step_depth_map(near=1.0, far=10.0, size=40)
        evidence = build_boundary_evidence(
            depth_map, FrameId.CAMERA_OPTICAL_LEFT,
            grid_rows=1, grid_cols=3, min_support_count=1,
            depth_step_threshold_m=0.15, orientation_change_threshold_rad=0.5236,
        )
        assert all(isinstance(cell, BoundaryEvidence) for cell in evidence)
        assert not any(hasattr(cell, "opening") or hasattr(cell, "passage") for cell in evidence)
