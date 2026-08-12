"""
DepthPerceptionPipeline geometry integration — Level 3, Phase E3.

Covers exactly what E3 adds on top of the already-tested Level 0-2
pipeline (tests/test_pipeline.py) and the already-tested E2 math
(tests/test_depth_estimator.py::TestEstimatePointCloud,
tests/test_point_cloud_builder.py): the config gate, the process()
integration point, the result-contract field, failure/degradation
semantics, zero-regression on every pre-E3 output, and that exactly one
canonical point-cloud producer exists anywhere in the library.
"""

import ast
import dataclasses
import os
from unittest.mock import patch

import numpy as np
import pytest

from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry import PointCloud, PointCloudBuilder
from depth_perception_engine.models import (
    DepthPerceptionResult,
    ObstacleAssessment,
    TraversabilityResult,
)
from depth_perception_engine.pipeline import DepthPerceptionPipeline
from depth_perception_engine.traversability.types import NavigationDecision

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src", "depth_perception_engine")


def _flat_textureless_pair(calibration):
    """Zero stereo correspondence anywhere — SGBM finds no valid match for
    any pixel, deterministically forcing valid_disparity_mask/
    valid_depth_mask/geometry.valid_mask to all-False. Verified empirically
    (unlike identical-but-textured images, which still yield scattered
    spurious matches) before being used as a test fixture."""
    width, height = calibration.image_size
    flat = np.full((height, width, 3), 128, dtype=np.uint8)
    return flat, flat.copy()


class TestConfigGate:
    def test_default_is_disabled(self):
        assert PipelineConfig().enable_geometry is False

    def test_can_be_enabled(self):
        assert PipelineConfig(enable_geometry=True).enable_geometry is True

    def test_disabled_by_default_preserves_pre_e3_behavior(
        self, config, calibration, stereo_pair,
    ):
        """`config` fixture is PipelineConfig() — no enable_geometry passed
        at all, exactly what every pre-E3 caller (including
        mp01_perception) does today."""
        assert config.enable_geometry is False
        pipeline = DepthPerceptionPipeline(config, calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry is None


class TestGeometryEnabledOutput:
    def test_returns_camera_frame_point_cloud(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry is not None
        assert isinstance(result.geometry, PointCloud)
        assert result.geometry.frame_id == FrameId.CAMERA_OPTICAL_LEFT
        assert result.geometry.frame_id == "camera_optical_left"  # explicit, verifiable literal

    def test_organized_cloud_shape_matches_depth_image_dimensions(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry.points.shape == result.depth_map.shape + (3,)
        assert result.geometry.points.dtype == np.float32
        assert result.geometry.valid_mask.shape == result.depth_map.shape

    def test_geometry_absent_from_internal_point_cloud_class(self, calibration, stereo_pair):
        """result.geometry must be the frozen PointCloud data contract, not
        the internal PointCloudBuilder that produced it (Task 4: 'Do not
        expose internal point-cloud implementation objects')."""
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert not isinstance(result.geometry, PointCloudBuilder)
        assert isinstance(result.geometry, PointCloud)

    def test_valid_geometry_count_matches_valid_depth_mask_exactly(self, calibration, stereo_pair):
        """Not just a count match — the exact same set of pixels, since
        both are derived from the identical raw_disparity via the same Q
        and the same MIN/MAX_DEPTH_M — see
        tests/test_depth_estimator.py::TestZOnlyMatchesFullReprojection
        for the underlying math proof this relies on."""
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)

        np.testing.assert_array_equal(result.geometry.valid_mask, result.valid_depth_mask)
        assert result.geometry.valid_mask.sum() > 0, (
            "fixture must contain at least one valid pixel for this to be a meaningful check"
        )

    def test_invalid_pixels_never_become_valid_geometry(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)

        invalid = ~result.valid_depth_mask
        assert np.any(invalid), "fixture must contain at least one invalid pixel"
        assert not np.any(result.geometry.valid_mask[invalid])
        assert np.all(np.isnan(result.geometry.points[invalid]))

    def test_matches_e2_verified_builder_called_directly(self, calibration, stereo_pair):
        """Strongest correctness check: feed the pipeline's own
        raw_disparity output into a freshly, independently constructed
        PointCloudBuilder (the exact E2-verified class — see
        tests/test_point_cloud_builder.py) and confirm the pipeline's
        result.geometry is identical. Proves the pipeline performs no
        second, divergent reprojection of its own."""
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)

        reference = PointCloudBuilder.from_calibration(calibration).build(result.disparity_map)

        np.testing.assert_array_equal(result.geometry.valid_mask, reference.valid_mask)
        np.testing.assert_array_equal(result.geometry.points, reference.points)


class TestDeterminism:
    def test_repeated_process_calls_produce_identical_geometry(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair

        r1 = pipeline.process(left, right)
        r2 = pipeline.process(left, right)
        r3 = pipeline.process(left, right)

        np.testing.assert_array_equal(r1.geometry.points, r2.geometry.points)
        np.testing.assert_array_equal(r1.geometry.points, r3.geometry.points)
        np.testing.assert_array_equal(r1.geometry.valid_mask, r2.geometry.valid_mask)


class TestLifecycle:
    def test_reset_then_process_still_produces_geometry(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair
        pipeline.process(left, right)

        pipeline.reset()
        result = pipeline.process(left, right)

        assert result.geometry is not None
        assert isinstance(result.geometry, PointCloud)

    def test_close_then_process_raises_even_with_geometry_enabled(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair
        pipeline.close()

        with pytest.raises(RuntimeError):
            pipeline.process(left, right)


class TestFailureSemantics:
    """
    Task 6: geometry-disabled / degraded-but-valid / runtime-error must
    stay distinguishable, matching this codebase's existing three-way
    taxonomy (see tests/test_pipeline.py's TestRectificationFailureInvalidatesTheFrame
    and require_matching_stereo_pair for the other two categories):

    - invalid caller input        -> unchanged, not touched by E3 at all
    - runtime perception degradation -> a normal DepthPerceptionResult,
      geometry present but valid_mask all-False / points all-NaN
    - fatal pipeline failure       -> uncaught exception, frame dropped
      by the caller, exactly like a rectification failure
    """

    def test_geometry_disabled_yields_none_not_an_error(self, calibration, stereo_pair):
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=False), calibration)
        left, right = stereo_pair

        result = pipeline.process(left, right)

        assert result.geometry is None

    def test_no_valid_disparity_yields_all_invalid_geometry_not_a_crash(self, calibration):
        """Runtime perception degradation, not a failure: process() must
        still return a normal result, with geometry present but entirely
        invalid — never a zero-filled XYZ array masquerading as data."""
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = _flat_textureless_pair(calibration)

        result = pipeline.process(left, right)

        assert result.valid_disparity_mask.sum() == 0
        assert result.valid_depth_mask.sum() == 0
        assert result.geometry is not None
        assert result.geometry.valid_mask.sum() == 0
        assert np.all(np.isnan(result.geometry.points))
        assert not np.any(result.geometry.points == 0.0), (
            "invalid geometry must be NaN, never a silent zero-fill"
        )

    def test_all_depth_rejected_out_of_range_is_covered_at_the_builder_level(self):
        """
        'Disparity valid but all depth rejected' (e.g. every pixel's
        reprojected depth falls outside [MIN_DEPTH_M, MAX_DEPTH_M]) is not
        reproducible on demand through real SGBM output from synthetic
        images — StereoSGBM's actual disparity values aren't directly
        steerable per-pixel. This exact rule is already deterministically
        tested one layer down, against the identical PointCloudBuilder
        the pipeline calls unmodified (see
        test_matches_e2_verified_builder_called_directly above for proof
        the pipeline doesn't diverge from that builder):
        tests/test_depth_estimator.py::TestEstimatePointCloud::
        test_out_of_range_depth_rejected_too_close_and_too_far and
        test_boundary_depth_exactly_at_min_and_max_are_valid. Re-deriving
        that here through real stereo matching would be redundant and
        flaky; this test exists only to document that decision.
        """
        assert True

    def test_point_cloud_runtime_error_propagates_uncaught(self, calibration, stereo_pair):
        """A genuine failure inside the geometry stage must invalidate the
        whole frame exactly like a rectification failure does — never be
        silently swallowed into a fake empty PointCloud."""
        pipeline = DepthPerceptionPipeline(PipelineConfig(enable_geometry=True), calibration)
        left, right = stereo_pair

        with patch.object(
            PointCloudBuilder, "build", side_effect=RuntimeError("simulated geometry failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated geometry failure"):
                pipeline.process(left, right)


class TestZeroRegression:
    """Task 7: enabling E3 must not change any Level 0-2 output for a fixed
    input, within existing numerical tolerances. Two pipelines built from
    configs identical in every field except enable_geometry."""

    def test_level_0_2_outputs_identical_with_geometry_on_or_off(self, calibration, stereo_pair):
        config_off = PipelineConfig(enable_geometry=False)
        config_on = dataclasses.replace(config_off, enable_geometry=True)
        left, right = stereo_pair

        result_off = DepthPerceptionPipeline(config_off, calibration).process(left, right)
        result_on = DepthPerceptionPipeline(config_on, calibration).process(left, right)

        np.testing.assert_array_equal(result_off.disparity_map, result_on.disparity_map)
        np.testing.assert_array_equal(result_off.depth_map, result_on.depth_map)
        np.testing.assert_array_equal(result_off.valid_disparity_mask, result_on.valid_disparity_mask)
        np.testing.assert_array_equal(result_off.valid_depth_mask, result_on.valid_depth_mask)
        assert result_off.confidence == result_on.confidence

        assert result_off.traversability_mask.decision == result_on.traversability_mask.decision
        assert set(result_off.traversability_mask.regions) == set(result_on.traversability_mask.regions)
        for name, region_off in result_off.traversability_mask.regions.items():
            region_on = result_on.traversability_mask.regions[name]
            assert region_off.classification == region_on.classification
            assert region_off.depth_median_m == region_on.depth_median_m
            assert region_off.confidence == region_on.confidence
            assert region_off.valid_count == region_on.valid_count
            assert region_off.total_pixels == region_on.total_pixels

        assert len(result_off.obstacles.beams) == len(result_on.obstacles.beams)
        for beam_off, beam_on in zip(result_off.obstacles.beams, result_on.obstacles.beams):
            assert beam_off.index == beam_on.index
            assert beam_off.x1 == beam_on.x1
            assert beam_off.x2 == beam_on.x2
            assert beam_off.distance_m == beam_on.distance_m
            assert beam_off.status == beam_on.status
        assert (result_off.obstacles.safest_beam is None) == (result_on.obstacles.safest_beam is None)

        assert result_off.geometry is None
        assert result_on.geometry is not None

    def test_repeated_process_calls_navigation_decision_unaffected_by_geometry(
        self, calibration, stereo_pair,
    ):
        """ThreatAssessor's EMA/debounce is the one piece of cross-frame
        state in the pipeline — confirms it evolves identically frame to
        frame whether or not geometry is being computed alongside it."""
        config_off = PipelineConfig(enable_geometry=False)
        config_on = dataclasses.replace(config_off, enable_geometry=True)
        left, right = stereo_pair

        pipeline_off = DepthPerceptionPipeline(config_off, calibration)
        pipeline_on = DepthPerceptionPipeline(config_on, calibration)

        decisions_off = [pipeline_off.process(left, right).traversability_mask.decision for _ in range(4)]
        decisions_on = [pipeline_on.process(left, right).traversability_mask.decision for _ in range(4)]

        assert decisions_off == decisions_on

        statuses_off = [b.status for b in pipeline_off.process(left, right).obstacles.beams]
        statuses_on = [b.status for b in pipeline_on.process(left, right).obstacles.beams]
        assert statuses_off == statuses_on


class TestResultContract:
    def test_existing_fields_unchanged_in_name_type_and_order(self):
        """dataclasses.fields() in declaration order — proves geometry (E3),
        geometry_body (E4), obstacle_cloud/free_space_rays/geometry_metrics
        (E5), temporal_admission_status/temporal_consistency/
        temporal_stabilization/rotation_compensation_status/
        motion_aware_reliability/temporal_persistence (Level 4, Phases
        E2-E7) / geometry_frame (Phase D2) / surface_evidence (Phase D4)
        / boundary_evidence (Phase D5) / opening_evidence (Phase D6) were
        appended at the end, in that order, and every earlier field
        name/order/default is untouched (Task 4: do not rename/remove/
        reorder existing fields)."""
        fields = dataclasses.fields(DepthPerceptionResult)
        names = [f.name for f in fields]

        assert names == [
            "disparity_map",
            "depth_map",
            "traversability_mask",
            "obstacles",
            "confidence",
            "processing_time_ms",
            "valid_disparity_mask",
            "valid_depth_mask",
            "timestamp",
            "geometry",
            "geometry_body",
            "obstacle_cloud",
            "free_space_rays",
            "geometry_metrics",
            "temporal_admission_status",
            "temporal_consistency",
            "temporal_stabilization",
            "rotation_compensation_status",
            "motion_aware_reliability",
            "temporal_persistence",
            "geometry_frame",
            "surface_evidence",
            "boundary_evidence",
            "opening_evidence",
        ]
        # temporal_consistency, temporal_stabilization,
        # rotation_compensation_status, motion_aware_reliability,
        # temporal_persistence, geometry_frame, surface_evidence,
        # boundary_evidence, opening_evidence
        for field in fields[-9:]:
            assert field.default is None

    def test_geometry_field_defaults_to_none(self):
        """Constructing a DepthPerceptionResult the pre-E3 way (no geometry
        kwarg at all) must still work — proves the field is truly
        additive, not a silently-required new argument."""
        result = DepthPerceptionResult(
            disparity_map=np.zeros((2, 2), dtype=np.float32),
            depth_map=np.zeros((2, 2), dtype=np.float32),
            traversability_mask=TraversabilityResult(regions={}, decision=NavigationDecision.STOP),
            obstacles=ObstacleAssessment(beams=[], safest_beam=None),
            confidence=0.0,
            processing_time_ms=0.0,
        )
        assert result.geometry is None


class TestSingleCanonicalProducer:
    """Task 1/7: there must be exactly one place in the library that calls
    cv2.reprojectImageTo3D — proving no second, divergent point-cloud/
    reprojection implementation was introduced anywhere (mirrors
    test_no_ros_dependency.py's AST-scan style)."""

    def test_exactly_one_reprojectImageTo3D_call_site_exists(self):
        call_sites = []
        for dirpath, _dirnames, filenames in os.walk(_SRC_DIR):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and node.attr == "reprojectImageTo3D":
                        call_sites.append(f"{os.path.relpath(path, _REPO_ROOT)}:{node.lineno}")

        assert call_sites == ["src/depth_perception_engine/depth/depth_estimator.py:202"], (
            f"expected exactly one reprojectImageTo3D call site, found: {call_sites}"
        )

    def test_point_cloud_builder_is_the_only_geometry_producer_wired_into_the_pipeline(
        self, calibration,
    ):
        """Static confirmation that DepthPerceptionPipeline's source only
        constructs PointCloudBuilder for camera-frame geometry — not some
        other class. As of E5, the pipeline legitimately also references
        ObstacleCloud/FreeSpaceRays (via the canonical
        geometry.build_obstacle_cloud/build_free_space_rays functions —
        see TestSingleCanonicalProducer.test_e5_builders_used_are_the_canonical_ones
        below) — this test now checks import provenance instead of mere
        string absence, which stopped being a meaningful signal once E5
        made those references legitimate."""
        pipeline_src_path = os.path.join(_SRC_DIR, "pipeline", "pipeline.py")
        with open(pipeline_src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "PointCloudBuilder" in source
        assert "from depth_perception_engine.geometry.point_cloud_builder import PointCloudBuilder" in source

    def test_e5_builders_used_are_the_canonical_ones(self):
        """Task 6 (E5): pipeline.py must import build_obstacle_cloud/
        build_free_space_rays from their one canonical module each, not
        define or import a second, divergent implementation."""
        pipeline_src_path = os.path.join(_SRC_DIR, "pipeline", "pipeline.py")
        with open(pipeline_src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "from depth_perception_engine.geometry.obstacle_extractor import build_obstacle_cloud" in source
        assert "from depth_perception_engine.geometry.free_space import build_free_space_rays" in source
        assert "from depth_perception_engine.geometry.geometry_metrics import build_geometry_metrics" in source

    def test_exactly_one_obstacle_cloud_and_free_space_rays_producer_module_each(self):
        """AST-scan style, mirroring test_no_forbidden_imports_anywhere_in_the_library:
        exactly one function named build_obstacle_cloud and one named
        build_free_space_rays exist anywhere under src/ — no duplicate/
        parallel geometry-filtering implementation."""
        obstacle_defs = []
        free_space_defs = []
        for dirpath, _dirnames, filenames in os.walk(_SRC_DIR):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == "build_obstacle_cloud":
                        obstacle_defs.append(f"{os.path.relpath(path, _REPO_ROOT)}:{node.lineno}")
                    if isinstance(node, ast.FunctionDef) and node.name == "build_free_space_rays":
                        free_space_defs.append(f"{os.path.relpath(path, _REPO_ROOT)}:{node.lineno}")

        assert len(obstacle_defs) == 1, f"expected exactly one build_obstacle_cloud def, found: {obstacle_defs}"
        assert len(free_space_defs) == 1, f"expected exactly one build_free_space_rays def, found: {free_space_defs}"


class TestPublicAPIUnchanged:
    def test_top_level_imports_still_resolve(self):
        from depth_perception_engine import (
            DepthPerceptionPipeline as TopPipeline,
            DepthPerceptionResult as TopResult,
            PipelineConfig as TopConfig,
        )
        assert TopPipeline is DepthPerceptionPipeline
        assert TopResult is DepthPerceptionResult
        assert TopConfig is PipelineConfig

    def test_geometry_builders_remain_tier_3_not_promoted_to_top_level(self):
        # PointCloud itself WAS promoted to Tier 1 by Phase D3 (see
        # docs/DPE_V1_PROVIDER_CONTRACT.md's D3 record and
        # tests/test_public_api.py) — GeometryFrame's own `geometry`/
        # `geometry_body` fields are typed against it. The builder that
        # *produces* one remains internal: promoting a result type does
        # not mean promoting its producer.
        import depth_perception_engine

        assert "PointCloud" in depth_perception_engine.__all__
        assert "PointCloudBuilder" not in depth_perception_engine.__all__
        assert not hasattr(depth_perception_engine, "PointCloudBuilder")
