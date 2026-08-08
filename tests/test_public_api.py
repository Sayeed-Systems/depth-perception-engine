"""
API boundary tests — public API freeze pass.

Protects the exact contract docs/PUBLIC_API.md documents: which symbols
are Tier 1/2 (top-level, stable), which stay Tier 3 (internal, no
guarantee), and that every duplicated import path resolves to one single
object. Tests the explicit, hardcoded contract — not `dir()` introspection
— so a future accidental export (or accidental removal) fails loudly here
instead of silently drifting.

Point 10 from the public-API-freeze task ("existing Level 0-2 tests remain
unchanged and pass") has no dedicated test here — it's proven by the full
`pytest tests/ -q` run this file is part of, not by any single assertion.
"""

import importlib

import depth_perception_engine as dpe
import depth_perception_engine.calibration as dpe_calibration
import depth_perception_engine.config as dpe_config
import depth_perception_engine.models as dpe_models
import depth_perception_engine.pipeline as dpe_pipeline
import depth_perception_engine.pipeline.api as dpe_pipeline_api
import depth_perception_engine.traversability as dpe_traversability

TIER_1_SYMBOLS = {
    "DepthPerceptionPipeline",
    "PipelineConfig",
    "StereoCalibration",
    "StereoObservation",
    "DepthPerceptionResult",
    "PipelineHealth",
    "TraversabilityResult",
    "ObstacleAssessment",
    "BeamReading",
    "NavigationDecision",
    "RegionClass",
    "RegionStats",
    "TextureClass",
    "load_stereo_calibration",
}

TIER_2_SYMBOLS = {
    "process_stereo_pair",
    "compute_disparity",
    "estimate_depth",
    "detect_obstacles",
    "classify_traversability",
}

EXPECTED_ALL = TIER_1_SYMBOLS | TIER_2_SYMBOLS | {"__version__"}

# symbol -> (top-level module, subpackage module) for identity checks
DUPLICATED_PATHS = {
    "DepthPerceptionPipeline": dpe_pipeline,
    "PipelineConfig": dpe_config,
    "StereoCalibration": dpe_calibration,
    "load_stereo_calibration": dpe_calibration,
    "StereoObservation": dpe_models,
    "DepthPerceptionResult": dpe_models,
    "PipelineHealth": dpe_models,
    "TraversabilityResult": dpe_models,
    "ObstacleAssessment": dpe_models,
    "BeamReading": dpe_models,
    "NavigationDecision": dpe_traversability,
    "RegionClass": dpe_traversability,
    "RegionStats": dpe_traversability,
    "TextureClass": dpe_traversability,
    "process_stereo_pair": dpe_pipeline,
    "compute_disparity": dpe_pipeline,
    "estimate_depth": dpe_pipeline,
    "detect_obstacles": dpe_pipeline,
    "classify_traversability": dpe_pipeline,
}

# Tier 2 functions are also reachable via their defining module directly.
TIER_2_API_MODULE_PATHS = {
    "process_stereo_pair": dpe_pipeline_api,
    "compute_disparity": dpe_pipeline_api,
    "estimate_depth": dpe_pipeline_api,
    "detect_obstacles": dpe_pipeline_api,
    "classify_traversability": dpe_pipeline_api,
}


class TestTier1ImportsFromRoot:
    def test_every_tier1_symbol_is_a_root_attribute(self):
        missing = [name for name in TIER_1_SYMBOLS if not hasattr(dpe, name)]
        assert not missing, f"Tier 1 symbols missing from package root: {missing}"

    def test_every_tier1_symbol_is_importable_directly(self):
        # Proves `from depth_perception_engine import X` works for each,
        # not just that the attribute happens to exist post-import.
        module = importlib.import_module("depth_perception_engine")
        for name in TIER_1_SYMBOLS:
            assert getattr(module, name) is not None


class TestTier2ImportsFromRoot:
    def test_every_tier2_symbol_is_a_root_attribute(self):
        missing = [name for name in TIER_2_SYMBOLS if not hasattr(dpe, name)]
        assert not missing, f"Tier 2 symbols missing from package root: {missing}"

    def test_tier2_symbols_are_callable(self):
        for name in TIER_2_SYMBOLS:
            assert callable(getattr(dpe, name)), f"{name} should be a callable function"


class TestAllIsExactlyIntentional:
    def test_all_equals_expected_set_exactly(self):
        actual = set(dpe.__all__)
        missing = EXPECTED_ALL - actual
        extra = actual - EXPECTED_ALL
        assert not missing, f"Expected in __all__ but missing: {missing}"
        assert not extra, f"Unexpected symbols in __all__ (undocumented export): {extra}"

    def test_all_has_no_duplicates(self):
        assert len(dpe.__all__) == len(set(dpe.__all__))

    def test_utils_is_not_in_all(self):
        assert "utils" not in dpe.__all__


class TestImportIdentity:
    """`from depth_perception_engine import X` and the subpackage-path
    equivalent must be the exact same object — no duplicate definitions."""

    def test_every_duplicated_symbol_has_identical_object_identity(self):
        mismatches = []
        for name, subpackage_module in DUPLICATED_PATHS.items():
            top_level_obj = getattr(dpe, name)
            subpackage_obj = getattr(subpackage_module, name)
            if top_level_obj is not subpackage_obj:
                mismatches.append(name)
        assert not mismatches, f"Identity mismatch (duplicate definition?) for: {mismatches}"

    def test_tier2_functions_also_identical_via_pipeline_api_module(self):
        mismatches = []
        for name, api_module in TIER_2_API_MODULE_PATHS.items():
            if getattr(dpe, name) is not getattr(api_module, name):
                mismatches.append(name)
        assert not mismatches, f"Identity mismatch against pipeline.api for: {mismatches}"

    def test_documented_readme_example_identity(self):
        """The exact assertion docs/PUBLIC_API.md and this task's spec both use."""
        from depth_perception_engine import DepthPerceptionPipeline as A
        from depth_perception_engine.pipeline import DepthPerceptionPipeline as B

        assert A is B


class TestNoInternalImportNeededForFullResultInterpretation:
    """Constructs a real result and reads every field using ONLY top-level
    imports — proves a Tier-1-only consumer (a future mp01_perception
    wrapper) never needs traversability.types, quality.*, depth.*,
    stereo.*, or utils.*."""

    def test_full_result_readable_with_tier1_symbols_only(self, config, calibration, stereo_pair):
        from depth_perception_engine import (
            BeamReading,
            DepthPerceptionPipeline,
            DepthPerceptionResult,
            NavigationDecision,
            ObstacleAssessment,
            PipelineConfig,
            RegionClass,
            RegionStats,
            StereoCalibration,
            TextureClass,
            TraversabilityResult,
        )

        assert isinstance(config, PipelineConfig)
        assert isinstance(calibration, StereoCalibration)

        left, right = stereo_pair
        pipeline = DepthPerceptionPipeline(config, calibration)
        result = pipeline.process(left, right)

        assert isinstance(result, DepthPerceptionResult)

        traversability = result.traversability_mask
        assert isinstance(traversability, TraversabilityResult)
        assert isinstance(traversability.decision, NavigationDecision)
        for region in traversability.regions.values():
            assert isinstance(region, RegionStats)
            assert isinstance(region.classification, RegionClass)
            assert isinstance(region.texture_class, TextureClass)

        obstacles = result.obstacles
        assert isinstance(obstacles, ObstacleAssessment)
        for beam in obstacles.beams:
            assert isinstance(beam, BeamReading)
            assert isinstance(beam.status, str)  # plain string status, not a Tier-1 enum today

        health = pipeline.health()
        assert health.frames_processed == 1


class TestNoImportSideEffects:
    def test_import_does_not_touch_cv2_video_capture_or_gui(self, mocker):
        import cv2

        video_capture_spy = mocker.spy(cv2, "VideoCapture")
        imshow_spy = mocker.spy(cv2, "imshow")

        importlib.reload(dpe)

        video_capture_spy.assert_not_called()
        imshow_spy.assert_not_called()

    def test_import_does_not_pull_in_rclpy(self):
        import sys

        importlib.reload(dpe)
        assert "rclpy" not in sys.modules

    def test_reimport_is_idempotent_and_side_effect_free(self):
        # Re-importing must not raise, print, or leave the module in a
        # different state — a lightweight proxy for "no runtime side
        # effects" without needing to enumerate every possible effect.
        first = importlib.import_module("depth_perception_engine")
        second = importlib.reload(first)
        assert second.__version__ == first.__version__
        assert set(second.__all__) == set(first.__all__)


class TestCanonicalClassNaming:
    def test_depth_perception_pipeline_is_exported(self):
        assert hasattr(dpe, "DepthPerceptionPipeline")

    def test_no_depth_perception_engine_symbol_exists(self):
        assert not hasattr(dpe, "DepthPerceptionEngine")
        assert "DepthPerceptionEngine" not in dpe.__all__

    def test_no_other_top_level_symbol_shaped_like_a_second_pipeline_class(self):
        # A second class *named like a pipeline/engine itself* (ending in
        # "Pipeline"/"Engine", e.g. a hypothetical "DepthPerceptionEngine"
        # or "GeometryPipeline") would be exactly the naming-churn this
        # task explicitly rejects. Deliberately a suffix check, not a
        # substring check — PipelineConfig/PipelineHealth are legitimate
        # value types *associated with* the one pipeline, not competing
        # processing classes, and must not trip this.
        pipeline_like = [
            name for name in dpe.__all__
            if name != "DepthPerceptionPipeline" and (name.endswith("Pipeline") or name.endswith("Engine"))
        ]
        assert not pipeline_like, f"Unexpected competing high-level class name(s): {pipeline_like}"


class TestInternalContractsStayNonPublic:
    """Level 3 (Phase E1) geometry/calibration-decomposition contracts are
    real, tested, and importable via their own submodules — but must never
    be promoted to the top-level package until something actually produces
    them (see docs/LEVEL3_PUBLIC_API.md)."""

    INTERNAL_SYMBOLS = [
        "PointCloud", "ObstacleCloud", "FreeSpaceRays", "GeometryMetrics",
        "RigCalibration", "CameraIntrinsics", "StereoExtrinsics",
        "RectificationParameters", "CameraModel", "RigidTransform", "FrameId",
        # internal stage/helper classes
        "DisparityEngine", "RectificationEngine", "DepthEstimator",
        "DistanceReader", "ThreatAssessor", "RegionAnalyzer",
        "SceneInterpreter", "FrameSplitter",
        # Level 3 producers (E2-E6) — geometry.* stays Tier 3 throughout,
        # see docs/LEVEL3_PUBLIC_API.md's E3 update for why PointCloud
        # itself was deliberately not promoted even once it had a real
        # producer.
        "PointCloudBuilder", "transform_point_cloud", "build_obstacle_cloud",
        "build_free_space_rays", "build_geometry_metrics", "GeometryQuality",
        "classify_geometry_quality",
    ]

    def test_no_internal_symbol_is_a_root_attribute(self):
        leaked = [name for name in self.INTERNAL_SYMBOLS if hasattr(dpe, name)]
        assert not leaked, f"Internal symbols leaked onto the package root: {leaked}"

    def test_no_internal_symbol_is_in_all(self):
        leaked = [name for name in self.INTERNAL_SYMBOLS if name in dpe.__all__]
        assert not leaked, f"Internal symbols leaked into __all__: {leaked}"

    def test_geometry_and_quality_submodules_are_still_reachable_as_internal_paths(self):
        # Confirms these weren't accidentally broken — just not promoted.
        from depth_perception_engine.geometry import PointCloud
        from depth_perception_engine.quality.frame_quality import looks_like_garbage_frame

        assert PointCloud is not None
        assert callable(looks_like_garbage_frame)


class TestPackageDocstringMatchesExports:
    def test_every_tier1_symbol_name_appears_in_module_docstring(self):
        doc = dpe.__doc__
        missing = [name for name in TIER_1_SYMBOLS if name not in doc]
        assert not missing, f"Tier 1 symbols not mentioned in package docstring: {missing}"

    def test_docstring_states_the_canonical_top_level_import(self):
        assert "from depth_perception_engine import" in dpe.__doc__

    def test_docstring_documents_no_depth_perception_engine_alias(self):
        assert "DepthPerceptionEngine" in dpe.__doc__  # explicitly states none exists
        assert "no `DepthPerceptionEngine` symbol" in dpe.__doc__.replace("\n", " ")
