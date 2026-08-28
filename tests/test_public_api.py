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
import depth_perception_engine.frames as dpe_frames
import depth_perception_engine.geometry as dpe_geometry
import depth_perception_engine.models as dpe_models
import depth_perception_engine.pipeline as dpe_pipeline
import depth_perception_engine.pipeline.api as dpe_pipeline_api
import depth_perception_engine.temporal as dpe_temporal
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
    # Phase D2: GeometryFrame — the final, authoritative DPE V1 provider
    # contract — plus the Level 4 temporal result contracts it carries.
    "GeometryFrame",
    "TemporalConsistency",
    "TemporalConsistencyState",
    "TemporalStabilization",
    "TemporalStabilizationState",
    "RotationCompensationStatus",
    "MotionAwareReliability",
    "MotionAwareReliabilityState",
    "TemporalPersistence",
    "TemporalPersistenceState",
    "TemporalPersistenceCellState",
    # Phase D3: the four Level 3 geometry result types GeometryFrame's own
    # fields are typed against, plus the two new neutral evidence
    # contracts extracted from traversability/obstacles.
    "PointCloud",
    "ObstacleCloud",
    "FreeSpaceRays",
    "GeometryMetrics",
    "RegionEvidence",
    "ClearanceEvidence",
    # Phase D4: local surface-normal/planarity evidence.
    "SurfaceEvidence",
    # Phase D5: geometric boundary/discontinuity evidence, plus its two
    # state-constant classes.
    "BoundaryEvidence",
    "BoundaryState",
    "BoundaryDirection",
    # Phase D6: geometric opening/passage-structure evidence.
    "OpeningEvidence",
    # Phase D7: ClearanceEvidence's own coverage/support state-constant
    # class (ClearanceEvidence itself already Tier 1 since D3).
    "ClearanceSupportState",
    # Phase D8: the structured geometric quality/uncertainty rollup.
    "GeometryFrameQuality",
    "GeometryFrameQualityState",
    # Phase D10: FrameId — the canonical public vocabulary required to
    # interpret every frame_id field across GeometryFrame's own type
    # graph (see docs/DPE_V1_PROVIDER_CONTRACT.md's D9/D10 records).
    "FrameId",
    # Phase D13: MotionHint — required to construct a complete public
    # INPUT contract (StereoObservation.motion_hint/.motion_hints,
    # DepthPerceptionPipeline.process()'s own motion_hint/motion_hints
    # parameters) without an internal depth_perception_engine.temporal
    # import. Not part of GeometryFrame's own output type graph (see
    # docs/DPE_V1_PROVIDER_CONTRACT.md's D13 record).
    "MotionHint",
}

TIER_2_SYMBOLS = {
    "process_stereo_pair",
    "compute_disparity",
    "estimate_depth",
    "detect_obstacles",
    "classify_traversability",
}

# The dual-interface refactor's own public entry point: the standalone /
# sensor-facing convenience interface (docs/DUAL_INTERFACE_ARCHITECTURE.md).
# Deliberately its own set rather than a Tier 1 addition — it is NOT part of
# the core/embedded contract an external consumer (a future
# hybrid_perception_engine) uses, and it is resolved LAZILY by the root
# package's PEP 562 __getattr__ so that importing the core never loads it.
# Its canonical import path is depth_perception_engine.standalone; the root
# export exists for symmetry with every other supported public symbol.
STANDALONE_SYMBOLS = {
    "StandaloneStereoInterface",
}

EXPECTED_ALL = TIER_1_SYMBOLS | TIER_2_SYMBOLS | STANDALONE_SYMBOLS | {"__version__"}

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
    "GeometryFrame": dpe_geometry,
    "TemporalConsistency": dpe_temporal,
    "TemporalConsistencyState": dpe_temporal,
    "TemporalStabilization": dpe_temporal,
    "TemporalStabilizationState": dpe_temporal,
    "RotationCompensationStatus": dpe_temporal,
    "MotionAwareReliability": dpe_temporal,
    "MotionAwareReliabilityState": dpe_temporal,
    "TemporalPersistence": dpe_temporal,
    "TemporalPersistenceState": dpe_temporal,
    "TemporalPersistenceCellState": dpe_temporal,
    "PointCloud": dpe_geometry,
    "ObstacleCloud": dpe_geometry,
    "FreeSpaceRays": dpe_geometry,
    "GeometryMetrics": dpe_geometry,
    "RegionEvidence": dpe_geometry,
    "ClearanceEvidence": dpe_geometry,
    "SurfaceEvidence": dpe_geometry,
    "BoundaryEvidence": dpe_geometry,
    "BoundaryState": dpe_geometry,
    "BoundaryDirection": dpe_geometry,
    "OpeningEvidence": dpe_geometry,
    "ClearanceSupportState": dpe_geometry,
    "GeometryFrameQuality": dpe_geometry,
    "GeometryFrameQualityState": dpe_geometry,
    "FrameId": dpe_frames,
    "MotionHint": dpe_temporal,
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
        "RigCalibration", "CameraIntrinsics", "StereoExtrinsics",
        "RectificationParameters", "CameraModel", "RigidTransform",
        # FrameId (unlike RigidTransform) was promoted to Tier 1 at Phase
        # D10 — it is the canonical public vocabulary required to
        # interpret every frame_id field across GeometryFrame's own type
        # graph, and promoting it required only API/export/test/doc
        # changes, no behavioral change (see docs/DPE_V1_PROVIDER_CONTRACT.md's
        # D10 record). RigidTransform stays Tier 3 — it is a pipeline
        # CONSTRUCTOR input (body_T_camera_left), never part of
        # GeometryFrame's own output type graph.
        # internal stage/helper classes
        "DisparityEngine", "RectificationEngine", "DepthEstimator",
        "DistanceReader", "ThreatAssessor", "RegionAnalyzer",
        "SceneInterpreter", "FrameSplitter",
        # Level 3 producers/algorithms stay Tier 3 even after Phase D3
        # promoted their RESULT types (PointCloud, ObstacleCloud,
        # FreeSpaceRays, GeometryMetrics — see TIER_1_SYMBOLS): a caller
        # consuming GeometryFrame reads these results, it never
        # constructs a PointCloudBuilder or calls build_obstacle_cloud()
        # itself. See docs/DPE_V1_PROVIDER_CONTRACT.md's D3 update.
        "PointCloudBuilder", "transform_point_cloud", "build_obstacle_cloud",
        "build_free_space_rays", "build_geometry_metrics", "GeometryQuality",
        "classify_geometry_quality",
        # Phase D4 — same reasoning: SurfaceEvidence (the result type) is
        # Tier 1, build_surface_evidence (the algorithm) stays Tier 3.
        "build_surface_evidence",
        # Phase D5 — same reasoning: BoundaryEvidence (the result type) is
        # Tier 1, build_boundary_evidence (the algorithm) stays Tier 3.
        "build_boundary_evidence",
        # Phase D6 — same reasoning: OpeningEvidence (the result type) is
        # Tier 1, build_opening_evidence (the algorithm) stays Tier 3.
        "build_opening_evidence",
        # MotionHint (unlike RigidTransform) was promoted to Tier 1 at
        # Phase D13 — it is required to construct a complete public INPUT
        # contract (StereoObservation.motion_hint/.motion_hints,
        # DepthPerceptionPipeline.process()'s own motion_hint/
        # motion_hints parameters); see docs/DPE_V1_PROVIDER_CONTRACT.md's
        # D13 record.
        # Level 4 (Phase E2) — temporal history stays Tier 3 too: it is
        # infrastructure DepthPerceptionPipeline owns internally, not a
        # symbol a normal caller is meant to construct directly.
        "TemporalRecord", "TemporalHistory", "TemporalAdmissionStatus",
        # Level 4 algorithm implementation functions stay Tier 3 even
        # after Phase D2 promoted their RESULT/EVIDENCE dataclasses
        # (TemporalConsistency, TemporalStabilization,
        # RotationCompensationStatus, MotionAwareReliability,
        # TemporalPersistence — see TIER_1_SYMBOLS) — a caller consuming
        # GeometryFrame reads these results, it never calls the functions
        # that produce them; those remain internal to process(). See
        # docs/DPE_V1_PROVIDER_CONTRACT.md's D2 update.
        "compute_temporal_consistency",
        "compute_temporal_stabilization",
        "compute_rotation_compensation",
        # compensate_prior_geometry_with_payload lives in the same E5
        # module (shares its reprojection math) but is a new symbol added
        # at E7; same Tier 3 reasoning.
        "compensate_prior_geometry_with_payload",
        "compute_motion_aware_reliability",
        # Level 4 (Phase E7) — the tracker itself is stateful
        # infrastructure DepthPerceptionPipeline owns internally, not a
        # symbol a normal caller is meant to construct directly — stays
        # Tier 3 even though its output type (TemporalPersistence) is
        # now Tier 1.
        "TemporalPersistenceTracker",
    ]

    def test_no_internal_symbol_is_a_root_attribute(self):
        leaked = [name for name in self.INTERNAL_SYMBOLS if hasattr(dpe, name)]
        assert not leaked, f"Internal symbols leaked onto the package root: {leaked}"

    def test_no_internal_symbol_is_in_all(self):
        leaked = [name for name in self.INTERNAL_SYMBOLS if name in dpe.__all__]
        assert not leaked, f"Internal symbols leaked into __all__: {leaked}"

    def test_geometry_and_quality_submodules_are_still_reachable_as_internal_paths(self):
        # Confirms these weren't accidentally broken — just not promoted.
        # PointCloudBuilder (the producer), unlike PointCloud (the result
        # type it builds, promoted Tier 1 by Phase D3), stays internal.
        from depth_perception_engine.geometry import PointCloudBuilder
        from depth_perception_engine.quality.frame_quality import looks_like_garbage_frame

        assert PointCloudBuilder is not None
        assert callable(looks_like_garbage_frame)


class TestGeometryFrameTypeGraphIsFullyPublic:
    """Phase D13 structural guard: automatically (not by hand-maintained
    list) proves every DPE-owned type reachable from GeometryFrame's own
    field annotations is Tier 1 — so a future phase that adds a new
    GeometryFrame field typed against a Tier-3/internal class fails HERE,
    at the type-graph level, rather than only being caught if someone
    remembers to update TIER_1_SYMBOLS by hand. This is the literal
    "cannot make GeometryFrame depend on internal-only types" guard the
    D13 task asked for.

    Method: typing.get_type_hints(GeometryFrame) resolves every field's
    real class (unwrapping Optional/List/Dict via get_origin/get_args
    recursively); any resolved class whose own module lives under
    depth_perception_engine is required to be a Tier 1 symbol, identical
    object identity to depth_perception_engine.<name>. Builtins (str,
    float) and numpy.ndarray are skipped — they carry no DPE-owned
    contract to promote.
    """

    @staticmethod
    def _dpe_owned_leaf_classes(tp):
        import typing

        origin = typing.get_origin(tp)
        if origin is None:
            if isinstance(tp, type) and tp.__module__.startswith("depth_perception_engine"):
                yield tp
            return
        for arg in typing.get_args(tp):
            if arg is type(None):
                continue
            yield from TestGeometryFrameTypeGraphIsFullyPublic._dpe_owned_leaf_classes(arg)

    def test_every_dpe_owned_type_reachable_from_geometry_frame_is_tier1(self):
        import typing

        from depth_perception_engine.geometry.provider import GeometryFrame

        hints = typing.get_type_hints(GeometryFrame)
        offenders = []
        for field_name, tp in hints.items():
            for cls in self._dpe_owned_leaf_classes(tp):
                if cls.__name__ not in TIER_1_SYMBOLS:
                    offenders.append(f"{field_name}: {cls.__module__}.{cls.__qualname__}")
                elif getattr(dpe, cls.__name__) is not cls:
                    offenders.append(f"{field_name}: {cls.__name__} is Tier 1 but object identity mismatch")
        assert not offenders, (
            "GeometryFrame depends on non-Tier-1 (or identity-mismatched) "
            f"DPE-owned type(s): {offenders}"
        )

    def test_at_least_the_known_evidence_families_were_actually_checked(self):
        # Sanity guard on the guard itself: proves the recursive walk
        # above is actually descending into Optional[List[...]]/
        # Optional[Dict[str, ...]] wrappers, not silently matching zero
        # fields (a test that can never fail is not a test).
        import typing

        from depth_perception_engine.geometry.provider import GeometryFrame

        hints = typing.get_type_hints(GeometryFrame)
        found_names = {
            cls.__name__
            for tp in hints.values()
            for cls in self._dpe_owned_leaf_classes(tp)
        }
        expected_minimum = {
            "PointCloud", "ObstacleCloud", "FreeSpaceRays", "GeometryMetrics",
            "TemporalConsistency", "TemporalStabilization", "MotionAwareReliability",
            "TemporalPersistence", "SurfaceEvidence", "BoundaryEvidence",
            "OpeningEvidence", "GeometryFrameQuality",
        }
        assert expected_minimum <= found_names, (
            f"Expected evidence families not found by the recursive walk: "
            f"{expected_minimum - found_names}"
        )


class TestCoreContractFloor:
    """Phase D13: the minimal INPUT/EXECUTION/OUTPUT contract named
    explicitly by the D13 task's own 'PUBLIC API RULE' — a floor that
    must never shrink. TestAllIsExactlyIntentional already fails on ANY
    drift from the full TIER_1_SYMBOLS set; this test is deliberately
    narrower and independent of that hand-maintained set, so even a
    (hypothetical) future accidental edit to TIER_1_SYMBOLS itself cannot
    silently carry a real regression through both guards at once."""

    CORE_INPUT = {"StereoObservation", "StereoCalibration", "MotionHint", "PipelineConfig"}
    CORE_EXECUTION = {"DepthPerceptionPipeline"}
    CORE_OUTPUT = {"GeometryFrame"}

    def test_core_input_execution_output_symbols_remain_tier1(self):
        core = self.CORE_INPUT | self.CORE_EXECUTION | self.CORE_OUTPUT
        missing_from_root = [name for name in core if not hasattr(dpe, name)]
        missing_from_all = [name for name in core if name not in dpe.__all__]
        assert not missing_from_root, f"Core contract symbol(s) missing from package root: {missing_from_root}"
        assert not missing_from_all, f"Core contract symbol(s) missing from __all__: {missing_from_all}"

    def test_core_output_alone_is_sufficient_to_construct_and_run_a_pipeline(self, config, calibration, stereo_pair):
        """A minimal proof that the CORE_INPUT/CORE_EXECUTION symbols
        alone (no other Tier 1 symbol) are sufficient to drive the
        pipeline and obtain a GeometryFrame — the literal D13 workflow:
        configure -> construct input -> run pipeline -> consume
        GeometryFrame."""
        from depth_perception_engine import DepthPerceptionPipeline, GeometryFrame, PipelineConfig, StereoCalibration

        assert isinstance(config, PipelineConfig)
        assert isinstance(calibration, StereoCalibration)
        left, right = stereo_pair

        full_config = PipelineConfig(enable_geometry=True, enable_geometry_frame=True)
        pipeline = DepthPerceptionPipeline(full_config, calibration)
        result = pipeline.process(left, right)
        assert isinstance(result.geometry_frame, GeometryFrame)


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
