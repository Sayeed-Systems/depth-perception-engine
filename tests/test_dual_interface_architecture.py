"""
DPE dual-interface architecture tests — see docs/DUAL_INTERFACE_ARCHITECTURE.md.

DPE is consumed through exactly two supported interfaces:

    CORE / EMBEDDED     depth_perception_engine.core
                        (DepthPerceptionPipeline + StereoObservation
                        -> GeometryFrame) — what a larger perception system
                        embedding DPE uses.

    STANDALONE          depth_perception_engine.standalone
                        (StandaloneStereoInterface) — raw/convenient input
                        adaptation that keeps DPE independently runnable for
                        development, tests, benchmarks and qualification.

These tests enforce the architectural invariant that makes that split safe:
BOTH interfaces reach the SAME geometry implementation
(DepthPerceptionPipeline.process_observation()) and produce the SAME
authoritative output contract (GeometryFrame) — there is no second geometry
pipeline, no standalone-specific frame type, and no runtime mode flag.

Every execution test below runs the REAL algorithms (real StereoSGBM, real
calibration fixture, real geometry/temporal stages) — no mocked provider
stands in for a geometry stage anywhere in this file. The only injected
double is a pass-through delegation SPY in TestNoDuplicatedProcessing, which
wraps (and still runs) the real engine.
"""

import ast
import dataclasses
import pathlib
import subprocess
import sys
import textwrap

import cv2
import numpy as np
import pytest

# --- the CORE / EMBEDDED public API, imported exactly as an embedded
# --- consumer would: the documented core namespace and nothing else.
from depth_perception_engine.core import (
    DepthPerceptionPipeline,
    FrameId,
    GeometryFrame,
    MotionHint,
    PipelineConfig,
    RigidTransform,
    StereoCalibration,
    StereoObservation,
)
# --- the STANDALONE public API, on its own canonical import path.
from depth_perception_engine.standalone import StandaloneStereoInterface

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CALIBRATION_PATH = str(_REPO_ROOT / "examples" / "config" / "stereo_calibration.xml")
_SRC_ROOT = _REPO_ROOT / "src" / "depth_perception_engine"


# ===================================================================
# Shared real-data fixtures
# ===================================================================
def _structured_stereo_pair(calibration, shift_px=24, seed=5):
    """A real, locally-structured stereo pair (not i.i.d. noise) with a known
    horizontal shift — the same technique tests/test_d10_black_box_provider.py
    and tests/test_d13_external_consumer.py already use, so every evidence
    family (surface/boundary/opening/clearance/temporal) is genuinely
    populated rather than degenerate."""
    width, height = calibration.image_size
    canvas_w = width + shift_px
    rng = np.random.default_rng(seed)
    low_res = rng.integers(0, 255, (height // 4 + 2, canvas_w // 4 + 2), dtype=np.uint8)
    canvas = cv2.resize(low_res, (canvas_w, height), interpolation=cv2.INTER_CUBIC)
    canvas_bgr = np.stack([canvas] * 3, axis=-1)
    return canvas_bgr[:, 0:width], canvas_bgr[:, shift_px:shift_px + width]


def _full_config():
    """Every evidence family opted in, so equivalence is proven across the
    WHOLE contract rather than a trivially small subset of it."""
    return PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        enable_geometry_frame=True,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=50,
    )


def _body_transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


@pytest.fixture
def stereo_structured(calibration):
    return _structured_stereo_pair(calibration)


# ===================================================================
# Deep GeometryFrame comparison
# ===================================================================
def _assert_values_equal(a, b, path):
    assert type(a) is type(b), f"{path}: type mismatch {type(a).__name__} vs {type(b).__name__}"

    if isinstance(a, np.ndarray):
        assert a.shape == b.shape, f"{path}: shape {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} vs {b.dtype}"
        if a.dtype.kind in "fc":
            # equal_nan: DPE deliberately uses NaN for "no measurement" in
            # PointCloud.points — two identical runs must agree there too.
            assert np.array_equal(a, b, equal_nan=True), f"{path}: array values differ"
        else:
            assert np.array_equal(a, b), f"{path}: array values differ"
        return

    if dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            _assert_values_equal(getattr(a, field.name), getattr(b, field.name), f"{path}.{field.name}")
        return

    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_values_equal(x, y, f"{path}[{i}]")
        return

    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: key sets differ"
        for key in a:
            _assert_values_equal(a[key], b[key], f"{path}[{key!r}]")
        return

    if isinstance(a, float) and np.isnan(a):
        assert np.isnan(b), f"{path}: {a} vs {b}"
        return

    assert a == b, f"{path}: {a!r} vs {b!r}"


def assert_geometry_frames_equal(frame_a, frame_b, label=""):
    """Field-for-field, recursively, across GeometryFrame's whole type graph.

    Deliberately EXACT (not approximate): both paths run the identical
    deterministic implementation on the identical input, so any numerical
    difference at all would mean the interface refactor changed DPE geometry.
    """
    assert isinstance(frame_a, GeometryFrame) and isinstance(frame_b, GeometryFrame)
    _assert_values_equal(frame_a, frame_b, label or "GeometryFrame")


def _count_populated_evidence_families(frame):
    """How many of GeometryFrame's own evidence families were genuinely
    populated — so an 'equivalence' claim can never rest on two empty
    frames trivially matching."""
    families = [
        frame.geometry, frame.geometry_body, frame.obstacle_cloud, frame.free_space_rays,
        frame.geometry_metrics, frame.region_evidence, frame.clearance_evidence,
        frame.surface_evidence, frame.boundary_evidence, frame.quality,
        frame.temporal_consistency, frame.temporal_stabilization, frame.temporal_persistence,
        frame.motion_aware_reliability, frame.rotation_compensation_status,
    ]
    return sum(1 for family in families if family is not None)


# ===================================================================
# 1. CORE DIRECT EXECUTION  (the most important new integration test)
# ===================================================================
class TestCoreDirectExecution:
    """canonical input -> DPE core -> GeometryFrame, with no standalone
    adapter constructed anywhere."""

    def test_canonical_observation_through_core_yields_geometry_frame(self, calibration, stereo_structured):
        left, right = stereo_structured
        pipeline = DepthPerceptionPipeline(
            _full_config(), calibration, body_T_camera_left=_body_transform(),
        )
        observation = StereoObservation(
            left_image=left, right_image=right, left_timestamp=0.0, right_timestamp=0.0,
        )

        geometry = pipeline.process_geometry_frame(observation)

        assert isinstance(geometry, GeometryFrame)
        assert geometry.frame_id == FrameId.CAMERA_OPTICAL_LEFT
        assert geometry.timestamp == 0.0
        assert geometry.depth_map.shape == (calibration.image_size[1], calibration.image_size[0])
        assert _count_populated_evidence_families(geometry) >= 10, (
            "Core direct execution produced a nearly empty GeometryFrame — the "
            "equivalence proofs below would then be vacuous."
        )

    def test_core_returns_a_geometry_frame_even_without_the_legacy_result_flag(
        self, calibration, stereo_structured,
    ):
        """process_geometry_frame() IS the caller's request for a
        GeometryFrame, so enable_geometry_frame (which gates only the legacy
        DepthPerceptionResult.geometry_frame field) must not be a hidden
        prerequisite for the embedded path."""
        left, right = stereo_structured
        config = dataclasses.replace(_full_config(), enable_geometry_frame=False)
        pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_body_transform())

        geometry = pipeline.process_geometry_frame(
            StereoObservation(left_image=left, right_image=right, left_timestamp=0.0)
        )
        assert isinstance(geometry, GeometryFrame)

    def test_flag_on_and_flag_off_geometry_frames_are_identical(self, calibration, stereo_structured):
        """The two branches inside process_geometry_frame() (reuse the frame
        already built during process_observation(), vs build it immediately
        afterwards) call the identical builder with identical arguments."""
        left, right = stereo_structured
        observation = StereoObservation(left_image=left, right_image=right, left_timestamp=0.0)

        on = DepthPerceptionPipeline(
            _full_config(), calibration, body_T_camera_left=_body_transform(),
        ).process_geometry_frame(observation)
        off = DepthPerceptionPipeline(
            dataclasses.replace(_full_config(), enable_geometry_frame=False),
            calibration, body_T_camera_left=_body_transform(),
        ).process_geometry_frame(observation)

        assert_geometry_frames_equal(on, off, "enable_geometry_frame on-vs-off")

    def test_core_construction_does_not_touch_the_filesystem_or_a_device(self, calibration, mocker):
        """The core takes a StereoCalibration OBJECT — calibration file
        loading is a standalone convenience, not a core responsibility."""
        video_capture = mocker.patch("cv2.VideoCapture")
        file_storage = mocker.patch("cv2.FileStorage")

        DepthPerceptionPipeline(_full_config(), calibration)

        video_capture.assert_not_called()
        file_storage.assert_not_called()


# ===================================================================
# 2. STANDALONE EXECUTION
# ===================================================================
class TestStandaloneExecution:
    """standalone input -> adapter -> SAME core -> GeometryFrame."""

    def test_standalone_from_calibration_file_runs_end_to_end(self, calibration, stereo_structured):
        left, right = stereo_structured
        dpe = StandaloneStereoInterface.from_calibration_file(
            _CALIBRATION_PATH, _full_config(), body_T_camera_left=_body_transform(),
        )

        geometry = dpe.process_geometry_frame(left, right, timestamp=0.0)

        assert isinstance(geometry, GeometryFrame)
        assert isinstance(dpe.calibration, StereoCalibration)
        assert _count_populated_evidence_families(geometry) >= 10

    def test_standalone_still_returns_the_legacy_result_shape(self, calibration, stereo_structured):
        """DPE remains independently runnable exactly as before — the
        development/benchmark path that reads DepthPerceptionResult is
        unchanged."""
        left, right = stereo_structured
        dpe = StandaloneStereoInterface(_full_config(), calibration)

        result = dpe.process(left, right, timestamp=0.0)

        assert result.depth_map.shape == left.shape[:2]
        assert isinstance(result.geometry_frame, GeometryFrame)

    def test_standalone_splits_a_combined_side_by_side_frame(self, calibration, stereo_structured):
        left, right = stereo_structured
        combined = np.concatenate([left, right], axis=1)

        # Two fresh interfaces, so cross-frame engine state cannot differ
        # between the two runs — the ONLY difference is how the pair reached
        # the core (split from one joined frame vs. supplied separately).
        via_combined = StandaloneStereoInterface(
            _full_config(), calibration, body_T_camera_left=_body_transform(),
        ).process_combined_frame_geometry(combined, timestamp=0.0)
        via_pair = StandaloneStereoInterface(
            _full_config(), calibration, body_T_camera_left=_body_transform(),
        ).process_geometry_frame(left, right, timestamp=0.0)

        assert isinstance(via_combined, GeometryFrame)
        assert_geometry_frames_equal(via_combined, via_pair, "combined-frame-vs-pair")

    def test_split_returns_views_not_copies(self, calibration, stereo_structured):
        """No image copying was introduced by separating the interfaces."""
        left, right = stereo_structured
        combined = np.ascontiguousarray(np.concatenate([left, right], axis=1))
        dpe = StandaloneStereoInterface(_full_config(), calibration)

        split_left, split_right = dpe.split_combined_frame(combined)

        assert split_left.base is combined
        assert split_right.base is combined

    def test_observation_holds_the_caller_arrays_by_reference(self, calibration, stereo_structured):
        left, right = stereo_structured
        dpe = StandaloneStereoInterface(_full_config(), calibration)

        observation = dpe.build_observation(left, right, timestamp=1.0)

        assert observation.left_image is left
        assert observation.right_image is right
        assert observation.left_timestamp == 1.0

    def test_standalone_lifecycle_delegates_to_the_core(self, calibration, stereo_structured):
        left, right = stereo_structured
        dpe = StandaloneStereoInterface(_full_config(), calibration)
        dpe.process(left, right, timestamp=0.0)

        assert dpe.health().frames_processed == 1
        dpe.reset()
        assert dpe.health().frames_processed == 0
        dpe.close()
        assert dpe.health().is_closed is True
        with pytest.raises(RuntimeError):
            dpe.process(left, right)


# ===================================================================
# 3. OUTPUT EQUIVALENCE (real algorithms, both paths)
# ===================================================================
class TestOutputEquivalence:
    def test_single_frame_geometry_frames_are_field_for_field_identical(self, calibration, stereo_structured):
        left, right = stereo_structured

        core = DepthPerceptionPipeline(_full_config(), calibration, body_T_camera_left=_body_transform())
        core_frame = core.process_geometry_frame(
            StereoObservation(left_image=left, right_image=right, left_timestamp=0.0)
        )

        standalone = StandaloneStereoInterface.from_calibration_file(
            _CALIBRATION_PATH, _full_config(), body_T_camera_left=_body_transform(),
        )
        standalone_frame = standalone.process_geometry_frame(left, right, timestamp=0.0)

        assert _count_populated_evidence_families(core_frame) >= 10
        assert_geometry_frames_equal(core_frame, standalone_frame, "single-frame core-vs-standalone")

    def test_multi_frame_sequence_including_temporal_state_is_identical(self, calibration, stereo_structured):
        """Temporal evidence (consistency/stabilization/persistence) only
        becomes non-trivial from the second frame onward, and it depends on
        cross-frame engine state — so equivalence must be proven over a
        SEQUENCE, not one frame."""
        left, right = stereo_structured
        core = DepthPerceptionPipeline(_full_config(), calibration, body_T_camera_left=_body_transform())
        standalone = StandaloneStereoInterface(
            _full_config(), calibration, body_T_camera_left=_body_transform(),
        )

        core_frames, standalone_frames = [], []
        for i in range(4):
            timestamp = float(i)
            core_frames.append(core.process_geometry_frame(
                StereoObservation(left_image=left, right_image=right, left_timestamp=timestamp)
            ))
            standalone_frames.append(standalone.process_geometry_frame(left, right, timestamp=timestamp))

        last = core_frames[-1]
        assert last.temporal_consistency is not None
        assert last.temporal_persistence is not None
        for i, (a, b) in enumerate(zip(core_frames, standalone_frames)):
            assert_geometry_frames_equal(a, b, f"frame[{i}] core-vs-standalone")

    def test_legacy_result_shape_is_also_equivalent_through_both_paths(self, calibration, stereo_structured):
        """Not only GeometryFrame: the legacy DepthPerceptionResult the
        standalone/development path still returns is unchanged too (with
        processing_time_ms excluded — it is a wall-clock measurement, not
        geometry)."""
        left, right = stereo_structured
        core = DepthPerceptionPipeline(_full_config(), calibration, body_T_camera_left=_body_transform())
        standalone = StandaloneStereoInterface(
            _full_config(), calibration, body_T_camera_left=_body_transform(),
        )

        core_result = core.process_observation(
            StereoObservation(left_image=left, right_image=right, left_timestamp=0.0)
        )
        standalone_result = standalone.process(left, right, timestamp=0.0)

        for field in dataclasses.fields(core_result):
            if field.name == "processing_time_ms":
                continue
            _assert_values_equal(
                getattr(core_result, field.name), getattr(standalone_result, field.name),
                f"DepthPerceptionResult.{field.name}",
            )


# ===================================================================
# 4. MOTION EQUIVALENCE
# ===================================================================
class TestMotionEquivalence:
    """Raw motion samples adapted by the standalone layer must deliver the
    SAME normalized MotionHint values into the SAME core motion path."""

    def test_raw_samples_normalize_to_the_same_motion_hint(self):
        explicit = MotionHint(
            timestamp=1.0, angular_velocity_rad_s=np.array([0.0, 0.0, 0.01], dtype=np.float64),
            frame_id=FrameId.BODY, valid=True,
        )
        from_pair = StandaloneStereoInterface.build_motion_hints([(1.0, (0.0, 0.0, 0.01))])[0]
        from_flat = StandaloneStereoInterface.build_motion_hints([(1.0, 0.0, 0.0, 0.01)])[0]

        for adapted in (from_pair, from_flat):
            assert adapted.timestamp == explicit.timestamp
            assert adapted.frame_id == explicit.frame_id
            assert adapted.valid == explicit.valid
            assert np.array_equal(adapted.angular_velocity_rad_s, explicit.angular_velocity_rad_s)
            assert adapted.angular_velocity_rad_s.shape == (3,)

    def test_already_built_motion_hints_pass_through_unchanged(self):
        hint = MotionHint(
            timestamp=2.0, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY,
        )
        assert StandaloneStereoInterface.build_motion_hints([hint])[0] is hint

    def test_none_stays_none(self):
        assert StandaloneStereoInterface.build_motion_hints(None) is None

    def test_conflicting_motion_arguments_are_rejected_not_silently_resolved(self, calibration):
        dpe = StandaloneStereoInterface(_full_config(), calibration)
        hint = MotionHint(timestamp=0.0, angular_velocity_rad_s=np.zeros(3), frame_id=FrameId.BODY)
        with pytest.raises(ValueError):
            dpe.build_observation(
                np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8),
                motion_hints=[hint], motion_samples=[(0.0, 0.0, 0.0, 0.0)],
            )

    def test_motion_driven_temporal_evidence_is_identical_through_both_paths(
        self, calibration, stereo_structured,
    ):
        """End-to-end, with real rotation compensation / motion-aware
        reliability / persistence actually exercised: the standalone path
        supplies RAW angular-rate tuples, the core path supplies explicit
        MotionHints, and the resulting GeometryFrames must match exactly."""
        left, right = stereo_structured
        core = DepthPerceptionPipeline(_full_config(), calibration, body_T_camera_left=_body_transform())
        standalone = StandaloneStereoInterface(
            _full_config(), calibration, body_T_camera_left=_body_transform(),
        )

        core_frames, standalone_frames = [], []
        for i in range(4):
            timestamp = float(i)
            rate = (0.0, 0.0, 0.01 * i)
            hint = MotionHint(
                timestamp=timestamp,
                angular_velocity_rad_s=np.array(rate, dtype=np.float64),
                frame_id=FrameId.BODY, valid=True,
            )
            core_frames.append(core.process_geometry_frame(StereoObservation(
                left_image=left, right_image=right, left_timestamp=timestamp,
                motion_hint=hint, motion_hints=[hint],
            )))
            standalone_frames.append(standalone.process_geometry_frame(
                left, right, timestamp=timestamp,
                motion_hint=hint, motion_samples=[(timestamp, rate)],
            ))

        last = core_frames[-1]
        assert last.rotation_compensation_status is not None
        assert last.motion_aware_reliability is not None
        for i, (a, b) in enumerate(zip(core_frames, standalone_frames)):
            assert_geometry_frames_equal(a, b, f"motion frame[{i}] core-vs-standalone")


# ===================================================================
# 5. NO DUPLICATED PROCESSING
# ===================================================================
class _DelegationSpy:
    """A pass-through recorder around the REAL engine — it forwards every
    call unchanged, so the geometry that comes back is genuinely produced by
    the core, not by a stub."""

    def __init__(self, engine):
        self._engine = engine
        self.observations = []
        self.geometry_frame_calls = 0

    def process_observation(self, observation):
        self.observations.append(observation)
        return self._engine.process_observation(observation)

    def process_geometry_frame(self, observation):
        self.geometry_frame_calls += 1
        self.observations.append(observation)
        return self._engine.process_geometry_frame(observation)

    def __getattr__(self, name):
        return getattr(self._engine, name)


class TestNoDuplicatedProcessing:
    def test_standalone_delegates_every_frame_to_the_injected_core(self, calibration, stereo_structured):
        left, right = stereo_structured
        real_engine = DepthPerceptionPipeline(_full_config(), calibration)
        spy = _DelegationSpy(real_engine)
        dpe = StandaloneStereoInterface(engine=spy)

        geometry = dpe.process_geometry_frame(left, right, timestamp=0.0)

        assert spy.geometry_frame_calls == 1
        assert len(spy.observations) == 1
        observation = spy.observations[0]
        assert isinstance(observation, StereoObservation)
        assert observation.left_image is left
        assert observation.right_image is right
        assert isinstance(geometry, GeometryFrame)

    def test_standalone_and_core_can_share_one_engine_instance(self, calibration, stereo_structured):
        """The strongest possible single-core proof: the very same engine
        object serves both interfaces, and its own frame counter advances
        once per call from either side."""
        left, right = stereo_structured
        engine = DepthPerceptionPipeline(_full_config(), calibration)
        dpe = StandaloneStereoInterface(engine=engine)

        assert dpe.engine is engine
        engine.process_observation(StereoObservation(left_image=left, right_image=right, left_timestamp=0.0))
        dpe.process(left, right, timestamp=1.0)

        assert engine.health().frames_processed == 2

    def test_standalone_module_imports_no_geometry_algorithm(self):
        """Structural proof that the adapter cannot contain a second
        geometry pipeline: it imports no algorithm module at all. Follows
        this repository's existing AST-scan guard style
        (tests/test_no_ros_dependency.py, tests/test_level4_architecture_guards.py,
        tests/test_d13_external_consumer.py)."""
        forbidden_prefixes = (
            "depth_perception_engine.geometry.boundary",
            "depth_perception_engine.geometry.free_space",
            "depth_perception_engine.geometry.geometry_metrics",
            "depth_perception_engine.geometry.obstacle_extractor",
            "depth_perception_engine.geometry.opening",
            "depth_perception_engine.geometry.point_cloud_builder",
            "depth_perception_engine.geometry.reliability",
            "depth_perception_engine.geometry.rigid_transform",
            "depth_perception_engine.geometry.surface",
            "depth_perception_engine.depth",
            "depth_perception_engine.stereo.disparity_engine",
            "depth_perception_engine.stereo.rectification",
            "depth_perception_engine.traversability",
            "depth_perception_engine.obstacles",
            "depth_perception_engine.fusion",
            "depth_perception_engine.quality",
            "depth_perception_engine.temporal.consistency",
            "depth_perception_engine.temporal.history",
            "depth_perception_engine.temporal.persistence",
            "depth_perception_engine.temporal.reliability",
            "depth_perception_engine.temporal.rotation_compensation",
            "depth_perception_engine.temporal.stabilization",
        )
        for path in sorted((_SRC_ROOT / "standalone").glob("*.py")):
            tree = ast.parse(path.read_text())
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            offenders = [m for m in imported if m.startswith(forbidden_prefixes)]
            assert not offenders, (
                f"{path.name} imports geometry algorithm module(s) {offenders} — the "
                "standalone adapter must delegate to DepthPerceptionPipeline, never "
                "implement or re-invoke a geometry stage itself."
            )

    def test_standalone_module_calls_no_algorithm_builder(self):
        """Complementary to the import scan: no `build_*`/`compute_*`
        geometry-stage call appears anywhere in the adapter's own source."""
        pattern_prefixes = ("build_obstacle_cloud", "build_free_space_rays", "build_geometry_metrics",
                            "build_surface_evidence", "build_boundary_evidence", "build_opening_evidence",
                            "build_geometry_frame", "build_result", "compute_disparity", "compute_temporal",
                            "compute_rotation_compensation", "compute_motion_aware_reliability",
                            "compute_shadow_zone_mask", "compute_ramp_zone_mask", "estimate")
        for path in sorted((_SRC_ROOT / "standalone").glob("*.py")):
            tree = ast.parse(path.read_text())
            called = [
                node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
            ]
            offenders = [name for name in called if name.startswith(pattern_prefixes)]
            assert not offenders, f"{path.name} calls geometry builder(s) {offenders}"


# ===================================================================
# 6. PUBLIC IMPORT SURFACE
# ===================================================================
def _embedded_consumer_workflow(calibration, left, right):
    """The exact call sequence an embedded consumer performs. Built from
    the public core namespace only — see TestPublicImportSurface, which
    AST-scans this module's own imports."""
    pipeline = DepthPerceptionPipeline(_full_config(), calibration)
    observation = StereoObservation(left_image=left, right_image=right, left_timestamp=0.0)
    return pipeline.process_geometry_frame(observation)


class TestPublicImportSurface:
    ALLOWED = (
        "depth_perception_engine",            # the package root (public)
        "depth_perception_engine.core",       # CORE / EMBEDDED namespace
        "depth_perception_engine.standalone", # STANDALONE namespace
    )

    def test_this_file_imports_only_the_two_public_interface_namespaces(self):
        tree = ast.parse(pathlib.Path(__file__).read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

        dpe_imports = [m for m in imported if m.startswith("depth_perception_engine")]
        assert dpe_imports, "sanity: this file must import DPE"
        offenders = [m for m in dpe_imports if m not in self.ALLOWED]
        assert not offenders, (
            f"Dual-interface test imported {offenders} — the whole point is that "
            "each interface is reachable through its own public namespace alone, "
            "with no internal (algorithms/pipeline/fusion/temporal) import needed."
        )

    def test_embedded_workflow_runs_using_core_imports_only(self, calibration, stereo_structured):
        left, right = stereo_structured
        geometry = _embedded_consumer_workflow(calibration, left, right)
        assert isinstance(geometry, GeometryFrame)

    def test_core_namespace_symbols_are_identical_objects_to_the_root_exports(self):
        import depth_perception_engine as dpe
        import depth_perception_engine.core as dpe_core

        # RigidTransform stays Tier 3 at the root (a constructor input, not
        # part of GeometryFrame's output type graph — see docs/PUBLIC_API.md);
        # every other core symbol is a root export and must be the same object.
        shared = [name for name in dpe_core.__all__ if hasattr(dpe, name)]
        assert len(shared) >= len(dpe_core.__all__) - 1
        for name in shared:
            assert getattr(dpe_core, name) is getattr(dpe, name), (
                f"{name} differs between depth_perception_engine.core and the package root — "
                "the core namespace must re-export the SAME objects, never redefine them."
            )

    def test_core_namespace_exposes_no_file_loading_or_sensor_helper(self):
        import depth_perception_engine.core as dpe_core

        for name in ("load_stereo_calibration", "FrameSplitter", "StandaloneStereoInterface"):
            assert not hasattr(dpe_core, name), (
                f"{name} leaked into the core/embedded namespace — sensor-facing "
                "convenience belongs to the standalone interface only."
            )

    def test_standalone_entry_point_is_also_reachable_from_the_package_root(self):
        import depth_perception_engine as dpe
        from depth_perception_engine.standalone import StandaloneStereoInterface as canonical

        assert dpe.StandaloneStereoInterface is canonical
        assert "StandaloneStereoInterface" in dpe.__all__


# ===================================================================
# 7. STANDALONE OPTIONALITY
# ===================================================================
_OPTIONALITY_PROGRAM = textwrap.dedent(
    """
    import sys

    import numpy as np

    from depth_perception_engine.core import (
        DepthPerceptionPipeline, PipelineConfig, StereoObservation, GeometryFrame,
    )
    from depth_perception_engine.calibration import load_stereo_calibration

    assert "depth_perception_engine.standalone" not in sys.modules, (
        "importing the DPE core loaded the standalone adapter"
    )

    calibration = load_stereo_calibration(sys.argv[1])
    width, height = calibration.image_size
    rng = np.random.default_rng(3)
    left = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

    pipeline = DepthPerceptionPipeline(
        PipelineConfig(enable_geometry=True, enable_geometry_frame=True), calibration,
    )
    geometry = pipeline.process_geometry_frame(
        StereoObservation(left_image=left, right_image=right, left_timestamp=0.0)
    )
    assert isinstance(geometry, GeometryFrame)

    assert "depth_perception_engine.standalone" not in sys.modules, (
        "running the DPE core loaded the standalone adapter"
    )
    assert "depth_perception_engine.standalone.interface" not in sys.modules

    print("OK")
    """
)


class TestStandaloneOptionality:
    def test_core_construction_and_execution_never_load_the_standalone_layer(self):
        """A fresh interpreter: construct and RUN the core, then assert the
        standalone adapter was never imported. This is what makes 'standalone
        is off inside an embedded consumer' a structural fact rather than a
        runtime flag."""
        completed = subprocess.run(
            [sys.executable, "-c", _OPTIONALITY_PROGRAM, _CALIBRATION_PATH],
            capture_output=True, text=True, cwd=str(_REPO_ROOT),
        )
        assert completed.returncode == 0, (
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        assert "OK" in completed.stdout

    def test_no_module_in_the_library_imports_the_standalone_subpackage(self):
        """Nothing inside DPE itself may depend on the standalone layer —
        the dependency direction is strictly standalone -> core."""
        offenders = []
        for path in sorted(_SRC_ROOT.rglob("*.py")):
            if "standalone" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            # MODULE-LEVEL imports only. The package root's own lazy PEP 562
            # __getattr__ (a function-body import, executed only if a caller
            # explicitly asks for StandaloneStereoInterface) is exactly the
            # mechanism that keeps the standalone layer optional — flagging it
            # would invert the rule this test exists to enforce.
            for node in ast.iter_child_nodes(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    if module.startswith("depth_perception_engine.standalone"):
                        offenders.append(f"{path.relative_to(_SRC_ROOT)}:{node.lineno}")
        assert not offenders, (
            f"Library module(s) import the standalone adapter at module level: "
            f"{offenders}. The standalone layer must remain a leaf that depends on "
            "the core, never the reverse — otherwise it becomes a hidden mandatory "
            "dependency."
        )

    def test_standalone_subpackage_is_not_a_core_import_side_effect_of_the_root(self):
        """Importing the package root must not eagerly load the standalone
        layer either — the root re-export is deliberately lazy."""
        program = textwrap.dedent(
            """
            import sys
            import depth_perception_engine  # noqa: F401
            assert "depth_perception_engine.standalone" not in sys.modules
            print("OK")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, cwd=str(_REPO_ROOT),
        )
        assert completed.returncode == 0, f"stderr:\n{completed.stderr}"
        assert "OK" in completed.stdout
