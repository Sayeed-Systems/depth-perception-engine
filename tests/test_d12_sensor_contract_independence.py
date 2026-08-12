"""
Sensor-contract independence validation — Phase D12 (see
docs/DPE_V1_PROVIDER_CONTRACT.md's D12 record).

Goal: prove DepthPerceptionPipeline depends only on its public,
sensor-facing contracts (StereoObservation, StereoCalibration, MotionHint,
PipelineConfig) and GeometryFrame is fully consumable without any
knowledge of what produced those contracts — no ROS, no mp01_sensors, no
physical camera/IMU driver, no simulator. Conceptual boundary this file
exercises:

    any sensor backend -> mp01_sensors contract -> StereoObservation +
    calibration + optional MotionHint -> DepthPerceptionPipeline ->
    GeometryFrame

DPE itself has no opinion above the StereoObservation/StereoCalibration/
MotionHint line — this file constructs every input purely from public
dataclasses/numpy arrays (never a ROS message, never an mp01_sensors
type, never a hardware/simulator handle) and proves the full pipeline,
including Level 4 temporal/rotation-compensation behavior, runs to
completion and produces a fully self-describing GeometryFrame.

STRUCTURAL RULE — enforced by TestPublicApiOnlyImportSurface below, not
merely by convention (same discipline as
tests/test_d10_black_box_provider.py): this file imports
depth_perception_engine only via its root package and the two documented
public subpackage paths (`depth_perception_engine.frames`,
`depth_perception_engine.temporal`) — never `.pipeline`, `.geometry`,
`.traversability`, `.obstacles`, `.fusion`, or any other internal module.

Per the task's own explicit instruction, this file does NOT add any
source/origin-identifying field to any DPE type (no `source`,
`sensor_id`, `is_simulated`, ...) to "prove" independence — instead it
proves DPE is source-blind by construction: the public contracts
themselves carry no such field (checked structurally below), and two
inputs built via genuinely different construction paths but carrying
identical values produce provably identical GeometryFrame output.
"""

import ast
import dataclasses
import pathlib

import cv2
import numpy as np
import pytest

import depth_perception_engine as dpe
from depth_perception_engine import (
    DepthPerceptionPipeline,
    GeometryFrame,
    PipelineConfig,
    StereoCalibration,
    StereoObservation,
    load_stereo_calibration,
)
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.temporal import MotionHint

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size


# ===================================================================
# Structural: this file itself never reaches into DPE internals
# ===================================================================
class TestPublicApiOnlyImportSurface:
    def test_this_test_file_imports_no_internal_dpe_module(self):
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
                f"D12 source-independence test imported {module!r} — only the "
                "root package and the frames/temporal public subpackages are "
                "permitted, proving this file (and by construction, any "
                "external caller) needs no DPE-internal or sensor-backend "
                "import to exercise the whole pipeline."
            )


# ===================================================================
# The public contracts themselves carry no sensor/platform-identity field
# ===================================================================
class TestPublicContractsCarryNoSourceIdentity:
    _FORBIDDEN_FIELD_NAMES = {
        "source", "sensor_id", "device_id", "driver", "topic", "topic_name",
        "vehicle_id", "vehicle_type", "platform", "platform_id", "robot_id",
        "is_simulated", "is_simulation", "backend", "hardware_id",
    }

    @pytest.mark.parametrize("cls", [StereoObservation, StereoCalibration, MotionHint, PipelineConfig])
    def test_no_source_identity_field(self, cls):
        field_names = {f.name for f in dataclasses.fields(cls)}
        hit = field_names & self._FORBIDDEN_FIELD_NAMES
        assert not hit, f"{cls.__name__} carries a source/origin-identity field: {hit}"


# ===================================================================
# Helpers — construct valid public inputs from nothing but numpy/dataclasses
# ===================================================================
def _raw_buffer_stereo_pair(seed: int = 11):
    """Mimics a real driver handing over a raw, already-decoded frame
    buffer: a numpy array built directly from an integer RNG, no
    intermediate encode/decode step. One construction PATH among two
    this file uses to build value-identical stereo evidence — neither
    path is DPE-visible; DPE only ever sees the resulting ndarray."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
    return left, right


def _encode_decode_roundtrip(image: np.ndarray) -> np.ndarray:
    """Mimics a transport layer that serializes/deserializes an image
    (e.g. a lossless PNG-encoded topic payload) before DPE ever sees it —
    a different construction PATH than _raw_buffer_stereo_pair, but
    lossless, so the resulting array is byte-identical."""
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


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


def _smoothed_stereo_pair(shift_px: int = 24, seed: int = 99):
    """A real stereo pair with genuine, stable local structure (smoothed
    low-frequency texture, the same technique
    tests/test_d10_black_box_provider.py uses) rather than i.i.d.
    per-pixel noise — needed for the MotionHint scenarios below, which
    must produce a real, comparable depth map robust to a small rotation
    compensation shift. Pure per-pixel noise defeats StereoSGBM's own
    correspondence entirely (confirmed by this repository's own D10/D11
    findings), which would make CONSISTENT/CONTRADICTORY outcomes an
    artifact of the fixture, not of rotation-compensation behavior."""
    canvas_w = _W + shift_px
    rng = np.random.default_rng(seed)
    low_res = rng.integers(0, 255, (_H // 4 + 2, canvas_w // 4 + 2), dtype=np.uint8)
    canvas = cv2.resize(low_res, (canvas_w, _H), interpolation=cv2.INTER_CUBIC)
    canvas_bgr = np.stack([canvas] * 3, axis=-1)
    left = canvas_bgr[:, 0:_W]
    right = canvas_bgr[:, shift_px:shift_px + _W]
    return left, right


def _hint(ts, omega, valid=True):
    return MotionHint(
        timestamp=ts,
        angular_velocity_rad_s=np.array(omega, dtype=np.float64),
        frame_id=FrameId.BODY,
        valid=valid,
    )


def _assert_geometry_frames_equal(gf_a: GeometryFrame, gf_b: GeometryFrame):
    """Field-by-field equality sufficient to prove two GeometryFrame
    instances describe the identical scene — mirrors
    test_d10_black_box_provider.py's determinism check."""
    np.testing.assert_array_equal(gf_a.disparity_map, gf_b.disparity_map)
    np.testing.assert_array_equal(gf_a.depth_map, gf_b.depth_map)
    np.testing.assert_array_equal(gf_a.obstacle_cloud.points, gf_b.obstacle_cloud.points)
    np.testing.assert_array_equal(gf_a.free_space_rays.ranges_m, gf_b.free_space_rays.ranges_m)
    assert gf_a.geometry_metrics == gf_b.geometry_metrics
    assert [s.normal.tolist() if s.normal is not None else None for s in gf_a.surface_evidence] == \
           [s.normal.tolist() if s.normal is not None else None for s in gf_b.surface_evidence]
    assert gf_a.boundary_evidence == gf_b.boundary_evidence
    assert gf_a.opening_evidence == gf_b.opening_evidence
    assert gf_a.clearance_evidence == gf_b.clearance_evidence


# ===================================================================
# 1. DPE runs end-to-end from purely public, constructed inputs
# ===================================================================
class TestPipelineRunsFromPublicContractsAlone:
    def test_process_from_raw_arrays_produces_full_geometry_frame(self):
        left, right = _raw_buffer_stereo_pair()
        result = _pipeline().process(left, right, left_timestamp=0.0, right_timestamp=0.0)
        assert isinstance(result.geometry_frame, GeometryFrame)
        assert result.geometry_frame.quality is not None

    def test_process_observation_from_a_hand_built_stereoobservation(self):
        """StereoObservation constructed with nothing but public dataclass
        fields and numpy arrays — no producer type from any sensor
        backend is required to build a legal one."""
        left, right = _raw_buffer_stereo_pair()
        observation = StereoObservation(
            left_image=left,
            right_image=right,
            left_timestamp=1.0,
            right_timestamp=1.0,
            frame_id=FrameId.CAMERA_OPTICAL_LEFT,
            motion_hint=_hint(1.0, [0.0, 0.0, 0.01]),
            motion_hints=[_hint(1.0, [0.0, 0.0, 0.01])],
        )
        result = _pipeline().process_observation(observation)
        assert isinstance(result.geometry_frame, GeometryFrame)
        assert result.geometry_frame.frame_id == dpe.FrameId.CAMERA_OPTICAL_LEFT

    def test_calibration_constructed_from_plain_matrices_without_the_file_loader(self):
        """StereoCalibration itself needs no file/loader path at all —
        proves the ONE place this library touches a filesystem
        (load_stereo_calibration) is a convenience, not a structural
        dependency; any caller (a real driver's own calibration service,
        a simulator's own camera_info, a hand-written test fixture) can
        construct the contract directly."""
        calib = StereoCalibration(
            image_size=_CALIBRATION.image_size,
            camera_matrix_left=_CALIBRATION.camera_matrix_left.copy(),
            dist_coeffs_left=_CALIBRATION.dist_coeffs_left.copy(),
            camera_matrix_right=_CALIBRATION.camera_matrix_right.copy(),
            dist_coeffs_right=_CALIBRATION.dist_coeffs_right.copy(),
            R1=_CALIBRATION.R1.copy(),
            R2=_CALIBRATION.R2.copy(),
            P1=_CALIBRATION.P1.copy(),
            P2=_CALIBRATION.P2.copy(),
            Q=_CALIBRATION.Q.copy(),
        )
        left, right = _raw_buffer_stereo_pair()
        pipeline = DepthPerceptionPipeline(_full_config(), calib, rectify=False)
        result = pipeline.process(left, right)
        assert isinstance(result.geometry_frame, GeometryFrame)


# ===================================================================
# 2. Equivalent evidence from different nominal origins => equivalent
#    GeometryFrame, with no source-specific field anywhere in the loop
# ===================================================================
class TestSourceOriginIndependence:
    def test_raw_buffer_and_encode_decode_roundtrip_paths_agree_on_pixels(self):
        """Sanity: the two construction paths this file uses to stand in
        for 'different nominal sensor origins' really do produce
        identical pixel content (a lossless PNG roundtrip), so the
        GeometryFrame-equality assertion below is testing DPE's own
        source-blindness, not accidentally testing image compression."""
        left, right = _raw_buffer_stereo_pair()
        left_rt = _encode_decode_roundtrip(left)
        right_rt = _encode_decode_roundtrip(right)
        np.testing.assert_array_equal(left, left_rt)
        np.testing.assert_array_equal(right, right_rt)

    def test_equivalent_evidence_from_two_construction_paths_yields_equivalent_geometry_frame(self):
        left_a, right_a = _raw_buffer_stereo_pair()
        left_b, right_b = _encode_decode_roundtrip(left_a), _encode_decode_roundtrip(right_a)

        result_a = _pipeline().process(left_a, right_a, left_timestamp=0.0, right_timestamp=0.0)
        result_b = _pipeline().process(left_b, right_b, left_timestamp=0.0, right_timestamp=0.0)

        _assert_geometry_frames_equal(result_a.geometry_frame, result_b.geometry_frame)

    def test_two_independently_constructed_calibrations_with_equal_values_agree(self):
        """A second StereoCalibration built by a completely separate
        construction call (fresh .copy()'d arrays, not the same object
        identity) but with equal values produces the identical
        GeometryFrame — DPE's behavior is a pure function of contract
        VALUES, never of which code path or object produced them."""
        def _rebuild_calibration():
            return StereoCalibration(
                image_size=_CALIBRATION.image_size,
                camera_matrix_left=_CALIBRATION.camera_matrix_left.copy(),
                dist_coeffs_left=_CALIBRATION.dist_coeffs_left.copy(),
                camera_matrix_right=_CALIBRATION.camera_matrix_right.copy(),
                dist_coeffs_right=_CALIBRATION.dist_coeffs_right.copy(),
                R1=_CALIBRATION.R1.copy(),
                R2=_CALIBRATION.R2.copy(),
                P1=_CALIBRATION.P1.copy(),
                P2=_CALIBRATION.P2.copy(),
                Q=_CALIBRATION.Q.copy(),
            )

        left, right = _raw_buffer_stereo_pair()
        transform = RigidTransform(
            rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )
        pipeline_a = DepthPerceptionPipeline(_full_config(), _rebuild_calibration(), rectify=False, body_T_camera_left=transform)
        pipeline_b = DepthPerceptionPipeline(_full_config(), _rebuild_calibration(), rectify=False, body_T_camera_left=transform)

        result_a = pipeline_a.process(left, right)
        result_b = pipeline_b.process(left, right)
        _assert_geometry_frames_equal(result_a.geometry_frame, result_b.geometry_frame)


# ===================================================================
# 3/4. MotionHint exercised entirely through its public contract —
# deterministic, physically-consistent sequences, checked at the
# GeometryFrame level (rotation_compensation_status,
# motion_aware_reliability, temporal_consistency).
# ===================================================================
class _MotionHintScenarioBase:
    """Two-frame sequence: frame 0 establishes the previous temporal
    record (no motion hint needed for admission), frame 1 supplies this
    scenario's MotionHint(s) for the interval between the two
    timestamps. Using the identical static scene for both frames means
    any CONTRADICTORY verdict could only come from a genuine rotation-
    compensation defect, never from an unrelated scene change — the same
    "identical repeated frame" idiom test_d10_black_box_provider.py's
    own Scenario 8 already established.
    """

    PREV_TS = 0.0
    CURR_TS = 1.0

    def _run(self, motion_hints):
        left, right = _smoothed_stereo_pair()
        pipeline = _pipeline()
        pipeline.process(left, right, left_timestamp=self.PREV_TS, right_timestamp=self.PREV_TS)
        result = pipeline.process(
            left, right,
            left_timestamp=self.CURR_TS, right_timestamp=self.CURR_TS,
            motion_hints=motion_hints,
        )
        gf = result.geometry_frame
        assert gf is not None
        return gf


class TestMotionHintScenarioNoHint(_MotionHintScenarioBase):
    def test_no_motion_hint_leaves_compensation_not_applied(self):
        gf = self._run(motion_hints=None)
        assert gf.rotation_compensation_status == "NOT_APPLIED"
        # E3 alone (no E5 compensation in the loop at all) still classifies
        # a genuinely identical repeated scene as CONSISTENT.
        assert gf.temporal_consistency.state == "CONSISTENT"
        assert gf.motion_aware_reliability.state in {"RELIABLE", "DEGRADED", "UNRELIABLE", "INSUFFICIENT_EVIDENCE"}


class TestMotionHintScenarioZeroAngularMotion(_MotionHintScenarioBase):
    def test_valid_zero_angular_motion_applies_and_stays_reliable(self):
        gf = self._run(motion_hints=[_hint(0.9, [0.0, 0.0, 0.0])])
        assert gf.rotation_compensation_status == "APPLIED"
        assert gf.temporal_consistency.state == "CONSISTENT"
        assert gf.motion_aware_reliability.state == "RELIABLE"


class TestMotionHintScenarioYawOnly(_MotionHintScenarioBase):
    def test_small_yaw_applies_and_stays_reliable(self):
        # omega about Z (yaw) for the full interval: angle ~= 0.02 rad,
        # comfortably under reliability_max_angular_motion_rad (~0.0873).
        gf = self._run(motion_hints=[_hint(1.0, [0.0, 0.0, 0.02])])
        assert gf.rotation_compensation_status == "APPLIED"
        assert gf.motion_aware_reliability.state == "RELIABLE"


class TestMotionHintScenarioPitchOnly(_MotionHintScenarioBase):
    def test_small_pitch_applies_and_stays_reliable(self):
        gf = self._run(motion_hints=[_hint(1.0, [0.0, 0.02, 0.0])])
        assert gf.rotation_compensation_status == "APPLIED"
        assert gf.motion_aware_reliability.state == "RELIABLE"


class TestMotionHintScenarioRollOnly(_MotionHintScenarioBase):
    def test_small_roll_applies_and_stays_reliable(self):
        gf = self._run(motion_hints=[_hint(1.0, [0.02, 0.0, 0.0])])
        assert gf.rotation_compensation_status == "APPLIED"
        assert gf.motion_aware_reliability.state == "RELIABLE"


class TestMotionHintScenarioCombinedAngularMotion(_MotionHintScenarioBase):
    def test_small_combined_motion_applies_and_stays_reliable(self):
        gf = self._run(motion_hints=[_hint(1.0, [0.01, 0.01, 0.01])])
        assert gf.rotation_compensation_status == "APPLIED"
        assert gf.motion_aware_reliability.state == "RELIABLE"

    def test_excessive_combined_motion_applies_but_never_reads_reliable(self):
        # Well beyond reliability_max_angular_motion_rad (~0.0873 rad):
        # compensation still runs (APPLIED, per its own "was a rotation
        # integrated and used" definition). Measured (not assumed):
        # a rotation this large reprojects essentially all prior-frame
        # geometry outside the current pixel grid, so comparable_count
        # collapses to 0 and temporal_consistency itself reads
        # INSUFFICIENT_EVIDENCE (NOT_COMPARABLE) rather than
        # CONTRADICTORY — which correctly cascades to
        # MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE per E6's own
        # documented priority order (temporal_consistency_state is
        # checked before the angular-magnitude threshold). Whichever of
        # the three non-RELIABLE states results, RELIABLE must never be
        # the verdict for a rotation this far outside the configured
        # trust bound — that is the one safety property this scenario
        # actually proves.
        gf = self._run(motion_hints=[_hint(1.0, [0.5, 0.5, 0.5])])
        assert gf.rotation_compensation_status == "APPLIED"
        assert gf.motion_aware_reliability.state != "RELIABLE"
        assert gf.motion_aware_reliability.state in {"DEGRADED", "UNRELIABLE", "INSUFFICIENT_EVIDENCE"}


class TestMotionHintScenarioInvalidHint(_MotionHintScenarioBase):
    def test_valid_false_hint_falls_back_exactly_like_no_hint(self):
        gf_invalid = self._run(motion_hints=[_hint(1.0, [0.0, 0.0, 0.02], valid=False)])
        gf_none = self._run(motion_hints=None)
        assert gf_invalid.rotation_compensation_status == "NOT_APPLIED"
        assert gf_invalid.rotation_compensation_status == gf_none.rotation_compensation_status
        assert gf_invalid.temporal_consistency.state == gf_none.temporal_consistency.state


class TestMotionHintScenarioStaleHint(_MotionHintScenarioBase):
    def test_hint_timestamped_before_the_previous_frame_is_rejected_as_stale(self):
        # Outside (previous_timestamp, current_timestamp] entirely.
        gf = self._run(motion_hints=[_hint(-1.0, [0.0, 0.0, 0.02])])
        assert gf.rotation_compensation_status == "NOT_APPLIED"


class TestMotionHintScenarioInadequateCoverage(_MotionHintScenarioBase):
    def test_hint_covering_only_the_start_of_the_interval_applies_but_degrades_on_coverage(self):
        # Only sample is far short of current_timestamp (1.0) -> a real,
        # small motion_coverage_fraction, under
        # reliability_min_motion_coverage_fraction (0.5) -> compensation
        # still APPLIED (a real sample was integrated), but E6 must not
        # call this RELIABLE given how little of the interval it covers.
        gf = self._run(motion_hints=[_hint(0.1, [0.0, 0.0, 0.01])])
        assert gf.rotation_compensation_status == "APPLIED"
        assert gf.motion_aware_reliability.state in {"DEGRADED", "UNRELIABLE"}

    def test_hint_covering_the_full_interval_is_not_flagged_for_coverage(self):
        gf = self._run(motion_hints=[_hint(1.0, [0.0, 0.0, 0.01])])
        assert gf.rotation_compensation_status == "APPLIED"
        assert gf.motion_aware_reliability.state == "RELIABLE"


# ===================================================================
# 5. GeometryFrame is the only thing an external consumer needs — no
# source-specific knowledge required to interpret it.
# ===================================================================
class TestGeometryFrameNeedsNoSourceKnowledge:
    def test_a_generic_consumer_function_interprets_geometry_frame_using_only_public_vocabulary(self):
        """A stand-in for a larger external perception system: reads
        GeometryFrame using only dpe.FrameId / dpe.GeometryFrameQualityState
        -style public constants, never a private import, never a branch
        on where the frame came from."""
        def consume(gf: GeometryFrame) -> dict:
            assert gf.frame_id == dpe.FrameId.CAMERA_OPTICAL_LEFT
            return {
                "valid_fraction": gf.geometry_metrics.valid_fraction,
                "quality": gf.quality.overall_state,
                "obstacle_count": gf.obstacle_cloud.points.shape[0],
                "opening_count": len(gf.opening_evidence),
            }

        left, right = _raw_buffer_stereo_pair()
        result = _pipeline().process(left, right)
        summary = consume(result.geometry_frame)
        assert summary["quality"] in {"VALID", "DEGRADED", "INSUFFICIENT"}
        assert summary["obstacle_count"] >= 0
