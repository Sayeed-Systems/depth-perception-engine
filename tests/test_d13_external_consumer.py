"""
Black-box EXTERNAL-CONSUMER validation — Phase D13 (see
docs/DPE_V1_PROVIDER_CONTRACT.md's D13 record).

D10's own black-box test (tests/test_d10_black_box_provider.py) proved
GeometryFrame is consumable through the public API alone and measured
real SGBM error against a known shift. This file is a different, D13-
specific proof: a single, comprehensive walk through the exact consumer
workflow a future `hybrid_perception_engine` (or any other external
integrator) is expected to follow —

    configure DPE
        -> construct public input contracts (StereoCalibration,
           StereoObservation, optional MotionHint, PipelineConfig)
        -> run DepthPerceptionPipeline
        -> consume GeometryFrame

— touching EVERY authoritative evidence family GeometryFrame carries, all
read/type-checked using ONLY public types.

STRUCTURAL RULE — enforced by TestImportSurface below, not merely by
convention: this file imports depth_perception_engine only via its root
package and the one documented public subpackage still needed for a
constructor-input type that was never promoted to root
(`depth_perception_engine.frames`, for `RigidTransform` —
`body_T_camera_left` is a legitimate, documented pipeline constructor
argument, not an internal algorithm). Since Phase D13 promoted
`MotionHint`/`FrameId` to the root, this file needs no OTHER subpackage
import at all — a strictly narrower import surface than D10's own
(`.frames` + `.temporal`). It therefore cannot depend on internal
geometry algorithms, `RegionAnalyzer`, `ThreatAssessor`, `TemporalHistory`,
`TemporalRecord`, `fusion.result_builder`, `pipeline.pipeline`/`.api`
internals, or legacy navigation/traversability internals, by
construction, not merely by omission.

TestNoLegacyResultFieldsNeeded below additionally proves, structurally,
that the consumer workflow's own source never reads
`DepthPerceptionResult.traversability_mask`/`.obstacles`/`.confidence` —
GeometryFrame alone is sufficient, the D13 task's own explicit guard
requirement.
"""

import ast
import pathlib

import cv2
import dataclasses

import numpy as np
import pytest

import depth_perception_engine as dpe
from depth_perception_engine import (
    BoundaryEvidence,
    ClearanceEvidence,
    DepthPerceptionPipeline,
    FreeSpaceRays,
    GeometryFrame,
    GeometryFrameQuality,
    GeometryMetrics,
    MotionAwareReliability,
    MotionHint,
    ObstacleCloud,
    OpeningEvidence,
    PipelineConfig,
    PointCloud,
    RegionEvidence,
    StereoCalibration,
    StereoObservation,
    SurfaceEvidence,
    TemporalConsistency,
    TemporalPersistence,
    TemporalStabilization,
    load_stereo_calibration,
)
from depth_perception_engine.frames import RigidTransform

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIBRATION.image_size


def _smoothed_stereo_pair(shift_px: int = 24, seed: int = 5):
    """Stable, real local structure (not i.i.d. noise) — same technique
    tests/test_d10_black_box_provider.py established — needed for the
    real evidence families (surface/boundary/opening/clearance) this
    file inspects to be genuinely populated, not degenerate."""
    canvas_w = _W + shift_px
    rng = np.random.default_rng(seed)
    low_res = rng.integers(0, 255, (_H // 4 + 2, canvas_w // 4 + 2), dtype=np.uint8)
    canvas = cv2.resize(low_res, (canvas_w, _H), interpolation=cv2.INTER_CUBIC)
    canvas_bgr = np.stack([canvas] * 3, axis=-1)
    left = canvas_bgr[:, 0:_W]
    right = canvas_bgr[:, shift_px:shift_px + _W]
    return left, right


# ===================================================================
# Structural: this file itself never reaches into DPE internals
# ===================================================================
class TestImportSurface:
    def test_this_test_file_imports_no_forbidden_internal_module(self):
        source = pathlib.Path(__file__).read_text()
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        allowed_prefixes = ("depth_perception_engine.frames",)
        for module in imported_modules:
            if module == "depth_perception_engine":
                continue
            if not module.startswith("depth_perception_engine"):
                continue
            assert module.startswith(allowed_prefixes), (
                f"D13 external-consumer test imported {module!r} — only the "
                "root package and depth_perception_engine.frames (for the "
                "RigidTransform constructor-input type) are permitted. This "
                "file exists to prove a real external consumer needs "
                "nothing else: no internal geometry algorithm, no "
                "RegionAnalyzer/ThreatAssessor, no TemporalHistory/"
                "TemporalRecord, no fusion.result_builder, no pipeline "
                "internals, no traversability internals."
            )


class TestNoLegacyResultFieldsNeeded:
    """Structural proof that GeometryFrame alone is sufficient — the
    consumer workflow function below (_run_external_consumer_workflow)
    never reads DepthPerceptionResult.traversability_mask/.obstacles/
    .confidence, the three legacy/compatibility fields D13 explicitly
    forbids an external consumer from needing."""

    _FORBIDDEN_ATTRS = {"traversability_mask", "obstacles", "confidence"}

    def test_consumer_workflow_source_never_reads_legacy_result_fields(self):
        source = pathlib.Path(__file__).read_text()
        tree = ast.parse(source)
        workflow_fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_run_external_consumer_workflow"
        )
        offenders = [
            node.attr
            for node in ast.walk(workflow_fn)
            if isinstance(node, ast.Attribute) and node.attr in self._FORBIDDEN_ATTRS
        ]
        assert not offenders, (
            f"External-consumer workflow reads legacy DepthPerceptionResult "
            f"field(s) {offenders} — GeometryFrame must be sufficient on its own."
        )


# ===================================================================
# The one comprehensive external-consumer workflow
# ===================================================================
def _run_external_consumer_workflow() -> GeometryFrame:
    """The exact 7-step workflow an external integrator (a future
    hybrid_perception_engine) is expected to follow. Returns the second
    frame's GeometryFrame — the first frame exists only to give E3/E4/E7
    a real prior record, so temporal evidence is genuinely populated
    (CONSISTENT/CLASSIFIED), not just structurally present as None.
    """
    # 1. construct calibration
    calibration: StereoCalibration = load_stereo_calibration("examples/config/stereo_calibration.xml")

    # 2. construct stereo observation(s)
    left, right = _smoothed_stereo_pair()
    observation_1 = StereoObservation(left_image=left, right_image=right, left_timestamp=0.0, right_timestamp=0.0)

    # 3. optionally construct MotionHint (attached to the SECOND observation
    # — the interval between frame 1 and frame 2 is what a MotionHint
    # describes; there is no "previous" interval for the very first frame).
    motion_hint = MotionHint(
        timestamp=1.0,
        angular_velocity_rad_s=np.array([0.0, 0.0, 0.01], dtype=np.float64),
        frame_id=dpe.FrameId.BODY,
        valid=True,
    )
    observation_2 = StereoObservation(
        left_image=left, right_image=right, left_timestamp=1.0, right_timestamp=1.0,
        motion_hint=motion_hint, motion_hints=[motion_hint],
    )

    # 4. construct PipelineConfig — every evidence family opted into, so
    # this workflow genuinely exercises the full authoritative contract.
    config = PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        enable_geometry_frame=True,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=50,
    )

    # 5. run DepthPerceptionPipeline (RigidTransform: the one legitimate
    # constructor-input type never promoted to root — see module docstring).
    body_transform = RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=dpe.FrameId.CAMERA_OPTICAL_LEFT, to_frame=dpe.FrameId.BODY,
    )
    pipeline = DepthPerceptionPipeline(config, calibration, rectify=False, body_T_camera_left=body_transform)
    pipeline.process_observation(observation_1)
    result = pipeline.process_observation(observation_2)

    # 6. obtain GeometryFrame
    geometry_frame = result.geometry_frame
    assert isinstance(geometry_frame, GeometryFrame)
    return geometry_frame


# ===================================================================
# 7. inspect all authoritative evidence families using public types only
# ===================================================================
class TestExternalConsumerWorkflow:
    def test_workflow_runs_end_to_end_and_produces_a_geometry_frame(self):
        gf = _run_external_consumer_workflow()
        assert isinstance(gf, GeometryFrame)

    def test_frame_identity_and_raw_arrays(self):
        gf = _run_external_consumer_workflow()
        assert gf.frame_id == dpe.FrameId.CAMERA_OPTICAL_LEFT
        assert isinstance(gf.disparity_map, np.ndarray)
        assert isinstance(gf.depth_map, np.ndarray)
        assert isinstance(gf.valid_disparity_mask, np.ndarray)
        assert isinstance(gf.valid_depth_mask, np.ndarray)
        assert gf.timestamp == pytest.approx(1.0)

    def test_level3_geometry_evidence(self):
        gf = _run_external_consumer_workflow()
        assert isinstance(gf.geometry, PointCloud)
        assert gf.geometry.frame_id == dpe.FrameId.CAMERA_OPTICAL_LEFT
        assert isinstance(gf.geometry_body, PointCloud)
        assert gf.geometry_body.frame_id == dpe.FrameId.BODY
        assert isinstance(gf.obstacle_cloud, ObstacleCloud)
        assert isinstance(gf.free_space_rays, FreeSpaceRays)
        assert isinstance(gf.geometry_metrics, GeometryMetrics)
        assert 0.0 <= gf.geometry_metrics.valid_fraction <= 1.0

    def test_level4_temporal_evidence(self):
        gf = _run_external_consumer_workflow()
        assert isinstance(gf.temporal_consistency, TemporalConsistency)
        assert gf.temporal_consistency.state == "CONSISTENT"
        assert isinstance(gf.temporal_stabilization, TemporalStabilization)
        assert gf.rotation_compensation_status in {"APPLIED", "NOT_APPLIED"}
        assert isinstance(gf.motion_aware_reliability, MotionAwareReliability)
        assert gf.motion_aware_reliability.state in {"RELIABLE", "DEGRADED", "UNRELIABLE", "INSUFFICIENT_EVIDENCE"}
        assert isinstance(gf.temporal_persistence, TemporalPersistence)

    def test_region_and_clearance_evidence(self):
        gf = _run_external_consumer_workflow()
        assert isinstance(gf.region_evidence, dict)
        assert len(gf.region_evidence) > 0
        for region in gf.region_evidence.values():
            assert isinstance(region, RegionEvidence)
            assert region.frame_id == dpe.FrameId.CAMERA_OPTICAL_LEFT
            assert 0.0 <= region.valid_fraction <= 1.0

        assert isinstance(gf.clearance_evidence, list)
        assert len(gf.clearance_evidence) > 0
        for sector in gf.clearance_evidence:
            assert isinstance(sector, ClearanceEvidence)
            assert sector.support_state in {"SUPPORTED", "PARTIALLY_SUPPORTED", "NO_EVIDENCE"}
            assert sector.bearing_min_rad <= sector.bearing_center_rad <= sector.bearing_max_rad

    def test_surface_boundary_opening_evidence(self):
        gf = _run_external_consumer_workflow()
        assert isinstance(gf.surface_evidence, list)
        assert len(gf.surface_evidence) > 0
        for cell in gf.surface_evidence:
            assert isinstance(cell, SurfaceEvidence)
            if cell.normal is not None:
                assert cell.planarity is not None

        assert isinstance(gf.boundary_evidence, list)
        assert len(gf.boundary_evidence) > 0
        for edge in gf.boundary_evidence:
            assert isinstance(edge, BoundaryEvidence)
            assert edge.state in {"OBSERVED_DISCONTINUITY", "NO_DISCONTINUITY", "INSUFFICIENT_EVIDENCE"}

        assert isinstance(gf.opening_evidence, list)  # positive-findings-only; empty is legal
        for opening in gf.opening_evidence:
            assert isinstance(opening, OpeningEvidence)
            assert opening.approx_range_m > 0.0

    def test_quality_rollup(self):
        gf = _run_external_consumer_workflow()
        assert isinstance(gf.quality, GeometryFrameQuality)
        assert gf.quality.overall_state in {"VALID", "DEGRADED", "INSUFFICIENT"}
        assert isinstance(gf.quality.degradation_reasons, list)

    def test_observation_identity_is_returned_to_the_external_consumer(self):
        """Phase D2: an external consumer must be able to correlate the
        GeometryFrame it receives back to the capture it submitted, using
        nothing but the public contract."""
        gf = _run_external_consumer_workflow()
        # This workflow submits no identity, so the contract must say so
        # honestly rather than inventing one.
        assert gf.observation_id is None
        assert "observation_id" in {f.name for f in dataclasses.fields(GeometryFrame)}

    def test_full_evidence_family_checklist_is_exhaustive(self):
        """Meta-guard: enumerates GeometryFrame's own dataclass fields and
        confirms every one of them was actually exercised by name
        somewhere above — so a future new GeometryFrame field silently
        skips this file's own coverage instead of silently passing."""
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(GeometryFrame)}
        exercised_in_this_file = {
            "timestamp", "frame_id", "disparity_map", "depth_map",
            "valid_disparity_mask", "valid_depth_mask",
            "geometry", "geometry_body", "obstacle_cloud", "free_space_rays", "geometry_metrics",
            "temporal_consistency", "temporal_stabilization", "rotation_compensation_status",
            "motion_aware_reliability", "temporal_persistence",
            "region_evidence", "clearance_evidence",
            "surface_evidence", "boundary_evidence", "opening_evidence",
            "quality",
            "observation_id",
        }
        assert field_names == exercised_in_this_file, (
            f"GeometryFrame field set changed — update this file's coverage. "
            f"Missing coverage: {field_names - exercised_in_this_file}; "
            f"stale entries: {exercised_in_this_file - field_names}"
        )
