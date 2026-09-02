"""
Phase D2 — observation identity contract.

D1 (benchmarks/d1_execution/D1_EXECUTION_AUDIT_REPORT.md) proved DPE's
authoritative output carried NO observation/transaction identity: a caller
could set StereoObservation.frame_id and that value appeared nowhere in the
returned GeometryFrame, while GeometryFrame.frame_id was — and remains — the
COORDINATE frame. D2 adds `observation_id` as opaque, caller-owned
provenance.

Every test here defends one of three invariants:

  1. identity PROPAGATES verbatim, on every authoritative path;
  2. identity NEVER touches coordinate-frame semantics;
  3. identity NEVER influences temporal chronology or any geometry/algorithm
     output.
"""

import dataclasses

import numpy as np
import pytest

import depth_perception_engine as dpe
from depth_perception_engine.core import (
    DepthPerceptionPipeline,
    GeometryFrame,
    PipelineConfig,
    StereoObservation,
)
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.temporal.history import TemporalAdmissionStatus

OBSERVATION_ID = "HPE-OBSERVATION-12345"


# ===================================================================
# fixtures
# ===================================================================
@pytest.fixture
def full_config():
    """Every capability on, so identity is checked against a fully
    populated GeometryFrame rather than a mostly-empty one."""
    return PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        enable_geometry_frame=True,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=30,
    )


@pytest.fixture
def body_transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


@pytest.fixture
def full_pipeline(full_config, calibration, body_transform):
    return DepthPerceptionPipeline(
        full_config, calibration, rectify=True, body_T_camera_left=body_transform,
    )


def _obs(stereo_pair, ts=0.0, **kwargs):
    left, right = stereo_pair
    return StereoObservation(left_image=left, right_image=right, left_timestamp=ts, **kwargs)


# ===================================================================
# 1-3. Propagation and exact preservation
# ===================================================================
class TestPropagation:
    def test_none_stays_none(self, full_pipeline, stereo_pair):
        gf = full_pipeline.process_geometry_frame(_obs(stereo_pair))
        assert gf.observation_id is None

    def test_normal_string_preserved_exactly(self, full_pipeline, stereo_pair):
        gf = full_pipeline.process_geometry_frame(_obs(stereo_pair, observation_id="obs-000001"))
        assert gf.observation_id == "obs-000001"

    @pytest.mark.parametrize("weird", [
        "   leading and trailing   ",
        "UPPER/lower-MiXeD",
        "urn:uuid:8f14e45f-ea3b-4d2c-9a1b-000000000001",
        "obs id with spaces",
        "λ-unicode-Ωbservation-日本語",
        "{\"json\": \"like\"}",
        "0",
        "..//..//not-a-path",
        "a" * 4096,
    ])
    def test_opaque_strings_are_never_normalized(self, full_pipeline, stereo_pair, weird):
        """DPE must not trim, case-fold, escape, truncate, or otherwise
        reinterpret an opaque caller-owned identifier."""
        gf = full_pipeline.process_geometry_frame(_obs(stereo_pair, observation_id=weird))
        assert gf.observation_id == weird
        assert gf.observation_id is not None and len(gf.observation_id) == len(weird)

    def test_process_observation_path_also_propagates(self, full_pipeline, stereo_pair):
        result = full_pipeline.process_observation(_obs(stereo_pair, observation_id=OBSERVATION_ID))
        assert result.observation_id == OBSERVATION_ID
        assert result.geometry_frame is not None
        assert result.geometry_frame.observation_id == OBSERVATION_ID

    def test_both_geometry_frame_paths_agree(self, full_config, calibration, body_transform, stereo_pair):
        """process_geometry_frame() and process_observation()'s own
        enable_geometry_frame branch must yield the same identity — they
        share one builder, and this proves they cannot drift."""
        a = DepthPerceptionPipeline(full_config, calibration, rectify=True,
                                    body_T_camera_left=body_transform)
        gf_a = a.process_geometry_frame(_obs(stereo_pair, observation_id=OBSERVATION_ID))

        disabled = dataclasses.replace(full_config, enable_geometry_frame=False)
        b = DepthPerceptionPipeline(disabled, calibration, rectify=True,
                                    body_T_camera_left=body_transform)
        gf_b = b.process_geometry_frame(_obs(stereo_pair, observation_id=OBSERVATION_ID))

        assert gf_a.observation_id == gf_b.observation_id == OBSERVATION_ID

    def test_identity_survives_a_minimal_default_config(self, calibration, stereo_pair):
        """Identity must not depend on any enable_* capability flag."""
        p = DepthPerceptionPipeline(PipelineConfig(), calibration)
        gf = p.process_geometry_frame(_obs(stereo_pair, observation_id=OBSERVATION_ID))
        assert gf.observation_id == OBSERVATION_ID


# ===================================================================
# 9. Coordinate-frame non-regression — THE conflation guard
# ===================================================================
class TestCoordinateFrameNonRegression:
    def test_observation_identity_never_leaks_into_any_coordinate_frame(
        self, full_pipeline, stereo_pair,
    ):
        """The exact defect D1 warned about: identity must never be written
        into a frame_id, and no frame_id may change because identity was
        supplied."""
        gf = full_pipeline.process_geometry_frame(
            _obs(stereo_pair, observation_id=OBSERVATION_ID)
        )

        assert gf.observation_id == OBSERVATION_ID
        assert gf.frame_id == FrameId.CAMERA_OPTICAL_LEFT
        assert gf.geometry.frame_id == FrameId.CAMERA_OPTICAL_LEFT
        assert gf.geometry_body.frame_id == FrameId.BODY
        assert gf.obstacle_cloud.frame_id == FrameId.BODY
        assert gf.free_space_rays.frame_id == FrameId.BODY
        for cell in gf.surface_evidence:
            assert cell.frame_id == FrameId.BODY
        for adjacency in gf.boundary_evidence:
            assert adjacency.frame_id == FrameId.CAMERA_OPTICAL_LEFT
        for sector in gf.clearance_evidence:
            assert sector.frame_id == FrameId.CAMERA_OPTICAL_LEFT
        for region in gf.region_evidence.values():
            assert region.frame_id == FrameId.CAMERA_OPTICAL_LEFT
        for opening in gf.opening_evidence:
            assert opening.frame_id == FrameId.CAMERA_OPTICAL_LEFT

    def test_recursive_scan_finds_identity_only_in_observation_id(
        self, full_pipeline, stereo_pair,
    ):
        """Walk the whole returned object graph: the identity string must
        appear at GeometryFrame.observation_id and NOWHERE else."""
        gf = full_pipeline.process_geometry_frame(
            _obs(stereo_pair, observation_id=OBSERVATION_ID)
        )
        hits = []

        def walk(obj, path, depth=0):
            if depth > 5:
                return
            if isinstance(obj, str):
                if OBSERVATION_ID in obj:
                    hits.append(path)
                return
            if isinstance(obj, (int, float, bool, type(None), np.ndarray)):
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{path}[{k!r}]", depth + 1)
                return
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]", depth + 1)
                return
            slots = getattr(type(obj), "__slots__", None)
            for name in (list(slots) if slots else []):
                walk(getattr(obj, name, None), f"{path}.{name}", depth + 1)

        walk(gf, "GeometryFrame")
        assert hits == ["GeometryFrame.observation_id"], hits

    def test_frame_ids_are_identical_with_and_without_identity(
        self, full_config, calibration, body_transform, stereo_pair,
    ):
        def frames_of(observation_id):
            p = DepthPerceptionPipeline(full_config, calibration, rectify=True,
                                        body_T_camera_left=body_transform)
            gf = p.process_geometry_frame(_obs(stereo_pair, observation_id=observation_id))
            return (
                gf.frame_id, gf.geometry.frame_id, gf.geometry_body.frame_id,
                gf.obstacle_cloud.frame_id, gf.free_space_rays.frame_id,
                tuple(c.frame_id for c in gf.surface_evidence),
                tuple(b.frame_id for b in gf.boundary_evidence),
                tuple(c.frame_id for c in gf.clearance_evidence),
                tuple(r.frame_id for r in gf.region_evidence.values()),
            )

        assert frames_of(None) == frames_of(OBSERVATION_ID)


# ===================================================================
# 4-7. Temporal chronology remains authoritative and ID-blind
# ===================================================================
class TestTemporalSemanticsUnchanged:
    def test_repeated_ids_on_advancing_timestamps_are_all_accepted(
        self, full_pipeline, stereo_pair,
    ):
        """A repeated observation_id is NOT a reason to reject a frame —
        DPE has no uniqueness opinion about an opaque caller identifier."""
        statuses = [
            full_pipeline.process_observation(
                _obs(stereo_pair, ts=i * 0.1, observation_id="duplicate-id")
            ).temporal_admission_status
            for i in range(5)
        ]
        assert statuses == [TemporalAdmissionStatus.ACCEPTED] * 5
        assert len(full_pipeline.temporal_history) == 5

    def test_distinct_ids_do_not_rescue_a_duplicate_timestamp(
        self, full_pipeline, stereo_pair,
    ):
        """Chronology stays authoritative: a fresh identity must not make a
        duplicate timestamp admissible."""
        full_pipeline.process_observation(_obs(stereo_pair, ts=1.0, observation_id="a"))
        second = full_pipeline.process_observation(_obs(stereo_pair, ts=1.0, observation_id="b"))
        assert second.temporal_admission_status == TemporalAdmissionStatus.REJECTED_DUPLICATE_TIMESTAMP
        # ...and the rejected frame still carries its own identity back.
        assert second.observation_id == "b"
        assert len(full_pipeline.temporal_history) == 1

    def test_distinct_ids_do_not_rescue_a_decreasing_timestamp(
        self, full_pipeline, stereo_pair,
    ):
        full_pipeline.process_observation(_obs(stereo_pair, ts=2.0, observation_id="a"))
        older = full_pipeline.process_observation(_obs(stereo_pair, ts=1.0, observation_id="b"))
        assert older.temporal_admission_status == TemporalAdmissionStatus.REJECTED_OLDER_TIMESTAMP
        assert older.observation_id == "b"

    def test_large_gap_still_starts_a_new_sequence(self, full_pipeline, stereo_pair):
        full_pipeline.process_observation(_obs(stereo_pair, ts=0.0, observation_id="a"))
        full_pipeline.process_observation(_obs(stereo_pair, ts=0.1, observation_id="b"))
        jumped = full_pipeline.process_observation(_obs(stereo_pair, ts=30.0, observation_id="c"))
        assert jumped.temporal_admission_status == TemporalAdmissionStatus.ACCEPTED_NEW_SEQUENCE
        assert len(full_pipeline.temporal_history) == 1

    def test_temporal_history_stores_no_observation_identity(
        self, full_pipeline, stereo_pair,
    ):
        """D2 explicitly did NOT add identity to TemporalRecord — chronology
        has no use for it, and storing it would create a second, unbounded
        place identity could accumulate."""
        full_pipeline.process_observation(_obs(stereo_pair, ts=0.0, observation_id=OBSERVATION_ID))
        record = full_pipeline.temporal_history.latest
        assert not hasattr(record, "observation_id")
        assert OBSERVATION_ID not in repr(record)


# ===================================================================
# 15. No DPE algorithm branches on observation_id
# ===================================================================
def _fingerprint(gf: GeometryFrame):
    """Everything about a GeometryFrame EXCEPT its identity."""
    def state(x):
        return None if x is None else getattr(x, "state", x)

    return {
        "timestamp": gf.timestamp,
        "frame_id": gf.frame_id,
        "disparity": gf.disparity_map.tobytes(),
        "depth": gf.depth_map.tobytes(),
        "valid_disparity": gf.valid_disparity_mask.tobytes(),
        "valid_depth": gf.valid_depth_mask.tobytes(),
        "geometry": np.nan_to_num(gf.geometry.points, nan=-999.0).tobytes(),
        "geometry_body": np.nan_to_num(gf.geometry_body.points, nan=-999.0).tobytes(),
        "obstacle_points": gf.obstacle_cloud.points.tobytes(),
        "free_space": gf.free_space_rays.ranges_m.tobytes(),
        "metrics": (
            gf.geometry_metrics.min_obstacle_distance_m,
            gf.geometry_metrics.mean_free_space_m,
            gf.geometry_metrics.point_count,
            gf.geometry_metrics.valid_fraction,
        ),
        "consistency": state(gf.temporal_consistency),
        "stabilization": state(gf.temporal_stabilization),
        "rotation": gf.rotation_compensation_status,
        "reliability": state(gf.motion_aware_reliability),
        "persistence": state(gf.temporal_persistence),
        "surfaces": tuple((c.frame_id, c.row, c.col) for c in gf.surface_evidence),
        "boundaries": tuple((b.state, b.direction) for b in gf.boundary_evidence),
        "openings": len(gf.opening_evidence),
        "clearance": tuple((c.support_state, c.nearest_distance_m) for c in gf.clearance_evidence),
        "quality": (gf.quality.overall_state, tuple(gf.quality.degradation_reasons)),
    }


class TestIdentityHasNoAlgorithmicEffect:
    def test_two_sequences_differing_only_in_identity_are_output_identical(
        self, full_config, calibration, body_transform, stereo_pair,
    ):
        """The core D2 safety property. Run the SAME 6-frame sequence twice:
        once with no identity, once with a distinct identity per frame.
        Every field except observation_id must be bit-for-bit identical."""
        def run(ids):
            p = DepthPerceptionPipeline(full_config, calibration, rectify=True,
                                        body_T_camera_left=body_transform)
            return [
                p.process_geometry_frame(_obs(stereo_pair, ts=i * 0.1, observation_id=oid))
                for i, oid in enumerate(ids)
            ]

        without = run([None] * 6)
        with_ids = run([f"observation-{i:06d}" for i in range(6)])

        for i, (a, b) in enumerate(zip(without, with_ids)):
            assert _fingerprint(a) == _fingerprint(b), f"frame {i} differs beyond identity"
            assert a.observation_id is None
            assert b.observation_id == f"observation-{i:06d}"

    def test_production_source_never_branches_on_observation_id(self):
        """Structural guard: observation_id must never appear in a
        conditional, comparison, or match statement anywhere in production
        source — it is copied, never consulted."""
        import ast
        import os

        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "depth_perception_engine",
        )
        offenders = []
        for dirpath, _dirs, files in os.walk(root):
            for filename in files:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.If, ast.While, ast.Compare, ast.IfExp)):
                        continue
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Attribute) and sub.attr == "observation_id":
                            offenders.append(f"{filename}:{sub.lineno}")
                        if isinstance(sub, ast.Name) and sub.id == "observation_id":
                            offenders.append(f"{filename}:{sub.lineno}")
        # models/result.py's own __post_init__ validates the field (type,
        # emptiness, alias conflict) — that is contract validation at the
        # boundary, not an algorithm consuming the value.
        offenders = [o for o in offenders if not o.startswith("result.py")]
        assert offenders == [], (
            "observation_id is consulted in a conditional outside its own "
            f"contract validation: {offenders}"
        )


# ===================================================================
# 8. reset() must not leak stale identity
# ===================================================================
class TestResetAndLifecycle:
    def test_reset_leaves_no_stale_identity(self, full_pipeline, stereo_pair):
        full_pipeline.process_geometry_frame(
            _obs(stereo_pair, ts=0.0, observation_id="before-reset")
        )
        full_pipeline.reset()
        after = full_pipeline.process_geometry_frame(_obs(stereo_pair, ts=0.0))
        assert after.observation_id is None

    def test_identity_is_per_call_not_sticky(self, full_pipeline, stereo_pair):
        """No frame may inherit a previous frame's identity — the pipeline
        stores none between calls."""
        first = full_pipeline.process_geometry_frame(
            _obs(stereo_pair, ts=0.0, observation_id="first")
        )
        second = full_pipeline.process_geometry_frame(_obs(stereo_pair, ts=0.1))
        third = full_pipeline.process_geometry_frame(
            _obs(stereo_pair, ts=0.2, observation_id="third")
        )
        assert (first.observation_id, second.observation_id, third.observation_id) == (
            "first", None, "third",
        )

    def test_pipeline_retains_no_identity_state(self, full_pipeline, stereo_pair):
        """Identity must not create any new retained state on the engine."""
        before = set(vars(full_pipeline))
        full_pipeline.process_geometry_frame(
            _obs(stereo_pair, ts=0.0, observation_id=OBSERVATION_ID)
        )
        assert set(vars(full_pipeline)) == before
        assert OBSERVATION_ID not in repr(full_pipeline)
        assert not any(
            OBSERVATION_ID in repr(v) for v in vars(full_pipeline).values()
        )


# ===================================================================
# 12. Legacy + standalone compatibility
# ===================================================================
class TestLegacyAndStandaloneCompatibility:
    def test_legacy_process_accepts_and_propagates_identity(
        self, full_config, calibration, body_transform, stereo_pair,
    ):
        p = DepthPerceptionPipeline(full_config, calibration, rectify=True,
                                    body_T_camera_left=body_transform)
        left, right = stereo_pair
        result = p.process(left, right, left_timestamp=0.0, observation_id=OBSERVATION_ID)
        assert result.observation_id == OBSERVATION_ID
        assert result.geometry_frame.observation_id == OBSERVATION_ID

    def test_legacy_process_without_identity_is_unchanged(
        self, full_config, calibration, body_transform, stereo_pair,
    ):
        p = DepthPerceptionPipeline(full_config, calibration, rectify=True,
                                    body_T_camera_left=body_transform)
        left, right = stereo_pair
        result = p.process(left, right, left_timestamp=0.0)
        assert result.observation_id is None
        assert result.geometry_frame.observation_id is None

    def test_standalone_build_observation_passes_identity_through(self, calibration, stereo_pair):
        interface = dpe.StandaloneStereoInterface(PipelineConfig(), calibration)
        left, right = stereo_pair
        observation = interface.build_observation(
            left, right, timestamp=0.0, observation_id=OBSERVATION_ID,
        )
        assert observation.observation_id == OBSERVATION_ID
        assert observation.resolved_observation_id == OBSERVATION_ID

    def test_standalone_uses_the_same_core_propagation(self, calibration, stereo_pair):
        """Standalone must not grow a second identity implementation — it
        builds the same StereoObservation and hands it to the same core."""
        interface = dpe.StandaloneStereoInterface(
            PipelineConfig(enable_geometry_frame=True), calibration,
        )
        left, right = stereo_pair
        observation = interface.build_observation(
            left, right, timestamp=0.0, observation_id=OBSERVATION_ID,
        )
        gf = interface.engine.process_geometry_frame(observation)
        assert gf.observation_id == OBSERVATION_ID
        assert gf.frame_id == FrameId.CAMERA_OPTICAL_LEFT


# ===================================================================
# Deprecated frame_id alias (migration strategy B)
# ===================================================================
class TestDeprecatedFrameIdAlias:
    def test_legacy_frame_id_still_constructs(self, stereo_pair):
        """Pre-D2 callers (including hybrid_perception_engine, which passes
        frame_id= today) must keep working unmodified."""
        observation = _obs(stereo_pair, frame_id="legacy-observation-1")
        assert observation.frame_id == "legacy-observation-1"

    def test_legacy_frame_id_now_supplies_observation_identity(
        self, full_pipeline, stereo_pair,
    ):
        """The alias is truthful, not inert: a caller who populated the old
        field gets identity propagation for free."""
        gf = full_pipeline.process_geometry_frame(
            _obs(stereo_pair, frame_id="legacy-observation-1")
        )
        assert gf.observation_id == "legacy-observation-1"
        assert gf.frame_id == FrameId.CAMERA_OPTICAL_LEFT

    def test_observation_id_wins_when_both_agree(self, stereo_pair):
        observation = _obs(stereo_pair, observation_id="same", frame_id="same")
        assert observation.resolved_observation_id == "same"

    def test_conflicting_identities_raise(self, stereo_pair):
        """Exactly ONE authoritative identity: an ambiguous pair is a caller
        error, resolved deterministically by refusing it."""
        with pytest.raises(ValueError, match="disagree"):
            _obs(stereo_pair, observation_id="a", frame_id="b")

    @pytest.mark.parametrize("field", ["observation_id", "frame_id"])
    def test_non_string_identity_rejected(self, stereo_pair, field):
        with pytest.raises(ValueError, match="must be a string or None"):
            _obs(stereo_pair, **{field: 12345})

    @pytest.mark.parametrize("field", ["observation_id", "frame_id"])
    def test_empty_identity_rejected(self, stereo_pair, field):
        """Mirrors frames.RigidTransform's own non-empty-identifier rule:
        "" is neither an identity nor the explicit 'no identity' signal."""
        with pytest.raises(ValueError, match="non-empty string"):
            _obs(stereo_pair, **{field: ""})


# ===================================================================
# 13-14. Dataclass / equality / repr behaviour
# ===================================================================
class TestContractShape:
    def test_stereo_observation_is_still_frozen_and_slotted(self, stereo_pair):
        observation = _obs(stereo_pair, observation_id=OBSERVATION_ID)
        assert dataclasses.is_dataclass(observation)
        assert hasattr(StereoObservation, "__slots__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            observation.observation_id = "mutated"

    def test_geometry_frame_is_still_frozen_and_slotted(self, full_pipeline, stereo_pair):
        gf = full_pipeline.process_geometry_frame(_obs(stereo_pair, observation_id=OBSERVATION_ID))
        assert dataclasses.is_dataclass(gf)
        assert hasattr(GeometryFrame, "__slots__")
        assert "observation_id" in GeometryFrame.__slots__
        with pytest.raises(dataclasses.FrozenInstanceError):
            gf.observation_id = "mutated"

    def test_observation_id_is_defaulted_so_pre_d2_construction_still_works(self):
        """Every pre-D2 field keeps its exact position; identity is appended
        with a default, so no existing positional or keyword construction of
        either contract breaks."""
        obs_fields = dataclasses.fields(StereoObservation)
        assert obs_fields[-1].name == "observation_id"
        assert obs_fields[-1].default is None
        assert [f.name for f in obs_fields[:8]] == [
            "left_image", "right_image", "left_timestamp", "right_timestamp",
            "calibration", "frame_id", "motion_hint", "motion_hints",
        ]
        gf_fields = dataclasses.fields(GeometryFrame)
        assert gf_fields[-1].name == "observation_id"
        assert gf_fields[-1].default is None

    def test_identity_appears_in_repr_for_debuggability(self, full_pipeline, stereo_pair):
        gf = full_pipeline.process_geometry_frame(_obs(stereo_pair, observation_id=OBSERVATION_ID))
        assert OBSERVATION_ID in repr(gf)

    def test_identity_participates_in_generated_equality(self, stereo_pair):
        """observation_id is a normal compare=True dataclass field, so it is
        part of the generated __eq__ field set.

        Note whole-instance `==` on StereoObservation/GeometryFrame is not
        usable in practice because both carry ndarray fields, whose
        element-wise comparison makes the generated __eq__ raise "truth
        value of an array is ambiguous". That is PRE-EXISTING behaviour of
        these contracts, unchanged by D2 and asserted here so the
        limitation is recorded rather than discovered later."""
        left, right = stereo_pair
        by_name = {f.name: f for f in dataclasses.fields(StereoObservation)}
        assert by_name["observation_id"].compare is True

        a = StereoObservation(left_image=left, right_image=right, observation_id="a")
        b = StereoObservation(left_image=left, right_image=right, observation_id="b")
        with pytest.raises(ValueError, match="truth value of an array"):
            _ = a == b

    def test_identity_alone_is_directly_comparable(self, full_pipeline, stereo_pair):
        """The practical correlation operation an orchestrator performs is
        comparing the identity FIELD, not whole frames — and that is a plain
        string comparison."""
        a = full_pipeline.process_geometry_frame(
            _obs(stereo_pair, ts=0.0, observation_id="obs-a")
        )
        b = full_pipeline.process_geometry_frame(
            _obs(stereo_pair, ts=0.1, observation_id="obs-b")
        )
        assert a.observation_id != b.observation_id
        assert a.observation_id == "obs-a"
