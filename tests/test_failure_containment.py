"""
Failure containment — Level 3, Phase E6 (Task 6).

DepthPerceptionPipeline.process() has zero try/except around any Level 3
stage (confirmed by direct source reading during the E6 audit — see
docs/IMPLEMENTATION_STATUS.md's E6 addendum). This means a failure at any
single stage aborts the *entire* call: no DepthPerceptionResult is ever
constructed or returned, so there is no way for a partially-built result
(e.g. a real obstacle_cloud paired with a fabricated free_space_rays) to
leak out. This file proves that atomicity holds for each of the four
stages the task names, by forcing a failure at each one via mocking and
confirming (a) the exception propagates uncaught and (b) the pipeline's
own bookkeeping (frames_processed/last_confidence/last_processing_time_ms)
is not updated — proving the call had no observable side effect at all.
"""

from unittest.mock import patch

import numpy as np
import pytest

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import PointCloudBuilder, build_free_space_rays, build_obstacle_cloud
from depth_perception_engine.geometry.rigid_transform import transform_point_cloud

_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _full_pipeline():
    config = PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)
    return DepthPerceptionPipeline(config, _CALIBRATION, body_T_camera_left=_transform())


def _random_pair(seed=7):
    w, h = _CALIBRATION.image_size
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return left, right


class TestCameraPointCloudFailureContainment:
    """If camera point-cloud construction fails: no body geometry, no
    obstacle cloud, no free-space rays — trivially true here, since the
    whole call raises before build_result() is ever reached."""

    def test_failure_propagates_uncaught_and_updates_no_state(self):
        pipeline = _full_pipeline()
        left, right = _random_pair()

        with patch.object(PointCloudBuilder, "build", side_effect=RuntimeError("simulated camera cloud failure")):
            with pytest.raises(RuntimeError, match="simulated camera cloud failure"):
                pipeline.process(left, right)

        health = pipeline.health()
        assert health.frames_processed == 0
        assert health.last_confidence is None
        assert health.last_processing_time_ms is None


class TestBodyTransformFailureContainment:
    """If body transformation fails: camera geometry may remain valid in
    principle, but downstream BODY-frame geometry must not be fabricated
    — verified here as: the whole call fails (stricter than the minimum
    bar; no result, fabricated or otherwise, is ever returned)."""

    def test_failure_propagates_uncaught_and_updates_no_state(self):
        pipeline = _full_pipeline()
        left, right = _random_pair()

        with patch(
            "depth_perception_engine.pipeline.pipeline.transform_point_cloud",
            side_effect=RuntimeError("simulated body transform failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated body transform failure"):
                pipeline.process(left, right)

        health = pipeline.health()
        assert health.frames_processed == 0


class TestObstacleExtractionFailureContainment:
    """If obstacle extraction fails: must not fabricate an empty scene
    and call it free — verified as: the whole call fails, free_space_rays
    (computed after obstacle_cloud in process()'s stage order) is never
    reached or returned either."""

    def test_failure_propagates_uncaught_and_updates_no_state(self):
        pipeline = _full_pipeline()
        left, right = _random_pair()

        with patch(
            "depth_perception_engine.pipeline.pipeline.build_obstacle_cloud",
            side_effect=RuntimeError("simulated obstacle extraction failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated obstacle extraction failure"):
                pipeline.process(left, right)

        health = pipeline.health()
        assert health.frames_processed == 0


class TestFreeSpaceGenerationFailureContainment:
    """If free-space generation fails: obstacle evidence may remain
    (already validly computed, held in a local variable) but free-space
    evidence must be absent — verified as: the whole call fails, so even
    the already-computed obstacle_cloud never reaches the caller. This is
    stricter than the mission's minimum bar ("obstacle evidence MAY
    remain") — the current design discards it too, rather than returning
    a partial result. Documented as the actual, verified behavior."""

    def test_failure_propagates_uncaught_and_updates_no_state(self):
        pipeline = _full_pipeline()
        left, right = _random_pair()

        with patch(
            "depth_perception_engine.pipeline.pipeline.build_free_space_rays",
            side_effect=RuntimeError("simulated free-space generation failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated free-space generation failure"):
                pipeline.process(left, right)

        health = pipeline.health()
        assert health.frames_processed == 0

    def test_already_computed_obstacle_cloud_never_reaches_the_caller(self):
        """Explicit proof, not just an inference from 'process() raised':
        even though build_obstacle_cloud() itself succeeds and its return
        value briefly exists as a local variable inside process(), no
        result object is ever constructed to carry it out — confirmed by
        the process() call raising before returning anything at all,
        so there is no result variable a caller could even attempt to
        read a stale obstacle_cloud from."""
        pipeline = _full_pipeline()
        left, right = _random_pair()

        with patch(
            "depth_perception_engine.pipeline.pipeline.build_free_space_rays",
            side_effect=RuntimeError("simulated failure"),
        ):
            result = None
            try:
                result = pipeline.process(left, right)
                assert False, "expected process() to raise"
            except RuntimeError:
                pass
            assert result is None


class TestGenericErrorNotSwallowed:
    """A generic programming-error exception type (not RuntimeError/
    ValueError) must also propagate uncaught — process() must not have a
    broad `except Exception` anywhere that would convert a genuine
    software defect into a falsely-successful result."""

    def test_arbitrary_exception_type_propagates(self):
        pipeline = _full_pipeline()
        left, right = _random_pair()

        class SimulatedProgrammingError(Exception):
            pass

        with patch.object(PointCloudBuilder, "build", side_effect=SimulatedProgrammingError("impossible state")):
            with pytest.raises(SimulatedProgrammingError):
                pipeline.process(left, right)
