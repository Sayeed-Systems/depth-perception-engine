"""Unit tests for frames.FrameId / RigidTransform."""

import numpy as np
import pytest

from depth_perception_engine.frames import FrameId, RigidTransform


class TestFrameId:
    def test_camera_optical_left_value(self):
        assert FrameId.CAMERA_OPTICAL_LEFT == "camera_optical_left"

    def test_body_value(self):
        assert FrameId.BODY == "body"

    def test_frame_ids_are_plain_strings_not_an_enum(self):
        # Deliberately not a closed Enum — any string is a valid frame id.
        assert isinstance(FrameId.CAMERA_OPTICAL_LEFT, str)
        assert isinstance(FrameId.BODY, str)


class TestRigidTransform:
    def test_valid_construction(self):
        rt = RigidTransform(
            rotation=np.eye(3),
            translation=np.zeros(3),
            from_frame=FrameId.CAMERA_OPTICAL_LEFT,
            to_frame=FrameId.BODY,
        )
        assert rt.from_frame == FrameId.CAMERA_OPTICAL_LEFT
        assert rt.to_frame == FrameId.BODY

    def test_wrong_rotation_shape_raises(self):
        with pytest.raises(ValueError):
            RigidTransform(
                rotation=np.eye(4),
                translation=np.zeros(3),
                from_frame="a", to_frame="b",
            )

    def test_non_ndarray_rotation_raises(self):
        with pytest.raises(ValueError):
            RigidTransform(
                rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                translation=np.zeros(3),
                from_frame="a", to_frame="b",
            )

    def test_wrong_translation_shape_raises(self):
        with pytest.raises(ValueError):
            RigidTransform(
                rotation=np.eye(3),
                translation=np.zeros(2),
                from_frame="a", to_frame="b",
            )

    def test_empty_from_frame_raises(self):
        with pytest.raises(ValueError):
            RigidTransform(
                rotation=np.eye(3), translation=np.zeros(3),
                from_frame="", to_frame="b",
            )

    def test_empty_to_frame_raises(self):
        with pytest.raises(ValueError):
            RigidTransform(
                rotation=np.eye(3), translation=np.zeros(3),
                from_frame="a", to_frame="",
            )

    def test_is_frozen(self):
        rt = RigidTransform(
            rotation=np.eye(3), translation=np.zeros(3),
            from_frame="a", to_frame="b",
        )
        with pytest.raises(AttributeError):
            rt.from_frame = "c"
