"""
Coordinate frame identity and rigid transforms.

Level 3 (Phase E1) contract-freezing module — see docs/COORDINATE_FRAMES.md
for the full specification this codifies. Nothing in this module computes
anything; it exists so a frame can be *named* and a transform *represented*,
ahead of any geometry code actually producing or consuming one.

Does NOT perform coordinate transformation, point cloud generation, or any
other Level 3+ computation — see docs/E2_IMPLEMENTATION_PLAN.md for where
that will eventually live.
"""

from dataclasses import dataclass

import numpy as np


class FrameId:
    """String constants for coordinate frame names.

    Deliberately plain strings, not a closed Enum — a geometry result needs
    to be able to name a frame (e.g. a second/rear camera, or a caller's own
    downstream frame) without this library knowing every frame that could
    ever exist. These two are the only frames this repository itself
    currently has any opinion about; see docs/COORDINATE_FRAMES.md.
    """

    CAMERA_OPTICAL_LEFT = "camera_optical_left"
    """Origin at the rectified left camera's optical center. Axes: X right,
    Y down, Z forward (into the scene) — the OpenCV/computer-vision
    convention this library's Q-matrix-based depth math already assumes
    today (right-handed: X x Y = Z). This is the frame depth_map,
    disparity_map, and (if ever computed) a point cloud already implicitly
    live in — this constant only gives that existing, already-in-use frame
    a name."""

    BODY = "body"
    """Recommended vehicle/robot body frame: X forward, Y left, Z up,
    right-handed (matches ROS REP-103). Not used anywhere in this
    repository today — no rigid transform into this frame exists yet. Named
    here so a future body-frame transform has an agreed-on target frame ID
    to reference, rather than inventing one ad hoc later."""


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """A rotation + translation from one named frame to another.

    Pure data — no apply-to-points method exists yet (that belongs to
    Level 3's actual geometry implementation, out of scope for this
    contract-freezing pass). rotation/translation are validated for shape
    only; orthonormality of `rotation` is the caller's responsibility, the
    same validation depth this codebase's other matrix-carrying contracts
    (e.g. calibration.StereoCalibration) already use.

    Attributes:
        rotation: (3, 3) rotation matrix.
        translation: (3,) translation vector, metres.
        from_frame: FrameId (or any caller-defined string) this transform
            maps points *from*.
        to_frame: FrameId (or any caller-defined string) this transform
            maps points *to*. Convention: a point p in `from_frame`
            transforms to `to_frame` as `rotation @ p + translation`.
    """

    rotation: np.ndarray
    translation: np.ndarray
    from_frame: str
    to_frame: str

    def __post_init__(self) -> None:
        if not isinstance(self.rotation, np.ndarray) or self.rotation.shape != (3, 3):
            raise ValueError(
                f"RigidTransform.rotation must be a (3, 3) ndarray, "
                f"got {getattr(self.rotation, 'shape', type(self.rotation).__name__)}."
            )
        if not isinstance(self.translation, np.ndarray) or self.translation.shape != (3,):
            raise ValueError(
                f"RigidTransform.translation must be a (3,) ndarray, "
                f"got {getattr(self.translation, 'shape', type(self.translation).__name__)}."
            )
        if not self.from_frame:
            raise ValueError("RigidTransform.from_frame must be a non-empty string.")
        if not self.to_frame:
            raise ValueError("RigidTransform.to_frame must be a non-empty string.")
