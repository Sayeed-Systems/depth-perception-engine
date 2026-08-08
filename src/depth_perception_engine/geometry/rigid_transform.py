"""
Rigid point-cloud transformation — Level 3, Phase E4.

The one canonical implementation of frames.RigidTransform's own documented
convention ("a point p in from_frame transforms to to_frame as
rotation @ p + translation" — see frames.py and docs/COORDINATE_FRAMES.md)
applied to an organized geometry.PointCloud. frames.RigidTransform's own
docstring explicitly deferred this: "Pure data — no apply-to-points method
exists yet (that belongs to Level 3's actual geometry implementation...)."
This module is that implementation.

Deliberately generic, not body-frame-specific: this function transforms a
PointCloud from whatever frame it is currently in to whatever frame the
given RigidTransform targets — it has no special knowledge of
FrameId.CAMERA_OPTICAL_LEFT or FrameId.BODY. DepthPerceptionPipeline (Phase
E4 pipeline integration) is the one place that happens to call it with a
camera-to-body transform; the math itself makes no assumption about which
two frames are involved, or where a stereo rig is physically mounted.
"""

import numpy as np

from depth_perception_engine.frames import RigidTransform
from depth_perception_engine.geometry.types import PointCloud


def transform_point_cloud(cloud: PointCloud, transform: RigidTransform) -> PointCloud:
    """Apply a rigid transform to every point in an organized PointCloud.

    Fully vectorized (one reshape + one matrix multiply over the whole
    (H, W, 3) array) — no Python loop per point. Deterministic: the same
    (cloud, transform) always produces bit-identical output.

    NaN handling requires no special-casing: IEEE-754 arithmetic already
    guarantees `rotation @ NaN + translation` is NaN in every output
    component (a zero entry in `rotation` times a NaN coordinate is still
    NaN, not 0 — this is standard floating-point behavior, not an edge
    case this function works around). So invalid (NaN) input points are
    invalid (NaN) output points automatically, with no additional masking
    logic — verified explicitly in
    tests/test_rigid_transform.py::TestInvalidPointsStayInvalid regardless.
    `valid_mask` itself is not recomputed from the transformed points (it
    would be redundant, and the task's own contract requires it be
    preserved exactly, not merely end up equal) — it is copied unchanged
    from `cloud`, since validity is a frame-independent property of a
    point (whether real stereo evidence exists for it), not a property of
    which frame it happens to be expressed in.

    Args:
        cloud: The source PointCloud. Not mutated by this call — a fresh
            PointCloud is always returned (see geometry/types.py's module
            docstring on ownership).
        transform: A RigidTransform whose `from_frame` must equal
            `cloud.frame_id` — see Raises below. Its `rotation`/
            `translation` must be finite; orthonormality of `rotation` is
            not checked here either (frames.RigidTransform's own docstring
            states this is the caller's/calibration-provider's
            responsibility, not this function's).

    Returns:
        A new PointCloud: `points` transformed (float32, same (H, W, 3)
        shape), `frame_id` set to `transform.to_frame`, `valid_mask`
        preserved exactly (a copy, not the same array object),
        `confidence` preserved exactly if present (frame-independent, same
        reasoning as valid_mask), `timestamp` preserved exactly.

    Raises:
        TypeError: If `cloud` is not a PointCloud or `transform` is not a
            RigidTransform.
        ValueError: If `transform.from_frame != cloud.frame_id` (this
            transform does not apply to this cloud's frame), or if
            `transform.rotation`/`transform.translation` contain a
            non-finite (NaN/Inf) entry — an invalid extrinsic calibration
            must never silently produce a transformed cloud.
    """
    if not isinstance(cloud, PointCloud):
        raise TypeError(f"cloud must be a geometry.PointCloud, got {type(cloud).__name__}.")
    if not isinstance(transform, RigidTransform):
        raise TypeError(f"transform must be a frames.RigidTransform, got {type(transform).__name__}.")
    if transform.from_frame != cloud.frame_id:
        raise ValueError(
            f"transform.from_frame ({transform.from_frame!r}) does not match "
            f"cloud.frame_id ({cloud.frame_id!r}) — this transform does not apply to this cloud."
        )
    if not np.all(np.isfinite(transform.rotation)) or not np.all(np.isfinite(transform.translation)):
        raise ValueError(
            "transform.rotation/translation contain a non-finite (NaN/Inf) entry — "
            "refusing to transform a cloud with an invalid extrinsic calibration."
        )

    height, width, _ = cloud.points.shape
    flat_points = cloud.points.reshape(-1, 3).astype(np.float64)

    # For row-vectors p (shape (N, 3)), p @ R.T is the vectorized form of
    # "R @ p for every row p" — standard identity, avoids any per-point
    # Python loop or per-point matrix-vector call.
    transformed_flat = flat_points @ transform.rotation.T + transform.translation
    transformed_points = transformed_flat.reshape(height, width, 3).astype(np.float32)

    return PointCloud(
        points=transformed_points,
        frame_id=transform.to_frame,
        valid_mask=cloud.valid_mask.copy(),
        confidence=None if cloud.confidence is None else cloud.confidence.copy(),
        timestamp=cloud.timestamp,
    )
