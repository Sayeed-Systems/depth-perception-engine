"""
Free-space ray extraction — Level 3, Phase E5.

Core safety rule (see docs/DATA_CONTRACTS.md's Geometry section and this
module's own tests): a valid stereo depth pixel means a ray from the
sensor origin through observed space to a measured surface point. The
space strictly between origin and that endpoint is observed free; the
endpoint itself is surface evidence, not free space; an invalid pixel
carries no such evidence at all and generates NO ray — it is neither
free nor occupied, it is UNKNOWN. This module never fabricates a ray for
an invalid pixel, and never treats a ray's own endpoint as free (the
frozen FreeSpaceRays contract's `[origin, origin + direction*range)`
half-open interval already encodes this: the point *at* `range` is the
surface, not included in "free").

Stores the frozen (origins, directions, ranges_m) parametrization exactly
as geometry.types.FreeSpaceRays defines it — not an (origin, endpoint)
pair representation; the two are mathematically equivalent
(endpoint = origin + direction * range) and this module's own tests
verify that equivalence directly.

No voxelization, no occupancy integration — this module produces
observations only; see docs/E6_IMPLEMENTATION_PLAN.md for what a future
occupancy consumer of these rays would look like, not built here.

Consumes only an existing geometry.PointCloud (in practice, the E4
body-frame cloud) — see obstacle_extractor.py's module docstring for the
identical "no recomputation, no parallel geometry chain" rationale, which
applies here unchanged.
"""

from typing import Optional

import numpy as np

from depth_perception_engine.geometry.types import FreeSpaceRays, PointCloud


def build_free_space_rays(
    cloud: PointCloud,
    origin: np.ndarray,
    stride: int = 1,
    reliability_mask: Optional[np.ndarray] = None,
) -> FreeSpaceRays:
    """Build FreeSpaceRays from a PointCloud's valid points.

    For every valid pixel (after `stride` decimation), one ray is
    produced: `origin` (the sensor origin) to the pixel's measured 3D
    point, stored as a unit `direction` and a `range_m` scalar such that
    `origin + direction * range_m` recovers the endpoint exactly. Invalid
    pixels contribute no ray at all — the output arrays are shorter than
    the input pixel count whenever any pixel is invalid; there is no
    placeholder/sentinel ray standing in for "no data".

    A degenerate case is defensively excluded: a point exactly coincident
    with `origin` (range == 0) cannot form a direction (0/0) and is
    dropped rather than emitting a NaN direction. This is not expected to
    occur for real data — every valid point's distance from the camera's
    own optical center is bounded below by
    depth.DepthEstimator.MIN_DEPTH_M (0.15 m) before any body-frame
    transform, and a rigid transform preserves distances — but is handled
    explicitly rather than left to produce a silent NaN.

    Fully vectorized: directions/ranges are computed once over the whole
    (decimated) grid via `np.linalg.norm`/broadcasting division, no
    Python loop over pixels.

    Args:
        cloud: Source PointCloud — frame-agnostic, see
            obstacle_extractor.build_obstacle_cloud's equivalent note. In
            this pipeline's actual usage, always the E4 body-frame cloud.
        origin: (3,) float, the common ray origin, expressed in the same
            frame as `cloud.frame_id` (the pipeline passes
            `body_T_camera_left.translation`).
        stride: Grid decimation factor, >= 1 — see
            obstacle_extractor.build_obstacle_cloud's identical
            `stride` semantics (2D grid decimation, not a count-based
            sample of the valid subset).
        reliability_mask: Optional (H, W) bool — see
            obstacle_extractor.build_obstacle_cloud's identical
            parameter. Applying the SAME mask to both builders is
            deliberate: a pixel excluded here must also be excluded from
            build_obstacle_cloud, and vice versa — the point stays
            UNKNOWN (neither confirmed obstacle nor confirmed free
            space), never asymmetrically "not an obstacle, so implicitly
            free." None (default) preserves this function's exact
            pre-I3 behavior.

    Returns:
        FreeSpaceRays in `cloud.frame_id`: `origins` (N, 3) float32 (every
        row identical to `origin` — a single-camera producer, per the
        type's own docstring), `directions` (N, 3) float32 unit vectors,
        `ranges_m` (N,) float32. No `timestamp`/`confidence` field —
        FreeSpaceRays' frozen contract does not define either.

    Raises:
        ValueError: If `stride < 1`.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    points = cloud.points[::stride, ::stride]
    valid = cloud.valid_mask[::stride, ::stride]
    if reliability_mask is not None:
        valid = valid & ~reliability_mask[::stride, ::stride]

    offsets = points - origin
    ranges = np.linalg.norm(offsets, axis=-1)

    # np.isfinite(ranges) guards a contract-violating Inf/NaN point marked
    # valid from producing a NaN direction (Inf/Inf) — see
    # obstacle_extractor.build_obstacle_cloud's identical guard and its
    # comment for why this is not redundant, just less obviously so here
    # since there is no configurable upper bound to coincide with +inf.
    selectable = valid & np.isfinite(ranges) & (ranges > 0.0)  # also excludes the origin-coincident case

    selected_ranges = ranges[selectable].astype(np.float32)
    selected_offsets = offsets[selectable]
    selected_directions = (selected_offsets / selected_ranges[:, np.newaxis]).astype(np.float32)

    n = selected_ranges.shape[0]
    origins = np.broadcast_to(origin.astype(np.float32), (n, 3)).copy()

    return FreeSpaceRays(
        origins=origins,
        directions=selected_directions,
        ranges_m=selected_ranges,
        frame_id=cloud.frame_id,
    )
