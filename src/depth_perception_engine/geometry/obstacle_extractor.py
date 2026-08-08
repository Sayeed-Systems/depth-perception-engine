"""
Obstacle-cloud extraction — Level 3, Phase E5.

Purely geometric: an "obstacle" here means valid, observed 3D surface
geometry within a configured range window — nothing more. No semantic
classification (floor/wall/person/object), no clustering, no
traversability judgement. That reuse-not-reinvent boundary matches this
codebase's established discipline (Level 0-2's RegionAnalyzer/
ThreatAssessor already own scene classification; this module does not
duplicate or second-guess it — see docs/E2_IMPLEMENTATION_PLAN.md's
original ObstacleExtractor sketch, which set this same boundary before
any of this was implemented).

Consumes only an existing geometry.PointCloud (in practice, the E4
body-frame cloud) — no disparity/depth recomputation, no reprojection, no
independent camera->body transform of its own. See docs/LEVEL3_ARCHITECTURE.md's
E5 update for the full single-chain rationale.
"""

import numpy as np

from depth_perception_engine.geometry.types import ObstacleCloud, PointCloud


def build_obstacle_cloud(
    cloud: PointCloud,
    origin: np.ndarray,
    min_range_m: float,
    max_range_m: float,
    stride: int = 1,
) -> ObstacleCloud:
    """Filter a PointCloud down to a compact, unorganized ObstacleCloud.

    A point is included if and only if: it is valid (`cloud.valid_mask`,
    which already excludes NaN/invalid-disparity/out-of-range-depth
    pixels — see geometry.PointCloudBuilder), its Euclidean distance from
    `origin` falls within `[min_range_m, max_range_m]`, and its pixel
    position survives the `stride` grid decimation described below. No
    invalid point can ever appear in the output — there is no code path
    that adds a point without first checking `valid_mask`.

    `stride`: a 2D grid decimation applied to the *source pixel grid*
    before any filtering — equivalent to `cloud.points[::stride, ::stride]`
    — not a "keep every Nth *valid* point" count-based sample. This is
    deterministic (depends only on pixel position, never on how many
    valid points preceded a given one) and matches the common
    organized-point-cloud downsampling convention.

    Fully vectorized: distance and range-membership are computed once
    over the whole (decimated) grid via `np.linalg.norm`, no Python loop
    over pixels or points.

    Args:
        cloud: Source PointCloud — any frame; this function has no
            special knowledge of BODY vs. CAMERA_OPTICAL_LEFT. In this
            pipeline's actual usage it is always the E4 body-frame cloud.
        origin: (3,) float, the point distances are measured from,
            expressed in the *same* frame as `cloud.frame_id` (the
            pipeline passes `body_T_camera_left.translation` — the
            camera's own position in body coordinates — not (0, 0, 0),
            since the body origin and the camera origin are not
            generally the same point). Not validated against `cloud`'s
            actual frame_id beyond shape, matching
            geometry.transform_point_cloud's equivalent trust boundary.
        min_range_m: Inclusive lower bound on distance from `origin`.
        max_range_m: Inclusive upper bound on distance from `origin`.
        stride: Grid decimation factor, >= 1. 1 (default) keeps every
            pixel — no downsampling.

    Returns:
        ObstacleCloud in `cloud.frame_id`: `points` (N, 3) float32,
        `distances_m` (N,) float32 (computed here, not left for the
        caller to re-derive), `confidence` (N,) float32 if `cloud`
        carries one, else None. No `timestamp` field — ObstacleCloud's
        frozen contract (geometry/types.py) does not define one; the
        frame's timestamp remains available via the PointCloud this was
        built from, or DepthPerceptionResult.timestamp one level up.

    Raises:
        ValueError: If `stride < 1`, or `min_range_m > max_range_m`.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    if min_range_m > max_range_m:
        raise ValueError(
            f"min_range_m ({min_range_m}) must be <= max_range_m ({max_range_m})."
        )

    points = cloud.points[::stride, ::stride]
    valid = cloud.valid_mask[::stride, ::stride]

    # NaN-safe: invalid pixels are NaN, so distances there are NaN too;
    # NaN comparisons are always False, so they never satisfy the range
    # bounds below regardless of min/max_range_m's values — but `valid`
    # is still ANDed in explicitly rather than relied upon implicitly,
    # so this behavior does not depend on that NaN-comparison subtlety.
    offsets = points - origin
    distances = np.linalg.norm(offsets, axis=-1)

    # np.isfinite(distances) is NOT redundant with the range comparisons
    # below when max_range_m is +inf (PipelineConfig's own default —
    # "unbounded" is expressed as +inf, not as skipping the check): a
    # contract-violating Inf point would otherwise satisfy `inf <= inf`
    # and be silently admitted. Explicit, not incidental — matches this
    # module's "invalid points must never become obstacles" mandate
    # regardless of what min/max_range_m happen to be configured to.
    in_range = valid & np.isfinite(distances) & (distances >= min_range_m) & (distances <= max_range_m)

    selected_points = points[in_range].astype(np.float32)
    selected_distances = distances[in_range].astype(np.float32)

    selected_confidence = None
    if cloud.confidence is not None:
        selected_confidence = cloud.confidence[::stride, ::stride][in_range].astype(np.float32)

    return ObstacleCloud(
        points=selected_points,
        frame_id=cloud.frame_id,
        distances_m=selected_distances,
        confidence=selected_confidence,
    )
