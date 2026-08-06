"""
Level 3 geometry result contracts — Level 3 (Phase E1).

Interfaces only. Nothing in this module computes a point cloud, extracts
an obstacle, casts a free-space ray, or estimates a metric — every type
here is pure data, with zero producer anywhere in this repository today.
See docs/E2_IMPLEMENTATION_PLAN.md for what will actually build these.

Not exported from the top-level depth_perception_engine package — reachable
only via `from depth_perception_engine.geometry import ...`, deliberately,
since none of this is real yet; promoting it to the top-level public API
would overclaim capability this repository does not have (see the
2026-08-05 perception capability audit, docs/VALIDATION_REPORT.md).

Ownership convention (all four types): a future producer allocates and
returns a fresh set of arrays on every call. A caller must not assume it
can safely mutate a previous call's arrays in place without checking for
aliasing — the same convention DepthPerceptionResult's arrays already use,
stated explicitly here since these are new types with no prior norm to
inherit.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class PointCloud:
    """An organized (pixel-aligned) 3D point cloud.

    Attributes:
        points: (H, W, 3) float32, metres, in `frame_id`. Invalid pixels —
            same invalid convention as depth_map (see
            docs/DATA_CONTRACTS.md) — are NaN, not zero, since (0, 0, 0) is
            a legitimate point on the optical axis and must not be
            confused with "no data".
        frame_id: coordinate frame the points are expressed in — see
            frames.FrameId. Expected to be FrameId.CAMERA_OPTICAL_LEFT for
            anything this pipeline itself would ever produce, but not
            hardcoded to that value here — a caller-supplied transform
            could produce a PointCloud in a different frame.
        valid_mask: (H, W) bool, True where `points` is not NaN — mirrors
            DepthPerceptionResult.valid_depth_mask's role, provided
            directly here rather than requiring a caller to re-derive it
            via `~np.isnan(points).any(axis=-1)`.
        confidence: (H, W) float32 in [0, 1], optional — per-pixel
            confidence, if the producer has one to offer (e.g. reusing
            RegionAnalyzer-style signals at pixel granularity instead of
            per-region).
        timestamp: opaque, caller-defined — same pass-through convention
            as DepthPerceptionResult.timestamp.
    """

    points: np.ndarray
    frame_id: str
    valid_mask: np.ndarray
    confidence: Optional[np.ndarray] = None
    timestamp: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ObstacleCloud:
    """A filtered, unorganized set of points classified as obstacles.

    Attributes:
        points: (N, 3) float32, metres, in `frame_id`. Unorganized — no
            relationship to source pixel position is preserved; N varies
            call to call.
        frame_id: see PointCloud.frame_id.
        distances_m: (N,) float32, optional — Euclidean distance from the
            camera origin to each point, if the producer computes it
            directly rather than requiring the caller to derive it from
            `points`.
        confidence: (N,) float32 in [0, 1], optional — per-point confidence.
    """

    points: np.ndarray
    frame_id: str
    distances_m: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class FreeSpaceRays:
    """A set of rays from the camera origin to the nearest obstacle (or max range) along each.

    Attributes:
        origins: (N, 3) float32, metres, in `frame_id` — typically all
            identical (the camera's own optical center) for a single-camera
            producer, but not constrained to be, so a future multi-camera
            fusion producer can return rays from more than one origin.
        directions: (N, 3) float32, unit vectors, in `frame_id`.
        ranges_m: (N,) float32 — distance along each ray before an obstacle
            is reached, or the sensor's max reliable range if none is
            found within it (mirroring DepthEstimator.MAX_DEPTH_M's
            existing clamp-not-null convention, rather than returning
            infinity).
        frame_id: see PointCloud.frame_id.
    """

    origins: np.ndarray
    directions: np.ndarray
    ranges_m: np.ndarray
    frame_id: str


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    """Scalar summary statistics over a PointCloud/ObstacleCloud/FreeSpaceRays result.

    Attributes:
        min_obstacle_distance_m: closest obstacle point's distance, or None
            if no obstacle points exist for this frame.
        mean_free_space_m: mean range across FreeSpaceRays, or None if no
            rays were cast.
        point_count: total point count the metrics were computed over.
        valid_fraction: fraction (0..1) of the source point cloud/rays that
            were valid, not fraction "occupied" or "free" — a data-quality
            metric, not a scene-content metric.
    """

    min_obstacle_distance_m: Optional[float]
    mean_free_space_m: Optional[float]
    point_count: int
    valid_fraction: float
