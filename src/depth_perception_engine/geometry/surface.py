"""
Local surface geometry — Phase D4.

Deterministic, bounded, purely geometric local-plane estimation over an
existing organized PointCloud (in this pipeline's actual usage, the same
body-frame cloud already built for obstacle_cloud/free_space_rays/
geometry_metrics — see geometry.rigid_transform.transform_point_cloud). No
disparity/depth recomputation, no reprojection, no second camera->body
transform, no semantic classification, no traversability judgement: this
module answers "what surface geometry is observable here?" — never "can a
vehicle traverse/land/navigate here?" It must never be asked to produce, and
does not produce, anything named "road"/"ground"/"wall"/"landing surface"/
"traversable" — see docs/DPE_V1_PROVIDER_CONTRACT.md's D4 record.

Algorithm: the source cloud's pixel grid is partitioned into a small,
configurable grid_rows x grid_cols grid (the exact `np.linspace` boundary
convention traversability.SceneInterpreter already uses for its own region
grid — reimplemented here, not imported, to keep this module free of any
traversability/behavioral dependency). For each cell with at least
min_support_count valid points, a local plane is fit via PCA: the 3x3
covariance matrix of the cell's valid points is eigendecomposed
(`np.linalg.eigh`, which returns eigenvalues in ascending order for a real
symmetric input); the eigenvector of the SMALLEST eigenvalue is the surface
normal (the direction of least variance — the classical "fit the plane
that best explains this local point set" estimator), and
`1 - 3*lambda_min/sum(lambda)` (clipped to [0, 1]) is a planarity/goodness-
of-fit score: 1.0 for points lying exactly on a plane, trending toward 0.0
as the local neighborhood becomes more isotropic/non-planar. This is O(1)
work per cell (a 3x3 eigendecomposition) plus one O(cell size) pass to
build the covariance matrix — bounded by grid_rows*grid_cols (a small,
configured constant), not by image resolution, and cheap enough for later
Jetson execution: negligible next to the SGBM disparity stage it runs
downstream of.

Coordinate frame: every SurfaceEvidence carries `frame_id`, copied directly
from the source `cloud.frame_id` — this function has no built-in opinion
about BODY vs. CAMERA_OPTICAL_LEFT (matches geometry.build_obstacle_cloud's
own genericity). `centroid_m`/`normal` are expressed in that same frame.

Normalization convention: `normal` is a unit vector (`np.linalg.norm(normal)
== 1.0`, up to floating-point precision) whenever it is not None.

Orientation/sign convention: an eigenvector's sign is mathematically
ambiguous (n and -n are both valid solutions to the same eigenproblem).
This is resolved deterministically by orienting `normal` to point toward
`viewpoint` (in practice, the camera's own optical-center position in
`cloud.frame_id` — the same `origin` this pipeline already computes for
build_obstacle_cloud/build_free_space_rays): `normal` is flipped iff
`normal . (viewpoint - centroid) < 0`. This is the standard, purely
geometric "orient toward the sensor" convention used throughout point-cloud
processing (e.g. PCL's viewpoint-based normal orientation) — it encodes
where the sensor observed the surface FROM, nothing about "up," "outward,"
or any physical/behavioral surface classification. On the (rare, effectively
measure-zero) exact-tie case `normal . (viewpoint - centroid) == 0`, no flip
is applied — the result is still fully deterministic (the same input always
produces the same output, via `np.linalg.eigh`'s own deterministic
convention), just not viewpoint-oriented in that specific degenerate case.

Invalid/insufficient-support behavior: a cell with fewer than
min_support_count valid points gets `centroid_m=None, normal=None,
planarity=None` — explicitly unknown, never a fabricated/interpolated
plane. `support_count`/`support_fraction` are always populated (0/0.0 is a
legitimate, informative value, not "no data"). A cell that clears
min_support_count but whose points are numerically degenerate (total
eigenvalue variance ~0 — every valid point coincides) also reports
`normal=None, planarity=None` (no well-defined plane exists), while
`centroid_m` — the mean position, always well-defined regardless of
degeneracy — is still populated.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from depth_perception_engine.geometry.types import PointCloud

_MIN_POINTS_FOR_PLANE = 3
_EIGENVALUE_DEGENERACY_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class SurfaceEvidence:
    """Local surface geometry for one cell of the surface grid — Phase D4.

    See this module's own docstring for the full frame/normalization/
    orientation/invalid-support contract. Every field is either spatial
    identity (`row`/`col`/`x1`/`y1`/`x2`/`y2`, the pixel bounds in the
    source cloud's own 2D grid this cell was decimated from — mirrors
    RegionEvidence's identical fields), coverage (`support_count`/
    `support_fraction`), or genuine geometric evidence
    (`centroid_m`/`normal`/`planarity`) — never a behavioral/navigation
    judgement.
    """

    frame_id: str
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int

    support_count: int
    support_fraction: float

    centroid_m: Optional[np.ndarray]
    normal: Optional[np.ndarray]
    planarity: Optional[float]


def build_surface_evidence(
    cloud: PointCloud,
    viewpoint: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    min_support_count: int,
    reliability_mask: Optional[np.ndarray] = None,
) -> List[SurfaceEvidence]:
    """Fit a local plane per grid cell over an existing organized PointCloud.

    Args:
        cloud: Source PointCloud — any frame; see module docstring. In
            this pipeline's actual usage, always the E4 body-frame cloud
            (`geometry_body`).
        viewpoint: (3,) float, expressed in the same frame as
            `cloud.frame_id` — the point each cell's normal is
            sign-oriented toward. In this pipeline's actual usage, the
            same `body_T_camera_left.translation` origin already used by
            build_obstacle_cloud/build_free_space_rays.
        grid_rows, grid_cols: surface grid dimensions, both >= 1.
        min_support_count: minimum valid points a cell must contain for a
            plane to be fit; must be >= 3 (fewer points cannot uniquely
            determine a plane).
        reliability_mask: Optional (H, W) bool, same organized grid as
            `cloud` — Phase I3, see geometry.reliability's module
            docstring. Where True, a point is excluded from this cell's
            plane fit — a confirmed occlusion-shadow point (a
            geometrically-fabricated extension of an adjacent real
            surface) must not contribute to that surface's own reported
            normal/planarity. None (default) preserves this function's
            exact pre-I3 behavior.

    Returns:
        A flat list of grid_rows * grid_cols SurfaceEvidence, one per
        cell, in row-major order.

    Raises:
        ValueError: If grid_rows/grid_cols < 1, or min_support_count < 3.
    """
    if grid_rows < 1 or grid_cols < 1:
        raise ValueError(
            f"grid_rows/grid_cols must both be >= 1, got rows={grid_rows}, cols={grid_cols}."
        )
    if min_support_count < _MIN_POINTS_FOR_PLANE:
        raise ValueError(
            f"min_support_count ({min_support_count}) must be >= {_MIN_POINTS_FOR_PLANE} — "
            "fewer points cannot uniquely determine a plane."
        )

    h, w = cloud.valid_mask.shape[:2]
    row_bounds = np.linspace(0, h, grid_rows + 1).astype(int)
    col_bounds = np.linspace(0, w, grid_cols + 1).astype(int)

    effective_valid_mask = cloud.valid_mask
    if reliability_mask is not None:
        effective_valid_mask = effective_valid_mask & ~reliability_mask

    evidence: List[SurfaceEvidence] = []
    for r in range(grid_rows):
        y1, y2 = int(row_bounds[r]), int(row_bounds[r + 1])
        for c in range(grid_cols):
            x1, x2 = int(col_bounds[c]), int(col_bounds[c + 1])

            cell_points = cloud.points[y1:y2, x1:x2]
            cell_valid = effective_valid_mask[y1:y2, x1:x2]
            total_pixels = int(cell_valid.size)

            valid_points = cell_points[cell_valid].astype(np.float64)
            support_count = int(valid_points.shape[0])
            support_fraction = (support_count / total_pixels) if total_pixels else 0.0

            centroid_m = None
            normal = None
            planarity = None

            if support_count >= min_support_count:
                centroid = valid_points.mean(axis=0)
                centroid_m = centroid.astype(np.float32)

                centered = valid_points - centroid
                cov = (centered.T @ centered) / support_count
                # Ascending eigenvalue order — guaranteed by eigh for a
                # real symmetric input. eigenvectors[:, 0] is the
                # smallest-eigenvalue (least-variance) direction: the
                # surface normal.
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                eigenvalue_sum = float(eigenvalues.sum())

                if eigenvalue_sum > _EIGENVALUE_DEGENERACY_EPS:
                    candidate_normal = eigenvectors[:, 0]
                    norm = float(np.linalg.norm(candidate_normal))
                    if norm > _EIGENVALUE_DEGENERACY_EPS:
                        candidate_normal = candidate_normal / norm

                        view_vector = viewpoint - centroid
                        if np.dot(candidate_normal, view_vector) < 0.0:
                            candidate_normal = -candidate_normal

                        normal = candidate_normal.astype(np.float32)
                        planarity = float(
                            np.clip(1.0 - 3.0 * eigenvalues[0] / eigenvalue_sum, 0.0, 1.0)
                        )

            evidence.append(SurfaceEvidence(
                frame_id=cloud.frame_id,
                row=r, col=c, x1=x1, y1=y1, x2=x2, y2=y2,
                support_count=support_count, support_fraction=support_fraction,
                centroid_m=centroid_m, normal=normal, planarity=planarity,
            ))

    return evidence
