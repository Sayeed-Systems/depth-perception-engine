"""
Level 3 geometry (Phase E1 interfaces + Phase E2's producer + Phase E4's
frame transform + Phase E5's obstacle/free-space/metrics producers + Phase
E6's quality classification).

Result types (PointCloud, ObstacleCloud, FreeSpaceRays, GeometryMetrics)
are defined in types.py. PointCloudBuilder (Phase E2) builds a PointCloud
from a disparity map. transform_point_cloud (Phase E4) rigidly transforms
an existing PointCloud from one named frame to another (e.g. camera optical
-> body), per frames.RigidTransform's own documented convention.
build_obstacle_cloud/build_free_space_rays/build_geometry_metrics (Phase
E5) derive structured spatial evidence from an existing PointCloud.
classify_geometry_quality/GeometryQuality (Phase E6) map a GeometryMetrics'
valid_fraction onto a HEALTHY/DEGRADED/NO_USABLE_GEOMETRY label — opt-in,
not wired into DepthPerceptionPipeline.process().
GeometryFrame (Phase D2, provider.py) is the final, authoritative DPE V1
provider contract — see its own module docstring and
docs/DPE_V1_PROVIDER_CONTRACT.md. RegionEvidence/ClearanceEvidence (Phase
D3, provider.py) are neutral geometric evidence extracted from the
traversability/obstacle layers, deliberately without their behavioral
labels — see provider.py's own module docstring.

As of Phase D3, PointCloud/ObstacleCloud/FreeSpaceRays/GeometryMetrics are
ALSO re-exported from the top-level depth_perception_engine package
(Tier 1) — the architect's D3 decision: every DPE-owned type appearing in
GeometryFrame's authoritative type graph must be publicly supported. Every
builder/algorithm function below (PointCloudBuilder, transform_point_cloud,
build_obstacle_cloud, build_free_space_rays, build_geometry_metrics,
GeometryQuality, classify_geometry_quality) remains deliberately Tier 3 —
promoting a result *type* does not mean promoting the *producer* that
builds it; see docs/DPE_V1_PROVIDER_CONTRACT.md's D3 record.

SurfaceEvidence/build_surface_evidence (Phase D4, surface.py) is the first
genuinely NEW geometric algorithm this pass adds: deterministic, bounded
local-plane (surface normal) estimation over an existing organized
PointCloud — see surface.py's own module docstring for the full frame/
normalization/orientation/invalid-support contract. SurfaceEvidence (the
result type) is Tier 1; build_surface_evidence (the algorithm) stays Tier
3, same "promote the type, not the producer" rule as Phase D3.

BoundaryEvidence/build_boundary_evidence (Phase D5, boundary.py) detects
depth and/or surface-orientation discontinuities between adjacent cells
of their own independent grid — reusing depth_map directly (Level 0,
always present) and SurfaceEvidence's normals opportunistically (only
when its grid happens to match) — see boundary.py's own module docstring
for the full discontinuity-definition/invalid-vs-boundary/coordinate
contract. BoundaryEvidence is Tier 1; build_boundary_evidence stays Tier
3, same rule.

OpeningEvidence/build_opening_evidence (Phase D6, opening.py) detects
geometrically supported gaps between confirmed flanking structures —
built from already-computed BoundaryEvidence (confirms real structural
transitions) plus depth_map (the one signal BoundaryEvidence's own
unsigned depth_step_m cannot give: which side is farther) — on the SAME
grid boundary_evidence was built with, never a redesigned/independent
one. See opening.py's own module docstring for the full admission-rule/
invalid-vs-opening/coordinate contract. OpeningEvidence is Tier 1;
build_opening_evidence stays Tier 3, same rule.

Phase D7 refined ClearanceEvidence (provider.py, unchanged location —
still a pure extraction type, now with a small trigonometric addition,
not enough to warrant its own algorithm file): added geometric
coverage/support (valid_count/total_pixels/coverage_fraction/
support_state — the last using the new ClearanceSupportState
constant class, also Tier 1) and calibrated horizontal bearing
(bearing_center_rad/bearing_min_rad/bearing_max_rad, derived from
already-available focal_length_px/principal_point_x_px via the standard
pinhole formula). See provider.py's own ClearanceEvidence/
ClearanceSupportState docstrings for the full contract.

Phase D8 (provider.py) added GeometryFrameQuality/
GeometryFrameQualityState — a structured, multi-dimension quality/
uncertainty rollup over GeometryFrame's own four frame-level
state-bearing evidence sources (geometry_metrics via the EXISTING
classify_geometry_quality() below, temporal_consistency,
motion_aware_reliability, temporal_persistence), deliberately NOT a
blended numeric score. NOTE the deliberately DISTINCT name from this
subpackage's own pre-existing GeometryQuality (a narrower, single-signal
HEALTHY/DEGRADED/NO_USABLE_GEOMETRY classification of one
GeometryMetrics.valid_fraction value, unchanged since Phase E6) — the two
are different types serving different purposes; GeometryFrameQuality's
own geometry_validity_state dimension is DERIVED FROM GeometryQuality via
classify_geometry_quality(), not a replacement for it. Also added
RegionEvidence.frame_id (the one other D3-era evidence type that still
lacked one). See provider.py's own GeometryFrameQuality docstring for the
full per-dimension derivation and docs/DPE_V1_PROVIDER_CONTRACT.md's D8
record.
"""

from depth_perception_engine.geometry.boundary import (
    BoundaryDirection,
    BoundaryEvidence,
    BoundaryState,
    build_boundary_evidence,
)
from depth_perception_engine.geometry.free_space import build_free_space_rays
from depth_perception_engine.geometry.geometry_metrics import (
    GeometryQuality,
    build_geometry_metrics,
    classify_geometry_quality,
)
from depth_perception_engine.geometry.obstacle_extractor import build_obstacle_cloud
from depth_perception_engine.geometry.opening import OpeningEvidence, build_opening_evidence
from depth_perception_engine.geometry.point_cloud_builder import PointCloudBuilder
from depth_perception_engine.geometry.provider import (
    ClearanceEvidence,
    ClearanceSupportState,
    GeometryFrame,
    GeometryFrameQuality,
    GeometryFrameQualityState,
    RegionEvidence,
)
from depth_perception_engine.geometry.rigid_transform import transform_point_cloud
from depth_perception_engine.geometry.surface import SurfaceEvidence, build_surface_evidence
from depth_perception_engine.geometry.types import (
    FreeSpaceRays,
    GeometryMetrics,
    ObstacleCloud,
    PointCloud,
)

__all__ = [
    "PointCloud",
    "ObstacleCloud",
    "FreeSpaceRays",
    "GeometryMetrics",
    "PointCloudBuilder",
    "transform_point_cloud",
    "build_obstacle_cloud",
    "build_free_space_rays",
    "build_geometry_metrics",
    "GeometryQuality",
    "classify_geometry_quality",
    "GeometryFrame",
    "RegionEvidence",
    "ClearanceEvidence",
    "ClearanceSupportState",
    "SurfaceEvidence",
    "build_surface_evidence",
    "BoundaryEvidence",
    "BoundaryState",
    "BoundaryDirection",
    "build_boundary_evidence",
    "OpeningEvidence",
    "build_opening_evidence",
    "GeometryFrameQuality",
    "GeometryFrameQualityState",
]
