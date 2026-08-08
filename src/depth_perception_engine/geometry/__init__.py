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
not wired into DepthPerceptionPipeline.process(). See
docs/E6_IMPLEMENTATION_PLAN.md for what's next and docs/LEVEL3_CONTRACTS.md
for the full contract rationale.

Deliberately not re-exported from the top-level depth_perception_engine
package — see this subpackage's types.py module docstring for why.
"""

from depth_perception_engine.geometry.free_space import build_free_space_rays
from depth_perception_engine.geometry.geometry_metrics import (
    GeometryQuality,
    build_geometry_metrics,
    classify_geometry_quality,
)
from depth_perception_engine.geometry.obstacle_extractor import build_obstacle_cloud
from depth_perception_engine.geometry.point_cloud_builder import PointCloudBuilder
from depth_perception_engine.geometry.rigid_transform import transform_point_cloud
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
]
