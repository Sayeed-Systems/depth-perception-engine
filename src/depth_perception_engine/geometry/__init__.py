"""
Level 3 geometry — interfaces only (Phase E1).

Result types (PointCloud, ObstacleCloud, FreeSpaceRays, GeometryMetrics)
are defined and frozen; nothing that produces them exists yet. See
docs/E2_IMPLEMENTATION_PLAN.md for what's next and
docs/LEVEL3_CONTRACTS.md for the full contract rationale.

Deliberately not re-exported from the top-level depth_perception_engine
package — see this subpackage's types.py module docstring for why.
"""

from depth_perception_engine.geometry.types import (
    FreeSpaceRays,
    GeometryMetrics,
    ObstacleCloud,
    PointCloud,
)

__all__ = ["PointCloud", "ObstacleCloud", "FreeSpaceRays", "GeometryMetrics"]
