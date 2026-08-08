"""
Result fusion — combines the disparity/depth/traversability/obstacle stage
outputs into one DepthPerceptionResult, including the single scalar
`confidence` field the original per-stage modules don't otherwise produce.

confidence is deliberately simple: the mean of the traversability grid's
per-region RegionAnalyzer confidence scores (each already a blend of
disparity validity, texture, and entropy — see
traversability.region_analyzer.RegionAnalyzer._confidence). This is a
transparent aggregate of an existing, already-computed signal, not a new
inference algorithm — 0.0 if the grid is empty.

This module holds no state and calls no cv2/algorithm code itself; it only
assembles values already computed by the other stages.
"""

from typing import Optional

from depth_perception_engine.geometry.types import FreeSpaceRays, GeometryMetrics, ObstacleCloud, PointCloud
from depth_perception_engine.models.result import (
    BeamReading,
    DepthPerceptionResult,
    ObstacleAssessment,
    TraversabilityResult,
)


def to_obstacle_assessment(raw: dict) -> ObstacleAssessment:
    """Wrap obstacles.ThreatAssessor.assess()'s raw dict into a typed ObstacleAssessment."""
    beams = [BeamReading(**b) for b in raw["beams"]]
    safest = BeamReading(**raw["safest_beam"]) if raw["safest_beam"] else None
    return ObstacleAssessment(beams=beams, safest_beam=safest)


def aggregate_confidence(traversability: TraversabilityResult) -> float:
    """Mean per-region confidence from the traversability grid, or 0.0 if empty."""
    regions = traversability.regions
    if not regions:
        return 0.0
    return sum(r.confidence for r in regions.values()) / len(regions)


def build_result(
    disparity_map,
    depth_map,
    traversability: TraversabilityResult,
    obstacles: ObstacleAssessment,
    processing_time_ms: float,
    timestamp=None,
    geometry: Optional[PointCloud] = None,
    geometry_body: Optional[PointCloud] = None,
    obstacle_cloud: Optional[ObstacleCloud] = None,
    free_space_rays: Optional[FreeSpaceRays] = None,
    geometry_metrics: Optional[GeometryMetrics] = None,
) -> DepthPerceptionResult:
    """Assemble the final structured pipeline output.

    valid_disparity_mask/valid_depth_mask are computed here (not left
    implicit) from this codebase's one, universal invalid-value convention
    — disparity <= 0, depth == 0 — used identically by DisparityEngine,
    DepthEstimator, and RegionAnalyzer already; see
    docs/DATA_CONTRACTS.md.

    geometry (E3) / geometry_body (E4) / obstacle_cloud, free_space_rays,
    geometry_metrics (E5): all pass-through only — this function does not
    build/transform/filter/aggregate any of them itself (that's
    geometry.PointCloudBuilder / geometry.transform_point_cloud /
    geometry.build_obstacle_cloud / geometry.build_free_space_rays /
    geometry.build_geometry_metrics, invoked by the caller before
    build_result()); all default to None, identical in spirit to
    timestamp's pass-through-or-None convention above, so every existing
    call site (including pipeline.api's process_stereo_pair, which never
    passes any of them) is unaffected.
    """
    return DepthPerceptionResult(
        disparity_map=disparity_map,
        depth_map=depth_map,
        traversability_mask=traversability,
        obstacles=obstacles,
        confidence=aggregate_confidence(traversability),
        processing_time_ms=processing_time_ms,
        valid_disparity_mask=disparity_map > 0,
        valid_depth_mask=depth_map > 0,
        timestamp=timestamp,
        geometry=geometry,
        geometry_body=geometry_body,
        obstacle_cloud=obstacle_cloud,
        free_space_rays=free_space_rays,
        geometry_metrics=geometry_metrics,
    )
