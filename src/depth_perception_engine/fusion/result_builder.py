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
from depth_perception_engine.temporal.types import (
    MotionAwareReliability,
    TemporalConsistency,
    TemporalPersistence,
    TemporalStabilization,
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
    temporal_admission_status: Optional[str] = None,
    temporal_consistency: Optional[TemporalConsistency] = None,
    temporal_stabilization: Optional[TemporalStabilization] = None,
    rotation_compensation_status: Optional[str] = None,
    motion_aware_reliability: Optional[MotionAwareReliability] = None,
    temporal_persistence: Optional[TemporalPersistence] = None,
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

    temporal_admission_status (Level 4, Phase E2) / temporal_consistency
    (Level 4, Phase E3) / temporal_stabilization (Level 4, Phase E4) /
    rotation_compensation_status (Level 4, Phase E5) /
    motion_aware_reliability (Level 4, Phase E6) / temporal_persistence
    (Level 4, Phase E7): pass-through only, same discipline as
    geometry/geometry_body/etc. above — this function does not compute
    any of them (that's temporal.TemporalHistory.admit() /
    temporal.compute_temporal_consistency() /
    temporal.compute_temporal_stabilization() /
    temporal.compute_rotation_compensation() /
    temporal.compute_motion_aware_reliability() /
    temporal.persistence.TemporalPersistenceTracker.update(), all called
    by the pipeline before this function runs). All default to None, so
    every pre-E2/E3/E4/E5/E6/E7 call site is unaffected.
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
        temporal_admission_status=temporal_admission_status,
        temporal_consistency=temporal_consistency,
        temporal_stabilization=temporal_stabilization,
        rotation_compensation_status=rotation_compensation_status,
        motion_aware_reliability=motion_aware_reliability,
        temporal_persistence=temporal_persistence,
    )
