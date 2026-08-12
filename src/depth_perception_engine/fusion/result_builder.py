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

import math
from typing import Dict, List, Optional

from depth_perception_engine.frames import FrameId
from depth_perception_engine.geometry.boundary import BoundaryEvidence
from depth_perception_engine.geometry.geometry_metrics import GeometryQuality, classify_geometry_quality
from depth_perception_engine.geometry.opening import OpeningEvidence
from depth_perception_engine.geometry.provider import (
    ClearanceEvidence,
    ClearanceSupportState,
    GeometryFrame,
    GeometryFrameQuality,
    GeometryFrameQualityState,
    RegionEvidence,
)
from depth_perception_engine.geometry.surface import SurfaceEvidence
from depth_perception_engine.geometry.types import FreeSpaceRays, GeometryMetrics, ObstacleCloud, PointCloud
from depth_perception_engine.models.result import (
    BeamReading,
    DepthPerceptionResult,
    ObstacleAssessment,
    TraversabilityResult,
)
from depth_perception_engine.temporal.consistency import TemporalConsistencyState
from depth_perception_engine.temporal.persistence import TemporalPersistenceState
from depth_perception_engine.temporal.reliability import MotionAwareReliabilityState
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
    surface_evidence: Optional[List[SurfaceEvidence]] = None,
    boundary_evidence: Optional[List[BoundaryEvidence]] = None,
    opening_evidence: Optional[List[OpeningEvidence]] = None,
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

    surface_evidence (Phase D4): pass-through only, identical discipline —
    this function does not compute it (that's
    geometry.surface.build_surface_evidence(), called by the pipeline
    before this function runs). Defaults to None, so every pre-D4 call
    site is unaffected.

    boundary_evidence (Phase D5): pass-through only, identical discipline —
    this function does not compute it (that's
    geometry.boundary.build_boundary_evidence(), called by the pipeline
    before this function runs). Defaults to None, so every pre-D5 call
    site is unaffected.

    opening_evidence (Phase D6): pass-through only, identical discipline —
    this function does not compute it (that's
    geometry.opening.build_opening_evidence(), called by the pipeline
    before this function runs). Defaults to None, so every pre-D6 call
    site is unaffected.
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
        surface_evidence=surface_evidence,
        boundary_evidence=boundary_evidence,
        opening_evidence=opening_evidence,
    )


def build_region_evidence(traversability: TraversabilityResult, frame_id: str) -> Dict[str, RegionEvidence]:
    """Extract neutral RegionEvidence from an already-computed TraversabilityResult
    — Phase D3/D8 (see docs/DPE_V1_PROVIDER_CONTRACT.md).

    Pure field-by-field extraction of RegionAnalyzer's already-computed
    RegionStats (via TraversabilityResult.regions) — no new statistic is
    computed, no threshold is re-applied. `RegionStats.classification`
    (RegionClass — a behavioral/occupancy label) is the one field
    deliberately NOT carried over; every other field maps directly, with
    `valid_pct` (0..100) re-expressed as `valid_fraction` (0..1) — a unit
    conversion, matching GeometryMetrics.valid_fraction's own convention,
    not a new computation. `frame_id` (Phase D8) is generic, copied
    straight through from the caller — matches every other evidence
    type's own per-object frame_id convention.
    """
    return {
        name: RegionEvidence(
            frame_id=frame_id,
            name=stats.name, row=stats.row, col=stats.col,
            x1=stats.x1, y1=stats.y1, x2=stats.x2, y2=stats.y2,
            valid_count=stats.valid_count, total_pixels=stats.total_pixels,
            valid_fraction=stats.valid_pct / 100.0,
            depth_avg_m=stats.depth_avg_m, depth_median_m=stats.depth_median_m,
            depth_min_m=stats.depth_min_m, depth_max_m=stats.depth_max_m,
            texture_score=stats.texture_score, entropy=stats.entropy,
            gradient_magnitude=stats.gradient_magnitude,
            texture_class=stats.texture_class,
            confidence=stats.confidence,
        )
        for name, stats in traversability.regions.items()
    }


def _bearing_rad(x_px: float, principal_point_x_px: float, focal_length_px: float) -> float:
    """Standard pinhole horizontal bearing — Phase D7.

    `atan2(x_px - principal_point_x_px, focal_length_px)`: the angle, in
    the CAMERA_OPTICAL_LEFT frame's own X-right/Z-forward plane, between
    the optical axis and the ray through image column `x_px`. Positive
    means `x_px` lies toward increasing image-x (right of the principal
    point); negative means left; exactly 0.0 at the principal point
    itself. See geometry.provider.ClearanceEvidence's own docstring for
    the full convention this is used to populate.
    """
    return math.atan2(x_px - principal_point_x_px, focal_length_px)


def build_clearance_evidence(
    obstacles: ObstacleAssessment,
    frame_id: str,
    focal_length_px: float,
    principal_point_x_px: float,
    min_coverage_fraction: float,
) -> List[ClearanceEvidence]:
    """Extract neutral ClearanceEvidence from an already-computed ObstacleAssessment
    — Phase D3/D7 (see docs/DPE_V1_PROVIDER_CONTRACT.md).

    Pure extraction of ThreatAssessor's already-computed per-beam scan (via
    the existing, Tier 1 BeamReading) — no new distance is measured, no
    caution_distance_m/clear_distance_m threshold is re-applied.
    `BeamReading.status` (CLEAR/CAUTION/BLOCKED/NO_DATA — a
    proximity-policy label) is deliberately NOT carried over.
    `distance_m <= 0.0` (BeamReading's own "0 = no data" convention) is
    re-expressed as `nearest_distance_m=None`/`has_evidence=False` — this
    codebase's standard "None means no data, never a fabricated 0.0"
    convention (e.g. GeometryMetrics.min_obstacle_distance_m) — a
    representation change, not a new computation.

    valid_count/total_pixels (Phase D7): BeamReading's own new D7 fields,
    reused directly — ThreatAssessor.assess() computed them once, not
    recomputed here. coverage_fraction is a trivial division (0.0 when
    total_pixels is 0, never a divide-by-zero, never a fabricated
    "fully covered" default). support_state classifies has_evidence/
    coverage_fraction into ClearanceSupportState's 3-way split — see that
    class's own docstring.

    bearing_center_rad/bearing_min_rad/bearing_max_rad (Phase D7): the
    standard pinhole bearing formula (_bearing_rad above) applied to this
    beam's own (x1+x2)/2, x1, and x2 pixel columns respectively — no new
    camera-geometry algorithm, a direct application of already-available
    calibration (focal_length_px/principal_point_x_px, the same
    Q-matrix-derived values rotation compensation and surface/boundary/
    opening geometry already use).
    """
    evidence = []
    for beam in obstacles.beams:
        has_evidence = beam.distance_m > 0.0
        coverage_fraction = (beam.valid_count / beam.total_pixels) if beam.total_pixels else 0.0

        if not has_evidence:
            support_state = ClearanceSupportState.NO_EVIDENCE
        elif coverage_fraction >= min_coverage_fraction:
            support_state = ClearanceSupportState.SUPPORTED
        else:
            support_state = ClearanceSupportState.PARTIALLY_SUPPORTED

        center_x_px = (beam.x1 + beam.x2) / 2.0
        evidence.append(ClearanceEvidence(
            frame_id=frame_id, index=beam.index, x1=beam.x1, x2=beam.x2,
            nearest_distance_m=beam.distance_m if has_evidence else None,
            has_evidence=has_evidence,
            valid_count=beam.valid_count, total_pixels=beam.total_pixels,
            coverage_fraction=coverage_fraction, support_state=support_state,
            bearing_center_rad=_bearing_rad(center_x_px, principal_point_x_px, focal_length_px),
            bearing_min_rad=_bearing_rad(beam.x1, principal_point_x_px, focal_length_px),
            bearing_max_rad=_bearing_rad(beam.x2, principal_point_x_px, focal_length_px),
        ))
    return evidence


_GEOMETRY_VALIDITY_QUALITY_MAP = {
    GeometryQuality.HEALTHY: GeometryFrameQualityState.VALID,
    GeometryQuality.DEGRADED: GeometryFrameQualityState.DEGRADED,
    GeometryQuality.NO_USABLE_GEOMETRY: GeometryFrameQualityState.INSUFFICIENT,
}

_TEMPORAL_CONSISTENCY_QUALITY_MAP = {
    TemporalConsistencyState.CONSISTENT: GeometryFrameQualityState.VALID,
    TemporalConsistencyState.CONTRADICTORY: GeometryFrameQualityState.DEGRADED,
    TemporalConsistencyState.NOT_COMPARABLE: GeometryFrameQualityState.INSUFFICIENT,
    TemporalConsistencyState.INSUFFICIENT_EVIDENCE: GeometryFrameQualityState.INSUFFICIENT,
}

_MOTION_RELIABILITY_QUALITY_MAP = {
    MotionAwareReliabilityState.RELIABLE: GeometryFrameQualityState.VALID,
    MotionAwareReliabilityState.DEGRADED: GeometryFrameQualityState.DEGRADED,
    MotionAwareReliabilityState.UNRELIABLE: GeometryFrameQualityState.DEGRADED,
    MotionAwareReliabilityState.INSUFFICIENT_EVIDENCE: GeometryFrameQualityState.INSUFFICIENT,
}

_PERSISTENCE_QUALITY_MAP = {
    TemporalPersistenceState.CLASSIFIED: GeometryFrameQualityState.VALID,
    TemporalPersistenceState.UNRELIABLE: GeometryFrameQualityState.DEGRADED,
    TemporalPersistenceState.INSUFFICIENT_EVIDENCE: GeometryFrameQualityState.INSUFFICIENT,
}


def build_geometry_frame_quality(
    geometry_metrics: Optional[GeometryMetrics],
    temporal_consistency: Optional[TemporalConsistency],
    motion_aware_reliability: Optional[MotionAwareReliability],
    temporal_persistence: Optional[TemporalPersistence],
    geometry_healthy_min_valid_fraction: float,
    geometry_degraded_min_valid_fraction: float,
) -> GeometryFrameQuality:
    """Roll up DPE's four frame-level state-bearing evidence sources into
    one GeometryFrameQuality — Phase D8 (see docs/DPE_V1_PROVIDER_CONTRACT.md).

    Pure dict-lookup classification of already-computed state values — no
    stereo/depth/surface/boundary/opening recomputation, no new algorithm.
    geometry_validity_state reuses the EXISTING
    geometry.geometry_metrics.classify_geometry_quality() classifier and
    the EXISTING PipelineConfig.geometry_healthy_min_valid_fraction/
    geometry_degraded_min_valid_fraction thresholds — no new threshold was
    introduced for this dimension. Each of the four dimensions stays None
    when its own underlying DepthPerceptionResult/GeometryFrame field is
    None (the capability was never enabled/computed this frame) — see
    GeometryFrameQuality's own docstring for the full absent-vs-degraded
    contract this distinction exists to satisfy.
    """
    geometry_validity_state = None
    if geometry_metrics is not None:
        raw_quality = classify_geometry_quality(
            geometry_metrics, geometry_healthy_min_valid_fraction, geometry_degraded_min_valid_fraction,
        )
        geometry_validity_state = _GEOMETRY_VALIDITY_QUALITY_MAP[raw_quality]

    temporal_consistency_state = None
    if temporal_consistency is not None:
        temporal_consistency_state = _TEMPORAL_CONSISTENCY_QUALITY_MAP[temporal_consistency.state]

    motion_reliability_state = None
    if motion_aware_reliability is not None:
        motion_reliability_state = _MOTION_RELIABILITY_QUALITY_MAP[motion_aware_reliability.state]

    persistence_state = None
    if temporal_persistence is not None:
        persistence_state = _PERSISTENCE_QUALITY_MAP[temporal_persistence.state]

    # Fixed, documented order — also governs degradation_reasons' own order.
    dimensions = (
        ("GEOMETRY_VALIDITY", geometry_validity_state),
        ("TEMPORAL_CONSISTENCY", temporal_consistency_state),
        ("MOTION_RELIABILITY", motion_reliability_state),
        ("PERSISTENCE", persistence_state),
    )
    defined_states = [state for _, state in dimensions if state is not None]

    if any(state == GeometryFrameQualityState.DEGRADED for state in defined_states):
        overall_state = GeometryFrameQualityState.DEGRADED
    elif not defined_states or any(state == GeometryFrameQualityState.INSUFFICIENT for state in defined_states):
        overall_state = GeometryFrameQualityState.INSUFFICIENT
    else:
        overall_state = GeometryFrameQualityState.VALID

    degradation_reasons = [
        f"{name}:{state}" for name, state in dimensions
        if state is not None and state != GeometryFrameQualityState.VALID
    ]

    return GeometryFrameQuality(
        overall_state=overall_state,
        geometry_validity_state=geometry_validity_state,
        temporal_consistency_state=temporal_consistency_state,
        motion_reliability_state=motion_reliability_state,
        persistence_state=persistence_state,
        degradation_reasons=degradation_reasons,
    )


def build_geometry_frame(
    result: DepthPerceptionResult,
    focal_length_px: float,
    principal_point_x_px: float,
    clearance_min_coverage_fraction: float,
    geometry_healthy_min_valid_fraction: float,
    geometry_degraded_min_valid_fraction: float,
) -> GeometryFrame:
    """Construct a GeometryFrame purely by reading fields off an already-built
    DepthPerceptionResult — Phase D2 (see docs/DPE_V1_PROVIDER_CONTRACT.md).

    Zero recomputation, zero duplicate algorithms: every field below is a
    direct attribute read of `result`, the same object build_result() just
    returned. frame_id is always FrameId.CAMERA_OPTICAL_LEFT — disparity_map/
    depth_map/geometry are always expressed in that frame in this pipeline;
    geometry_body/obstacle_cloud/free_space_rays each declare their own
    (BODY) frame_id on themselves, per GeometryFrame's own frame convention.

    focal_length_px/principal_point_x_px/clearance_min_coverage_fraction
    (Phase D7) are the one exception to "every field is a direct attribute
    read of result": DepthPerceptionResult carries no calibration-derived
    field of its own (calibration is a pipeline-construction-time input,
    never stored per-frame), so build_clearance_evidence()'s bearing
    computation needs them threaded through explicitly by the caller
    (pipeline.py, which already holds them as
    self._rectified_focal_length_px/self._rectified_principal_point_px).
    Still zero recomputation of anything ALGORITHMIC — these are passed
    straight through to a trigonometric formula, not re-derived from
    scratch.

    region_evidence/clearance_evidence (Phase D3) are always populated
    (never None) because `result.traversability_mask`/`result.obstacles`
    are themselves unconditional, non-Optional DepthPerceptionResult
    fields — there is no PipelineConfig.enable_* flag that could make
    either one absent.

    surface_evidence (Phase D4) is read the same way — result.surface_evidence
    was itself computed once, upstream, by geometry.surface.
    build_surface_evidence() (a real new pipeline stage, gated by
    PipelineConfig.enable_surface_geometry) and is simply referenced here,
    not recomputed.

    boundary_evidence (Phase D5) is read the same way — result.boundary_evidence
    was itself computed once, upstream, by geometry.boundary.
    build_boundary_evidence() (a real new pipeline stage, gated by
    PipelineConfig.enable_boundary_geometry) and is simply referenced
    here, not recomputed.

    opening_evidence (Phase D6) is read the same way — result.opening_evidence
    was itself computed once, upstream, by geometry.opening.
    build_opening_evidence() (a real new pipeline stage, gated by
    PipelineConfig.enable_opening_geometry) and is simply referenced
    here, not recomputed.

    quality (Phase D8): geometry_healthy_min_valid_fraction/
    geometry_degraded_min_valid_fraction are the SAME thresholds
    classify_geometry_quality() already used before D8 — no new
    threshold. build_geometry_frame_quality() reads only fields already
    on `result`/already computed above (geometry_metrics,
    temporal_consistency, motion_aware_reliability, temporal_persistence)
    — zero recomputation, zero new algorithm, same discipline as
    region_evidence/clearance_evidence above.
    """
    return GeometryFrame(
        timestamp=result.timestamp,
        frame_id=FrameId.CAMERA_OPTICAL_LEFT,
        disparity_map=result.disparity_map,
        depth_map=result.depth_map,
        valid_disparity_mask=result.valid_disparity_mask,
        valid_depth_mask=result.valid_depth_mask,
        geometry=result.geometry,
        geometry_body=result.geometry_body,
        obstacle_cloud=result.obstacle_cloud,
        free_space_rays=result.free_space_rays,
        geometry_metrics=result.geometry_metrics,
        temporal_consistency=result.temporal_consistency,
        temporal_stabilization=result.temporal_stabilization,
        rotation_compensation_status=result.rotation_compensation_status,
        motion_aware_reliability=result.motion_aware_reliability,
        temporal_persistence=result.temporal_persistence,
        region_evidence=build_region_evidence(result.traversability_mask, FrameId.CAMERA_OPTICAL_LEFT),
        clearance_evidence=build_clearance_evidence(
            result.obstacles, FrameId.CAMERA_OPTICAL_LEFT,
            focal_length_px, principal_point_x_px, clearance_min_coverage_fraction,
        ),
        surface_evidence=result.surface_evidence,
        boundary_evidence=result.boundary_evidence,
        opening_evidence=result.opening_evidence,
        quality=build_geometry_frame_quality(
            result.geometry_metrics, result.temporal_consistency,
            result.motion_aware_reliability, result.temporal_persistence,
            geometry_healthy_min_valid_fraction, geometry_degraded_min_valid_fraction,
        ),
    )
