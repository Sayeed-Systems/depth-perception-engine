"""
Geometry metrics aggregation — Level 3, Phase E5 — plus quality
classification — Level 3, Phase E6.

Populates exactly the four fields geometry.types.GeometryMetrics froze at
E1 — no new fields added. That frozen shape is narrower than what a
richer metrics set might include (max/farthest distance, separate
obstacle-point/ray counts, an explicit unknown-fraction, spatial
coverage) — those are real, useful, and NOT built here; extending
GeometryMetrics itself is a decision for a future pass to make
deliberately, not something to add ad hoc while implementing E5. See
docs/E6_IMPLEMENTATION_PLAN.md.

Each field below has one precise definition (Task 8's own requirement) —
stated in build_geometry_metrics's docstring, not left to be inferred
from the frozen type's already-fairly-precise docstring alone.

Pure aggregation over already-computed values — no new geometric
computation, no reprojection, no re-derivation of validity.

E6 addendum: classify_geometry_quality() below maps a single, already
precisely-defined GeometryMetrics field (valid_fraction) onto one of
three plain string labels (GeometryQuality.HEALTHY/DEGRADED/
NO_USABLE_GEOMETRY) — mirroring obstacles.ThreatAssessor's existing
CLEAR/CAUTION/BLOCKED/NO_DATA precedent (plain string constants, not a
new Enum type) rather than inventing a new frozen contract. Deliberately
NOT a blended/composite score — Task 5's explicit "do not combine
unrelated quantities into an undocumented scalar" — and NOT wired into
DepthPerceptionPipeline.process() or any result field: purely opt-in,
callable on a result's already-present geometry_metrics whenever a
caller wants a categorical judgement instead of reading the raw fraction
themselves. See PipelineConfig.geometry_healthy_min_valid_fraction/
geometry_degraded_min_valid_fraction's own docstring for why these
thresholds are a policy choice, not a physical constant.
"""

from typing import Optional

import numpy as np

from depth_perception_engine.geometry.types import FreeSpaceRays, GeometryMetrics, ObstacleCloud, PointCloud


class GeometryQuality:
    """Plain string constants for classify_geometry_quality()'s return
    value — not a closed Enum, matching obstacles.ThreatAssessor's
    existing CLEAR/CAUTION/BLOCKED/NO_DATA convention for exactly this
    kind of categorical-but-lightweight state in this codebase."""

    HEALTHY = "HEALTHY"
    """valid_fraction >= healthy_min_valid_fraction — sufficient valid
    geometry to be trusted at face value, per the configured threshold."""

    DEGRADED = "DEGRADED"
    """degraded_min_valid_fraction <= valid_fraction < healthy_min_valid_fraction
    — processing succeeded and some real geometric evidence exists, but
    coverage is limited; treat any decision derived from it cautiously."""

    NO_USABLE_GEOMETRY = "NO_USABLE_GEOMETRY"
    """valid_fraction < degraded_min_valid_fraction (includes exactly
    0.0) — processing succeeded (this is not a failure — see
    docs/IMPLEMENTATION_STATUS.md's E6 addendum for the full
    failure-vs-degradation taxonomy) but there is no reliable spatial
    evidence this frame. Must never be treated as "clear"/"free" by a
    downstream consumer — the correct reading is UNKNOWN, not safe."""


def build_geometry_metrics(
    cloud: PointCloud,
    obstacle_cloud: Optional[ObstacleCloud],
    free_space_rays: Optional[FreeSpaceRays],
) -> GeometryMetrics:
    """Summarize a PointCloud/ObstacleCloud/FreeSpaceRays triple into one GeometryMetrics.

    Precise definitions:
        point_count: the number of valid (non-NaN) points in `cloud` —
            `int(cloud.valid_mask.sum())`. Not the obstacle-point count or
            ray count (those aren't separately representable in the
            frozen GeometryMetrics shape — see this module's docstring).
        valid_fraction: `point_count / cloud.valid_mask.size` — the
            fraction of `cloud`'s total pixel grid that carried valid
            geometry. A data-quality metric (matches the frozen
            docstring's own "not fraction 'occupied'/'free'" caveat), not
            a scene-content judgement.
        min_obstacle_distance_m: `float(obstacle_cloud.distances_m.min())`
            if `obstacle_cloud` is non-None and non-empty; `None`
            otherwise (either because obstacle geometry wasn't computed
            this call, or because it was computed and found zero
            in-range points — both cases are "no obstacle evidence this
            frame", which is exactly what `None` already means per the
            frozen type's own docstring).
        mean_free_space_m: `float(free_space_rays.ranges_m.mean())` if
            `free_space_rays` is non-None and non-empty; `None` otherwise
            (same two-cases-collapse-to-None reasoning as above).

    Args:
        cloud: The source PointCloud (in this pipeline's usage, the E4
            body-frame cloud) that `point_count`/`valid_fraction` are
            computed over.
        obstacle_cloud: This frame's ObstacleCloud, or None if obstacle
            geometry was not computed this call (e.g.
            PipelineConfig.enable_obstacle_geometry is False).
        free_space_rays: This frame's FreeSpaceRays, or None if free-space
            geometry was not computed this call.

    Returns:
        A GeometryMetrics using exactly its four frozen fields.
    """
    valid_count = int(cloud.valid_mask.sum())
    total = int(cloud.valid_mask.size)
    valid_fraction = (valid_count / total) if total > 0 else 0.0

    min_obstacle_distance_m = None
    if obstacle_cloud is not None and obstacle_cloud.distances_m is not None and obstacle_cloud.distances_m.size > 0:
        min_obstacle_distance_m = float(np.min(obstacle_cloud.distances_m))

    mean_free_space_m = None
    if free_space_rays is not None and free_space_rays.ranges_m.size > 0:
        mean_free_space_m = float(np.mean(free_space_rays.ranges_m))

    return GeometryMetrics(
        min_obstacle_distance_m=min_obstacle_distance_m,
        mean_free_space_m=mean_free_space_m,
        point_count=valid_count,
        valid_fraction=valid_fraction,
    )


def classify_geometry_quality(
    metrics: GeometryMetrics,
    healthy_min_valid_fraction: float,
    degraded_min_valid_fraction: float,
) -> str:
    """Classify a frame's geometry coverage into one GeometryQuality tier.

    Based on exactly one field, `metrics.valid_fraction` — deliberately
    not a blend of multiple signals (Task 5: "do not combine unrelated
    quantities into an undocumented scalar"). Boundaries are inclusive on
    their lower edge, matching PipelineConfig's own validated ordering
    (`degraded_min_valid_fraction <= healthy_min_valid_fraction`):

        valid_fraction >= healthy_min_valid_fraction   -> HEALTHY
        degraded_min_valid_fraction <= valid_fraction
            < healthy_min_valid_fraction                -> DEGRADED
        valid_fraction < degraded_min_valid_fraction    -> NO_USABLE_GEOMETRY

    The HEALTHY check runs first and is itself inclusive, so if the two
    thresholds are set equal, DEGRADED becomes unreachable at exactly
    that value (it resolves to HEALTHY, not DEGRADED) — a legal, if
    unusual, configuration; see tests/test_geometry_quality.py::
    TestBoundaryConditions::test_equal_thresholds_collapse_degraded_tier_to_the_empty_set.

    This function does not itself decide what "sufficient" means — the
    two threshold arguments are a policy choice the caller supplies (in
    practice, PipelineConfig.geometry_healthy_min_valid_fraction/
    geometry_degraded_min_valid_fraction); see those fields' own
    docstring for why no universal default is correct.

    Args:
        metrics: A GeometryMetrics, e.g. DepthPerceptionResult.geometry_metrics.
        healthy_min_valid_fraction: Inclusive lower bound for HEALTHY.
        degraded_min_valid_fraction: Inclusive lower bound for DEGRADED.

    Returns:
        One of GeometryQuality.HEALTHY / DEGRADED / NO_USABLE_GEOMETRY.

    Raises:
        ValueError: If the two thresholds are not both in [0, 1] with
            `degraded_min_valid_fraction <= healthy_min_valid_fraction`
            — the same ordering PipelineConfig.__post_init__ already
            enforces, re-checked here since this function does not
            require a PipelineConfig, only the two raw floats.
    """
    if not (0.0 <= degraded_min_valid_fraction <= 1.0):
        raise ValueError(
            f"degraded_min_valid_fraction ({degraded_min_valid_fraction}) must be in [0, 1]."
        )
    if not (0.0 <= healthy_min_valid_fraction <= 1.0):
        raise ValueError(
            f"healthy_min_valid_fraction ({healthy_min_valid_fraction}) must be in [0, 1]."
        )
    if not (degraded_min_valid_fraction <= healthy_min_valid_fraction):
        raise ValueError(
            "Require degraded_min_valid_fraction <= healthy_min_valid_fraction, got "
            f"degraded_min_valid_fraction={degraded_min_valid_fraction}, "
            f"healthy_min_valid_fraction={healthy_min_valid_fraction}."
        )

    if metrics.valid_fraction >= healthy_min_valid_fraction:
        return GeometryQuality.HEALTHY
    if metrics.valid_fraction >= degraded_min_valid_fraction:
        return GeometryQuality.DEGRADED
    return GeometryQuality.NO_USABLE_GEOMETRY
