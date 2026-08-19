"""
Local geometric reliability — Phase I3.

Addresses a specific, confirmed false-authoritative-geometry mechanism:
a genuine occlusion/dis-occlusion strip adjacent to a real depth
discontinuity can read as 100% "valid" disparity (StereoSGBM's own
smoothness-regularized cost aggregation extends the confidently-matched
near surface's disparity across the ambiguous, no-true-correspondence
region — a "foreground fattening" artifact, not random noise) while still
being fabricated. Per-pixel statistical checks (local variance/outlier
rejection, invalid-neighbor density) cannot detect this: the fabricated
values are numerically indistinguishable from genuine extensions of the
real, adjacent surface (empirically measured, Phase I3 offline
benchmarks, benchmarks/i3_occlusion_safety/).

What DOES work: the contamination is geometrically predictable, not
just statistically hidden. Wherever a large disparity DROP exists within
a short lookahead window (a genuine near->far transition), classical
stereo-occlusion geometry guarantees a shadow zone of width approximately
equal to that disparity difference exists on the near (higher-disparity)
side of the transition — this is a physical consequence of observing a
depth step from two different viewpoints, not an implementation
artifact, and holds regardless of what value StereoSGBM happens to fill
that zone with. This module computes that shadow zone directly from the
disparity map — no optical flow, no neural inference, no new sensor
input, no per-pixel Python loop (bounded-width vectorized shift/OR).

This is a RELIABILITY signal, not a validity/sentinel value: it never
sets disparity/depth to invalid, and does not change
DepthPerceptionResult.valid_disparity_mask/valid_depth_mask or
PointCloud.valid_mask's own frozen "this pixel has valid depth" contract
(Level 0-2 and the E2-E5 PointCloud/ObstacleCloud/FreeSpaceRays chain are
completely unaffected unless a caller explicitly threads this mask in).
Callers (pipeline.py) use it to additionally discount/exclude
shadow-zone pixels from the SPECIFIC Level-3/4 evidence builders whose
own "support"/"confirmed" semantics this class of contamination was found
to defeat (obstacle_cloud, free_space_rays, boundary/surface per-cell
support) — one shared mask, computed once, rather than a separate
heuristic reimplemented in each builder.
"""

import numpy as np


def compute_shadow_zone_mask(
    disparity_map: np.ndarray,
    valid_mask: np.ndarray,
    lookahead_px: int,
    gradient_threshold_px: float,
    max_width_px: int,
) -> np.ndarray:
    """Flag pixels within a geometrically-predicted occlusion shadow zone.

    For every column x, compares disparity[x] against disparity[x +
    lookahead_px] (a wider baseline than an adjacent-pixel difference,
    robust to StereoSGBM's own few-pixel smoothing of a true step — an
    immediate 1px diff systematically under-measures the real jump size;
    empirically, benchmarks/i3_occlusion_safety/ found lookahead_px=8
    recovers ~99.6% of a known synthetic occlusion strip's true extent
    with zero false triggers anywhere on flat/textured, discontinuity-free
    scenes). Wherever that comparison shows a genuine drop (near -> far,
    i.e. disparity decreasing left-to-right) of at least
    `gradient_threshold_px`, marks the `round(gap)` columns immediately
    to the LEFT of x (capped at `max_width_px`) as shadow-zone — matching
    the exact near-side occlusion-strip placement this module's own
    calibration fixtures (benchmarks/i1_stereo_accuracy/fixtures.py's
    `make_discontinuity_fixture(..., occlusion=True)`) use to construct a
    known-ground-truth dis-occlusion region.

    Fully vectorized: one array comparison for the gradient/trigger test,
    then a `max_width_px`-bounded loop of whole-array shift/OR
    operations (not a per-pixel Python loop) to expand each trigger into
    its shadow-zone width.

    Args:
        disparity_map: (H, W) float, raw disparity in pixels (same array
            DisparityEngine/DepthEstimator already operate on).
        valid_mask: (H, W) bool — only valid-on-both-sides comparisons
            can trigger (an already-invalid pixel needs no additional
            reliability signal; it is already excluded downstream).
        lookahead_px: comparison baseline, columns.
        gradient_threshold_px: minimum disparity drop (pixels) within
            `lookahead_px` to count as a genuine transition worth
            shadow-flagging.
        max_width_px: hard cap on shadow-zone width per trigger (bounds
            both worst-case coverage loss for very large jumps and the
            cost of this function itself).

    Returns:
        (H, W) bool — True where a pixel falls inside a predicted
        shadow zone. All-False whenever no qualifying transition exists
        anywhere in the frame (e.g. any flat/single-plane scene) —
        verified to trigger 0 times across benchmarks/i1_stereo_accuracy/'s
        A/B/C flat fixtures at every tested depth.
    """
    h, w = disparity_map.shape[:2]
    shadow = np.zeros((h, w), dtype=bool)
    if lookahead_px < 1 or max_width_px < 1 or w <= lookahead_px:
        return shadow

    left = disparity_map[:, : w - lookahead_px]
    right = disparity_map[:, lookahead_px:]
    left_valid = valid_mask[:, : w - lookahead_px]
    right_valid = valid_mask[:, lookahead_px:]

    gap = left - right
    trigger = (gap >= gradient_threshold_px) & left_valid & right_valid
    width = np.where(trigger, np.minimum(np.round(gap).astype(np.int64), max_width_px), 0)

    full_width = np.zeros((h, w), dtype=np.int64)
    full_width[:, : w - lookahead_px] = width

    for k in range(1, min(max_width_px, w - 1) + 1):
        shadow[:, : w - k] |= full_width[:, k:w] >= k

    return shadow


def compute_ramp_zone_mask(
    disparity_map: np.ndarray,
    valid_mask: np.ndarray,
    window_px: int,
    gradient_threshold_px: float,
) -> np.ndarray:
    """Flag pixels sitting inside a WIDE, direction-agnostic disparity
    transition — a second, distinct contamination mechanism from
    compute_shadow_zone_mask's classical occlusion-shadow geometry.

    Phase I6.3: root-caused directly (a real depth profile trace across a
    multi_zone ClearanceEvidence false-clear sector) to StereoSGBM's own
    smoothness-regularization radius (tied to block_size) spreading a
    strong disparity step across a ~20-column gradual ramp, rather than
    the narrow (~disparity-gap-width), geometrically-predicted occlusion
    strip compute_shadow_zone_mask models. That ramp is far wider than
    compute_shadow_zone_mask's own width model, and (confirmed directly,
    both forward and mirrored) does not overlap it at all — a genuinely
    separate mechanism, not a parameter-tuning gap in the existing one.

    Mechanism: for every pixel, the rolling max-min disparity RANGE within
    a window_px-wide window centered on it. Flagged wherever that range is
    >= gradient_threshold_px. Deliberately direction-agnostic (unlike
    compute_shadow_zone_mask's one-directional "decreasing left-to-right"
    trigger) because a wide SGBM smoothing ramp can run either way and the
    contamination is symmetric — proven necessary: a bidirectional/
    mirrored variant of compute_shadow_zone_mask itself still measured
    0.0% overlap with the known false-clear case, because the problem is
    the (narrow) WIDTH MODEL, not direction.

    This is a second, independent reliability signal alongside
    compute_shadow_zone_mask, not a replacement — pipeline.py unions the
    two (where both are enabled) into one combined mask before it reaches
    ThreatAssessor.assess()'s contamination check. It is NOT threaded into
    build_obstacle_cloud/build_free_space_rays/build_surface_evidence/
    build_boundary_evidence — those already have their own separately-
    validated (I3/I4, 100% precision/recall) shadow_zone_mask-only
    behavior, and widening their exclusion was out of scope for what this
    phase measured; only the ClearanceEvidence false-clear path was
    root-caused and validated against this signal
    (benchmarks/i5_surface_opening_clearance/clearance_rootcause/
    ramp_zone_gate_prototype.py — window_px=24 gives real margin below
    every measured true-positive overlap while the cost is bounded and
    concentrated exclusively in genuinely transition-adjacent sectors).

    Vectorized: a `window_px // 2`-bounded loop of whole-array shift/
    minimum/maximum operations (not a per-pixel Python loop), matching
    compute_shadow_zone_mask's own style. Invalid pixels are excluded via
    +-inf sentinels so they can never win a rolling min/max, then masked
    out of the returned array entirely.

    Args:
        disparity_map: (H, W) float, raw disparity in pixels.
        valid_mask: (H, W) bool — only valid pixels can be flagged or
            contribute to another pixel's rolling range.
        window_px: width (in columns) of the centered window each pixel's
            local disparity range is computed over.
        gradient_threshold_px: minimum local disparity range (pixels)
            within that window to count as ramp-zone contamination.

    Returns:
        (H, W) bool — True where a pixel's local window shows a
        disparity range >= gradient_threshold_px. All-False on any flat/
        single-plane/discontinuity-free scene (a constant-disparity
        window always has range 0).
    """
    h, w = disparity_map.shape[:2]
    if window_px < 1:
        return np.zeros((h, w), dtype=bool)

    big = np.float32(1.0e6)
    d_max = np.where(valid_mask, disparity_map, -big).astype(np.float32)
    d_min = np.where(valid_mask, disparity_map, big).astype(np.float32)
    roll_max = d_max.copy()
    roll_min = d_min.copy()

    half = max(1, window_px // 2)
    for k in range(1, min(half, w - 1) + 1):
        roll_max[:, k:] = np.maximum(roll_max[:, k:], d_max[:, : w - k])
        roll_min[:, k:] = np.minimum(roll_min[:, k:], d_min[:, : w - k])
        roll_max[:, : w - k] = np.maximum(roll_max[:, : w - k], d_max[:, k:])
        roll_min[:, : w - k] = np.minimum(roll_min[:, : w - k], d_min[:, k:])

    rng = roll_max - roll_min
    return (rng >= gradient_threshold_px) & valid_mask
