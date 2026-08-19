"""
Geometric openings / passage structure — Phase D6.

Deterministic, bounded detection of "a geometrically supported gap between
surrounding physical structures" — never "safe/passable/traversable for
this vehicle." No PASSABLE/TRAVERSABLE/SAFE/DOORWAY/WINDOW/FLY_THROUGH
concept, no vehicle-size assumption, no platform-specific clearance
threshold appears anywhere in this module. Opening inference is never
triggered by two image edges alone — see "What counts as an opening"
below for the full, multi-part evidentiary requirement.

Built entirely from already-computed evidence, on the SAME grid
geometry.boundary already used (grid_rows/grid_cols here are always the
caller's own boundary_grid_rows/boundary_grid_cols — this is consuming
boundary's already-chosen grid, not inventing a third independent one,
and is not a "redesign" of BoundaryEvidence's own contract):

    geometry.boundary.BoundaryEvidence — confirms REAL structural
        transitions exist (state == OBSERVED_DISCONTINUITY) flanking a
        candidate gap. Reused directly, never reclassified.

    depth_map — the one piece BoundaryEvidence does not expose:
        per-cell ABSOLUTE median depth (BoundaryEvidence.depth_step_m is
        an unsigned magnitude; distinguishing "farther" (an opening) from
        "nearer" (a protruding obstacle) requires the sign, which requires
        each side's own absolute depth). Re-deriving a per-cell median
        from depth_map is the same trivial, already-precedented
        aggregation geometry.boundary/geometry.surface/RegionAnalyzer each
        already perform independently — not a re-derivation of depth
        estimation itself, and not "duplicating boundary computation" (the
        discontinuity CLASSIFICATION itself is reused, not recomputed).

    focal_length_px — already-available calibration-derived scalar (the
        same one the pipeline already threads through for rotation
        compensation) — used only for a simple, explicitly-approximate
        pinhole back-projection of a pixel EXTENT at a KNOWN depth
        (apparent_size_m = pixel_extent * range_m / focal_length_px), not
        a second depth-estimation algorithm.

Deliberately NOT consumed: geometry.FreeSpaceRays / geometry.ObstacleCloud
(both unorganized (N,) arrays with no per-grid-cell correlation to this
module's own row/col grid without re-deriving one — which would duplicate
obstacle_extractor's/free_space's own stride-decimation logic) and
geometry.surface.SurfaceEvidence (a real, identified, NOT-yet-built
refinement — corroborating "does the gap read as a coherent receding
surface or truly unbounded" is left for a future decision, not built
here, to keep this contract genuinely minimal). "Free-space support" is
instead expressed through this module's own support_fraction/
farther-than-reference-depth signal, consistent with FreeSpaceRays' own
definition of free space (observed space strictly between origin and a
measured endpoint).

What counts as an opening (the full admission rule — ALL of the
following, never any one alone):
    1. At least one of the two flanking cell-adjacencies (left, right, or
       both) is a REAL, already-confirmed geometry.boundary.BoundaryEvidence
       RIGHT-edge with state == OBSERVED_DISCONTINUITY. A span with NEITHER
       side confirmed (e.g. a smoothly receding surface with no genuine
       structural jump anywhere — see TestContinuousWallProducesNoFalseOpening)
       is never admitted, regardless of how its depth varies. If only one
       side is confirmed (the other is the image edge itself), the opening
       is admitted as PARTIAL (`at_image_boundary=True`) — see point 4.
    2. Every cell inside the candidate span has at least min_support_count
       valid depth pixels (same "insufficient support" gate boundary.py
       already established — never lower this to admit a low-evidence
       region).
    3. Every cell inside the span has its own median depth at least
       `min_range_ratio` times the FARTHER of the confirmed flanking
       structure cell(s)' own median depth — i.e., the candidate region
       must recede meaningfully beyond BOTH known structures, not merely
       differ from one side. This is the check that structurally prevents
       a near-side protrusion (an obstacle poking forward, which also
       produces a large |depth step| but in the WRONG direction) from
       ever being mistaken for an opening.
    4. A span touching the image edge (column 0 or the last column) with
       no cell beyond it to check is admitted only if its OTHER side has a
       real confirmed flank — reported with `at_image_boundary=True`, an
       explicit, honest signal that this opening's true extent beyond the
       observed frame is unknown, not a silent full/partial ambiguity.

OpeningEvidence is a POSITIVE-FINDINGS-ONLY list — unlike BoundaryEvidence/
SurfaceEvidence (one record per grid cell/pair, always, including an
explicit insufficient/no-finding state), a span with insufficient support
or no qualifying gap produces NO record at all. Absence from the list IS
how "not an opening"/"insufficient evidence" is represented — a parallel
record for every one of the combinatorially many non-opening spans would
be uninformative. A caller asking "were there any openings this frame" or
"is there evidence of insufficient support" reads `len(evidence)` and
`GeometryFrame.boundary_evidence`/`.geometry_metrics` respectively.

Scope: horizontal (left-right, along one grid row) spans only. Vertical/
2D opening-region detection is a real, identified extension NOT built
here — a deliberate scope-narrowing to keep this initial contract minimal
(see docs/DPE_V1_PROVIDER_CONTRACT.md's D6 record).

No `BoundaryEvidence`/`RegionEvidence`/`SurfaceEvidence`/`ClearanceEvidence`
grid is redesigned or forced into alignment by this module — this module
only ever CONSUMES the grid it is given.

Stable references, not fragile indices: OpeningEvidence does not carry a
positional index into `boundary_evidence` (BoundaryEvidence has no stable
ID field, and a list-position reference would be fragile — exactly what
the D6 design brief warns against). Its own `row`/`col_start`/`col_end`
are themselves sufficient, stable, self-describing geometric identifiers —
a consumer holding both `GeometryFrame.boundary_evidence` and
`.opening_evidence` can independently correlate them by matching
`row`/`col`/`direction`, without DPE inventing a brittle cross-reference.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from depth_perception_engine.geometry.boundary import BoundaryDirection, BoundaryEvidence, BoundaryState


@dataclass(frozen=True, slots=True)
class OpeningEvidence:
    """One confirmed geometric opening — a supported gap between two (or,
    at an image edge, one) real flanking structures — Phase D6. See this
    module's own docstring for the full admission rule and evidentiary
    contract.
    """

    frame_id: str
    row: int
    col_start: int
    col_end: int

    x1: int
    y1: int
    x2: int
    y2: int

    approx_range_m: float
    approx_width_m: float
    approx_height_m: float

    support_fraction: float
    at_image_boundary: bool


def build_opening_evidence(
    boundary_evidence: List[BoundaryEvidence],
    depth_map: np.ndarray,
    frame_id: str,
    grid_rows: int,
    grid_cols: int,
    min_support_count: int,
    min_range_ratio: float,
    focal_length_px: float,
    min_merge_depth_diff_m: float = 0.0,
) -> List[OpeningEvidence]:
    """Detect horizontal opening candidates per row of a grid_rows x
    grid_cols partition of depth_map, using already-computed
    boundary_evidence to confirm flanking structure.

    Args:
        boundary_evidence: this frame's already-computed BoundaryEvidence
            list (from geometry.boundary.build_boundary_evidence), built
            with the SAME grid_rows/grid_cols passed here.
        depth_map: metric depth map, (H, W), metres; invalid == 0.0 — same
            universal convention every other stage in this codebase uses.
        frame_id: copied directly onto every returned OpeningEvidence.
        grid_rows, grid_cols: MUST match the grid boundary_evidence was
            built with — this function does not (and cannot) verify that
            independently; passing a mismatched grid produces meaningless
            (row, col) lookups against boundary_evidence.
        min_support_count: minimum valid depth pixels a cell must contain
            to be considered part of, or a reference for, a candidate
            opening.
        min_range_ratio: a candidate span's own median depth must be at
            least this many times the farther confirmed flanking
            structure's median depth. Must be > 1.0.
        focal_length_px: calibration-derived focal length (pixels), used
            only for the explicitly-approximate width/height
            back-projection described in the module docstring.
        min_merge_depth_diff_m: Phase I5.1 (see
            benchmarks/i5_surface_opening_clearance/opening_rootcause/).
            A confirmed RIGHT-edge OBSERVED_DISCONTINUITY is treated as a
            real internal split only if its two flanking cells' own
            median depths differ by MORE than this value — measured root
            cause: a grid cell exactly at a genuine scene-depth
            transition can catch a thin sliver of the other side's depth
            (SGBM's own step-smoothing) and fit a spuriously-oriented
            normal, confirming OBSERVED_DISCONTINUITY between two cells
            whose median depth is (near-)identical, which otherwise
            fragments a genuine multi-cell-wide opening into pieces that
            each fail min_range_ratio against the other (now
            misclassified as a flank reference). This is a span-assembly
            recalibration, not a change to `geometry.boundary`'s own
            admission logic (two earlier attempts to gate the
            ORIENTATION signal itself — on SurfaceEvidence.planarity, and
            on depth_step_m — were each measured to regress genuine
            boundary recall and are NOT used here; see this module's own
            git history / DPE_V1_PROVIDER_CONTRACT record for that
            account). Measured insensitive across 0.01-0.10m; default
            0.0 preserves this function's exact pre-I5.1 behavior for
            any caller not passing it; pipeline.py supplies
            PipelineConfig.opening_min_merge_depth_diff_m (default 0.05)
            for the real pipeline. A `None`-median cell is never merged
            across (see the dead-zone hard-split logic below, unrelated
            to this parameter).

    Returns:
        A flat list of confirmed OpeningEvidence — empty when nothing
        qualifies this frame (see "positive-findings-only" in the module
        docstring). Deterministic: identical inputs always produce an
        identical (element-wise equal) output list.

    Raises:
        ValueError: If grid_rows/grid_cols < 1, min_support_count < 1,
            min_range_ratio <= 1.0, or focal_length_px <= 0.0.
    """
    if grid_rows < 1 or grid_cols < 1:
        raise ValueError(
            f"grid_rows/grid_cols must both be >= 1, got rows={grid_rows}, cols={grid_cols}."
        )
    if min_support_count < 1:
        raise ValueError(f"min_support_count ({min_support_count}) must be >= 1.")
    if min_range_ratio <= 1.0:
        raise ValueError(f"min_range_ratio ({min_range_ratio}) must be > 1.0.")
    if focal_length_px <= 0.0:
        raise ValueError(f"focal_length_px ({focal_length_px}) must be > 0.")

    valid_depth_mask = depth_map > 0.0
    h, w = depth_map.shape[:2]
    row_bounds = np.linspace(0, h, grid_rows + 1).astype(int)
    col_bounds = np.linspace(0, w, grid_cols + 1).astype(int)

    cells = [[None] * grid_cols for _ in range(grid_rows)]
    for r in range(grid_rows):
        y1, y2 = int(row_bounds[r]), int(row_bounds[r + 1])
        for c in range(grid_cols):
            x1, x2 = int(col_bounds[c]), int(col_bounds[c + 1])
            cell_valid = valid_depth_mask[y1:y2, x1:x2]
            cell_depth = depth_map[y1:y2, x1:x2]
            total_pixels = int(cell_valid.size)
            support_count = int(cell_valid.sum())
            support_fraction = (support_count / total_pixels) if total_pixels else 0.0
            median_depth_m = (
                float(np.median(cell_depth[cell_valid])) if support_count >= min_support_count else None
            )
            cells[r][c] = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "support_fraction": support_fraction,
                "median_depth_m": median_depth_m,
            }

    right_state = {
        (be.row, be.col): be.state
        for be in boundary_evidence
        if be.direction == BoundaryDirection.RIGHT
    }

    evidence: List[OpeningEvidence] = []
    for r in range(grid_rows):
        # Phase I5.1: a real flank edge is a confirmed OBSERVED_DISCONTINUITY
        # whose two flanking cells' own median depths genuinely differ by
        # more than min_merge_depth_diff_m — see this function's own
        # docstring for the measured spurious-split root cause this
        # excludes. (OBSERVED_DISCONTINUITY already implies both medians
        # are non-None, per geometry.boundary's own admission logic.)
        flank_edges = []
        for e in range(grid_cols - 1):
            if right_state.get((r, e)) != BoundaryState.OBSERVED_DISCONTINUITY:
                continue
            left_median = cells[r][e]["median_depth_m"]
            right_median = cells[r][e + 1]["median_depth_m"]
            if (
                left_median is not None and right_median is not None
                and abs(left_median - right_median) <= min_merge_depth_diff_m
            ):
                continue
            flank_edges.append(e)

        # Phase I5.1: an edge adjacent to a None-median (structurally
        # unobservable / insufficient-support) cell can never be safely
        # bridged by a span (span_cells below rejects any span containing
        # a None cell) — without an explicit split point there, the WHOLE
        # span between the nearest real flanks is discarded even when the
        # unobservable portion is a small, avoidable fraction of it.
        # These are ARTIFICIAL (non-real) split points — never a
        # confirmed flank, only a place a span may not cross — disjoint
        # from flank_edges by construction (OBSERVED_DISCONTINUITY already
        # implies both medians non-None).
        hard_splits = {
            e for e in range(grid_cols - 1)
            if cells[r][e]["median_depth_m"] is None or cells[r][e + 1]["median_depth_m"] is None
        }

        flank_edge_set = set(flank_edges)
        all_splits = sorted(flank_edge_set | hard_splits)
        boundaries = (
            [(-1, False)] + [(e, e in flank_edge_set) for e in all_splits] + [(grid_cols - 1, False)]
        )

        for i in range(len(boundaries) - 1):
            left_edge, left_is_real = boundaries[i]
            right_edge, right_is_real = boundaries[i + 1]
            c_start, c_end = left_edge + 1, right_edge
            if c_start > c_end or (not left_is_real and not right_is_real):
                continue

            span_cells = [cells[r][c] for c in range(c_start, c_end + 1)]
            if any(cell["median_depth_m"] is None for cell in span_cells):
                continue

            reference_depths = []
            if left_is_real:
                reference_depths.append(cells[r][c_start - 1]["median_depth_m"])
            if right_is_real:
                reference_depths.append(cells[r][c_end + 1]["median_depth_m"])
            if any(depth is None for depth in reference_depths):
                continue
            reference_depth_m = max(reference_depths)
            if reference_depth_m <= 0.0:
                continue

            if not all(
                cell["median_depth_m"] >= min_range_ratio * reference_depth_m for cell in span_cells
            ):
                continue

            approx_range_m = float(np.mean([cell["median_depth_m"] for cell in span_cells]))
            support_fraction = float(min(cell["support_fraction"] for cell in span_cells))

            x1, x2 = span_cells[0]["x1"], span_cells[-1]["x2"]
            y1, y2 = span_cells[0]["y1"], span_cells[0]["y2"]

            evidence.append(OpeningEvidence(
                frame_id=frame_id, row=r, col_start=c_start, col_end=c_end,
                x1=x1, y1=y1, x2=x2, y2=y2,
                approx_range_m=approx_range_m,
                approx_width_m=(x2 - x1) * approx_range_m / focal_length_px,
                approx_height_m=(y2 - y1) * approx_range_m / focal_length_px,
                support_fraction=support_fraction,
                at_image_boundary=(not left_is_real) or (not right_is_real),
            ))

    return evidence
