"""
Geometric boundaries and discontinuities — Phase D5.

Deterministic, bounded detection of where observed physical geometry
changes significantly or terminates — purely geometric evidence, never a
semantic edge class ("curb"/"shoreline"/"road edge"/"wall edge"), never a
behavioral judgement ("obstacle to avoid"/"safe"/"traversable"), and never
an inferred opening (two adjacent boundaries do not, by themselves, imply
a passage — that reasoning is explicitly out of scope, reserved for a
future phase). No "road"/"ground"/"wall"/"landing"/"traversable" concept
appears anywhere in this module.

Two independent, purely geometric signals are evaluated per adjacent cell
pair:

    depth discontinuity: |median_depth(cell_A) - median_depth(cell_B)|
        exceeds depth_step_threshold_m. Reuses only depth_map (a Level 0
        output, always present regardless of PipelineConfig.enable_geometry)
        — no disparity/depth recomputation, no PointCloud dependency.

    surface-orientation discontinuity: the angle between cell_A's and
        cell_B's already-computed geometry.surface.SurfaceEvidence.normal
        exceeds orientation_change_threshold_rad. OPPORTUNISTIC only: used
        when SurfaceEvidence is available AND its own grid happens to have
        the exact same row/col dimensions as this stage's own grid — see
        "Grid independence" below. Reuses SurfaceEvidence's already-computed
        normal directly; no new plane fit, no recomputation.

Either signal alone is sufficient to report OBSERVED_DISCONTINUITY — this
directly implements the D5 task's own explicit requirement that a
surface-orientation change CAN produce supported discontinuity evidence
even where the metric depth step is small (e.g. a corner between two
surfaces at the same distance but different orientation).

Grid independence: this stage partitions depth_map into its OWN
independently-configured grid_rows x grid_cols grid — reimplemented (same
np.linspace boundary convention SceneInterpreter/geometry.surface already
use), not shared with, imported from, or forced onto RegionEvidence's or
SurfaceEvidence's own grids. SurfaceEvidence's normals are consulted only
when the caller's surface_grid_rows/surface_grid_cols happen to exactly
match this stage's own grid_rows/grid_cols (both default to 3x3, so they
align out of the box without DPE forcing anything) — when they don't
match, orientation_change_rad simply stays None for every cell pair,
falling back to depth-only evidence. Neither RegionEvidence nor
SurfaceEvidence is modified by this module.

Invalid vs. boundary — the core safety rule: A MISSING DEPTH MEASUREMENT
MUST NOT AUTOMATICALLY BECOME A PHYSICAL BOUNDARY. If EITHER cell in a
pair has fewer than min_support_count valid depth pixels, the pair's
state is INSUFFICIENT_EVIDENCE — depth_step_m and orientation_change_rad
both stay None — regardless of how different the two cells' available
evidence might otherwise look. This also covers the "observed geometry
termination" case named in the D5 design brief: a cell whose neighbor has
NO valid depth at all (rather than merely a large depth step) cannot be
told apart, from evidence alone, between "a real object edge where depth
data genuinely stops" and "a stereo-matching failure (occlusion,
textureless surface)" — DPE has no way to distinguish these, so it
reports the honest state (INSUFFICIENT_EVIDENCE), not a fabricated
"termination" claim dressed up as a third state. `support_fraction_from`/
`support_fraction_to` remain populated in every case (0.0 is a
legitimate, informative value, not "no data") so a caller can see exactly
why a pair was insufficient.

Coordinate/spatial contract: every BoundaryEvidence carries `frame_id`
(copied directly from the caller, generic — this module has no built-in
BODY/CAMERA_OPTICAL_LEFT opinion, matching geometry.surface's own
genericity; in this pipeline's actual usage it is always
FrameId.CAMERA_OPTICAL_LEFT, since depth_map is a camera-optical-frame
raster) and `x1`/`y1`/`x2`/`y2` — the union (bounding box) of the two
adjacent cells' own pixel bounds, a pixel-space region a caller can use to
locate the boundary in disparity_map/depth_map/the source image without
reaching into DPE internals.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from depth_perception_engine.geometry.surface import SurfaceEvidence


class BoundaryState:
    """Plain string constants for BoundaryEvidence.state — not a closed
    Enum, matching this codebase's ThreatAssessor/GeometryQuality
    precedent for lightweight categorical state."""

    OBSERVED_DISCONTINUITY = "OBSERVED_DISCONTINUITY"
    """A supported geometric change: both adjacent cells had sufficient
    depth evidence, and their metric depth step and/or surface-orientation
    change exceeded the configured threshold."""

    NO_DISCONTINUITY = "NO_DISCONTINUITY"
    """Both adjacent cells had sufficient depth evidence, and neither
    signal exceeded its threshold — a supported finding of continuity,
    not an absence of evaluation."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    """Either adjacent cell had fewer than min_support_count valid depth
    pixels — no comparison was possible. Never reinterpreted as a
    boundary, and never reinterpreted as confirmed continuity either."""


class BoundaryDirection:
    """Plain string constants identifying which neighboring cell a
    BoundaryEvidence evaluates against, relative to its own (row, col)."""

    RIGHT = "RIGHT"
    """Compared against the cell at (row, col + 1)."""

    DOWN = "DOWN"
    """Compared against the cell at (row + 1, col)."""


@dataclass(frozen=True, slots=True)
class BoundaryEvidence:
    """One evaluated adjacency between two neighboring cells of the
    boundary grid — Phase D5. See this module's own docstring for the
    full discontinuity-definition, invalid-vs-boundary, and coordinate
    contract.
    """

    frame_id: str
    row: int
    col: int
    direction: str

    x1: int
    y1: int
    x2: int
    y2: int

    support_fraction_from: float
    support_fraction_to: float

    state: str
    depth_step_m: Optional[float]
    orientation_change_rad: Optional[float]


def build_boundary_evidence(
    depth_map: np.ndarray,
    frame_id: str,
    grid_rows: int,
    grid_cols: int,
    min_support_count: int,
    depth_step_threshold_m: float,
    orientation_change_threshold_rad: float,
    surface_evidence: Optional[List[SurfaceEvidence]] = None,
    surface_grid_rows: Optional[int] = None,
    surface_grid_cols: Optional[int] = None,
    reliability_mask: Optional[np.ndarray] = None,
    min_confirmation_support_fraction: float = 0.0,
) -> List[BoundaryEvidence]:
    """Evaluate depth/surface-orientation discontinuities between every
    pair of horizontally/vertically adjacent cells of a grid_rows x
    grid_cols partition of depth_map.

    Args:
        depth_map: metric depth map, (H, W), metres; invalid == 0.0 — the
            same universal invalid-value convention DisparityEngine/
            DepthEstimator/RegionAnalyzer/build_result already use (see
            docs/DATA_CONTRACTS.md). Validity is derived here as
            `depth_map > 0`, not passed in separately — no new convention.
        frame_id: copied directly onto every returned BoundaryEvidence;
            see module docstring.
        grid_rows, grid_cols: this stage's own grid dimensions, both >= 1
            — independent of RegionEvidence's/SurfaceEvidence's own grids.
        min_support_count: minimum valid depth pixels a cell must contain
            to be compared at all; must be >= 1.
        depth_step_threshold_m: minimum |median depth difference| between
            two adjacent cells to report OBSERVED_DISCONTINUITY.
        orientation_change_threshold_rad: minimum angle (radians) between
            two adjacent cells' SurfaceEvidence.normal to report
            OBSERVED_DISCONTINUITY via the orientation signal.
        surface_evidence: this frame's already-computed SurfaceEvidence
            list (row-major order), or None if not available this frame.
        surface_grid_rows, surface_grid_cols: the grid `surface_evidence`
            was built with. Orientation evidence is only consulted when
            these exactly equal grid_rows/grid_cols — see "Grid
            independence" in the module docstring.
        reliability_mask: Optional (H, W) bool, same grid as `depth_map`
            — Phase I3, see geometry.reliability's module docstring.
            Where True, a pixel is excluded from its cell's own
            support_count/support_fraction/median_depth_m — a confirmed
            occlusion-shadow pixel must not inflate a cell's reported
            support or bias its median toward fabricated content. This
            does not redefine `depth_map > 0.0` validity itself, only
            what counts as SUPPORT for this stage's own per-cell
            aggregation. None (default) preserves this function's exact
            pre-I3 behavior.
        min_confirmation_support_fraction: Phase I4 (see
            benchmarks/i4_boundary_precision/ — 60+ decorrelated-noise
            seeds measured a false-confirmed-pair support_fraction
            ceiling of ~0.027; 126 genuine-boundary cases across strong/
            weak/distant/narrow/occluded/weak-texture fixtures measured a
            floor of ~0.44 — a >16x gap, zero overlap). Either cell's
            support_fraction below this value routes the pair to
            INSUFFICIENT_EVIDENCE regardless of depth_step/orientation —
            "enough valid pixels to compute a median" (the pre-existing
            min_support_count floor) is not the same claim as "enough to
            trust a POSITIVE discontinuity finding from it." Default 0.0
            preserves this function's exact pre-I4 behavior for any
            caller not passing it; pipeline.py supplies
            PipelineConfig.boundary_min_confirmation_support_fraction
            (default 0.10) for the real pipeline.
    Returns:
        A flat list covering every RIGHT edge (grid_rows * (grid_cols - 1)
        of them) followed by every DOWN edge (grid_cols * (grid_rows - 1)
        of them), in row-major cell order.

    Raises:
        ValueError: If grid_rows/grid_cols < 1, min_support_count < 1, or
            depth_step_threshold_m <= 0.
    """
    if grid_rows < 1 or grid_cols < 1:
        raise ValueError(
            f"grid_rows/grid_cols must both be >= 1, got rows={grid_rows}, cols={grid_cols}."
        )
    if min_support_count < 1:
        raise ValueError(f"min_support_count ({min_support_count}) must be >= 1.")
    if depth_step_threshold_m <= 0.0:
        raise ValueError(f"depth_step_threshold_m ({depth_step_threshold_m}) must be > 0.")

    valid_depth_mask = depth_map > 0.0
    if reliability_mask is not None:
        valid_depth_mask = valid_depth_mask & ~reliability_mask

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

    surface_grid_matches = (
        surface_evidence is not None
        and surface_grid_rows == grid_rows
        and surface_grid_cols == grid_cols
    )

    def normal_at(row: int, col: int):
        if not surface_grid_matches:
            return None
        return surface_evidence[row * grid_cols + col].normal

    evidence: List[BoundaryEvidence] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            neighbors = []
            if c + 1 < grid_cols:
                neighbors.append((BoundaryDirection.RIGHT, r, c + 1))
            if r + 1 < grid_rows:
                neighbors.append((BoundaryDirection.DOWN, r + 1, c))

            for direction, nr, nc in neighbors:
                a, b = cells[r][c], cells[nr][nc]
                x1, y1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
                x2, y2 = max(a["x2"], b["x2"]), max(a["y2"], b["y2"])

                # Phase I4: a fractional-support floor, IN ADDITION TO the
                # absolute min_support_count check just below — "enough
                # pixels to compute a median at all" is a weaker claim
                # than "enough of the cell is genuinely explained to
                # trust a POSITIVE discontinuity finding from it." See
                # this function's own docstring and
                # benchmarks/i4_boundary_precision/ for the measured
                # separation this threshold is calibrated against.
                insufficient = (
                    a["median_depth_m"] is None or b["median_depth_m"] is None
                    or a["support_fraction"] < min_confirmation_support_fraction
                    or b["support_fraction"] < min_confirmation_support_fraction
                )
                if insufficient:
                    evidence.append(BoundaryEvidence(
                        frame_id=frame_id, row=r, col=c, direction=direction,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        support_fraction_from=a["support_fraction"],
                        support_fraction_to=b["support_fraction"],
                        state=BoundaryState.INSUFFICIENT_EVIDENCE,
                        depth_step_m=None, orientation_change_rad=None,
                    ))
                    continue

                depth_step_m = abs(a["median_depth_m"] - b["median_depth_m"])

                orientation_change_rad = None
                normal_a, normal_b = normal_at(r, c), normal_at(nr, nc)
                if normal_a is not None and normal_b is not None:
                    cos_angle = float(np.clip(np.dot(normal_a, normal_b), -1.0, 1.0))
                    orientation_change_rad = float(np.arccos(cos_angle))

                is_discontinuous = depth_step_m >= depth_step_threshold_m or (
                    orientation_change_rad is not None
                    and orientation_change_rad >= orientation_change_threshold_rad
                )
                state = (
                    BoundaryState.OBSERVED_DISCONTINUITY if is_discontinuous
                    else BoundaryState.NO_DISCONTINUITY
                )

                evidence.append(BoundaryEvidence(
                    frame_id=frame_id, row=r, col=c, direction=direction,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    support_fraction_from=a["support_fraction"],
                    support_fraction_to=b["support_fraction"],
                    state=state, depth_step_m=depth_step_m,
                    orientation_change_rad=orientation_change_rad,
                ))

    return evidence
