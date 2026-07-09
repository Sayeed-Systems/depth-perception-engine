"""
Pipeline configuration.

A single, plain, typed dataclass bundling every tunable threshold the
pipeline needs — the same values that used to live as loose module-level
constants in the original standalone main.py. Defaults here match that
file's tuned values exactly, so building a pipeline with PipelineConfig()
reproduces the original desk-tested behavior.

Deliberately excluded: anything specific to flight-velocity planning
(corridor fraction, max forward speed, yaw rate, creep speed). Those drive
examples/navigation/velocity_planner.py, which is a downstream, demo-only
consumer of this engine's output — not part of the Depth Perception Engine
itself. See docs/INTEGRATION_READINESS.md.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PipelineConfig:
    # --- stereo disparity (StereoSGBM) ---
    min_disparity: int = 0
    num_disparities: int = 128       # must be divisible by 16; ~0.31m-10m @ 64mm baseline
    block_size: int = 13             # must be odd; larger = smoother disparity

    # --- shared distance thresholds (obstacles + traversability) ---
    caution_distance_m: float = 0.60   # nearer than this: BLOCKED / OBSTACLE
    clear_distance_m: float = 1.20     # farther than this: CLEAR

    # --- obstacles (per-beam threat assessment) ---
    n_beams: int = 20
    obstacle_percentile: int = 15
    obstacle_min_valid_px: int = 5
    obstacle_blocked_invalid_ratio: float = 0.85
    obstacle_ema_alpha: float = 0.30
    obstacle_debounce_frames: int = 3
    # None => resolved to num_disparities at pipeline construction time,
    # matching the original main.py's ThreatAssessor(dead_zone_px=NUM_DISPARITIES).
    obstacle_dead_zone_px: Optional[int] = None

    # --- traversability (region grid) ---
    traversability_grid_rows: int = 3
    traversability_grid_cols: int = 3
    traversability_ambiguous_fraction_thresh: float = 0.50

    def resolved_obstacle_dead_zone_px(self) -> int:
        """obstacle_dead_zone_px if set, else num_disparities."""
        return (
            self.obstacle_dead_zone_px
            if self.obstacle_dead_zone_px is not None
            else self.num_disparities
        )
