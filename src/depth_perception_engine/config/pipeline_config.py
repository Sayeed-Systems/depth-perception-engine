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

    # --- geometry (Level 3, Phase E3) ---
    # Gates DepthPerceptionPipeline.process()'s camera-optical-frame
    # PointCloud stage (geometry.PointCloudBuilder — see
    # docs/LEVEL3_ARCHITECTURE.md). Default is explicitly False: geometry
    # is additional compute cost (see docs/VALIDATION_REPORT.md's E3
    # benchmark) that every existing caller — including mp01_perception,
    # which does not know this field exists — must not start paying for
    # or receiving without opting in. False also reproduces pre-E3
    # behavior exactly: DepthPerceptionResult.geometry stays None, and no
    # PointCloudBuilder is even constructed. No __post_init__ check is
    # added below for this field: it is a plain bool with no invalid
    # *value* to reject (unlike e.g. block_size, which has a real
    # constraint), so a value check would be validation theater.
    enable_geometry: bool = False

    # --- E5 spatial evidence (ObstacleCloud / FreeSpaceRays) ---
    # Both default False — additional compute cost on top of enable_geometry,
    # same "opt-in, existing callers unaffected" discipline as
    # enable_geometry itself. Neither has any effect unless enable_geometry
    # is also True AND a body_T_camera_left extrinsic was supplied to the
    # pipeline (E5 operates on the E4 body-frame cloud only — see
    # docs/LEVEL3_ARCHITECTURE.md's E5 update); with no body-frame cloud
    # there is nothing for either to filter/cast rays from.
    enable_obstacle_geometry: bool = False
    enable_free_space_rays: bool = False

    # Range window (Euclidean distance from the camera's own origin, not
    # the body origin — matches ObstacleCloud.distances_m's frozen
    # docstring) an ObstacleCloud point must fall within. Defaults are
    # deliberately unbounded ("no additional restriction beyond what the
    # sensor itself already enforces upstream" — every valid point is
    # already clamped to [DepthEstimator.MIN_DEPTH_M, MAX_DEPTH_M] in
    # camera-frame depth before this stage ever runs), not a new
    # MP01-specific or hardware-specific number.
    obstacle_min_range_m: float = 0.0
    obstacle_max_range_m: float = float("inf")

    # 2D grid decimation applied identically by both build_obstacle_cloud
    # and build_free_space_rays (see their shared `stride` semantics) —
    # one shared knob rather than two independent ones, since both
    # producers read the same source PointCloud and "keep every Nth
    # pixel" means the same thing for either. 1 = no downsampling
    # (deterministic default — every pixel considered, same as every
    # other geometry stage's default behavior).
    geometry_sampling_stride: int = 1

    # --- E6 geometry quality classification thresholds ---
    # Consumed by geometry.classify_geometry_quality(), an opt-in helper
    # (not auto-invoked by process() — see docs/IMPLEMENTATION_STATUS.md's
    # E6 addendum for why) that maps GeometryMetrics.valid_fraction — a
    # single, already-precisely-defined metric, not a new blended score —
    # onto one of GeometryQuality.HEALTHY/DEGRADED/NO_USABLE_GEOMETRY.
    # These two fractions are a POLICY choice, not a physical constant:
    # there is no universally correct "enough valid geometry" threshold —
    # it depends on sensor, scene, and how the caller intends to use the
    # geometry. The defaults below are conservative, undocumented-against-
    # any-real-dataset placeholders, not tuned/validated values — a real
    # deployment is expected to override them for its own sensor/scene.
    # valid_fraction >= geometry_healthy_min_valid_fraction        -> HEALTHY
    # geometry_degraded_min_valid_fraction <= valid_fraction < ... -> DEGRADED
    # valid_fraction < geometry_degraded_min_valid_fraction        -> NO_USABLE_GEOMETRY
    geometry_healthy_min_valid_fraction: float = 0.5
    geometry_degraded_min_valid_fraction: float = 0.05

    def resolved_obstacle_dead_zone_px(self) -> int:
        """obstacle_dead_zone_px if set, else num_disparities."""
        return (
            self.obstacle_dead_zone_px
            if self.obstacle_dead_zone_px is not None
            else self.num_disparities
        )

    def __post_init__(self) -> None:
        """Validate at construction time, not three layers deeper inside
        DisparityEngine/ThreatAssessor/SceneInterpreter — mirrors the
        validation pattern StereoCalibration and RegionAnalyzer already use
        elsewhere in this codebase."""
        if self.num_disparities % 16 != 0:
            raise ValueError(
                f"num_disparities ({self.num_disparities}) must be divisible by 16."
            )
        if self.block_size % 2 == 0 or self.block_size < 1:
            raise ValueError(
                f"block_size ({self.block_size}) must be a positive odd integer."
            )
        if self.caution_distance_m <= 0.0:
            raise ValueError(
                f"caution_distance_m ({self.caution_distance_m}) must be positive."
            )
        if not (self.caution_distance_m < self.clear_distance_m):
            raise ValueError(
                "Require caution_distance_m < clear_distance_m, got "
                f"caution_distance_m={self.caution_distance_m}, "
                f"clear_distance_m={self.clear_distance_m}."
            )
        if self.n_beams < 1:
            raise ValueError(f"n_beams ({self.n_beams}) must be at least 1.")
        if not (0 <= self.obstacle_percentile <= 100):
            raise ValueError(
                f"obstacle_percentile ({self.obstacle_percentile}) must be in [0, 100]."
            )
        if self.obstacle_min_valid_px < 0:
            raise ValueError(
                f"obstacle_min_valid_px ({self.obstacle_min_valid_px}) must be >= 0."
            )
        if not (0.0 <= self.obstacle_blocked_invalid_ratio <= 1.0):
            raise ValueError(
                "obstacle_blocked_invalid_ratio "
                f"({self.obstacle_blocked_invalid_ratio}) must be in [0, 1]."
            )
        if not (0.0 < self.obstacle_ema_alpha <= 1.0):
            raise ValueError(
                f"obstacle_ema_alpha ({self.obstacle_ema_alpha}) must be in (0, 1]."
            )
        if self.obstacle_debounce_frames < 1:
            raise ValueError(
                f"obstacle_debounce_frames ({self.obstacle_debounce_frames}) must be at least 1."
            )
        if self.obstacle_dead_zone_px is not None and self.obstacle_dead_zone_px < 0:
            raise ValueError(
                f"obstacle_dead_zone_px ({self.obstacle_dead_zone_px}) must be >= 0 when set."
            )
        if self.traversability_grid_rows < 1 or self.traversability_grid_cols < 1:
            raise ValueError(
                "traversability_grid_rows/cols must both be at least 1, got "
                f"rows={self.traversability_grid_rows}, cols={self.traversability_grid_cols}."
            )
        if not (0.0 <= self.traversability_ambiguous_fraction_thresh <= 1.0):
            raise ValueError(
                "traversability_ambiguous_fraction_thresh "
                f"({self.traversability_ambiguous_fraction_thresh}) must be in [0, 1]."
            )
        if self.obstacle_min_range_m < 0.0:
            raise ValueError(
                f"obstacle_min_range_m ({self.obstacle_min_range_m}) must be >= 0."
            )
        if not (self.obstacle_min_range_m <= self.obstacle_max_range_m):
            raise ValueError(
                "Require obstacle_min_range_m <= obstacle_max_range_m, got "
                f"obstacle_min_range_m={self.obstacle_min_range_m}, "
                f"obstacle_max_range_m={self.obstacle_max_range_m}."
            )
        if self.geometry_sampling_stride < 1:
            raise ValueError(
                f"geometry_sampling_stride ({self.geometry_sampling_stride}) must be >= 1."
            )
        if not (0.0 <= self.geometry_degraded_min_valid_fraction <= 1.0):
            raise ValueError(
                "geometry_degraded_min_valid_fraction "
                f"({self.geometry_degraded_min_valid_fraction}) must be in [0, 1]."
            )
        if not (0.0 <= self.geometry_healthy_min_valid_fraction <= 1.0):
            raise ValueError(
                "geometry_healthy_min_valid_fraction "
                f"({self.geometry_healthy_min_valid_fraction}) must be in [0, 1]."
            )
        if not (self.geometry_degraded_min_valid_fraction <= self.geometry_healthy_min_valid_fraction):
            raise ValueError(
                "Require geometry_degraded_min_valid_fraction <= geometry_healthy_min_valid_fraction, got "
                f"geometry_degraded_min_valid_fraction={self.geometry_degraded_min_valid_fraction}, "
                f"geometry_healthy_min_valid_fraction={self.geometry_healthy_min_valid_fraction}."
            )
