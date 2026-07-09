"""
Scene interpreter — partitions a frame into a configurable grid, runs
RegionAnalyzer over each cell, and derives one global navigation decision
from the resulting region classifications.

This sits after depth estimation and before any flight-control layer:

    DepthEstimator -> SceneInterpreter -> (downstream planning)

It is intentionally extensible: object-detection or semantic perception
results could later be merged into the same RegionStats/RegionClass model
(e.g. by overriding a region's classification when a detector reports an
object inside it) without changing this module's grid/decision structure.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from depth_perception_engine.traversability.region_analyzer import RegionAnalyzer
from depth_perception_engine.traversability.types import (
    NavigationDecision,
    RegionClass,
    RegionStats,
)

logger = logging.getLogger(__name__)

SceneState = Dict[str, RegionStats]

_AMBIGUOUS_CLASSES = (RegionClass.LOW_CONFIDENCE, RegionClass.LOW_TEXTURE_UNKNOWN)
_BLOCKED_CLASSES   = (RegionClass.OBSTACLE, RegionClass.PROBABLE_WALL)


class SceneInterpreter:
    """Grid-based traversability layer producing a global navigation decision."""

    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        region_analyzer: Optional[RegionAnalyzer] = None,
        caution_m: float = 0.60,
        clear_m: float = 1.20,
        ambiguous_fraction_thresh: float = 0.50,
    ) -> None:
        """
        Args:
            rows, cols:                grid dimensions (default 3x3)
            region_analyzer:           injected analyzer; built from caution_m/clear_m if omitted
            caution_m:                 forward-cell distance below which to slow down
            clear_m:                   forward-cell distance above which full speed is safe
            ambiguous_fraction_thresh: if at least this fraction of all regions are
                                       LOW_CONFIDENCE/LOW_TEXTURE_UNKNOWN, the scene as a
                                       whole is too uncertain to trust any single
                                       direction -> ROTATE_AND_SCAN

        Raises:
            ValueError: If rows/cols are not positive, or thresholds are out of range.
        """
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must both be >= 1.")
        if not (0.0 < caution_m < clear_m):
            raise ValueError("Require 0 < caution_m < clear_m.")
        if not (0.0 <= ambiguous_fraction_thresh <= 1.0):
            raise ValueError("ambiguous_fraction_thresh must be in [0, 1].")

        self._rows = rows
        self._cols = cols
        self._caution_m = caution_m
        self._clear_m = clear_m
        self._ambiguous_fraction_thresh = ambiguous_fraction_thresh
        self._analyzer = region_analyzer or RegionAnalyzer(
            obstacle_caution_m=caution_m, obstacle_clear_m=clear_m
        )
        self._region_names = self._build_region_names(rows, cols)

        logger.info("SceneInterpreter initialised — grid=%dx%d", rows, cols)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_region_names(rows: int, cols: int) -> List[List[str]]:
        """Use TL/TC/TR.../BL/BC/BR naming for the canonical 3x3 grid; RxCy otherwise."""
        if rows == 3 and cols == 3:
            return [
                ["TL", "TC", "TR"],
                ["ML", "MC", "MR"],
                ["BL", "BC", "BR"],
            ]
        return [[f"R{r}C{c}" for c in range(cols)] for r in range(rows)]

    # ------------------------------------------------------------------
    def analyze(
        self,
        gray: np.ndarray,
        raw_disp: np.ndarray,
        depth_map: np.ndarray,
    ) -> SceneState:
        """Partition the frame and analyze every region.

        Args:
            gray:      grayscale rectified left frame (H, W)
            raw_disp:  raw disparity map, same (H, W); invalid <= 0
            depth_map: metric depth map, same (H, W); invalid == 0

        Raises:
            ValueError: If the three inputs don't share the same spatial shape.
        """
        shape = gray.shape[:2]
        if raw_disp.shape[:2] != shape or depth_map.shape[:2] != shape:
            raise ValueError(
                f"gray/raw_disp/depth_map must share shape: gray={shape}, "
                f"raw_disp={raw_disp.shape[:2]}, depth_map={depth_map.shape[:2]}"
            )

        h, w = shape
        row_bounds = np.linspace(0, h, self._rows + 1).astype(int)
        col_bounds = np.linspace(0, w, self._cols + 1).astype(int)

        scene: SceneState = {}
        for r in range(self._rows):
            y1, y2 = int(row_bounds[r]), int(row_bounds[r + 1])
            for c in range(self._cols):
                x1, x2 = int(col_bounds[c]), int(col_bounds[c + 1])
                name = self._region_names[r][c]
                scene[name] = self._analyzer.analyze(
                    name, r, c, x1, y1, x2, y2,
                    gray[y1:y2, x1:x2],
                    raw_disp[y1:y2, x1:x2],
                    depth_map[y1:y2, x1:x2],
                )

        return scene

    # ------------------------------------------------------------------
    def decide_navigation(self, scene: SceneState) -> NavigationDecision:
        """Derive one global recommendation from the full region grid.

        Priority order (first match wins):
            1. ROTATE_AND_SCAN — too much of the scene is ambiguous to trust
            2. forward blocked — STOP, or TURN_LEFT/TURN_RIGHT toward whichever
               side is clear (preferring the side with more depth headroom
               if both are clear)
            3. forward ambiguous — SLOW_DOWN (not confirmed blocked, but not
               confirmed clear either)
            4. forward clear but within caution range — SLOW_DOWN
            5. forward clear and beyond clear_m — MOVE_FORWARD
            6. fallback — HOVER
        """
        if not scene:
            return NavigationDecision.HOVER

        ambiguous_count = sum(1 for r in scene.values() if r.classification in _AMBIGUOUS_CLASSES)
        if ambiguous_count / len(scene) >= self._ambiguous_fraction_thresh:
            return NavigationDecision.ROTATE_AND_SCAN

        forward, left, right = self._forward_left_right(scene)
        if forward is None:
            return NavigationDecision.HOVER

        if forward.classification in _BLOCKED_CLASSES:
            return self._turn_decision(left, right)

        if forward.classification in _AMBIGUOUS_CLASSES:
            return NavigationDecision.SLOW_DOWN

        # forward.classification == CLEAR
        if forward.depth_avg_m > 0.0 and forward.depth_avg_m < self._clear_m:
            return NavigationDecision.SLOW_DOWN
        if forward.depth_avg_m >= self._clear_m:
            return NavigationDecision.MOVE_FORWARD

        return NavigationDecision.HOVER

    # ------------------------------------------------------------------
    def _forward_left_right(
        self, scene: SceneState
    ) -> Tuple[Optional[RegionStats], Optional[RegionStats], Optional[RegionStats]]:
        mid_r, mid_c = self._rows // 2, self._cols // 2
        forward = scene.get(self._region_names[mid_r][mid_c])
        left  = scene.get(self._region_names[mid_r][mid_c - 1]) if mid_c - 1 >= 0 else None
        right = scene.get(self._region_names[mid_r][mid_c + 1]) if mid_c + 1 < self._cols else None
        return forward, left, right

    @staticmethod
    def _turn_decision(
        left: Optional[RegionStats], right: Optional[RegionStats]
    ) -> NavigationDecision:
        left_clear  = left  is not None and left.classification == RegionClass.CLEAR
        right_clear = right is not None and right.classification == RegionClass.CLEAR

        if left_clear and not right_clear:
            return NavigationDecision.TURN_LEFT
        if right_clear and not left_clear:
            return NavigationDecision.TURN_RIGHT
        if left_clear and right_clear:
            return (
                NavigationDecision.TURN_LEFT
                if left.depth_avg_m >= right.depth_avg_m
                else NavigationDecision.TURN_RIGHT
            )
        return NavigationDecision.STOP

    # ------------------------------------------------------------------
    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def __repr__(self) -> str:
        return f"SceneInterpreter(grid={self._rows}x{self._cols})"
