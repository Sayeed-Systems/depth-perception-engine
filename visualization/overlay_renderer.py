"""
Overlay renderer module.

Responsibilities: take a BGR camera frame, a depth map, and a distance
measurement, then draw informational overlays (distance badge, depth colour
map thumbnail, centre-ROI crosshair, FPS counter, beam profile strip) and
return the annotated frame.

Does NOT perform camera acquisition, frame splitting, rectification,
disparity computation, depth estimation, or object detection.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from perception.region_types import NavigationDecision, RegionClass, RegionStats

logger = logging.getLogger(__name__)

# Colour constants (BGR)
_GREEN  = (0,   220,   0)
_YELLOW = (0,   220, 220)
_RED    = (0,    60, 220)
_GREY   = (80,   80,  80)
_WHITE  = (255, 255, 255)
_BLACK  = (0,     0,   0)
_CYAN   = (220, 220,   0)

_STATUS_COLOR = {
    "CLEAR":   _GREEN,
    "CAUTION": _YELLOW,
    "BLOCKED": _RED,
    "NO_DATA": _GREY,
}

_REGION_CLASS_COLOR = {
    RegionClass.CLEAR:               _GREEN,
    RegionClass.OBSTACLE:            _RED,
    RegionClass.PROBABLE_WALL:       _RED,
    RegionClass.LOW_TEXTURE_UNKNOWN: _YELLOW,
    RegionClass.LOW_CONFIDENCE:      _YELLOW,
}

_PROFILE_H   = 28   # height of the beam profile strip in pixels
_PROFILE_MAXD = 5.0  # distance at which bar height = 0 (fully clear)


class OverlayRenderer:
    """Renders depth and distance overlays onto a BGR camera frame."""

    _CROSS_HALF: int = 12

    def __init__(
        self,
        roi_width:         int   = 50,
        roi_height:        int   = 50,
        depth_thumb_scale: float = 0.35,
    ) -> None:
        if roi_width < 1 or roi_height < 1:
            raise ValueError("roi_width and roi_height must be >= 1.")
        if not (0.0 <= depth_thumb_scale <= 1.0):
            raise ValueError("depth_thumb_scale must be in [0, 1].")

        self._roi_w       = roi_width
        self._roi_h       = roi_height
        self._thumb_scale = depth_thumb_scale

        logger.info(
            "OverlayRenderer initialised — roi=%dx%d, thumb_scale=%.2f",
            roi_width, roi_height, depth_thumb_scale,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def render(
        self,
        frame:       np.ndarray,
        depth_map:   Optional[np.ndarray],
        measurement: Optional[Dict[str, float]],
        fps:         float = 0.0,
        beams:       Optional[List[Dict]] = None,
        nav:         Optional[Dict] = None,
    ) -> np.ndarray:
        """Draw overlays on *frame* and return the annotated copy."""
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Depth thumbnail (top-right corner)
        if depth_map is not None and self._thumb_scale > 0.0:
            out = self._draw_depth_thumb(out, depth_map)

        # 2. Localized obstacle zones — red overlay only over the beam's
        # own x-range, so the mark sits on the actual obstacle, not the
        # whole frame.
        if beams:
            self._draw_obstacle_zones(out, beams)

        # 2b. Forward flight corridor guide + live clear/blocked readout
        if nav:
            self._draw_corridor(out, nav)

        # 3. Centre ROI crosshair + box
        cx, cy = w // 2, h // 2
        rx1 = max(0, cx - self._roi_w // 2)
        ry1 = max(0, cy - self._roi_h // 2)
        rx2 = min(w, rx1 + self._roi_w)
        ry2 = min(h, ry1 + self._roi_h)
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), _GREEN, 1, cv2.LINE_AA)
        ch = self._CROSS_HALF
        cv2.line(out, (cx - ch, cy), (cx + ch, cy), _GREEN, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy - ch), (cx, cy + ch), _GREEN, 1, cv2.LINE_AA)

        # 4. Centre distance badge (above profile strip)
        badge_y = h - _PROFILE_H - 18
        if measurement is not None:
            dist_m  = measurement.get("distance_meters", 0.0)
            dist_cm = measurement.get("distance_centimeters", 0.0)
            if dist_m > 0.0:
                self._draw_badge(out, f"{dist_m:.2f} m  ({dist_cm:.0f} cm)",
                                 (w // 2, badge_y), self._distance_color(dist_m))
            else:
                self._draw_badge(out, "-- no depth --", (w // 2, badge_y), _RED)

        # 5. Beam profile strip (bottom of frame)
        if beams:
            self._draw_profile(out, beams)

        # 6. FPS badge (top-left)
        if fps > 0.0:
            cv2.putText(out, f"FPS {fps:.1f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, _CYAN, 2, cv2.LINE_AA)

        return out

    def render_scene_grid(
        self,
        frame:    np.ndarray,
        scene:    Dict[str, RegionStats],
        decision: NavigationDecision,
        fps:      float = 0.0,
    ) -> np.ndarray:
        """Draw the perception-fusion grid: per-region classification, confidence,
        depth, texture, invalid ratio, and the resulting global navigation decision.

        This is a distinct, denser view from render() — toggled separately so it
        doesn't compete visually with the corridor/beam display during normal flight.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        if not scene:
            return out

        rows = max(r.row for r in scene.values()) + 1
        cols = max(r.col for r in scene.values()) + 1

        for r in range(1, rows):
            y = int(h * r / rows)
            cv2.line(out, (0, y), (w, y), _GREY, 1, cv2.LINE_AA)
        for c in range(1, cols):
            x = int(w * c / cols)
            cv2.line(out, (x, 0), (x, h), _GREY, 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for name, region in scene.items():
            color = _REGION_CLASS_COLOR.get(region.classification, _GREY)
            cv2.rectangle(out, (region.x1, region.y1), (region.x2 - 1, region.y2 - 1),
                          color, 1, cv2.LINE_AA)

            lines = [
                f"{name} {region.classification.value}",
                f"d={region.depth_avg_m:.2f}m c={region.confidence:.2f}",
                f"inv={region.invalid_pct:.0f}% {region.texture_class.value}",
            ]
            ty = region.y1 + 13
            for line in lines:
                cv2.putText(out, line, (region.x1 + 3, ty), font, 0.32, color, 1, cv2.LINE_AA)
                ty += 12

        cv2.rectangle(out, (0, 0), (w, 22), _BLACK, -1)
        cv2.putText(out, f"NAV: {decision.value}", (8, 16), font, 0.55, _CYAN, 1, cv2.LINE_AA)
        if fps > 0.0:
            (tw, _), _ = cv2.getTextSize(f"FPS {fps:.1f}", font, 0.5, 1)
            cv2.putText(out, f"FPS {fps:.1f}", (w - tw - 8, 16), font, 0.5, _CYAN, 1, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_corridor(frame: np.ndarray, nav: Dict) -> None:
        """Outline the forward flight corridor and show its clear/blocked status.

        The corridor is the beam range the navigation planner actually
        gates forward speed on (not the whole frame), so this draws dashed
        vertical guides at its edges plus a status badge — distinct from
        the per-beam obstacle zones, which mark exactly where blockage is.
        """
        h, w = frame.shape[:2]
        x1 = max(0, min(nav.get("corridor_x1", 0), w - 1))
        x2 = max(0, min(nav.get("corridor_x2", 0), w - 1))
        status = nav.get("status", "NO_DATA")
        color  = _STATUS_COLOR.get(status, _GREY)

        for x in (x1, x2):
            for y in range(0, h, 14):
                cv2.line(frame, (x, y), (x, min(h, y + 7)), _CYAN, 1, cv2.LINE_AA)

        ahead_m = nav.get("ahead_distance_m", 0.0)
        if status == "BLOCKED":
            text = "AHEAD: BLOCKED"
        elif ahead_m > 0.0:
            text = f"AHEAD: {ahead_m:.2f} m [{status}]"
        else:
            text = f"AHEAD: clear [{status}]"

        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
        tx = max(0, (x1 + x2) // 2 - tw // 2)
        ty = th + 10
        cv2.rectangle(frame, (tx - 6, ty - th - 6), (tx + tw + 6, ty + bl + 4), _BLACK, -1)
        cv2.putText(frame, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_obstacle_zones(frame: np.ndarray, beams: List[Dict]) -> None:
        """Highlight only the beam columns reporting BLOCKED, with a distance label.

        Unlike a full-frame border, this marks just the obstacle's own
        x-range so the alert is localized to where the object actually is.
        """
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.5
        thickness = 1
        h         = frame.shape[0]

        for b in beams:
            if b["status"] != "BLOCKED":
                continue

            x1, x2 = b["x1"], b["x2"]
            overlay = frame[0:h, x1:x2].copy()
            cv2.rectangle(overlay, (0, 0), (x2 - x1 - 1, h - 1), _RED, -1)
            cv2.addWeighted(overlay, 0.25, frame[0:h, x1:x2], 0.75, 0, dst=frame[0:h, x1:x2])
            cv2.rectangle(frame, (x1, 0), (x2 - 1, h - 1), _RED, 2)

            d_m = b["distance_m"]
            text = f"{d_m:.2f}m" if d_m > 0.0 else "obstacle"
            (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
            tx = x1 + max(0, ((x2 - x1) - tw) // 2)
            cv2.putText(frame, text, (tx, 16), font, scale, _WHITE, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_profile(frame: np.ndarray, beams: List[Dict]) -> None:
        """Draw a radar-like bar chart strip at the bottom of the frame.

        Each beam is one vertical bar. Bar height is inversely proportional
        to distance — closer obstacles produce taller bars. Colour follows
        the beam status (CLEAR/CAUTION/BLOCKED/NO_DATA).
        """
        h, w = frame.shape[:2]
        strip_y = h - _PROFILE_H

        # Dark background for the strip
        cv2.rectangle(frame, (0, strip_y), (w, h), (20, 20, 20), -1)

        for b in beams:
            x1     = b["x1"]
            x2     = b["x2"] - 1
            d_m    = b["distance_m"]
            color  = _STATUS_COLOR.get(b["status"], _GREY)

            if d_m > 0.0:
                # Taller bar = closer obstacle
                frac  = max(0.0, 1.0 - d_m / _PROFILE_MAXD)
                bar_h = max(3, int(_PROFILE_H * frac))
            else:
                bar_h = 3  # thin grey tick = no data

            bar_y = strip_y + (_PROFILE_H - bar_h)
            cv2.rectangle(frame, (x1, bar_y), (x2, h), color, -1)

    def _draw_depth_thumb(
        self, frame: np.ndarray, depth_map: np.ndarray
    ) -> np.ndarray:
        """Paste a colourised depth thumbnail into the top-right corner."""
        h, w = frame.shape[:2]
        thumb_w = max(1, int(w * self._thumb_scale))
        thumb_h = max(1, int(h * self._thumb_scale))

        valid = depth_map > 0.0
        vis   = np.zeros_like(depth_map, dtype=np.uint8)
        if valid.any():
            d_min = float(depth_map[valid].min())
            d_max = float(depth_map[valid].max())
            if d_max > d_min:
                norm = np.where(
                    valid,
                    255.0 * (1.0 - (depth_map - d_min) / (d_max - d_min)),
                    0.0,
                )
                vis = norm.clip(0, 255).astype(np.uint8)

        depth_color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        thumb = cv2.resize(depth_color, (thumb_w, thumb_h),
                           interpolation=cv2.INTER_NEAREST)

        x0 = w - thumb_w - 2
        y0 = 2
        cv2.rectangle(frame, (x0 - 1, y0 - 1),
                      (x0 + thumb_w, y0 + thumb_h), _WHITE, 1)
        frame[y0: y0 + thumb_h, x0: x0 + thumb_w] = thumb
        return frame

    @staticmethod
    def _draw_badge(
        frame:  np.ndarray,
        text:   str,
        centre: Tuple[int, int],
        color:  Tuple[int, int, int],
    ) -> None:
        """Draw a filled rectangle badge with *text* centred at *centre*."""
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.75
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        cx, cy = centre
        pad = 6
        x1 = cx - tw // 2 - pad
        y1 = cy - th - pad
        x2 = cx + tw // 2 + pad
        y2 = cy + baseline + pad
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BLACK, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color,  1)
        cv2.putText(frame, text, (cx - tw // 2, cy),
                    font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _distance_color(dist_m: float) -> Tuple[int, int, int]:
        if dist_m < 0.5:
            return _RED
        if dist_m < 1.5:
            return _YELLOW
        return _GREEN

    def __repr__(self) -> str:
        return (
            f"OverlayRenderer(roi={self._roi_w}x{self._roi_h}, "
            f"thumb_scale={self._thumb_scale})"
        )
