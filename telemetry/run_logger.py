"""
Run logger — records every flight/test run of the pipeline to disk in
real time, so behavior can be reviewed or compared after the fact without
re-running the hardware.

Each run gets its own timestamped directory under runs/:
    runs/<YYYY-MM-DD_HH-MM-SS>/
        system_info.json — static description of what this system is and what
                            it's trying to do: pipeline stages, hardware, what
                            each status/threshold/telemetry field means. Lets
                            an outside viewer (human or LLM) understand a run
                            folder without any prior context from this repo.
        config.json     — full snapshot of the thresholds/parameters used
        run.log         — human-readable event log (startup, errors, shutdown)
        telemetry.jsonl — one line per frame: distance, nav decision, beam
                          statuses; line-buffered so it's readable mid-run
        summary.json    — written on close: duration, frame count, avg FPS,
                          time spent in each nav status, closest approach
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SYSTEM_INFO = {
    "system": "Depth Perception Engine — stereo-camera obstacle avoidance for a small indoor quadcopter",
    "goal": (
        "Estimate depth from a USB stereo camera, decide whether the space "
        "directly ahead is safe to fly through, and produce a forward speed "
        "+ yaw command. The drone should keep moving through open corridors "
        "and only slow/stop for obstacles actually in its flight path — not "
        "for walls merely visible off to the side."
    ),
    "pipeline_stages": [
        "camera capture (640x240 raw side-by-side frame, 25 fps)",
        "frame_splitter — split raw frame into left/right halves",
        "rectification — undistort + align left/right using stereo_calibration.xml",
        "disparity_engine — SGBM stereo matching -> raw disparity map",
        "depth_estimator — disparity -> metric depth map (metres)",
        "threat_assessment — scan depth map in 20 vertical beams, classify each "
            "CLEAR/CAUTION/BLOCKED/NO_DATA, with EMA smoothing + debouncing",
        "velocity_planner — convert beams into forward_mps + yaw_rate_dps, "
            "gating speed only on the beams inside the forward flight corridor",
        "scene_interpreter (perception-fusion grid) — partitions the frame into a "
            "3x3 grid (configurable), classifies each region from disparity validity, "
            "texture, entropy, gradient magnitude and confidence, then derives one "
            "global navigation decision (MOVE_FORWARD/TURN_LEFT/TURN_RIGHT/SLOW_DOWN/"
            "STOP/HOVER/ROTATE_AND_SCAN). Runs independently of the beam corridor above; "
            "no detector or neural network involved — purely classical image statistics.",
        "overlay_renderer / run_logger — live on-screen display and disk telemetry",
    ],
    "hardware": {
        "camera": "USB stereo camera, native 640x240 side-by-side (320x240 per eye)",
        "baseline_mm": 64,
        "stereo_blind_floor_m": 0.31,  # closer than this, SGBM cannot resolve disparity at all
        "sgbm_dead_zone": (
            "The leftmost ~num_disparities pixel columns of the rectified frame "
            "are structurally invalid (no valid disparity range to search), "
            "regardless of scene content. Always reported NO_DATA, never BLOCKED."
        ),
    },
    "beam_status_meaning": {
        "CLEAR":   "beam's nearest surface is farther than clear_m — full speed allowed",
        "CAUTION": "beam's nearest surface is between caution_m and clear_m — speed scales down",
        "BLOCKED": "beam's nearest surface is closer than caution_m, OR the beam has no "
                   "valid depth but mostly invalid (blocked) disparity — catches textureless "
                   "obstacles like a blank wall that produce no stereo correspondence",
        "NO_DATA": "insufficient valid pixels to make any determination (e.g. SGBM dead zone)",
    },
    "nav_field_meaning": {
        "forward_mps":      "commanded forward speed in metres/second",
        "yaw_rate_dps":      "commanded yaw rate in degrees/second; positive = turn right, "
                             "steers toward the beam with the most clearance",
        "status":            "worst condition found in the forward flight corridor (not the "
                             "whole frame) — this is what actually gates forward_mps",
        "ahead_distance_m":  "distance reading from the specific beam(s) that produced status; "
                             "0.0 means that status has no associated distance (e.g. a "
                             "texture-blocked beam, or a NO_DATA creep fallback)",
    },
    "region_class_meaning": {
        "CLEAR":               "confident reading (good disparity validity + texture), nothing close",
        "OBSTACLE":            "confident reading, nearest surface in this region is within caution_m",
        "LOW_TEXTURE_UNKNOWN": "too little valid disparity and texture to judge either way — "
                                "genuinely ambiguous, not assumed clear",
        "PROBABLE_WALL":       "very high invalid-disparity ratio AND very low texture AND very low "
                                "confidence — most likely a near, textureless surface (e.g. a blank "
                                "wall) that stereo cannot range, treated as a probable obstacle, "
                                "never as free space",
        "LOW_CONFIDENCE":      "generic catch-all when the blended confidence score is too low to "
                                "trust, regardless of the specific reason",
    },
    "scene_decision_meaning": {
        "MOVE_FORWARD":    "forward region is CLEAR and beyond clear_m",
        "TURN_LEFT":        "forward region blocked; left region is the better/only clear option",
        "TURN_RIGHT":        "forward region blocked; right region is the better/only clear option",
        "SLOW_DOWN":         "forward region is CLEAR but within caution range, or its reading is "
                             "ambiguous (LOW_TEXTURE_UNKNOWN/LOW_CONFIDENCE) rather than confirmed clear",
        "STOP":              "forward region blocked and neither side is confidently clear",
        "HOVER":             "no usable scene data this frame (e.g. depth estimation failed)",
        "ROTATE_AND_SCAN":   "at least half of all grid regions are ambiguous — the whole scene is "
                             "too uncertain to trust any single direction",
    },
    "telemetry_jsonl_fields": {
        "t":                    "seconds since run start",
        "frame":                "frame index since run start",
        "fps":                  "measured pipeline frame rate at this point",
        "center_distance_m":    "EMA-smoothed distance from a small fixed ROI at the exact "
                                 "frame center — a separate, narrower reading than nav_ahead_distance_m",
        "nav_status":           "see beam_status_meaning",
        "nav_forward_mps":      "see nav_field_meaning.forward_mps",
        "nav_yaw_rate_dps":     "see nav_field_meaning.yaw_rate_dps",
        "nav_ahead_distance_m": "see nav_field_meaning.ahead_distance_m",
        "beam_statuses":        "status of all 20 beams left-to-right, for full-frame context",
        "scene_decision":       "see scene_decision_meaning — global recommendation from the perception grid",
        "scene_regions":        "per-region classification from the perception grid, see region_class_meaning",
    },
    "known_limitations": [
        "No flight-controller wiring yet — this only computes the commands, it does not "
            "send them to a drone (e.g. via MAVSDK/MAVLink).",
        "Indoor desk/room testing will show more BLOCKED/CAUTION time than real flight in "
            "open spaces, because nearby walls are themselves within the safety thresholds — "
            "this is intended behavior, not a bug (see status_seconds in summary.json).",
    ],
}


class RunLogger:
    def __init__(self, root_dir: str = "runs", config: Optional[Dict] = None) -> None:
        timestamp     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir  = Path(root_dir) / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        (self.run_dir / "system_info.json").write_text(json.dumps(SYSTEM_INFO, indent=2))

        if config is not None:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

        self._jsonl_file = open(self.run_dir / "telemetry.jsonl", "a", buffering=1)

        self._logger = logging.getLogger(f"run_logger.{timestamp}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        handler = logging.FileHandler(self.run_dir / "run.log")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self._logger.addHandler(handler)

        self._start_time     = time.time()
        self._frame_count    = 0
        self._fps_sum        = 0.0
        self._status_seconds = {"CLEAR": 0.0, "CAUTION": 0.0, "BLOCKED": 0.0, "NO_DATA": 0.0}
        self._last_status      = None
        self._last_status_time = self._start_time
        self._min_ahead_distance_m: Optional[float] = None

        self.info(f"Run started — artifacts in {self.run_dir}")

    # ------------------------------------------------------------------
    def info(self, message: str) -> None:
        self._logger.info(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def log_frame(
        self,
        frame_idx:         int,
        fps:                float,
        center_distance_m:  float,
        nav:                Dict,
        beams:              List[Dict],
        scene_decision:     Optional[str] = None,
        scene_regions:      Optional[Dict[str, str]] = None,
    ) -> None:
        """Append one telemetry record. Called once per processed frame.

        scene_decision/scene_regions are optional so older callers (or a
        pipeline run without the perception-fusion grid enabled) keep working.
        """
        now = time.time()

        record = {
            "t":                    round(now - self._start_time, 3),
            "frame":                frame_idx,
            "fps":                  round(fps, 2),
            "center_distance_m":    round(center_distance_m, 3),
            "nav_status":           nav["status"],
            "nav_forward_mps":      nav["forward_mps"],
            "nav_yaw_rate_dps":     round(nav["yaw_rate_dps"], 2),
            "nav_ahead_distance_m": round(nav["ahead_distance_m"], 3),
            "beam_statuses":        [b["status"] for b in beams],
        }
        if scene_decision is not None:
            record["scene_decision"] = scene_decision
        if scene_regions is not None:
            record["scene_regions"] = scene_regions
        self._jsonl_file.write(json.dumps(record) + "\n")

        self._frame_count += 1
        self._fps_sum     += fps

        if self._last_status is not None:
            self._status_seconds[self._last_status] += now - self._last_status_time
        self._last_status      = nav["status"]
        self._last_status_time = now

        ahead = nav["ahead_distance_m"]
        if ahead > 0.0 and (self._min_ahead_distance_m is None or ahead < self._min_ahead_distance_m):
            self._min_ahead_distance_m = ahead

    def close(self) -> None:
        """Flush a run summary and close all open handles. Safe to call once."""
        now = time.time()
        if self._last_status is not None:
            self._status_seconds[self._last_status] += now - self._last_status_time

        summary = {
            "duration_s":          round(now - self._start_time, 2),
            "frame_count":         self._frame_count,
            "avg_fps":             round(self._fps_sum / self._frame_count, 2) if self._frame_count else 0.0,
            "status_seconds":      {k: round(v, 2) for k, v in self._status_seconds.items()},
            "min_ahead_distance_m": self._min_ahead_distance_m,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.info(f"Run closed — summary: {summary}")

        self._jsonl_file.close()
        for h in self._logger.handlers[:]:
            h.close()
            self._logger.removeHandler(h)
