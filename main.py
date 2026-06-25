"""
Depth Perception Engine — main entry point.

Pipeline:
    camera (640x240 raw) → split → rectify → disparity → depth → distance → display

Camera is opened directly (not via StereoCameraStream) because this USB stereo
camera does not support MJPEG; StereoCameraStream's MJPEG codec forcing produces
black frames on this hardware.
"""

import logging
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or from inside the package
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from stereo.frame_splitter     import FrameSplitter
from stereo.rectification      import RectificationEngine
from stereo.disparity_engine   import DisparityEngine
from depth.depth_estimator     import DepthEstimator
from depth.distance_reader     import DistanceReader
from fusion.threat_assessment  import ThreatAssessor
from navigation.velocity_planner import VelocityPlanner
from perception.scene_interpreter import SceneInterpreter
from visualization.overlay_renderer import OverlayRenderer
from telemetry.run_logger      import RunLogger

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX       = 0
FRAME_W, FRAME_H   = 640, 240   # native stereo side-by-side resolution
CALIBRATION_FILE   = os.path.join(_PROJECT_ROOT, "config", "stereo_calibration.xml")
NUM_DISPARITIES    = 128        # covers ~0.31 m – 10 m with

# 64mm baseline
BLOCK_SIZE         = 13         # larger block = smoother disparity at 320×240
WARMUP_FRAMES      = 60         # discard initial auto-exposure frames
STALE_FRAMES_MAX   = 4          # clear EMA if no valid reading for this many frames
DEPTH_PERCENTILE   = 25         # use 25th percentile of ROI — biases toward nearer surface
CLOSE_RANGE_M      = 0.70       # readings below this are considered "approaching"
TOO_CLOSE_THRESH   = 0.85       # fraction of invalid ROI pixels that signals "too close"
EMA_ALPHA          = 0.20       # EMA smoothing weight — lower = smoother, higher = faster

# Drone navigation thresholds — tighter than a generic 1 m cutoff so the
# drone keeps moving through normal indoor corridor widths instead of
# hovering whenever a side wall is within a meter. BLOCKED triggers below
# caution_m; well clear of the ~0.31 m stereo blind floor.
NAV_CAUTION_M      = 0.60       # corridor distance below this → slow down
NAV_CLEAR_M        = 1.20       # corridor distance above this → full speed
NAV_MAX_FORWARD_MPS = 0.6
NAV_CREEP_MPS      = 0.15       # speed used when corridor data is unreliable
NAV_MAX_YAW_DPS    = 25.0
NAV_CORRIDOR_FRAC  = 0.35       # fraction of the valid beam range (centred) treated as "ahead"
N_BEAMS            = 20
RECTIFIED_WIDTH    = FRAME_W // 2   # left/right split width after FrameSplitter
BEAM_WIDTH_PX      = RECTIFIED_WIDTH / N_BEAMS
# Leading beams fully inside the SGBM dead zone — structurally invalid
# regardless of scene content, must never be treated as an obstacle.
NAV_DEAD_ZONE_BEAMS = int(np.ceil(NUM_DISPARITIES / BEAM_WIDTH_PX))

# Perception-fusion grid — partitions each frame for region-level
# classification (CLEAR/OBSTACLE/LOW_TEXTURE_UNKNOWN/PROBABLE_WALL/
# LOW_CONFIDENCE), independent of the beam-based corridor above. Reuses
# the same caution/clear thresholds for consistency between the two layers.
SCENE_GRID_ROWS = 3
SCENE_GRID_COLS = 3
SCENE_AMBIGUOUS_FRACTION_THRESH = 0.50


def _config_snapshot() -> dict:
    """Capture every tunable threshold/parameter in effect for this run."""
    return {
        "camera_index":        CAMERA_INDEX,
        "frame_w":              FRAME_W,
        "frame_h":              FRAME_H,
        "calibration_file":     CALIBRATION_FILE,
        "num_disparities":      NUM_DISPARITIES,
        "block_size":           BLOCK_SIZE,
        "warmup_frames":        WARMUP_FRAMES,
        "stale_frames_max":     STALE_FRAMES_MAX,
        "depth_percentile":     DEPTH_PERCENTILE,
        "close_range_m":        CLOSE_RANGE_M,
        "too_close_thresh":     TOO_CLOSE_THRESH,
        "ema_alpha":            EMA_ALPHA,
        "nav_caution_m":        NAV_CAUTION_M,
        "nav_clear_m":          NAV_CLEAR_M,
        "nav_max_forward_mps":  NAV_MAX_FORWARD_MPS,
        "nav_creep_mps":        NAV_CREEP_MPS,
        "nav_max_yaw_dps":      NAV_MAX_YAW_DPS,
        "nav_corridor_frac":    NAV_CORRIDOR_FRAC,
        "n_beams":              N_BEAMS,
        "nav_dead_zone_beams":  NAV_DEAD_ZONE_BEAMS,
        "scene_grid_rows":      SCENE_GRID_ROWS,
        "scene_grid_cols":      SCENE_GRID_COLS,
        "scene_ambiguous_fraction_thresh": SCENE_AMBIGUOUS_FRACTION_THRESH,
    }


def open_camera() -> cv2.VideoCapture:
    """Open the USB stereo camera without forcing a codec."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(WARMUP_FRAMES):
        cap.read()
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")
    return cap


def build_pipeline():
    """Construct and return all pipeline components."""
    splitter  = FrameSplitter()

    rectifier = RectificationEngine(calibration_file=CALIBRATION_FILE)
    rectifier.load_calibration()
    rectifier.initialize_rectification()

    disparity = DisparityEngine(
        min_disparity=0,
        num_disparities=NUM_DISPARITIES,
        block_size=BLOCK_SIZE,
    )

    estimator = DepthEstimator.from_calibration_file(CALIBRATION_FILE)

    reader   = DistanceReader(roi_width=80, roi_height=80)
    assessor = ThreatAssessor(
        n_beams=N_BEAMS,
        clear_m=NAV_CLEAR_M,
        caution_m=NAV_CAUTION_M,
        percentile=15,
        dead_zone_px=NUM_DISPARITIES,
    )
    planner  = VelocityPlanner(
        n_beams=N_BEAMS,
        corridor_frac=NAV_CORRIDOR_FRAC,
        caution_m=NAV_CAUTION_M,
        clear_m=NAV_CLEAR_M,
        max_forward_mps=NAV_MAX_FORWARD_MPS,
        creep_mps=NAV_CREEP_MPS,
        max_yaw_dps=NAV_MAX_YAW_DPS,
        dead_zone_beams=NAV_DEAD_ZONE_BEAMS,
    )
    scene_interpreter = SceneInterpreter(
        rows=SCENE_GRID_ROWS,
        cols=SCENE_GRID_COLS,
        caution_m=NAV_CAUTION_M,
        clear_m=NAV_CLEAR_M,
        ambiguous_fraction_thresh=SCENE_AMBIGUOUS_FRACTION_THRESH,
    )
    renderer = OverlayRenderer(roi_width=80, roi_height=80, depth_thumb_scale=0.35)

    return splitter, rectifier, disparity, estimator, reader, assessor, planner, scene_interpreter, renderer


def main() -> None:
    run_logger = RunLogger(config=_config_snapshot())
    print(f"Logging run artifacts to {run_logger.run_dir}")

    print("Loading calibration and building pipeline...")
    try:
        (splitter, rectifier, disparity_engine, estimator, reader,
         assessor, planner, scene_interpreter, renderer) = build_pipeline()
        run_logger.info("Pipeline built successfully")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Pipeline setup failed: {exc}")
        run_logger.error(f"Pipeline setup failed: {exc}")
        run_logger.close()
        raise SystemExit(1)

    print("Opening camera...")
    try:
        cap = open_camera()
        run_logger.info(f"Camera {CAMERA_INDEX} opened at {FRAME_W}x{FRAME_H}")
    except RuntimeError as exc:
        print(f"Camera error: {exc}")
        run_logger.error(f"Camera error: {exc}")
        run_logger.close()
        raise SystemExit(1)

    print("Running — press 'q' to quit, 'd' to toggle disparity view, 'g' to toggle scene grid.")
    show_disparity = False
    show_grid       = False
    frame_count = 0
    fps = 0.0
    t_fps = time.perf_counter()

    _ema_dist       = 0.0    # EMA for centre ROI distance (0 = uninitialised)
    _last_print     = 0.0
    _stale_count    = 0
    _recently_close = False
    _beams          = []
    _nav            = {"forward_mps": 0.0, "yaw_rate_dps": 0.0, "status": "NO_DATA",
                        "ahead_distance_m": 0.0, "corridor_x1": 0, "corridor_x2": 0}
    _scene          = {}
    _scene_decision = None
    _total_frames   = 0

    try:
        while True:
            ret, raw = cap.read()
            if not ret or raw is None:
                continue

            _too_close = False

            # --- split ---
            try:
                left_raw, right_raw = splitter.split(raw)
            except (ValueError, TypeError):
                continue

            # --- rectify ---
            try:
                left_rect, right_rect = rectifier.rectify(left_raw, right_raw)
            except (ValueError, RuntimeError):
                left_rect, right_rect = left_raw, right_raw

            # --- disparity ---
            try:
                raw_disp, disp_vis = disparity_engine.compute_disparity(left_rect, right_rect)
            except (ValueError, TypeError, RuntimeError):
                continue

            # Detect "too close": center ROI mostly invalid disparity means the
            # object is nearer than SGBM's max-disparity limit (~32 cm at 128 disp),
            # or it's a textureless surface (e.g. a wall) producing no stereo match
            # at all — either way, something is right in front of the camera.
            _dh, _dw = raw_disp.shape[:2]
            _cy, _cx = _dh // 2, _dw // 2
            _r = min(40, _cy, _cx)
            _roi_d = raw_disp[_cy - _r:_cy + _r, _cx - _r:_cx + _r]
            _too_close = bool((_roi_d <= 0).mean() > TOO_CLOSE_THRESH)

            # --- depth ---
            try:
                depth_map = estimator.estimate(raw_disp)
            except (TypeError, ValueError, RuntimeError):
                depth_map = None

            # --- distance ---
            measurement = None
            if depth_map is not None:
                try:
                    roi_vals   = reader.extract_roi(depth_map)
                    valid_vals = reader.filter_depth_values(roi_vals)

                    # IQR outlier rejection — strip rogue SGBM pixels before sampling
                    if valid_vals.size >= 10:
                        q1, q3 = np.percentile(valid_vals, [25, 75])
                        iqr = q3 - q1
                        if iqr > 0:
                            valid_vals = valid_vals[
                                (valid_vals >= q1 - 1.5 * iqr) &
                                (valid_vals <= q3 + 1.5 * iqr)
                            ]
                        d_m = float(np.percentile(valid_vals, DEPTH_PERCENTILE)) if valid_vals.size >= 5 else 0.0
                    else:
                        d_m = 0.0

                    if d_m > 0.0:
                        # EMA update — smooth out frame-to-frame noise
                        _ema_dist = d_m if _ema_dist <= 0.0 else EMA_ALPHA * d_m + (1.0 - EMA_ALPHA) * _ema_dist
                        _stale_count    = 0
                        _recently_close = d_m < CLOSE_RANGE_M
                        raw_meas = reader.compute_distance(valid_vals)
                        measurement = {**raw_meas,
                                       "distance_meters":      _ema_dist,
                                       "distance_centimeters": _ema_dist * 100.0}
                    else:
                        if _too_close and _recently_close:
                            _ema_dist    = 0.0
                            _stale_count = 0
                        else:
                            _recently_close = False
                            _stale_count += 1
                            if _stale_count >= STALE_FRAMES_MAX:
                                _ema_dist = 0.0
                except (TypeError, ValueError, RuntimeError):
                    pass

            # --- threat assessment (per-zone obstacle presence + distance) ---
            if depth_map is not None:
                result  = assessor.assess(depth_map, raw_disp)
                _beams  = result["beams"]
                _nav    = planner.plan(_beams, result["safest_beam"])
            else:
                _beams = []

            # --- perception-fusion grid (region classification + global decision) ---
            if depth_map is not None:
                gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY) if left_rect.ndim == 3 else left_rect
                _scene = scene_interpreter.analyze(gray, raw_disp, depth_map)
                _scene_decision = scene_interpreter.decide_navigation(_scene)
            else:
                _scene = {}
                _scene_decision = None

            # --- print to terminal every 0.3 s ---
            now = time.perf_counter()
            if now - _last_print >= 0.3:
                _last_print = now
                if measurement and measurement["distance_meters"] > 0:
                    dm = measurement["distance_meters"]
                    print(f"\rDistance: {dm:.2f} m  ({dm*100:.0f} cm)   "
                          f"Ahead: {_nav['ahead_distance_m']:.2f}m [{_nav['status']}]  "
                          f"FPS: {fps:.1f}   ", end="", flush=True)
                elif _too_close and _recently_close:
                    print(f"\rDistance: 0.00 m  (< min range)   "
                          f"Ahead: {_nav['ahead_distance_m']:.2f}m [{_nav['status']}]  "
                          f"FPS: {fps:.1f}   ", end="", flush=True)
                else:
                    print(f"\r-- no depth --   "
                          f"Ahead: {_nav['ahead_distance_m']:.2f}m [{_nav['status']}]  "
                          f"FPS: {fps:.1f}   ", end="", flush=True)

            # --- telemetry (real-time, one record per processed frame) ---
            _total_frames += 1
            run_logger.log_frame(
                frame_idx=_total_frames,
                fps=fps,
                center_distance_m=measurement["distance_meters"] if measurement else 0.0,
                nav=_nav,
                beams=_beams,
                scene_decision=_scene_decision.value if _scene_decision else None,
                scene_regions={name: r.classification.value for name, r in _scene.items()},
            )

            # --- FPS ---
            frame_count += 1
            elapsed = time.perf_counter() - t_fps
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_fps = time.perf_counter()

            # --- display ---
            if show_grid:
                annotated = renderer.render_scene_grid(left_rect, _scene, _scene_decision, fps)
                cv2.imshow("Depth Perception Engine", annotated)
            elif show_disparity:
                disp_bgr = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
                cv2.putText(disp_bgr, f"FPS {fps:.1f}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2)
                cv2.imshow("Depth Perception Engine", disp_bgr)
            else:
                annotated = renderer.render(left_rect, depth_map, measurement, fps,
                                            beams=_beams, nav=_nav)
                cv2.imshow("Depth Perception Engine", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_disparity = not show_disparity
                mode = "disparity" if show_disparity else "depth overlay"
                print(f"  View: {mode}")
            elif key == ord('g'):
                show_grid = not show_grid
                print(f"  Scene grid: {'on' if show_grid else 'off'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        run_logger.info("Shutting down")
        run_logger.close()
        print(f"Done. Run artifacts saved to {run_logger.run_dir}")


if __name__ == "__main__":
    main()
