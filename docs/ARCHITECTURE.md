# Architecture

## Module boundaries

| Module | Responsibility | Depends on |
|---|---|---|
| `calibration/` | `StereoCalibration` data model (validated, frozen) + `load_stereo_calibration(path)` file loader. The **only** place in the library that touches a file path, and only when explicitly called. | nothing else in this package |
| `stereo/` | `FrameSplitter` (optional), `RectificationEngine`, `DisparityEngine` (StereoSGBM) | `calibration/` (for the Q/R/P matrices) |
| `depth/` | `DepthEstimator` (disparity → metric depth, closed-form from the Q matrix), `DistanceReader` (single-point ROI reading — not part of the canonical result, see below) | `calibration/` |
| `quality/` | `looks_like_garbage_frame` — adjacent-pixel correlation check for corrupt/uncorrelated-noise frames, upstream of any stereo processing | nothing else in this package |
| `traversability/` | `RegionAnalyzer` + `SceneInterpreter` — grid-based region classification and one global `NavigationDecision` | nothing else in this package |
| `obstacles/` | `ThreatAssessor` — per-beam nearest-obstacle scan, EMA-smoothed and debounced *across calls* (this is the one piece of genuinely stateful algorithm code in the library) | nothing else in this package |
| `fusion/` | Assembles the above stage outputs into one `DepthPerceptionResult`, including the aggregate `confidence` score and the validity masks | `models/` |
| `config/` | `PipelineConfig` — every tunable threshold as one validated dataclass | nothing else in this package |
| `models/` | `StereoObservation`, `DepthPerceptionResult`, `TraversabilityResult`, `ObstacleAssessment`, `BeamReading`, `PipelineHealth` — typed values, never bare dicts | `traversability/` (for `RegionStats`/`NavigationDecision`) |
| `utils/` | Small shared helpers (generic stereo-pair shape validation, timing) used by the pipeline glue, not by any one algorithm | nothing else in this package |
| `pipeline/` | The public entry point — see below | everything above |

## Canonical execution path

There is exactly one algorithm path, reachable two ways:

```
StereoObservation / (left_image, right_image)
        |
        v
require_matching_stereo_pair()          utils/validation.py
        |
        v
rectify (if rectify=True)               stereo.RectificationEngine
        | (raises on failure — see "Failure semantics" below, does NOT
        |  silently fall back to the unrectified pair)
        v
grayscale conversion (once, reused
below for both SGBM and texture)
        |
        v
disparity (StereoSGBM)                  stereo.DisparityEngine
        |
        v
metric depth (closed-form Z from Q)     depth.DepthEstimator
        |
        v
region grid classification              traversability.RegionAnalyzer
+ global navigation decision            traversability.SceneInterpreter
        |
        v
per-beam obstacle scan                  obstacles.ThreatAssessor
(EMA-smoothed, debounced)
        |
        v
DepthPerceptionResult                   fusion.result_builder
```

**Two entry points, one path — not two competing pipelines:**
- `pipeline.DepthPerceptionPipeline` (class, stateful) — `.process()`, `.process_observation()`. Holds `ThreatAssessor` persistently across calls, because its EMA/debounce state must survive between frames; rebuilding it every frame throws that smoothing away. **This is the entry point any long-running caller (ROS node, live demo) should use.**
- `pipeline.api` (module, five stateless functions: `process_stereo_pair`, `compute_disparity`, `estimate_depth`, `classify_traversability`, `detect_obstacles`) — thin wrappers that construct the same underlying stage classes fresh on every call. Useful for one-shot scripts and tests where per-frame smoothing doesn't matter; **not** suitable for a caller that processes a video stream, since a fresh `ThreatAssessor` every call means no debounce ever engages.

Both call the identical underlying stage classes — confirmed by direct comparison during this recovery pass, no logic divergence between them.

**`DistanceReader` is not part of the canonical result.** It's a real, tested, but separate single-point ROI convenience (`depth.DistanceReader`), used by `examples/live_demo.py` for its on-screen distance badge. It is not wired into `DepthPerceptionResult` and never will be without a deliberate decision to do so — the canonical per-frame output is `disparity_map`/`depth_map`/`traversability_mask`/`obstacles` only.

## The replaceable algorithm boundary

`DisparityEngine` wraps OpenCV's `StereoSGBM` behind `compute_disparity(left, right) -> (raw_disparity, visualization)`. Nothing above it in the pipeline branches on which matcher is in use — a future alternative matcher only needs to satisfy that same two-value return contract and be swapped into `DepthPerceptionPipeline.__init__`. No matcher-specific logic currently leaks into `pipeline.py` itself.

## No-ROS guarantee

Enforced two ways, both unconditional and both run on every `pytest` invocation (`tests/test_no_ros_dependency.py`):
1. A real-absence import check (`rclpy`/`sensor_msgs`/`cv_bridge` genuinely not importable in a bare test environment) — skipped automatically when running inside a sourced ROS 2 environment, since ROS being *installed system-wide* is not evidence this library imports it.
2. A static AST scan of every `.py` file under `src/` for forbidden imports — runs unconditionally in every environment, including inside a sourced ROS overlay. This is the check that actually proves the guarantee; verified passing both standalone and inside the `mp01_ros2` docker container (which does have ROS Humble sourced) during this recovery pass.

A third, related check (`test_no_gui_or_hardware_capture_calls_in_the_library`) statically scans for `cv2.imshow`/`waitKey`/`namedWindow`/`VideoCapture`/`destroyAllWindows` anywhere under `src/` — none exist; all camera/GUI code lives in `examples/`, which is not imported by anything under `src/`.
