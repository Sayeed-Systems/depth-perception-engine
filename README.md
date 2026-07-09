# Depth Perception Engine

A standalone, ROS-free stereo depth / traversability / obstacle perception
library. Geometry-only — no object detection, no learned models — designed
as the perception layer for a UAV navigating indoor environments.

Originally built and validated as a single hardware desk-test script; now
refactored into an installable library (`src/depth_perception_engine/`) so
it can be imported by MP-01's ROS2 `mp01_perception` package (or any other
caller) without dragging in ROS, camera hardware, or a GUI. The algorithms
themselves are unchanged from the validated desk-test version — this was a
structural refactor, not an algorithm rewrite.

![Demo](docs/assets/demo.gif)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # library only — numpy + opencv-python-headless
pip install -e ".[dev]"       # + pytest, for running tests/
```

No camera, no display, and no ROS installation required for the library
itself. See [Running the examples](#running-the-examples) below for the
one extra step needed for the GUI demo.

## Quick start

```python
from depth_perception_engine.calibration import load_stereo_calibration
from depth_perception_engine.config import PipelineConfig
from depth_perception_engine.pipeline import DepthPerceptionPipeline

calibration = load_stereo_calibration("examples/config/stereo_calibration.xml")
pipeline = DepthPerceptionPipeline(PipelineConfig(), calibration)   # build once

result = pipeline.process(left_image, right_image)   # call per frame — both plain NumPy arrays

result.disparity_map          # float32 (H, W)
result.depth_map              # float32 (H, W), metres
result.traversability_mask    # TraversabilityResult: per-region grid + NavigationDecision
result.obstacles              # ObstacleAssessment: per-beam nearest-obstacle scan
result.confidence              # float, 0..1
result.processing_time_ms     # float
```

See [`examples/synthetic_demo.py`](examples/synthetic_demo.py) for this
same call shape run standalone (no camera), and
[`docs/INTEGRATION_READINESS.md`](docs/INTEGRATION_READINESS.md) for
exactly how MP-01's ROS2 `mp01_perception` package will call this later.

## Pipeline

```
(left_image, right_image) — already-split NumPy arrays, e.g. from ROS
  → rectify (calibrated)                     stereo.RectificationEngine
  → disparity (SGBM)                         stereo.DisparityEngine
  → depth estimation                          depth.DepthEstimator
  → traversability grid + nav decision        traversability.SceneInterpreter
  → per-beam obstacle scan                    obstacles.ThreatAssessor
  → fused DepthPerceptionResult                fusion.result_builder
```

`pipeline.DepthPerceptionPipeline` wires all of the above and holds it
persistently across frames — this matters because `obstacles.ThreatAssessor`
EMA-smooths and debounces its output over time; rebuilding it every frame
throws that smoothing away. `pipeline.api` also exposes each stage as a
plain, stateless function (`compute_disparity`, `estimate_depth`,
`classify_traversability`, `detect_obstacles`, `process_stereo_pair`) for
one-shot use, scripting, or tests — see that module's docstring.

## Why geometry-only

Object detection (YOLO) was deliberately excluded, not just deferred as an
afterthought: obstacle avoidance only requires *where* something is, not
*what* it is, and the compute/power budget on the target hardware (NVIDIA
Jetson Orin Nano) is reserved for future VIO/SLAM work instead. See
Section 12 of [`docs/report.html`](docs/report.html) for the full reasoning.

## Project layout

```
src/depth_perception_engine/   the installable library — see below
tests/                         pytest suite: imports, pipeline, API, no-ROS-dependency proof
examples/                      runnable scripts + demo-only code (camera, GUI, flight-command
                                planning, disk telemetry) — none of this is part of the library
docs/                          architecture blueprint, diagrams, roadmap, integration readiness
runs/                          per-run telemetry from past examples/live_demo.py sessions
```

### `src/depth_perception_engine/`

| Module | Responsibility |
|---|---|
| `calibration/` | `StereoCalibration` data model + `load_stereo_calibration(path)` file loader. The **only** place in the library that touches a file path, and only when explicitly called. |
| `stereo/` | `FrameSplitter` (optional utility), `RectificationEngine`, `DisparityEngine` (SGBM) |
| `depth/` | `DepthEstimator` (disparity → metric depth via the calibration's Q matrix), `DistanceReader` (single-point ROI distance reading) |
| `traversability/` | `RegionAnalyzer` + `SceneInterpreter` — grid-based region classification and a global `NavigationDecision` |
| `obstacles/` | `ThreatAssessor` — per-beam nearest-obstacle scan, EMA-smoothed and debounced |
| `fusion/` | Combines the above into one `DepthPerceptionResult`, including the aggregate `confidence` score |
| `config/` | `PipelineConfig` — every tunable threshold as one plain dataclass |
| `models/` | `DepthPerceptionResult`, `TraversabilityResult`, `ObstacleAssessment`, `BeamReading` — typed outputs, never bare dicts |
| `utils/` | Small shared helpers (input validation, timing) used by the pipeline glue, not by any one algorithm |
| `pipeline/` | `DepthPerceptionPipeline` (stateful, recommended) + the stateless `pipeline.api` functions |

No module under `src/` imports `rclpy`, `sensor_msgs`, or `cv_bridge`, opens
a camera device, or calls any `cv2.imshow`/`waitKey`/GUI function — enforced
by `tests/test_no_ros_dependency.py`, both by static source scan and by the
simple fact that this test environment doesn't have ROS installed at all.

### `examples/`

Everything here is a *consumer* of the library, never a dependency of it:

| File | What it needs beyond the library |
|---|---|
| `live_demo.py` | Real USB stereo camera, `cv2.imshow` (full `opencv-python`, not headless — see `pyproject.toml`'s comment on this) |
| `synthetic_demo.py` | Nothing — synthetic NumPy arrays, no camera, no GUI |
| `navigation/velocity_planner.py` | Converts obstacle beam data into forward speed / yaw rate — flight-command generation, downstream of and out of scope for the perception library itself |
| `visualization/overlay_renderer.py` | `cv2` drawing for `live_demo.py`'s on-screen display |
| `telemetry/run_logger.py` | Disk-based per-run logging (`runs/<timestamp>/`) for desk-test sessions |

## Running the examples

```bash
# No hardware needed:
python examples/synthetic_demo.py

# Real camera + on-screen display:
pip uninstall opencv-python-headless && pip install opencv-python
python examples/live_demo.py
```

`live_demo.py` requires a USB stereo camera presenting as a single
side-by-side 640×240 video device, and reads calibration from
`examples/config/stereo_calibration.xml`. Camera index is set at the top of
the file (`CAMERA_INDEX`) — verify with `v4l2-ctl --list-devices` if
obstacle distances look wrong (e.g. picking up a laptop webcam instead of
the stereo rig). Controls while running: `q` to quit, `d` to toggle the
disparity view, `g` to toggle the traversability grid overlay. Each run
logs config, system info, and per-frame telemetry to `runs/<timestamp>/`.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

19 tests: every module imports cleanly, `DepthPerceptionPipeline` can be
built and `.process()`d (including repeated calls, and rejecting mismatched
stereo pairs), every output field is the documented structured type (never
a bare dict), and — the requirement this whole refactor exists for — no ROS
dependency exists anywhere in the library.

## Status

Rectification, disparity, depth estimation, obstacle beams, and the
traversability grid are validated on real hardware and now packaged as an
installable library (this refactor). `examples/live_demo.py` additionally
wires in a reactive velocity planner for standalone desk-testing, though
flight-command generation is intentionally *not* part of the library's own
scope — see [`docs/INTEGRATION_READINESS.md`](docs/INTEGRATION_READINESS.md).
IMU, VIO/SLAM, lidar fusion, and the rover stack are planned but not yet
implemented — see the status table in `docs/report.html` for the full
breakdown.

## License

[MIT](LICENSE)
