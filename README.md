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

## Refactor: before → after

The library was restructured from a single flat script into an installable
package. No algorithm, threshold, or tuned parameter changed — only where
the code lives and how it's called.

**Before** — one entry point, ten flat top-level folders, camera/GUI/disk
code mixed in with the algorithm, no tests, not installable:

```
depth_perception_engine/
├── main.py                       # camera + GUI + algorithm, all in one loop
├── stereo/                       # rectification.py had a hardcoded default XML path
├── depth/                        # depth_estimator.py duplicated rectification.py's
│                                  # calibration-loading code
├── perception/                   # name didn't describe what it did
├── fusion/threat_assessment.py   # actually obstacle detection, not fusion
├── navigation/velocity_planner.py
├── visualization/overlay_renderer.py
├── telemetry/run_logger.py
└── config/stereo_calibration.xml

no tests/, no pyproject.toml, not pip-installable
```

**After** — an installable package with a clear library/example boundary:

```
depth_perception_engine/
├── pyproject.toml                # pip install -e .
├── src/depth_perception_engine/
│   ├── calibration/               # new — one file-loading path, no hardcoded default
│   ├── stereo/
│   ├── depth/
│   ├── traversability/            # renamed from perception/ — matches what it does
│   ├── obstacles/                 # renamed from fusion/threat_assessment.py
│   ├── fusion/                    # new — result assembly + confidence scoring
│   ├── config/                    # new — PipelineConfig dataclass
│   ├── models/                    # new — typed results, no bare dicts
│   ├── utils/                     # new — shared validation/timing helpers
│   └── pipeline/                  # new — the public entry point
├── examples/
│   ├── live_demo.py               # was main.py
│   ├── synthetic_demo.py          # new — no camera, no GUI
│   ├── navigation/
│   ├── visualization/
│   └── telemetry/
└── tests/                          # new — 19 tests
```

Operations performed:

| # | Change |
|---|---|
| 1 | Extracted calibration file-loading into `calibration/`, removing duplicated `cv2.FileStorage` parsing from both `rectification.py` and `depth_estimator.py` |
| 2 | Removed the hardcoded default calibration path baked into `RectificationEngine` — every entry point now requires a `StereoCalibration` object passed in explicitly |
| 3 | Renamed `perception/` → `traversability/` and `fusion/threat_assessment.py` → `obstacles/` to match what each module actually computes |
| 4 | Wrapped every previously loose dict (`beams`, `scene`, distance measurements) in a typed dataclass: `DepthPerceptionResult`, `TraversabilityResult`, `ObstacleAssessment`, `BeamReading` |
| 5 | Added `PipelineConfig` — every tunable threshold that used to be a loose module-level constant in `main.py`, now one dataclass |
| 6 | Added `pipeline.DepthPerceptionPipeline` (stateful, holds engines across frames) and five stateless functions (`compute_disparity`, `estimate_depth`, `classify_traversability`, `detect_obstacles`, `process_stereo_pair`) as the public API |
| 7 | Moved camera capture, `cv2` GUI overlays, flight-command velocity planning, and disk telemetry logging into `examples/` — none of it is reachable from the library |
| 8 | Added `pyproject.toml` (src-layout, `opencv-python-headless` for the library core so it never pulls in a GUI-capable OpenCV build) |
| 9 | Added `tests/` (19 tests): every module imports cleanly, the pipeline builds and runs, every output is the documented structured type, and no `rclpy`/`sensor_msgs`/`cv_bridge`/camera/GUI dependency exists anywhere under `src/` |
| 10 | Added `docs/INTEGRATION_READINESS.md`, describing exactly how this library will be called from ROS2's `mp01_perception` package later |

| | Before | After |
|---|---|---|
| Entry point | `main.py`, one script | `pip install -e .` + `DepthPerceptionPipeline` |
| Lines (algorithm code) | 3,335, flat | 2,565 in `src/`, 1,206 in `examples/` |
| Tests | none | 19 |
| ROS/camera/GUI coupling | mixed into the algorithm modules | isolated to `examples/`, verified absent from `src/` |
| Calibration loading | duplicated, hardcoded default path | one function, no default, caller supplies the path |

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
result.valid_disparity_mask   # bool (H, W) — disparity_map > 0
result.valid_depth_mask       # bool (H, W) — depth_map > 0
result.traversability_mask    # TraversabilityResult: per-region grid + NavigationDecision
result.obstacles              # ObstacleAssessment: per-beam nearest-obstacle scan
result.confidence              # float, 0..1
result.processing_time_ms     # float
result.timestamp              # Optional[float] — only set if you pass left_timestamp/right_timestamp

pipeline.health()              # PipelineHealth: is_closed, frames_processed, last_confidence, ...
pipeline.reset()                # clears cross-frame smoothing state, keeps calibration/config
pipeline.close()                # marks the pipeline unusable; further process() calls raise
```

`from_config()` and `process_observation()` are also available for callers
that prefer that shape — see `docs/DATA_CONTRACTS.md`'s `StereoObservation`
section. Both are equivalent to the constructor/`process()` calls above.

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
| `quality/` | `looks_like_garbage_frame` — adjacent-pixel correlation check that flags corrupt/uncorrelated-noise frames before any stereo processing runs |
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

131 tests (grew from the 19 the src/ refactor above originally added, most
recently in a 2026-08-05 baseline-recovery pass — see
`docs/VALIDATION_REPORT.md`): every module imports cleanly,
`DepthPerceptionPipeline` can be built and `.process()`d (including
repeated calls, its full lifecycle — `reset()`/`close()`/`health()` — and
rejecting mismatched stereo pairs), every output field is the documented
structured type (never a bare dict), depth math is verified both
differentially against OpenCV and against independent hand-computed known
values, and — the requirement this whole refactor exists for — no ROS
dependency exists anywhere in the library. See `docs/ARCHITECTURE.md`,
`docs/DATA_CONTRACTS.md`, and `docs/CALIBRATION.md` for the full module
boundaries, output contracts, and calibration conventions;
`docs/IMPLEMENTATION_STATUS.md` for what's implemented versus deferred.

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
