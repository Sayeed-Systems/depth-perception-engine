# Depth Perception Engine

A real-time, geometry-only obstacle avoidance pipeline built on a USB stereo camera. Designed as the perception layer for a UAV navigating indoor environments, targeting deployment on an NVIDIA Jetson Orin Nano.

No object detection, no learned models — just stereo depth, validated against real desk-test hardware, driving a reactive velocity planner.

![Demo](docs/assets/demo.gif)

## Pipeline

```
USB stereo camera (640×240 side-by-side)
  → split into left/right
  → rectify (calibrated)
  → disparity (SGBM)
  → depth + distance estimation
  → obstacle representation (20-beam corridor + 3×3 scene grid)
  → reactive velocity planner (forward speed / yaw rate)
  → telemetry logging + live overlay
```

See [`docs/report.html`](docs/report.html) for the full architecture blueprint, diagrams, and current implementation status against the target multi-agent (UAV + rover) system.

## Why geometry-only

Object detection (YOLO) was deliberately excluded, not just deferred as an afterthought: obstacle avoidance only requires *where* something is, not *what* it is, and the compute/power budget on the Jetson Orin Nano is reserved for future VIO/SLAM work instead. See Section 12 of the architecture doc for the full reasoning.

## Project layout

```
stereo/        frame splitting, rectification, SGBM disparity
depth/         depth estimation, distance reading
fusion/        threat assessment (beam-based obstacle distances)
navigation/    reactive velocity planner
perception/    scene grid region classification
visualization/ live debug overlay (disparity, ROI, scene grid)
telemetry/     per-run logging (config, system info, JSONL telemetry)
config/        camera + stereo calibration files
docs/          architecture blueprint, diagrams, roadmap
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy
python main.py
```

Requires a USB stereo camera presenting as a single side-by-side 640×240 video device, and a calibration file at `config/stereo_calibration.xml`. Camera index is set in `main.py` (`CAMERA_INDEX`) — verify with `v4l2-ctl --list-devices` if obstacle distances look wrong (e.g. picking up a laptop webcam instead of the stereo rig).

Controls while running: `q` to quit, `d` to toggle the disparity view, `g` to toggle the scene grid overlay.

Each run logs config, system info, and per-frame telemetry to `runs/<timestamp>/`.

## Status

Capture, rectification, disparity, depth estimation, obstacle beams, scene grid, and the reactive velocity planner are built and wired end-to-end, validated on real hardware. IMU, VIO/SLAM, lidar fusion, and the rover stack are planned but not yet implemented — see the status table in `docs/report.html` for the full breakdown.

## License

[MIT](LICENSE)