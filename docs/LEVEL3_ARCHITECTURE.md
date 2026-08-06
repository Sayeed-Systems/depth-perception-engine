# Level 3 architecture report (Phase E1)

## Current coordinate system, as found

Before this pass: entirely implicit. `DepthEstimator.estimate()`/`estimate_point_cloud()` operate in the rectified left camera's optical frame (OpenCV convention: X right, Y down, Z forward, right-handed) purely because that's what `cv2.reprojectImageTo3D`/the calibration's `Q` matrix already assume — nothing in this repository ever states this in code, a docstring, or a doc file. No body frame, no rigid-transform type, no frame-naming convention, no camera-model concept existed anywhere (confirmed by direct grep across `src/` this pass: zero hits for frame/handedness/body-frame/camera-model terms).

## Problems discovered

1. **Implicit, unnamed coordinate frame.** A future consumer of `depth_map` or a point cloud has no way to know which frame the values are in without reading source comments — and even those comments (added in the previous session's `docs/DATA_CONTRACTS.md`) were an inference from OpenCV's known conventions, not a stated, validated contract.
2. **No body-frame relationship exists at all.** Blocks any future vehicle-relative geometry (clearance, collision volume — Level 4+) until someone defines a convention; nothing to build on, not even a placeholder.
3. **`StereoCalibration` conflates intrinsic + extrinsic (left-right) + rectification into one flat bag.** Baseline/focal-length are only derivable ad hoc, buried as private state inside `DepthEstimator.__init__` — not independently testable or reusable by anything else that might need them (e.g. a future geometry module).
4. **`DepthEstimator.estimate_point_cloud()` is dead code.** Found in the previous session's perception capability audit: real implementation, zero callers, zero tests, no frame identity, no validity/confidence attachment. Confirmed still true this pass — not touched, since E1 explicitly forbids implementing geometry.
5. **`StereoObservation` had no reserved extensibility point for calibration.** A future multi-rig/multi-camera caller had nowhere to attach per-observation calibration without changing `process_observation()`'s signature.
6. **No geometry result types existed at all** — Level 3 work would otherwise start by inventing shapes/units/ownership conventions ad hoc, mid-implementation, rather than against a reviewed contract.

## Proposed architecture

Additive-only, layered on top of the untouched Level 0–2 pipeline:

```
src/depth_perception_engine/
├── frames.py                 # NEW — FrameId, RigidTransform (naming + transform representation)
├── calibration/
│   ├── models.py             # UNCHANGED — StereoCalibration, still what Rectification/DepthEstimator consume
│   ├── contracts.py          # NEW — CameraModel, CameraIntrinsics, StereoExtrinsics,
│   │                         #        RectificationParameters, RigCalibration (all derived views)
│   └── loader.py             # UNCHANGED
├── geometry/                 # NEW package — interfaces only
│   ├── __init__.py
│   └── types.py              # PointCloud, ObstacleCloud, FreeSpaceRays, GeometryMetrics
├── models/result.py          # StereoObservation gains one Optional field (calibration); nothing else changes
└── pipeline/                 # UNCHANGED
```

Future, E2+, **not built in this pass**:
```
geometry/
├── point_cloud_builder.py    # wires estimate_point_cloud() into a real execution path
├── obstacle_extractor.py     # PointCloud + RegionStats/ObstacleAssessment -> ObstacleCloud
└── free_space.py             # beam-scan geometry -> FreeSpaceRays
```

## Reasoning

- **Additive-only mirrors this session's established precedent.** The previous session's baseline recovery pass (lifecycle API, `StereoObservation`, validity masks) proved this pattern works: every new field/type defaults to inert/`None`, every existing call site keeps working unmodified, `mp01_perception` requires zero changes. E1 follows the identical discipline.
- **`StereoCalibration` itself stays untouched** rather than being decomposed in place, specifically because `RectificationEngine`/`DepthEstimator`/`calibration/loader.py` consume its exact current shape — changing it would count as modifying the processing pipeline, which E1 explicitly forbids, even if the change were shape-only with no math difference. Read-only derived views (`.from_calibration(...)`) achieve the same "distinguish intrinsic/extrinsic/rectification" goal with zero risk to the tested, hardware-verified consumers.
- **Geometry types live in a new top-level `geometry/` package, not inside `models/`,** because they represent a genuinely distinct future capability tier (Level 3), not more output shapes for the existing Level 0–2 pipeline — keeping them physically separate makes "this is not real yet" legible from the directory structure alone, and matches the existing precedent of one subpackage per capability area (`traversability/`, `obstacles/`, `quality/`).
- **Nothing new is exported from the top-level `depth_perception_engine` package.** Promoting unproduced contracts to the top-level namespace would overclaim capability (a caller seeing `depth_perception_engine.PointCloud` would reasonably assume something produces one) and contradicts the encapsulation goal this task itself asks for — see `docs/LEVEL3_PUBLIC_API.md`.

## What this pass explicitly did not do

Implement Level 3 geometry, point cloud generation, obstacle extraction, occupancy mapping, IMU integration, temporal fusion, ROS integration, or any change to `stereo/`, `depth/depth_estimator.py`'s existing methods, `traversability/`, `obstacles/`, `fusion/`, `pipeline/pipeline.py`, `config/pipeline_config.py`, or `mp01_perception`. Verified: `pytest tests/ -q` — 162 passed (131 before this pass, +31 new contract tests, zero existing test modified in a way that changes its assertions), confirming the existing Level 0–2 behavior is byte-for-byte unchanged.
