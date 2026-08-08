# Coordinate frames

> **Update (2026-08-06, Phase E4):** body-frame transformation is now implemented — `geometry.transform_point_cloud()`. The "Body Frame" and "Relationship between the two frames" sections below predate that (E1, contracts-only) and are updated in place rather than left as a historical record, since E4 is a direct continuation of exactly what those sections anticipated, using the exact convention they already froze — nothing below contradicts what E1 stated, it is now backed by real code. See `docs/IMPLEMENTATION_STATUS.md`'s E4 addendum for the full change record.

Level 3, Phase E1 — the authoritative frame specification. Before this pass, **no coordinate frame was named or documented anywhere in this repository** (verified by direct grep across `src/` for frame/handedness/body-frame terms — zero hits). Everything that already produces 3D-ish output (`depth_map`, the unused `DepthEstimator.estimate_point_cloud()`) lived implicitly in "whatever frame OpenCV's conventions imply," never stated. This document states it, and nothing else — no coordinate transformation code changes as a result of writing it down.

## Camera Optical Frame — `frames.FrameId.CAMERA_OPTICAL_LEFT`

The frame this library's math already, implicitly, operates in today.

| Property | Value |
|---|---|
| Origin | The rectified left camera's optical center |
| X axis | Right |
| Y axis | Down |
| Z axis | Forward (into the scene, along the optical axis) |
| Handedness | Right-handed — verified this pass: for unit axes, `X × Y = (1,0,0) × (0,1,0) = (0,0,1) = Z` |
| Units | Metres |
| Convention source | Standard OpenCV/computer-vision camera convention — the same one `cv2.stereoRectify`/`cv2.reprojectImageTo3D` already assume; this library never overrides it |

This is the frame `depth_map`, `disparity_map` (which is 2D but pixel-indexed in this frame's image plane), and `DepthEstimator.estimate_point_cloud()`'s XYZ output (if it were ever called — see `docs/IMPLEMENTATION_STATUS.md`) already exist in. Nothing about this pass changes that — it only gives the existing, already-in-use frame a name (`FrameId.CAMERA_OPTICAL_LEFT`, `src/depth_perception_engine/frames.py`) so a future geometry result can state which frame it's in instead of leaving it to be inferred.

**Millimetres vs. metres, restated:** the calibration's own internal object points (and therefore `Q`'s implied units) are millimetres — `DepthEstimator.estimate()`/`estimate_point_cloud()` both divide by 1000 exactly once before returning. This was already true and tested before this pass; stated here for completeness alongside the frame it applies to.

## Body Frame — `frames.FrameId.BODY`

Convention (frozen at E1, unchanged by E4): X forward, Y left, Z up, right-handed — matches ROS REP-103. Chosen because this library's output will eventually reach a ROS consumer (`mp01_perception`) even though this library itself stays ROS-free; matching REP-103 avoids a translation step at that boundary later.

**Units: metres**, same as the camera optical frame — `RigidTransform.translation` is documented in `frames.py` as metres, and `geometry.transform_point_cloud()` performs no unit conversion of its own; a caller supplying a millimetre-scale translation by mistake will get a wrong but silently-plausible answer (this is not detected, the same way `StereoCalibration`'s matrix entries are not finite-checked — see `docs/CALIBRATION.md`'s documented gap).

**As of Phase E4:** a real, tested producer exists — `geometry.transform_point_cloud(cloud, transform)` (`src/depth_perception_engine/geometry/rigid_transform.py`) — wired into `DepthPerceptionPipeline` behind the optional `body_T_camera_left` constructor parameter (see `docs/LEVEL3_ARCHITECTURE.md`'s E4 update and `docs/IMPLEMENTATION_STATUS.md`). Vehicle-relative reasoning beyond the coordinate transform itself (collision envelopes, occupancy, risk scoring) remains **out of scope, Level 4+ (E5 and beyond)** — see `docs/E5_IMPLEMENTATION_PLAN.md`.

## Relationship between the two frames

A `frames.RigidTransform` (rotation + translation + named `from_frame`/`to_frame`) is the vehicle for expressing "the camera's pose in the body frame" — `body_T_camera_left`, by this repo's naming convention (`to_frame` first, matching the common robotics `A_T_B` = "transform that expresses B's frame in A's coordinates" convention). **The authoritative, implemented convention (unchanged since E1, verified consistent between `frames.py`'s docstring and this document before E4 implementation began — no mismatch was found):** a point `p` in `from_frame` transforms to `to_frame` as `rotation @ p + translation`. Concretely, for `body_T_camera_left` (`from_frame=CAMERA_OPTICAL_LEFT`, `to_frame=BODY`):

```
p_body = rotation @ p_camera + translation
```

`calibration.contracts.RigCalibration` is where this transform is optionally carried alongside a `StereoCalibration` — see `docs/LEVEL3_CONTRACTS.md`. `RigCalibration.body_T_camera_left` defaults to `None`, meaning "not yet calibrated/unknown" — explicitly **not** "identity" or "zero offset." **This is enforced, not just documented:** `DepthPerceptionPipeline`'s optional `body_T_camera_left` constructor parameter also defaults to `None`, and when it is `None`, `DepthPerceptionResult.geometry_body` stays `None` — the pipeline never fabricates a body-frame cloud by assuming identity. A caller gets body-frame geometry only once it actually supplies a calibrated extrinsic.

### Calibration source and the real-hardware replacement workflow

**Development/test transform values used in this repository (`tests/test_rigid_transform.py`, `tests/test_pipeline_body_frame.py`, `examples/benchmark_body_transform.py`) are illustrative test inputs only** — synthetic rotations/translations chosen to exercise the math (90° rotations, arbitrary small translations), not measurements of any real airframe. **Real aircraft deployment must supply measured/calibrated camera-to-body extrinsics** — physically measured (or vision/target-based extrinsic calibration) rotation and translation from the rectified left camera's optical center to the vehicle body origin, provided as a real `frames.RigidTransform` and passed to `DepthPerceptionPipeline(..., body_T_camera_left=that_transform)`.

Replacing a synthetic/illustrative transform with a real one requires **calibration/configuration only** — supplying a different `RigidTransform` value at construction. No change to `geometry/rigid_transform.py`'s transformation code, `DepthPerceptionPipeline`'s integration logic, or any other part of this library is required or expected. `transform_point_cloud()` makes zero assumptions about where the stereo rig is physically mounted — it has no hardcoded dimensions, offsets, or orientation of any kind (verified: `tests/test_rigid_transform.py::TestFrameIdAndTimestamp::test_frame_id_is_generic_not_hardcoded_to_body` proves the function has no special-cased knowledge of `BODY` at all).

This library does not itself measure or compute a camera-to-body extrinsic (no `cv2.stereoRectify`-style extrinsic calibration routine exists here, matching `docs/CALIBRATION.md`'s existing "does not run calibration itself, assumes an external tool already produced it" design for `StereoCalibration`) — obtaining the real value is the deploying system's responsibility, the same way obtaining a real `StereoCalibration` already is.

## Naming convention

`frames.FrameId` is a small class of plain string constants, **not** a closed `Enum`. This is deliberate: a geometry result (`PointCloud.frame_id`, etc. — see `docs/LEVEL3_CONTRACTS.md`) needs to be able to name a frame this library has no opinion about (a second camera, a downstream consumer's own frame) without requiring an exhaustive enum membership this library would have to keep growing. `FrameId.CAMERA_OPTICAL_LEFT` and `FrameId.BODY` are simply the two names this repository itself currently has anything to say about.

## What this document does NOT do (as of E1) / current status (as of E4)

At E1, this document did not add a coordinate transformation function anywhere (`RigidTransform` had no "apply to points" method), did not change `depth_map`'s values/shape/meaning, and required no existing caller to change anything.

**As of E4:** the transformation function now exists (`geometry.transform_point_cloud()`, applied by `DepthPerceptionPipeline` when `body_T_camera_left` is supplied) and does not change `depth_map`, `disparity_map`, or any Level 0-2 output — verified in `tests/test_pipeline_body_frame.py::TestZeroRegression`. **`mp01_perception` still requires zero changes**: `body_T_camera_left` defaults to `None` on every constructor path, so an unmodified caller sees byte-identical behavior to pre-E4 — `DepthPerceptionResult.geometry_body` simply stays `None`, same as `geometry` does for a caller that never sets `PipelineConfig.enable_geometry`.
