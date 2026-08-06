# Coordinate frames

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

**Not used anywhere in this repository today.** No rigid transform into this frame exists, and nothing computes vehicle-relative geometry (that's Level 4, out of scope for E1 and explicitly not started).

Recommended convention (not yet adopted by any code, stated here so a future transform has an agreed-on target rather than inventing one ad hoc): X forward, Y left, Z up, right-handed — matches ROS REP-103. Chosen because this library's output will eventually reach a ROS consumer (`mp01_perception`) even though this library itself stays ROS-free; matching REP-103 avoids a translation step at that boundary later. This is a recommendation for E2+, not a constraint this pass enforces anywhere.

## Relationship between the two frames

A `frames.RigidTransform` (rotation + translation + named `from_frame`/`to_frame`) is the vehicle for expressing "the camera's pose in the body frame" — `body_T_camera_left`, by this repo's naming convention (`to_frame` first, matching the common robotics `A_T_B` = "transform that expresses B's frame in A's coordinates" convention). Convention, restated: a point `p` in `from_frame` transforms to `to_frame` as `rotation @ p + translation`.

`calibration.contracts.RigCalibration` (new this pass) is where this transform is optionally carried alongside a `StereoCalibration` — see `docs/LEVEL3_CONTRACTS.md`. No calibration file, loader, or pipeline in this repository currently populates one; `RigCalibration.body_T_camera_left` defaults to `None`, meaning "not yet calibrated/unknown" — explicitly **not** "identity" or "zero offset." A future caller must not assume `None` means the camera sits at the body origin.

## Naming convention

`frames.FrameId` is a small class of plain string constants, **not** a closed `Enum`. This is deliberate: a geometry result (`PointCloud.frame_id`, etc. — see `docs/LEVEL3_CONTRACTS.md`) needs to be able to name a frame this library has no opinion about (a second camera, a downstream consumer's own frame) without requiring an exhaustive enum membership this library would have to keep growing. `FrameId.CAMERA_OPTICAL_LEFT` and `FrameId.BODY` are simply the two names this repository itself currently has anything to say about.

## What this document does NOT do

It does not add a coordinate transformation function anywhere (`RigidTransform` has no "apply to points" method — that's real Level 3 geometry code, explicitly out of scope for E1). It does not change `depth_map`'s values, shape, or meaning. It does not require any existing caller (including `mp01_perception`) to change anything — nothing currently reads `frames.py` or `RigCalibration` outside this repository's own new tests.
