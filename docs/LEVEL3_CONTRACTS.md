# Level 3 data contracts (Phase E1)

Everything below is a **frozen interface**, not an implementation. Nothing in this document describes code that computes geometry — every type here either already existed (documented for completeness) or is new, additive, and unreferenced by the current pipeline. See `docs/E2_IMPLEMENTATION_PLAN.md` for what actually produces these later.

## Observation contract — `models.StereoObservation`

Already existed (added in the previous session's baseline recovery pass): `left_image`, `right_image`, `left_timestamp`, `right_timestamp`, `frame_id`. This pass adds one field:

| Field | Type | Status |
|---|---|---|
| `calibration` | `Optional[calibration.StereoCalibration]`, default `None` | **New this pass.** Reserved for future multi-rig/multi-camera use. `None` means "use the pipeline's own calibration" — **not** "no calibration exists." `DepthPerceptionPipeline.process_observation()` does not read this field today; verified by a dedicated regression test (`tests/test_pipeline.py::TestProcessObservation::test_observation_calibration_field_defaults_to_none_and_is_unused`) proving its presence/absence produces byte-identical output. |

No other field changed. This satisfies the "observation contract" ask (left image, right image, timestamps, stereo calibration, optional metadata via `frame_id`) without altering `process_observation()`'s behavior.

## Calibration contract — `calibration.contracts`

`calibration.StereoCalibration` (unchanged) remains the one contract `RectificationEngine`/`DepthEstimator` actually consume. The new module adds **read-only decomposed views**, each derived from an existing `StereoCalibration` via a `.from_calibration(...)` classmethod — pure extraction, no new math:

| Type | Distinguishes | Fields | Derivation |
|---|---|---|---|
| `CameraModel` (`str, Enum`) | Camera model | `PINHOLE` (only value) | N/A — states an existing, previously-unstated assumption |
| `CameraIntrinsics` | Intrinsic calibration | `camera_matrix`, `dist_coeffs`, `image_size`, `camera_model` | `.from_calibration(calibration, side="left"\|"right")` |
| `StereoExtrinsics` | Extrinsic (left-right) calibration | `baseline_m`, `focal_length_px` | `.from_calibration(calibration)` — **identical formula** to `DepthEstimator.__init__`'s private derivation (`focal_length_px = abs(Q[2,3])`, `baseline_m = abs(1/Q[3,2])/1000`); cross-checked by test against `DepthEstimator.from_calibration(...).baseline_m`/`.focal_length_px` for the real fixture calibration — exact match confirmed |
| `RectificationParameters` | Rectification parameters | `R1`, `R2`, `P1`, `P2`, `Q` | `.from_calibration(calibration)` |
| `RigCalibration` | Coordinate-frame transform | `stereo: StereoCalibration`, `body_T_camera_left: Optional[RigidTransform] = None`, `camera_frame_id: str = FrameId.CAMERA_OPTICAL_LEFT` | Constructed directly — wraps, does not derive from, a `StereoCalibration` |

`Image dimensions` and `Baseline` (explicitly named in the E1 request) live inside `CameraIntrinsics.image_size` and `StereoExtrinsics.baseline_m` respectively — not duplicated as separate top-level types, since they're properties of intrinsics/extrinsics, not independent categories.

**Validation:** `RigCalibration.__post_init__` enforces that a supplied `body_T_camera_left` actually targets `FrameId.BODY` and originates from `camera_frame_id` — a real, if narrow, validation rule (frame-identity consistency), not just a shape check. `CameraIntrinsics.from_calibration` rejects an unrecognized `side` argument. Neither `StereoCalibration` itself nor any of its existing validation was touched.

**Nothing in this repository constructs a `RigCalibration` outside tests.** `DepthPerceptionPipeline` still takes a plain `StereoCalibration`, exactly as before this pass.

## Geometry contracts — `geometry.types`

Interfaces only — see that module's own docstrings for the full per-field unit/shape/ownership documentation; summarized here:

| Type | Represents | Shape | Frame-aware | Producer today |
|---|---|---|---|---|
| `PointCloud` | Organized (pixel-aligned) 3D points | `points: (H, W, 3)`, `valid_mask: (H, W)` | Yes (`frame_id`) | **None** |
| `ObstacleCloud` | Unorganized obstacle points | `points: (N, 3)` | Yes | **None** |
| `FreeSpaceRays` | Rays to nearest obstacle/max range | `origins/directions: (N, 3)`, `ranges_m: (N,)` | Yes | **None** |
| `GeometryMetrics` | Scalar summary stats | scalars | No | **None** |

All four: `frozen=True, slots=True` dataclasses, `float32` arrays, metres. `PointCloud.points` uses `NaN` (not `0.0`) for invalid pixels — a deliberate departure from `DepthPerceptionResult.depth_map`'s existing `0.0`-means-invalid convention, because `(0, 0, 0)` is a legitimate point directly on the optical axis for a point cloud and must not collide with "no data" the way a scalar depth of `0.0` safely can (0 m depth is never physically meaningful for a single scalar, but the coordinate origin is a real point in 3-space). This inconsistency between `depth_map` and any future `PointCloud` is intentional and documented here explicitly so it isn't mistaken for an oversight later.

Ownership: a future producer allocates and returns fresh arrays on every call — no type here promises safe in-place reuse across calls.

## Result contract — deferred, not changed

`models.DepthPerceptionResult` is **not modified this pass.** `docs/E2_IMPLEMENTATION_PLAN.md` describes adding an optional `geometry: Optional[<GeometryResult>]` field once a real producer exists — not before, per this task's explicit non-goal against implementing geometry now.
