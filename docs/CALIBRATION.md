# Calibration

## Required values

`StereoCalibration` (`calibration.models.StereoCalibration`) is a frozen dataclass carrying exactly what `RectificationEngine` and `DepthEstimator` need — everything a standard OpenCV `cv2.stereoRectify()` call produces:

| Field | Shape | What it is |
|---|---|---|
| `image_size` | `(width, height)` tuple | The resolution the calibration was computed for — per-eye, after any stereo split. Every incoming frame's per-eye size must match this exactly, or rectification raises (see below). |
| `camera_matrix_left`, `camera_matrix_right` | `(3, 3)` | Intrinsic matrices |
| `dist_coeffs_left`, `dist_coeffs_right` | `(1, N)` | Distortion coefficients (any length OpenCV accepts — not shape-constrained beyond non-empty) |
| `R1`, `R2` | `(3, 3)` | Rectification rotations |
| `P1`, `P2` | `(3, 4)` | Rectified projection matrices |
| `Q` | `(4, 4)` | Disparity-to-depth mapping matrix |

All nine matrix fields are shape-validated at construction (`__post_init__` raises `ValueError` immediately on a missing/empty/wrong-shaped matrix — never silently accepted); `image_size` must be strictly positive in both dimensions. **Not currently validated:** individual matrix entries being finite (no NaN/Inf check) — a real, documented gap, see `docs/IMPLEMENTATION_STATUS.md`.

## What this library does NOT do

It does not run `cv2.stereoRectify()` itself, and does not accept raw intrinsics + a raw left-to-right rotation/translation pair to compute `R1`/`R2`/`P1`/`P2`/`Q` on the fly. It assumes an external calibration tool already produced the rectified form and expects to load it as-is. This is a deliberate, pre-existing design choice (not a defect introduced or left unfixed by this recovery pass) — worth knowing if a future calibration source only provides raw `R`/`T` extrinsics, since a conversion step (a straightforward `cv2.stereoRectify()` call) would need to happen before constructing a `StereoCalibration`.

## Loading

`calibration.load_stereo_calibration(path: str) -> StereoCalibration` — the **only** place in the library that touches a file path, and only when explicitly called by the caller; nothing in the library assumes or defaults to a particular file. Reads an OpenCV `FileStorage` **XML** file with these exact key names: `image_width`, `image_height`, `camera_matrix_left`, `dist_coeffs_left`, `camera_matrix_right`, `dist_coeffs_right`, `R1`, `R2`, `P1`, `P2`, `Q`. Raises `FileNotFoundError` if the path doesn't exist, `RuntimeError` if `cv2.FileStorage` can't open it or a key is missing/unparseable, `ValueError` (from `StereoCalibration` itself) if a parsed matrix has the wrong shape.

`examples/config/stereo_calibration.xml` is this repo's own desk-test hardware fixture (a real, physical 64mm-baseline global-shutter stereo rig — not synthetic) — used by `examples/live_demo.py` and the test suite's `calibration` fixture. It is **not** a generic default; any other rig needs its own calibration file supplied explicitly.

**Known dead file removed in this recovery pass:** `calibration.yml` (repo root) and `config/camera.yaml` were committed, unused-anywhere, stray artifacts in a different format (plain OpenCV YAML, not the XML this loader actually reads) — deleted, see `docs/VALIDATION_REPORT.md`.

## Baseline convention

The baseline is never stored as its own field — it's derived from `Q` by `DepthEstimator.__init__`:

```python
focal_length_px = abs(Q[2, 3])
Tx = Q[3, 2]
baseline_m = abs(1.0 / Tx) / 1000.0 if Tx != 0.0 else 0.0
```

`Q[3, 2]` (`-1/Tx` in the standard OpenCV convention, `Tx` in millimetres) is where the baseline actually lives; `abs()` is applied on read so the reported `baseline_m`/`focal_length_px` are always positive regardless of the sign convention a particular calibration tool used — but the **sign is preserved** in the raw `Q` matrix used for the actual depth computation, since that sign is what makes the resulting depth come out positive for a real rig. For this repo's fixture calibration: baseline ≈ 64.7 mm, focal length ≈ 614.5 px (matches the physical rig's known ~65mm baseline).

## Rectification assumptions

`RectificationEngine.rectify(left, right)` requires each incoming image's `(H, W)` to exactly match `StereoCalibration.image_size` — a size mismatch (e.g. a frame that no longer matches the loaded calibration) raises `ValueError` rather than silently attempting rectification maps built for a different resolution. This is deliberate: `DepthPerceptionPipeline.process()` does **not** catch this and fall back to processing the unrectified pair — a rectification failure invalidates the whole frame (see `docs/ARCHITECTURE.md`'s "Failure semantics" and the regression test `TestRectificationFailureInvalidatesTheFrame` in `tests/test_pipeline.py`).

`rectify=False` (a constructor parameter on `DepthPerceptionPipeline`) is a legitimate, documented opt-out for callers that already supply pre-rectified images — not a workaround for calibration mismatches.

## Depth units, restated

`depth_map` values are always metres, always `float32`, always `0.0` for invalid pixels — see `docs/DATA_CONTRACTS.md`. There is no unit ambiguity anywhere in the calibration→depth path: `Q`'s object points are in millimetres by the convention this loader/estimator assume (matching this repo's own calibration tooling), converted to metres exactly once, inside `DepthEstimator.estimate()`.
