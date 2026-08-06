# Data contracts

Every shape/dtype/unit/validity convention the library uses, in one place. If a value isn't documented here, treat that as a gap to fix, not as license to guess.

## Images

`left_image`/`right_image` (as passed to `process()`, or inside `StereoObservation`): `numpy.ndarray`, shape `(H, W)` grayscale or `(H, W, 3)` BGR (OpenCV convention — **not** RGB) or `(H, W, 4)` BGRA, `uint8`. Both images in a pair must share identical `(H, W)` — enforced by `utils.validation.require_matching_stereo_pair`, raises `ValueError`/`TypeError` otherwise. No implicit resizing anywhere in the library.

## `StereoObservation` (`models.StereoObservation`)

Optional convenience container — `process()` still takes `left_image`/`right_image` directly; use `process_observation()` to hand one of these to the pipeline instead.

| Field | Type | Notes |
|---|---|---|
| `left_image`, `right_image` | `np.ndarray` | as above |
| `left_timestamp`, `right_timestamp` | `Optional[float]` | **Opaque, caller-defined.** This library performs no unit conversion, synchronization, or skew checking on these — a caller using ROS time, wall-clock seconds, or a monotonic counter all work identically as far as this library is concerned. `left_timestamp` wins as the result's `timestamp` if both are set. |
| `frame_id` | `Optional[str]` | Carried on the observation only — not currently propagated onto `DepthPerceptionResult`. Available for a caller's own bookkeeping. |

## `StereoCalibration` (`calibration.StereoCalibration`)

Frozen, validated at construction (`__post_init__` raises `ValueError` on any violation below) — see `docs/CALIBRATION.md` for the full field-by-field breakdown and units. Loaded via `calibration.load_stereo_calibration(path)`, the only file-path-touching function in the library.

## `PipelineConfig` (`config.PipelineConfig`)

Plain dataclass, every tunable threshold in one place, validated at construction (`__post_init__`, added in this recovery pass — see `docs/IMPLEMENTATION_STATUS.md`). `PipelineConfig()` with no arguments reproduces the original hardware-desk-test tuned defaults exactly. See the class docstring in `src/depth_perception_engine/config/pipeline_config.py` for every field's meaning and unit.

## `DepthPerceptionResult` (`models.DepthPerceptionResult`)

The one top-level output shape, returned by both `DepthPerceptionPipeline.process()`/`.process_observation()` and `pipeline.api.process_stereo_pair()`.

| Field | Type / shape | Unit | Validity convention |
|---|---|---|---|
| `disparity_map` | `float32`, `(H, W)` | pixels | Invalid pixels (no stereo correspondence found) are `<= 0`. **Not** OpenCV's raw fixed-point `int16` output — already divided by 16 into true pixel units. |
| `depth_map` | `float32`, `(H, W)` | **metres** | Invalid pixels are exactly `0.0`. Valid range is clamped to `[DepthEstimator.MIN_DEPTH_M, DepthEstimator.MAX_DEPTH_M]` = `[0.15, 8.0]` m for the project's 64mm-baseline rig; anything outside that range is zeroed as invalid rather than returned as an out-of-trust-range number. |
| `valid_disparity_mask` | `bool`, `(H, W)` | — | `disparity_map > 0`, computed once in `fusion.result_builder` (added in this recovery pass) rather than left for every caller to re-derive from the sign convention above. |
| `valid_depth_mask` | `bool`, `(H, W)` | — | `depth_map > 0`. |
| `traversability_mask` | `TraversabilityResult` | — | **Naming wart, not fixed in this pass** — despite the name, this is not a pixel mask; it's a per-region grid + navigation decision (see below). Left as-is because `mp01_perception` reads this field by name today; see `docs/IMPLEMENTATION_STATUS.md`. |
| `obstacles` | `ObstacleAssessment` | — | see below |
| `confidence` | `float` | `[0.0, 1.0]` | Mean of the traversability grid's per-region confidence scores; `0.0` if the grid is empty. Not a probability in any calibrated sense — a transparent aggregate of an already-computed signal. |
| `processing_time_ms` | `float` | milliseconds | Wall-clock time for one `process()` call (`time.perf_counter()`-based), excluding image acquisition (the caller's problem, not this library's). |
| `timestamp` | `Optional[float]` | caller-defined | `None` unless `left_timestamp`/`right_timestamp` was passed to `process()` — see `StereoObservation` above for the opaque-unit caveat. |

### `TraversabilityResult`

`regions: Dict[str, RegionStats]` (e.g. `"TL".."BR"` for the default 3×3 grid — **not** a pixel-aligned array; this is the real granularity the underlying `RegionAnalyzer` computes, not forced into a same-shape-as-image mask that would misrepresent it) + `decision: NavigationDecision` (one global enum value derived from the whole grid).

`RegionStats` (`traversability.types.RegionStats`) per-region fields include `classification` (`RegionClass` enum: `CLEAR`, `OBSTACLE`, `PROBABLE_WALL`, `LOW_TEXTURE_UNKNOWN`, `LOW_CONFIDENCE`, `UNKNOWN`), `depth_median_m`, `confidence` (`[0,1]`), `valid_pct`/`invalid_pct`, `valid_count`/`total_pixels` (absolute counts, added during the mp01_perception safety remediation because `valid_pct` alone is a lossy region-size-relative percentage not safely comparable across regions of different sizes).

`UNKNOWN` is a **hard gate**: any region with fewer valid disparity pixels than `RegionAnalyzer`'s `min_valid_pixels` threshold is unconditionally `UNKNOWN`, checked before any other signal (texture/entropy/confidence) — a region with zero stereo evidence can never be reported as any other class, including `CLEAR`. `SceneInterpreter` treats `UNKNOWN` as ambiguous; a region grid containing an `UNKNOWN` region directly ahead can never itself produce a `MOVE_FORWARD` navigation decision.

### `ObstacleAssessment`

`beams: List[BeamReading]` + `safest_beam: Optional[BeamReading]`. Each `BeamReading`: `index`, `x1`/`x2` (pixel column bounds), `distance_m` (`0.0` = no data), `status` (one of `ThreatAssessor.CLEAR`/`CAUTION`/`BLOCKED`/`NO_DATA`, plain strings not an enum — pre-existing, not changed in this pass). A beam with no valid depth but a high fraction of invalid (blocked) disparity pixels is reclassified `BLOCKED` rather than `NO_DATA` — catches textureless obstacles (e.g. a blank wall) that produce no stereo correspondence at all regardless of distance. Status changes are debounced (must persist `debounce_frames` consecutive calls) **except** when depth evidence disappears entirely (`evidence_lost`), which bypasses debounce immediately — a stale `CLEAR` must never keep being reported once the evidence behind it is gone.

## `PipelineHealth` (`models.PipelineHealth`)

Added in this recovery pass. `is_closed: bool`, `frames_processed: int`, `last_confidence: Optional[float]`, `last_processing_time_ms: Optional[float]` — the last two are `None` until `process()` has been called at least once since construction or the last `reset()`. Reports the pipeline's own lifecycle state, not a re-diagnosis of the scene.

## Coordinate frames

All image-space coordinates (`x1`/`x2` beam bounds, region grid pixel bounds) are in the **rectified left image's** pixel space — the frame everything downstream of `RectificationEngine.rectify()` operates in. `depth_map`/disparity values are per-pixel scalars, not associated with a 3D coordinate frame — this library does not currently expose X/Y (only Z/depth); `DepthEstimator.estimate_point_cloud()` exists for the rare caller that wants the full `(H, W, 3)` X/Y/Z point cloud, in the camera's own rectified-left optical frame (OpenCV convention: X right, Y down, Z forward), metres.
