# Integration Readiness — mp01_perception (ROS2)

> **STATUS UPDATE (2026-08-05, baseline recovery pass):** the integration
> described below has since actually happened — `mp01_perception`'s
> `PerceptionProcessor`/`PerceptionNode` already construct one
> `DepthPerceptionPipeline` and call `.process()` per frame, not the toy
> pass-through shown in this doc's original "today" snippet. This document
> is kept as-is below for its historical design rationale (still accurate —
> the actual integration followed this plan) rather than rewritten; see
> `docs/IMPLEMENTATION_STATUS.md` in this repo for what's current.
>
> **STATUS UPDATE (2026-08-05, public API freeze pass):** §2's code example
> below has been corrected from subpackage-style imports to the canonical
> top-level form — `load_stereo_calibration`/`PipelineConfig`/
> `DepthPerceptionPipeline` were always top-level exports; showing them
> imported from `.calibration`/`.config`/`.pipeline` was itself an instance
> of the exact import ambiguity `docs/PUBLIC_API.md` now resolves. Verified
> against `mp01_perception`'s actual, current `perception_processor.py`
> this pass: it already uses the top-level form independently — this
> doc's example now matches real usage, not just the original plan.

Status (as originally written): this library is ready to be imported. **No
integration has been performed yet** — `mp01_ws` is untouched. This document
describes exactly what that integration will look like when it happens, so
it can be done without re-deriving the design.

## The seam, recap

`mp01_perception` (MP-01's ROS2 perception package, a separate repository)
already documents its own integration seam: replacing the body of
`PerceptionProcessor.process()` in
`mp01_perception/mp01_perception/perception_processor.py` is the **only**
change its ROS pipeline expects. Every other file — `StereoSubscriber`,
`ImageAdapter`, `PerceptionPublisher`, `Diagnostics`, `PerceptionNode`, the
launch file, the config YAML — stays as-is, because none of them know or
care what `PerceptionProcessor.process()` does internally.

Today, that method is a pass-through:

```python
class PerceptionProcessor:
    def process(self, left_image, right_image) -> PerceptionResult:
        return PerceptionResult(left_image=left_image, right_image=right_image)
```

This document is about what replaces it.

## 1. Install this library into the ROS2 workspace's Python environment

```bash
pip install -e /path/to/depth_perception_engine
```

Editable install (`-e`) so changes to this repo are picked up without
reinstalling — matches how `mp01_perception` itself is already built
(`ament_python`, installed in development mode via `colcon build`).
`depth_perception_engine`'s only runtime dependencies are `numpy` and
`opencv-python-headless` — both already present in any ROS2 Humble
environment that has `cv_bridge` working, so this adds no new system
dependency.

## 2. Build one `DepthPerceptionPipeline`, once — not per frame

`PerceptionProcessor.__init__` is where this belongs, mirroring how
`PerceptionNode.__init__` already builds `ImageAdapter`, `PerceptionProcessor`,
and `Diagnostics` once:

```python
from depth_perception_engine import (
    DepthPerceptionPipeline,
    PipelineConfig,
    load_stereo_calibration,
)


class PerceptionProcessor:
    def __init__(self, calibration_file: str):
        calibration = load_stereo_calibration(calibration_file)
        self._pipeline = DepthPerceptionPipeline(PipelineConfig(), calibration)

    def process(self, left_image, right_image) -> PerceptionResult:
        result = self._pipeline.process(left_image, right_image)
        ...
```

**Why once, not per call:** `DepthPerceptionPipeline` holds a persistent
`obstacles.ThreatAssessor` internally, which EMA-smooths and debounces
obstacle readings *across frames*. Constructing a fresh pipeline (or using
the stateless `pipeline.api.process_stereo_pair()` function) on every frame
throws that smoothing away and reproduces raw per-frame SGBM noise — this
was true of the original desk-test pipeline too (`main.py`'s
`build_pipeline()` was called once, not per frame), and it carries over
unchanged into this library. See `pipeline/api.py`'s module docstring.

`calibration_file` needs to be a **real calibration for mp01_camera's
actual stereo rig**, not `examples/config/stereo_calibration.xml` (that
file is this repo's own desk-test hardware fixture). It should come in as
a `perception_node` ROS parameter (the same pattern `mp01_perception`
already uses for `input_topic_namespace` etc. in `perception.yaml`), not a
hardcoded path — this library never assumes or defaults to one (see
`calibration/loader.py`'s docstring).

## 3. Per-frame call — the actual `process()` body swap

```python
def process(self, left_image: np.ndarray, right_image: np.ndarray) -> PerceptionResult:
    result = self._pipeline.process(left_image, right_image)
    # map result.* onto whatever mp01_perception decides to publish —
    # see "Output mapping" below for what's available.
    return PerceptionResult(left_image=..., right_image=...)
```

`left_image`/`right_image` are already the plain NumPy arrays
`mp01_perception`'s `ImageAdapter.to_numpy()` produces from the incoming
`sensor_msgs/Image` pair — no format conversion is needed at this boundary.
`DepthPerceptionPipeline.process()` never imports `rclpy`, never touches a
ROS message, and never opens a camera — it only ever sees NumPy arrays in,
a `DepthPerceptionResult` out.

## 4. Output mapping — what `DepthPerceptionResult` carries

| Field | Type | A ROS-side home for it (illustrative — mp01_perception's decision, not this repo's) |
|---|---|---|
| `disparity_map` | `np.ndarray` float32 (H, W) | Debug-only topic, or dropped if unused downstream |
| `depth_map` | `np.ndarray` float32 (H, W), metres | `sensor_msgs/Image` (32FC1) on a new `/perception/front/depth` topic |
| `traversability_mask` | `TraversabilityResult` (per-region grid + `NavigationDecision`) | A new small message type, or flattened into `diagnostic_msgs/KeyValue` pairs if a custom `.msg` isn't wanted yet |
| `obstacles` | `ObstacleAssessment` (per-beam scan) | Same — new message type, or `DiagnosticArray` values for a first pass |
| `confidence` | `float`, 0..1 | A `KeyValue` in `mp01_perception`'s existing `/diagnostics` publish, alongside `fps`/`latency_ms`/`dropped_frames` |
| `processing_time_ms` | `float` | Same — folds naturally into the existing diagnostics cycle; compare against `mp01_perception`'s own measured `latency_ms` to see the real algorithm's added cost over the Layer 1 pass-through floor |

None of this requires a new topic or message type on day one — `confidence`
and `processing_time_ms` alone are enough to prove the wiring works,
published as additional `KeyValue`s on the diagnostics `PerceptionNode`
already publishes.

## 5. What does NOT change in `mp01_perception`

Per `mp01_perception`'s own `docs/FUTURE_INTEGRATION.md`:

- `StereoSubscriber` — still just subscribes and time-synchronizes.
- `ImageAdapter` — still just `sensor_msgs/Image ⇄ NumPy`.
- `PerceptionPublisher` — gains new `create_publisher()` calls only if new
  topics are added (e.g. for the depth map); its existing publish-ordering
  guarantee (build every message before publishing any) is unaffected.
- `Diagnostics` — unchanged; already times the whole `_on_stereo_pair`
  cycle, so it will automatically reflect the new processing cost.
- `PerceptionNode`, the launch file, `perception.yaml` — unchanged, aside
  from possibly adding a `calibration_file` parameter (see §2).

## 6. Performance expectation

`mp01_perception`'s validated Layer 1 pass-through floor is ~1.5–2.6 ms at
60 fps (see its `docs/PERFORMANCE_REPORT.md`). This library's own
`processing_time_ms` on a 320×240 stereo pair, measured in this repo's own
test/example runs, is in the tens of milliseconds — dominated by
`StereoSGBM.compute()`. Total per-frame latency once integrated will be
roughly `mp01_perception`'s existing floor **plus** this library's
`processing_time_ms`, not a replacement for it — budget accordingly against
the target frame rate; the existing `Diagnostics.fps` measurement will show
directly whether it's sustainable at 60 fps on target hardware (Jetson Orin
Nano) once actually wired in.

## 7. Suggested integration order

1. Add `calibration_file` (and, if needed, `rectify: bool`) parameters to
   `perception.yaml` / `PerceptionNode`.
2. `pip install -e` this library into the ROS2 workspace's environment;
   add it to `mp01_perception/package.xml` as a dependency once packaged
   for `rosdep`, or document the manual `pip install -e` step for now.
3. Swap `PerceptionProcessor`'s constructor and `process()` body as in §2–3.
4. Extend `PerceptionResult` (or replace it) with whatever subset of
   `DepthPerceptionResult`'s fields `PerceptionPublisher` will publish.
5. Re-run `mp01_perception`'s existing test suite and validation sequence
   (`docs/VALIDATION_REPORT.md`'s ten-point checklist) — every check other
   than "pass-through validation" (which necessarily changes, since the
   output is no longer identity) should still pass unmodified.
