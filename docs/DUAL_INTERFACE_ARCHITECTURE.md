# DPE Dual-Interface Architecture

**Status:** structural refactor only. No DPE algorithm, threshold, calibration
mathematic, geometric result, or `GeometryFrame` semantic was changed by this
pass — see "What did NOT change" at the bottom.

DPE is consumed through exactly **two** supported interfaces. They differ only
in *how a valid input reaches the engine*, never in what the engine then does.

```
STANDALONE DEVELOPMENT / QUALIFICATION

    raw/convenient stereo + calibration path + optional raw motion samples
                                   |
                                   v
                  standalone.StandaloneStereoInterface
                                   |
                          input adaptation only
                                   |
                                   v
        DepthPerceptionPipeline.process_observation()   <-- THE core
                                   |
                                   v
                             GeometryFrame


EMBEDDED INTEGRATION (a future hybrid_perception_engine)

    consumer-owned observation, already prepared
                                   |
                                   v
        DepthPerceptionPipeline.process_geometry_frame()
              -> process_observation()                  <-- THE SAME core
                                   |
                                   v
                             GeometryFrame
```

---

## 1. DPE remains a standalone geometry engine

`depth_perception_engine` is still an independently installable, independently
runnable, ROS-free library. `pytest tests/`, every `examples/` script, and every
`benchmarks/` suite run without any external perception system present. Nothing
in this refactor makes DPE a component that only works inside something else.

## 2. It has a standalone convenience / sensor-input interface

`depth_perception_engine.standalone.StandaloneStereoInterface` — the formalized,
supported public entry point for running DPE by itself. It owns exactly the four
sensor-facing conveniences the core must not own:

| Convenience | Method | Previously lived in |
|---|---|---|
| Calibration **file** loading | `from_calibration_file(path, config)` | `calibration.load_stereo_calibration` called ad hoc by every example/test |
| Combined side-by-side frame splitting | `split_combined_frame()`, `process_combined_frame*()` | `stereo.FrameSplitter`, wired by hand in `examples/live_demo.py` |
| Raw angular-rate normalization | `build_motion_hint()`, `build_motion_hints()` | constructed by hand at each call site |
| Loose per-frame arguments → canonical input | `build_observation()`, `process()`, `process_geometry_frame()` | `DepthPerceptionPipeline.process(left, right, ...)` |

It implements **no** geometry. Enforced structurally by
`tests/test_dual_interface_architecture.py::TestNoDuplicatedProcessing`, which
AST-scans the subpackage and fails if it imports any algorithm module or calls
any geometry builder.

## 3. It has a clean embedded / core public interface

`depth_perception_engine.core` names the complete embedded contract in one
place:

```python
from depth_perception_engine.core import (
    DepthPerceptionPipeline,   # the engine
    PipelineConfig,            # configuration
    StereoCalibration,         # calibration value object
    RigidTransform,            # optional camera->body extrinsic
    StereoObservation,         # THE canonical core input
    MotionHint,                # optional normalized motion input
    GeometryFrame,             # THE authoritative output
    FrameId,                   # frame-name vocabulary
)
```

Every symbol there is the *same object* the package root already exported — the
namespace introduces no new class and no wrapper. What it adds is a structurally
verifiable boundary: it imports nothing from `standalone`, and it deliberately
does **not** re-export `load_stereo_calibration` or `FrameSplitter`.

The one genuinely new core method is
`DepthPerceptionPipeline.process_geometry_frame(observation) -> GeometryFrame`,
so an embedded consumer never has to reach through the legacy
`DepthPerceptionResult` wrapper to obtain DPE's authoritative output.

## 4. An embedded consumer should use ONLY the core interface

```python
from depth_perception_engine.core import (
    DepthPerceptionPipeline, PipelineConfig, StereoObservation, GeometryFrame,
)

pipeline = DepthPerceptionPipeline(config, calibration)          # construct once
observation = StereoObservation(                                  # consumer-prepared
    left_image=left, right_image=right, left_timestamp=t,
    motion_hint=hint, motion_hints=[hint],
)
geometry: GeometryFrame = pipeline.process_geometry_frame(observation)   # per frame
```

No standalone adapter, no DPE internals, no algorithm stage invoked
individually, no ROS, no DPE test utilities.

## 5. Standalone adapters are NOT part of an embedded runtime path

This is structural, not a flag. There is no `standalone_mode`,
`sensor_interface_enabled`, or `hpe_mode` boolean anywhere in DPE. An embedded
consumer simply never imports `depth_perception_engine.standalone`, so that
layer is **absent** from its process — proven in a fresh interpreter by
`TestStandaloneOptionality`, which constructs and runs the core and then asserts
`depth_perception_engine.standalone` is not in `sys.modules`. The package root's
own re-export of `StandaloneStereoInterface` is deliberately lazy (PEP 562
module `__getattr__`) so that even `import depth_perception_engine` does not
pull the standalone layer in.

The dependency direction is strictly one-way: **standalone → core**. No module
inside the library imports the standalone subpackage at module level (enforced
by an AST scan over all of `src/`).

## 6. Both interfaces reach the same processing implementation

`DepthPerceptionPipeline.process_observation()` is the single geometry
implementation. Everything else is a funnel into it:

- `DepthPerceptionPipeline.process(left, right, ...)` now *builds* a
  `StereoObservation` and calls `process_observation()` (before this pass the
  relationship was inverted — `process_observation()` unpacked into `process()`).
- `DepthPerceptionPipeline.process_geometry_frame(obs)` calls
  `process_observation()` exactly once and then the same
  `_build_geometry_frame()` helper that `process_observation()` itself uses.
- `StandaloneStereoInterface.process*()` builds a `StereoObservation` and calls
  the engine it holds.

Guarantees that this cannot drift:

1. **One implementation, textually.** There is exactly one body; the other entry
   points are three-line delegations.
2. **A shared-engine test.** `TestNoDuplicatedProcessing` hands the *same*
   `DepthPerceptionPipeline` instance to both interfaces and shows the engine's
   own frame counter advancing once per call from either side.
3. **A delegation spy.** A pass-through recorder wrapping the real engine proves
   the standalone interface forwards a `StereoObservation` and nothing else.
4. **Output equivalence over real algorithms** (below).

## 7. `GeometryFrame` remains the single authoritative output

There is no `StandaloneGeometryFrame` and no consumer-specific frame type, and
none may be added. Both paths return the same `GeometryFrame` class with the
same 22 fields and the same semantics frozen at Phase D2-D8. Equivalence is
proven field-for-field, recursively, across the whole type graph — including
NaN-aware array comparison — over a multi-frame sequence with every evidence
family enabled (geometry, body geometry, obstacle cloud, free-space rays,
geometry metrics, region/clearance/surface/boundary/opening evidence, all five
temporal fields, and quality). See
`tests/test_dual_interface_architecture.py::TestOutputEquivalence`.

`DepthPerceptionResult.geometry_frame` keeps its exact prior meaning: a
migration/compatibility field gated by `PipelineConfig.enable_geometry_frame`
(default `False`). `process_geometry_frame()` returns a frame regardless of that
flag, because *calling it is* the request for one — and both branches call the
identical builder with identical arguments, which is itself asserted as a test.

## 8. DPE remains platform-agnostic

No aerial/ground/marine/rover/drone/boat concept, no vehicle size, no platform
identity, and no planning constraint appears in either interface. The canonical
input carries only observation identity, timestamps, the stereo pair, and
optional normalized motion. Enforced unchanged by
`tests/test_level4_architecture_guards.py`, which scans all of `src/` including
the two new subpackages.

## 9. DPE has no knowledge of NPE / embedded-consumer fusion

DPE imports nothing from any external perception system, contains no fusion
behavior, no semantics, no neural inference, and no consumer-specific
accommodation. `GeometryFrame` was already designed as a neutral provider
contract (Phase D2-D8); this pass added an entry point to it, not a field.

## 10. Concurrency is not DPE's responsibility

No thread, process, queue, lock, executor, async construct, or shared-memory
mechanism was added. Both interfaces are synchronous and single-threaded; frame
scheduling belongs to whatever runs DPE.

---

## Canonical input contract

`models.StereoObservation` — pre-existing, not invented here. It already carried
exactly what DPE requires, so nothing was added to it:

| Field | Meaning |
|---|---|
| `left_image` / `right_image` | the stereo pair, used by reference |
| `left_timestamp` / `right_timestamp` | opaque caller-defined floats |
| `motion_hint` | optional normalized `MotionHint` for temporal-record association |
| `motion_hints` | optional bounded sequence for rotation compensation |
| `observation_id` | **Phase D2** — opaque caller-owned observation/transaction identity, copied verbatim onto `GeometryFrame.observation_id`; never interpreted |
| `frame_id` | **deprecated** alias of `observation_id` (was reserved+unread pre-D2; now supplies identity when `observation_id` is `None`) |
| `calibration` | reserved for future multi-rig use (unread; the pipeline's own calibration is always used) |

The core never consumes a ROS `sensor_msgs/Imu` or any raw device message. The
standalone layer adapts convenient forms — `(timestamp, (wx, wy, wz))` or
`(timestamp, wx, wy, wz)` — into this same `MotionHint`; an embedded consumer
performs its own routing into the identical contract. Motion **mathematics**
(integration, compensation, reliability, persistence) exists in exactly one
place: the core's E5/E6/E7 stages, untouched.

## Zero added copies

Images pass by reference from a caller, through `StereoObservation`, into the
core. `split_combined_frame()` returns NumPy *views* (plain slicing). Nothing in
either interface calls `.copy()`, reshapes, or reconstructs an array — asserted
by `test_split_returns_views_not_copies` and
`test_observation_holds_the_caller_arrays_by_reference`.

## What did NOT change

Disparity, depth, SGBM parameters, obstacle extraction, surfaces, normals,
boundaries, openings, clearance, temporal algorithms, reliability logic,
thresholds, calibration mathematics, quality semantics, `GeometryFrame`
semantics, `PipelineConfig` fields, and every existing public symbol. The
existing regression suite passes unchanged and unweakened; no existing test was
skipped, loosened, or deleted.

## Known pre-existing item, reported not fixed

`pipeline.api.process_stereo_pair` (Tier 2, stateless) inlines its own
Level 0-2 sequence — rectify → disparity → depth → scene → threat →
`build_result` — rather than delegating to `DepthPerceptionPipeline`. That
predates this refactor, is documented Tier 2 behavior (fresh engines per call,
deliberately no cross-frame smoothing), and produces **no** Level 3/4 geometry
and **no** `GeometryFrame`, so it is not a second geometry engine. It was left
untouched: collapsing it into the core would change behavior, which this
architecture-only stage forbids. Flagged here so a future pass can decide
deliberately.
