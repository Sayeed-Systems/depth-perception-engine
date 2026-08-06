# Public API

The authoritative reference for what is, and is not, this library's stable public contract. Supersedes any older doc's implicit assumptions about import style — where another doc's code example disagrees with this one, this one is correct (and the other should be treated as needing a fix, not as an alternate truth).

## Naming: repository vs. class

- **Repository/library name:** `depth_perception_engine`.
- **Canonical processing object:** `DepthPerceptionPipeline`.

These are deliberately different names for different things — the repository is not named after its one high-level class, and the class is not renamed to match the repository. There is no `DepthPerceptionEngine` symbol anywhere in this codebase, and none is planned. An alias under that name was considered and rejected: `DepthPerceptionPipeline` is already the coherent, tested, hardware-verified name every existing caller (including `mp01_perception`'s real, current code) already uses — introducing a second name for the same class would be naming churn with no technical justification, exactly the outcome this document exists to prevent.

## The canonical import

```python
from depth_perception_engine import (
    DepthPerceptionPipeline,
    PipelineConfig,
    StereoObservation,
    load_stereo_calibration,
)

calibration = load_stereo_calibration("/path/to/stereo_calibration.xml")
pipeline = DepthPerceptionPipeline(PipelineConfig(), calibration)   # build once, reuse across frames
```

Two ways to call it, both real, both documented — do not assume `.process()` accepts a `StereoObservation` directly; it does not:

```python
result = pipeline.process(left_image, right_image)          # primary: two plain NumPy arrays
# or, if you already have a StereoObservation:
result = pipeline.process_observation(observation)           # observation: StereoObservation
```

`mp01_perception`'s actual `perception_processor.py` uses the first form today (`from depth_perception_engine import (DepthPerceptionPipeline, DepthPerceptionResult, PipelineConfig, load_stereo_calibration)` then `.process(left, right)`) — already, independently, exactly Tier-1-only, verified by direct inspection of its source this pass.

## Tier 1 — primary stable API

The only tier `mp01_perception` and normal users should depend on. All top-level, all in `__all__`.

| Symbol | Defined in | Why it's Tier 1 |
|---|---|---|
| `DepthPerceptionPipeline` | `pipeline/pipeline.py` | The one entry point |
| `PipelineConfig` | `config/pipeline_config.py` | Required to construct a pipeline |
| `StereoCalibration` | `calibration/models.py` | Required to construct a pipeline |
| `load_stereo_calibration` | `calibration/loader.py` | The only way to obtain a `StereoCalibration` from a file |
| `StereoObservation` | `models/result.py` | Alternate input shape for `process_observation()` |
| `DepthPerceptionResult` | `models/result.py` | The one output shape |
| `PipelineHealth` | `models/result.py` | `DepthPerceptionPipeline.health()`'s return type |
| `TraversabilityResult` | `models/result.py` | `DepthPerceptionResult.traversability_mask`'s type |
| `ObstacleAssessment` | `models/result.py` | `DepthPerceptionResult.obstacles`'s type |
| `BeamReading` | `models/result.py` | `ObstacleAssessment.beams`'s element type |
| `NavigationDecision` | `traversability/types.py` | `TraversabilityResult.decision`'s type |
| `RegionClass` | `traversability/types.py` | `RegionStats.classification`'s type |
| `RegionStats` | `traversability/types.py` | `TraversabilityResult.regions`'s value type |
| `TextureClass` | `traversability/types.py` | `RegionStats.texture_class`'s type |

The last four were **promoted to Tier 1 by this pass** — previously only reachable via `depth_perception_engine.traversability.types`, an internal-looking path. This was not a style preference: `mp01_perception`'s real, current code (`validity_gate.py`, `perception_publisher.py`) already imports `RegionClass`/`NavigationDecision`/`TextureClass` from that internal path today, because there was no top-level alternative. `DepthPerceptionResult.traversability_mask` structurally embeds these types — a caller cannot compare `result.traversability_mask.decision` against anything, or type-hint `result.traversability_mask.regions: Dict[str, RegionStats]`, without them. Promoting them closes a real, proven gap rather than anticipating a hypothetical one.

`RegionAnalyzer`, `SceneInterpreter`, `SceneState` (the classes that *produce* these types) remain Tier 3 — a caller receives and reads `RegionStats`/`NavigationDecision` values, it does not construct a `SceneInterpreter` itself.

## Tier 2 — advanced functional API

Public, top-level, documented — for research and isolated single-stage testing, not for a normal integration:

| Symbol | Defined in |
|---|---|
| `process_stereo_pair` | `pipeline/api.py` |
| `compute_disparity` | `pipeline/api.py` |
| `estimate_depth` | `pipeline/api.py` |
| `detect_obstacles` | `pipeline/api.py` |
| `classify_traversability` | `pipeline/api.py` |

Each is a stateless, single-call wrapper around one pipeline stage — useful for a script or test that wants (say) just disparity without a full `DepthPerceptionPipeline`. **Do not use these for a video stream**: they construct fresh stage objects on every call, so `detect_obstacles`' `ThreatAssessor` never accumulates the EMA/debounce state that makes its output stable across frames — that's the entire reason `DepthPerceptionPipeline` (Tier 1) exists as a persistent object instead. Contracts and test coverage: `tests/test_api_functions.py`.

## Tier 3 — internal

Everything not listed above. Examples: `RegionAnalyzer`, `SceneInterpreter`, `DisparityEngine`, `RectificationEngine`, `DepthEstimator`, `DistanceReader`, `ThreatAssessor`, `FrameSplitter`, `looks_like_garbage_frame`, `frames.RigidTransform`, `frames.FrameId`, `calibration.contracts.*` (`CameraIntrinsics`, `StereoExtrinsics`, `RectificationParameters`, `RigCalibration`, `CameraModel`), `geometry.*` (`PointCloud`, `ObstacleCloud`, `FreeSpaceRays`, `GeometryMetrics`), `utils.*`, `fusion.result_builder.*`.

**Submodules remain importable** (`depth_perception_engine.pipeline`, `depth_perception_engine.traversability`, etc. — some, like `depth_perception_engine.utils`, are even reachable as bare attributes after `import depth_perception_engine`, an unavoidable consequence of Python binding parent-package attributes whenever a submodule is imported anywhere in the import chain — verified this pass via `sys.modules` inspection). **This is acceptable and not something to engineer around** — but none of it is part of the documented, stable contract, and no consumer should be instructed to rely on it. `utils` specifically is never in `__all__` and never will be.

Level 3 contracts (`frames.*`, `calibration.contracts.*`, `geometry.*`, added in the Phase E1 pass) are Tier 3 by explicit design — see `docs/LEVEL3_PUBLIC_API.md` for why they're deliberately not promoted: nothing produces one yet, and promoting an unproduced type would overclaim capability.

## Backward compatibility

Existing subpackage imports keep working — nothing was removed:

```python
from depth_perception_engine.pipeline import DepthPerceptionPipeline    # still works
from depth_perception_engine.calibration import StereoCalibration       # still works
```

These are **compatibility paths**, not the canonical style for new code — the top-level import is canonical. Every duplicated path resolves to the exact same object (proven, not assumed — see `tests/test_public_api.py::TestImportIdentity`):

```python
from depth_perception_engine import DepthPerceptionPipeline as A
from depth_perception_engine.pipeline import DepthPerceptionPipeline as B
assert A is B
```

No symbol is ever defined twice — every top-level export is a plain re-import of the one class/function/type already defined in its home module.

## API stability policy / semantic versioning expectations

- **Tier 1 and Tier 2** are covered by semantic versioning as of `0.1.0`: a breaking change to a Tier 1/2 symbol's name, shape, or behavior requires a major version bump (post-1.0) or an explicit, documented migration note (pre-1.0, where breaking changes are still expected occasionally per SemVer's own pre-1.0 carve-out). Adding a new Tier 1/2 symbol, or a new optional field/parameter with a backward-compatible default (the pattern every extension this repository has made so far has followed — see `docs/VALIDATION_REPORT.md`, `docs/LEVEL3_CONTRACTS.md`), is a minor-version-level change.
- **Tier 3** carries no stability guarantee whatsoever — internal modules may be renamed, restructured, or removed without notice, precisely because nothing external is meant to depend on them. If you find yourself importing from Tier 3, that is itself a signal the symbol you need should be requested as a Tier 1/2 promotion instead (as happened with `RegionClass`/`NavigationDecision`/`TextureClass`/`RegionStats` in this pass).
- Compatibility subpackage-path imports (`from depth_perception_engine.pipeline import ...` for a Tier 1/2 symbol) are held to the same stability guarantee as the top-level path for as long as they're documented here as valid — they are not deprecated, only non-canonical.

## Proof: a future `mp01_perception` wrapper needs only Tier 1

```python
from depth_perception_engine import (
    DepthPerceptionPipeline,
    PipelineConfig,
    StereoObservation,
    DepthPerceptionResult,
    load_stereo_calibration,
)
```

is sufficient to construct a pipeline, process frames, and read every field of the result — including `result.traversability_mask.decision`/`.regions[...].classification` and `result.obstacles.beams[...].status` — without reaching into `depth_perception_engine.traversability.types`, `.quality.frame_quality`, `.depth.depth_estimator`, `.stereo.disparity_engine`, or `.utils`. Proven by `tests/test_public_api.py::TestNoInternalImportNeededForFullResultInterpretation`, which constructs a real `DepthPerceptionResult` and reads every one of its fields using only symbols imported from the package root.

**Known, real boundary violations in `mp01_perception`'s current code** (verified by direct inspection this pass — not fixed here, since modifying `mp01_perception` is out of this task's scope):

| File | Internal import | Status |
|---|---|---|
| `validity_gate.py`, `perception_publisher.py` | `depth_perception_engine.traversability.types.{RegionClass, NavigationDecision, TextureClass}` | **Resolved on the engine side** — all three (plus `RegionStats`) are now Tier 1. `mp01_perception` can switch to the top-level import whenever it's next touched; the internal path still works unchanged in the meantime. |
| `calibration_integrity.py` | `depth_perception_engine.calibration.models.StereoCalibration` | Not resolved — `StereoCalibration` was *already* top-level; this is a redundant, avoidable internal-path import on `mp01_perception`'s side, not an engine-side gap. |
| `frame_validation.py` | `depth_perception_engine.quality.frame_quality.looks_like_garbage_frame` | Not resolved — `looks_like_garbage_frame` is a genuine internal utility, not named in this task's Tier 1/Tier 2 lists. Remains a real, pre-existing boundary item (first flagged in the previous baseline-recovery pass) for a future task to decide: promote it to Tier 2, or accept it as a deliberate, narrow exception. |

`perception_processor.py` requires no changes — already exactly Tier-1-only.
