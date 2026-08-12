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
| `GeometryFrame` | `geometry/provider.py` | **Phase D2.** The final, authoritative DPE V1 provider contract — see `docs/DPE_V1_PROVIDER_CONTRACT.md` |
| `TemporalConsistency` / `TemporalConsistencyState` | `temporal/types.py` / `temporal/consistency.py` | **Phase D2.** `GeometryFrame.temporal_consistency`'s type |
| `TemporalStabilization` / `TemporalStabilizationState` | `temporal/types.py` / `temporal/stabilization.py` | **Phase D2.** `GeometryFrame.temporal_stabilization`'s type |
| `RotationCompensationStatus` | `temporal/rotation_compensation.py` | **Phase D2.** `GeometryFrame.rotation_compensation_status`'s value set |
| `MotionAwareReliability` / `MotionAwareReliabilityState` | `temporal/types.py` / `temporal/reliability.py` | **Phase D2.** `GeometryFrame.motion_aware_reliability`'s type |
| `TemporalPersistence` / `TemporalPersistenceState` / `TemporalPersistenceCellState` | `temporal/types.py` / `temporal/persistence.py` | **Phase D2.** `GeometryFrame.temporal_persistence`'s type |
| `PointCloud` | `geometry/types.py` | **Phase D3.** `GeometryFrame.geometry`/`.geometry_body`'s type |
| `ObstacleCloud` | `geometry/types.py` | **Phase D3.** `GeometryFrame.obstacle_cloud`'s type |
| `FreeSpaceRays` | `geometry/types.py` | **Phase D3.** `GeometryFrame.free_space_rays`'s type |
| `GeometryMetrics` | `geometry/types.py` | **Phase D3.** `GeometryFrame.geometry_metrics`'s type |
| `RegionEvidence` | `geometry/provider.py` | **Phase D3, `frame_id` added D8.** Neutral per-region geometric evidence (extracted from `RegionAnalyzer`'s output, without `RegionClass`) — `GeometryFrame.region_evidence`'s value type |
| `ClearanceEvidence` | `geometry/provider.py` | **Phase D3, refined D7.** Neutral per-sector directional-distance evidence — coverage/support and calibrated bearing added in D7 — `GeometryFrame.clearance_evidence`'s element type |
| `SurfaceEvidence` | `geometry/surface.py` | **Phase D4.** Local surface-normal/planarity evidence, deterministically fit from `geometry_body` — `GeometryFrame.surface_evidence`'s element type. The first genuinely new geometric algorithm added since D1; see `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D4 record |
| `BoundaryEvidence` | `geometry/boundary.py` | **Phase D5.** Depth/surface-orientation discontinuity evidence between adjacent cells — `GeometryFrame.boundary_evidence`'s element type |
| `BoundaryState` | `geometry/boundary.py` | **Phase D5.** `BoundaryEvidence.state`'s value set (OBSERVED_DISCONTINUITY/NO_DISCONTINUITY/INSUFFICIENT_EVIDENCE) |
| `BoundaryDirection` | `geometry/boundary.py` | **Phase D5.** `BoundaryEvidence.direction`'s value set (RIGHT/DOWN) |
| `OpeningEvidence` | `geometry/opening.py` | **Phase D6.** A confirmed geometrically supported gap between flanking structures — `GeometryFrame.opening_evidence`'s element type |
| `ClearanceSupportState` | `geometry/provider.py` | **Phase D7.** `ClearanceEvidence.support_state`'s value set (SUPPORTED/PARTIALLY_SUPPORTED/NO_EVIDENCE) |
| `GeometryFrameQuality` | `geometry/provider.py` | **Phase D8.** Structured geometric quality/uncertainty rollup — `GeometryFrame.quality`'s type |
| `GeometryFrameQualityState` | `geometry/provider.py` | **Phase D8.** Shared value set every `GeometryFrameQuality` dimension uses (VALID/DEGRADED/INSUFFICIENT) |
| `FrameId` | `frames.py` | **Phase D10.** The canonical public vocabulary for every `frame_id` string value across `GeometryFrame`'s own type graph (`CAMERA_OPTICAL_LEFT`/`BODY`) — D9 found it was the one categorical-value source never promoted, unlike every other state-bearing field's own constant class; promoting it was a pure API/export/test/doc change, zero behavioral change. `frames.RigidTransform` stays Tier 3 — a pipeline constructor input, never part of `GeometryFrame`'s own output type graph. See `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D9/D10 records |
| `MotionHint` | `temporal/types.py` | **Phase D13.** Required to construct a complete public INPUT contract — `StereoObservation.motion_hint`/`.motion_hints`, and `DepthPerceptionPipeline.process()`'s own `motion_hint`/`motion_hints` parameters, are typed against it. Not part of `GeometryFrame`'s own output type graph (D2 deliberately did not promote it for that reason — it is a pipeline INPUT, not an evidence type `GeometryFrame` carries) — promoted on the input side instead, symmetric to how D10 promoted `FrameId` on the output side. Pure API/export/test/doc change; `MotionHint`'s own shape/validation/consumption is completely unchanged. See `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D13 record |

`NavigationDecision`/`RegionClass`/`RegionStats`/`TextureClass` were **promoted to Tier 1 by** the public-API-freeze pass — previously only reachable via `depth_perception_engine.traversability.types`, an internal-looking path. This was not a style preference: `mp01_perception`'s real, current code (`validity_gate.py`, `perception_publisher.py`) already imports `RegionClass`/`NavigationDecision`/`TextureClass` from that internal path today, because there was no top-level alternative. `DepthPerceptionResult.traversability_mask` structurally embeds these types — a caller cannot compare `result.traversability_mask.decision` against anything, or type-hint `result.traversability_mask.regions: Dict[str, RegionStats]`, without them. Promoting them closes a real, proven gap rather than anticipating a hypothetical one.

`GeometryFrame` and the ten temporal symbols after it were **promoted to Tier 1 by Phase D2** for the identical reason: `GeometryFrame` is DPE's own final V1 provider contract (`docs/DPE_V1_PROVIDER_CONTRACT.md`), and four of its fields are typed against those temporal result classes — a caller cannot meaningfully interpret `geometry_frame.temporal_persistence.state` without `TemporalPersistenceState` being reachable. `TemporalHistory`, `TemporalRecord`, `TemporalAdmissionStatus`, `TemporalPersistenceTracker`, and every `compute_*` algorithm function were deliberately **not** promoted — they are chronology bookkeeping, stateful internal collaborators, or algorithm implementations, none of which `GeometryFrame` itself exposes; see `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D2 record for the full boundary. `MotionHint` was also deliberately not promoted at D2 (it is a pipeline INPUT type, not one of `GeometryFrame`'s own output fields) — **Phase D13 later promoted it anyway**, not as an output-type-graph member but as one of the four named Tier 1 INPUT contracts (see the `MotionHint` table row above and `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D13 record).

`PointCloud`/`ObstacleCloud`/`FreeSpaceRays`/`GeometryMetrics` were **promoted to Tier 1 by Phase D3**, resolving the gap D2 flagged above: `GeometryFrame.geometry`/`.geometry_body`/`.obstacle_cloud`/`.free_space_rays`/`.geometry_metrics` are typed against them. `RegionEvidence`/`ClearanceEvidence` are **new types introduced by Phase D3** — neutral geometric evidence (spatial extent, depth statistics, validity/coverage, texture/observability, nearest supported distance) extracted from `RegionAnalyzer`'s/`ThreatAssessor`'s already-computed output, deliberately WITHOUT their behavioral labels (`RegionClass`, `BeamReading.status`'s CLEAR/CAUTION/BLOCKED). No unified `GeometryFrame`-level confidence score was introduced by D3 — `RegionEvidence.confidence` is `RegionAnalyzer`'s own already-existing *per-region* blend, reused as-is. `PointCloudBuilder`, `transform_point_cloud`, `build_obstacle_cloud`, `build_free_space_rays`, `build_geometry_metrics`, `GeometryQuality`, `classify_geometry_quality` remain Tier 3 — promoting a result type does not promote its producer. See `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D3 record for the full boundary.

`SurfaceEvidence` is a **new type introduced by Phase D4** — deterministic, bounded local-plane (surface normal) estimation over the `geometry_body` PointCloud, the first genuinely new geometric algorithm added since the D1/D2/D3 passes (which only assembled or extracted already-existing evidence). Answers "what surface geometry is observable here," never "can a vehicle traverse/land/navigate here" — no occupancy, traversability, or vehicle-specific concept exists anywhere in it. `geometry.surface.build_surface_evidence` (the algorithm) stays Tier 3 — same "promote the type, not the producer" rule as D3. See `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D4 record for the full frame/normalization/orientation/invalid-support contract.

`BoundaryEvidence` (plus `BoundaryState`/`BoundaryDirection`) is a **new type introduced by Phase D5** — deterministic detection of depth and/or surface-orientation discontinuities between adjacent cells of its own independent grid, reusing `depth_map` directly (Level 0, independent of `enable_geometry`) and `SurfaceEvidence`'s normals opportunistically (only when its grid happens to match — `RegionEvidence`'s/`SurfaceEvidence`'s own grids were never redesigned or forced to align). Three explicit states so a missing depth measurement is never silently promoted into a physical boundary. Never a semantic edge class, never a behavioral judgment, never an inferred opening — that reasoning is out of scope. `geometry.boundary.build_boundary_evidence` (the algorithm) stays Tier 3 — same "promote the type, not the producer" rule. See `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D5 record for the full discontinuity-definition/invalid-vs-boundary/coordinate contract.

`OpeningEvidence` is a **new type introduced by Phase D6** — a confirmed "geometrically supported gap between surrounding physical structures," built from this frame's already-confirmed `BoundaryEvidence` plus `depth_map`, on the SAME grid `BoundaryEvidence` itself used (never an independent one, never a forced redesign of `BoundaryEvidence`). Unlike `BoundaryEvidence`/`SurfaceEvidence` (one record per cell/pair, always), `OpeningEvidence` is a positive-findings-only list — an empty list means no opening was confirmed, not that nothing was evaluated. Never PASSABLE/TRAVERSABLE/SAFE/DOORWAY/WINDOW/FLY_THROUGH, no vehicle-size assumption, no platform-specific clearance threshold — never triggered by two image edges alone. `geometry.opening.build_opening_evidence` (the algorithm) stays Tier 3 — same "promote the type, not the producer" rule. See `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D6 record for the full admission-rule/invalid-vs-opening/coordinate contract.

`ClearanceEvidence` was **refined, not replaced, by Phase D7** — its existing D3 fields (`index`/`x1`/`x2`/`nearest_distance_m`/`has_evidence`) are unchanged; new fields add geometric coverage/support (`valid_count`/`total_pixels`/`coverage_fraction`/`support_state`, the last using the new `ClearanceSupportState` Tier 1 constant class) and calibrated horizontal bearing (`bearing_center_rad`/`bearing_min_rad`/`bearing_max_rad`, derived from already-available focal length/principal point via the standard pinhole formula — no new camera-geometry algorithm). `nearest_distance_m`'s own derivation is completely unchanged. This closes the coverage gap D3 originally flagged and D4/D5/D6 each carried forward. See `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D7 record for the full bearing/coverage/frame contract.

`GeometryFrameQuality` (plus `GeometryFrameQualityState`) is a **new type introduced by Phase D8** — a structured, multi-dimension geometric quality/uncertainty rollup, deliberately named DISTINCTLY from the pre-existing, narrower `geometry.GeometryQuality` (unchanged, single-signal, Phase E6) to avoid any collision or confusion between the two. Summarizes `geometry_metrics` (via the existing `classify_geometry_quality()`), `temporal_consistency`, `motion_aware_reliability`, and `temporal_persistence` into a shared VALID/DEGRADED/INSUFFICIENT vocabulary — deliberately NOT a blended numeric confidence score (if an aggregate could not be mathematically justified, none was invented). Each dimension is `None` when its capability was never enabled/computed (absent), distinct from `INSUFFICIENT` (ran, not enough evidence) and `DEGRADED` (ran, active problem). `RegionEvidence` also gained a `frame_id` field in this pass (the one other D3-era evidence type that still lacked one). `DepthPerceptionResult.confidence` is explicitly NOT superseded or redefined by this type — see `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D8 record for the full per-dimension derivation and absent-vs-degraded contract.

`RegionAnalyzer`, `SceneInterpreter`, `SceneState` (the classes that *produce* these types) remain Tier 3 — a caller receives and reads `RegionStats`/`NavigationDecision` values, it does not construct a `SceneInterpreter` itself. Likewise, `temporal.TemporalHistory`/`persistence.TemporalPersistenceTracker` and `geometry.PointCloudBuilder`/etc. (the classes/functions that *produce* the Level 3/4 result types) remain Tier 3.

## The supported DPE V1 consumer workflow (Phase D13)

`GeometryFrame` is the FINAL, authoritative DPE V1 perception-data contract for any external system (present or future — e.g. a future `hybrid_perception_engine`). The supported workflow is exactly:

```
configure DPE (PipelineConfig)
    -> construct public input contracts (StereoCalibration, StereoObservation, optional MotionHint)
    -> run DepthPerceptionPipeline
    -> consume GeometryFrame
```

Minimal example, using only Tier 1 imports:

```python
from depth_perception_engine import (
    DepthPerceptionPipeline,
    GeometryFrame,
    MotionHint,
    PipelineConfig,
    StereoObservation,
    load_stereo_calibration,
)

calibration = load_stereo_calibration("/path/to/stereo_calibration.xml")
config = PipelineConfig(enable_geometry=True, enable_geometry_frame=True)
pipeline = DepthPerceptionPipeline(config, calibration)   # build once, reuse across frames

observation = StereoObservation(left_image=left_image, right_image=right_image, left_timestamp=t)
result = pipeline.process_observation(observation)

geometry_frame: GeometryFrame = result.geometry_frame
print(geometry_frame.quality.overall_state, geometry_frame.geometry_metrics.valid_fraction)
```

`MotionHint` is imported above for completeness — it is only needed when a caller actually has angular-rate evidence to supply (`observation.motion_hint`/`.motion_hints`, consumed only when `PipelineConfig.enable_temporal`/`.enable_rotation_compensation` are set); omitting it is always legal (see the `MotionHint` table row above).

**`GeometryFrame`'s complete type graph is Tier 1, verified structurally, not by hand-maintained list** (Phase D13, `tests/test_public_api.py::TestGeometryFrameTypeGraphIsFullyPublic`): every field's annotated type — `timestamp`/`frame_id`/the four raw arrays/masks, `geometry`/`geometry_body`/`obstacle_cloud`/`free_space_rays`/`geometry_metrics`, the five Level 4 temporal fields, `region_evidence`/`clearance_evidence`/`surface_evidence`/`boundary_evidence`/`opening_evidence`, and `quality` — resolves (directly, or through `Optional`/`List`/`Dict`) only to a Tier 1 symbol or a plain builtin/`numpy.ndarray`. This re-confirms D9's own audit finding (the one gap, `FrameId`, was already closed at D10) and guards against future regression: a new `GeometryFrame` field typed against a not-yet-promoted class fails this test automatically, not only when someone remembers to update `docs/PUBLIC_API.md` by hand.

**GeometryFrame alone is sufficient — no legacy `DepthPerceptionResult` field is required.** An external consumer never needs to read `DepthPerceptionResult.traversability_mask`/`.obstacles`/`.confidence` to interpret DPE's authoritative geometric evidence; those remain compatibility-only fields for existing callers (`mp01_perception`). Proven structurally in `tests/test_d13_external_consumer.py::TestNoLegacyResultFieldsNeeded`, which AST-scans the delivered consumer-workflow function itself for any of those three attribute names.

**Comprehensive black-box proof:** `tests/test_d13_external_consumer.py` walks the full workflow above (calibration -> `StereoObservation` -> optional `MotionHint` -> `PipelineConfig` -> `DepthPerceptionPipeline` -> `GeometryFrame`) and inspects every one of `GeometryFrame`'s 22 fields using only public types — enforced by an AST import-surface scan permitting only the root package plus `depth_perception_engine.frames` (needed solely for `RigidTransform`, a legitimate pipeline constructor input never promoted to root — see `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D9 record for why). No internal geometry algorithm, `RegionAnalyzer`, `ThreatAssessor`, `TemporalHistory`/`TemporalRecord`, `fusion.result_builder`, or pipeline/traversability internals is reachable from this file by construction.

**Known caveat carried forward, not resolved by D13 (per its own explicit instruction not to fix behavior):** `SurfaceEvidence` is advisory under low-texture/decorrelated-stereo conditions — Phase D11 found real `StereoSGBM`'s smoothness prior can report near-perfect (`~0.99`) planarity on cells built from pure decorrelated noise (see `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D11 Finding 1). `SurfaceEvidence` must not alone be treated as authoritative geometry by an external consumer; corroborate with `geometry_metrics`/`quality` under low-texture conditions.

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

Everything not listed above. Examples: `RegionAnalyzer`, `SceneInterpreter`, `DisparityEngine`, `RectificationEngine`, `DepthEstimator`, `DistanceReader`, `ThreatAssessor`, `FrameSplitter`, `looks_like_garbage_frame`, `frames.RigidTransform` (`frames.FrameId` was promoted to Tier 1 at Phase D10 — see above), `calibration.contracts.*` (`CameraIntrinsics`, `StereoExtrinsics`, `RectificationParameters`, `RigCalibration`, `CameraModel`), `geometry.PointCloudBuilder`/`transform_point_cloud`/`build_obstacle_cloud`/`build_free_space_rays`/`build_geometry_metrics`/`GeometryQuality`/`classify_geometry_quality` (the Level 3 *producers* — as of Phase D3 their result types `PointCloud`/`ObstacleCloud`/`FreeSpaceRays`/`GeometryMetrics` are Tier 1, see above), `utils.*`, `fusion.result_builder.*`.

**Submodules remain importable** (`depth_perception_engine.pipeline`, `depth_perception_engine.traversability`, etc. — some, like `depth_perception_engine.utils`, are even reachable as bare attributes after `import depth_perception_engine`, an unavoidable consequence of Python binding parent-package attributes whenever a submodule is imported anywhere in the import chain — verified this pass via `sys.modules` inspection). **This is acceptable and not something to engineer around** — but none of it is part of the documented, stable contract, and no consumer should be instructed to rely on it. `utils` specifically is never in `__all__` and never will be.

`frames.*`/`calibration.contracts.*` remain Tier 3 by explicit design — see `docs/LEVEL3_PUBLIC_API.md` for the original Phase E1 reasoning (at that time, nothing produced a `geometry.*` result type yet either, and promoting an unproduced type would have overclaimed capability). `geometry.*`'s own result types (`PointCloud`/`ObstacleCloud`/`FreeSpaceRays`/`GeometryMetrics`) no longer fall under that reasoning — they gained a real producer at Phase E2-E5 and were promoted to Tier 1 at Phase D3 (above); `geometry.*`'s *producer* functions/classes remain Tier 3.

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
