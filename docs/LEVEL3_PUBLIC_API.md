# Public API specification (Level 3, Phase E1)

> **Superseded as the general reference by `docs/PUBLIC_API.md`** (public API freeze pass, 2026-08-05), which documents the full Tier 1/2/3 contract, stability policy, and the `mp01_perception` wrapper-readiness proof for the whole library — not just the Level 3 question this doc originally answered. This doc's own finding (below) is unchanged and still correct; kept as the historical record of *why* no redesign was needed for Level 3 specifically. One update from that pass: `NavigationDecision`/`RegionClass`/`RegionStats`/`TextureClass` (referenced throughout this doc as living in `traversability.types`) are now Tier 1, top-level exports — see `docs/PUBLIC_API.md` for why.

## Finding: no redesign needed

The E1 request asked whether the public API needs redesigning to reach `engine.process(observation) -> DepthPerceptionResult`. It already does — added in the previous session's baseline recovery pass, before this E1 pass began:

```python
pipeline = DepthPerceptionPipeline.from_config(config, calibration)   # or the plain constructor
result = pipeline.process_observation(observation)                    # observation: StereoObservation
# or, unchanged since the original refactor:
result = pipeline.process(left_image, right_image)
```

`DepthPerceptionPipeline` is the one class `mp01_perception` (or any caller) needs. This pass adds **zero** new public methods to it — adding more now, ahead of an actual Level 3 producer needing them, would be overengineering against this task's own instruction ("avoid overengineering, design only what is necessary").

## What is frozen as of this pass

| Surface | Status |
|---|---|
| `DepthPerceptionPipeline(config, calibration, rectify=True)` | Frozen — constructor shape unchanged since the original refactor |
| `.from_config(config, calibration, rectify=True)` | Frozen — added last session |
| `.process(left_image, right_image, left_timestamp=None, right_timestamp=None)` | Frozen — two-arg core call unchanged since the original refactor; timestamp kwargs added last session |
| `.process_observation(observation: StereoObservation)` | Frozen — added last session, matches the target `engine.process(observation)` shape |
| `.reset()` / `.close()` / `.health()` | Frozen — added last session |
| `DepthPerceptionResult` | Frozen — **not touched this pass**; no `geometry` field exists yet (see below) |
| `calibration.contracts.*`, `frames.*`, `geometry.*` | **New this pass**, not yet consumed by anything in `DepthPerceptionPipeline` — see `docs/LEVEL3_CONTRACTS.md` |

## Encapsulation — "nothing outside the repository should import internal modules directly"

Audited this pass: `mp01_perception` already violates this in one place, pre-dating E1 — `mp01_perception/frame_validation.py` imports `depth_perception_engine.quality.frame_quality.looks_like_garbage_frame` directly (an internal module, not re-exported from the top-level `depth_perception_engine.__all__`). This was found and documented (not fixed — out of scope, would require touching `mp01_perception`) during the previous session's baseline recovery pass; restated here since it's directly relevant to this task's stated goal. **Not addressed by E1** — fixing it means either re-exporting `quality` at the top level or changing `mp01_perception`'s import, and this task's scope is this repository's own contracts, not that boundary.

The new E1 additions (`calibration.contracts`, `frames`, `geometry`) are deliberately **not** re-exported from the top-level `depth_perception_engine` package — only reachable via explicit submodule imports (`from depth_perception_engine.geometry import PointCloud`, etc.). This is intentional: promoting unused, unproduced contracts to the top-level public surface would itself invite exactly the kind of internal-module-reaching this task wants to prevent, just one level up (a caller importing `depth_perception_engine.PointCloud` today would get a type with no producer — misleading, not helpful).

## What E2 will need to add here (not built now)

- `DepthPerceptionResult.geometry: Optional[<GeometryResult>] = None` — additive, default `None`, so existing callers (including `mp01_perception`) are unaffected until they opt in.
- `PipelineConfig.enable_geometry: bool = False` — a gate, so the added compute cost of Level 3 processing is opt-in, not paid by every existing caller by default.
- No change anticipated to `.process()`/`.process_observation()`'s call signature — the existing shape already accommodates returning a richer `DepthPerceptionResult`.

## Future extension check (success criteria from the E1 request)

- **IMU integration without redesigning the public API:** plausible under the current shape — a future `StereoObservation` (or a new sibling type) could gain an optional `imu_samples` field the same way `calibration` was added this pass (additive, default `None`), and `.process_observation()` would grow to consume it without changing `.process()`'s existing two-arg shape at all.
- **Multi-camera fusion without redesigning the public API:** plausible — `StereoObservation.calibration` (this pass) and `RigCalibration.camera_frame_id` (this pass) already anticipate more than one named camera frame existing; a future multi-camera caller could construct one `DepthPerceptionPipeline` per rig and fuse `DepthPerceptionResult`s downstream, or a future variant could accept a sequence of observations — either path is additive to what exists today, not a breaking change to `DepthPerceptionPipeline.process()`.

Neither of these is being built now — this section only confirms E1's contracts don't foreclose them, per this task's explicit success criteria.
