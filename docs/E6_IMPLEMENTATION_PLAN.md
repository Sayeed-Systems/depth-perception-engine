# E6 implementation plan (write-up only — not started)

> **Update (2026-08-06):** the actual Level 3, Phase E6 pass took a different direction than this plan anticipated. This document (written at the end of E5) speculated E6 would build occupancy/voxel mapping and temporal fusion; the actual E6 task was scoped as **robustness, degradation semantics, and performance hardening** of the E2-E5 chain instead (adversarial input testing, an explicit UNKNOWN-space safety invariant, a geometry-quality classifier, failure containment, determinism, recovery, and full performance/memory characterization — see `docs/IMPLEMENTATION_STATUS.md`'s E6 addendum). Every item below (occupancy, temporal fusion, vehicle envelope, richer `GeometryMetrics`, confidence activation) remains **entirely unbuilt** and has been carried forward into `docs/E7_IMPLEMENTATION_PLAN.md` rather than lost — this document is kept as the historical record of that original speculation, not corrected in place.

This describes what a real Level 3+ (E6 and beyond) pass would build against what E1-E5 actually delivered. **Nothing below is implemented.** No code in this pass builds an occupancy grid, fuses geometry across frames, reasons about vehicle dimensions, or scores collision risk.

## Where E5 left off

Single-frame geometry is now complete end to end: `Stereo → Disparity → Metric Depth → Camera-frame PointCloud (E2/E3) → Body-frame PointCloud (E4) → ObstacleCloud / FreeSpaceRays / GeometryMetrics (E5)`. Every stage is calibration-driven, additive, and independently gated. There is still **no persistent state across frames** anywhere in this chain (`ThreatAssessor`'s per-beam EMA/debounce, Level 0-2, is the one pre-existing exception — untouched by E2-E5) and **no occupancy representation** — `FreeSpaceRays` are observations, not an integrated map.

## 1. Occupancy / voxel mapping — consumes `FreeSpaceRays` + `ObstacleCloud`, does not replace them

A real occupancy grid (or other volumetric representation) would ray-cast each `FreeSpaceRays` ray through a voxel grid (marking traversed cells free, the endpoint cell occupied) and would need to decide:
- **Voxel resolution and extent** — a real config surface this repository has never needed before (E5's own non-goals explicitly excluded this: "Do NOT voxelize the ray. Voxel occupancy integration belongs to Level 8" per the E5 task itself — restated here as still out of scope for E6 unless a future task explicitly authorizes it).
- **Single-frame vs. accumulated occupancy** — a single frame's rays alone make a poor occupancy grid (sparse, noisy); this almost certainly implies temporal accumulation (item 2 below) is a prerequisite, not an independent feature.
- Where this lives architecturally: a new top-level capability area (`occupancy/`?), not inside `geometry/` — `geometry/` has consistently held single-frame, stateless producers (see `docs/LEVEL3_ARCHITECTURE.md`'s reasoning for why `geometry/` is a distinct package from `traversability/`/`obstacles/` in the first place); an occupancy map is neither single-frame nor stateless, so it does not belong there by the same logic.

## 2. Temporal fusion — the real prerequisite most later capabilities need

Every stage through E5 is stateless per call (`ThreatAssessor`'s EMA/debounce is the sole exception, and it smooths a 2D per-beam scan, not 3D geometry). A real temporal-fusion layer would need:
- A frame-to-frame association/registration story — this pipeline has no localization/VIO of any kind (explicitly out of scope, non-goal in every phase so far including this plan), so naive accumulation across frames without ego-motion correction would smear a moving sensor's geometry incorrectly. This is a real, hard prerequisite gap, not a detail — flagged here explicitly rather than glossed over.
- A decision about what persists: raw point history, an occupancy grid, or some compressed intermediate — depends on decision 1 above.

## 3. Vehicle envelope / collision-risk scoring — explicitly deferred again

Every phase from E4 through this plan has excluded vehicle dimensions, safety margins, and collision-risk scoring from scope, each time restating that the decision of *where* this belongs (this repository vs. `mp01_perception` vs. a new consumer entirely) has not been made. This plan does not resolve it either — restated here so a future E6+ pass does not have to rediscover that the question is still open.

## 4. Richer `GeometryMetrics` — a real, scoped extension candidate

E5's `build_geometry_metrics()` populated exactly the 4 fields `GeometryMetrics` froze at E1. A richer set was identified as useful but out of scope (`docs/LEVEL3_ARCHITECTURE.md`'s E5 update, `docs/DATA_CONTRACTS.md`'s "Spatial evidence" section):
- `max_obstacle_distance_m` (farthest observed surface, symmetric with the existing `min_obstacle_distance_m`)
- separate `obstacle_point_count`/`free_space_ray_count` (currently only the source cloud's `point_count` is exposed — an obstacle-range-filtered count is not currently derivable from `GeometryMetrics` alone, only from `len(obstacle_cloud.points)` directly)
- `unknown_fraction` (currently only `valid_fraction` exists; `1 - valid_fraction` already recovers this arithmetically, so this is a convenience field, not new information)
- spatial coverage (angular or areal — no single well-defined formula was settled on even provisionally; would need its own design pass, not a one-line addition)

This is the most self-contained, lowest-risk item in this plan — no new frame, no new producer chain, just extending one already-real, already-wired-in dataclass. A future pass could do this alone without needing to resolve items 1-3 first.

## 5. `ObstacleCloud`/`PointCloud` confidence — currently always `None` in practice

`PointCloud.confidence`/`ObstacleCloud.confidence` are both frozen, optional, and already threaded through every producer in this pipeline (`PointCloudBuilder`, `transform_point_cloud`, `build_obstacle_cloud` all pass it through when present) — but nothing upstream has ever populated it; it is `None` end to end today. A future per-pixel confidence signal (reusing `RegionAnalyzer`-style texture/entropy signals at pixel granularity, as `geometry/types.py`'s own original docstring speculated) would activate this pass-through with zero changes to any of the E2-E5 producers themselves.

## Suggested sequencing

1. `GeometryMetrics` extension (item 4) — standalone, no prerequisites, could happen independently of everything else in this plan.
2. Resolve the vehicle-envelope/collision-risk scope question (item 3) — a decision, not code.
3. Localization/VIO prerequisite investigation (part of item 2) — almost certainly needs its own dedicated phase before real temporal fusion is attemptable.
4. Temporal fusion (item 2).
5. Occupancy/voxel mapping (item 1), once 2-4 exist.
6. Confidence signal activation (item 5) — can happen in parallel with any of the above, genuinely independent.

## Explicitly not part of this plan either

ROS integration, `mp01_perception` modifications, semantic perception, learned/neural models, multi-camera fusion — all remain out of scope, matching every prior phase's non-goals.
