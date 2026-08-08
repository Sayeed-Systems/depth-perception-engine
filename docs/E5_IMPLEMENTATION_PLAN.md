# E5 implementation plan (write-up only — not started)

This describes what a real Level 3, Phase E5 pass would build against the contracts E1 froze and E2/E3/E4 actually implemented. **Nothing below is implemented.** No code in this pass constructs an `ObstacleCloud`, casts a `FreeSpaceRays`, computes a `GeometryMetrics`, or reasons about vehicle dimensions/collision envelopes.

## Where E4 left off

`DepthPerceptionResult.geometry` (camera-optical-frame `PointCloud`, E2/E3) and `DepthPerceptionResult.geometry_body` (body-frame `PointCloud`, E4) are both real, tested, wired into `DepthPerceptionPipeline`, gated by `PipelineConfig.enable_geometry` and the optional `body_T_camera_left` constructor parameter respectively. `ObstacleCloud`, `FreeSpaceRays`, and `GeometryMetrics` (`geometry/types.py`, frozen at E1) still have **zero producers** — E5 is where that changes, per `docs/LEVEL3_CONTRACTS.md`'s producer table.

## 1. `geometry/obstacle_extractor.py` — `ObstacleExtractor`

Consumes `geometry_body` (preferred — vehicle-relative geometry is more directly useful downstream than camera-relative; falls back to `geometry` if no body extrinsic is configured, a decision E5 should make explicit, not silently pick one) plus the existing `RegionStats`/`ObstacleAssessment` (already computed by `traversability`/`obstacles`, unchanged) to build a `geometry.ObstacleCloud`.

- Filters `PointCloud.points` to only those falling within regions/beams already classified `OBSTACLE`/`BLOCKED` by the existing Level 0-2 logic — deliberately **reusing** the existing, tested obstacle classification rather than re-deriving obstacle-ness from raw geometry, so E5 doesn't duplicate or risk diverging from Level 0-2's already-verified behavior (same discipline E2's plan already committed to for this exact step).
- Populates `distances_m`/`confidence` from the same per-region/per-beam values already computed, not a new geometric distance calculation.
- Output is unorganized (`(N, 3)`, N varies per call) — a real change in shape discipline from the organized `PointCloud` inputs, needs its own shape/ownership tests distinct from `tests/test_rigid_transform.py`'s organized-cloud assumptions.

## 2. `geometry/free_space.py` — `FreeSpaceEstimator`

Produces `geometry.FreeSpaceRays` from the existing 20-beam scan geometry (`ThreatAssessor`'s beam column bounds, unchanged) — one ray per beam, direction derived from the beam's pixel column plus calibration intrinsics (and, if body-frame output is requested, further rotated by `body_T_camera_left.rotation` — a direction is not a point, so it transforms by rotation only, not translation; this distinction does not exist yet anywhere in this codebase and needs its own dedicated, tested helper, not a reuse of `transform_point_cloud`, which is point-only), range from the beam's already-computed `distance_m` (or `DepthEstimator.MAX_DEPTH_M` if the beam is `CLEAR`/`NO_DATA`, matching `FreeSpaceRays.ranges_m`'s documented clamp-not-infinity convention).

## 3. `GeometryMetrics` aggregation

Once 1-2 exist, this is a small summary step over their outputs (`min_obstacle_distance_m`, `mean_free_space_m`, `point_count`, `valid_fraction`) — no new geometric computation, per its own frozen docstring.

## 4. Vehicle envelope / collision reasoning — explicitly deferred past E5 too, pending scope decision

This task's own non-goals (E4) list "vehicle envelope logic," "collision envelopes," and "risk scoring" as out of scope — that exclusion is not automatically lifted for E5 either; whether collision-envelope reasoning belongs in E5, a later E6, or outside this repository entirely (e.g. as `mp01_perception`-side logic consuming `ObstacleCloud`) is a real open scope question this plan does not resolve, flagged here so it isn't silently assumed either way when E5 actually starts.

## 5. Wiring into `DepthPerceptionResult` — last step, not first, same discipline as E2→E3

Only after steps 1-3 exist and are independently tested:
- Add `DepthPerceptionResult.obstacle_cloud: Optional[geometry.ObstacleCloud] = None` / `.free_space: Optional[geometry.FreeSpaceRays] = None` / `.geometry_metrics: Optional[geometry.GeometryMetrics] = None` — additive, default `None`, reusing the exact frozen types directly (same "don't invent a composite" reasoning `docs/LEVEL3_ARCHITECTURE.md`'s E3/E4 updates both used for `PointCloud`).
- A new config gate (or reuse of `enable_geometry`, decided at E5 start, not here) so the added compute cost stays opt-in.
- `fusion/result_builder.py`'s `build_result()` gains the corresponding optional pass-through parameters, mirroring `geometry`/`geometry_body`'s existing pattern exactly.

## 6. Suggested sequencing

1. `ObstacleExtractor` + tests (reuses Level 0-2 classification, adds no new obstacle-detection logic).
2. A dedicated direction-only (rotation-not-translation) transform helper, if `FreeSpaceRays` needs body-frame directions — a real, small, new piece of math distinct from `transform_point_cloud`, not a reuse of it.
3. `FreeSpaceEstimator` + tests.
4. `GeometryMetrics` aggregation (once 1-3 exist, this is a small summary step over their outputs).
5. Wire into `DepthPerceptionResult`/`PipelineConfig`, gated off by default.
6. Re-run the full `pytest` suite + `colcon test` (both packages) + a live-hardware capture, exactly as every previous phase did, before considering E5 complete.
7. Resolve the vehicle-envelope/collision-reasoning scope question (item 4 above) explicitly, before writing any code toward it.

## Explicitly not part of E5 either

IMU integration, temporal fusion, occupancy mapping, semantic perception, multi-camera fusion, ROS adapters, world-frame/localization-dependent geometry — all remain deferred past E5, matching every prior phase's non-goals. World-frame transformation specifically has no frozen convention anywhere yet (unlike body-frame, which E1 froze and E4 implemented) — a future phase would need its own frame-freezing pass first, the same way E1 preceded E2-E4 for the body frame.
