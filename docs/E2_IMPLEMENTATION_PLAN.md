# E2 implementation plan (write-up only — not started)

This describes what a real Level 3 implementation (E2) would build against the contracts E1 froze. **Nothing below is implemented.** No code in this pass constructs a `PointCloud`, extracts an obstacle, casts a free-space ray, or computes a geometry metric.

## 1. `geometry/point_cloud_builder.py` — `PointCloudBuilder`

Wires the existing, currently-dead `DepthEstimator.estimate_point_cloud()` into a real execution path — directly addresses the dead-code finding from the previous session's perception capability audit.

- Input: a `raw_disparity` array (as already produced inside `DepthPerceptionPipeline.process()`) + the `StereoCalibration` already in use.
- Calls `DepthEstimator.estimate_point_cloud()` (unchanged — no math edit) to get `(points_3d, valid_mask)`.
- Wraps the result into a `geometry.PointCloud`: sets `frame_id=FrameId.CAMERA_OPTICAL_LEFT` (matching what `estimate_point_cloud()` already, implicitly, produces — see `docs/COORDINATE_FRAMES.md`), converts the existing `0.0`-invalid convention to `PointCloud`'s `NaN`-invalid convention (a real, small, testable conversion step — the one deliberate inconsistency documented in `docs/LEVEL3_CONTRACTS.md`), optionally attaches per-pixel `confidence` by reusing `RegionAnalyzer`-style signals if a per-pixel (not per-region) confidence signal is worth adding at that point — undecided until E2 actually starts.
- Testing plan: unit tests against known synthetic disparity planes (mirroring `test_depth_estimator.py::TestAnalyticKnownDepth`'s pattern — hand-computed expected XYZ, not just a differential check), plus a real-hardware capture re-run (this repo already has a working capture harness from the previous session) to confirm the wired-in point cloud is physically sensible against the same wall/cable scene already used for Level 0–2 verification.

## 2. `geometry/obstacle_extractor.py` — `ObstacleExtractor`

Consumes a `PointCloud` (from step 1) plus the existing `RegionStats`/`ObstacleAssessment` (already computed by `traversability`/`obstacles` — unchanged) to build a `geometry.ObstacleCloud`.

- Filters `PointCloud.points` to only those falling within regions/beams already classified `OBSTACLE`/`BLOCKED` by the existing Level 0–2 logic — deliberately **reusing** the existing, tested obstacle classification rather than re-deriving obstacle-ness from raw geometry, so E2 doesn't duplicate or risk diverging from Level 1's already-verified behavior.
- Populates `distances_m`/`confidence` from the same per-region/per-beam values already computed, not a new geometric distance calculation.

## 3. `geometry/free_space.py` — `FreeSpaceEstimator`

Produces `geometry.FreeSpaceRays` from the existing 20-beam scan geometry (`ThreatAssessor`'s beam column bounds, unchanged) — one ray per beam, direction derived from the beam's pixel column + calibration intrinsics, range from the beam's already-computed `distance_m` (or `DepthEstimator.MAX_DEPTH_M` if the beam is `CLEAR`/`NO_DATA` with no obstacle found, matching `FreeSpaceRays.ranges_m`'s documented clamp-not-infinity convention).

## 4. Wiring into `DepthPerceptionResult` — last step, not first

Only after steps 1–3 exist and are independently tested:
- Add `DepthPerceptionResult.geometry: Optional[GeometryResult] = None` (a new small composite type bundling `PointCloud`/`ObstacleCloud`/`FreeSpaceRays`/`GeometryMetrics` — exact shape TBD at E2 start, not frozen by E1).
- Add `PipelineConfig.enable_geometry: bool = False` — gates the added compute cost; `DepthPerceptionPipeline.process()` only calls the new builders when `True`.
- `fusion/result_builder.py`'s `build_result()` gains an optional geometry-building step, gated by the same flag.

This ordering (build the pieces standalone and tested, wire in last, gated off by default) is what keeps `mp01_perception` at zero risk throughout E2 — exactly the same discipline E1 and the previous recovery pass both followed.

## 5. Suggested sequencing

1. `PointCloudBuilder` + tests (closes the known dead-code finding).
2. `ObstacleExtractor` + tests (reuses Level 1 classification, adds no new obstacle-detection logic).
3. `FreeSpaceEstimator` + tests.
4. `GeometryMetrics` aggregation (once 1–3 exist, this is a small summary step over their outputs).
5. Wire into `DepthPerceptionResult`/`PipelineConfig`, gated off by default.
6. Re-run the full `pytest` suite + `colcon test` (both packages) + a live-hardware capture, exactly as both previous passes did, before considering E2 complete.

## Explicitly not part of E2 either

IMU integration, temporal fusion, occupancy mapping, semantic perception, multi-camera fusion, ROS adapters — all remain deferred past E2 as well; this plan only covers reaching a real, tested Level 3 (3D geometric understanding), matching the scope the previous perception capability audit identified as the actual next gap.
