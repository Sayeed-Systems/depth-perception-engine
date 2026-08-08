# E7 plan (write-up only — not started)

This describes what a real Level 3, Phase E7 pass would build/validate against what E1-E6 actually delivered. **Nothing below is implemented.** No code in this pass adds occupancy mapping, temporal fusion, vehicle envelope reasoning, or runs an integrated real-hardware validation campaign.

## Where E6 left off

Single-frame geometry (E2-E5) is now hardened: an explicit adversarial input matrix (`tests/test_adversarial_geometry.py`, scenarios A-P) and the mandatory UNKNOWN-space safety invariant are both tested and passing; failure containment, determinism, and healthy→invalid→healthy recovery are all proven at the pipeline level; full per-stage performance and long-run memory behavior are characterized (not just claimed) at two resolutions. See `docs/VALIDATION_REPORT.md`'s E6 addendum for the complete results. **Level 3 is still not complete** — E6 hardened what E2-E5 built; it did not add occupancy, temporal fusion, or vehicle-relative reasoning, all of which remain open.

## Two genuinely different directions E7 could take — this plan does not choose between them

### Direction A: continue the capability roadmap `docs/E6_IMPLEMENTATION_PLAN.md` (E5-era) sketched

Unchanged from that document, carried forward verbatim since none of it has been built:

1. **Richer `GeometryMetrics`** — `max_obstacle_distance_m`, separate `obstacle_point_count`/`free_space_ray_count`, `unknown_fraction`, spatial coverage. Still the lowest-risk, most self-contained item — no new frame, no new producer chain, extends one already-wired-in dataclass.
2. **`ObstacleCloud`/`PointCloud.confidence` activation** — the pass-through already exists end to end (E2-E5); nothing populates it yet.
3. **Temporal fusion** — real prerequisite gap restated: no localization/VIO exists anywhere in this repository, so naive multi-frame accumulation would smear a moving sensor's geometry. Needs its own dedicated investigation before attempting.
4. **Occupancy/voxel mapping** — consumes `FreeSpaceRays`/`ObstacleCloud`, needs voxel resolution/extent decisions this repository has never made, and almost certainly needs (3) first.
5. **Vehicle envelope / collision-risk scoring** — scope question (this repo vs. `mp01_perception` vs. elsewhere) still entirely open, restated at every phase since E4.

### Direction B: integrated / system-level validation (what "E7 integrated-validation plan" more literally suggests)

E2-E6 validated this library in isolation — synthetic fixtures, mocked failures, a shared/virtualized development container's own benchmark numbers. None of E2-E6 re-ran against:

- **Real hardware.** `docs/VALIDATION_REPORT.md`'s only real-camera capture predates E2 (the original baseline-recovery pass, 2026-08-05, Level 0-2 only). Every E2-E6 number is synthetic. A real capture run through the full E2-E6 chain (camera → disparity → depth → camera cloud → body cloud → obstacle cloud → free-space rays → metrics → quality classification) has never happened.
- **`colcon test` inside the `mp01_ros2` container.** Every prior *baseline* recovery pass re-ran this (real ROS Humble, real flake8/pep257 lint); no E2-E6 phase has. This is a real, concrete, low-effort validation gap — the container and workflow already exist (`docs/VALIDATION_REPORT.md`'s original "Commands run" section documents exactly how), it has simply not been re-run since Level 3 work began.
- **`mp01_perception` integration re-verification.** Confirmed by construction throughout E2-E6 that every new field/parameter defaults to inert (`enable_geometry=False`, `body_T_camera_left=None`, etc.), so `mp01_perception`'s existing calls are unaffected — but this has been verified by *reading* `mp01_perception`'s source and by unit/integration tests in *this* repository, never by actually running the updated library inside `mp01_perception` end to end.
- **A non-synthetic adversarial scene.** E6's adversarial matrix uses synthetic images (flat/textureless, sparse-patch, random noise) — real degraded conditions (motion blur, glare, actual occlusion, real low-light noise) were not captured or tested against.

This direction requires no new source code in `depth_perception_engine` at all — it is a validation campaign, not a feature.

## This plan takes no position on A vs. B

Both are legitimate "E7" readings of "integrated-validation plan," and they are not mutually exclusive, but they have different owners and different costs (B needs physical hardware access and the `mp01_ros2` container; A is pure software). Resolving which one E7 actually means is the first concrete decision a future pass needs to make — not assumed here.

## Explicitly not part of this plan either

ROS integration into this library itself (as opposed to *validating inside* the existing `mp01_ros2` ROS overlay, Direction B above — a real, meaningful distinction), `mp01_perception` source modifications, semantic perception, learned/neural models, IMU, planning, localization — all remain out of scope, matching every prior phase's non-goals.
