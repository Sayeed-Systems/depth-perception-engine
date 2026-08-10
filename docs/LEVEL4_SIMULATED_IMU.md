# Simulated IMU policy (Level 4, Phase E1)

**No simulator is implemented in this pass.** This document freezes the architectural boundary a future simulated-IMU producer (and a future real-IMU producer) must both honor — nothing more.

## Why simulation is needed at all

Level 4 development does not currently have access to a real IMU. Later phases (E5+) will therefore develop and validate short-duration rotational motion compensation against a **simulated** IMU input first. This is a real, acknowledged constraint on how Level 4 will be developed — but it must never become a constraint baked into the core perception API's shape or behavior.

## The one rule this document exists to enforce

**Simulation must not leak into the core perception API.** Concretely:

- The core engine (`depth_perception_engine`'s `src/` tree) receives exactly one motion-input contract — `temporal.MotionHint` (`docs/LEVEL4_CONTRACTS.md`) — regardless of whether the value behind it came from a simulator or a real sensor.
- No code under `src/depth_perception_engine/` may contain a source-specific branch of any kind: no `if simulated_imu:`, no `if real_imu:`, no `if source == "simulation":`, no equivalent. `tests/test_level4_architecture_guards.py` statically scans `src/` for this pattern (identifiers containing `simulated_imu`/`real_imu`, and string comparisons against `"simulation"`/`"simulated"`) and fails the build if one appears.
- `temporal.MotionHint` itself has no `source`/`is_simulated`/`provenance` field — see `docs/LEVEL4_CONTRACTS.md`'s "Why no `source`/`is_simulated` field" section. A field like that would be a standing invitation for exactly the branch this rule forbids.

## Future architecture (not built yet)

```
simulated source ─┐
                   ├─> temporal.MotionHint (canonical, producer-agnostic)
real IMU source ───┘
                              |
                              v
                  depth_perception_engine
                  (consumes MotionHint only —
                   never asks who produced it)
```

The engine does not care, and must never be made to care, which producer supplied a given `MotionHint`. A simulated producer and a real producer are interchangeable at the `MotionHint` boundary by construction — if a future implementation ever needs to tell them apart inside `src/depth_perception_engine/`, that is itself a signal the boundary has been violated and needs to be fixed, not worked around.

## Where the simulator will eventually live (not decided in detail, but scoped)

The simulated-IMU producer itself — whatever generates a stream of plausible `MotionHint` values for development/testing — belongs in **tests, examples, or simulation tooling**, never in `src/depth_perception_engine/`'s core algorithm modules. Candidate locations for a future phase to choose between (not decided here): a `tests/` fixture/helper (if only test code needs it), or an `examples/`-style script (if it should also be runnable/demonstrable standalone, matching this repository's existing precedent of keeping standalone tools like `examples/visualize_level3.py` out of `src/` entirely — see `docs/LEVEL3_ARCHITECTURE.md`'s E7 update). Either choice keeps the simulator outside the package that ships as `depth_perception_engine` and gets imported by a real consumer.

## What this document explicitly does not do

Implement a simulated IMU. Generate any synthetic `MotionHint` stream. Add a `source` field to `MotionHint`. Add any simulation-aware branch anywhere in `src/depth_perception_engine/`. All of these remain out of scope for E1 and are deferred to whichever future phase (E5, per the Level 4 task's own numbering) actually builds temporal motion compensation.
