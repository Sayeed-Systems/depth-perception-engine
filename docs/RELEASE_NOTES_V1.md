# depth_perception_engine — MP01 V1 release notes

**Status: FREEZE-READY (Phase D16).** This document is the single, concise, release-focused summary of what `depth_perception_engine` (DPE) is, what it guarantees, and what it explicitly does not do, at the point its V1 provider contract was frozen. It supersedes nothing — `docs/PUBLIC_API.md` remains the authoritative Tier 1/2/3 reference and `docs/DPE_V1_PROVIDER_CONTRACT.md` remains the authoritative phase-by-phase design/validation record (D1-D16) — this file exists so a future integrator (principally a `hybrid_perception_engine` developer) can read one page and know whether DPE is ready to build on.

## Purpose

DPE is a standalone, ROS-free, hardware/simulator-agnostic Python library that turns a rectified (or rectifiable) stereo image pair, plus calibration and optional short-window angular-rate evidence, into structured, deterministic geometric and temporal perception evidence — nothing upstream of that (camera drivers, ROS topics, sensor transport) and nothing downstream of it (navigation policy, planning, control) is DPE's concern.

## Major capabilities

- Rectification, stereo disparity (SGBM), metric depth estimation.
- Level 3 metric geometry: camera- and body-frame point clouds, obstacle/free-space spatial evidence, geometry quality metrics.
- Neutral per-region and per-sector evidence: `RegionEvidence` (spatial/depth/texture statistics without occupancy labels), `ClearanceEvidence` (calibrated directional distance/bearing with coverage/support classification).
- Local surface geometry: `SurfaceEvidence` (deterministic per-cell plane fit — centroid/normal/planarity).
- Geometric discontinuities: `BoundaryEvidence` (depth- and/or surface-orientation-based, explicit insufficient-evidence state — never fabricated from missing data).
- Geometric openings: `OpeningEvidence` (positive-findings-only, geometrically confirmed gaps between real flanking structure).
- Level 4 temporal geometric reasoning: bounded chronology (`TemporalHistory`), read-only cross-frame consistency, deterministic stabilization, short-window rotation compensation from optional `MotionHint` evidence, motion-aware reliability assessment, per-cell temporal persistence classification.
- Structured geometric quality/uncertainty rollup (`GeometryFrameQuality`) — a shared VALID/DEGRADED/INSUFFICIENT vocabulary across geometry validity, temporal consistency, motion reliability, and persistence, with an explicit absent-vs-degraded-vs-insufficient distinction, never a blended confidence score.

## The GeometryFrame provider boundary

`GeometryFrame` (`DepthPerceptionResult.geometry_frame`, opt-in via `PipelineConfig.enable_geometry_frame`) is the **one, final, authoritative** output contract any external perception system is meant to consume DPE evidence through. A consumer never needs to import DPE algorithm internals, `TemporalHistory`/`TemporalRecord`, `RegionAnalyzer`/`ThreatAssessor` internals, `NavigationDecision`, intermediate pipeline state, or any other `DepthPerceptionResult` field to interpret DPE's geometric evidence completely (proven structurally — `tests/test_public_api.py::TestGeometryFrameTypeGraphIsFullyPublic`, `tests/test_d13_external_consumer.py::TestNoLegacyResultFieldsNeeded`). `DepthPerceptionResult`'s other fields (`traversability_mask`, `obstacles`, `confidence`) remain real, tested, indefinitely-supported **compatibility** APIs for existing callers (`mp01_perception`) — not removed, not the V1 contract.

## Release contract

**INPUT** (construct these; no ROS/hardware/simulator type is ever required):
- `StereoObservation`
- `StereoCalibration`
- `MotionHint` (optional — a missing hint always falls back cleanly, never blocks Level 3 perception)
- `PipelineConfig`

**EXECUTION:**
- `DepthPerceptionPipeline` (`.process()` / `.process_observation()`)

**AUTHORITATIVE OUTPUT:**
- `GeometryFrame`

**DPE owns:** stereo/disparity/depth · metric geometry · spatial evidence · region evidence · directional clearance · surfaces · boundaries/discontinuities · openings · temporal geometric reasoning · rotation compensation · persistence · structured geometric quality.

**DPE does NOT own:** neural semantics · object classification · localization/state estimation · persistent/global mapping · planning · platform identity · capability management · controls · ROS/hardware/simulator integration.

## Validation summary (D9-D15)

| Phase | Scope | Result |
|---|---|---|
| D9 | Provider-contract completeness audit | FREEZE-READY |
| D10 | Controlled ground-truth geometry validation (analytic + real SGBM) | PASS |
| D11 | Degradation/failure validation (15+ scenarios) | PARTIAL — one documented finding, see Known caveat |
| D12 | Sensor-contract independence (zero ROS/hardware/simulator dependency; MotionHint contract matrix) | PASS |
| D13 | Public API / provider hardening (GeometryFrame type-graph fully Tier 1; black-box external-consumer test) | READY |
| D14 | Performance / bounded-resource validation | PASS (characterized, not threshold-graded — see below) |
| D15 | Packaging / reproducibility / external-consumer validation | PASS |
| D16 | Final freeze audit (this phase) | see report |

Full narrative and every measured number for each phase: `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D9-D16 records.

## Performance / reproducibility status

Measured honestly on one development container (not target/Jetson hardware) — no real-time rate requirement is documented for this library by any consumer, so no threshold is claimed. At 320x240 (the real hardware calibration resolution): ~21-47 FPS depending on which optional evidence families are enabled (core geometry fastest, full V1 candidate configuration slowest). `TemporalHistory` and `TemporalPersistenceTracker` state confirmed bounded over a 500-frame sustained run; `GeometryFrame`'s large fields confirmed zero-duplication (same objects as `DepthPerceptionResult`'s own fields, not copies) by direct identity check; `reset()` confirmed to return the pipeline exactly to its documented first-frame baseline. Packaging: `python -m build` produces a runtime-only wheel (no `tests`/`docs`/`examples` leakage); installed into a fully isolated environment outside this repository, every required Tier 1 symbol imports correctly from the installed artifact; a standalone external-consumer script (zero internal-module or repository-file dependency) ran the complete input -> pipeline -> `GeometryFrame` workflow and produced **bit-exact** output across two independent process invocations. Dependency version ranges (`numpy>=1.23`, `opencv-python-headless>=4.7`, `requires-python>=3.9`) are intentionally broad, not pinned — the conventional posture for a library meant to be embedded in a larger consumer's own dependency resolution.

## Known caveats (preserved, not resolved)

**`SurfaceEvidence` is advisory under low-texture/decorrelated stereo conditions and must not alone be treated as authoritative geometry.** Phase D11 found real `StereoSGBM`'s smoothness prior can report high-confidence (`~0.99`) planarity on cells built from pure decorrelated noise, where no real surface exists — a confidence-signal over-claim specific to `SurfaceEvidence`. **This caveat does NOT invalidate metric depth, obstacle, or free-space geometry** — those remained correctly gated by validity masks in every scenario D11 tested; no fabricated depth/obstacle/free-space evidence was ever produced. A consumer should treat `SurfaceEvidence` as corroborating evidence, not sole authority, under low-texture conditions — cross-check against `geometry_metrics`/`quality` when texture is sparse.

**RESOLVED (Phase I6.2/I6.3) — `ClearanceEvidence` false-clear safety gap.** The post-freeze I1–I6 improvement series (see `docs/VALIDATION_REPORT.md`'s own I1–I6 addendum) initially found `ClearanceEvidence` sectors that reported an obstacle farther/clearer than reality ("false-clear"). That figure was first measured at 28/252 sectors (11.1%), but 24 of those 28 were later found to be an artifact of a benchmark-methodology bug (stale `ThreatAssessor` EMA/debounce state leaking across unrelated fixtures sharing one pipeline instance in the measurement script, not a pipeline defect — fixed in `benchmarks/i5_surface_opening_clearance/clearance/measure.py`). The true, methodology-clean false-clear count was 4/252 (1.6%), traced to two distinct root causes: (1) genuine occlusion-shadow contamination — closed by threading the existing Phase I3 `compute_shadow_zone_mask` reliability signal into `ThreatAssessor.assess()`/`ClearanceEvidence` construction (`PipelineConfig.clearance_shadow_zone_gating_enabled`); (2) a wider (~20px), direction-agnostic StereoSGBM smoothness-regularization ramp that the narrow occlusion-shadow model couldn't see — closed by a second, independent reliability signal, `geometry.reliability.compute_ramp_zone_mask` (`PipelineConfig.clearance_ramp_zone_gating_enabled`). Neither mechanism attempts to *recover* the true near-obstacle value (proven not reliably recoverable once SGBM has smeared a transition — see Phase I6.1's contiguity/nearest-cluster prototypes) — instead, a beam whose IQR-kept population is significantly contaminated by either signal is downgraded from `SUPPORTED` to `PARTIALLY_SUPPORTED` rather than asserted as authoritative. **Result: 0/252 false-clear sectors** in the qualified benchmark, down from the corrected 4/252 baseline, with the worst-case `SUPPORTED`-sector error now 4.4% (down from 139%). The remaining, accepted trade-off: ~30/252 sectors immediately adjacent to a genuine transition (previously `SUPPORTED`, sometimes with a correct reading by chance) now read `PARTIALLY_SUPPORTED` instead — an intentional, conservative-direction cost, not a new defect. Pure decorrelated noise still never produces a confidently-`SUPPORTED` sector (0 of 630 tested).

## Explicit non-responsibilities

DPE performs no ROS/topic/node logic, no camera/IMU driver or device-path handling, no simulator integration, no vehicle/platform identity, no neural inference, no object classification, no localization or global mapping, no path planning, and no control/actuation. These remain the explicit responsibility of the sensor backend below DPE (`mp01_sensors` or equivalent) and the perception/planning system above it (a future `hybrid_perception_engine` and beyond).

## Version

Current package version: `1.0.1` — the D16 freeze was released as `1.0.0`; `1.0.1` is a packaging-only patch (Phase D17) fixing an external-install defect (a missing `setup.py` legacy-build-path fallback caused some non-PEP-517-isolated build paths to produce `UNKNOWN` package metadata instead of `depth-perception-engine`). No algorithm, `GeometryFrame` contract, or evidence-semantics change is included in `1.0.1` — see `docs/DPE_V1_PROVIDER_CONTRACT.md`'s D16 (freeze rationale) and D17 (packaging repair) records.

## What's next

No further DPE feature development is anticipated for the MP01 V1 scope this freeze covers. The next development track is `neural_perception_engine` (NPE), consuming `GeometryFrame` as its one geometric-evidence input, per the architecture this whole D-phase series was designed around.
