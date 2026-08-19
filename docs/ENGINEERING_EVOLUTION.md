# Engineering Evolution — IA0 through v1.1.1

This document preserves the debugging/optimization story behind DPE's
post-freeze improvement series (I1-I6.3) and the v1.1.1 structural closure,
in enough detail to be useful in an engineering interview or a future
regression investigation. It does not sanitize dead ends — rejected
approaches are recorded because they are evidence the design was
evidence-driven, not because every experiment worked.

Every number below is cited to a script or artifact in this repository.
`benchmarks/*/results/*.json` are `.gitignore`d (regeneratable, not
committed) — re-run the cited script to reproduce a number, or read
`benchmarks/reporting/release_metrics_manifest.py`, which recomputes each
headline figure with its own citation. Where a "before" number could not be
independently re-derived on the current, frozen v1.1.1 source (it would
require reverting a shipped fix), it is cited to `docs/VALIDATION_REPORT.md`'s
own I1-I6 addendum instead, and marked as such.

For the full D-phase (pre-freeze, V1 development) history, see
`docs/DPE_V1_PROVIDER_CONTRACT.md` and `docs/VALIDATION_REPORT.md` directly —
this document starts from the post-D17 (`v1.0.1`) frozen baseline.

---

## Baseline provenance reconciliation (IA0)

**Problem.** Before starting the I1-I6 series, an earlier task prompt cited
a "MP01 Gazebo/RViz simulation validation report" with specific numbers
(depth error, opening counts, latency) as the pre-existing baseline to
improve against.

**Observation.** That report does not exist anywhere in this repository or
its git history. `benchmarks/i0_baseline/`'s own README documents this
finding, independently confirmed twice.

**Resolution.** The I1-I6 series treated the cited report as an unverified
external claim throughout and built its own real, reproducible, in-repo
evidence base instead: `benchmarks/i0_baseline/` (a recorded snapshot of
`v1.0.1`'s own behavior, `baseline_v1.0.1.json`) plus the new
`i1_stereo_accuracy/`, `i3_occlusion_safety/`, `i4_boundary_precision/`,
`i5_surface_opening_clearance/`, `i6_temporal/` suites. This document, and
`docs/VALIDATION_MATRIX.md`, cite only that in-repo evidence.

**Performance-provenance note (important, and directly relevant to how
"final v1.1.1 performance" should be read below).** Standalone DPE latency,
measured on the *same* 320×240/full-V1-evidence-config scope, has been
recorded at several different mean values across sessions in this same
development sandbox:

| Source | Mean latency | FPS |
|---|---|---|
| `docs/DPE_V1_PROVIDER_CONTRACT.md` D14 record (dedicated performance pass) | 35.86 ms | 27.9 |
| `benchmarks/results/i6_final_qualification.json` (`compare_to_baseline` candidate side, during the I6 session) | 23.6 ms | 42.4 |
| This document's own v1.1.1 documentation pass (`benchmarks/i0_baseline/compare_to_baseline.py`, run fresh) | 38.39 ms | 26.0 |

This is real, disclosed run-to-run/session-to-session variance on a shared
development container — not three different versions of DPE, and not
evidence of a regression between them (no source file affecting this path
changed between the D14 and I6 sessions; D14 predates the I1-I6 series
entirely). **The "Performance at a Glance" figure in `README.md` uses the
freshest measurement from this documentation pass, not the fastest
historical one** — see "Compute performance" below for the full number set.

A separate figure, ~97 ms mean latency, appears in an **external**
integration repository's own MP01-sim validation report (not part of this
repository, not reproducible from anything committed here). That number
measures a different boundary entirely — end-to-end simulation/ROS/wrapper
latency in a downstream consumer's own integration test, not standalone DPE
compute cost — and is not cited as a DPE performance number anywhere in
this repository's documentation. Conflating the two would misrepresent
both.

---

## I1 — Stereo / disparity accuracy

**Problem.** A re-audit of the shipped `v1.0.1` pipeline found
`stereo/disparity_engine.py`'s `StereoSGBM` smoothness-penalty terms
(`P1`/`P2`) computed using OpenCV's documented **3-channel** heuristic
(`8 * 3 * block_size**2` / `32 * 3 * block_size**2`) even though
`compute_disparity()` always converts both frames to grayscale (1 channel)
before calling `.compute()` — the smoothness prior was roughly 3× stronger
than the input actually warranted.

**Hypothesis.** An over-strong smoothness prior would over-smooth real
disparity transitions and, more dangerously, let `StereoSGBM`'s own
confidence machinery report "valid" disparity on input with **no real
stereo correspondence at all** (pure decorrelated noise) by extending a
neighboring confident match across a region that shouldn't match anything.

**Experiment.** `benchmarks/i1_stereo_accuracy/measure.py` ran a fixture
sweep (scenarios A-G: high/moderate/weak texture, periodic pattern,
occlusion present/absent, decorrelated noise) at multiple depths and seeds,
before and after changing the channel multiplier to 1 and raising
`uniquenessRatio` from 10 to 20 (a further sweep,
`benchmarks/i1_stereo_accuracy/sweep.py`, found 20 cut false-valid
disparity a further ~86% relative over the channel fix alone, with zero
measured cost to accuracy or latency).

**Root cause.** Confirmed exactly as hypothesized:
`benchmarks/i1_stereo_accuracy/results/{baseline_current.json,
final_after.json}`, scenario G (decorrelated noise):

| | Before | v1.1.1 |
|---|---|---|
| False-valid disparity fraction | 41.788% | 0.267% |

**Fix.** `src/depth_perception_engine/stereo/disparity_engine.py`:
`P1 = 8*1*block_size**2`, `P2 = 32*1*block_size**2`; `uniquenessRatio: 10 →
20`; `PipelineConfig.block_size` default `13 → 9` (git `c2b1906`).

**Before → after, per-distance depth error (median relative %, synthetic
fixtures, `benchmarks/i1_stereo_accuracy/results/{baseline_current,
final_after}.json`):**

| Distance | Before | v1.1.1 |
|---|---|---|
| 1 m | 0.561% | 0.405% |
| 2 m | 0.561% | 0.561% |
| 3 m | 1.501% | 1.018% |
| 5 m | 0.561% | 0.561% |
| 6 m | 4.443% | 3.574% |

**Remaining limitation, reported honestly.** Weak-texture (scenario C) at
6 m still measures 5.296% median relative error, `final_after.json`
scenario C — a narrow, harder case than the other 6 m scenarios averaged
together above, still present in v1.1.1. Residual false-valid disparity
(0.267%, not 0%) is not itself proof of correct downstream geometry — it is
a disparity-stage metric; `GeometryMetrics`/`GeometryFrameQuality` gate
whether that disparity is trusted further downstream, not this benchmark.

---

## I2 — Validity / observable ROI

**Problem.** A previously-reported "whole-frame valid fraction ≈59.9%"
figure was being read as "DPE only successfully reconstructs 60% of the
observable scene" — an incorrect and needlessly pessimistic reading.

**Investigation.** `numDisparities=128` on a 320px-wide frame makes the
leftmost 128 columns (40% of width) structurally unable to produce a
disparity match, *regardless of scene content* — a direct consequence of
the search window, not an implementation defect.
`benchmarks/i2_depth_validity/results/step1_2_roi_and_gates.json` confirms
this exactly: `theoretical_whole_frame_observable_fraction = 0.60`,
matching the measured `whole_frame_valid_mean ≈ 0.599`.

**Corrected metric.** Scoped to the columns that are actually observable
(≥128px), valid fraction measures **99.34-99.96%** across the full
texture/depth sweep — the pre-existing "≥70% observable-geometry" target
was already met once measured at the right scope; no code change was
required.

**Why no gate was relaxed.** The dead zone is a real, physical search-window
limit — masking depth there is correct behavior, not a defect to code
around. Reporting the corrected ROI-scoped metric, rather than loosening any
validity gate to make the whole-frame number look better, was the chosen
fix — a documentation/measurement correction, not an algorithm change.

**Connection to I3.** Building the corrected ROI methodology (fixture-by-
fixture rectification review) surfaced a separate, real issue during this
same investigation: a localized occlusion-strip contamination pattern near
real depth transitions — carried forward into I3.

---

## I3 — Occlusion / dis-occlusion contamination

**Problem.** A genuine occlusion strip immediately adjacent to a real depth
step can read as high-confidence "valid" disparity — `StereoSGBM`'s own
smoothness prior extends the confidently-matched near surface across the
ambiguous occlusion zone. This is **not random noise**; at the pixel level
it is numerically indistinguishable from genuine correspondence, and
propagates into `obstacle_cloud`/`free_space_rays`/`BoundaryEvidence` while
`GeometryFrameQuality.overall_state` still reads `VALID`.

**Rejected approaches, and why:**
- **Statistical outlier filtering per cell** — fabricated disparity from
  the smoothness prior *looks* numerically coherent (it's a real, locally-
  consistent SGBM output), so per-cell statistical outlier tests could not
  reliably distinguish it from genuine data.
- **Planarity as a discriminator** — a contaminated region can still fit a
  plausible local plane; planarity alone doesn't separate "real surface"
  from "smoothness-prior extension of a neighboring real surface."
- **Invalid-neighbor density** — not discriminating; contaminated regions
  are, by construction, densely "valid," not sparse.
- **Generic per-cell outlier rejection** — damaged genuine boundaries when
  tuned aggressively enough to catch contamination, an unacceptable
  trade-off (this is exactly what I4's own boundary-precision work
  independently confirmed and fixed properly instead — see below).

**Chosen mechanism.** A **geometric**, not statistical, reliability signal:
`geometry.reliability.compute_shadow_zone_mask()` — predicts the shadow
region a real depth step would geometrically project, independent of what
the disparity map itself reports there. Confirmed live in source
(`src/depth_perception_engine/config/pipeline_config.py`):

| Config | Value |
|---|---|
| `geometry_shadow_zone_lookahead_px` | 8 px |
| `geometry_shadow_zone_gradient_threshold_px` | 3.0 px |
| `geometry_shadow_zone_max_width_px` | 40 px |

**One shared signal, not several heuristics.** `shadow_zone_mask` is
computed once per frame and threaded into every affected Level 3/4 builder
(`obstacle_cloud`, `free_space_rays`, `surface_evidence`, `boundary_evidence`,
and later, at I6.2, `ClearanceEvidence`) — a single, consistent reliability
signal reused everywhere it's relevant, rather than a separate ad hoc
detector per consumer.

**Measured effect.** Occlusion-strip contamination reduced ~13.6%
(`docs/VALIDATION_REPORT.md` I1-I6 addendum); boundary-specific false
positives closed entirely — see I4.

---

## I4 — Boundary precision

**Problem.** Low-support, noise-driven cells could still trip
`BoundaryEvidence`'s discontinuity admission, producing false positives
uncorrelated with any real depth transition.

**Fix.** `geometry/boundary.py`'s admission logic recalibrated to require
meaningful fractional support before admitting a discontinuity (Phase I4,
alongside I3's shadow-zone signal threading).

| Metric | Before | v1.1.1 |
|---|---|---|
| Precision | 87.8% | **100.0%** (TP=126, FP=0 — fresh rerun, `benchmarks/i4_boundary_precision/collect.py`) |
| Recall | 100.0% | **100.0%** (FN=0, unchanged) |
| Noise-driven low-support false positives | present | **0** |
| Genuine TN (correctly-quiet uniform pairs) | — | 54 |

**The important engineering result:** precision improved without
sacrificing measured recall — the boundary admission fix removed exactly
the false-positive population it targeted, with zero regression to genuine
(including weak/distant/narrow/occluded) boundaries.

**A number that could not be verified, and was not published.** A
"96.9%" historical precision figure was requested for inclusion in this
documentation pass. It does not appear anywhere in this repository's
committed docs, benchmark scripts, or git history — grep across
`docs/*.md` finds only 87.8%. It is omitted here rather than guessed; if a
second historical precision figure genuinely exists, it lives outside this
repository and should be sourced and cited properly before being added.

---

## I5 / I5.1 — Surface, opening, clearance

### Surface

**Measurement, and a real methodology subtlety found while re-verifying it
for this documentation pass.** `benchmarks/i5_surface_opening_clearance/
surface/measure.py`'s own top-level summary print statement buckets cells
by `planarity >= 0.95` alone, with no distinction between two different
fixture *families* in the same file: (1) well-posed fronto/yaw/pitch/
combined-slant planes at multiple ranges/textures (items 1-3 in `main()`),
and (2) deliberately adversarial `mixed_surface_cell`/`partial_invalid_cell`
fixtures (item 4, added later specifically to characterize the *separate*
partial-coverage failure mode below). A partial-coverage cell can still
self-report `planarity >= 0.95` despite covering only ~20-23% of its own
cell — confirmed directly: 28 such records exist in the current fixture
set. Re-running the script's own unscoped summary today therefore prints
`p95≈75-79°`, not the previously-documented `1.42°` — **not because the
underlying accuracy regressed, but because the print statement mixes two
fixture populations that were combined in one file for convenience, not
because they measure the same thing.**

Excluding those two labels (i.e., scoping to the original well-posed
fixture family the `1.42°` figure was always meant to describe) reproduces
it exactly:

| Metric (well-posed, full-coverage, planarity ≥0.95, n=32 cells) | Value |
|---|---|
| Median angular error | 0.515° |
| **p95 angular error** | **1.417°** |
| Max | 1.655° |

Target: ≤10° (preferred ≤5°) — comfortably met.

**Partial-coverage cells — a separate, already-known limitation, not mixed
into the figure above:**

| Metric (partial-coverage/mixed-surface fixtures, n=18 scored cells) | Value |
|---|---|
| Mean angular error | 36.7° |
| Range | 0.17° - 81.5° |

This is the same limitation independently confirmed during the v1.1.1
structural-closure pass's own Box-3 surface-normal audit (see
`docs/DPE_V1_PROVIDER_CONTRACT.md`'s D18 record): a cell's `support_fraction`
— not `planarity` alone — predicts whether its normal direction is
trustworthy. Documented, not fixed (would require a robust/weighted fitting
method, out of scope for a synthetic-fixture-driven pass — see "v1.1.1
structural closure" below).

### Opening

**Problem.** `geometry/opening.py`'s span-assembly logic could
spuriously split a single real opening into multiple sub-spans, and
rejected an entire multi-cell span outright when any one cell inside it
was a structurally-unobservable (SGBM dead-zone) cell rather than treating
that cell as a non-informative gap.

**Fix.** Span-assembly recalibrated to merge spuriously-split same-depth
spans and treat dead-zone cells as artificial, non-flank split points.

| Metric | Before | v1.1.1 |
|---|---|---|
| Precision | 100.0% | **100.0%** (TP=50, FP=0 — fresh rerun, unchanged) |
| Recall | 54.5% | **90.9%** (FN=5, all in the `partial_invalid` scenario) |
| Width error, median (confirmed) | — | 0.561% |
| Range error, median (confirmed) | — | 0.561% |
| Negative-fixture false openings | 0 | 0 (unchanged, reconfirmed via `benchmarks/i3_occlusion_safety/validate.py`) |

A verification note from this documentation pass: the opening benchmark's
own `single_step_not_opening` scenario carries `gt_expect_confirm: null`
(deliberately unscored — a diagnostic case, not a negative test). A first
precision/recall recomputation during this audit mis-treated `null` as
`False` and produced a spurious 5-false-positive result; correcting the
filter reproduced the documented 100%/90.9% exactly. Recorded here as an
example of exactly the kind of self-check this documentation pass is
supposed to perform before publishing a number.

### Clearance — magnitude accuracy (I5.1)

**Problem.** `ClearanceEvidence.nearest_distance_m` can exhibit **13-94%**
relative error in sectors whose column overlaps a real depth transition or
a narrow obstacle occupying a minority of the sector's own pixels.

**Root cause.** `obstacles.ThreatAssessor`'s IQR-filter + fixed-percentile
column aggregation: on a column with a real bimodal mixture (a genuine near
obstacle plus a numerically larger far/background population), IQR outlier
rejection can discard the smaller near cluster as a statistical outlier —
the aggregation reports a value derived mostly or entirely from the
background.

**Rejected recovery approaches.** Nearest-cluster / statistical recovery of
the true near-obstacle value was investigated and rejected: genuine
transition regions were found to contain **smeared intermediate disparity
values**, not a clean, separable hidden near cluster — there is no reliable
way to recover the true value once SGBM has already smeared a transition
(`benchmarks/i5_surface_opening_clearance/clearance_rootcause/
contiguity_gate_prototype.py`'s own investigation).

**Verdict.** No safe *accuracy* fix was found; the 13-94% magnitude
limitation remains present and unresolved in v1.1.1, by deliberate choice —
every aggregation alternative that fixed the idealized/synthetic model
regressed real pipeline data in a different way. The engineering priority
shifted from "fix the magnitude" to "never let a wrong value be reported as
an authoritative safety guarantee" — see I6 below.

---

## I6 / I6.3 — Final clearance safety closure

This section is deliberately the most detailed, because it is the clearest
example in this codebase's history of the benchmark methodology itself
being audited and corrected before trusting its result.

**A first re-audit found a previously-recorded safety statement — "no
false-clear sector was observed" — was inaccurate.** A fresh rerun measured
**28 of 252** ground-truthed `SUPPORTED` sectors (11.1%) as false-clear
(reported range >5cm farther than true range).

**That number was not accepted at face value.** Tracing it down surfaced
two separate, independently-verified findings:

**1. A benchmark-methodology bug.** `clearance/measure.py` reused *one*
`DepthPerceptionPipeline`/`ThreatAssessor` instance across ~9 unrelated
synthetic fixtures run in sequence. `ThreatAssessor`'s per-beam EMA-
smoothing/debounce state — a deliberate design for real video continuity —
leaked across those unrelated scenes. Fixed (fresh pipeline per fixture)
and directly reverified: **24 of the 28** false-clear sectors were this
artifact, not a pipeline defect. Corrected, methodology-clean baseline:
**4/252 (1.6%)**.

**2. Two genuine, distinct root causes for the remaining 4:**
- **Classical occlusion-shadow contamination** (`narrow_obstacle` beam8,
  3 sectors after the methodology fix) — closed by threading the existing
  I3 `compute_shadow_zone_mask` signal into
  `ThreatAssessor.assess()`/`ClearanceEvidence` construction
  (`clearance_shadow_zone_gating_enabled`, `contamination_threshold=0.30`,
  chosen with margin below the measured 0.42-0.67 true-positive overlap).
- **A wider (~20px), direction-agnostic SGBM smoothness-regularization
  ramp** (`multi_zone` beam13, 3 sectors) — invisible to the narrow
  occlusion-shadow model even mirrored bidirectionally. Closed by a second,
  independent signal, `geometry.reliability.compute_ramp_zone_mask`
  (`clearance_ramp_zone_window_px=24`, chosen with margin above the
  measured ~20px ramp width — 20px is the point the 0.30 threshold is
  first crossed; `clearance_ramp_zone_gradient_threshold_px=2.0`).

**Neither mechanism recovers the true value** — I5.1's own contiguity/
nearest-cluster prototypes already proved that's not reliably possible once
SGBM has smeared a transition. Both instead downgrade an otherwise-
`SUPPORTED` contaminated beam to `PARTIALLY_SUPPORTED` — an honest "less
confident" signal, never a fabricated corrected value.

**Final, reverified result:**

| Stage | False-clear sectors | % |
|---|---|---|
| Initial measurement | 28/252 | 11.1% |
| After benchmark-methodology fix | 4/252 | 1.6% |
| **After reliability gating (final)** | **0/252** | **0.0%** |

| Metric | Before fix | v1.1.1 |
|---|---|---|
| Worst-case `SUPPORTED`-sector error | 139.252% | 4.443% |

**Accepted cost:** ~30/252 sectors immediately adjacent to a genuine
transition — previously `SUPPORTED`, sometimes correct by chance — now read
`PARTIALLY_SUPPORTED` instead. Conservative-direction, individually
verified to sit next to a real transition, not scattered/spurious. Pure
decorrelated noise still never produces a confidently-`SUPPORTED` sector
(0/630 tested).

**No `GeometryFrame` contract change.** `ClearanceEvidence`'s fields/types
are unchanged; only which of the *existing* `support_state` values a
contaminated beam receives changed.

**What this is evidence of.** The benchmark itself was audited, an
incorrect measurement was corrected rather than trusted, the real
remaining failure was isolated to two distinct, independently-verified
mechanisms, and both were closed without touching the `GeometryFrame`
contract or fabricating a recovered value anywhere. This is the "0/252"
result on this qualified benchmark's own fixture population — see the
next section for why that phrase matters.

**`ClearanceEvidence` is evidence, not a universal guarantee.** During the
v1.1.1 structural-closure pass, an external MP01-sim integration exercising
`ClearanceEvidence` against a *different* scene (outside the 252 qualified
sectors) independently found 3 sectors reproducing the same class of
contamination this section closed for its own fixtures — proof that
"0/252" is this benchmark's own result on its own fixture population, not a
universal safety guarantee for every possible scene. See
`docs/RELEASE_NOTES_V1.md`'s "Known caveats" for the consumer-facing
statement this finding prompted. No DPE algorithm or threshold was changed
in response — see below.

---

## v1.1.1 — Structural closure

No perception-optimization work. Four parts, full detail in
`docs/DPE_V1_PROVIDER_CONTRACT.md`'s own D18 record — summarized here:

**A. Packaging repair.** A real `v1.1.0` defect: `pip install`/`pip wheel`
under genuine PEP 517 build isolation on Python 3.10 (Ubuntu 22.04
dist-packages layout) silently produced an empty wheel — correct metadata,
zero actual code, no error. Root-caused via a 7-step elimination sequence to
pip's own real isolated-subprocess build invocation specifically (not any
one setuptools version, not a bare-environment effect). Fixed by declaring
`packages`/`package_dir` explicitly in `setup.py` instead of relying on
`pyproject.toml`'s TOML-driven auto-discovery. Verified: isolated wheel,
normal install, editable install, and a real git-tag install, all on the
exact previously-failing environment.

**B. Surface-normal regression audit.** An external integration observed
`obstacle_box_3`'s `SurfaceEvidence` angular error change 20.3°→48.1°
between v1.0.1 and v1.1.0 on its own Gazebo fixture. Directly A/B-tested the
only suspect (I3's shadow-zone mask, added to `surface.py` in the same
commit) — byte-identical result with it on/off, ruled out. Traced instead
to the same release's already-applied, already-validated I1 SGBM P1/P2 fix
shifting which pixels are valid in a small, partial-coverage grid cell —
landing squarely in the *already-documented* partial-coverage limitation
above (support_fraction ~0.73, well within the 0.2-0.75 range this same
document already characterizes as unreliable). No structural defect found;
nothing changed, per this phase's own explicit no-tuning-against-a-fixture
scope.

**C. Contract documentation.** `GeometryFrame`'s contract unchanged.
Explicit "`ClearanceEvidence` is evidence, not guarantee" statement added to
`docs/RELEASE_NOTES_V1.md`, directly motivated by finding B's own sibling
finding above (the external integration's 3-sector false-clear on an
unqualified scene).

**D. Regression.** `pytest` 953/953 (950 D17 baseline + 3 new packaging
tests). `compare_to_baseline` 69/79 exact zero-delta. I4 boundary 100%/100%.
I5 clearance 0/252 false-clear, reconfirmed.

**DPE v1.1.1 — SOFTWARE/SIMULATION DEVELOPMENT FROZEN.** Further algorithm
optimization requires evidence from real stereo-camera/Jetson hardware
qualification — every remaining characterized limitation in this document
(weak-texture long-range depth, clearance transition-sector magnitude,
partial-coverage surface normals) is a synthetic-fixture finding; whether
and how each one matters on a real sensor is genuinely unknown until
measured on one.
