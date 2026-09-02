# D2 — Observation Identity Contract

**Phase D2 — DPE observation identity contract repair + final qualification.**
Closes the single P0 defect found by D1
(`benchmarks/d1_execution/D1_EXECUTION_AUDIT_REPORT.md`, section 6).

This document is BOTH the frozen public contract for observation identity and the
D2 qualification report.

---

## 1. Starting repository state

| Item | Value |
|---|---|
| Branch | `release/v1.2.0` |
| HEAD at start | `f4ce645e73bef01930a59b3ed7d68c4adda1b165` |
| `git status --short` at start | `?? benchmarks/d1_execution/` (D1 artefacts only) |
| Package version | `1.2.0` (**unreleased** — latest tag is `v1.1.1`) |
| Baseline suite | **983 passed, 0 failed, 0 skipped, 4 known warnings** (20.17 s) |

The 4 warnings are pre-existing `RuntimeWarning: invalid value encountered in cast`
from `temporal/rotation_compensation.py:223-224`, emitted by
`tests/test_d11_degradation_validation.py`'s deliberate NaN/Inf degradation tests.
Unchanged by D2.

---

## 2. The exact P0 defect

D1 proved, by direct probe and by repository-wide source scan:

1. `StereoObservation.frame_id: Optional[str]` existed and was documented as
   *"optional caller-side observation identity"*.
2. **No production code path read it.** A scan for `observation.frame_id` /
   `obs.frame_id` across all of `src/depth_perception_engine/` returned zero hits.
3. `StereoObservation(frame_id="observation-X")` → `process_geometry_frame()`
   produced a `GeometryFrame` in which the string `"observation-X"` appeared
   **nowhere** in the entire object graph.
4. `GeometryFrame.frame_id` is **not** observation identity — it is the
   coordinate frame, hardcoded to `FrameId.CAMERA_OPTICAL_LEFT`, and every nested
   `.frame_id` is likewise a coordinate frame (`camera_optical_left` / `body`).
5. `GeometryFrame` therefore carried **no immutable observation/transaction
   identity at all**.
6. `timestamp` cannot substitute: optional, caller-defined, unit-unenforced, not
   guaranteed unique, and a duplicate-timestamp frame still returns a full
   `GeometryFrame`.

This blocked the invariant `GeometryFrame X + SemanticFrame X`, where both must
provably originate from the same observation.

---

## 3. Chosen public-contract migration

**Strategy B — additive `observation_id`, with the pre-existing `frame_id`
retained as a deprecated, deterministic alias.**

```
StereoObservation.observation_id : Optional[str] = None   # authoritative
StereoObservation.frame_id       : Optional[str] = None   # DEPRECATED alias
StereoObservation.resolved_observation_id  ->  the ONE authoritative value

GeometryFrame.observation_id     : Optional[str] = None   # verbatim copy
GeometryFrame.frame_id           : str                    # COORDINATE frame, unchanged
```

Resolution rule (`StereoObservation.resolved_observation_id`):

| `observation_id` | `frame_id` | Result |
|---|---|---|
| `None` | `None` | `None` — "no identity supplied", always legal |
| `"X"` | `None` | `"X"` |
| `None` | `"X"` | `"X"` — legacy caller keeps working, now propagates |
| `"X"` | `"X"` | `"X"` |
| `"X"` | `"Y"` | **`ValueError`** — ambiguous; refused deterministically |

There is exactly ONE authoritative identity. No DPE code path reads
`.observation_id` or `.frame_id` directly for propagation — every path goes
through `resolved_observation_id`.

---

## 4. Why this migration was selected

A clean rename (strategy A) was evaluated first and **rejected on repository
evidence**, not on caution.

**Evidence gathered (AST scan of `src/`, `tests/`, `examples/`, `benchmarks/`):**

* 35 `StereoObservation(...)` construction sites exist in this repository; only
  **4** pass `frame_id=`, and all 4 are incidental (three tests using it as filler,
  plus D1's own benchmark probe). Nothing asserts on it.
* **Zero** `.frame_id` attribute reads on a `StereoObservation` receiver anywhere.

That alone would have permitted a rename. Two further findings changed the answer:

**(a) A real external consumer already passes it.**
`hybrid_perception_engine/src/hybrid_perception_engine/providers/dpe_provider.py:126`
constructs:

```python
dpe_observation = StereoObservation(
    ..., frame_id=observation.frame_id, ...
)
```

A rename would break that at construction with `TypeError: unexpected keyword
argument 'frame_id'`. D2's own instruction is explicit: *"Do not blindly break
compatibility."*

**(b) That consumer already means observation identity by it.**
`hybrid_perception_engine/contracts/observation.py`'s `SensorObservation`
docstring states:

> `frame_id` is the single, authoritative **observation identity** HPE assigns to
> both providers — see contracts/provenance.py … for why this (not either engine's
> own returned frame_id) is the correlation source of truth.

So aliasing `frame_id → observation_id` **preserves exactly what the only real
caller already means**. It is a truthful migration, not a compatibility hack — and
that same docstring is independent confirmation of the D1 defect, seen from the
consumer side: HPE had to keep its own side-channel provenance precisely because
DPE never echoed the identity back.

**Counter-evidence acknowledged.** `tests/test_d12_sensor_contract_independence.py:220`
passed `frame_id=FrameId.CAMERA_OPTICAL_LEFT` — a *coordinate-frame constant* — into
the observation, and HPE's own test fixture builds `GeometryFrame(frame_id="frame-0001")`.
Both are real instances of engineers conflating the two axes. This is why the field is
**deprecated with a documented removal path** rather than blessed: the name is genuinely
bad. D2's rule *"Do not preserve a bad ambiguous field merely because it existed"* is
honoured by deprecating it and making `observation_id` authoritative — not by keeping
two independent identities.

**Version context.** `v1.2.0` is unreleased (latest tag `v1.1.1`), but `v1.1.1`
shipped `frame_id` as a public field of a root-exported dataclass. Removal belongs in
a major version, not a patch phase whose charter is a contract *repair*.

---

## 5. Exact files changed

### Production source (5 files, +165 / −4 lines)

| File | Change |
|---|---|
| `models/result.py` | `StereoObservation`: `+observation_id`, `+__post_init__` validation, `+resolved_observation_id` property, deprecation docs for `frame_id`. `DepthPerceptionResult`: `+observation_id`. |
| `geometry/provider.py` | `GeometryFrame`: `+observation_id` (defaulted, appended last); module docstring gains an explicit **identity vs coordinate-frame** section. |
| `fusion/result_builder.py` | `build_result(..., observation_id=None)` forwards to the result; `build_geometry_frame()` copies `result.observation_id` verbatim. |
| `pipeline/pipeline.py` | `process_observation()` reads `observation.resolved_observation_id` once and forwards it to `build_result()`; legacy `process()` gains a trailing `observation_id` parameter; stale "frame_id is unread" docstring corrected. |
| `standalone/interface.py` | `build_observation(..., observation_id=None)` pass-through. No second propagation implementation. |

### Tests (4 files)

| File | Change |
|---|---|
| `tests/test_d2_observation_identity.py` | **New** — 45 tests (the D2 matrix). |
| `tests/test_geometry_frame.py` | Approved-field-list guard updated. |
| `tests/test_pipeline_geometry.py` | Field-order guard updated. |
| `tests/test_d13_external_consumer.py` | Exhaustiveness guard updated **plus a real new coverage test** (the guard demands genuine exercise, not an allowlist entry). |

Those three guard failures were the intended safety net firing: each pins an exact
contract field list so a contract change cannot happen silently.

### Documentation (3 files)

`docs/D2_OBSERVATION_IDENTITY_CONTRACT.md` (this file, new),
`docs/DATA_CONTRACTS.md`, `docs/DUAL_INTERFACE_ARCHITECTURE.md`.

### Benchmarks

`benchmarks/d2_observation_identity/` (new) — focused A/B only.

---

## 6. `StereoObservation` — before / after

**Before (v1.1.1 / pre-D2)**

```python
left_image, right_image, left_timestamp, right_timestamp,
calibration, frame_id, motion_hint, motion_hints
# no __post_init__; frame_id inert
```

**After (D2)**

```python
left_image, right_image, left_timestamp, right_timestamp,
calibration, frame_id, motion_hint, motion_hints,
observation_id                       # <- appended LAST
+ __post_init__                      # identity validation only
+ resolved_observation_id (property) # the ONE authoritative accessor
```

`observation_id` is appended **last** deliberately: every pre-D2 positional
construction up to and including `motion_hints` keeps its exact meaning. Field
order is a compatibility artefact, not a statement about importance.

`__post_init__` is new and validates **identity only** — it imposes no format,
length, charset, uniqueness or ordering constraint:

* `str` or `None`, else `ValueError`;
* non-empty when supplied, else `ValueError` (mirrors
  `frames.RigidTransform.__post_init__`'s own frozen non-empty-identifier rule;
  `""` is neither an identity nor the explicit "no identity" signal, which is `None`);
* conflicting `observation_id` vs `frame_id` → `ValueError`.

---

## 7. `GeometryFrame` — before / after

**Before:** 22 fields, all required, no identity.
**After:** 23 fields — `observation_id: Optional[str] = None` appended last.

Defaulted and trailing so that **existing keyword construction keeps working
unmodified**, including external fixtures: `hybrid_perception_engine/tests/conftest.py:52`
and `hybrid_perception_engine/benchmarks/h5_performance_baseline.py:97` both build
`GeometryFrame(...)` with all-keyword arguments and no `observation_id`. A required
field would have broken both.

The module docstring now opens with an explicit two-axis statement so the
distinction cannot be missed by a future implementer.

---

## 8. Observation-ID propagation graph

```
StereoObservation
  ├── observation_id ─────┐
  └── frame_id (deprec.) ─┤
                          v
        resolved_observation_id          models/result.py  (the ONE resolver)
                          |
                          v
   DepthPerceptionPipeline.process_observation()            pipeline.py:414
     read ONCE, verbatim; nothing between here and
     build_result() inspects, parses, or branches on it
                          |
                          v
        build_result(..., observation_id=...)          result_builder.py:87,165
                          |
                          v
        DepthPerceptionResult.observation_id
                          |
                          v
        build_geometry_frame(result)                   result_builder.py:462
          observation_id=result.observation_id
                          |
                          v
        GeometryFrame.observation_id
```

**One geometry implementation, one propagation path.** Routing identity through
`DepthPerceptionResult` is deliberate and structural: `build_geometry_frame()`
builds `GeometryFrame` purely by reading fields off `result`, so **both**
GeometryFrame-producing paths — `process_observation()`'s own
`enable_geometry_frame` branch and `process_geometry_frame()` — share that one
builder and are incapable of drifting apart. Proved by
`test_both_geometry_frame_paths_agree`.

Entry points that can now carry identity:

| Entry point | Identity | Notes |
|---|---|---|
| `process_geometry_frame(obs)` | ✅ | **The HPE runtime path** |
| `process_observation(obs)` | ✅ | on `DepthPerceptionResult` *and* its `geometry_frame` |
| `process(left, right, ..., observation_id=...)` | ✅ | trailing defaulted parameter; no existing call site shifts |
| `StandaloneStereoInterface.build_observation(..., observation_id=...)` | ✅ | pure pass-through into the same core |

`TemporalRecord` deliberately did **not** gain the field — chronology has no use
for it, and storing it would create a second place identity could accumulate.

---

## 9. Coordinate-frame preservation evidence

Every coordinate frame is unchanged, verified three independent ways.

**(a) Direct assertion** (`test_observation_identity_never_leaks_into_any_coordinate_frame`)
with `observation_id = "HPE-OBSERVATION-12345"`:

| Field | Value after D2 |
|---|---|
| `GeometryFrame.observation_id` | `"HPE-OBSERVATION-12345"` |
| `GeometryFrame.frame_id` | `camera_optical_left` |
| `geometry.frame_id` | `camera_optical_left` |
| `geometry_body.frame_id` | `body` |
| `obstacle_cloud.frame_id` | `body` |
| `free_space_rays.frame_id` | `body` |
| every `surface_evidence[i].frame_id` | `body` |
| every `boundary_evidence[i].frame_id` | `camera_optical_left` |
| every `clearance_evidence[i].frame_id` | `camera_optical_left` |
| every `region_evidence[k].frame_id` | `camera_optical_left` |
| every `opening_evidence[i].frame_id` | `camera_optical_left` |

**(b) Recursive object-graph scan** (`test_recursive_scan_finds_identity_only_in_observation_id`)
— walks the entire returned graph to depth 5 and asserts the identity string appears
at exactly `["GeometryFrame.observation_id"]` and nowhere else. This is the exact
inverse of D1's probe, which found zero occurrences.

**(c) Differential** (`test_frame_ids_are_identical_with_and_without_identity`) —
the full tuple of frame IDs is identical with `observation_id=None` and with an
identity supplied.

---

## 10. Temporal non-regression evidence

`TemporalHistory` still keys on **timestamp only**, exactly as D1 measured.

| Scenario | Result | Test |
|---|---|---|
| 5 frames sharing ONE repeated identity, advancing timestamps | all `ACCEPTED`, history = 5 | `test_repeated_ids_on_advancing_timestamps_are_all_accepted` |
| Distinct identities, **duplicate** timestamp | `REJECTED_DUPLICATE_TIMESTAMP` (identity does not rescue it) | `test_distinct_ids_do_not_rescue_a_duplicate_timestamp` |
| Distinct identities, **decreasing** timestamp | `REJECTED_OLDER_TIMESTAMP` | `test_distinct_ids_do_not_rescue_a_decreasing_timestamp` |
| 30 s forward jump with identities | `ACCEPTED_NEW_SEQUENCE`, history reset to 1 | `test_large_gap_still_starts_a_new_sequence` |
| `TemporalRecord` contents | carries no identity; identity absent from its `repr` | `test_temporal_history_stores_no_observation_identity` |

A chronology-rejected frame still returns its own identity — provenance is reported
even when the frame is temporally rejected.

### Numerical non-regression — byte-identical

Pristine pre-D2 source (`git archive f4ce645`) and current source were each driven
through an identical 6-frame qualified run and fingerprinted (SHA-256 of every
raster, plus every evidence family):

```
pre-D2  sha256: c52a7c1a946bb3b1cc84c85f81743cffa3d8a3b173ff205444619ef718c02d1f
post-D2 sha256: c52a7c1a946bb3b1cc84c85f81743cffa3d8a3b173ff205444619ef718c02d1f
RESULT: BYTE-IDENTICAL
```

Covering: `disparity_map`, `depth_map`, `valid_disparity_mask`, `valid_depth_mask`,
`geometry`, `geometry_body`, `obstacle_cloud`, `free_space_rays`, `geometry_metrics`,
`temporal_consistency`, `temporal_stabilization`, `rotation_compensation_status`,
`motion_aware_reliability`, `temporal_persistence`, `surface_evidence`,
`boundary_evidence`, `opening_evidence`, `clearance_evidence`, `region_evidence`,
`quality`. **No tolerance was loosened anywhere** — this is exact equality, not
approximate.

Additionally, `test_two_sequences_differing_only_in_identity_are_output_identical`
runs the same 6-frame sequence with and without identities and asserts every field
except `observation_id` is bit-for-bit identical, and
`test_production_never_branches_on_observation_id` is an AST guard proving
`observation_id` never appears in a conditional, comparison, or `match` anywhere in
production source outside its own contract validation.

---

## 11. Standalone / legacy compatibility

| Consumer | Status |
|---|---|
| `DepthPerceptionPipeline.process(...)` | ✅ unchanged; new trailing `observation_id` parameter, defaulted |
| `StandaloneStereoInterface` | ✅ pass-through only; still delegates to the one core |
| `pipeline.api` Tier-2 helpers | ✅ untouched |
| **`hybrid_perception_engine`** (`frame_id=` at `dpe_provider.py:126`) | ✅ **keeps working, and now receives identity back** on `GeometryFrame.observation_id` |
| **HPE `GeometryFrame(...)` fixtures** (conftest, h5 benchmark) | ✅ unaffected — new field is defaulted and trailing |
| `mp01_perception` (uses `process()`) | ✅ unaffected — `process()` never exposed `frame_id` |

HPE was **not modified**, as required. It gains identity propagation for free
because its existing `frame_id=` argument is now the deprecated alias.

---

## 12. Performance A/B

Same warmed-pipeline methodology as D1: one instance per arm, warmup discarded,
identical images/timestamps/motion, arms interleaved in **ABBA** order across 4
trials (n = 1000 per arm) so machine-load drift cannot bias one arm.

| Arm | median | P95 |
|---|---|---|
| A — `observation_id=None` | 34.550 ms | 39.846 ms |
| B — `observation_id="benchmark-observation-X"` | 33.790 ms | 40.481 ms |
| **Δ** | **−0.760 ms (−2.20 %)** | +0.635 ms |

The median delta is **negative** — identity cannot make the pipeline faster, so the
sign itself proves this is noise. Confirmed by the noise floor: within-arm spread
across trials was **3.20 ms / 4.42 ms**, several times the |Δ|. (A first,
non-interleaved run gave +1.05 ms; the sign flipped under ABBA, which is exactly
what an ordering artefact does.)

Because a ~1 ms full-pipeline noise floor cannot resolve a sub-microsecond change,
the added work was also measured **directly** (n = 200 000):

| Operation | Cost |
|---|---|
| `StereoObservation(...)` without identity | 1.081 µs |
| `StereoObservation(...)` with identity (incl. new `__post_init__`) | 1.167 µs |
| construction delta | **+0.087 µs** |
| `resolved_observation_id` property read | **+0.032 µs** |
| **total added per frame** | **0.118 µs = 0.00034 % of a frame** |

### Memory / retention

600 frames each carrying a **unique 220-character** identity:

* steady-state RSS growth **−0.29 MB** (flat), versus **−0.24 MB** for the same run
  without identities — indistinguishable;
* `TemporalHistory` still bounded at exactly 30 / `temporal_max_records`;
* a live-object scan found **0** identity strings still reachable anywhere in the
  process after the run;
* the pipeline's attribute set is unchanged, and no identity appears in its `repr`.

DPE retains no identity in any history or buffer.

---

## 13. Dependency / architecture guard result

| Guard | Result |
|---|---|
| `threading` / `queue` / `asyncio` / `multiprocessing` / `concurrent.futures` imports in production source | **none** |
| `rclpy` / ROS / CUDA / TensorRT / torch imports | **none** |
| Imports of HPE / NPE / `mp01_perception` | **none** (only prose mentions in docstrings) |
| Locks / semaphores / executors / queues | **none** |
| Internal input queue added | **no** |
| Internal result buffer added | **no** |
| Execution policy added | **no** |
| One-active-call-per-instance model | **unchanged** |
| DPE made thread-safe | **no** (deliberately) |

`threading` does appear in `sys.modules` after importing DPE — it is pulled in
transitively by **NumPy and OpenCV** (verified: `import numpy` alone loads it).
DPE's own source contains **zero** occurrences of the string `threading`.

---

## 14. New tests

`tests/test_d2_observation_identity.py` — **45 tests**, mapping to the D2 matrix:

| # | Requirement | Class |
|---|---|---|
| 1 | `observation_id=None` → `None` | `TestPropagation` |
| 2 | normal string preserved exactly | `TestPropagation` |
| 3 | 9 unusual-but-legal opaque strings, no normalization (whitespace, unicode, JSON-like, 4096-char, path-like) | `TestPropagation` |
| 4 | repeated IDs not rejected | `TestTemporalSemanticsUnchanged` |
| 5 | distinct IDs on duplicate timestamps | `TestTemporalSemanticsUnchanged` |
| 6 | distinct IDs on decreasing timestamps | `TestTemporalSemanticsUnchanged` |
| 7 | large gap → `ACCEPTED_NEW_SEQUENCE` | `TestTemporalSemanticsUnchanged` |
| 8 | `reset()` leaks no stale identity | `TestResetAndLifecycle` |
| 9 | coordinate-frame non-regression (3 tests incl. recursive scan) | `TestCoordinateFrameNonRegression` |
| 10 | `process_geometry_frame` propagation | `TestPropagation` |
| 11 | `process_observation` propagation | `TestPropagation` |
| 12 | standalone / legacy compatibility | `TestLegacyAndStandaloneCompatibility` |
| 13 | frozen + slotted behaviour preserved | `TestContractShape` |
| 14 | equality / repr behaviour | `TestContractShape` |
| 15 | no algorithm branches on identity (differential + AST guard) | `TestIdentityHasNoAlgorithmicEffect` |
| — | deprecated-alias contract + conflict + validation | `TestDeprecatedFrameIdAlias` |

Plus 1 new test in `tests/test_d13_external_consumer.py`.

**One test was corrected rather than made to pass.** An initial
`test_equality_distinguishes_observations_by_identity` asserted `a != b` on two
`StereoObservation`s. That is false — and was false before D2: these contracts carry
ndarray fields, so the generated `__eq__` raises *"truth value of an array is
ambiguous"*. The test now records that pre-existing limitation explicitly and
asserts what is actually true (identity is a `compare=True` field; correlation is a
plain string comparison on the field itself).

---

## 15. Full regression result

| | Baseline (pre-D2) | Final (post-D2) | Δ |
|---|---|---|---|
| passed | 983 | **1029** | **+46** |
| failed | 0 | **0** | 0 |
| skipped | 0 | **0** | 0 |
| errors | 0 | **0** | 0 |
| warnings | 4 | **4** | 0 |
| runtime | 20.17 s | 21.43 s | +1.26 s |

+46 = 45 new D2 tests + 1 new `test_d13` coverage test. No test was deleted,
skipped, or had its tolerance loosened.

---

## 16. Remaining issues

**P0 — none.** The D1 P0 is closed.

| Class | Issue |
|---|---|
| **P1** | `StereoObservation.frame_id` remains as a deprecated alias. It should be **removed in the next major version**. Removal is a breaking change requiring a coordinated update of `hybrid_perception_engine/providers/dpe_provider.py:126` (one line: `frame_id=` → `observation_id=`). Tracked, documented, and covered by tests; not a blocker. |
| **P1** | `StereoObservation` gained a `__post_init__` where it previously had none. Identity-only validation, but it means a construction passing a non-string/empty identity now raises where it previously silently accepted. Intentional and consistent with `RigidTransform`/`StereoCalibration`/`MotionHint`; noted because it is technically a behaviour change. |
| **P2** | Two in-repo sites still conflate the axes cosmetically: `tests/test_d12_sensor_contract_independence.py:220` passes `frame_id=FrameId.CAMERA_OPTICAL_LEFT` (now interpreted as an observation id — harmless, its assertion is about `GeometryFrame.frame_id` which is hardcoded), and HPE's own fixture builds `GeometryFrame(frame_id="frame-0001")`. Neither affects behaviour; both are naming hygiene for a later pass. |
| **P3** | `DepthPerceptionResult.observation_id` is carried mainly as the structural route that keeps both GeometryFrame paths sharing one builder. It is also genuinely useful to `process()`/`process_observation()` callers, so it is not dead weight — recorded only so the rationale is not lost. |
| **P3** | Version number left at `1.2.0`. D2 adds a backward-compatible public field plus a deprecation; **recommendation only** (no release operation performed): this warrants a minor bump at release time, and the `frame_id` removal warrants a major. |

---

## 17. The frozen contract — for an external orchestrator

> **`observation_id`** — opaque, caller-owned **transaction identity** used for
> provenance and correlation by external orchestration layers. DPE copies it
> verbatim from `StereoObservation` to `GeometryFrame` and interprets it in no
> other way: it never generates, parses, normalizes, sequences, or branches on the
> value, and never uses it for temporal admission or any algorithmic decision.
> `None` is legal and means "no identity supplied".
>
> **`frame_id`** — **coordinate-frame identity**, describing the coordinate system
> geometric evidence is expressed in (`camera_optical_left`, `body`). Entirely
> unrelated to `observation_id`. Unchanged by D2.
>
> **`timestamp`** — capture/observation time, consumed by DPE's temporal
> algorithms. **Not guaranteed unique, optional, caller-defined, unit-unenforced,
> and NOT an observation transaction identity.**

**Joining provider results.** An external orchestrator should join provider
results **by `observation_id`**. `timestamp` may be used as *secondary validation*
(e.g. sanity-checking that two joined frames carry the plausible capture time) but
must never be the primary exact join key, for the six reasons in section 2.

DPE has no opinion about, and no dependency on, how such an orchestrator is built;
this section describes an external use case only.

---

## 18. Final verdict

Observation identity is explicit, propagates verbatim on every authoritative path,
leaves coordinate-frame semantics untouched, has zero effect on temporal behaviour
or numerical output (byte-identical), costs 0.118 µs per frame, retains nothing,
and adds no execution policy of any kind.

**D2 OBSERVATION IDENTITY CONTRACT: PASS — DPE READY TO FREEZE FOR HPE**
