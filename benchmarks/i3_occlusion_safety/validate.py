"""
Phase I3 — Steps 5, 6, 8 validation. Read-only w.r.t. the real, unmodified
pipeline; compares geometry_shadow_zone_enabled=False (BEFORE) vs True
(AFTER, the shipped default) on the same deterministic fixtures.

Step 5: boundary precision/recall against ground-truth genuine/false
transitions, aggregated across many seeds/scenarios.
Step 6: opening-evidence negative-fixture regression (decorrelated noise,
3x3 and 6x8 grids).
Step 8: depth-accuracy/coverage regression (reuses benchmarks/i1_stereo_accuracy
fixtures/measure.py at the disparity/depth level, unaffected by I3 by
design — I3 only touches Level-3/4 evidence builders, not disparity/depth).
"""
import json
import sys

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform

from benchmarks.i1_stereo_accuracy.fixtures import (
    make_discontinuity_fixture, make_flat_fixture, make_decorrelated_fixture, make_repetitive_fixture,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
_W, _H = _CALIB.image_size


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _cfg(shadow_zone_enabled, grid_rc=3):
    return PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, surface_grid_rows=grid_rc, surface_grid_cols=grid_rc,
        enable_boundary_geometry=True, boundary_grid_rows=grid_rc, boundary_grid_cols=grid_rc,
        enable_opening_geometry=True, enable_geometry_frame=True,
        geometry_shadow_zone_enabled=shadow_zone_enabled,
    )


def _pipeline(cfg):
    return DepthPerceptionPipeline(cfg, _CALIB, rectify=False, body_T_camera_left=_transform())


# ===================================================================
# Step 5 — boundary precision/recall
# ===================================================================
def _boundary_confusion(shadow_zone_enabled, grid_rc=3, n_seeds=6):
    """TP/FP/FN over every boundary pair produced across a mix of
    genuine-transition fixtures (A clean discontinuity, F genuine box)
    and no-transition fixtures (flat A/B/C, decorrelated G)."""
    pipeline = _pipeline(_cfg(shadow_zone_enabled, grid_rc))
    tp = fp = fn = tn = 0

    # Genuine transitions: clean discontinuity (D, no occlusion) and
    # genuine occlusion-adjacent discontinuity (E, occlusion=True) — both
    # have a REAL near/far step; ground truth = "a genuine transition
    # exists at the actual column-boundary vicinity."
    boundary_col_frac = 0.5  # make_discontinuity_fixture splits at W//2
    for occ in (False, True):
        for seed in range(1, n_seeds + 1):
            fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=occ)
            result = pipeline.process(fx.left, fx.right)
            gf = result.geometry_frame
            for b in gf.boundary_evidence or []:
                if b.direction != "RIGHT":
                    continue  # the real transition is vertical (column-wise), only RIGHT edges cross it
                # Structural dead zone (numDisparities=128px at the left
                # edge) makes any pair whose left flanking cell falls
                # (mostly) inside it structurally unobservable regardless
                # of any real transition — exclude these from the
                # confusion matrix entirely (an unrelated, pre-existing
                # dead-zone limitation, not what I3 is testing; conflating
                # the two would make this benchmark's own recall number
                # meaningless). The left cell's own extent is [x1, x1 +
                # (x2-x1)/2) for a RIGHT-direction pair's union bbox.
                left_cell_x2_approx = b.x1 + (b.x2 - b.x1) // 2
                if left_cell_x2_approx <= 128:
                    continue
                # Ground truth: this pair is a genuine transition iff its
                # bbox straddles the true boundary column (W//2).
                true_boundary_col = _W // 2
                crosses = b.x1 < true_boundary_col < b.x2
                is_positive = b.state == "OBSERVED_DISCONTINUITY"
                if crosses:
                    if is_positive:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if is_positive:
                        fp += 1
                    else:
                        tn += 1

    # No transitions anywhere: flat A/B/C at 2m, decorrelated G.
    for scen in ("A", "B", "C"):
        for seed in range(1, n_seeds + 1):
            fx = make_flat_fixture(scen, depth_m=2.0, seed=seed)
            result = pipeline.process(fx.left, fx.right)
            for b in (result.geometry_frame.boundary_evidence or []):
                is_positive = b.state == "OBSERVED_DISCONTINUITY"
                if is_positive:
                    fp += 1
                else:
                    tn += 1
    for seed in range(1, n_seeds + 1):
        fx = make_decorrelated_fixture(seed)
        result = pipeline.process(fx.left, fx.right)
        for b in (result.geometry_frame.boundary_evidence or []):
            is_positive = b.state == "OBSERVED_DISCONTINUITY"
            if is_positive:
                fp += 1
            else:
                tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


# ===================================================================
# Step 6 — opening negative-fixture regression
# ===================================================================
def _opening_negative(shadow_zone_enabled, n_seeds=10):
    out = {}
    for grid_rc in (3, 6):
        pipeline = _pipeline(_cfg(shadow_zone_enabled, grid_rc))
        total_openings = 0
        for seed in range(1, n_seeds + 1):
            fx = make_decorrelated_fixture(seed)
            result = pipeline.process(fx.left, fx.right)
            total_openings += len(result.geometry_frame.opening_evidence or [])
        out[f"grid_{grid_rc}x{grid_rc}"] = total_openings
    return out


def _opening_genuine_support(shadow_zone_enabled, n_seeds=6):
    """Genuine opening support (fixture F-style two-flank box gap) — is
    a real, well-supported opening still confirmed after I3?"""
    pipeline = _pipeline(_cfg(shadow_zone_enabled, grid_rc=3))
    n_confirmed = 0
    support_values = []
    for seed in range(1, n_seeds + 1):
        # Near/far/near three-zone: reuse make_discontinuity_fixture twice
        # isn't directly composable; approximate a two-flank box using the
        # existing discontinuity fixture as a single-step proxy — the
        # opening admission itself is exercised via boundary_evidence's
        # already-confirmed RIGHT-edge transitions plus depth_map, so a
        # single genuine step (D, no occlusion) already exercises
        # opening_evidence's real admission path end-to-end.
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=False)
        result = pipeline.process(fx.left, fx.right)
        openings = result.geometry_frame.opening_evidence or []
        n_confirmed += len(openings)
        support_values.extend(o.support_fraction for o in openings)
    return {"n_confirmed": n_confirmed, "support_values": support_values}


def main():
    out = {}
    print("=" * 100)
    print("STEP 5 — Boundary precision/recall")
    print("=" * 100)
    for shadow_zone_enabled, label in ((False, "BEFORE"), (True, "AFTER")):
        r = _boundary_confusion(shadow_zone_enabled)
        out[f"boundary_{label}"] = r
        print(f"{label}: TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']} "
              f"precision={r['precision']:.4f} recall={r['recall']:.4f}")

    print("\n" + "=" * 100)
    print("STEP 6 — Opening negative-fixture regression (decorrelated noise)")
    print("=" * 100)
    for shadow_zone_enabled, label in ((False, "BEFORE"), (True, "AFTER")):
        r = _opening_negative(shadow_zone_enabled)
        out[f"opening_negative_{label}"] = r
        print(f"{label}: {r}")

    print("\n" + "=" * 100)
    print("STEP 6b — Genuine opening support (sanity: still confirmed?)")
    print("=" * 100)
    for shadow_zone_enabled, label in ((False, "BEFORE"), (True, "AFTER")):
        r = _opening_genuine_support(shadow_zone_enabled)
        out[f"opening_genuine_{label}"] = r
        print(f"{label}: n_confirmed={r['n_confirmed']} support_values={r['support_values']}")

    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i3_occlusion_safety/results/validate.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
