"""
Phase I2, Step 6 — negative-fixture safety recheck (CORRECTED methodology:
rectify=False for synthetic fixtures, matching fixtures.py's own intended
usage and test_d10_black_box_provider.py's established precedent — I1.1's
own safety_closure.py used the DepthPerceptionPipeline default rectify=True
against synthetic unrectified images, which corrupts correspondence; this
script fixes that methodology bug) + the isolated-noise-pocket structural
test I1.1 flagged as untested.
"""
import sys, json
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")
import numpy as np
from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from benchmarks.i1_stereo_accuracy.fixtures import (
    make_flat_fixture, make_discontinuity_fixture, make_decorrelated_fixture, W, H,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")


def _transform():
    return RigidTransform(rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
                           from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY)


def _full_v1_config(**overrides):
    defaults = dict(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, enable_boundary_geometry=True, enable_opening_geometry=True,
        enable_geometry_frame=True,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _pipeline(cfg, rectify=False):
    return DepthPerceptionPipeline(cfg, _CALIB, rectify=rectify, body_T_camera_left=_transform())


def _trace(result):
    gf = result.geometry_frame
    body_valid = int(result.geometry_body.valid_mask.sum()) if result.geometry_body is not None else None
    obs = int(result.obstacle_cloud.points.shape[0]) if result.obstacle_cloud is not None else None
    rays = int(result.free_space_rays.ranges_m.shape[0]) if result.free_space_rays is not None else None
    return {
        "whole_frame_valid_fraction": float(result.valid_disparity_mask.mean()),
        "body_valid": body_valid, "obstacle_count": obs, "ray_count": rays,
        "unknown_invariant_holds": (obs == body_valid and rays == body_valid),
        "clearance_supported": sum(1 for c in gf.clearance_evidence if c.support_state == "SUPPORTED") if gf.clearance_evidence else 0,
        "clearance_n": len(gf.clearance_evidence) if gf.clearance_evidence else 0,
        "boundary_observed": sum(1 for b in gf.boundary_evidence if b.state == "OBSERVED_DISCONTINUITY") if gf.boundary_evidence else 0,
        "opening_n": len(gf.opening_evidence) if gf.opening_evidence else 0,
        "opening_support_fractions": [round(o.support_fraction, 4) for o in gf.opening_evidence] if gf.opening_evidence else [],
        "quality_overall": gf.quality.overall_state if gf.quality else None,
        "quality_geometry_validity": gf.quality.geometry_validity_state if gf.quality else None,
        "quality_reasons": gf.quality.degradation_reasons if gf.quality else None,
    }


def recheck_negatives():
    print("[STEP6] Negative-fixture recheck, CORRECTED methodology (rectify=False)\n")
    cfg = _full_v1_config()
    pipeline = _pipeline(cfg, rectify=False)
    out = []

    for seed in range(1, 6):
        fx = make_decorrelated_fixture(seed)
        r = _trace(pipeline.process(fx.left, fx.right))
        r.update({"scenario": "G_noise", "seed": seed})
        out.append(r)
        print(f"  G_noise seed{seed}: valid_frac={r['whole_frame_valid_fraction']:.4f} "
              f"quality={r['quality_overall']}/{r['quality_geometry_validity']} "
              f"invariant_holds={r['unknown_invariant_holds']} openings={r['opening_n']}")

    for seed in range(1, 6):
        fx = make_discontinuity_fixture(near_m=1.5, far_m=5.0, seed=seed, occlusion=True)
        r = _trace(pipeline.process(fx.left, fx.right))
        r.update({"scenario": "E_occlusion", "seed": seed})
        out.append(r)
        print(f"  E_occlusion seed{seed}: valid_frac={r['whole_frame_valid_fraction']:.4f} "
              f"quality={r['quality_overall']}/{r['quality_geometry_validity']} "
              f"invariant_holds={r['unknown_invariant_holds']} openings={r['opening_n']} "
              f"clearance_supported={r['clearance_supported']}/{r['clearance_n']}")

    for seed in range(1, 6):
        fx = make_flat_fixture("C", depth_m=6.0, seed=seed)
        r = _trace(pipeline.process(fx.left, fx.right))
        r.update({"scenario": "C_weak_6m", "seed": seed})
        out.append(r)
        print(f"  C_weak_6m seed{seed}: valid_frac={r['whole_frame_valid_fraction']:.4f} "
              f"quality={r['quality_overall']}/{r['quality_geometry_validity']} "
              f"invariant_holds={r['unknown_invariant_holds']} openings={r['opening_n']}")

    return out


def isolated_noise_pocket():
    print("\n[STEP6] Isolated-noise-pocket structural test\n")
    cfg = _full_v1_config()
    pipeline_a = _pipeline(cfg, rectify=False)
    pipeline_b = _pipeline(cfg, rectify=False)

    results = []
    for (size, cy, cx) in [(30, 120, 220), (40, 80, 250), (40, 160, 200)]:
        fx = make_flat_fixture("A", depth_m=2.0, seed=1)
        base_left, base_right = fx.left.copy(), fx.right.copy()

        control = pipeline_a.process(base_left, base_right)
        control_trace = _trace(control)

        rng = np.random.default_rng(999)
        noisy_left = base_left.copy()
        noisy_right = base_right.copy()
        patch_l = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        patch_r = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)  # independent -> zero correspondence
        noisy_left[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2] = patch_l
        noisy_right[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2] = patch_r

        test = pipeline_b.process(noisy_left, noisy_right)
        test_trace = _trace(test)

        finding = {
            "patch": {"size": size, "cy": cy, "cx": cx},
            "control_quality_overall": control_trace["quality_overall"],
            "control_valid_frac": control_trace["whole_frame_valid_fraction"],
            "control_openings": control_trace["opening_n"],
            "control_clearance_supported": control_trace["clearance_supported"],
            "test_quality_overall": test_trace["quality_overall"],
            "test_valid_frac": test_trace["whole_frame_valid_fraction"],
            "test_openings": test_trace["opening_n"],
            "test_opening_support": test_trace["opening_support_fractions"],
            "test_clearance_supported": test_trace["clearance_supported"],
            "new_opening_appeared": test_trace["opening_n"] > control_trace["opening_n"],
            "new_supported_clearance_appeared": test_trace["clearance_supported"] > control_trace["clearance_supported"],
            "frame_stayed_valid_overall": test_trace["quality_overall"] == "VALID",
        }
        results.append(finding)
        print(f"  patch size={size} @({cy},{cx}): control quality={control_trace['quality_overall']} "
              f"valid={control_trace['whole_frame_valid_fraction']:.4f} openings={control_trace['opening_n']}  |  "
              f"test quality={test_trace['quality_overall']} valid={test_trace['whole_frame_valid_fraction']:.4f} "
              f"openings={test_trace['opening_n']} (support={test_trace['opening_support_fractions']}) "
              f"new_opening={finding['new_opening_appeared']} "
              f"new_supported_clearance={finding['new_supported_clearance_appeared']}")

    risky = [f for f in results if f["frame_stayed_valid_overall"] and
             (f["new_opening_appeared"] or f["new_supported_clearance_appeared"])]
    print(f"\n  RISKY co-occurrences (VALID frame + new opening/clearance from noise pocket): {len(risky)}/{len(results)}")
    return results, risky


if __name__ == "__main__":
    neg = recheck_negatives()
    pocket_results, pocket_risky = isolated_noise_pocket()
    with open("benchmarks/i2_depth_validity/results/step6_negative_and_pocket.json", "w") as f:
        json.dump({"negative_recheck": neg, "isolated_pocket": pocket_results,
                    "risky_cooccurrences": pocket_risky}, f, indent=2)
    print("\nWrote benchmarks/i2_depth_validity/results/step6_negative_and_pocket.json")
