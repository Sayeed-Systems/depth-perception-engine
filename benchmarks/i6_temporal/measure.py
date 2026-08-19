"""
Phase I6 Part A -- quantitative temporal qualification. Read-only w.r.t.
src/depth_perception_engine/. Real, unmodified DepthPerceptionPipeline only.
"""
import json
import sys
import time

sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.temporal import MotionHint
from depth_perception_engine.temporal.rotation_compensation import (
    integrate_angular_velocity, compensate_prior_geometry, compute_rotation_compensation,
)
from depth_perception_engine.temporal.consistency import compute_temporal_consistency
from depth_perception_engine.temporal.types import TemporalRecord

from benchmarks.i6_temporal.fixtures import (
    static_pair, two_object_pair, flat_pair, decorrelated_pair, textureless_pair, gap_pair, wall_pair,
)

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
_RESULTS = {}


def _transform():
    return RigidTransform(rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
                           from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY)


def _full_cfg(**overrides):
    d = dict(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_surface_geometry=True, surface_grid_rows=3, surface_grid_cols=6,
        enable_boundary_geometry=True, boundary_grid_rows=3, boundary_grid_cols=6,
        enable_opening_geometry=True,
        enable_geometry_frame=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
    )
    d.update(overrides)
    return PipelineConfig(**d)


def _pipeline(cfg):
    return DepthPerceptionPipeline(cfg, _CALIB, rectify=False, body_T_camera_left=_transform())


def _hint(ts, wz=0.0, wy=0.0, wx=0.0, valid=True):
    return MotionHint(timestamp=ts, angular_velocity_rad_s=np.array([wx, wy, wz]), frame_id=FrameId.BODY, valid=valid)


def qf(x):
    return None if x is None else (float(x) if not (isinstance(x, float) and np.isnan(x)) else "NaN")


# ===================================================================
# 1-2. Static scene, static + "noise" (different seed, same physical depth)
# ===================================================================
def scenario_1_2():
    pipeline = _pipeline(_full_cfg())
    left, right = static_pair(depth_m=2.0, seed=1)
    states = []
    agree = []
    for i in range(6):
        r = pipeline.process(left, right, left_timestamp=0.1 * i)
        gf = r.geometry_frame
        states.append(gf.temporal_consistency.state if gf.temporal_consistency else None)
        agree.append(gf.temporal_consistency.agreement_fraction if gf.temporal_consistency else None)
    static_result = {"states": states, "agreement_fraction": agree}

    pipeline2 = _pipeline(_full_cfg())
    states2 = []
    agree2 = []
    for i in range(6):
        left_n, right_n = static_pair(depth_m=2.0, seed=1 + (i % 2))  # alternate seed -> "measurement noise"
        r = pipeline2.process(left_n, right_n, left_timestamp=0.1 * i)
        gf = r.geometry_frame
        states2.append(gf.temporal_consistency.state if gf.temporal_consistency else None)
        agree2.append(gf.temporal_consistency.agreement_fraction if gf.temporal_consistency else None)
    noisy_result = {"states": states2, "agreement_fraction": agree2}

    # static agreement % across all comparable (non-first) frames, both runs
    fracs = [a for a in (agree[1:] + agree2[1:]) if a is not None]
    static_agreement_pct = 100.0 * float(np.mean(fracs)) if fracs else None
    _RESULTS["1_static"] = static_result
    _RESULTS["2_static_noise"] = noisy_result
    _RESULTS["static_agreement_pct"] = static_agreement_pct
    print(f"[1-2] static agreement % = {static_agreement_pct}")


# ===================================================================
# 3-5. Rotation (quantitative A/B/C), zero-translation, changing angular velocity
# ===================================================================
def scenario_3_4_5():
    # Function-level proof (established technique, tests/test_rotation_compensation.py)
    rng = np.random.default_rng(7)
    h, w = 60, 60
    previous_snapshot = rng.uniform(2.0, 4.0, size=(h, w)).astype(np.float32)
    focal_length_px = 400.0
    principal_point_px = (float(w) / 2.0, float(h) / 2.0)
    stride = 1
    omega = np.array([0.0, 0.03, 0.0])
    hint = MotionHint(timestamp=1.0, angular_velocity_rad_s=omega, frame_id=FrameId.BODY)
    delta_r_true = integrate_angular_velocity([hint], previous_timestamp=0.0)
    current_snapshot = compensate_prior_geometry(previous_snapshot, delta_r_true, stride, focal_length_px, principal_point_px)
    previous_record = TemporalRecord(timestamp=0.0, confidence=0.9, depth_snapshot_m=previous_snapshot)
    raw_consistency = compute_temporal_consistency(current_snapshot, previous_record, agreement_tolerance_m=0.05, min_agreement_fraction=0.1)
    compensated_record, status = compute_rotation_compensation(
        previous_record, [hint], previous_timestamp=0.0, current_timestamp=1.0,
        stride=stride, focal_length_px=focal_length_px, principal_point_px=principal_point_px,
    )
    compensated_consistency = compute_temporal_consistency(current_snapshot, compensated_record, agreement_tolerance_m=0.05, min_agreement_fraction=0.1)
    raw_fraction = raw_consistency.agreement_fraction or 0.0
    compensated_fraction = compensated_consistency.agreement_fraction or 0.0
    func_level = {
        "status": status, "B_uncompensated_agreement": raw_fraction, "C_compensated_agreement": compensated_fraction,
        "improvement": compensated_fraction - raw_fraction,
    }
    print(f"[3] function-level: B={raw_fraction:.4f} C={compensated_fraction:.4f} status={status}")

    # Pipeline-level A/B/C on a static (non-rotating in image content) scene
    # with an attached rotation MotionHint -- measures whether ON/OFF
    # compensation changes reported state given actual real motion-hint
    # wiring through the pipeline (the image itself doesn't rotate since we
    # don't have a real rotating-camera capture; this isolates the
    # configuration-level wiring, complementing the function-level proof
    # above which is the actual quantitative rotation-improvement proof).
    left, right = static_pair(depth_m=2.0, seed=3)
    pipeline_results = {}
    for label, cfg in [
        ("A_temporal_off", _full_cfg(enable_temporal=False, enable_temporal_stabilization=False,
                                       enable_rotation_compensation=False, enable_motion_aware_reliability=False,
                                       enable_temporal_persistence=False)),
        ("B_temporal_on_compensation_off", _full_cfg(enable_rotation_compensation=False)),
        ("C_temporal_on_compensation_on", _full_cfg(enable_rotation_compensation=True)),
    ]:
        pipeline = _pipeline(cfg)
        r0 = pipeline.process(left, right, left_timestamp=0.0, motion_hints=[_hint(0.0)])
        r1 = pipeline.process(left, right, left_timestamp=0.1, motion_hints=[_hint(0.1, wy=0.03)])
        gf = r1.geometry_frame
        pipeline_results[label] = {
            "temporal_consistency_state": gf.temporal_consistency.state if gf and gf.temporal_consistency else None,
            "rotation_compensation_status": gf.rotation_compensation_status if gf else None,
            "motion_aware_reliability_state": gf.motion_aware_reliability.state if gf and gf.motion_aware_reliability else None,
        }
    print(f"[3] pipeline-level A/B/C: {pipeline_results}")

    # 5. changing angular velocity across a sequence
    pipeline = _pipeline(_full_cfg())
    omegas = [0.0, 0.01, 0.03, 0.06, 0.09, 0.02]
    changing_states = []
    for i, wz in enumerate(omegas):
        r = pipeline.process(left, right, left_timestamp=0.1 * i, motion_hints=[_hint(0.1 * i, wy=wz)])
        gf = r.geometry_frame
        changing_states.append({
            "omega": wz,
            "reliability": gf.motion_aware_reliability.state if gf and gf.motion_aware_reliability else None,
            "rotation_status": gf.rotation_compensation_status if gf else None,
        })
    print(f"[5] changing angular velocity: {changing_states}")

    _RESULTS["3_4_rotation_function_level"] = func_level
    _RESULTS["3_rotation_pipeline_level_ABC"] = pipeline_results
    _RESULTS["5_changing_angular_velocity"] = changing_states


# ===================================================================
# 6-8. Missing / invalid / NaN-Inf MotionHint
# ===================================================================
def scenario_6_7_8():
    left, right = static_pair(depth_m=2.0, seed=4)
    out = {}
    # 6. missing
    pipeline = _pipeline(_full_cfg())
    pipeline.process(left, right, left_timestamp=0.0)
    r = pipeline.process(left, right, left_timestamp=0.1)  # no motion_hints at all
    gf = r.geometry_frame
    out["6_missing"] = {
        "rotation_status": gf.rotation_compensation_status,
        "reliability_state": gf.motion_aware_reliability.state if gf.motion_aware_reliability else None,
    }
    # 7. invalid (valid=False)
    pipeline = _pipeline(_full_cfg())
    pipeline.process(left, right, left_timestamp=0.0)
    r = pipeline.process(left, right, left_timestamp=0.1, motion_hints=[_hint(0.1, wy=0.03, valid=False)])
    gf = r.geometry_frame
    out["7_invalid"] = {
        "rotation_status": gf.rotation_compensation_status,
        "reliability_state": gf.motion_aware_reliability.state if gf.motion_aware_reliability else None,
    }
    # 8. NaN/Inf
    pipeline = _pipeline(_full_cfg())
    pipeline.process(left, right, left_timestamp=0.0)
    nan_hint = MotionHint(timestamp=0.1, angular_velocity_rad_s=np.array([0.0, np.nan, 0.0]), frame_id=FrameId.BODY, valid=True)
    r = pipeline.process(left, right, left_timestamp=0.1, motion_hints=[nan_hint])
    gf = r.geometry_frame
    body_valid = r.geometry_body.valid_mask if r.geometry_body is not None else None
    body_points_finite = bool(np.all(np.isfinite(r.geometry_body.points[body_valid]))) if body_valid is not None else None
    out["8_nan"] = {
        "rotation_status": gf.rotation_compensation_status,
        "reliability_state": gf.motion_aware_reliability.state if gf.motion_aware_reliability else None,
        "temporal_consistency_state": gf.temporal_consistency.state if gf.temporal_consistency else None,
        "geometry_body_points_all_finite_where_valid": body_points_finite,
        "quality_overall_state": gf.quality.overall_state if gf.quality else None,
    }
    inf_hint = MotionHint(timestamp=0.2, angular_velocity_rad_s=np.array([0.0, np.inf, 0.0]), frame_id=FrameId.BODY, valid=True)
    r2 = pipeline.process(left, right, left_timestamp=0.2, motion_hints=[inf_hint])
    gf2 = r2.geometry_frame
    out["8_inf"] = {
        "rotation_status": gf2.rotation_compensation_status,
        "reliability_state": gf2.motion_aware_reliability.state if gf2.motion_aware_reliability else None,
        "quality_overall_state": gf2.quality.overall_state if gf2.quality else None,
    }
    print(f"[6-8] {out}")
    _RESULTS["6_7_8_motionhint_edge_cases"] = out


# ===================================================================
# 9-11. Timestamp gap, out-of-order, stale history
# ===================================================================
def scenario_9_10_11():
    left, right = static_pair(depth_m=2.0, seed=5)
    out = {}
    cfg = _full_cfg()  # temporal_gap_limit_s=0.5, temporal_max_age_s=1.0 (defaults)

    # 9. gap > temporal_gap_limit_s (0.5s default)
    pipeline = _pipeline(cfg)
    pipeline.process(left, right, left_timestamp=0.0)
    r = pipeline.process(left, right, left_timestamp=5.0)  # 5s gap, way > 0.5s limit
    gf = r.geometry_frame
    out["9_gap"] = {
        "temporal_admission_status": r.temporal_admission_status,
        "temporal_consistency_state": gf.temporal_consistency.state if gf.temporal_consistency else None,
    }

    # 10. out-of-order timestamp
    pipeline = _pipeline(cfg)
    pipeline.process(left, right, left_timestamp=5.0)
    r = pipeline.process(left, right, left_timestamp=2.0)  # earlier than 5.0
    gf = r.geometry_frame
    out["10_out_of_order"] = {
        "temporal_admission_status": r.temporal_admission_status,
        "temporal_consistency_state": gf.temporal_consistency.state if gf.temporal_consistency else None,
    }

    # 11. stale history (spacing beyond temporal_max_age_s=1.0, but under gap_limit... actually gap_limit(0.5)<max_age(1.0) so any admitted gap already exceeds max_age only if >1.0; test a 0.4s gap (under gap_limit) repeated to accumulate age beyond max_age_s via history age filtering, then check state)
    pipeline = _pipeline(cfg)
    ts = 0.0
    last_state = None
    for i in range(4):
        r = pipeline.process(left, right, left_timestamp=ts)
        last_state = r.geometry_frame.temporal_consistency.state if r.geometry_frame.temporal_consistency else None
        ts += 0.4
    out["11_stale_progression_last_state"] = last_state
    out["11_history_len"] = len(pipeline.temporal_history) if pipeline.temporal_history is not None else None

    print(f"[9-11] {out}")
    _RESULTS["9_10_11_timestamp_handling"] = out


# ===================================================================
# 12-14. Sudden geometry change, obstacle appearing/disappearing
# ===================================================================
def scenario_12_13_14():
    pipeline = _pipeline(_full_cfg(persistence_min_support_count=2, persistence_max_dropout_frames=1, persistence_expiration_absence_frames=5))
    left_a, right_a = flat_pair(depth_m=5.0, seed=6)  # far, no obstacle
    left_b, right_b = two_object_pair(near_m=1.5, far_m=5.0, seed=6)  # near obstacle appears

    seq = []
    ts = 0.0
    # baseline: no obstacle for 3 frames
    for i in range(3):
        r = pipeline.process(left_a, right_a, left_timestamp=ts)
        seq.append({"frame": i, "phase": "no_obstacle", "obstacle_pts": r.obstacle_cloud.points.shape[0] if r.obstacle_cloud is not None else None,
                     "persistence_state": r.geometry_frame.temporal_persistence.state if r.geometry_frame.temporal_persistence else None})
        ts += 0.1
    # obstacle appears for 5 frames
    for i in range(5):
        r = pipeline.process(left_b, right_b, left_timestamp=ts)
        seq.append({"frame": 3 + i, "phase": "obstacle_appears", "obstacle_pts": r.obstacle_cloud.points.shape[0] if r.obstacle_cloud is not None else None,
                     "persistence_state": r.geometry_frame.temporal_persistence.state if r.geometry_frame.temporal_persistence else None,
                     "persistent_count": r.geometry_frame.temporal_persistence.persistent_count if r.geometry_frame.temporal_persistence else None})
        ts += 0.1
    # obstacle disappears -- track exact frame persistence expires (persistence_expiration_absence_frames=5)
    disappear_frames = []
    for i in range(8):
        r = pipeline.process(left_a, right_a, left_timestamp=ts)
        tp = r.geometry_frame.temporal_persistence
        disappear_frames.append({
            "frame": 8 + i, "persistent_count": tp.persistent_count if tp else None,
            "disappearing_count": tp.disappearing_count if tp else None, "expired_count": tp.expired_count if tp else None,
            "state": tp.state if tp else None,
        })
        ts += 0.1
    seq.extend(disappear_frames)
    print(f"[12-14] obstacle appear/disappear sequence tail: {disappear_frames}")
    _RESULTS["12_13_14_obstacle_lifecycle"] = seq


# ===================================================================
# 15. Opening appearing/disappearing
# ===================================================================
def scenario_15():
    pipeline = _pipeline(_full_cfg())
    left_wall, right_wall = wall_pair(depth_m=2.0, seed=7)
    left_gap, right_gap, gt = gap_pair(seed=7)
    seq = []
    ts = 0.0
    for i in range(3):
        r = pipeline.process(left_wall, right_wall, left_timestamp=ts)
        n_open = len(r.geometry_frame.opening_evidence or [])
        seq.append({"frame": i, "phase": "wall", "n_openings": n_open})
        ts += 0.1
    for i in range(3):
        r = pipeline.process(left_gap, right_gap, left_timestamp=ts)
        n_open = len(r.geometry_frame.opening_evidence or [])
        seq.append({"frame": 3 + i, "phase": "gap_appears", "n_openings": n_open})
        ts += 0.1
    for i in range(3):
        r = pipeline.process(left_wall, right_wall, left_timestamp=ts)
        n_open = len(r.geometry_frame.opening_evidence or [])
        seq.append({"frame": 6 + i, "phase": "gap_closes", "n_openings": n_open})
        ts += 0.1
    print(f"[15] opening lifecycle: {seq}")
    _RESULTS["15_opening_lifecycle"] = seq


# ===================================================================
# 16. Degradation then recovery
# ===================================================================
def scenario_16():
    pipeline = _pipeline(_full_cfg())
    left_h, right_h = static_pair(depth_m=2.0, seed=8)
    left_bad, right_bad = decorrelated_pair(seed=8)
    ts = 0.0

    def _rec(r, frame, phase):
        q = r.geometry_frame.quality
        return {
            "frame": frame, "phase": phase,
            "overall": q.overall_state if q else None,
            "geometry_validity": q.geometry_validity_state if q else None,
            "temporal_consistency": q.temporal_consistency_state if q else None,
        }

    seq = []
    for i in range(3):
        r = pipeline.process(left_h, right_h, left_timestamp=ts, motion_hints=[_hint(ts)])
        seq.append(_rec(r, i, "healthy"))
        ts += 0.1
    r = pipeline.process(left_bad, right_bad, left_timestamp=ts, motion_hints=[_hint(ts)])
    seq.append(_rec(r, 3, "degraded"))
    ts += 0.1
    recovery_latency_overall = None
    recovery_latency_temporal_consistency = None
    for i in range(6):
        r = pipeline.process(left_h, right_h, left_timestamp=ts, motion_hints=[_hint(ts)])
        rec = _rec(r, 4 + i, "recovering")
        seq.append(rec)
        if rec["overall"] == "VALID" and recovery_latency_overall is None:
            recovery_latency_overall = i + 1
        if rec["temporal_consistency"] == "VALID" and recovery_latency_temporal_consistency is None:
            recovery_latency_temporal_consistency = i + 1
        ts += 0.1
    print(f"[16] degradation/recovery: {seq}  recovery_latency_overall={recovery_latency_overall} "
          f"recovery_latency_temporal_consistency={recovery_latency_temporal_consistency}")
    _RESULTS["16_degradation_recovery"] = {
        "sequence": seq,
        "recovery_latency_overall_frames": recovery_latency_overall,
        "recovery_latency_temporal_consistency_frames": recovery_latency_temporal_consistency,
    }


# ===================================================================
# 17. reset()
# ===================================================================
def scenario_17():
    pipeline = _pipeline(_full_cfg())
    left, right = static_pair(depth_m=2.0, seed=9)
    for i in range(5):
        pipeline.process(left, right, left_timestamp=0.1 * i)
    before_health = pipeline.health()
    before_hist_len = len(pipeline.temporal_history) if pipeline.temporal_history is not None else None
    pipeline.reset()
    after_reset_health = pipeline.health()
    after_reset_hist_len = len(pipeline.temporal_history) if pipeline.temporal_history is not None else None
    r = pipeline.process(left, right, left_timestamp=100.0)
    result = {
        "before_frames_processed": before_health.frames_processed,
        "before_history_len": before_hist_len,
        "after_reset_frames_processed": after_reset_health.frames_processed,
        "after_reset_history_len": after_reset_hist_len,
        "post_reset_first_frame_temporal_consistency_state": r.geometry_frame.temporal_consistency.state if r.geometry_frame.temporal_consistency else None,
        "post_reset_frames_processed": pipeline.health().frames_processed,
    }
    print(f"[17] reset() verification: {result}")
    _RESULTS["17_reset"] = result


# ===================================================================
# 18. Long bounded sequence
# ===================================================================
def scenario_18(n_frames=300):
    pipeline = _pipeline(_full_cfg())
    left, right = static_pair(depth_m=2.0, seed=10)
    hist_lens = []
    latencies = []
    ts = 0.0
    for i in range(n_frames):
        t0 = time.perf_counter()
        pipeline.process(left, right, left_timestamp=ts, motion_hints=[_hint(ts, wy=0.0)])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if i % 25 == 0:
            hist_lens.append({"frame": i, "history_len": len(pipeline.temporal_history) if pipeline.temporal_history is not None else None})
        ts += 0.05
    latencies = np.array(latencies)
    first_half = latencies[:n_frames // 2]
    second_half = latencies[n_frames // 2:]
    result = {
        "n_frames": n_frames,
        "history_len_samples": hist_lens,
        "final_history_len": len(pipeline.temporal_history) if pipeline.temporal_history is not None else None,
        "latency_mean_ms": float(latencies.mean()), "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_max_ms": float(latencies.max()),
        "first_half_mean_ms": float(first_half.mean()), "second_half_mean_ms": float(second_half.mean()),
        "drift_ms": float(second_half.mean() - first_half.mean()),
    }
    print(f"[18] long-run: {result}")
    _RESULTS["18_long_run"] = result


def main():
    scenario_1_2()
    scenario_3_4_5()
    scenario_6_7_8()
    scenario_9_10_11()
    scenario_12_13_14()
    scenario_15()
    scenario_16()
    scenario_17()
    scenario_18()

    path = "/home/sayeed/PycharmProjects/depth_perception_engine/benchmarks/i6_temporal/results/measure.json"
    with open(path, "w") as f:
        json.dump(_RESULTS, f, indent=2, default=str)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
