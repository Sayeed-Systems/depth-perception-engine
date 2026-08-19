import sys
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine/src")
sys.path.insert(0, "/home/sayeed/PycharmProjects/depth_perception_engine")

import numpy as np
from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from benchmarks.i1_stereo_accuracy.fixtures import make_decorrelated_fixture, W as _W, H as _H
from benchmarks.i4_boundary_precision.fixtures import make_narrow_obstacle_fixture
from benchmarks.i5_surface_opening_clearance.clearance.fixtures import make_multi_zone_fixture

_CALIB = load_stereo_calibration("examples/config/stereo_calibration.xml")
N_BEAMS = 20
MIN_VALID = 5
PERCENTILE = 15
GAP_TOL = 0.02
MIN_SUPP = 12
DEAD_ZONE_PX = 128


def _transform():
    return RigidTransform(rotation=np.eye(3), translation=np.array([0.05,0.0,0.02]),
                           from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY)


def _pipeline():
    cfg = PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
                          enable_surface_geometry=True, enable_boundary_geometry=True,
                          enable_opening_geometry=True, enable_geometry_frame=True)
    return DepthPerceptionPipeline(cfg, _CALIB, rectify=False, body_T_camera_left=_transform())


def current_algorithm(col):
    valid = col[(col > 0) & np.isfinite(col)]
    if valid.size < MIN_VALID:
        return 0.0
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        valid = valid[(valid >= q1 - 1.5*iqr) & (valid <= q3 + 1.5*iqr)]
    return float(np.percentile(valid, PERCENTILE)) if valid.size >= MIN_VALID else 0.0


def candidate_algorithm(col):
    valid = col[(col > 0) & np.isfinite(col)]
    if valid.size < MIN_VALID:
        return 0.0
    s = np.sort(valid)
    gaps = np.diff(s)
    split_idx = np.where(gaps > GAP_TOL)[0] + 1
    for c in np.split(s, split_idx):
        if c.size >= MIN_SUPP:
            return float(np.median(c))
    return 0.0


def scan_beams(depth_map):
    h, w = depth_map.shape[:2]
    beam_w = w / N_BEAMS
    rows = []
    for i in range(N_BEAMS):
        x1, x2 = int(i*beam_w), int((i+1)*beam_w)
        col = depth_map[:, x1:x2].flatten().astype(np.float32)
        rows.append((i, x1, x2, current_algorithm(col), candidate_algorithm(col)))
    return rows


def true_range(x1, x2, zones):
    best = None
    for c0f, c1f, depth_m in zones:
        c0, c1 = int(c0f*_W), int(c1f*_W)
        ov0, ov1 = max(c0, x1, DEAD_ZONE_PX), min(c1, x2)
        if ov1 > ov0:
            best = depth_m if best is None else min(best, depth_m)
    return best


pipeline = _pipeline()

print("=== multi_zone (real pipeline) ===")
zones = [(0.0, 0.33, 1.5), (0.33, 0.66, 4.0), (0.66, 1.0, 2.5)]
for seed in range(1, 4):
    fx = make_multi_zone_fixture(zones, seed=seed)
    result = pipeline.process(fx.left, fx.right)
    for i, x1, x2, cur, cand in scan_beams(result.depth_map):
        tr = true_range(x1, x2, zones)
        if tr is None:
            continue
        cur_err = 100.0*abs(cur-tr)/tr if cur > 0 else None
        cand_err = 100.0*abs(cand-tr)/tr if cand > 0 else None
        flag = "  <-- improved" if (cur_err is not None and cur_err > 5 and cand_err is not None and cand_err <= 5) else ""
        print(f"seed={seed} beam={i} x=[{x1},{x2}] true={tr:.3f} cur={cur:.3f}({cur_err}) cand={cand:.3f}({cand_err}){flag}")

print()
print("=== narrow_obstacle (real pipeline) ===")
for seed in range(1, 4):
    fx = make_narrow_obstacle_fixture(near_m=2.0, far_m=5.0, seed=seed)
    result = pipeline.process(fx.left, fx.right)
    zones2 = [(0.0, 0.4, 5.0), (0.4, 0.6, 2.0), (0.6, 1.0, 5.0)]
    for i, x1, x2, cur, cand in scan_beams(result.depth_map):
        tr = true_range(x1, x2, zones2)
        if tr is None:
            continue
        cur_err = 100.0*abs(cur-tr)/tr if cur > 0 else None
        cand_err = 100.0*abs(cand-tr)/tr if cand > 0 else None
        flag = "  <-- improved" if (cur_err is not None and cur_err > 5 and cand_err is not None and cand_err <= 5) else ""
        print(f"seed={seed} beam={i} x=[{x1},{x2}] true={tr:.3f} cur={cur:.3f}({cur_err}) cand={cand:.3f}({cand_err}){flag}")

print()
print("=== pure_noise (real pipeline, 30 seeds) — count of beams reporting nonzero distance ===")
cur_nonzero = 0
cand_nonzero = 0
total_beams = 0
for seed in range(1, 31):
    fx = make_decorrelated_fixture(seed)
    result = pipeline.process(fx.left, fx.right)
    for i, x1, x2, cur, cand in scan_beams(result.depth_map):
        total_beams += 1
        if cur > 0:
            cur_nonzero += 1
        if cand > 0:
            cand_nonzero += 1
print(f"total_beams={total_beams} current_nonzero={cur_nonzero} candidate_nonzero={cand_nonzero}")
