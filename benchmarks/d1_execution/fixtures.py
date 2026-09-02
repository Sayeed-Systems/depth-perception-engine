"""
D1 fixtures — calibration, the qualified configuration, and stereo scenes.

Reuses this repository's OWN already-reviewed fixtures rather than
inventing new ones:

  * calibration: examples/config/stereo_calibration.xml (the same real
    hardware calibration tests/conftest.py and every I-phase benchmark
    load).
  * qualified configuration: the "full V1 candidate" PipelineConfig frozen
    by benchmarks/i0_baseline/scenarios.py::latency_scenario, which is the
    configuration DPE's own qualification benchmarks were run under.
  * stereo scenes: benchmarks/i1_stereo_accuracy/fixtures.py's
    low-frequency-canvas + disparity-remap technique (via
    benchmarks/i6_temporal/fixtures.py), which produces a genuinely
    disparity-consistent stereo pair rather than i.i.d. noise.
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from depth_perception_engine import PipelineConfig  # noqa: E402
from depth_perception_engine.calibration import load_stereo_calibration  # noqa: E402
from depth_perception_engine.calibration.models import StereoCalibration  # noqa: E402
from depth_perception_engine.frames import FrameId, RigidTransform  # noqa: E402
from depth_perception_engine.temporal.types import MotionHint  # noqa: E402

CALIBRATION_PATH = os.path.join(_REPO_ROOT, "examples", "config", "stereo_calibration.xml")


def calibration() -> StereoCalibration:
    return load_stereo_calibration(CALIBRATION_PATH)


def body_transform() -> RigidTransform:
    """The same camera->body extrinsic benchmarks/i0_baseline/measure.py and
    benchmarks/i6_temporal/measure.py already use."""
    return RigidTransform(
        rotation=np.eye(3),
        translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT,
        to_frame=FrameId.BODY,
    )


def qualified_config(**overrides) -> PipelineConfig:
    """The normal qualified configuration — verbatim the full V1 candidate
    config frozen in benchmarks/i0_baseline/scenarios.py::latency_scenario."""
    d = dict(
        enable_geometry=True,
        enable_obstacle_geometry=True,
        enable_free_space_rays=True,
        enable_surface_geometry=True,
        enable_boundary_geometry=True,
        enable_opening_geometry=True,
        enable_temporal=True,
        enable_temporal_stabilization=True,
        enable_rotation_compensation=True,
        enable_motion_aware_reliability=True,
        enable_temporal_persistence=True,
        enable_geometry_frame=True,
        temporal_gap_limit_s=5.0,
        temporal_max_age_s=100.0,
        temporal_max_records=30,
    )
    d.update(overrides)
    return PipelineConfig(**d)


def non_temporal_config(**overrides) -> PipelineConfig:
    """The qualified configuration with every temporal capability disabled
    through supported configuration flags only — no code path is replaced,
    and the authoritative process_geometry_frame() path is still used."""
    return qualified_config(
        enable_temporal=False,
        enable_temporal_stabilization=False,
        enable_rotation_compensation=False,
        enable_motion_aware_reliability=False,
        enable_temporal_persistence=False,
        **overrides,
    )


# ---------------------------------------------------------------------
# Stereo scenes
# ---------------------------------------------------------------------
def scene_pair(depth_m: float = 2.0, seed: int = 1):
    """A disparity-consistent textured stereo pair at the qualified
    resolution (320x240), from I1/I6's own fixture technique."""
    from benchmarks.i6_temporal.fixtures import static_pair
    return static_pair(depth_m=depth_m, seed=seed)


def two_object_scene(near_m: float = 1.0, far_m: float = 4.0, seed: int = 1):
    from benchmarks.i6_temporal.fixtures import two_object_pair
    return two_object_pair(near_m=near_m, far_m=far_m, seed=seed)


def scene_sequence(n: int, depth_m: float = 2.0, seed0: int = 1):
    """n distinct but physically similar scenes — a different texture seed
    per frame, same plane depth, so temporal stages see a realistic
    frame-to-frame relationship rather than a byte-identical repeat."""
    return [scene_pair(depth_m=depth_m, seed=seed0 + i) for i in range(n)]


def motion_hint(ts: float, wz: float = 0.0, wy: float = 0.0, wx: float = 0.0, valid: bool = True) -> MotionHint:
    return MotionHint(
        timestamp=ts,
        angular_velocity_rad_s=np.array([wx, wy, wz], dtype=np.float64),
        frame_id=FrameId.BODY,
        valid=valid,
    )


def motion_hint_window(t_prev: float, t_now: float, wz: float = 0.05, n: int = 5):
    """A representative bounded motion window spanning [t_prev, t_now] —
    the shape DepthPerceptionPipeline's rotation-compensation stage
    actually consumes (StereoObservation.motion_hints)."""
    return [motion_hint(t_prev + (t_now - t_prev) * i / (n - 1), wz=wz) for i in range(n)]


# ---------------------------------------------------------------------
# Derived higher-resolution calibration (clearly labelled, NOT hardware)
# ---------------------------------------------------------------------
def scaled_calibration(base: StereoCalibration, factor: int) -> StereoCalibration:
    """Exactly-derived calibration for the SAME physical rig sampled at
    `factor`x the qualified resolution.

    This is not a fabricated calibration: for an ideal pinhole camera,
    sampling the same sensor at k times the pixel density scales
    fx/fy/cx/cy by exactly k and leaves the normalized distortion
    coefficients and the physical baseline unchanged. Under the standard
    Q form produced by cv2.stereoRectify:

        Q = [[1, 0, 0,   -cx],
             [0, 1, 0,   -cy],
             [0, 0, 0,     f],
             [0, 0, -1/Tx, 0]]

    only -cx/-cy/f are pixel quantities, so they scale by k; Q[3,2] =
    -1/Tx is a pure function of the physical baseline and is left EXACTLY
    unchanged (which is what keeps metric depth identical rather than
    silently rescaling the world). Reported separately from the qualified
    320x240 result and never presented as a qualified configuration.
    """
    k = float(factor)
    w, h = base.image_size

    def _scale_k(m):
        out = m.astype(np.float64).copy()
        out[0, 0] *= k
        out[1, 1] *= k
        out[0, 2] *= k
        out[1, 2] *= k
        return out

    def _scale_p(p):
        out = p.astype(np.float64).copy()
        out[0, 0] *= k
        out[1, 1] *= k
        out[0, 2] *= k
        out[1, 2] *= k
        out[0, 3] *= k  # Tx*f: f scaled, Tx unchanged
        return out

    Q = base.Q.astype(np.float64).copy()
    Q[0, 3] *= k
    Q[1, 3] *= k
    Q[2, 3] *= k
    # Q[3, 2] = -1/Tx deliberately UNCHANGED — the physical baseline did
    # not change, and neither must metric depth.

    return StereoCalibration(
        image_size=(int(w * factor), int(h * factor)),
        camera_matrix_left=_scale_k(base.camera_matrix_left),
        dist_coeffs_left=base.dist_coeffs_left.copy(),
        camera_matrix_right=_scale_k(base.camera_matrix_right),
        dist_coeffs_right=base.dist_coeffs_right.copy(),
        R1=base.R1.copy(),
        R2=base.R2.copy(),
        P1=_scale_p(base.P1),
        P2=_scale_p(base.P2),
        Q=Q,
    )


def upscale_pair(left, right, factor: int):
    import cv2
    h, w = left.shape[:2]
    size = (w * factor, h * factor)
    return (
        cv2.resize(left, size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(right, size, interpolation=cv2.INTER_LINEAR),
    )
