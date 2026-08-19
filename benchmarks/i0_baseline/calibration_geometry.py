"""
Shared calibration-derived geometry constants for the I0 benchmark
freeze — Phase I0 (see docs/DPE_V1_PROVIDER_CONTRACT.md's D10 record for
the precedent this reuses).

The ONLY source of ground truth for every scenario in scenarios.py is
this repository's own checked-in calibration config
(examples/config/stereo_calibration.xml) plus PipelineConfig's own
documented defaults (config/pipeline_config.py) — no scenario invents an
intrinsic, baseline, or threshold value independently of those two
sources. This repo has no SDF/URDF (it is a pure Python library with
zero Gazebo/ROS dependency — see docs/VALIDATION_REPORT.md); the
calibration XML + PipelineConfig defaults are this repo's equivalent
"config geometry" ground-truth source.

fx/cx/baseline_m below are derived from the calibration's own Q matrix
using the standard OpenCV reprojection convention:

    Q = [[1, 0, 0,   -cx],
         [0, 1, 0,   -cy],
         [0, 0, 0,    fx],
         [0, 0, 1/Tx,  0]]

(Tx in millimetres in this calibration file's convention — confirmed by
cross-checking against tests/test_d10_integrated_ground_truth.py's own
independently hand-derived constants, which this module's values match
exactly.) This is a read-only derivation; it does not modify or
reinterpret StereoCalibration, PointCloudBuilder, or any other DPE code.
"""

from depth_perception_engine import load_stereo_calibration
from depth_perception_engine.calibration.models import StereoCalibration

CALIBRATION_PATH = "examples/config/stereo_calibration.xml"


def load_calibration() -> StereoCalibration:
    return load_stereo_calibration(CALIBRATION_PATH)


def derive_geometry(calibration: StereoCalibration) -> dict:
    """Return {fx_px, cx_px, cy_px, baseline_m, width, height} derived
    purely from calibration.Q — the same convention this repo's own D10
    ground-truth tests already use."""
    q = calibration.Q
    fx_px = float(q[2, 3])
    cx_px = float(-q[0, 3])
    cy_px = float(-q[1, 3])
    baseline_m = float((1.0 / q[3, 2]) / 1000.0)
    width, height = calibration.image_size
    return {
        "fx_px": fx_px,
        "cx_px": cx_px,
        "cy_px": cy_px,
        "baseline_m": baseline_m,
        "width": int(width),
        "height": int(height),
    }
