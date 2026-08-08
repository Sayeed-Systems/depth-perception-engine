"""
Regenerates docs/assets/09_level3_healthy_scene.png from a fresh, real
live capture — with a visible countdown so the operator can position the
camera/scene before the shot is taken.

Reuses visualize_level3.py's own capture/transform/figure-building
pieces unmodified (imported, not duplicated) — this script only adds the
countdown and targets the fixed README asset path.

Run:
    python examples/capture_readme_snapshot.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_level3 import _capture_live_pair, _illustrative_transform, build_figure  # noqa: E402

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration

_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "stereo_calibration.xml")
_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs", "assets", "09_level3_healthy_scene.png",
)
_COUNTDOWN_S = 6


def main() -> None:
    calibration = load_stereo_calibration(_CALIBRATION_PATH)
    config = PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)
    transform = _illustrative_transform()
    pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)

    print("Position the camera/scene now.")
    for remaining in range(_COUNTDOWN_S, 0, -1):
        print(f"  capturing in {remaining}...", flush=True)
        time.sleep(1)
    print("  CAPTURING NOW")

    left, right = _capture_live_pair(calibration)
    result = pipeline.process(left, right)
    build_figure(left, result, transform, _OUTPUT_PATH)


if __name__ == "__main__":
    main()
