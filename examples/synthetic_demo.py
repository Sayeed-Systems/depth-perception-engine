"""
Synthetic demo — no camera, no GUI, no hardware required.

Demonstrates the exact call shape mp01_perception's perception_processor.py
will use later (see ../docs/INTEGRATION_READINESS.md): build a
DepthPerceptionPipeline once from a loaded calibration, then call
.process(left, right) per frame with plain NumPy arrays.

Run:
    pip install -e ..
    python examples/synthetic_demo.py
"""

import os

import numpy as np

from depth_perception_engine import (
    DepthPerceptionPipeline,
    PipelineConfig,
    load_stereo_calibration,
)

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_CALIBRATION_FILE = os.path.join(_EXAMPLES_DIR, "config", "stereo_calibration.xml")


def _synthetic_stereo_pair(width: int, height: int, seed: int) -> tuple:
    """Random BGR image pair — stands in for a real rectified camera frame.

    Random noise has essentially no stereo correspondence, so SGBM will
    mostly report invalid disparity; that's fine here — this demo exists to
    prove the call shape and output structure work, not to produce a
    meaningful depth reading. See tests/ for the same idea used as
    assertions instead of a printout.
    """
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return left, right


def main() -> None:
    calibration = load_stereo_calibration(_CALIBRATION_FILE)
    config = PipelineConfig()

    # Built once — holds obstacle-detection temporal smoothing across frames,
    # exactly as mp01_perception's perception_processor.py should hold one
    # instance across ROS callbacks.
    pipeline = DepthPerceptionPipeline(config, calibration)
    print(f"Pipeline ready: {pipeline}")

    width, height = calibration.image_size
    for frame_idx in range(3):
        left, right = _synthetic_stereo_pair(width, height, seed=frame_idx)
        result = pipeline.process(left, right)

        print(f"\n--- frame {frame_idx} ---")
        print(f"disparity_map:       shape={result.disparity_map.shape} dtype={result.disparity_map.dtype}")
        print(f"depth_map:           shape={result.depth_map.shape} dtype={result.depth_map.dtype}")
        print(f"traversability:      decision={result.traversability_mask.decision.value}, "
              f"regions={list(result.traversability_mask.regions.keys())}")
        print(f"obstacles:           {len(result.obstacles.beams)} beams, "
              f"safest={result.obstacles.safest_beam}")
        print(f"confidence:          {result.confidence:.3f}")
        print(f"processing_time_ms:  {result.processing_time_ms:.2f}")


if __name__ == "__main__":
    main()
