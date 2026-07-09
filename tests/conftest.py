"""Shared pytest fixtures — synthetic stereo images and a real calibration fixture."""

import os

import numpy as np
import pytest

from depth_perception_engine.calibration import load_stereo_calibration
from depth_perception_engine.config import PipelineConfig

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CALIBRATION_FILE = os.path.join(_REPO_ROOT, "examples", "config", "stereo_calibration.xml")


@pytest.fixture(scope="session")
def calibration():
    """A real StereoCalibration loaded from the project's example fixture file.

    Using a real calibration (rather than fabricated matrices) means
    cv2.initUndistortRectifyMap/reprojectImageTo3D are exercised with
    genuinely valid data, the same way the original hardware demo did.
    """
    return load_stereo_calibration(_CALIBRATION_FILE)


@pytest.fixture
def config():
    return PipelineConfig()


@pytest.fixture
def stereo_pair(calibration):
    """A synthetic (left, right) BGR image pair matching the calibration's image size."""
    width, height = calibration.image_size
    rng = np.random.default_rng(42)
    left = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return left, right
