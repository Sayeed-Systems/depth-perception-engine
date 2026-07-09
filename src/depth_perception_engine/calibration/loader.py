"""
Stereo calibration loader.

Reads an OpenCV FileStorage XML file (the format written by this project's
original stereo_calibration.py tool) into a StereoCalibration object.

This is the ONE place in the library that touches a file path — and it is
never called implicitly. Every pipeline entry point requires a
StereoCalibration object to be constructed and handed in explicitly by the
caller (see pipeline.api / pipeline.pipeline), so the library itself never
assumes or defaults to any particular file on disk.
"""

import os

import cv2

from depth_perception_engine.calibration.models import StereoCalibration


def load_stereo_calibration(path: str) -> StereoCalibration:
    """Load a StereoCalibration from an OpenCV FileStorage XML file.

    Args:
        path: Path to the calibration XML file. Must be supplied by the
              caller — there is no default.

    Returns:
        A validated StereoCalibration.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If the file cannot be opened or a required key is
                      missing.
        ValueError: If a matrix has an unexpected shape (raised by
                    StereoCalibration itself).
    """
    if not path:
        raise ValueError("A calibration file path must be provided.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"cv2.FileStorage could not open: {path}")

    try:
        w = int(fs.getNode("image_width").real())
        h = int(fs.getNode("image_height").real())
        camera_matrix_left = fs.getNode("camera_matrix_left").mat()
        dist_coeffs_left = fs.getNode("dist_coeffs_left").mat()
        camera_matrix_right = fs.getNode("camera_matrix_right").mat()
        dist_coeffs_right = fs.getNode("dist_coeffs_right").mat()
        R1 = fs.getNode("R1").mat()
        R2 = fs.getNode("R2").mat()
        P1 = fs.getNode("P1").mat()
        P2 = fs.getNode("P2").mat()
        Q = fs.getNode("Q").mat()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse calibration file '{path}': {exc}") from exc
    finally:
        fs.release()

    return StereoCalibration(
        image_size=(w, h),
        camera_matrix_left=camera_matrix_left,
        dist_coeffs_left=dist_coeffs_left,
        camera_matrix_right=camera_matrix_right,
        dist_coeffs_right=dist_coeffs_right,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    )
