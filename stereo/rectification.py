"""
Stereo rectification module.

Responsibilities: load stereo calibration parameters, initialise undistort-
rectify maps for both cameras, and remap raw stereo image pairs to a common
rectified plane.

Does NOT perform camera acquisition, frame splitting, disparity computation,
depth estimation, distance measurement, object detection, visualisation
dashboard logic, or neural network inference.
"""

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default calibration file written by stereo_calibration.py
_THIS_DIR: str = os.path.dirname(os.path.abspath(str(__file__)))
_DEFAULT_CALIBRATION_FILE: str = os.path.join(
    os.path.dirname(_THIS_DIR), "config", "stereo_calibration.xml"
)


class RectificationEngine:
    """Loads stereo calibration data and rectifies stereo image pairs."""

    def __init__(self, calibration_file: str = _DEFAULT_CALIBRATION_FILE) -> None:
        """
        Args:
            calibration_file: Path to an OpenCV FileStorage XML file produced
                              by :class:`StereoCalibration`.
        """
        self._calibration_file: str = calibration_file

        # Calibration intrinsics / extrinsics
        self._camera_matrix_left: Optional[np.ndarray] = None
        self._dist_coeffs_left: Optional[np.ndarray] = None
        self._camera_matrix_right: Optional[np.ndarray] = None
        self._dist_coeffs_right: Optional[np.ndarray] = None
        self._R1: Optional[np.ndarray] = None
        self._R2: Optional[np.ndarray] = None
        self._P1: Optional[np.ndarray] = None
        self._P2: Optional[np.ndarray] = None
        self._Q: Optional[np.ndarray] = None

        # Expected image size from calibration (width, height)
        self._image_size: Optional[Tuple[int, int]] = None

        # Rectification maps (CV_16SC2 for remap speed)
        self._map1_left: Optional[np.ndarray] = None
        self._map2_left: Optional[np.ndarray] = None
        self._map1_right: Optional[np.ndarray] = None
        self._map2_right: Optional[np.ndarray] = None

        logger.info(
            "RectificationEngine created — calibration file: %s",
            calibration_file,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_calibration(self, path: Optional[str] = None) -> None:
        """Load stereo calibration parameters from an OpenCV FileStorage file.

        Reads all matrices required to build the rectification maps:
        camera matrices, distortion coefficients, rectification rotations
        (R1, R2), projection matrices (P1, P2), and the Q matrix.

        Args:
            path: Override the path set at construction time.  If ``None``
                  the constructor value is used.

        Raises:
            FileNotFoundError: If the calibration file does not exist.
            RuntimeError: If any required calibration key is missing or the
                          file cannot be parsed.
        """
        target: str = str(path or self._calibration_file)
        logger.info("Loading stereo calibration from: %s", target)

        if not os.path.isfile(target):
            raise FileNotFoundError(
                f"Calibration file not found: {target}\n"
                "Run stereo_calibration.py first to generate it."
            )

        fs = cv2.FileStorage(target, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"cv2.FileStorage could not open: {target}")

        try:
            w = int(fs.getNode("image_width").real())
            h = int(fs.getNode("image_height").real())
            self._image_size = (w, h)
            self._camera_matrix_left = fs.getNode("camera_matrix_left").mat()
            self._dist_coeffs_left = fs.getNode("dist_coeffs_left").mat()
            self._camera_matrix_right = fs.getNode("camera_matrix_right").mat()
            self._dist_coeffs_right = fs.getNode("dist_coeffs_right").mat()
            self._R1 = fs.getNode("R1").mat()
            self._R2 = fs.getNode("R2").mat()
            self._P1 = fs.getNode("P1").mat()
            self._P2 = fs.getNode("P2").mat()
            self._Q = fs.getNode("Q").mat()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse calibration file '{target}': {exc}"
            ) from exc
        finally:
            fs.release()

        self.validate_calibration()
        # validate_calibration() raises if image_size is None, so it is safe here.
        w_log, h_log = self._image_size  # type: ignore[misc]
        logger.info(
            "Calibration loaded successfully — image size: %dx%d", w_log, h_log
        )

    def validate_calibration(self) -> None:
        """Verify that all required calibration matrices are present and valid.

        Raises:
            RuntimeError: If any required matrix is missing, empty, or has an
                          unexpected shape.
        """
        required: dict = {
            "camera_matrix_left": (self._camera_matrix_left, (3, 3)),
            "camera_matrix_right": (self._camera_matrix_right, (3, 3)),
            "dist_coeffs_left": (self._dist_coeffs_left, None),
            "dist_coeffs_right": (self._dist_coeffs_right, None),
            "R1": (self._R1, (3, 3)),
            "R2": (self._R2, (3, 3)),
            "P1": (self._P1, (3, 4)),
            "P2": (self._P2, (3, 4)),
            "Q": (self._Q, (4, 4)),
        }

        for name, (mat, expected_shape) in required.items():
            if mat is None or (hasattr(mat, "size") and mat.size == 0):
                raise RuntimeError(
                    f"Calibration validation failed: '{name}' is missing or empty."
                )
            if expected_shape is not None and mat.shape != expected_shape:
                raise RuntimeError(
                    f"Calibration validation failed: '{name}' has shape "
                    f"{mat.shape}, expected {expected_shape}."
                )

        if self._image_size is None or self._image_size[0] <= 0 or self._image_size[1] <= 0:
            raise RuntimeError(
                "Calibration validation failed: image_size is missing or invalid."
            )

        logger.debug("Calibration validated successfully.")

    def initialize_rectification(self) -> None:
        """Build the undistort-rectify maps for both cameras.

        Must be called after :meth:`load_calibration`.  Computes two pairs of
        maps (one per camera) using :func:`cv2.initUndistortRectifyMap` with
        ``CV_16SC2`` precision, which is optimised for use with
        :func:`cv2.remap`.

        Raises:
            RuntimeError: If calibration data has not been loaded or validated.
        """
        if not self._calibration_loaded():
            raise RuntimeError(
                "Calibration not loaded. Call load_calibration() first."
            )

        # Narrow types: _calibration_loaded() guarantees all of these are set.
        assert self._camera_matrix_left is not None
        assert self._dist_coeffs_left is not None
        assert self._camera_matrix_right is not None
        assert self._dist_coeffs_right is not None
        assert self._R1 is not None
        assert self._R2 is not None
        assert self._P1 is not None
        assert self._P2 is not None
        assert self._image_size is not None

        logger.info("Initializing rectification maps...")

        self._map1_left, self._map2_left = cv2.initUndistortRectifyMap(
            self._camera_matrix_left,
            self._dist_coeffs_left,
            self._R1,
            self._P1,
            self._image_size,
            cv2.CV_16SC2,
        )

        self._map1_right, self._map2_right = cv2.initUndistortRectifyMap(
            self._camera_matrix_right,
            self._dist_coeffs_right,
            self._R2,
            self._P2,
            self._image_size,
            cv2.CV_16SC2,
        )

        logger.info(
            "Rectification maps created — size: %dx%d",
            self._image_size[0],
            self._image_size[1],
        )

    def rectify(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify a stereo image pair using pre-computed maps.

        Applies :func:`cv2.remap` with bilinear interpolation to both frames.

        Args:
            left_frame: Raw left camera image (BGR or grayscale ndarray).
            right_frame: Raw right camera image (BGR or grayscale ndarray).

        Returns:
            ``(left_rectified, right_rectified)`` — rectified images with the
            same dtype and number of channels as the inputs.

        Raises:
            RuntimeError: If rectification maps have not been initialised.
            ValueError: If either frame is ``None``, not a numpy ndarray,
                        has fewer than 2 dimensions, or does not match the
                        calibrated image size.
        """
        if not self.is_initialized():
            raise RuntimeError(
                "Rectification maps not initialised. "
                "Call initialize_rectification() first."
            )

        self._validate_frame(left_frame, "left")
        self._validate_frame(right_frame, "right")

        # Narrow: is_initialized() guarantees maps are set.
        assert self._map1_left is not None
        assert self._map2_left is not None
        assert self._map1_right is not None
        assert self._map2_right is not None

        left_rectified: np.ndarray = cv2.remap(
            left_frame,
            self._map1_left,
            self._map2_left,
            cv2.INTER_LINEAR,
        )
        right_rectified: np.ndarray = cv2.remap(
            right_frame,
            self._map1_right,
            self._map2_right,
            cv2.INTER_LINEAR,
        )

        logger.debug("Stereo images rectified.")
        return left_rectified, right_rectified

    def is_initialized(self) -> bool:
        """Return ``True`` if the rectification maps are ready for use."""
        return all(
            m is not None
            for m in (
                self._map1_left,
                self._map2_left,
                self._map1_right,
                self._map2_right,
            )
        )

    # ------------------------------------------------------------------
    # Properties — read-only access for downstream modules
    # ------------------------------------------------------------------

    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        """``(width, height)`` from the calibration file, or ``None``."""
        return self._image_size

    @property
    def Q(self) -> Optional[np.ndarray]:
        """Disparity-to-depth mapping matrix (4×4), or ``None``."""
        return self._Q

    @property
    def P1(self) -> Optional[np.ndarray]:
        """Left rectified projection matrix (3×4), or ``None``."""
        return self._P1

    @property
    def P2(self) -> Optional[np.ndarray]:
        """Right rectified projection matrix (3×4), or ``None``."""
        return self._P2

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calibration_loaded(self) -> bool:
        """Return ``True`` if all core calibration matrices are present."""
        return (
            self._image_size is not None
            and all(
                m is not None
                for m in (
                    self._camera_matrix_left,
                    self._dist_coeffs_left,
                    self._camera_matrix_right,
                    self._dist_coeffs_right,
                    self._R1,
                    self._R2,
                    self._P1,
                    self._P2,
                    self._Q,
                )
            )
        )

    def _validate_frame(self, frame: np.ndarray, side: str) -> None:
        """Raise if *frame* is unsuitable for remapping."""
        if frame is None:
            raise ValueError(f"{side} frame is None.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(
                f"{side} frame must be a numpy.ndarray, "
                f"got {type(frame).__name__}."
            )
        if frame.ndim < 2:
            raise ValueError(
                f"{side} frame has fewer than 2 dimensions ({frame.ndim})."
            )
        if self._image_size is not None:
            exp_w, exp_h = self._image_size
            h, w = frame.shape[:2]
            if (w, h) != (exp_w, exp_h):
                raise ValueError(
                    f"{side} frame size ({w}×{h}) does not match calibrated "
                    f"size ({exp_w}×{exp_h})."
                )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RectificationEngine("
            f"calibration_file={self._calibration_file!r}, "
            f"initialized={self.is_initialized()})"
        )


# ---------------------------------------------------------------------------
# Manual test — live rectification preview with horizontal guide lines
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from camera_stream import StereoCameraStream
    from frame_splitter import FrameSplitter

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    engine = RectificationEngine()

    try:
        engine.load_calibration()
        engine.initialize_rectification()
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Rectification setup failed: %s", exc)
        raise SystemExit(1)

    cam = StereoCameraStream(camera_index=0, width=1280, height=480)
    splitter = FrameSplitter()

    if not cam.open():
        logger.error("Could not open camera. Exiting.")
        raise SystemExit(1)

    logger.info("Press 'q' to quit.")

    # Guide-line appearance (spacing in pixels, BGR colour)
    guide_step = 30
    guide_color = (0, 255, 0)

    def draw_guide_lines(img: np.ndarray) -> np.ndarray:
        """Return a copy of *img* with evenly-spaced horizontal guide lines."""
        out = img.copy()
        for y in range(guide_step, out.shape[0], guide_step):
            cv2.line(out, (0, y), (out.shape[1], y), guide_color, 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    frame_count = 0
    fps = 0.0
    t_start = time.perf_counter()

    try:
        while True:
            ok, raw_frame = cam.read_frame()
            if not ok or raw_frame is None:
                logger.warning("Empty frame — skipping.")
                continue

            try:
                left_raw, right_raw = splitter.split(raw_frame)
                left_rect, right_rect = engine.rectify(left_raw, right_raw)
            except (ValueError, RuntimeError) as exc:
                logger.error("Processing error: %s", exc)
                continue

            # FPS calculation
            frame_count += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.perf_counter()

            # Draw guide lines on all four images
            left_raw_disp = draw_guide_lines(left_raw)
            right_raw_disp = draw_guide_lines(right_raw)
            left_rect_disp = draw_guide_lines(left_rect)
            right_rect_disp = draw_guide_lines(right_rect)

            # FPS overlay on raw-left window only
            cv2.putText(
                left_raw_disp,
                f"FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Raw Left", left_raw_disp)
            cv2.imshow("Raw Right", right_raw_disp)
            cv2.imshow("Rectified Left", left_rect_disp)
            cv2.imshow("Rectified Right", right_rect_disp)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key received.")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed.")
