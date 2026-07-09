"""
Shared input validation for the pipeline's public entry points.

Kept minimal and separate from the algorithm modules (stereo/, depth/,
traversability/, obstacles/), which already validate their own specific
inputs — this only covers the generic "is this even a usable image pair"
check shared by every top-level pipeline.api function.
"""

import numpy as np


def require_matching_stereo_pair(left_image: np.ndarray, right_image: np.ndarray) -> None:
    """Raise if *left_image*/*right_image* aren't a usable stereo pair.

    Raises:
        TypeError: If either image is not a numpy ndarray.
        ValueError: If either image is None, has fewer than 2 dimensions,
                    or the two images have mismatched (height, width).
    """
    for name, image in (("left_image", left_image), ("right_image", right_image)):
        if image is None:
            raise ValueError(f"{name} is None.")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray, got {type(image).__name__}.")
        if image.ndim < 2:
            raise ValueError(f"{name} must have at least 2 dimensions, got {image.ndim}.")

    if left_image.shape[:2] != right_image.shape[:2]:
        raise ValueError(
            f"left_image shape {left_image.shape[:2]} does not match "
            f"right_image shape {right_image.shape[:2]}."
        )
