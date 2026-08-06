"""
Corrupt-frame detection.

A genuine camera frame — however textured, grainy, or noisy-looking —
still has strong positive correlation between horizontally adjacent
pixels, since real scenes are locally smooth on average. Fully
uncorrelated per-pixel noise has near-zero adjacent-pixel correlation
regardless of how "noisy" it looks by eye, which is what actually
distinguishes it from legitimate high-detail texture — a raw
gradient/variance threshold can't tell the two apart, since a
richly-textured real frame can have just as much pixel-to-pixel
variation as garbage.

Confirmed live against two independent, unrelated corruption sources
this heuristic needed to catch: an intermittent Ignition Gazebo
headless/software-rendering artifact, and USB capture corruption from a
real camera under CPU starvation. Real frames from both measured ~0.99
correlation; corrupt frames from both measured ~0.0 — comfortable margin
either side of the default threshold.

Does NOT
---------
- Perform any perception (rectification, disparity, depth).
- Know about ROS messages, topics, or nodes.
"""

import cv2
import numpy as np

_DEFAULT_CORRELATION_THRESHOLD = 0.3


def looks_like_garbage_frame(
    image: np.ndarray, threshold: float = _DEFAULT_CORRELATION_THRESHOLD
) -> bool:
    """
    Return True if *image* looks like uncorrelated noise, not a real frame.

    Subsamples every 4th pixel in each dimension before computing
    correlation — full-resolution precision is unnecessary for this
    and it runs on the hot path.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sample = gray[::4, ::4].astype(np.float64)
    if sample.shape[0] < 2 or sample.shape[1] < 2:
        return False

    a = sample[:, :-1].flatten()
    b = sample[:, 1:].flatten()
    # A flat/blank frame (all-zero std) is legitimate, if useless —
    # not garbage — and corrcoef would otherwise divide by zero.
    if a.std() < 1e-6 or b.std() < 1e-6:
        return False

    correlation = np.corrcoef(a, b)[0, 1]
    return bool(correlation < threshold)
