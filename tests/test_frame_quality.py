"""Unit tests for quality.frame_quality."""

import numpy as np

from depth_perception_engine.quality.frame_quality import looks_like_garbage_frame


class TestLooksLikeGarbageFrame:
    def test_smooth_gradient_is_not_garbage(self):
        row = np.linspace(0, 255, 320, dtype=np.uint8)
        image = np.tile(row, (240, 1)).astype(np.uint8)
        image = np.stack([image] * 3, axis=-1)

        assert looks_like_garbage_frame(image) is False

    def test_uncorrelated_random_noise_is_garbage(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)

        assert looks_like_garbage_frame(image) is True

    def test_flat_image_is_not_garbage(self):
        image = np.full((240, 320, 3), 128, dtype=np.uint8)

        assert looks_like_garbage_frame(image) is False

    def test_grayscale_input_is_accepted(self):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, (240, 320), dtype=np.uint8)

        assert looks_like_garbage_frame(image) is True

    def test_tiny_image_is_not_garbage(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)

        assert looks_like_garbage_frame(image) is False

    def test_threshold_is_configurable(self):
        rng = np.random.default_rng(2)
        image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)

        assert looks_like_garbage_frame(image, threshold=-1.0) is False
