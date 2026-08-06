"""Frame-level image quality checks, upstream of any stereo/depth processing."""

from depth_perception_engine.quality.frame_quality import looks_like_garbage_frame

__all__ = ["looks_like_garbage_frame"]
