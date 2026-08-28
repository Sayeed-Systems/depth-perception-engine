"""
STANDALONE / SENSOR-FACING API — DPE's development-and-qualification
interface.

    from depth_perception_engine.standalone import StandaloneStereoInterface

This subpackage exists ONLY to make `depth_perception_engine` independently
runnable (development, unit tests, benchmarks, datasets, physical stereo/
motion qualification, debugging) by adapting raw/convenient inputs —
calibration file paths, combined side-by-side frames, plain angular-rate
tuples, loose per-frame arguments — into the canonical core input contract
(`models.StereoObservation`), then delegating to the ONE core engine
(`DepthPerceptionPipeline.process_observation()`).

It implements no geometry and duplicates no algorithm. It is NOT part of an
embedded consumer's runtime path: a larger perception system embedding DPE
imports `depth_perception_engine.core` (or `DepthPerceptionPipeline`
directly) and never imports this subpackage at all — structural absence, not
a mode flag. See docs/DUAL_INTERFACE_ARCHITECTURE.md.
"""

from depth_perception_engine.standalone.interface import RawMotionSample, StandaloneStereoInterface

__all__ = ["StandaloneStereoInterface", "RawMotionSample"]
