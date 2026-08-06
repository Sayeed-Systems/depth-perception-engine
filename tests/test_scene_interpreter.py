"""
Regression coverage for SceneInterpreter's handling of UNKNOWN regions.

Complements test_region_analyzer.py: that file proves a degenerate region
classifies as UNKNOWN; this file proves UNKNOWN propagates safely through
the navigation decision — it can never itself produce MOVE_FORWARD, and
enough UNKNOWN regions trigger ROTATE_AND_SCAN like any other ambiguous
class.
"""

from depth_perception_engine.traversability.scene_interpreter import SceneInterpreter
from depth_perception_engine.traversability.types import (
    NavigationDecision,
    RegionClass,
    RegionStats,
    TextureClass,
)


def _region(name, classification, depth_avg_m=0.0, row=1, col=1):
    return RegionStats(
        name=name, row=row, col=col, x1=0, y1=0, x2=10, y2=10,
        valid_pct=0.0, invalid_pct=100.0, invalid_ratio=1.0,
        valid_count=0, total_pixels=100,
        depth_avg_m=depth_avg_m, depth_median_m=depth_avg_m,
        depth_min_m=0.0, depth_max_m=0.0,
        texture_score=0.0, entropy=0.0, gradient_magnitude=0.0, confidence=0.0,
        texture_class=TextureClass.LOW_TEXTURE, classification=classification,
    )


def _full_scene(forward, left=None, right=None, fill=RegionClass.CLEAR):
    """3x3 scene with every region defaulted to `fill`, forward/left/right overridden."""
    names = [
        ["TL", "TC", "TR"],
        ["ML", "MC", "MR"],
        ["BL", "BC", "BR"],
    ]
    scene = {}
    for r, row_names in enumerate(names):
        for c, name in enumerate(row_names):
            scene[name] = _region(name, fill, depth_avg_m=2.0, row=r, col=c)
    scene["MC"] = _region("MC", forward, depth_avg_m=2.0, row=1, col=1)
    if left is not None:
        scene["ML"] = _region("ML", left, depth_avg_m=2.0, row=1, col=0)
    if right is not None:
        scene["MR"] = _region("MR", right, depth_avg_m=2.0, row=1, col=2)
    return scene


class TestUnknownNeverProducesMoveForward:
    def test_unknown_forward_region_yields_slow_down_not_move_forward(self):
        interpreter = SceneInterpreter()
        scene = _full_scene(forward=RegionClass.UNKNOWN)

        decision = interpreter.decide_navigation(scene)

        assert decision == NavigationDecision.SLOW_DOWN
        assert decision != NavigationDecision.MOVE_FORWARD

    def test_move_forward_still_reachable_when_forward_is_genuinely_clear(self):
        """Sanity check: the fix must not have broken the legitimate happy path."""
        interpreter = SceneInterpreter(clear_m=1.20)
        scene = _full_scene(forward=RegionClass.CLEAR)

        decision = interpreter.decide_navigation(scene)

        assert decision == NavigationDecision.MOVE_FORWARD

    def test_majority_unknown_scene_triggers_rotate_and_scan(self):
        interpreter = SceneInterpreter(ambiguous_fraction_thresh=0.50)
        scene = _full_scene(
            forward=RegionClass.UNKNOWN, fill=RegionClass.UNKNOWN,
        )

        decision = interpreter.decide_navigation(scene)

        assert decision == NavigationDecision.ROTATE_AND_SCAN

    def test_unknown_left_and_right_cannot_be_turned_into(self):
        """An UNKNOWN side must not be treated as a safe turn target — only
        a genuinely CLEAR side may be turned into."""
        interpreter = SceneInterpreter()
        scene = _full_scene(
            forward=RegionClass.OBSTACLE,
            left=RegionClass.UNKNOWN,
            right=RegionClass.UNKNOWN,
        )

        decision = interpreter.decide_navigation(scene)

        assert decision == NavigationDecision.STOP
        assert decision not in (NavigationDecision.TURN_LEFT, NavigationDecision.TURN_RIGHT)

    def test_obstacle_forward_with_one_genuinely_clear_side_still_turns(self):
        """Sanity check: a real CLEAR side must still be usable — the fix
        only removes UNKNOWN as a valid turn target, not CLEAR."""
        interpreter = SceneInterpreter()
        scene = _full_scene(
            forward=RegionClass.OBSTACLE,
            left=RegionClass.CLEAR,
            right=RegionClass.UNKNOWN,
        )

        decision = interpreter.decide_navigation(scene)

        assert decision == NavigationDecision.TURN_LEFT


class TestEndToEndThroughRegionAnalyzer:
    """One test that goes through the real RegionAnalyzer (not hand-built
    RegionStats) to prove the two modules compose correctly, not just that
    each behaves correctly in isolation with hand-crafted fixtures."""

    def test_all_black_frame_never_yields_move_forward(self):
        import numpy as np

        interpreter = SceneInterpreter(rows=3, cols=3)
        h, w = 90, 90
        gray = np.zeros((h, w), dtype=np.uint8)
        disp = np.zeros((h, w), dtype=np.float32)
        depth = np.zeros((h, w), dtype=np.float32)

        scene = interpreter.analyze(gray, disp, depth)
        decision = interpreter.decide_navigation(scene)

        assert all(r.classification == RegionClass.UNKNOWN for r in scene.values())
        assert decision != NavigationDecision.MOVE_FORWARD
        assert decision == NavigationDecision.ROTATE_AND_SCAN
