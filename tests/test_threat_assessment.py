"""
Coverage for ThreatAssessor — previously untested (no test file existed
for this module). Focuses on the temporal state machine (EMA smoothing,
status debounce) and specifically on the stale-status-leak fix made
during remediation (Issue B): a beam that loses depth evidence must have
its status updated immediately, not delayed by the ordinary debounce
window that legitimately smooths noise between real readings.
"""

import numpy as np

from depth_perception_engine.obstacles.threat_assessment import ThreatAssessor

_H, _W = 60, 200  # 200px wide, 20 beams -> 10px/beam


def _depth_map(fill_m, h=_H, w=_W):
    return np.full((h, w), fill_m, dtype=np.float32)


def _zeros(h=_H, w=_W):
    return np.zeros((h, w), dtype=np.float32)


class TestBasicStatusClassification:
    def test_far_depth_is_clear(self):
        # debounce_frames=1: this test targets per-frame classification,
        # not debounce timing (covered separately in TestStatusDebounce) —
        # _stable_status starts at NO_DATA, so a >1 debounce would report
        # the initial state, not this frame's classification, on call 1.
        assessor = ThreatAssessor(
            n_beams=4, clear_m=2.0, caution_m=1.0, min_valid=1, debounce_frames=1,
        )
        result = assessor.assess(_depth_map(3.0, w=40))
        assert all(b["status"] == ThreatAssessor.CLEAR for b in result["beams"])

    def test_near_depth_is_blocked_after_debounce(self):
        assessor = ThreatAssessor(
            n_beams=4, clear_m=2.0, caution_m=1.0, min_valid=1, debounce_frames=1,
        )
        result = assessor.assess(_depth_map(0.3, w=40))
        assert all(b["status"] == ThreatAssessor.BLOCKED for b in result["beams"])

    def test_zero_depth_map_is_no_data(self):
        assessor = ThreatAssessor(n_beams=4, min_valid=1)
        result = assessor.assess(_zeros(w=40))
        assert all(b["status"] == ThreatAssessor.NO_DATA for b in result["beams"])
        assert all(b["distance_m"] == 0.0 for b in result["beams"])
        assert result["safest_beam"] is None


class TestStatusDebounce:
    def test_single_frame_flicker_does_not_change_stable_status(self):
        """A one-frame-only reading must not flip the reported status —
        this is the noise-rejection behavior debounce exists to provide,
        and the Issue B fix must not have broken it for ordinary
        evidenced readings."""
        assessor = ThreatAssessor(
            n_beams=1, clear_m=2.0, caution_m=1.0, min_valid=1, debounce_frames=3,
        )
        for _ in range(3):  # 3 calls needed to make CLEAR the stable status
            assessor.assess(_depth_map(3.0, w=10))
        result = assessor.assess(_depth_map(0.3, w=10))  # single-frame blip toward BLOCKED

        assert result["beams"][0]["status"] == ThreatAssessor.CLEAR

    def test_persistent_change_eventually_updates_after_debounce_frames(self):
        assessor = ThreatAssessor(
            n_beams=1, clear_m=2.0, caution_m=1.0, min_valid=1, debounce_frames=3,
        )
        assessor.assess(_depth_map(3.0, w=10))
        for _ in range(3):
            result = assessor.assess(_depth_map(0.3, w=10))

        assert result["beams"][0]["status"] == ThreatAssessor.BLOCKED


class TestEvidenceLossBypassesDebounce:
    """Regression coverage for the Issue B fix: loss of evidence (zero
    valid depth pixels this frame) must update status immediately, not
    after debounce_frames of persistence — otherwise a stale
    CLEAR/CAUTION/BLOCKED status leaks into the output for several frames
    after the underlying evidence is already gone."""

    def test_status_drops_to_no_data_on_the_very_first_invalid_frame(self):
        assessor = ThreatAssessor(
            n_beams=1, clear_m=2.0, caution_m=1.0, min_valid=1, debounce_frames=5,
        )
        # Establish a confident CLEAR reading over several good frames.
        for _ in range(5):
            assessor.assess(_depth_map(3.0, w=10))
        assert assessor.assess(_depth_map(3.0, w=10))["beams"][0]["status"] == ThreatAssessor.CLEAR

        # Evidence vanishes completely on a single frame.
        result = assessor.assess(_zeros(w=10))

        assert result["beams"][0]["status"] == ThreatAssessor.NO_DATA, (
            "stale CLEAR status leaked past the frame where evidence was lost"
        )
        assert result["beams"][0]["distance_m"] == 0.0

    def test_status_and_distance_never_disagree_on_an_invalid_frame(self):
        """The specific inconsistency this fix closes: distance_m already
        reset to 0.0 immediately (pre-existing behavior) while status
        lagged behind (the bug) — assert they now change together."""
        assessor = ThreatAssessor(
            n_beams=1, clear_m=2.0, caution_m=1.0, min_valid=1, debounce_frames=10,
        )
        for _ in range(3):
            assessor.assess(_depth_map(3.0, w=10))

        result = assessor.assess(_zeros(w=10))
        beam = result["beams"][0]

        is_stale_pair = beam["status"] == ThreatAssessor.CLEAR and beam["distance_m"] == 0.0
        assert not is_stale_pair, "status=CLEAR but distance_m=0.0 — stale/inconsistent output"
        assert beam["status"] == ThreatAssessor.NO_DATA

    def test_evidence_loss_immediately_after_a_blocked_reading_also_updates_at_once(self):
        assessor = ThreatAssessor(
            n_beams=1, clear_m=2.0, caution_m=1.0, min_valid=1, debounce_frames=3,
        )
        for _ in range(3):  # 3 calls needed to make BLOCKED the stable status
            assessor.assess(_depth_map(0.3, w=10))
        assert assessor.assess(_depth_map(0.3, w=10))["beams"][0]["status"] == ThreatAssessor.BLOCKED

        result = assessor.assess(_zeros(w=10))

        assert result["beams"][0]["status"] == ThreatAssessor.NO_DATA

    def test_recovery_after_evidence_loss_still_requires_normal_debounce(self):
        """The fix only bypasses debounce for the LOSS of evidence, not its
        return — recovering from NO_DATA back to a confident status must
        still require persistence, same as any other transition, so a
        single-frame spurious reading right after a dropout doesn't
        instantly re-arm a confident status."""
        assessor = ThreatAssessor(
            n_beams=1, clear_m=2.0, caution_m=1.0, min_valid=1, debounce_frames=3,
        )
        assessor.assess(_depth_map(3.0, w=10))
        assessor.assess(_zeros(w=10))  # evidence lost -> immediate NO_DATA
        result = assessor.assess(_depth_map(3.0, w=10))  # single-frame recovery attempt

        assert result["beams"][0]["status"] == ThreatAssessor.NO_DATA, (
            "recovery from NO_DATA must still be debounced, not immediate"
        )


class TestDeadZoneReclassification:
    def test_dead_zone_column_stays_no_data_even_with_high_invalid_disparity(self):
        assessor = ThreatAssessor(
            n_beams=1, min_valid=1, dead_zone_px=50, blocked_invalid_ratio=0.5,
        )
        depth = _zeros(w=10)
        disp = np.zeros((_H, 10), dtype=np.float32)  # 100% invalid disparity

        result = assessor.assess(depth, raw_disp=disp)

        assert result["beams"][0]["status"] == ThreatAssessor.NO_DATA

    def test_beyond_dead_zone_high_invalid_disparity_reclassifies_to_blocked(self):
        assessor = ThreatAssessor(
            n_beams=1, min_valid=1, dead_zone_px=0, blocked_invalid_ratio=0.5, debounce_frames=1,
        )
        depth = _zeros(w=10)
        disp = np.zeros((_H, 10), dtype=np.float32)

        result = assessor.assess(depth, raw_disp=disp)

        assert result["beams"][0]["status"] == ThreatAssessor.BLOCKED


class TestSafestBeam:
    def test_safest_beam_ignores_no_data_beams(self):
        assessor = ThreatAssessor(
            n_beams=2, clear_m=2.0, caution_m=1.0, min_valid=1, debounce_frames=1,
        )
        depth = np.zeros((_H, 20), dtype=np.float32)
        depth[:, :10] = 3.0  # beam 0: CLEAR
        # beam 1 stays zero -> NO_DATA

        result = assessor.assess(depth)

        assert result["safest_beam"] is not None
        assert result["safest_beam"]["index"] == 0

    def test_safest_beam_is_none_when_every_beam_is_no_data(self):
        assessor = ThreatAssessor(n_beams=2, min_valid=1)
        result = assessor.assess(_zeros(w=20))
        assert result["safest_beam"] is None
