"""
Velocity planner — converts ThreatAssessor beams into a forward speed and
yaw rate for autonomous indoor flight.

Why not a single global threshold: a fixed "stop if anything is within
1 m" rule treats a wall 1 m to the side the same as a wall 1 m straight
ahead, so in any normal indoor corridor (where side walls are routinely
close) the drone would just hover. Instead, only the beams inside the
forward flight corridor gate forward speed; all beams (corridor + sides)
contribute to yaw, steering toward the clearest direction.

Forward speed scales smoothly between caution_m and clear_m rather than
snapping from full speed to zero, so the drone decelerates into tight
spaces instead of slamming into a binary wall.

Example-only: this is flight-command generation, downstream of and
out of scope for the Depth Perception Engine library itself (it consumes
ObstacleAssessment-shaped beam data, not the other way around). See
docs/INTEGRATION_READINESS.md — mp01_perception calls the library for
depth/traversability/obstacle output only; velocity planning belongs to
MP-01's mission/planning layer, not its perception layer.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class VelocityPlanner:
    BLOCKED = "BLOCKED"
    CAUTION = "CAUTION"
    CLEAR   = "CLEAR"

    def __init__(
        self,
        n_beams:          int   = 20,
        corridor_frac:    float = 0.35,  # fraction of the *valid* beam range (centred) treated as "ahead"
        caution_m:        float = 1.0,
        clear_m:          float = 2.0,
        max_forward_mps:  float = 0.6,
        creep_mps:        float = 0.15,  # speed used when corridor data is unreliable (NO_DATA)
        max_yaw_dps:      float = 25.0,
        dead_zone_beams:  int   = 0,     # leading beams structurally invalid (SGBM dead zone)
    ) -> None:
        if not (0.0 < corridor_frac <= 1.0):
            raise ValueError("corridor_frac must be in (0, 1].")
        self._n_beams         = n_beams
        self._caution_m       = caution_m
        self._clear_m         = clear_m
        self._max_forward_mps = max_forward_mps
        self._creep_mps       = creep_mps
        self._max_yaw_dps     = max_yaw_dps

        # Centre the corridor on the true optical centre of the frame (the
        # dead zone is one-sided, so the real forward direction is still
        # outside it) and only clip the left edge if it would otherwise
        # dip into the dead zone. Re-centering on the valid sub-range
        # instead would bias the corridor off to one side of "ahead".
        half_width = n_beams * corridor_frac / 2.0
        center     = n_beams / 2.0
        self._corridor_lo = max(dead_zone_beams, int(round(center - half_width)))
        self._corridor_hi = int(round(center + half_width))

    def plan(self, beams: List[Dict], safest_beam: Optional[Dict]) -> Dict:
        """Return navigation data derived from the current beam scan.

        Returns:
            forward_mps      — commanded forward speed
            yaw_rate_dps     — steer toward safest_beam (+ = right), regardless
                                of corridor status, so the drone keeps
                                orienting toward open space even while braking
            status           — worst condition found in the forward corridor;
                                drives forward_mps
            ahead_distance_m — nearest measured distance within the forward
                                corridor (0.0 if no beam there has a reading)
            corridor_x1/x2   — pixel bounds of the forward corridor, for display
        """
        corridor = [b for b in beams if self._corridor_lo <= b["index"] < self._corridor_hi]

        forward_mps, status, ahead_distance_m = self._corridor_speed(corridor)
        yaw_rate_dps = self._yaw_rate(safest_beam)

        corridor_x1 = corridor[0]["x1"] if corridor else 0
        corridor_x2 = corridor[-1]["x2"] if corridor else 0

        return {
            "forward_mps":      forward_mps,
            "yaw_rate_dps":     yaw_rate_dps,
            "status":           status,
            "ahead_distance_m": ahead_distance_m,
            "corridor_x1":      corridor_x1,
            "corridor_x2":      corridor_x2,
        }

    # ------------------------------------------------------------------
    def _corridor_speed(self, corridor: List[Dict]) -> Tuple[float, str, float]:
        """Return (forward_mps, status, ahead_distance_m).

        ahead_distance_m is always derived from the same beams that decided
        *status* — never from an unrelated beam elsewhere in the corridor —
        so a displayed distance always matches the reason given for it.
        """
        blocked = [b for b in corridor if b["status"] == self.BLOCKED]
        if blocked:
            dists = [b["distance_m"] for b in blocked if b["distance_m"] > 0.0]
            return 0.0, self.BLOCKED, (min(dists) if dists else 0.0)

        caution_dists = [b["distance_m"] for b in corridor if b["status"] == self.CAUTION]
        if caution_dists:
            nearest = min(caution_dists)
            frac = (nearest - self._caution_m) / (self._clear_m - self._caution_m)
            frac = float(np.clip(frac, 0.0, 1.0))
            speed = self._creep_mps + frac * (self._max_forward_mps - self._creep_mps)
            return speed, self.CAUTION, nearest

        # All corridor beams are CLEAR or NO_DATA (no obstacle signal) —
        # NO_DATA could be an unresolved near-textureless surface, so cap
        # speed at a creep rate rather than assuming it's open air. There is
        # no real distance reading behind this caution, so report 0.0
        # rather than borrowing an unrelated CLEAR beam's distance.
        if any(b["status"] == "NO_DATA" for b in corridor):
            return self._creep_mps, self.CAUTION, 0.0

        clear_dists = [b["distance_m"] for b in corridor if b["distance_m"] > 0.0]
        return self._max_forward_mps, self.CLEAR, (min(clear_dists) if clear_dists else 0.0)

    def _yaw_rate(self, safest_beam: Optional[Dict]) -> float:
        if safest_beam is None:
            return 0.0
        center = (self._n_beams - 1) / 2.0
        offset = (safest_beam["index"] - center) / center  # -1..1, left..right
        return float(np.clip(offset, -1.0, 1.0)) * self._max_yaw_dps
