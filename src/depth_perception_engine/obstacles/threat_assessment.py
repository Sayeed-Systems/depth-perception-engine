"""
Threat assessor — scans the depth map in N vertical beams and assigns
a navigability status to each beam based on the nearest obstacle detected.

Each beam samples a vertical column slice of the depth map using IQR
filtering and a low percentile (biased toward closer surfaces). Dead-zone
columns (SGBM disparity gap on the left) naturally produce NO_DATA rather
than false readings.

When raw disparity is supplied, a beam with no valid depth but a high
fraction of blocked (invalid) disparity pixels is reclassified as BLOCKED
instead of NO_DATA — this catches textureless obstacles (e.g. a blank wall)
that produce no stereo correspondence regardless of distance.

Each beam's distance is EMA-smoothed and its status is debounced (must
persist for several consecutive frames before changing) so per-frame SGBM
noise doesn't make readings jump or the obstacle overlay flicker. That
debounce is bypassed the moment a beam has no real depth evidence at all
(distance_m's EMA already resets to 0.0 immediately in that case) — a
stale CLEAR/CAUTION/BLOCKED status must never keep being reported once
the evidence behind it is gone, even for the few frames debounce would
otherwise hold it for. See the `evidence_lost` handling in assess().

Holds per-beam temporal state across assess() calls (see the constructor) —
constructing a fresh ThreatAssessor per frame throws that smoothing away.
DepthPerceptionPipeline holds one instance for this reason; the stateless
pipeline.api.detect_obstacles() convenience function does not.
"""

import numpy as np
from typing import Dict, List, Optional


class ThreatAssessor:
    CLEAR   = "CLEAR"     # beyond clear_m
    CAUTION = "CAUTION"   # between caution_m and clear_m
    BLOCKED = "BLOCKED"   # closer than caution_m
    NO_DATA = "NO_DATA"   # insufficient valid pixels

    def __init__(
        self,
        n_beams:              int   = 20,
        clear_m:              float = 2.0,
        caution_m:            float = 1.0,
        percentile:           int   = 15,
        min_valid:            int   = 5,
        blocked_invalid_ratio: float = 0.85,
        ema_alpha:             float = 0.30,
        debounce_frames:       int   = 3,
        dead_zone_px:          int   = 0,
    ) -> None:
        self._n_beams              = n_beams
        self._clear_m              = clear_m
        self._caution_m            = caution_m
        self._percentile           = percentile
        self._min_valid            = min_valid
        self._blocked_invalid_ratio = blocked_invalid_ratio
        self._ema_alpha             = ema_alpha
        self._debounce_frames       = debounce_frames
        self._dead_zone_px          = dead_zone_px

        # Per-beam temporal state — persists across assess() calls so
        # readings and overlay status stay stable frame-to-frame.
        self._ema_dist       = [0.0] * n_beams
        self._stable_status  = [self.NO_DATA] * n_beams
        self._pending_status = [None] * n_beams
        self._pending_count  = [0] * n_beams

    def assess(self, depth_map: np.ndarray, raw_disp: Optional[np.ndarray] = None) -> Dict:
        """Scan depth_map in vertical beams and return navigability data.

        Args:
            depth_map: float32 (H, W) depth map in metres, zeros = invalid.
            raw_disp:  optional (H, W) raw disparity map used to detect
                       textureless obstacles that produce no depth at all.

        Returns:
            Dict with:
                beams        — list of beam dicts (see below)
                safest_beam  — beam dict with the greatest distance, or None
            Each beam dict:
                index, x1, x2   — beam index and pixel column bounds
                distance_m      — IQR-filtered percentile depth (0 = no data)
                status          — CLEAR / CAUTION / BLOCKED / NO_DATA
                valid_count     — Phase D7: raw count of valid (>0, finite)
                                   depth pixels in this beam's column,
                                   captured BEFORE IQR filtering — the same
                                   `valid` array the distance/status
                                   computation below already builds, not a
                                   second scan. Added for
                                   geometry.provider.ClearanceEvidence's
                                   coverage fields; does not change
                                   distance_m/status derivation at all.
                total_pixels    — Phase D7: this beam column's total pixel
                                   count (`col.size`).
        """
        h, w = depth_map.shape[:2]
        beam_w = w / self._n_beams
        beams: List[Dict] = []

        for i in range(self._n_beams):
            x1 = int(i * beam_w)
            x2 = int((i + 1) * beam_w)

            # Flatten each non-contiguous vertical slab before the repeated
            # validity/IQR/percentile scans. Although this makes a copy, the
            # contiguous access pattern is measurably faster than applying
            # those operations directly to the strided 2-D view.
            col   = depth_map[:, x1:x2].flatten().astype(np.float32)
            valid = col[(col > 0) & np.isfinite(col)]
            # Snapshot BEFORE any IQR-filtering reassignment below —
            # `valid` itself is narrowed further inside the branch that
            # follows, but valid_count/total_pixels describe genuine
            # geometric SUPPORT (how much real depth evidence existed at
            # all), not the outlier-rejected subset distance_m happens to
            # be computed from.
            valid_count = int(valid.size)
            total_pixels = int(col.size)

            if valid.size >= self._min_valid:
                q1, q3 = np.percentile(valid, [25, 75])
                iqr = q3 - q1
                if iqr > 0:
                    valid = valid[
                        (valid >= q1 - 1.5 * iqr) &
                        (valid <= q3 + 1.5 * iqr)
                    ]
                d_m = float(np.percentile(valid, self._percentile)) if valid.size >= self._min_valid else 0.0
            else:
                d_m = 0.0

            # Loss-of-evidence for this frame, before any reclassification
            # below — used to bypass status debounce entirely (see that
            # block for why this must be decided here, not from the final
            # post-reclassification `status`).
            evidence_lost = d_m <= 0.0

            if evidence_lost:
                status = self.NO_DATA
            elif d_m < self._caution_m:
                status = self.BLOCKED
            elif d_m < self._clear_m:
                status = self.CAUTION
            else:
                status = self.CLEAR

            # Skip beams that fall (even partially) inside the SGBM dead
            # zone — those columns are structurally invalid regardless of
            # what's in front of the camera, so they must stay NO_DATA
            # rather than being mistaken for a blocked obstacle.
            if status == self.NO_DATA and raw_disp is not None and x1 >= self._dead_zone_px:
                disp_col = raw_disp[:, x1:x2]
                if disp_col.size and (disp_col <= 0).mean() >= self._blocked_invalid_ratio:
                    status = self.BLOCKED

            # EMA-smooth the distance — snap on first reading, reset on loss
            # of signal so a stale distance doesn't linger once invalid.
            if not evidence_lost:
                prev = self._ema_dist[i]
                self._ema_dist[i] = d_m if prev <= 0.0 else (
                    self._ema_alpha * d_m + (1.0 - self._ema_alpha) * prev
                )
            else:
                self._ema_dist[i] = 0.0

            # Debounce status — require the new status to persist for
            # several consecutive frames before it actually changes, so a
            # single noisy frame can't flip the overlay on/off. EXCEPTION:
            # a frame with no real depth evidence (evidence_lost, decided
            # above — independent of whatever `status` got reclassified
            # to) applies immediately, bypassing debounce entirely.
            #
            # Why: ordinary debounce noise-rejection is for smoothing
            # genuine, evidenced readings that flicker between
            # CLEAR/CAUTION/BLOCKED — valuable there. But applying that
            # same delay when evidence has vanished entirely means a
            # stale, no-longer-trustworthy status (e.g. CLEAR from
            # several frames ago) keeps being reported for up to
            # debounce_frames-1 more frames after a real fault (camera
            # dropout, corruption). That is a stale-value leak: this
            # beam's `distance_m` has already reset to 0.0 immediately
            # above, but without this exception `status` would lag
            # behind it, so a consumer reading status alone could act on
            # "CLEAR" for frames where distance_m is simultaneously 0.0.
            # Confirmed missing live during remediation (Issue B).
            if evidence_lost:
                self._stable_status[i]  = status
                self._pending_status[i] = None
                self._pending_count[i]  = 0
            elif status == self._stable_status[i]:
                self._pending_status[i] = None
                self._pending_count[i]  = 0
            else:
                if status == self._pending_status[i]:
                    self._pending_count[i] += 1
                else:
                    self._pending_status[i] = status
                    self._pending_count[i]  = 1
                if self._pending_count[i] >= self._debounce_frames:
                    self._stable_status[i]  = status
                    self._pending_status[i] = None
                    self._pending_count[i]  = 0

            beams.append({
                "index":      i,
                "x1":         x1,
                "x2":         x2,
                "distance_m": self._ema_dist[i],
                "status":     self._stable_status[i],
                "valid_count": valid_count,
                "total_pixels": total_pixels,
            })

        valid_beams = [b for b in beams if b["status"] != self.NO_DATA]
        safest: Optional[Dict] = max(valid_beams, key=lambda b: b["distance_m"]) if valid_beams else None

        return {"beams": beams, "safest_beam": safest}
