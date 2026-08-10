"""
Short-window rotational motion compensation — Level 4, Phase E5.

Mirrors temporal/consistency.py's and temporal/stabilization.py's own
precedent: a small class of plain string constants
(RotationCompensationStatus) plus pure, stateless functions. Owns exactly
two things: short-window gyro integration (select_motion_hint_samples,
integrate_angular_velocity) and prior-geometry rotation
(compensate_prior_geometry) — nothing else. See
docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md for the full derivation and
per-decision rationale.

E5 never becomes a pose estimator: no camera orientation R_cam(t) is ever
computed, stored, or returned anywhere in this module — only the ephemeral
relative rotation between two specific frames, used once and discarded.
No translation term appears in any formula here. No orientation, velocity,
or bias state survives across calls — every function is pure, and
compute_rotation_compensation() is called fresh, with fresh inputs, once
per DepthPerceptionPipeline.process() call.

E5 operates upstream of, and never modifies, temporal/consistency.py or
temporal/stabilization.py — it only decides which TemporalRecord value
DepthPerceptionPipeline.process() passes into their existing, frozen
functions.

Level 4, Phase E7 addendum: this module gained one additive sibling
function, compensate_prior_geometry_with_payload(), sharing
compensate_prior_geometry()'s own exact reprojection math (factored into
the private _reproject_source_to_target() helper both now call) so that
temporal.persistence.TemporalPersistenceTracker can carry its own
per-cell bookkeeping state (support count, absence streak, first-observed
timestamp) through the identical rotation-compensated mapping the depth
channel itself uses — see docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md's "reuse
E5, don't reimplement" decision. compensate_prior_geometry()'s own
pre-E7 behavior is unchanged (still the exact same arithmetic, verified
byte-identical by the pre-existing E5 test suite, unmodified).
"""

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from depth_perception_engine.temporal.types import MotionHint, TemporalRecord


class RotationCompensationStatus:
    """Plain string constants for compute_rotation_compensation()'s
    outcome — mirrors temporal.TemporalAdmissionStatus/
    TemporalConsistencyState/TemporalStabilizationState's existing
    precedent (plain string constants, not a new Enum type)."""

    APPLIED = "APPLIED"
    """A relative rotation was successfully integrated from at least one
    admissible MotionHint sample and used to re-project the previous
    frame's geometry onto the current frame's own pixel grid."""

    NOT_APPLIED = "NOT_APPLIED"
    """No compensation was applied — missing/invalid/stale/insufficient
    motion hints, no prior record to compensate, or E5 itself disabled.
    previous_for_comparison is passed into E3/E4 completely unchanged in
    this case — see docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md's Decision 6."""


# ----------------------------------------------------------------------
# Short-window gyro integration
# ----------------------------------------------------------------------
def select_motion_hint_samples(
    samples: Optional[Sequence[MotionHint]],
    previous_timestamp: Optional[float],
    current_timestamp: Optional[float],
) -> List[MotionHint]:
    """Filter samples to those admissible for integration between
    previous_timestamp and current_timestamp.

    A sample is rejected if it is None, MotionHint.valid is False, its
    timestamp is missing/non-finite, it falls outside the half-open
    interval (previous_timestamp, current_timestamp], or its timestamp
    does not strictly exceed the last *accepted* sample's timestamp
    (starting from previous_timestamp) — this single rule rejects
    non-monotonic and duplicate-timestamp samples identically. Processes
    `samples` in the order given, never pre-sorted — silently reordering
    a misbehaving caller's sequence would mask a real bug, not fix one.

    Returns an empty list (never raises) for any degenerate input:
    samples is None/empty, timestamps missing/non-finite, or
    current_timestamp <= previous_timestamp.
    """
    if not samples:
        return []
    if previous_timestamp is None or current_timestamp is None:
        return []
    if not math.isfinite(previous_timestamp) or not math.isfinite(current_timestamp):
        return []
    if not (current_timestamp > previous_timestamp):
        return []

    accepted: List[MotionHint] = []
    last_ts = previous_timestamp
    for sample in samples:
        if sample is None or not sample.valid:
            continue
        ts = sample.timestamp
        if ts is None or not math.isfinite(ts):
            continue
        if not (previous_timestamp < ts <= current_timestamp):
            continue
        if not (ts > last_ts):
            continue
        accepted.append(sample)
        last_ts = ts
    return accepted


def integrate_angular_velocity(accepted_samples: Sequence[MotionHint], previous_timestamp: float) -> np.ndarray:
    """Integrate an already-filtered, chronologically-ordered sequence of
    MotionHint samples into ΔR_prev_to_curr.

    Per-sample increment ΔR_k = Exp([ω_k]_x · Δt_k), Δt_k = t_k - t_{k-1}
    (t_0 := previous_timestamp) — zero-order hold, never extrapolated past
    each sample's own timestamp. Composed chronologically via
    post-multiplication (ΔR_total = ΔR_1 @ ... @ ΔR_n — the standard
    "compose in the local/body frame" rule, correct because a gyroscope
    reports angular velocity in its own instantaneous frame). The
    returned value is ΔR_total's transpose — the rotation that takes a
    point expressed in the previous frame's camera coordinates to the
    current frame's own camera coordinates — see
    docs/LEVEL4_E5_IMPLEMENTATION_PLAN.md's Decision 1 for the full
    derivation.

    A small Python loop over samples, not vectorized — deliberately: this
    loop is bounded by sample COUNT (typically a handful of IMU readings
    between two camera frames), never by pixel count, so it is not the
    class of performance regression tests/test_performance_guards.py
    guards against (a per-*pixel* loop). A closed-form matrix-chain
    composition is more code and no clearer than this loop for a
    "mathematically inspectable" (Decision 9) computation.

    Args:
        accepted_samples: non-empty, chronologically ordered (as
            select_motion_hint_samples() already guarantees).
        previous_timestamp: the previous frame's own timestamp, seconds.

    Returns:
        (3, 3) float32 rotation matrix, ΔR_prev_to_curr.
    """
    delta_r_total = np.eye(3, dtype=np.float64)
    t_prev = previous_timestamp
    for sample in accepted_samples:
        dt = sample.timestamp - t_prev
        rotation_vector = sample.angular_velocity_rad_s.astype(np.float64) * dt
        delta_r_k, _ = cv2.Rodrigues(rotation_vector)
        delta_r_total = delta_r_total @ delta_r_k
        t_prev = sample.timestamp
    return delta_r_total.T.astype(np.float32)


# ----------------------------------------------------------------------
# Prior-geometry rotation
# ----------------------------------------------------------------------
def _reproject_source_to_target(
    previous_depth_snapshot_m: np.ndarray,
    delta_r_prev_to_curr: np.ndarray,
    stride: int,
    focal_length_px: float,
    principal_point_px: Tuple[float, float],
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared projection-index computation underlying both
    compensate_prior_geometry() and (Level 4, Phase E7)
    compensate_prior_geometry_with_payload() — computes, once, which
    source (previous-frame) grid cells reproject into which target
    (current-frame) grid cells under delta_r_prev_to_curr, and the
    nearest-wins (farthest-first stable sort) scatter order, WITHOUT
    itself writing out any depth or payload values. Both public functions
    below apply this exact same mapping to whichever channel(s) they
    scatter, so there remains exactly ONE reprojection implementation in
    this module — Phase E7 extends this module additively (a payload-
    carrying sibling function) rather than writing a second, independent
    motion-compensation path. See docs/LEVEL4_E7_IMPLEMENTATION_PLAN.md's
    "reuse E5, don't reimplement" decision. A pure refactor of
    compensate_prior_geometry()'s own pre-E7 body — byte-for-byte the
    same arithmetic, unchanged.

    Returns:
        h_dec, w_dec: previous_depth_snapshot_m's own shape.
        scatter_mask: (h_dec, w_dec) bool — which SOURCE cells reproject
            in front of the camera and land in-bounds on the target grid.
        flat_target: 1D int64 array, one destination flat (row-major)
            index per True entry of scatter_mask, in scatter_mask's own
            row-major (C) order.
        order: 1D int64 array — stable, farthest-re-projected-depth-first
            sort order to apply (via flat_target[order]) to any channel
            aligned with flat_target, so nearest-wins occlusion resolution
            is identical for every channel scattered through this mapping.
        z_r: (h_dec, w_dec) float64 — the full re-projected depth field
            (only entries where scatter_mask is True are meaningful).
    """
    h_dec, w_dec = previous_depth_snapshot_m.shape
    cx, cy = principal_point_px
    f = focal_length_px

    i_idx, j_idx = np.indices((h_dec, w_dec), dtype=np.float64)
    v = i_idx * stride
    u = j_idx * stride

    z = previous_depth_snapshot_m.astype(np.float64)
    valid = z > 0.0
    x = (u - cx) * z / f
    y = (v - cy) * z / f

    r = delta_r_prev_to_curr.astype(np.float64)
    x_r = r[0, 0] * x + r[0, 1] * y + r[0, 2] * z
    y_r = r[1, 0] * x + r[1, 1] * y + r[1, 2] * z
    z_r = r[2, 0] * x + r[2, 1] * y + r[2, 2] * z

    in_front = valid & (z_r > 0.0)
    safe_z_r = np.where(in_front, z_r, 1.0)
    u_r = f * x_r / safe_z_r + cx
    v_r = f * y_r / safe_z_r + cy

    j_r = np.round(u_r / stride).astype(np.int64)
    i_r = np.round(v_r / stride).astype(np.int64)
    in_bounds = (i_r >= 0) & (i_r < h_dec) & (j_r >= 0) & (j_r < w_dec)

    scatter_mask = in_front & in_bounds
    flat_target = (i_r * w_dec + j_r)[scatter_mask]
    flat_z = z_r[scatter_mask]

    # Farthest first, so the nearest (smallest z) write happens last at
    # any target cell hit by more than one reprojected point —
    # deterministic, ties broken by original (post-filter) order via a
    # stable sort.
    order = np.argsort(-flat_z, kind="stable") if flat_target.size > 0 else np.zeros(0, dtype=np.int64)

    return h_dec, w_dec, scatter_mask, flat_target, order, z_r


def compensate_prior_geometry(
    previous_depth_snapshot_m: np.ndarray,
    delta_r_prev_to_curr: np.ndarray,
    stride: int,
    focal_length_px: float,
    principal_point_px: Tuple[float, float],
) -> np.ndarray:
    """Back-project previous_depth_snapshot_m into 3D camera-frame points,
    rotate by delta_r_prev_to_curr, and re-project onto the SAME decimated
    grid shape (current frame's own grid — identical shape/stride/
    intrinsics within one pipeline instance).

    Fully vectorized (no per-pixel Python loop) — see
    tests/test_rotation_compensation.py::TestPureFunctions::
    test_compensate_prior_geometry_has_no_for_or_while_loop.

    Back-projection/re-projection use a plain pinhole model
    (X = (u-cx)·Z/f, Y=(v-cy)·Z/f) — correct here because depth_snapshot_m
    is already post-rectification (same intrinsics DepthEstimator/
    PointCloudBuilder already use, derived from the same Q matrix, no
    distortion handling needed). No translation term appears in any
    function.

    Occlusion at the target grid is resolved deterministically:
    nearest-wins (smallest re-projected depth) via a stable sort plus
    fancy-index assignment (last write wins) — no interpolation, no
    hole-filling. A target cell with no re-projected point stays 0.0
    (invalid) — never fabricated.

    Args:
        previous_depth_snapshot_m: (H, W) float array, depth_map's own
            0.0-is-invalid convention, decimated by `stride`.
        delta_r_prev_to_curr: (3, 3) rotation matrix.
        stride: the same PipelineConfig.temporal_consistency_sampling_stride
            this snapshot was decimated with.
        focal_length_px: rectified focal length (fx == fy convention this
            library's Q-matrix-derived intrinsics already use elsewhere —
            see calibration.contracts.StereoExtrinsics.from_calibration).
        principal_point_px: (cx, cy), rectified, full-resolution pixel
            units.

    Returns:
        (H, W) float32 array, same shape/convention as the input.
    """
    h_dec, w_dec, scatter_mask, flat_target, order, z_r = _reproject_source_to_target(
        previous_depth_snapshot_m, delta_r_prev_to_curr, stride, focal_length_px, principal_point_px,
    )
    flat_z = z_r[scatter_mask]

    compensated_flat = np.zeros(h_dec * w_dec, dtype=np.float32)
    if flat_target.size > 0:
        compensated_flat[flat_target[order]] = flat_z[order].astype(np.float32)

    return compensated_flat.reshape(h_dec, w_dec)


def compensate_prior_geometry_with_payload(
    previous_depth_snapshot_m: np.ndarray,
    payload_channels: Sequence[np.ndarray],
    delta_r_prev_to_curr: np.ndarray,
    stride: int,
    focal_length_px: float,
    principal_point_px: Tuple[float, float],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Level 4, Phase E7. Identical reprojection to
    compensate_prior_geometry() (reuses _reproject_source_to_target()'s
    exact, single mapping — this is NOT a second motion-compensation
    implementation, see that function's own docstring), but additionally
    carries an arbitrary number of same-shape auxiliary payload channels
    through the identical source-to-target mapping and nearest-wins
    occlusion order, so per-cell bookkeeping state (e.g.
    temporal.persistence.TemporalPersistenceTracker's own support-count/
    absence-streak/first-observed-timestamp grids) stays spatially
    consistent with the depth channel's own rotation-compensated
    movement — a payload value is scattered to exactly the same target
    cell, in exactly the same occlusion order, as the depth value
    computed from the SAME source cell. A target cell with no reprojected
    source point gets 0.0 in every channel (both depth and every
    payload) — never fabricated, same 0.0-invalid convention as
    compensate_prior_geometry() itself.

    Args:
        previous_depth_snapshot_m: same as compensate_prior_geometry() —
            this is the ONLY channel used to derive 3D geometry (X, Y
            from this channel's own Z values); payload channels never
            influence the projection itself.
        payload_channels: sequence of (h_dec, w_dec) arrays (any numeric
            dtype), one entry per auxiliary channel to co-warp. Each is
            read, never mutated.
        delta_r_prev_to_curr, stride, focal_length_px, principal_point_px:
            same as compensate_prior_geometry().

    Returns:
        (compensated_depth, compensated_payloads): compensated_depth is
        byte-identical to what compensate_prior_geometry() returns for
        the same first/third/fourth/fifth/sixth arguments;
        compensated_payloads is a list, same length/order as
        payload_channels, each entry a (h_dec, w_dec) float32 array.
    """
    h_dec, w_dec, scatter_mask, flat_target, order, z_r = _reproject_source_to_target(
        previous_depth_snapshot_m, delta_r_prev_to_curr, stride, focal_length_px, principal_point_px,
    )
    flat_z = z_r[scatter_mask]

    compensated_depth_flat = np.zeros(h_dec * w_dec, dtype=np.float32)
    if flat_target.size > 0:
        compensated_depth_flat[flat_target[order]] = flat_z[order].astype(np.float32)
    compensated_depth = compensated_depth_flat.reshape(h_dec, w_dec)

    compensated_payloads: List[np.ndarray] = []
    for payload in payload_channels:
        flat_payload = payload.astype(np.float64)[scatter_mask]
        compensated_payload_flat = np.zeros(h_dec * w_dec, dtype=np.float32)
        if flat_target.size > 0:
            compensated_payload_flat[flat_target[order]] = flat_payload[order].astype(np.float32)
        compensated_payloads.append(compensated_payload_flat.reshape(h_dec, w_dec))

    return compensated_depth, compensated_payloads


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def compute_rotation_compensation(
    previous_record: Optional[TemporalRecord],
    motion_hints: Optional[Sequence[MotionHint]],
    previous_timestamp: Optional[float],
    current_timestamp: Optional[float],
    stride: int,
    focal_length_px: float,
    principal_point_px: Tuple[float, float],
) -> Tuple[Optional[TemporalRecord], str]:
    """Attempt rotation compensation of previous_record's depth_snapshot_m.

    Returns (record_to_use, status):
        - On any failure/fallback condition (Decision 6): (previous_record,
          RotationCompensationStatus.NOT_APPLIED) — previous_record itself,
          completely unchanged (identity, not a copy), so a caller that
          ignores `status` and just uses `record_to_use` gets exactly
          pre-E5 behavior.
        - On success: (a NEW TemporalRecord — same timestamp/confidence/
          geometry_quality/motion_hint as previous_record, depth_snapshot_m
          replaced via dataclasses.replace — RotationCompensationStatus.APPLIED).

    Never raises on ordinary inputs (missing/invalid/stale/insufficient
    motion hints, no prior record) — every one of those is a valid,
    non-exceptional NOT_APPLIED outcome.
    """
    if previous_record is None or previous_record.depth_snapshot_m is None:
        return previous_record, RotationCompensationStatus.NOT_APPLIED

    accepted = select_motion_hint_samples(motion_hints, previous_timestamp, current_timestamp)
    if not accepted:
        return previous_record, RotationCompensationStatus.NOT_APPLIED

    delta_r = integrate_angular_velocity(accepted, previous_timestamp)

    compensated_snapshot = compensate_prior_geometry(
        previous_record.depth_snapshot_m, delta_r, stride, focal_length_px, principal_point_px,
    )
    compensated_record = dataclasses.replace(previous_record, depth_snapshot_m=compensated_snapshot)
    return compensated_record, RotationCompensationStatus.APPLIED
