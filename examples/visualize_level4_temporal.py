"""
Level 4 temporal visual-proof tool — Level 4, Phase E8, Part B.

Standalone, outside the core engine, same discipline as
examples/visualize_level3.py: the core library
(src/depth_perception_engine/) never imports matplotlib, and this script
never calls any temporal.* algorithm function directly — every value
drawn here is read straight off a real, public DepthPerceptionResult
produced by the real, unmodified DepthPerceptionPipeline. This script
does not compute temporal consistency, stabilization, rotation
compensation, reliability, or persistence itself — it only visualizes
what process() already returned. Requires the optional "viz" extra:
`pip install -e ".[viz]"` (see pyproject.toml).

Drives the pipeline with a synthetic, deterministic, offline stereo
sequence (no camera needed) designed to exercise every temporal state in
one run — Level 4 Phase E7's own README precedent
(tests/test_adversarial_geometry.py) for how a flat/textureless stereo
pair reliably yields zero valid geometry, and how a small random-textured
patch on a flat background yields a small, controllable set of
occupied cells, are reused directly:

    frame 0: flat            -> nothing occupied (baseline)
    frame 1: scene A appears -> NEW
    frame 2: scene A repeats -> PERSISTENT (consistency/stabilization
                                  both become meaningful for the first
                                  time here)
    frame 3: scene A repeats, WITH a simulated MotionHint attached
              -> rotation_compensation_status APPLIED
    frame 4: flat again       -> one dropout frame (persistence survives,
                                  within the configured grace window)
    frame 5: flat again       -> DISAPPEARING (grace window exceeded)
    frame 6: flat again       -> EXPIRED (reverts to NO_EVIDENCE, not FREE)
    frame 7: scene B appears  -> NEW again, unrelated to scene A's history

"Simulated IMU/motion hints", per this phase's own instruction: frame 3's
temporal.MotionHint is constructed directly in this script (a plain
dataclass value, not a hardware reading) — see
docs/LEVEL4_SIMULATED_IMU.md for why a MotionHint's shape is
indistinguishable whether simulated or real.

Run:
    pip install -e ".[viz]"
    python examples/visualize_level4_temporal.py
    python examples/visualize_level4_temporal.py --output /tmp/my_run.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless-safe by default
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.temporal import MotionHint
from depth_perception_engine.temporal.persistence import TemporalPersistenceCellState

_CALIBRATION_FILE = "examples/config/stereo_calibration.xml"

# TemporalPersistenceCellState.NO_EVIDENCE/NEW/PERSISTENT/DISAPPEARING = 0/1/2/3.
_PERSISTENCE_COLORS = ["#2b2b2b", "#e8d34a", "#3aa35a", "#d9622b"]
_PERSISTENCE_LABELS = ["NO_EVIDENCE (=UNKNOWN)", "NEW", "PERSISTENT", "DISAPPEARING"]
_PERSISTENCE_CMAP = ListedColormap(_PERSISTENCE_COLORS)


def _illustrative_transform() -> RigidTransform:
    """Synthetic, illustrative only — NOT a measured MP01 extrinsic, same
    convention examples/visualize_level3.py already established."""
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.05, 0.0, 0.02]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _flat_pair(w, h):
    left = np.full((h, w, 3), 128, dtype=np.uint8)
    return left, left.copy()


def _random_pair(w, h, seed):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return left, right


def _build_sequence(w, h):
    """Returns a list of (label, left, right, timestamp, motion_hints)."""
    scene_a = _random_pair(w, h, seed=11)
    scene_b = _random_pair(w, h, seed=22)
    flat = _flat_pair(w, h)

    # Deliberately tiny: on a synthetic, spatially-uncorrelated random-
    # noise texture (zero real-world autocorrelation between neighboring
    # pixels), even a physically negligible rotation can shift a nearest-
    # neighbor reprojected grid cell onto an unrelated random value and
    # register as a contradiction — a real artifact of using random noise
    # as a stand-in scene, not a bug (see this repository's own
    # tests/test_level4_integration_e8.py::TestRotationCompensationWiring
    # docstring for the same caveat). This magnitude was chosen
    # empirically to stay well inside CONSISTENT/RELIABLE territory for
    # this specific calibration/resolution, so this frame's panel shows a
    # clean APPLIED+RELIABLE illustration rather than an incidental
    # CONTRADICTORY/UNRELIABLE one.
    hint = MotionHint(
        timestamp=3.05, angular_velocity_rad_s=np.array([0.0, 0.001, 0.0]), frame_id=FrameId.BODY,
    )

    return [
        ("flat (baseline)", *flat, 0.0, None),
        ("scene A appears", *scene_a, 1.0, None),
        ("scene A repeats", *scene_a, 2.0, None),
        ("scene A + simulated MotionHint", *scene_a, 3.1, [hint]),
        ("flat (dropout 1)", *flat, 4.0, None),
        ("flat (dropout 2)", *flat, 5.0, None),
        ("flat (dropout 3)", *flat, 6.0, None),
        ("scene B appears", *scene_b, 7.0, None),
    ]


def _panel_depth(ax, depth_map, title):
    vis = np.ma.masked_where(depth_map <= 0.0, depth_map)
    im = ax.imshow(vis, cmap="plasma")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


def _panel_persistence(ax, temporal_persistence):
    ax.set_title("persistence state_grid", fontsize=9)
    ax.axis("off")
    if temporal_persistence is None or temporal_persistence.state_grid is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        return
    ax.imshow(temporal_persistence.state_grid, cmap=_PERSISTENCE_CMAP, vmin=0, vmax=3, interpolation="nearest")


def _telemetry_text(label, timestamp, result):
    tc = result.temporal_consistency
    ts = result.temporal_stabilization
    mar = result.motion_aware_reliability
    tp = result.temporal_persistence

    lines = [
        f"{label}",
        f"t={timestamp:.2f}  admission={result.temporal_admission_status}",
        f"consistency={tc.state if tc else 'N/A'}",
        f"stabilization={ts.state if ts else 'N/A'}",
        f"rotation_compensation={result.rotation_compensation_status}",
        f"reliability={mar.state if mar else 'N/A'}",
        f"persistence={tp.state if tp else 'N/A'}",
        f"  new={tp.new_count} persistent={tp.persistent_count} "
        f"disappearing={tp.disappearing_count} expired={tp.expired_count}" if tp else "",
    ]
    return "\n".join(lines)


def build_figure(rows, output_path: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(15, 2.35 * n), constrained_layout=True)

    for i, (label, timestamp, result) in enumerate(rows):
        _panel_depth(axes[i, 0], result.depth_map, "raw depth_map" if i == 0 else "")
        stabilized = result.temporal_stabilization.stabilized_depth_m if result.temporal_stabilization else None
        if stabilized is not None:
            _panel_depth(axes[i, 1], stabilized, "stabilized_depth_m" if i == 0 else "")
        else:
            axes[i, 1].axis("off")
            axes[i, 1].set_title("stabilized_depth_m" if i == 0 else "", fontsize=9)
            axes[i, 1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[i, 1].transAxes)
        _panel_persistence(axes[i, 2], result.temporal_persistence)

        axes[i, 3].axis("off")
        axes[i, 3].text(
            0.0, 0.5, _telemetry_text(label, timestamp, result),
            va="center", fontfamily="monospace", fontsize=7.5, transform=axes[i, 3].transAxes,
        )

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _PERSISTENCE_COLORS]
    fig.legend(handles, _PERSISTENCE_LABELS, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Level 4 temporal chain — synthetic sequence "
        "(raw geometry -> consistency -> stabilization -> rotation compensation -> reliability -> persistence)",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, default="/tmp/level4_temporal_visualization.png")
    args = parser.parse_args()

    calibration = load_stereo_calibration(_CALIBRATION_FILE)
    w, h = calibration.image_size

    config = PipelineConfig(
        enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True,
        enable_temporal=True, enable_temporal_stabilization=True, enable_rotation_compensation=True,
        enable_motion_aware_reliability=True, enable_temporal_persistence=True,
        persistence_min_support_count=2, persistence_max_dropout_frames=1, persistence_expiration_absence_frames=2,
        temporal_gap_limit_s=5.0, temporal_max_age_s=100.0, temporal_max_records=20,
    )
    pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=_illustrative_transform())

    rows = []
    for label, left, right, timestamp, motion_hints in _build_sequence(w, h):
        result = pipeline.process(left, right, left_timestamp=timestamp, motion_hints=motion_hints)
        rows.append((label, timestamp, result))
        print(f"frame processed: {label:35s} t={timestamp:6.2f}  "
              f"persistence={result.temporal_persistence.state:14s} "
              f"new={result.temporal_persistence.new_count:5d} "
              f"persistent={result.temporal_persistence.persistent_count:5d} "
              f"disappearing={result.temporal_persistence.disappearing_count:5d} "
              f"expired={result.temporal_persistence.expired_count:5d}")

    build_figure(rows, args.output)


if __name__ == "__main__":
    main()
