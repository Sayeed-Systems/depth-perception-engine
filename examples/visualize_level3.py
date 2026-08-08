"""
Level 3 visual validation tool — Level 3, Phase E7 (Tasks 4, 5, 6).

Standalone, outside the core engine — the core library
(src/depth_perception_engine/) never imports matplotlib, and this script
never imports rclpy/sensor_msgs/cv_bridge either. Requires the optional
"viz" extra: `pip install -e ".[viz]"` (see pyproject.toml). The core
library and its test suite work with zero extras installed — this script
is the only thing in the repository that needs matplotlib.

Produces one multi-panel figure from a single stereo pair (either two
saved .npy arrays, or one live single-shot capture with --live):

    1. Left camera image
    2. Disparity visualization
    3. Metric depth visualization
    4. 3D BODY-frame point cloud, with explicit BODY origin + axes triad
       (X forward, Y left, Z up — the exact convention frozen in
       docs/COORDINATE_FRAMES.md), the camera's own position marked,
       ObstacleCloud points, and a sampled subset of FreeSpaceRays
    5. Telemetry panel (confidence, geometry quality, latency, counts)

Coordinate handling (Task 5): matplotlib's mplot3d does not impose any
axis convention of its own — it plots whatever (x, y, z) triples it is
given, with no forced "up" axis. So BODY_X/Y/Z are plotted directly onto
matplotlib's x/y/z with NO data conversion or axis flip anywhere in this
script — verified by inspection, not assumed. The only thing chosen for
aesthetics is the *camera view angle* (`ax.view_init`), which changes
how the same, unmodified data is looked at, not the data itself; this
distinction is deliberate, per this task's "do not silently flip axes
for aesthetics" instruction.

Free/occupied/unknown semantics (Task 6): SURFACE/OCCUPIED evidence is
drawn as solid markers (from the real ObstacleCloud, unmodified). FREE
observed space is drawn as line segments (from the real FreeSpaceRays,
unmodified) from the camera origin to each ray's endpoint. UNKNOWN space
is drawn as nothing at all — no marker, no line, no fabricated geometry
— with an explicit caption stating so, so an absence of drawing is never
mistaken for an absence of information.

The body-frame extrinsic used here is illustrative/synthetic, exactly
like every E4-E6 test fixture — see docs/COORDINATE_FRAMES.md's "real
hardware replacement workflow" section. Swap _illustrative_transform()
for a real measured extrinsic to visualize a real airframe's geometry;
no other change to this script is needed.

Run:
    pip install -e ".[viz]"
    python examples/visualize_level3.py --live                    # one real camera frame
    python examples/visualize_level3.py --left L.npy --right R.npy  # saved frames
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")  # headless-safe by default; --show switches to an interactive backend
import matplotlib.pyplot as plt
import numpy as np

from depth_perception_engine import DepthPerceptionPipeline, PipelineConfig, load_stereo_calibration
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import classify_geometry_quality

_CALIBRATION_FILE = "examples/config/stereo_calibration.xml"


def _illustrative_transform() -> RigidTransform:
    """Synthetic, illustrative only — NOT a measured MP01 extrinsic. See
    this module's docstring and docs/COORDINATE_FRAMES.md."""
    angle = np.deg2rad(10.0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    translation = np.array([0.10, 0.0, 0.05])  # camera 10cm forward, 5cm up, from BODY origin
    return RigidTransform(
        rotation=rotation, translation=translation,
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


def _capture_live_pair(calibration):
    """Single-shot real capture — camera I/O lives here, in examples/,
    never in the core engine. Mirrors examples/live_demo.py's own
    camera-opening approach (this USB stereo camera needs MJPEG off)."""
    import cv2

    width, height = calibration.image_size
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 0.")
    try:
        for _ in range(10):  # discard a few frames for auto-exposure to settle
            cap.read()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read a frame from the camera.")
    finally:
        cap.release()

    frame_h, frame_w = frame.shape[:2]
    half_w = frame_w // 2
    left, right = frame[:, :half_w], frame[:, half_w:]
    return left, right


def _draw_axes_triad(ax, origin, rotation=np.eye(3), length=0.15, label_prefix=""):
    colors = {"X": "red", "Y": "green", "Z": "blue"}
    for axis_name, axis_vec, color in zip("XYZ", np.eye(3), (colors["X"], colors["Y"], colors["Z"])):
        direction = rotation @ axis_vec
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            length=length, color=color, linewidth=2,
        )
        tip = origin + direction * length * 1.15
        ax.text(tip[0], tip[1], tip[2], f"{label_prefix}{axis_name}", color=color, fontsize=9)


def build_figure(left_image: np.ndarray, result, transform, output_path: str) -> None:
    fig = plt.figure(figsize=(20, 11), constrained_layout=True)

    ax_left = fig.add_subplot(2, 3, 1)
    left_rgb = left_image[:, :, ::-1] if left_image.ndim == 3 else left_image  # BGR -> RGB for display only
    ax_left.imshow(left_rgb, cmap=None if left_image.ndim == 3 else "gray")
    ax_left.set_title("Left image")
    ax_left.axis("off")

    ax_disp = fig.add_subplot(2, 3, 2)
    disp_vis = np.ma.masked_where(result.disparity_map <= 0, result.disparity_map)
    im_disp = ax_disp.imshow(disp_vis, cmap="viridis")
    ax_disp.set_title("Disparity (px) — invalid pixels blank")
    fig.colorbar(im_disp, ax=ax_disp, fraction=0.046)
    ax_disp.axis("off")

    ax_depth = fig.add_subplot(2, 3, 3)
    depth_vis = np.ma.masked_where(result.depth_map <= 0, result.depth_map)
    im_depth = ax_depth.imshow(depth_vis, cmap="plasma")
    ax_depth.set_title("Metric depth (m) — invalid pixels blank")
    fig.colorbar(im_depth, ax=ax_depth, fraction=0.046)
    ax_depth.axis("off")

    ax3d = fig.add_subplot(2, 3, 4, projection="3d")
    ax3d.set_title("BODY-frame geometry")
    ax3d.set_xlabel("X — forward (m)")
    ax3d.set_ylabel("Y — left (m)")
    ax3d.set_zlabel("Z — up (m)")

    body_origin = np.zeros(3)
    _draw_axes_triad(ax3d, body_origin, length=0.4, label_prefix="")
    ax3d.scatter(*body_origin, color="black", marker="o", s=60, label="BODY origin")

    camera_position = transform.translation
    ax3d.scatter(*camera_position, color="orange", marker="^", s=80, label="camera position")

    if result.obstacle_cloud is not None and result.obstacle_cloud.points.shape[0] > 0:
        # Plot-time-only subsampling (never touches the engine's own
        # output arrays) — purely to keep the figure legible; the full,
        # unmodified counts are reported in the telemetry panel.
        pts = result.obstacle_cloud.points
        stride = max(1, pts.shape[0] // 3000)
        ax3d.scatter(
            pts[::stride, 0], pts[::stride, 1], pts[::stride, 2],
            color="crimson", marker=".", s=4, alpha=0.6, label="ObstacleCloud (SURFACE/OCCUPIED)",
        )

    if result.free_space_rays is not None and result.free_space_rays.ranges_m.shape[0] > 0:
        rays = result.free_space_rays
        n = rays.ranges_m.shape[0]
        sample_n = min(150, n)
        idx = np.linspace(0, n - 1, sample_n).astype(int)
        endpoints = rays.origins + rays.directions * rays.ranges_m[:, np.newaxis]
        for i in idx:
            ax3d.plot(
                [rays.origins[i, 0], endpoints[i, 0]],
                [rays.origins[i, 1], endpoints[i, 1]],
                [rays.origins[i, 2], endpoints[i, 2]],
                color="mediumseagreen", linewidth=0.4, alpha=0.35,
            )
        ax3d.plot([], [], color="mediumseagreen", label="FreeSpaceRays (FREE, sampled)")

    ax3d.legend(loc="upper left", fontsize=7)
    ax3d.text2D(
        0.02, -0.08,
        "Blank/undrawn regions = UNKNOWN (no valid measurement) — never rendered as free or occupied.",
        transform=ax3d.transAxes, fontsize=8, color="dimgray",
    )
    ax3d.view_init(elev=20, azim=-60)  # camera viewing angle only — no data transformed

    ax_telemetry = fig.add_subplot(2, 3, 5)
    ax_telemetry.axis("off")
    quality = None
    if result.geometry_metrics is not None:
        quality = classify_geometry_quality(result.geometry_metrics, 0.5, 0.05)
    lines = [
        f"confidence: {result.confidence:.3f}",
        f"processing_time_ms: {result.processing_time_ms:.2f}",
        f"navigation decision: {result.traversability_mask.decision.value}",
        "",
        f"valid_disparity: {int(result.valid_disparity_mask.sum())} / {result.valid_disparity_mask.size} "
        f"({100.0 * result.valid_disparity_mask.mean():.1f}%)",
        f"valid_depth: {int(result.valid_depth_mask.sum())} / {result.valid_depth_mask.size} "
        f"({100.0 * result.valid_depth_mask.mean():.1f}%)",
    ]
    if result.geometry_metrics is not None:
        m = result.geometry_metrics
        lines += [
            "",
            f"geometry valid_fraction: {m.valid_fraction:.3f}",
            f"geometry point_count: {m.point_count}",
            f"min_obstacle_distance_m: {m.min_obstacle_distance_m}",
            f"mean_free_space_m: {m.mean_free_space_m}",
            f"quality: {quality}",
        ]
    if result.obstacle_cloud is not None:
        lines.append(f"obstacle_cloud points: {result.obstacle_cloud.points.shape[0]}")
    if result.free_space_rays is not None:
        lines.append(f"free_space_rays: {result.free_space_rays.ranges_m.shape[0]}")
    ax_telemetry.text(0.0, 1.0, "\n".join(lines), va="top", fontfamily="monospace", fontsize=9)
    ax_telemetry.set_title("Telemetry")

    fig.suptitle("Level 3 geometry — one frame", fontsize=13)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Capture one frame from the real camera.")
    parser.add_argument("--left", type=str, default=None, help="Path to a saved left image (.npy).")
    parser.add_argument("--right", type=str, default=None, help="Path to a saved right image (.npy).")
    parser.add_argument("--output", type=str, default="/tmp/level3_visualization.png")
    args = parser.parse_args()

    calibration = load_stereo_calibration(_CALIBRATION_FILE)

    if args.live:
        left, right = _capture_live_pair(calibration)
    elif args.left and args.right:
        left, right = np.load(args.left), np.load(args.right)
    else:
        raise SystemExit("Provide --live, or both --left and --right.")

    config = PipelineConfig(enable_geometry=True, enable_obstacle_geometry=True, enable_free_space_rays=True)
    transform = _illustrative_transform()
    pipeline = DepthPerceptionPipeline(config, calibration, body_T_camera_left=transform)

    t0 = time.perf_counter()
    result = pipeline.process(left, right)
    print(f"process() latency: {(time.perf_counter() - t0) * 1000.0:.2f} ms")

    build_figure(left, result, transform, args.output)


if __name__ == "__main__":
    main()
