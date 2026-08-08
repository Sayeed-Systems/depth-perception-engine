"""
Synthetic ground-truth acceptance — Level 3, Phase E7 (Task 2).

Deterministic scenarios where the correct geometry is known exactly in
advance (hand-computed, not derived from the code under test), run
through the real, unmodified geometry chain: DepthEstimator ->
PointCloudBuilder -> transform_point_cloud -> build_obstacle_cloud ->
build_free_space_rays -> build_geometry_metrics -> classify_geometry_quality.
Uses the repository's real hardware calibration (examples/config/
stereo_calibration.xml) so results are tied to an actual rig, not an
arbitrary synthetic Q.

Each test's docstring states what is being verified and the numeric
tolerance used. Run with `-v` (or `-s` for the printed values in
TestScenario1FlatFrontoParallelPlane) to see individual PASS/FAIL results
— see docs/VALIDATION_REPORT.md's E7 addendum for the captured summary.
"""

import numpy as np
import pytest

from depth_perception_engine import load_stereo_calibration
from depth_perception_engine.depth.depth_estimator import DepthEstimator
from depth_perception_engine.frames import FrameId, RigidTransform
from depth_perception_engine.geometry import (
    GeometryQuality,
    PointCloudBuilder,
    build_free_space_rays,
    build_geometry_metrics,
    build_obstacle_cloud,
    classify_geometry_quality,
    transform_point_cloud,
)
_CALIBRATION = load_stereo_calibration("examples/config/stereo_calibration.xml")
_FX = 614.5223992233675
_BASELINE_M = 0.0647261287661154
_CX = 155.57466888427734
_CY = 185.47198152542114
_TOL_M = 1e-3  # 1 mm — generous relative to float32 + mm->m round trip


def _expected_depth_m(disparity_px: float) -> float:
    return (_FX * _BASELINE_M) / disparity_px


def _builder():
    return PointCloudBuilder.from_calibration(_CALIBRATION)


def _transform():
    return RigidTransform(
        rotation=np.eye(3), translation=np.array([0.10, -0.02, 0.05]),
        from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
    )


class TestScenario1FlatFrontoParallelPlane:
    """A single depth plane, perpendicular to the optical axis, filling
    the whole frame — constant disparity everywhere.

    PASS criteria (tolerance 1e-3 m unless noted):
    - disparity: exactly the configured value at every pixel.
    - metric depth: matches f*B/d within 1e-3 m at every pixel.
    - XYZ surface: Z is constant across the plane (the defining property
      of fronto-parallel); centre pixel's X,Y ~= 0.
    - body-frame transform: centre point's body XYZ == camera XYZ +
      translation exactly (identity rotation chosen for this scenario).
    - obstacle endpoints: every obstacle point's Z equals the plane depth.
    - free-space rays: every ray's reconstructed endpoint matches its
      source point to float32 precision.
    """

    DISPARITY_PX = 50.0

    def test_full_chain_matches_hand_computed_plane_depth(self, capsys):
        h, w = _CALIBRATION.image_size[1], _CALIBRATION.image_size[0]
        disparity = np.full((h, w), self.DISPARITY_PX, dtype=np.float32)
        expected_depth = _expected_depth_m(self.DISPARITY_PX)

        estimator = DepthEstimator.from_calibration(_CALIBRATION)
        depth_map = estimator.estimate(disparity)
        assert np.all(depth_map > 0), "expected every pixel valid for this in-range disparity"
        np.testing.assert_allclose(depth_map, expected_depth, atol=_TOL_M)

        cloud = _builder().build(disparity)
        assert np.all(cloud.valid_mask)
        np.testing.assert_allclose(cloud.points[:, :, 2], expected_depth, atol=_TOL_M)
        cv, cu = int(_CY), int(_CX)
        np.testing.assert_allclose(cloud.points[cv, cu, :2], [0.0, 0.0], atol=_TOL_M)

        transform = _transform()
        body_cloud = transform_point_cloud(cloud, transform)
        expected_center_body = cloud.points[cv, cu] + transform.translation  # identity rotation
        np.testing.assert_allclose(body_cloud.points[cv, cu], expected_center_body, atol=_TOL_M)

        origin = transform.translation
        obstacle_cloud = build_obstacle_cloud(body_cloud, origin, min_range_m=0.0, max_range_m=100.0)
        # Body-frame Z = camera-frame depth + translation.z (identity rotation
        # in this scenario) — not the raw camera-frame depth itself.
        expected_body_z = expected_depth + transform.translation[2]
        np.testing.assert_allclose(obstacle_cloud.points[:, 2], expected_body_z, atol=_TOL_M)

        rays = build_free_space_rays(body_cloud, origin)
        endpoints = rays.origins + rays.directions * rays.ranges_m[:, np.newaxis]
        valid_points_flat = body_cloud.points[body_cloud.valid_mask]
        np.testing.assert_allclose(endpoints, valid_points_flat, atol=1e-4)

        print(f"\n[Scenario 1] disparity={self.DISPARITY_PX}px -> expected_depth={expected_depth:.4f}m, "
              f"measured (centre pixel)={depth_map[cv, cu]:.4f}m, "
              f"obstacle_points={obstacle_cloud.points.shape[0]}, rays={rays.ranges_m.shape[0]} -> PASS")


class TestScenario2NearObjectFarBackground:
    """A near patch and a far background at two distinct, known depths.

    PASS criteria: near depth < far depth (ordering); each region's
    measured depth matches its hand-computed value within 1e-3 m; the
    near patch's free-space rays terminate at the near depth, not the
    far one (no incorrect pass-through)."""

    NEAR_DISPARITY_PX = 100.0
    FAR_DISPARITY_PX = 20.0

    def test_near_far_ordering_and_ray_termination(self, capsys):
        h, w = _CALIBRATION.image_size[1], _CALIBRATION.image_size[0]
        disparity = np.full((h, w), self.FAR_DISPARITY_PX, dtype=np.float32)
        near_rows, near_cols = slice(0, h // 2), slice(0, w // 2)
        disparity[near_rows, near_cols] = self.NEAR_DISPARITY_PX

        expected_near = _expected_depth_m(self.NEAR_DISPARITY_PX)
        expected_far = _expected_depth_m(self.FAR_DISPARITY_PX)
        assert expected_near < expected_far, "fixture must actually be near+far"

        cloud = _builder().build(disparity)
        np.testing.assert_allclose(cloud.points[near_rows, near_cols, 2], expected_near, atol=_TOL_M)
        far_rows, far_cols = slice(h // 2, h), slice(w // 2, w)
        np.testing.assert_allclose(cloud.points[far_rows, far_cols, 2], expected_far, atol=_TOL_M)

        transform = _transform()
        body_cloud = transform_point_cloud(cloud, transform)
        origin = transform.translation
        rays = build_free_space_rays(body_cloud, origin)
        endpoints = rays.origins + rays.directions * rays.ranges_m[:, np.newaxis]

        # Every endpoint Z (camera-forward component survives translation-only
        # transform as a pure offset) must equal one of the two known depths
        # plus the Z-translation — never a blended/incorrect value.
        expected_zs = {round(expected_near + transform.translation[2], 3), round(expected_far + transform.translation[2], 3)}
        actual_zs = {round(float(z), 3) for z in endpoints[:, 2]}
        assert actual_zs.issubset(expected_zs), (
            f"ray endpoints contain unexpected Z values: {actual_zs - expected_zs}"
        )

        print(f"\n[Scenario 2] near={expected_near:.4f}m far={expected_far:.4f}m "
              f"(ordering correct: {expected_near < expected_far}) -> PASS")


class TestScenario3PartialInvalidRegion:
    """A small, exactly-known grid with a known count of valid and
    invalid pixels.

    PASS criteria (exact, not approximate — counts are hand-picked):
    - invalid pixels: valid_mask False, points NaN, exactly at their
      designated positions.
    - obstacle_cloud.points.shape[0] == exact valid pixel count.
    - free_space_rays.ranges_m.shape[0] == exact valid pixel count.
    """

    def test_exact_valid_invalid_counts_through_full_chain(self, capsys):
        disparity = np.full((4, 5), 60.0, dtype=np.float32)  # 20 pixels total
        invalid_positions = [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (3, 0)]  # exactly 6 invalid
        for r, c in invalid_positions:
            disparity[r, c] = 0.0
        expected_valid_count = disparity.size - len(invalid_positions)

        cloud = _builder().build(disparity)
        assert int(cloud.valid_mask.sum()) == expected_valid_count
        for r, c in invalid_positions:
            assert not cloud.valid_mask[r, c]
            assert np.all(np.isnan(cloud.points[r, c]))

        transform = _transform()
        body_cloud = transform_point_cloud(cloud, transform)
        origin = transform.translation
        obstacle_cloud = build_obstacle_cloud(body_cloud, origin, min_range_m=0.0, max_range_m=100.0)
        rays = build_free_space_rays(body_cloud, origin)

        assert obstacle_cloud.points.shape[0] == expected_valid_count
        assert rays.ranges_m.shape[0] == expected_valid_count

        print(f"\n[Scenario 3] total_pixels=20 invalid=6 -> expected_valid={expected_valid_count}, "
              f"obstacle_points={obstacle_cloud.points.shape[0]}, rays={rays.ranges_m.shape[0]} -> PASS")


class TestScenario4KnownBodyTransform:
    """A known, non-trivial rotation (90 degrees about Y) plus known
    translation, applied to a known camera-frame point.

    PASS criteria: body-frame XYZ differs from camera-frame XYZ by
    EXACTLY the hand-computed rotation+translation (atol 1e-4 m,
    accounting for one float32 round trip) — not merely "differs"."""

    def test_camera_and_body_outputs_differ_exactly_as_expected(self, capsys):
        disparity_px = 80.0
        h, w = _CALIBRATION.image_size[1], _CALIBRATION.image_size[0]
        disparity = np.full((h, w), disparity_px, dtype=np.float32)
        cloud = _builder().build(disparity)

        cv, cu = h // 4, w // 4  # an off-centre pixel, so rotation has a visible effect
        camera_point = cloud.points[cv, cu].astype(np.float64)

        angle = np.deg2rad(90.0)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])  # Ry(90): (x,y,z)->(z,y,-x)
        translation = np.array([0.2, 0.1, -0.05])
        transform = RigidTransform(
            rotation=rotation, translation=translation,
            from_frame=FrameId.CAMERA_OPTICAL_LEFT, to_frame=FrameId.BODY,
        )

        body_cloud = transform_point_cloud(cloud, transform)
        body_point = body_cloud.points[cv, cu].astype(np.float64)

        expected_body_point = rotation @ camera_point + translation
        np.testing.assert_allclose(body_point, expected_body_point, atol=1e-4)
        # And explicitly NOT equal to the untransformed camera point —
        # proving the transform actually did something, not a silent no-op.
        assert not np.allclose(body_point, camera_point, atol=1e-3)

        print(f"\n[Scenario 4] camera_point={camera_point}, body_point={body_point}, "
              f"expected={expected_body_point} -> PASS")


class TestScenario5EmptyTexturelessScene:
    """An entirely invalid (zero) disparity map — the synthetic proxy for
    a textureless real scene (SGBM finds no correspondence anywhere).

    PASS criteria: zero valid points, zero obstacle points, zero rays,
    valid_fraction == 0.0, classify_geometry_quality ==
    NO_USABLE_GEOMETRY, and — the core anti-fabrication check — every
    point in the (H, W, 3) array is NaN, never 0.0 or any other
    plausible-looking value.
    """

    def test_no_fabricated_geometry_and_correct_degraded_classification(self, capsys):
        h, w = _CALIBRATION.image_size[1], _CALIBRATION.image_size[0]
        disparity = np.zeros((h, w), dtype=np.float32)

        cloud = _builder().build(disparity)
        assert not np.any(cloud.valid_mask)
        assert np.all(np.isnan(cloud.points))

        transform = _transform()
        body_cloud = transform_point_cloud(cloud, transform)
        origin = transform.translation
        obstacle_cloud = build_obstacle_cloud(body_cloud, origin, min_range_m=0.0, max_range_m=100.0)
        rays = build_free_space_rays(body_cloud, origin)
        metrics = build_geometry_metrics(body_cloud, obstacle_cloud, rays)

        assert obstacle_cloud.points.shape[0] == 0
        assert rays.ranges_m.shape[0] == 0
        assert metrics.valid_fraction == 0.0
        assert metrics.point_count == 0

        quality = classify_geometry_quality(metrics, healthy_min_valid_fraction=0.5, degraded_min_valid_fraction=0.05)
        assert quality == GeometryQuality.NO_USABLE_GEOMETRY

        print(f"\n[Scenario 5] valid_fraction={metrics.valid_fraction}, quality={quality} -> PASS")


class TestScenario6MixedGeometryScene:
    """A realistic composite: valid near obstacle + valid far background
    + an invalid (textureless) region + an out-of-range (too-far, gets
    clamped invalid by DepthEstimator) region, all in one frame.

    PASS criteria: each region is independently, correctly represented —
    valid regions produce obstacle+ray evidence with the right depth;
    invalid and out-of-range regions produce neither; total counts are
    hand-computed exactly.
    """

    def test_all_four_regions_independently_correct(self, capsys):
        h, w = _CALIBRATION.image_size[1], _CALIBRATION.image_size[0]
        quarter_h, quarter_w = h // 2, w // 2
        disparity = np.zeros((h, w), dtype=np.float32)  # start fully invalid (textureless baseline)

        near_disparity_px = 120.0
        far_disparity_px = 15.0
        too_far_disparity_px = (_FX * _BASELINE_M) / (DepthEstimator.MAX_DEPTH_M * 3.0)  # clamped invalid

        # Quadrant layout: TL=near valid, TR=far valid, BL=stays invalid (0.0), BR=too-far (invalid via clamp)
        disparity[0:quarter_h, 0:quarter_w] = near_disparity_px
        disparity[0:quarter_h, quarter_w:w] = far_disparity_px
        disparity[quarter_h:h, quarter_w:w] = too_far_disparity_px

        expected_near = _expected_depth_m(near_disparity_px)
        expected_far = _expected_depth_m(far_disparity_px)
        expected_valid_count = quarter_h * quarter_w + quarter_h * (w - quarter_w)  # TL + TR

        cloud = _builder().build(disparity)
        assert int(cloud.valid_mask.sum()) == expected_valid_count
        assert not np.any(cloud.valid_mask[quarter_h:h, 0:quarter_w])  # BL: textureless, stays invalid
        assert not np.any(cloud.valid_mask[quarter_h:h, quarter_w:w])  # BR: too-far, clamped invalid
        np.testing.assert_allclose(cloud.points[0:quarter_h, 0:quarter_w, 2], expected_near, atol=_TOL_M)
        np.testing.assert_allclose(cloud.points[0:quarter_h, quarter_w:w, 2], expected_far, atol=_TOL_M)

        transform = _transform()
        body_cloud = transform_point_cloud(cloud, transform)
        origin = transform.translation
        obstacle_cloud = build_obstacle_cloud(body_cloud, origin, min_range_m=0.0, max_range_m=100.0)
        rays = build_free_space_rays(body_cloud, origin)
        metrics = build_geometry_metrics(body_cloud, obstacle_cloud, rays)

        assert obstacle_cloud.points.shape[0] == expected_valid_count
        assert rays.ranges_m.shape[0] == expected_valid_count
        expected_fraction = expected_valid_count / (h * w)
        assert metrics.valid_fraction == pytest.approx(expected_fraction, rel=1e-6)

        print(f"\n[Scenario 6] near={expected_near:.4f}m far={expected_far:.4f}m "
              f"valid_count={expected_valid_count}/{h*w} ({expected_fraction:.1%}), "
              f"obstacle_points={obstacle_cloud.points.shape[0]}, rays={rays.ranges_m.shape[0]} -> PASS")
