"""
Phase I1 — deterministic, ground-truth stereo fixtures for offline stereo/
disparity accuracy characterization. No Gazebo, no network, no randomness
beyond fixed `numpy.random.default_rng(seed)` calls.

Technique: reuses this repository's own proven, already-tested fixture
construction method (tests/test_d10_black_box_provider.py::_engineered_stereo_pair,
tests/test_d12_sensor_contract_independence.py::_smoothed_stereo_pair) —
render a smoothed low-frequency-noise canvas wider than the viewport, crop
the left image directly, and derive the right image by resampling the SAME
canvas at each pixel's true disparity (bilinear interpolation for exact
subpixel shifts, replicate-border for out-of-canvas reads). i.i.d. per-pixel
noise is deliberately NOT used anywhere here — this repo's own D10/D12 work
already found it defeats real StereoSGBM's smoothness-regularized cost
aggregation entirely (median disparity comes back unrelated to true shift).

Real calibration values (examples/config/stereo_calibration.xml), extracted
exactly as tests/test_d10_black_box_provider.py already does:
    FX = 614.5223992233675 px
    BASELINE_M = 0.0647261287661154 m
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

W, H = 320, 240
FX = 614.5223992233675
BASELINE_M = 0.0647261287661154

# Benchmark depths (metres) and their exact required disparity (px) at the
# real calibration above, d = FX * BASELINE_M / Z:
BENCH_DEPTHS_M = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def disparity_for_depth(depth_m: float) -> float:
    return FX * BASELINE_M / depth_m


def depth_for_disparity(disp_px: float) -> float:
    return FX * BASELINE_M / disp_px


# Pad wide enough for the largest benchmark disparity (~79.6px at 0.5m) plus
# margin for any per-fixture max shift used in discontinuity/occlusion cases.
_MAX_SHIFT_MARGIN = 200


def _low_freq_canvas(canvas_w: int, canvas_h: int, texture_scale: int, seed: int) -> np.ndarray:
    """A smoothed grayscale canvas — texture_scale controls spatial
    frequency (smaller = finer/higher texture detail, larger = coarser/
    weaker texture), same low-res-then-cubic-upsample technique
    tests/test_d10_black_box_provider.py::_engineered_stereo_pair uses."""
    rng = np.random.default_rng(seed)
    low_h = max(2, canvas_h // texture_scale + 2)
    low_w = max(2, canvas_w // texture_scale + 2)
    low_res = rng.integers(0, 255, (low_h, low_w), dtype=np.uint8)
    canvas = cv2.resize(low_res, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)
    return canvas.astype(np.float32)


def _remap_by_disparity(canvas: np.ndarray, disp_map: np.ndarray, x0: int) -> np.ndarray:
    """Sample `canvas` to build the right-eye image: right(x,y) =
    canvas(x0 + x + disp_map(x,y), y), bilinear (subpixel-exact).

    Sign convention matches tests/test_d10_black_box_provider.py::
    _engineered_stereo_pair exactly: left(x) = canvas(x0+x), right(x) =
    canvas(x0+x+d) -> the same world column c = x0+x_left appears in the
    right crop at x_right = x_left - d, i.e. x_left - x_right = +d, the
    standard rectified-stereo convention StereoSGBM's minDisparity=0
    (non-negative-only) search assumes. (A first draft of this function
    used `x0 + x - disp_map`, the opposite sign — verified experimentally
    to produce a right image whose true disparity relative to left is
    -d, which minDisparity=0 cannot find at all; caught by the resulting
    disparity/depth errors being absurdly large before any candidate
    config was ever compared against this baseline.)"""
    h, w = disp_map.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (x0 + xs + disp_map).astype(np.float32)
    map_y = ys
    return cv2.remap(canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray.astype(np.uint8)] * 3, axis=-1)


@dataclass
class Fixture:
    name: str
    scenario: str  # 'A'..'G'
    depth_m: Optional[float]
    true_disparity_px: Optional[float]
    left: np.ndarray
    right: np.ndarray
    # ground-truth validity mask: True where a genuine, unambiguous
    # correspondence exists in the constructed scene (used only for the
    # occlusion strip in scenario E and to define "should be invalid"
    # regions elsewhere; None means "the whole frame should be a single
    # uniform valid disparity", the common case for A/B/C/F).
    gt_invalid_mask: Optional[np.ndarray] = None
    # for D/E: ground truth per-region disparity map (H, W), for computing
    # per-pixel expected disparity/depth rather than one scalar.
    gt_disparity_map: Optional[np.ndarray] = None


# Texture scales (canvas-downsample factor before cubic upsample):
# smaller = sharper/higher-frequency texture, larger = smoother/weaker.
_TEXTURE_SCALE = {"A": 2, "B": 6, "C": 24}


def make_flat_fixture(scenario: str, depth_m: float, seed: int) -> Fixture:
    """Scenarios A (high texture), B (moderate), C (weak) — one uniform
    fronto-parallel plane at `depth_m`, ground truth disparity constant
    everywhere."""
    d = disparity_for_depth(depth_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, _TEXTURE_SCALE[scenario], seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]
    disp_map = np.full((H, W), d, dtype=np.float32)
    right_gray = _remap_by_disparity(canvas, disp_map, x0)
    return Fixture(
        name=f"{scenario}_{depth_m}m_seed{seed}", scenario=scenario, depth_m=depth_m,
        true_disparity_px=d, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
    )


def make_repetitive_fixture(depth_m: float, seed: int, period_px: int = 12) -> Fixture:
    """Scenario F — a repeating (periodic) pattern, stresses SGBM's
    uniquenessRatio/ambiguity handling directly (many equally-good matches
    at multiples of `period_px`)."""
    d = disparity_for_depth(depth_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    rng = np.random.default_rng(seed)
    # one random period-wide strip, tiled horizontally -> genuinely periodic
    strip = rng.integers(0, 255, (H, period_px), dtype=np.uint8).astype(np.float32)
    n_tiles = canvas_w // period_px + 2
    canvas = np.tile(strip, (1, n_tiles))[:, :canvas_w]
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]
    disp_map = np.full((H, W), d, dtype=np.float32)
    right_gray = _remap_by_disparity(canvas, disp_map, x0)
    return Fixture(
        name=f"F_{depth_m}m_seed{seed}", scenario="F", depth_m=depth_m,
        true_disparity_px=d, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
    )


def make_discontinuity_fixture(near_m: float, far_m: float, seed: int, occlusion: bool) -> Fixture:
    """Scenario D (occlusion=False) / E (occlusion=True) — a near region
    (left half) and a far region (right half) of the frame at two known
    depths, sharp boundary at the frame's horizontal midline. For E, the
    right-eye strip immediately right of the boundary (of width
    round(d_near - d_far), the true dis-occlusion width for this near/far
    step) is overwritten with independent random content in the RIGHT
    image only — no true correspondence exists there, matching a real
    stereo occlusion (that strip is visible to the far-region's own
    camera geometry only in one eye)."""
    d_near = disparity_for_depth(near_m)
    d_far = disparity_for_depth(far_m)
    canvas_w = W + _MAX_SHIFT_MARGIN
    canvas = _low_freq_canvas(canvas_w, H, _TEXTURE_SCALE["B"], seed)
    x0 = _MAX_SHIFT_MARGIN // 2
    left_gray = canvas[:, x0:x0 + W]

    boundary_col = W // 2
    disp_map = np.full((H, W), d_far, dtype=np.float32)
    disp_map[:, :boundary_col] = d_near
    right_gray = _remap_by_disparity(canvas, disp_map, x0)

    gt_invalid = np.zeros((H, W), dtype=bool)
    scenario = "D"
    if occlusion:
        scenario = "E"
        strip_w = max(1, int(round(d_near - d_far)))
        occ_rng = np.random.default_rng(seed + 10_000)
        # Strip is on the near side, immediately left of the boundary —
        # in the RIGHT image these columns should show what the far
        # surface (now revealed) would look like, which the left image's
        # near-surface occludes; here we instead inject uncorrelated noise
        # to remove genuine correspondence, and mark ground truth invalid.
        c0 = max(0, boundary_col - strip_w)
        right_gray = right_gray.copy()
        right_gray[:, c0:boundary_col] = occ_rng.integers(0, 255, (H, boundary_col - c0), dtype=np.uint8)
        gt_invalid[:, c0:boundary_col] = True

    return Fixture(
        name=f"{scenario}_{near_m}m-{far_m}m_seed{seed}", scenario=scenario, depth_m=None,
        true_disparity_px=None, left=_to_bgr(left_gray), right=_to_bgr(right_gray),
        gt_invalid_mask=gt_invalid, gt_disparity_map=disp_map,
    )


def make_decorrelated_fixture(seed: int) -> Fixture:
    """Scenario G — independent i.i.d. noise in left and right eyes, zero
    true correspondence anywhere. Ground truth: every pixel should be
    reported invalid; any pixel reporting a confident valid disparity is a
    false-valid."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 255, (H, W), dtype=np.uint8)
    right = rng.integers(0, 255, (H, W), dtype=np.uint8)
    return Fixture(
        name=f"G_decorrelated_seed{seed}", scenario="G", depth_m=None, true_disparity_px=None,
        left=_to_bgr(left), right=_to_bgr(right), gt_invalid_mask=np.ones((H, W), dtype=bool),
    )


def build_all_fixtures(seeds=(1, 2, 3)):
    fixtures = []
    for scenario in ("A", "B", "C"):
        for depth_m in BENCH_DEPTHS_M:
            for seed in seeds:
                fixtures.append(make_flat_fixture(scenario, depth_m, seed))
    for depth_m in BENCH_DEPTHS_M:
        for seed in seeds:
            fixtures.append(make_repetitive_fixture(depth_m, seed))
    # D/E: near region fixed near-ish, far region swept across bench depths
    # (near must stay closer than far and inside the searchable disparity
    # range; use 1.0m near, sweep far over depths > 1.0m).
    for far_m in [2.0, 3.0, 4.0, 5.0, 6.0]:
        for seed in seeds:
            fixtures.append(make_discontinuity_fixture(1.0, far_m, seed, occlusion=False))
            fixtures.append(make_discontinuity_fixture(1.0, far_m, seed, occlusion=True))
    for seed in (1, 2, 3, 4, 5):
        fixtures.append(make_decorrelated_fixture(seed))
    return fixtures


if __name__ == "__main__":
    fx_check = disparity_for_depth(1.0)
    print(f"FX={FX}, BASELINE_M={BASELINE_M}")
    for z in BENCH_DEPTHS_M:
        print(f"  depth={z:>4.1f}m -> true disparity={disparity_for_depth(z):.3f}px")
    fs = build_all_fixtures()
    print(f"Built {len(fs)} fixtures total.")
