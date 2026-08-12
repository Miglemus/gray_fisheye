"""Non-regression tests for `scripts/radial_eval.py`.

The whole "the gain lands at the periphery" claim is read off this script, and the script
just grew SSIM, LPIPS and a mask-radius knob. These tests pin it against the two things
that were already trusted before the extension:

  1. the canonical published gray numbers for `tunnel`, which come from a completely
     independent implementation -- `/workspace/gray/scripts/masked_eval.py` writing
     `dataset/fisheye_baselines/masked_metrics.json` on CUDA in float32, through
     `gray.utils.masked_psnr / masked_ssim` and its own `frame_lpips` -- against
     radial_eval's numpy/torch path here. PSNR 28.53749677, SSIM 0.95152459,
     LPIPS 0.15892051;
  2. the multi-camera-rig fix: `masks.json` must give each view the mask of the lens that
     took it, never camera 1's to everybody.

Run:
    python -m pytest tests/test_radial_eval.py -q
`pyproject.toml` sets `addopts = --forked`, so every test is its own process.

Measured agreement (both devices, tunnel, 28 views): PSNR 7.0e-7, SSIM 2.0e-7,
LPIPS 4.4e-4. The LPIPS one is not noise and not ours -- see
`test_lpips_gap_to_canonical_is_the_shared_pass_bbox_cache`, which reproduces the published
number to 3e-9 once the canonical pass's (buggy) shared crop is adopted.

⚠ The SSIM/LPIPS tests use CUDA when it is available, so run the suite THROUGH PUEUE, or
force `RADIAL_EVAL_DEVICE=cpu` (whole file ~5.5 min on CPU against ~2 min on the 11 GB
card; CPU and CUDA agree to 4e-9 on LPIPS, so either is a valid pass).
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import radial_eval  # noqa: E402
from radial_eval import evaluate, find_split, load_masks  # noqa: E402

TUNNEL = "/workspace/gray/out/tunnel_fisheye_baseline"
RIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "out/fullcircle_rttpf/dark_refit_rttpf",
)
CANONICAL = "/workspace/dataset/fisheye_baselines/masked_metrics.json"

# * The published gray tunnel row of the shared masked pass (radius 0.95, 28 test views).
CANONICAL_PSNR = 28.53749677113124
CANONICAL_SSIM = 0.951524589742933
CANONICAL_LPIPS = 0.15892050681369646

needs_tunnel = pytest.mark.skipif(not os.path.isdir(TUNNEL), reason=f"{TUNNEL} absent")
needs_rig = pytest.mark.skipif(not os.path.isdir(RIG), reason=f"{RIG} absent")


def _device():
    forced = os.environ.get("RADIAL_EVAL_DEVICE")
    if forced:
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------
# 1. the canonical number the extension must not move
# ---------------------------------------------------------------------------------------


@needs_tunnel
def test_psnr_reproduces_the_canonical_disk_number():
    """28.537 on out/tunnel_fisheye_baseline -- the published gray tunnel score.

    radial_eval sums squared error in float64 numpy; the canonical pass does it in float32
    on CUDA. 1e-4 dB is three orders of magnitude below the +-0.06 dB run-to-run noise and
    the two agree to 7e-7 in practice.
    """
    result = evaluate(TUNNEL, 6)
    assert result["views"] == 28
    assert abs(result["disk_per_view_mean"] - CANONICAL_PSNR) < 1e-4


@needs_tunnel
def test_canonical_constants_still_match_the_shared_table():
    """Guard against the canonical store being rewritten under us (it has happened)."""
    if not os.path.exists(CANONICAL):
        pytest.skip(f"{CANONICAL} absent")
    with open(CANONICAL) as handle:
        row = json.load(handle)["tunnel"]["gray"]
    assert row["n"] == 28
    assert abs(row["psnr"] - CANONICAL_PSNR) < 1e-9
    assert abs(row["ssim"] - CANONICAL_SSIM) < 1e-12
    assert abs(row["lpips"] - CANONICAL_LPIPS) < 1e-12


@needs_tunnel
def test_adding_metrics_does_not_move_psnr():
    """The PSNR path must be byte-identical whether or not SSIM/LPIPS are requested."""
    plain = evaluate(TUNNEL, 6, metrics=("psnr",))
    with_ssim = evaluate(TUNNEL, 6, metrics=("psnr", "ssim"), device="cpu")
    assert with_ssim["disk_per_view_mean"] == plain["disk_per_view_mean"]
    assert with_ssim["rings"] == plain["rings"]
    assert with_ssim["metrics"]["psnr"]["rings"] == plain["metrics"]["psnr"]["rings"]


@needs_tunnel
def test_psnr_rings_partition_the_disk():
    """Recombining the ring MSEs by pixel count must give the pooled disk PSNR back."""
    result = evaluate(TUNNEL, 6)
    entry = result["metrics"]["psnr"]
    squared = sum(
        10.0 ** (-value / 10.0) * count
        for value, count in zip(entry["rings"], entry["ring_counts"])
    )
    pooled = -10.0 * np.log10(squared / sum(entry["ring_counts"]))
    assert abs(pooled - entry["disk_pooled"]) < 1e-9


# ---------------------------------------------------------------------------------------
# 2. SSIM
# ---------------------------------------------------------------------------------------


@needs_tunnel
def test_ssim_frame_convention_reproduces_canonical():
    """`frame_per_view_mean` IS gray.utils.masked_ssim, so it must hit the published row."""
    result = evaluate(TUNNEL, 6, metrics=("ssim",), device=_device())
    assert abs(result["metrics"]["ssim"]["frame_per_view_mean"] - CANONICAL_SSIM) < 1e-4


@needs_tunnel
def test_ssim_disk_is_far_below_the_frame_convention():
    """The published SSIM is inflated by the black surround -- pin the size of the gap.

    On a circular-frame fisheye 55.8 % of the frame is outside the lens disk and is zeroed
    in BOTH images, so it scores ~1.0. Measured: 0.9515 frame vs 0.8929 disk. Anyone who
    compares a ring SSIM against the published 0.9515 is comparing two different things.
    """
    result = evaluate(TUNNEL, 6, metrics=("ssim",), device=_device())
    entry = result["metrics"]["ssim"]
    assert entry["frame_per_view_mean"] - entry["disk_pooled"] > 0.05


@needs_tunnel
def test_ssim_rings_partition_the_disk():
    result = evaluate(TUNNEL, 6, metrics=("ssim",), device=_device())
    entry = result["metrics"]["ssim"]
    pooled = sum(v * c for v, c in zip(entry["rings"], entry["ring_counts"]))
    pooled /= sum(entry["ring_counts"])
    assert abs(pooled - entry["disk_pooled"]) < 1e-9


# ---------------------------------------------------------------------------------------
# 3. LPIPS
# ---------------------------------------------------------------------------------------


@needs_tunnel
def test_lpips_frame_reproduces_canonical():
    """`frame_per_view_mean` IS scripts/masked_eval.py:frame_lpips -- to 4.4e-4, not to 0.

    The residual is NOT float noise (CPU and CUDA agree here to 4e-9, measured). It is a
    bug in the canonical pass, localised exactly by
    `test_lpips_gap_to_canonical_is_the_shared_pass_bbox_cache` below: masked_eval.py keys
    `_bbox_cache` on `mask.shape`, and all four myscenes scenes render at 1368x912, so
    every scene is cropped to whichever scene ran FIRST -- atrium. This script uses each
    scene's own bounding box, which is the correct thing and is why it lands 4.4e-4 away.
    """
    result = evaluate(TUNNEL, 6, metrics=("lpips",), device=_device())
    assert abs(result["metrics"]["lpips"]["frame_per_view_mean"] - CANONICAL_LPIPS) < 1e-3
    # * and pin our own, correct value tightly so it cannot drift under cover of that slack
    assert abs(result["metrics"]["lpips"]["frame_per_view_mean"] - 0.15847896) < 1e-6


@needs_tunnel
def test_lpips_gap_to_canonical_is_the_shared_pass_bbox_cache():
    """Reproduce the canonical LPIPS EXACTLY by adopting the crop it actually used.

    `/workspace/gray/scripts/masked_eval.py:118` is `_bbox_cache = {}` keyed by
    `key = mask.shape`. `main()` iterates SCENES = [atrium, tunnel, library, reception] and
    every one of them is 1368x912, so atrium's disk bbox (40, 876, 266, 1102) is reused for
    the other three, whose own bboxes are 1-2 px different. Feeding atrium's bbox to this
    script's `lpips_map` returns the published tunnel number to 3e-9 -- which proves the
    two implementations compute the same function and that the whole gap is the crop.

    The bias is shared by every method within the run (one cache entry for all of them), so
    cross-method LPIPS *rankings* survive; the absolute values do not, and they would move
    if anyone reordered SCENES.
    """
    import torch
    from PIL import Image

    device = _device()
    renders, ground_truth, mask_path, parent = find_split(TUNNEL)
    view_masks, _ = load_masks(parent, mask_path, renders)
    mask = torch.from_numpy(view_masks[0].copy()).to(device)
    atrium_bbox = (40, 876, 266, 1102)

    def load(path):
        array = np.asarray(Image.open(path).convert("RGB")).copy()
        return torch.from_numpy(array).permute(2, 0, 1).to(device).float() / 255.0

    values = [
        radial_eval.lpips_map(load(r), load(g), mask, atrium_bbox, device)[1]
        for r, g in zip(renders, ground_truth)
    ]
    assert abs(float(np.mean(values)) - CANONICAL_LPIPS) < 1e-7


@needs_tunnel
def test_lpips_map_integrates_back_to_the_scalar():
    """The ring decomposition must sum back to the canonical scalar, not merely near it."""
    result = evaluate(TUNNEL, 6, metrics=("lpips",), device=_device())
    entry = result["metrics"]["lpips"]
    assert entry["lpips_map_residual"] < 1e-6
    # * the constant that had to be applied to make that true: small, but not zero, and
    # * worth watching -- it grows if a crop size becomes badly divisible by 16.
    assert entry["lpips_resample_correction"] < 1e-2


@needs_tunnel
def test_lpips_matches_piq_on_a_single_view():
    """Independent check of the map machinery against piq's own LPIPS object."""
    import torch
    from PIL import Image
    from piq import LPIPS

    device = _device()
    renders, ground_truth, mask_path, parent = find_split(TUNNEL)
    view_masks, _ = load_masks(parent, mask_path, renders)
    mask = torch.from_numpy(view_masks[0].copy()).to(device)
    rows, cols = np.where(view_masks[0])
    bbox = (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)

    def load(path):
        array = np.asarray(Image.open(path).convert("RGB")).copy()
        return torch.from_numpy(array).permute(2, 0, 1).to(device).float() / 255.0

    render, gt = load(renders[0]), load(ground_truth[0])
    _, scalar, _ = radial_eval.lpips_map(render, gt, mask, bbox, device)

    y0, y1, x0, x1 = bbox
    m = mask.unsqueeze(0).float()
    reference = LPIPS(reduction="none").to(device)
    with torch.no_grad():
        expected = reference(
            (render * m)[:, y0:y1, x0:x1].unsqueeze(0), (gt * m)[:, y0:y1, x0:x1].unsqueeze(0)
        ).item()
    assert abs(scalar - expected) < 1e-6


# ---------------------------------------------------------------------------------------
# 3 bis. the published ring rows, reproduced from the exact run pairs
# ---------------------------------------------------------------------------------------

# * IMPLEMENTATION.md "Where the gain lands" quotes two ring rows. Neither records which
# * run pair produced it, and the pairs are NOT the ones the attribution table uses -- the
# * tunnel row is scored against `tmp/r4/tunnel_off` (28.483), not against the published
# * `out/tunnel_fisheye_baseline` (28.537). Pinning both here so the provenance stops being
# * folklore: if a future change to this script moves either row, it is this script that
# * moved, not the runs.
PUBLISHED_RINGS = {
    "tunnel": (
        "/workspace/gray/tmp/r4/tunnel_off",
        "/workspace/gray/tmp/final/tunnel_noncentral",
        [0.399, 0.130, 0.191, 0.161, 0.231, 0.206],
        0.246,
    ),
    "workshop": (
        "/workspace/gray/tmp/noncentral/fix15k_workshop",
        "/workspace/gray/tmp/final/workshop_noncentral",
        [0.648, 0.938, 0.933, 0.561, 0.663, 1.305],
        0.743,
    ),
}


@pytest.mark.parametrize("scene", sorted(PUBLISHED_RINGS))
def test_published_ring_rows_reproduce(scene):
    off_dir, nc_dir, expected, expected_disk = PUBLISHED_RINGS[scene]
    if not (os.path.isdir(off_dir) and os.path.isdir(nc_dir)):
        pytest.skip(f"{off_dir} / {nc_dir} absent")
    off, nc = evaluate(off_dir, 6), evaluate(nc_dir, 6)
    deltas = [b - a for a, b in zip(off["rings"], nc["rings"])]
    assert deltas == pytest.approx(expected, abs=1e-3)
    # * the row's last column is the PER-VIEW disk mean, not the pooled one.
    disk = nc["disk_per_view_mean"] - off["disk_per_view_mean"]
    assert disk == pytest.approx(expected_disk, abs=1e-3)


# ---------------------------------------------------------------------------------------
# 4. the multi-camera-rig trap that was already paid for once
# ---------------------------------------------------------------------------------------


@needs_rig
def test_rig_views_get_their_own_lens_mask():
    """masks.json must be honoured: two lenses, two disks, and they really do differ.

    The bug this pins: `valid_mask.png` is only the FIRST camera's. Applying it to every
    view counts invalid pixels at the rim -- exactly where the camera residual acts. The
    bias is identical in every run so it cancels in a rung-vs-rung delta and corrupts only
    the absolute number, which is precisely why it survived unnoticed.
    """
    renders, _, mask_path, parent = find_split(RIG)
    assert os.path.exists(os.path.join(parent, "masks.json"))
    view_masks, _ = load_masks(parent, mask_path, renders)
    distinct = {id(m): m for m in view_masks}
    assert len(distinct) == 2, "the rig collapsed back to one mask"
    first, second = list(distinct.values())
    assert (first ^ second).sum() > 1000, "the two lens disks came out identical"

    result = evaluate(RIG, 6)
    assert result["distinct_masks"] == 2


@needs_rig
def test_rig_rebuilt_masks_agree_with_the_saved_ones():
    """The radius knob must resolve per-camera intrinsics, not just camera 1's."""
    saved = evaluate(RIG, 6, mask_radius="saved")
    rebuilt = evaluate(RIG, 6, mask_radius="0.95")
    assert rebuilt["distinct_masks"] == 2
    assert rebuilt["disk_per_view_mean"] == saved["disk_per_view_mean"]


# ---------------------------------------------------------------------------------------
# 5. the mask-radius knob
# ---------------------------------------------------------------------------------------


@needs_tunnel
def test_radius_095_rebuilds_the_saved_mask_bit_for_bit():
    """The sweep is self-checking: its canonical rung must reproduce the saved masks.

    render.py wrote these masks with `fisheye_mask_radius_scale = 0.95` from the same
    `geometric_valid_mask_*` code, so `--mask-radius 0.95` has to land on the identical
    boolean array -- and therefore the identical PSNR, to the last bit.
    """
    saved = evaluate(TUNNEL, 6, mask_radius="saved")
    rebuilt = evaluate(TUNNEL, 6, mask_radius="0.95")
    assert rebuilt["disk_per_view_mean"] == saved["disk_per_view_mean"]
    assert rebuilt["valid_fraction"] == saved["valid_fraction"]
    assert rebuilt["rings"] == saved["rings"]


@needs_tunnel
def test_radius_sweep_widens_the_disk_monotonically():
    fractions = [
        evaluate(TUNNEL, 6, mask_radius=r)["valid_fraction"] for r in ("0.85", "0.95", "1.00")
    ]
    assert fractions[0] < fractions[1] < fractions[2]


@needs_tunnel
def test_ring_edge_fraction_flags_the_outer_ring():
    """The diagnostic that stops the contaminated outer SSIM/LPIPS ring being over-read."""
    result = evaluate(TUNNEL, 6)
    edge = result["ring_edge_fraction_5px"]
    # * measured on tunnel: [0, 0, 0, 0, 0, 0.139] -- only the rim ring touches the void.
    assert all(value == 0.0 for value in edge[:-1])
    assert edge[-1] > 0.05
