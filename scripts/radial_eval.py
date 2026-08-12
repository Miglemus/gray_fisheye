"""Radially-binned masked metrics for fisheye runs -- PSNR, SSIM and LPIPS.

Why this exists: a camera-model residual does not act uniformly over the field. A radial
d(theta) residual vanishes at theta = 0 and the non-central term scales as
sin(theta) * z(theta) / depth, so both are peripheral by construction. A single full-disk
PSNR averages that signal away -- +0.2 dB overall is uninformative, +1.5 dB on the outer
ring is a result. gray had no such tooling.

Comparability rules this script enforces:
  * the mask is whatever render.py saved next to the renders (`valid_mask.png`, or the
    per-camera `valid_mask_cam<uid>.png` named by `masks.json`), which is built from the
    COLMAP intrinsics and is therefore identical for every ablation rung -- the learned
    residual never touches `cam_info.intrinsics`;
  * bins are equal-area annuli in *pixel* radius about the COLMAP principal point, i.e. a
    fixed function of the pixel, so ring k means the same pixels in every run;
  * PSNR is pooled over views within a ring (sum of squared error over all valid pixels of
    that ring, then converted once), which is far more stable than averaging per-view dB;
  * SSIM and LPIPS use gray's shared-protocol definitions (`gray.utils.masked_ssim` and
    `scripts/masked_eval.py:frame_lpips`) so the disk-level numbers are directly
    comparable with the canonical cross-method tables.

usage:
  python scripts/radial_eval.py --runs out/a out/b --labels baseline noncentral [--rings 6]
  python scripts/radial_eval.py --runs out/a --metrics psnr ssim lpips --mask-radius 0.85


HOW SSIM AND LPIPS ARE MADE RADIAL (the method note -- read before quoting a ring number)
-----------------------------------------------------------------------------------------
Both are metrics with *spatial support*: SSIM pools an 11x11 gaussian window, LPIPS pools
VGG16 features whose deepest layer (relu5_3) has a receptive field of ~212 px. There are
two ways to make either of them radial:

  (a) crop the image to the annulus and score the crop, or
  (b) compute the metric MAP over the whole (masked) image and average that map over the
      annulus.

This script does (b), deliberately. Cropping to an annulus manufactures two fresh image
borders per ring; SSIM's windows straddle them and LPIPS' features see a hard edge that is
not in either image, so (a) reports ring structure that is an artefact of the cropping.
With (b) every pixel is scored in its true photometric neighbourhood and the ring only
decides which already-computed scores are averaged.

The price of (b), and it is real:

  * a ring value is not independent of its neighbours -- a defect one window (SSIM) or one
    receptive field (LPIPS) outside a ring still moves that ring's number. The ring
    profile is therefore *blurred* by roughly that support. For LPIPS the blur is large
    (relu5_3's receptive field in VGG16 is ~212 px), so a 6-ring LPIPS profile on a 419 px
    disk radius is much smoother than the underlying signal; treat sharp ring-to-ring
    LPIPS steps with suspicion.
  * ⚠ THE OUTERMOST RING IS CONTAMINATED FOR SSIM AND LPIPS, NOT FOR PSNR. Both metrics
    are computed on the mask-zeroed images, so a window/receptive field straddling the rim
    sees the black exterior in BOTH images and scores it as agreement. Measured on
    `out/tunnel_fisheye_baseline`: LPIPS rises monotonically 0.161 -> 0.241 out to ring 4
    and then *falls* to 0.194 on ring 5; SSIM likewise ticks up from 0.8781 to 0.8802. The
    fall is the zeroed surround, not a better reconstruction. `ring_edge_fraction_5px`
    reports how much of each ring lies within one SSIM window of the rim; for LPIPS assume
    the whole outer ring is affected. When quoting "the gain lands at the periphery", quote
    PSNR on the outer ring and use SSIM/LPIPS on rings 0..n-2.
  * the rings of one metric still partition the disk exactly, so the pooled disk value is
    the count-weighted average of the rings (asserted in tests/test_radial_eval.py) --
    except on the pixels each metric cannot score (see `*_support` below).

Support, i.e. which pixels each metric can score at all:
  * PSNR   -- every valid pixel.
  * SSIM   -- piq convolves with `padding=0`, so the map is [H-10, W-10]; the outer 5-pixel
    frame of the image has no SSIM. That never touches the lens disk on these scenes but
    it is reported as `support_fraction` so it cannot bite silently.
  * LPIPS  -- the canonical `frame_lpips` zeroes invalid pixels in BOTH images and then
    crops to the lens disk's bounding box; the map is defined on that crop only.

Two SSIM/LPIPS disk conventions are reported, and they are NOT the same number:
  * `disk_*`  -- the map averaged over VALID pixels only. This is the honest "how good is
    the reconstruction" number and the one the rings decompose.
  * `frame_*` -- the convention of the canonical shared pass. `gray.utils.masked_ssim`
    zeroes the invalid pixels in both images and then averages the SSIM map over the WHOLE
    frame, so the black surround (55.8 % of the frame on myscenes) scores ~1.0 and inflates
    the number; `frame_lpips` averages over the whole bbox crop. `frame_*` is what
    `dataset/*/masked_metrics.json` contains, so it is what a cross-method comparison must
    use -- and it is what the regression test pins. Never mix the two.

⚠ ONE KNOWN DISAGREEMENT WITH THE CANONICAL STORE, AND IT IS THE STORE THAT IS WRONG.
On `tunnel` this script reports frame LPIPS 0.158479 against the published 0.158921
(+4.4e-4). It is not float noise: CPU and CUDA agree here to 4e-9. The cause is
`/workspace/gray/scripts/masked_eval.py:118`, `_bbox_cache = {}` keyed on `mask.shape`.
All four myscenes scenes render at 1368x912, so the bbox computed for whichever scene ran
FIRST (atrium, (40, 876, 266, 1102)) is reused for tunnel, library and reception, whose
own disks differ by 1-2 px. Feeding atrium's bbox to `lpips_map` here reproduces the
published number to 3e-9 (`tests/test_radial_eval.py`). The bias is one constant crop for
every method in the run, so cross-method LPIPS RANKINGS are unaffected; absolute LPIPS
values for tunnel/library/reception are off by ~4e-4 and would move if anyone reordered
`SCENES`. PSNR and SSIM in that store are unaffected (no crop is involved) and this script
reproduces them to 7e-7 and 2e-7.


THE MASK-RADIUS SWEEP IS NOT MEANINGFUL ON EVERY LENS
-----------------------------------------------------------------------------------------
`--mask-radius` moves the theta cutoff, but `geometric_valid_mask_*` also drops any pixel
where the 100-iteration inversion of the distortion polynomial fails to converge
(`err_sq < 1e-5`). On a lens whose polynomial stops being invertible before the cutoff,
the second criterion binds and the radius knob does nothing. Measured valid fraction of
the frame:

  | run                       | r=0.85 | r=0.95 | r=1.00 |
  |---------------------------|--------|--------|--------|
  | tunnel (myscenes rttpf)   | 0.3731 | 0.4420 | 0.4734 |
  | workshop_immervision      | 0.4785 | 0.4785 | 0.4785 |
  | FullCircle dark (rig)     | 0.4826 | 0.6061 | 0.6670 |

On `workshop_immervision` the three radii differ by SEVEN pixels out of 1.56 M: its disk
is inversion-limited at about 76 deg, not cutoff-limited. A "the ranking is stable under
three mask radii" claim is therefore vacuous on that scene -- the three radii ARE the same
mask. Check `valid_fraction` before reading anything into a per-scene robustness row.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# * Canonical shared-protocol constants, kept identical to scripts/masked_eval.py.
SSIM_KERNEL_SIZE = 11
SSIM_KERNEL_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03

METRIC_NAMES = ("psnr", "ssim", "lpips")
# * higher-is-better flag, used only to orient the printed deltas.
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False}


def find_split(run_dir, camera_model="rad_tan_thin_prism_fisheye", split="test"):
    renders = sorted(glob.glob(os.path.join(run_dir, split, "*", camera_model, "renders", "*.png")))
    if not renders:
        raise SystemExit(
            f"no {camera_model} renders under {run_dir}/{split}. "
            "run.sh only renders pinhole -- use scripts/eval_rttpf.sh"
        )
    render_dir = os.path.dirname(renders[0])
    parent = os.path.dirname(render_dir)
    ground_truth = sorted(glob.glob(os.path.join(parent, "gt", "*.png")))
    mask_path = os.path.join(parent, "valid_mask.png")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(os.path.dirname(parent), "valid_mask.png")
    return renders, ground_truth, mask_path, parent


def load_masks(parent, mask_path, renders):
    """(per-view masks, union) -- honouring a multi-camera rig's per-view mask index.

    On a single-camera scene `valid_mask.png` is the mask for every view. On a rig it is
    only the FIRST camera's: render.py also writes `valid_mask_cam<k>.png` plus a
    `masks.json` naming the mask of each render. On FullCircle the two lenses' disks differ
    by 1864 px (0.36 % of the frame) at the rim, so scoring cam2's views against cam1's
    disk quietly counts invalid pixels. That bias is identical in every run and so cancels
    in a rung-vs-rung delta, but it corrupts the absolute number -- and the outer ring is
    exactly where the residual is supposed to act.

    The ring geometry is built from the UNION so that ring k means the same pixels in every
    run and for both cameras; only the per-view validity differs.
    """
    index = os.path.join(parent, "masks.json")
    if os.path.exists(index):
        with open(index) as handle:
            mapping = json.load(handle)
        cache, per_view = {}, []
        for path in renders:
            name = mapping.get(os.path.basename(path))
            if name is None:
                per_view = []
                break
            if name not in cache:
                cache[name] = np.asarray(
                    Image.open(os.path.join(parent, name)).convert("L")) > 127
            per_view.append(cache[name])
        if per_view:
            return per_view, np.logical_or.reduce(list(cache.values()))
    shared = np.asarray(Image.open(mask_path).convert("L")) > 127
    return [shared] * len(renders), shared


def _camera_uid_per_view(run_dir, parent, renders, split):
    """uid of the COLMAP camera that took each render.

    `masks.json` is the authority (it is written from the very list render.py iterated).
    Without it, a run is only unambiguous if the split has ONE camera -- refusing rather
    than guessing is the whole point of the multi-rig fix, so a rig without masks.json is
    an error, not a silent fall back to camera 1.
    """
    index = os.path.join(parent, "masks.json")
    if os.path.exists(index):
        with open(index) as handle:
            mapping = json.load(handle)
        uids = []
        for path in renders:
            name = mapping.get(os.path.basename(path))
            if name is None:
                uids = []
                break
            uids.append(int(name.split("valid_mask_cam")[1].split(".")[0]))
        if uids:
            return uids

    cameras = _load_cameras_json(run_dir)
    want_test = split == "test"
    split_uids = sorted({c["uid"] for c in cameras if bool(c.get("is_test", False)) == want_test})
    if len(split_uids) != 1:
        raise SystemExit(
            f"{parent} has no masks.json and its {split} split spans cameras {split_uids}; "
            "cannot assign a per-view mask without guessing. Re-render with a render.py "
            "that writes masks.json."
        )
    return [split_uids[0]] * len(renders)


def _load_cameras_json(run_dir):
    path = os.path.join(run_dir, "cameras.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- needed to rebuild masks at a custom radius")
    with open(path) as handle:
        return json.load(handle)


def rebuild_masks(run_dir, parent, renders, radius_scale, split, height, width, device="cpu"):
    """Per-view masks regenerated from the run's own intrinsics at an arbitrary radius.

    Validated to be **bit-for-bit** the mask render.py saved when `radius_scale` equals the
    run's `fisheye_mask_radius_scale` (0.95 on every myscenes run) -- so a sweep including
    the canonical radius is self-checking. `cameras.json` stores the intrinsics at the
    training resolution, which is also the render resolution, but the scaling is applied
    anyway so a re-render at another size stays correct.
    """
    import torch

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from gray.fisheye_mask import (  # noqa: E402
        geometric_valid_mask_opencv_fisheye,
        geometric_valid_mask_rad_tan_thin_prism_fisheye,
        geometric_valid_mask_thin_prism_fisheye,
    )

    builders = {
        "opencv_fisheye": geometric_valid_mask_opencv_fisheye,
        "thin_prism_fisheye": geometric_valid_mask_thin_prism_fisheye,
        "rad_tan_thin_prism_fisheye": geometric_valid_mask_rad_tan_thin_prism_fisheye,
    }

    cameras = _load_cameras_json(run_dir)
    by_uid = {}
    for cam in cameras:
        by_uid.setdefault(cam["uid"], cam)

    uids = _camera_uid_per_view(run_dir, parent, renders, split)
    built = {}
    for uid in sorted(set(uids)):
        cam = by_uid[uid]
        model = cam["model"]
        if model not in builders:
            raise SystemExit(
                f"camera {uid} of {run_dir} is `{model}`; a mask-radius sweep only means "
                "something for a fisheye model"
            )
        intr = list(cam["intrinsics"])
        sx, sy = width / cam["image_width"], height / cam["image_height"]
        intr[0] *= sx  # fx
        intr[2] *= sx  # cx
        intr[1] *= sy  # fy
        intr[3] *= sy  # cy
        tensor = torch.tensor(intr, dtype=torch.float32, device=device)
        built[uid] = builders[model](tensor, height, width, device, radius_scale).cpu().numpy()

    per_view = [built[uid] for uid in uids]
    return per_view, np.logical_or.reduce(list(built.values()))


def ring_index(mask, num_rings):
    """Equal-area annuli of the valid disk, indexed by radius about the disk centre.

    Equal area keeps every ring statistically comparable; radius is taken about the disk's
    bounding-box centre, which for these calibrations is the principal point to within a
    fraction of a pixel (myscenes has cx, cy pinned at the exact sensor centre).

    NOTE the geometry follows the mask actually in use, so shrinking `--mask-radius`
    shrinks the annuli too: ring k at r=0.85 is NOT the same pixels as ring k at r=0.95.
    Compare ring tables within a radius, never across. `disk_radius_px` is reported so the
    two can be related.
    """
    rows, cols = np.where(mask)
    centre_y = (rows.min() + rows.max()) / 2.0
    centre_x = (cols.min() + cols.max()) / 2.0
    radius = ((rows.max() - rows.min()) + (cols.max() - cols.min())) / 4.0
    grid_y, grid_x = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    normalized = np.sqrt((grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2) / radius
    edges = [math.sqrt(k / num_rings) for k in range(num_rings + 1)]
    # * The outermost bin must swallow everything past the nominal disk radius (the mask is
    # * not a perfect circle), but the reported edges stay the nominal ones.
    digitize_edges = list(edges)
    digitize_edges[-1] = max(edges[-1], float(normalized.max()) + 1.0)
    return np.digitize(normalized, digitize_edges) - 1, edges, (centre_y, centre_x, radius)


# --------------------------------------------------------------------------------------
# metric maps
# --------------------------------------------------------------------------------------


def ssim_map(render, gt, mask):
    """Per-pixel SSIM map of `gray.utils.masked_ssim`, as a [H, W] array with NaN border.

    `masked_ssim` is `piq.ssim(render*m, gt*m, downsample=False)`, i.e. the mean of the
    per-channel SSIM maps of the mask-zeroed images. This reproduces those maps and
    averages over channels, leaving the reduction to the caller. piq convolves with
    `padding=0`, so the outermost 5 pixels of the frame carry no SSIM and are returned NaN.

    Inputs are torch [3, H, W] in [0, 1] and a bool [H, W] mask, all on the same device.
    """
    import torch
    import torch.nn.functional as F
    from piq.functional import gaussian_filter

    m = mask.unsqueeze(0).to(render.dtype)
    x = (render * m).unsqueeze(0)
    y = (gt * m).unsqueeze(0)

    c1, c2 = SSIM_K1**2, SSIM_K2**2
    channels = x.size(1)
    kernel = gaussian_filter(
        SSIM_KERNEL_SIZE, SSIM_KERNEL_SIGMA, device=x.device, dtype=x.dtype
    ).repeat(channels, 1, 1, 1)

    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=channels)
    mu_xx, mu_yy, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_xx = F.conv2d(x**2, weight=kernel, stride=1, padding=0, groups=channels) - mu_xx
    sigma_yy = F.conv2d(y**2, weight=kernel, stride=1, padding=0, groups=channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=channels) - mu_xy
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    valid = ss.mean(1)[0]  # [H-10, W-10]

    pad = (SSIM_KERNEL_SIZE - 1) // 2
    full = torch.full(mask.shape, float("nan"), dtype=valid.dtype, device=valid.device)
    full[pad : mask.shape[0] - pad, pad : mask.shape[1] - pad] = valid
    return full


_LPIPS = {}


def _lpips_module(device):
    """piq.LPIPS -- the same backend gray's metrics.py and masked_eval.py use."""
    if device not in _LPIPS:
        from piq import LPIPS

        _LPIPS[device] = LPIPS(reduction="none").to(device)
    return _LPIPS[device]


def lpips_map(render, gt, mask, bbox, device):
    """(map, scalar) for the canonical `frame_lpips`, as a [H, W] array with NaN outside.

    The canonical definition (`scripts/masked_eval.py:frame_lpips`) is: zero the invalid
    pixels in BOTH images, crop to the lens disk's bounding box, then LPIPS. LPIPS itself
    is `sum over layers of mean over space of (channel-weighted squared feature
    difference)`. Dropping only the final spatial mean gives a per-layer spatial map; each
    is resampled (nearest, so no new values are invented) to the crop resolution.

    One correction is applied and it matters: VGG16 pools by 2, so a layer at stride 2^k
    has floor(H/2^k) rows and nearest-upsampling it back to H over-weights the last row
    whenever the stride does not divide H (838 px crop on tunnel: the deep layers are off
    by up to 9e-4 relative). Each layer's map is therefore rescaled by the constant that
    restores its exact spatial mean, so `mean(map) == scalar` holds to float precision and
    the ring decomposition sums back to the canonical LPIPS. The correction is a per-layer
    constant -- it cannot move a ring relative to another within a layer -- and its size is
    reported as `lpips_resample_correction`.

    Lower is better, and the map is a *distance*: the ring with the LARGEST value is the
    worst ring.
    """
    import torch
    import torch.nn.functional as F

    y0, y1, x0, x1 = bbox
    m = mask.unsqueeze(0).to(render.dtype)
    x = (render * m)[:, y0:y1, x0:x1].unsqueeze(0)
    y = (gt * m)[:, y0:y1, x0:x1].unsqueeze(0)

    module = _lpips_module(device)
    with torch.no_grad():
        x_features = module.get_features(x)
        y_features = module.get_features(y)
        distances = module.compute_distance(x_features, y_features)
        scalar = 0.0
        correction = 0.0
        crop = torch.zeros(x.shape[-2:], dtype=x.dtype, device=x.device)
        for distance, weight in zip(distances, module.weights):
            weighted = (distance * weight.to(distance)).sum(dim=1, keepdim=True)
            layer_mean = float(weighted.mean())
            scalar += layer_mean
            up = F.interpolate(weighted, size=x.shape[-2:], mode="nearest")[0, 0]
            up_mean = float(up.mean())
            if up_mean > 0:
                correction = max(correction, abs(up_mean - layer_mean) / max(layer_mean, 1e-12))
                up = up * (layer_mean / up_mean)
            crop += up

    full = torch.full(mask.shape, float("nan"), dtype=crop.dtype, device=crop.device)
    full[y0:y1, x0:x1] = crop
    return full, scalar, correction


# --------------------------------------------------------------------------------------


def ring_edge_fraction(union, bins, num_rings, distance_px=(SSIM_KERNEL_SIZE - 1) // 2):
    """Fraction of each ring's valid pixels lying within `distance_px` of the invalid region.

    This is the diagnostic for the outer-ring contamination described in the module
    docstring: SSIM and LPIPS score the mask-ZEROED images, so any pixel whose support
    reaches past the rim is partly measuring the agreement of two black regions. PSNR has
    no support and is unaffected. Computed from the union mask, i.e. the same geometry the
    rings are built on.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:  # * a diagnostic must never break the PSNR path
        return [float("nan")] * num_rings

    distance = distance_transform_edt(union)
    near = union & (distance <= distance_px)
    out = []
    for k in range(num_rings):
        ring = union & (bins == k)
        total = int(ring.sum())
        out.append(float((ring & near).sum() / total) if total else float("nan"))
    return out


def _accumulate(store, key, values, selection):
    """Sum/count a metric map over a boolean selection."""
    if not selection.any():
        return
    picked = values[selection]
    store[key][0] += float(np.nansum(picked))
    store[key][1] += int(np.count_nonzero(~np.isnan(picked)))


def evaluate(
    run_dir,
    num_rings,
    metrics=("psnr",),
    mask_radius="saved",
    device=None,
    camera_model="rad_tan_thin_prism_fisheye",
    split="test",
):
    """Ring-resolved masked metrics for one run.

    `metrics` defaults to PSNR ONLY so that existing callers (scripts/myscenes_table.py)
    stay CPU-cheap and unchanged; the CLI defaults to all three. `mask_radius` is either
    "saved" (the PNG masks render.py wrote, the canonical path) or a float, which rebuilds
    the masks geometrically from the run's own intrinsics.
    """
    metrics = tuple(metrics)
    unknown = set(metrics) - set(METRIC_NAMES)
    if unknown:
        raise SystemExit(f"unknown metrics {sorted(unknown)}; choose from {METRIC_NAMES}")
    needs_torch = bool({"ssim", "lpips"} & set(metrics))

    renders, ground_truth, mask_path, parent = find_split(run_dir, camera_model, split)
    if len(renders) != len(ground_truth):
        raise SystemExit(
            f"{parent}: {len(renders)} renders vs {len(ground_truth)} gt -- refusing to pair"
        )

    if mask_radius == "saved":
        view_masks, union = load_masks(parent, mask_path, renders)
    else:
        probe = np.asarray(Image.open(renders[0]).convert("RGB"))
        view_masks, union = rebuild_masks(
            run_dir, parent, renders, float(mask_radius), split, probe.shape[0], probe.shape[1]
        )

    bins, edges, geometry = ring_index(union, num_rings)

    torch = None
    if needs_torch:
        import torch as _torch

        torch = _torch
        if device is None:
            device = "cuda" if _torch.cuda.is_available() else "cpu"

    # * Cached per distinct mask object: a rig has two, not one per view.
    ring_masks, supports, bboxes = {}, {}, {}
    torch_masks = {}
    for mask in view_masks:
        if id(mask) in ring_masks:
            continue
        ring_masks[id(mask)] = [mask & (bins == k) for k in range(num_rings)]
        support = {"psnr": np.ones_like(mask)}
        if "ssim" in metrics:
            pad = (SSIM_KERNEL_SIZE - 1) // 2
            interior = np.zeros_like(mask)
            interior[pad : mask.shape[0] - pad, pad : mask.shape[1] - pad] = True
            support["ssim"] = interior
        if "lpips" in metrics:
            rows, cols = np.where(mask)
            bboxes[id(mask)] = (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)
            crop = np.zeros_like(mask)
            y0, y1, x0, x1 = bboxes[id(mask)]
            crop[y0:y1, x0:x1] = True
            support["lpips"] = crop
        supports[id(mask)] = support
        if needs_torch:
            torch_masks[id(mask)] = torch.from_numpy(mask).to(device)

    squared = np.zeros(num_rings)
    counts = np.zeros(num_rings)
    disk_squared = 0.0
    disk_count = 0
    per_view = []

    # * [sum, count] accumulators for the map-based metrics.
    pooled = {
        name: {"disk": [0.0, 0]} | {f"ring{k}": [0.0, 0] for k in range(num_rings)}
        for name in metrics
        if name != "psnr"
    }
    per_view_maps = {name: [] for name in metrics if name != "psnr"}
    frame_values = {name: [] for name in metrics if name != "psnr"}
    lpips_residuals = []
    lpips_corrections = []

    for render_path, gt_path, mask in zip(renders, ground_truth, view_masks):
        render_u8 = np.asarray(Image.open(render_path).convert("RGB"))
        gt_u8 = np.asarray(Image.open(gt_path).convert("RGB"))
        if render_u8.shape != gt_u8.shape:
            raise SystemExit(f"shape mismatch {render_path} {render_u8.shape} vs {gt_u8.shape}")

        # ---- PSNR: unchanged float64 numpy path, bit-for-bit the pre-existing one.
        render = render_u8.astype(np.float64) / 255.0
        gt = gt_u8.astype(np.float64) / 255.0
        error = ((render - gt) ** 2).sum(-1)
        rings = ring_masks[id(mask)]
        for k in range(num_rings):
            squared[k] += error[rings[k]].sum()
            counts[k] += rings[k].sum() * 3
        view_squared = error[mask].sum()
        disk_squared += view_squared
        disk_count += mask.sum() * 3
        per_view.append(-10.0 * math.log10(max(view_squared / (mask.sum() * 3), 1e-12)))

        if not needs_torch:
            continue

        render_t = torch.from_numpy(render_u8.copy()).permute(2, 0, 1).to(device).float() / 255.0
        gt_t = torch.from_numpy(gt_u8.copy()).permute(2, 0, 1).to(device).float() / 255.0
        mask_t = torch_masks[id(mask)]
        support = supports[id(mask)]

        maps = {}
        if "ssim" in metrics:
            maps["ssim"] = ssim_map(render_t, gt_t, mask_t).cpu().numpy().astype(np.float64)
            frame_values["ssim"].append(float(np.nanmean(maps["ssim"])))
        if "lpips" in metrics:
            field, scalar, correction = lpips_map(
                render_t, gt_t, mask_t, bboxes[id(mask)], device
            )
            maps["lpips"] = field.cpu().numpy().astype(np.float64)
            frame_values["lpips"].append(scalar)
            lpips_residuals.append(abs(float(np.nanmean(maps["lpips"])) - scalar))
            lpips_corrections.append(correction)

        for name, values in maps.items():
            sup = support[name]
            disk_sel = mask & sup
            _accumulate(pooled[name], "disk", values, disk_sel)
            per_view_maps[name].append(float(np.nanmean(values[disk_sel])))
            for k in range(num_rings):
                _accumulate(pooled[name], f"ring{k}", values, rings[k] & sup)

    to_db = lambda s, c: -10.0 * math.log10(max(s / max(c, 1), 1e-12))  # noqa: E731
    mean = lambda pair: (pair[0] / pair[1]) if pair[1] else float("nan")  # noqa: E731

    result = {
        "run": run_dir,
        "views": len(renders),
        "rings": [to_db(squared[k], counts[k]) for k in range(num_rings)],
        "ring_edges": edges[: num_rings + 1],
        "disk_pooled": to_db(disk_squared, disk_count),
        "disk_per_view_mean": float(np.mean(per_view)),
        "valid_fraction": float(np.mean([m.mean() for m in view_masks])),
        "distinct_masks": len(ring_masks),
        "mask_radius": mask_radius,
        "disk_centre_px": [float(geometry[1]), float(geometry[0])],
        "disk_radius_px": float(geometry[2]),
        "ring_edge_fraction_5px": ring_edge_fraction(union, bins, num_rings),
        "metrics": {
            "psnr": {
                "rings": [to_db(squared[k], counts[k]) for k in range(num_rings)],
                "ring_counts": [int(counts[k] // 3) for k in range(num_rings)],
                "disk_pooled": to_db(disk_squared, disk_count),
                "disk_per_view_mean": float(np.mean(per_view)),
            }
        },
    }

    for name in metrics:
        if name == "psnr":
            continue
        entry = {
            "rings": [mean(pooled[name][f"ring{k}"]) for k in range(num_rings)],
            "ring_counts": [pooled[name][f"ring{k}"][1] for k in range(num_rings)],
            "disk_pooled": mean(pooled[name]["disk"]),
            "disk_per_view_mean": float(np.mean(per_view_maps[name])),
            "frame_per_view_mean": float(np.mean(frame_values[name])),
            "support_fraction": float(
                np.mean([supports[id(m)][name][m].mean() for m in view_masks])
            ),
        }
        if name == "lpips":
            entry["lpips_map_residual"] = float(np.max(lpips_residuals)) if lpips_residuals else 0.0
            entry["lpips_resample_correction"] = (
                float(np.max(lpips_corrections)) if lpips_corrections else 0.0
            )
        result["metrics"][name] = entry

    return result


def _print_table(metric, labels, results, num_rings, width):
    header = "label".ljust(width) + "".join(f"ring{k}".rjust(9) for k in range(num_rings))
    header += "   disk(pool)  disk(view)   n"
    arrow = "higher is better" if HIGHER_IS_BETTER[metric] else "LOWER is better"
    fmt = "{:9.3f}" if metric == "psnr" else "{:9.4f}"
    wide = "{:10.3f}" if metric == "psnr" else "{:10.4f}"
    print()
    print(f"=== {metric.upper()} ({arrow}) ===")
    print(header)
    print("-" * len(header))
    for label, result in zip(labels, results):
        entry = result["metrics"][metric]
        row = label.ljust(width) + "".join(fmt.format(v) for v in entry["rings"])
        row += "   " + wide.format(entry["disk_pooled"])
        row += "  " + wide.format(entry["disk_per_view_mean"])
        row += f" {result['views']:3d}"
        print(row)
    if metric != "psnr":
        frame = "  ".join(
            f"{label}={result['metrics'][metric]['frame_per_view_mean']:.4f}"
            for label, result in zip(labels, results)
        )
        print(f"{'frame convention (canonical shared pass)'.ljust(width)}{frame}")
    if len(results) > 1:
        print(f"deltas vs {labels[0]}")
        print("-" * len(header))
        base = results[0]["metrics"][metric]
        dfmt = "{:+9.3f}" if metric == "psnr" else "{:+9.4f}"
        dwide = "{:+10.3f}" if metric == "psnr" else "{:+10.4f}"
        for label, result in zip(labels[1:], results[1:]):
            entry = result["metrics"][metric]
            row = label.ljust(width)
            row += "".join(dfmt.format(v - b) for v, b in zip(entry["rings"], base["rings"]))
            row += "   " + dwide.format(entry["disk_pooled"] - base["disk_pooled"])
            row += "  " + dwide.format(entry["disk_per_view_mean"] - base["disk_per_view_mean"])
            print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--rings", type=int, default=6)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRIC_NAMES),
        choices=list(METRIC_NAMES),
        help="default: all three. `psnr` alone needs no GPU and no torch.",
    )
    parser.add_argument(
        "--mask-radius",
        default="saved",
        help="`saved` (default) uses the masks render.py wrote; a float (0.85 / 0.95 / 1.00) "
        "rebuilds them from the run's own intrinsics. 0.95 rebuilds bit-for-bit identical "
        "to `saved` on every run whose fisheye_mask_radius_scale is 0.95.",
    )
    parser.add_argument("--device", default=None, help="torch device for SSIM/LPIPS")
    parser.add_argument("--camera-model", default="rad_tan_thin_prism_fisheye")
    parser.add_argument("--split", default="test")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match --runs")

    results = [
        evaluate(
            run,
            args.rings,
            metrics=args.metrics,
            mask_radius=args.mask_radius,
            device=args.device,
            camera_model=args.camera_model,
            split=args.split,
        )
        for run in args.runs
    ]

    width = max(len(label) for label in labels) + 2
    if set(args.metrics) - {"psnr"}:
        width = max(width, len("frame convention (canonical shared pass)") + 1)
    for metric in METRIC_NAMES:
        if metric in args.metrics:
            _print_table(metric, labels, results, args.rings, width)

    print()
    print("ring edges (fraction of disk radius): " + ", ".join(f"{e:.3f}" for e in results[0]["ring_edges"]))
    print(f"valid fraction of frame: {results[0]['valid_fraction']:.4f}")
    print(f"mask radius: {results[0]['mask_radius']}   disk radius: {results[0]['disk_radius_px']:.1f} px")
    print(f"distinct masks: {results[0]['distinct_masks']}")
    if set(args.metrics) - {"psnr"}:
        edge = ", ".join(f"{v:.3f}" for v in results[0]["ring_edge_fraction_5px"])
        print(f"ring fraction within one SSIM window of the rim: {edge}")
        print("  ^ SSIM/LPIPS on the outermost ring are pulled toward the zeroed exterior;")
        print("    quote PSNR there, or read SSIM/LPIPS on the inner rings. See the docstring.")
    if "lpips" in args.metrics:
        entry = results[0]["metrics"]["lpips"]
        print(
            f"lpips map/scalar residual (max over views): {entry['lpips_map_residual']:.2e}"
            f"   resample correction: {entry['lpips_resample_correction']:.2e}"
        )

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump({label: r for label, r in zip(labels, results)}, handle, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
