"""Where in the image does the learned camera actually buy its PSNR?

For each scene, average the per-pixel squared error of the `off` and `noncentral` test
renders over the whole test split, then look at the difference. If the gain comes from a
camera-geometry fix, the improvement must be *structured in image space*: it has to grow
with |x - cx| for an fx-vs-fy error, and with radius for a residual radial distortion. If
it came from extra fitting capacity it would be spread roughly uniformly, or follow scene
content instead of image coordinates.

Also emits the horizontal and radial profiles the report plots, plus a coarse downsampled
map per scene.

Outputs tmp/mipnerf360/analysis/error_maps.json.
"""

import json
import os

import numpy as np
from PIL import Image

ROOT = "/workspace/gray/worktrees/noncentral-camera"
OUT = f"{ROOT}/tmp/mipnerf360/analysis/error_maps.json"
SCENES = ["bicycle", "garden", "stump", "bonsai", "counter", "kitchen", "room"]
PAIR = ("off", "noncentral")
BLOCK = 16  # * downsample factor for the stored map


def load_sq_error(run):
    "Mean over the test split of the per-pixel squared error, [H, W]."
    base = f"{run}/test/15000/pinhole"
    names = sorted(os.listdir(f"{base}/renders"))
    total = None
    for name in names:
        render = np.asarray(Image.open(f"{base}/renders/{name}"), dtype=np.float64) / 255.0
        gt = np.asarray(Image.open(f"{base}/gt/{name}"), dtype=np.float64) / 255.0
        err = ((render - gt) ** 2).mean(-1)
        total = err if total is None else total + err
    return total / len(names), len(names)


def block_mean(array, block):
    h, w = array.shape
    hh, ww = h // block, w // block
    return array[: hh * block, : ww * block].reshape(hh, block, ww, block).mean((1, 3))


def main():
    payload = {}
    print(f"{'scene':9s} {'n':>3s} {'psnr_off':>9s} {'psnr_nc':>9s} {'dPSNR':>7s} "
          f"{'corr(|x-cx|)':>13s} {'corr(r)':>8s} {'edge/centre':>12s}")
    for scene in SCENES:
        runs = {tag: f"{ROOT}/tmp/mipnerf360/{scene}_{tag}" for tag in PAIR}
        if not all(os.path.exists(f"{r}/test/15000/pinhole/renders") for r in runs.values()):
            continue
        errs = {}
        for tag, run in runs.items():
            errs[tag], count = load_sq_error(run)
        gain = errs["off"] - errs["noncentral"]  # * > 0 where the learned camera helps

        height, width = gain.shape
        yy, xx = np.mgrid[0:height, 0:width]
        cx, cy = width / 2.0, height / 2.0
        ax = np.abs(xx - cx)
        radius = np.hypot(xx - cx, yy - cy)

        flat = gain.ravel()
        corr_x = float(np.corrcoef(ax.ravel(), flat)[0, 1])
        corr_r = float(np.corrcoef(radius.ravel(), flat)[0, 1])
        # * Ratio of mean gain in the outer third of |x - cx| to the inner third.
        outer = flat[(ax.ravel() > 2 * ax.max() / 3)].mean()
        inner = flat[(ax.ravel() < ax.max() / 3)].mean()

        # * Column profile: mean gain per image column, expressed as a local PSNR delta.
        col_off = errs["off"].mean(0)
        col_nc = errs["noncentral"].mean(0)
        row_off = errs["off"].mean(1)
        row_nc = errs["noncentral"].mean(1)
        psnr = lambda mse: float(-10.0 * np.log10(mse))

        payload[scene] = {
            "n_test": count,
            "psnr_off": psnr(errs["off"].mean()),
            "psnr_noncentral": psnr(errs["noncentral"].mean()),
            "corr_absx": corr_x,
            "corr_radius": corr_r,
            "edge_over_centre": float(outer / inner) if inner != 0 else float("nan"),
            "col_dpsnr": [psnr(a) - psnr(b) for a, b in zip(col_nc, col_off)],
            "row_dpsnr": [psnr(a) - psnr(b) for a, b in zip(row_nc, row_off)],
            "map": [[float(v) for v in row] for row in block_mean(gain, BLOCK)],
            "map_block": BLOCK,
            "width": width,
            "height": height,
        }
        p = payload[scene]
        print(f"{scene:9s} {count:3d} {p['psnr_off']:9.3f} {p['psnr_noncentral']:9.3f} "
              f"{p['psnr_noncentral'] - p['psnr_off']:7.3f} {corr_x:13.3f} {corr_r:8.3f} "
              f"{p['edge_over_centre']:12.2f}")

    with open(OUT, "w") as handle:
        json.dump(payload, handle)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
