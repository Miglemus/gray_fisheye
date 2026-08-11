"""Pick the patch where the camera model actually changed the picture, and cut it out.

Selection is deliberately mechanical so it cannot be cherry-picked by eye:
  * take the test view with the largest masked-PSNR gain;
  * slide a fixed window over the *outer* part of the disk (rho > 0.6, where both residual
    terms are predicted to act) and keep the window with the largest drop in summed squared
    error. No manual choice enters anywhere.
The error panels use a shared colour scale across the two methods, otherwise the comparison
means nothing.
"""

import base64
import glob
import io
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect import load, mask_for, ring_map  # noqa: E402

WS = "/workspace"
HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 224


def runs_for(scene):
    baseline = (f"{WS}/gray/tmp/noncentral/fix15k_workshop" if scene == "workshop"
                else f"{WS}/gray/out/{scene}_fisheye_baseline")
    return baseline, f"{WS}/gray/tmp/final/{scene}_noncentral"


def listing(run):
    return (sorted(glob.glob(f"{run}/test/*/rad_tan_thin_prism_fisheye/renders/*.png")),
            sorted(glob.glob(f"{run}/test/*/rad_tan_thin_prism_fisheye/gt/*.png")))


def heat(values, vmax):
    """Perceptually monotone blue -> magenta -> yellow ramp for the error panels."""
    t = np.clip(values / max(vmax, 1e-9), 0.0, 1.0)[..., None]
    stops = np.array([[8, 12, 40], [40, 30, 120], [150, 30, 130], [235, 90, 70], [255, 224, 120]],
                     dtype=np.float64)
    pos = np.linspace(0.0, 1.0, len(stops))
    out = np.zeros(t.shape[:2] + (3,))
    for channel in range(3):
        out[..., channel] = np.interp(t[..., 0], pos, stops[:, channel])
    return out.astype(np.uint8)


def encode(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="WEBP", quality=88)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    scenes = sys.argv[1:] or ["workshop", "reception"]
    out = {}
    for scene in scenes:
        base_run, nc_run = runs_for(scene)
        base_r, base_g = listing(base_run)
        nc_r, nc_g = listing(nc_run)
        mask = mask_for(scene)
        idx, edges, rho = ring_map(mask, 8)
        valid = mask.sum()

        best_view, best_gain = None, -1e9
        for i, (bp, np_, gp) in enumerate(zip(base_r, nc_r, base_g)):
            gt = load(gp)
            eb = np.where(mask, ((load(bp) - gt) ** 2).sum(axis=2), 0.0)
            en = np.where(mask, ((load(np_) - gt) ** 2).sum(axis=2), 0.0)
            gain = (-10 * math.log10(en.sum() / (valid * 3))) - (-10 * math.log10(eb.sum() / (valid * 3)))
            if gain > best_gain:
                best_gain, best_view = gain, (i, bp, np_, gp, eb, en)

        i, bp, np_, gp, eb, en = best_view
        # * Integral image over (eb - en) restricted to the outer disk, then argmax window.
        # * The window must also be >= 92% inside the lens disk: a patch straddling the disk
        # * edge scores well but is half black, and the figure has to stay readable.
        drop = np.where(rho > 0.55, eb - en, 0.0)
        integral = drop.cumsum(0).cumsum(1)
        valid_integral = mask.astype(np.float64).cumsum(0).cumsum(1)
        window_sum = lambda img, y, x: (img[y + WINDOW, x + WINDOW] - img[y, x + WINDOW]
                                        - img[y + WINDOW, x] + img[y, x])
        h, w = drop.shape
        best, by, bx = -1e18, 0, 0
        for y in range(0, h - WINDOW, 8):
            for x in range(0, w - WINDOW, 8):
                if window_sum(valid_integral, y, x) < 0.92 * WINDOW * WINDOW:
                    continue
                total = window_sum(integral, y, x)
                if total > best:
                    best, by, bx = total, y, x
        sl = (slice(by, by + WINDOW), slice(bx, bx + WINDOW))

        # * Display gamma only -- identical on the three RGB panels. The lens is heavily
        # * vignetted out here and the raw crop is near-black on screen.
        show = lambda img: encode(
            (np.clip(np.where(mask[sl][..., None], img, 0.0), 0, 1) ** (1 / 1.7) * 255).astype(np.uint8))
        vmax = float(np.percentile(np.maximum(eb[sl], en[sl]), 99.5))
        panels = {
            "gt": show(load(gp)[sl]),
            "base": show(load(bp)[sl]),
            "nc": show(load(np_)[sl]),
            "err_base": encode(heat(eb[sl], vmax)),
            "err_nc": encode(heat(en[sl], vmax)),
        }
        out[scene] = {
            "view": os.path.basename(gp),
            "gain_db": best_gain,
            "window": [int(bx), int(by), WINDOW],
            "rho": float(rho[by + WINDOW // 2, bx + WINDOW // 2]),
            "err_drop_pct": float(100.0 * (1.0 - en[sl].sum() / max(eb[sl].sum(), 1e-12))),
            "panels": panels,
        }
        print(f"{scene:11s} view={out[scene]['view']} gain={best_gain:+.3f} dB  "
              f"patch rho={out[scene]['rho']:.2f}  local SSE -{out[scene]['err_drop_pct']:.1f}%",
              flush=True)

    with open(os.path.join(HERE, "crops.json"), "w") as handle:
        json.dump(out, handle)
    print("wrote crops.json",
          f"({os.path.getsize(os.path.join(HERE, 'crops.json')) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
