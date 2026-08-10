"""Per-view and radially-binned masked PSNR for the report figures.

Everything here is CPU numpy and reproduces `gray.utils.masked_psnr` exactly:
    mse = sum((r - g)^2 * mask) / (n_valid * channels);  psnr = -10 log10 mse
The mask is the run's own `valid_mask.png` (radius 0.95, built from the COLMAP
intrinsics), which is identical for every method and every ablation rung.

Rings are equal-area annuli of the valid disk, so ring k covers the same pixels in
every run. PSNR is *pooled* within a ring (sum of squared error over all views, then
one dB conversion), not averaged per view -- far more stable on small pixel counts.
"""

import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

WS = "/workspace"
SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
NUM_RINGS = 8


def gray_run(scene):
    return f"{WS}/gray/out/{scene}_fisheye_baseline"


def resolve(scene, method):
    """(renders, gts, base) for one (scene, method), mirroring the viewer's resolver."""
    if method == "gray":
        base = gray_run(scene)
        r = sorted(glob.glob(f"{base}/test/*/rad_tan_thin_prism_fisheye/renders/*.png"))
        g = sorted(glob.glob(f"{base}/test/*/rad_tan_thin_prism_fisheye/gt/*.png"))
    elif method == "gray-non-central":
        base = f"{WS}/gray/tmp/final/{scene}_noncentral"
        r = sorted(glob.glob(f"{base}/test/*/rad_tan_thin_prism_fisheye/renders/*.png"))
        g = sorted(glob.glob(f"{base}/test/*/rad_tan_thin_prism_fisheye/gt/*.png"))
    elif method == "gray-workshopfix":
        base = f"{WS}/gray/tmp/noncentral/fix15k_workshop"
        r = sorted(glob.glob(f"{base}/test/*/rad_tan_thin_prism_fisheye/renders/*.png"))
        g = sorted(glob.glob(f"{base}/test/*/rad_tan_thin_prism_fisheye/gt/*.png"))
    elif method == "SPaGS":
        root = f"{WS}/SPaGS-rttpf/nerficg/output/SPaGS"
        base, r, g = None, [], []
        for cand in sorted(glob.glob(f"{root}/{scene}_masked_*"), reverse=True):
            rr = sorted(glob.glob(f"{cand}/test/ours_*/renders/*.png"))
            gg = sorted(glob.glob(f"{cand}/test/ours_*/gt/*.png"))
            if rr and len(rr) == len(gg):
                base, r, g = cand, rr, gg
                break
    else:
        raise ValueError(method)
    if not r or len(r) != len(g):
        return None
    return r, g, base


def mask_for(scene):
    """The shared valid mask: taken from gray's own baseline render tree.

    Using ONE mask for every method is the project rule; the methods render on the same
    pixel grid so the disk is identical anyway, but reading it from a single place makes
    that impossible to get wrong by accident.
    """
    hits = sorted(glob.glob(f"{gray_run(scene)}/test/*/rad_tan_thin_prism_fisheye/valid_mask.png"))
    return np.asarray(Image.open(hits[0]).convert("L")) > 127


def ring_map(mask, num_rings):
    """Equal-area annulus index per pixel, plus each ring's outer normalized radius."""
    rows, cols = np.where(mask)
    cy = (rows.min() + rows.max()) / 2.0
    cx = (cols.min() + cols.max()) / 2.0
    radius = ((rows.max() - rows.min()) + (cols.max() - cols.min())) / 4.0
    gy, gx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    rho = np.sqrt((gy - cy) ** 2 + (gx - cx) ** 2) / radius
    # * equal area  <=>  equal steps in rho^2
    edges = np.sqrt(np.linspace(0.0, 1.0, num_rings + 1))
    idx = np.clip(np.searchsorted(edges, rho, side="right") - 1, 0, num_rings - 1)
    idx = np.where(mask, idx, -1)
    return idx, edges, rho


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def score(scene, method, mask, idx, num_rings):
    got = resolve(scene, method)
    if got is None:
        return None
    renders, gts, base = got
    per_view = []
    ring_se = np.zeros(num_rings)
    ring_n = np.zeros(num_rings)
    valid = mask.sum()
    for rp, gp in zip(renders, gts):
        err = ((load(rp) - load(gp)) ** 2).sum(axis=2)  # [H, W], summed over channels
        masked = np.where(mask, err, 0.0)
        per_view.append(-10.0 * math.log10(max(masked.sum() / (valid * 3.0), 1e-12)))
        for k in range(num_rings):
            sel = idx == k
            ring_se[k] += masked[sel].sum()
            ring_n[k] += sel.sum()
    rings = [-10.0 * math.log10(max(ring_se[k] / (ring_n[k] * 3.0), 1e-12)) for k in range(num_rings)]
    return {
        "base": base,
        "views": len(renders),
        "psnr": float(np.mean(per_view)),
        "per_view": [float(v) for v in per_view],
        "names": [os.path.basename(p) for p in renders],
        "rings": rings,
        "ring_pixels": ring_n.tolist(),
    }


def main():
    out = {"num_rings": NUM_RINGS, "scenes": {}}
    for scene in SCENES:
        mask = mask_for(scene)
        idx, edges, _ = ring_map(mask, NUM_RINGS)
        entry = {"ring_edges": edges.tolist(), "methods": {}}
        methods = ["gray", "gray-non-central", "SPaGS"]
        if scene == "workshop":
            methods.append("gray-workshopfix")
        for method in methods:
            result = score(scene, method, mask, idx, NUM_RINGS)
            if result is None:
                print(f"  !! {scene}/{method}: no renders", file=sys.stderr)
                continue
            entry["methods"][method] = result
            print(f"{scene:11s} {method:18s} {result['psnr']:7.3f}  ({result['views']} views)",
                  flush=True)
        out["scenes"][scene] = entry
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rings.json")
    with open(dest, "w") as handle:
        json.dump(out, handle)
    print("wrote", dest)


if __name__ == "__main__":
    main()
