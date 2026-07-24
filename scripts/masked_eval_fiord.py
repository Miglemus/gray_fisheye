"""Unified masked evaluation for the FIORD fisheye baselines (gray, DFGS, 3dgrut).

Same protocol as scripts/masked_eval.py (myscenes), extended to FIORD's
per-camera masks: for every test view the mask is the CANONICAL shared file
valid_mask_cam<uid>.png of that view's colmap camera (the same files every
method trained with), resolved through gray's render masks.json.

  PSNR  : masked, MSE over valid px only   (gray.utils.masked_psnr)
  SSIM  : masked, invalid px zeroed in both (gray.utils.masked_ssim)
  LPIPS : masked — invalid zeroed in BOTH, then bbox-crop to the disk
          (per-camera bbox), LPIPS(reduction='none')

Alignment: all three methods use the identical every-8th-of-sorted-names test
split; renders are index-named. GT is taken from gray's saved gt (bit-identical
to the dataset input_4 images) for every method, and each method's render count
must match. Writes /workspace/dataset/fiord_baselines/{fiord_results.csv,
fiord_masked_metrics.json}.

Run via pueue (GPU): SSIM/LPIPS on gpu.
"""

import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/gray")
from gray.utils import masked_psnr, masked_ssim  # noqa: E402

DEVICE = "cuda"
BASE = "/workspace/dataset/fiord_baselines"
GRAY_OUT = "/workspace/gray/worktrees/fiord-multicam/out"
SCENES = ["bridge_out", "building_in", "building_out", "corridor_out", "hall_in",
          "kitchen_in", "meetingroom_in", "night_out", "road_out", "upstairs_in"]

_lpips = None


def lpips_fn():
    global _lpips
    if _lpips is None:
        from piq import LPIPS
        _lpips = LPIPS(reduction="none").to(DEVICE)
    return _lpips


def load_png(p):
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    ).permute(2, 0, 1).to(DEVICE)


_bbox = {}


def frame_lpips(render, gt, mask, key):
    if key not in _bbox:
        ys, xs = torch.where(mask)
        _bbox[key] = (ys.min().item(), ys.max().item() + 1,
                      xs.min().item(), xs.max().item() + 1)
    y0, y1, x0, x1 = _bbox[key]
    m = mask.unsqueeze(0).float()
    r = (render * m)[:, y0:y1, x0:x1].unsqueeze(0)
    g = (gt * m)[:, y0:y1, x0:x1].unsqueeze(0)
    with torch.no_grad():
        return float(lpips_fn()(r, g).item())


def method_renders(method, scene):
    if method == "gray":
        pats = [f"{GRAY_OUT}/{scene}_fisheye_baseline/test/*/opencv_fisheye/renders/*.png"]
    elif method == "DFGS":
        pats = [f"/workspace/DirectFisheye-GS-rttpf/out/{scene}_fisheye_baseline/test/ours_*/renders/*.png"]
    elif method == "3dgrut":
        pats = [f"/workspace/3dgrut/out/{scene}_fisheye_baseline/*/ours_30000/renders/*.png"]
    else:
        raise ValueError(method)
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits
    return []


def main():
    results = {}
    for scene in SCENES:
        gdir = f"{GRAY_OUT}/{scene}_fisheye_baseline/test/15000/opencv_fisheye"
        gts = sorted(glob.glob(f"{gdir}/gt/*.png"))
        mask_index = json.load(open(f"{gdir}/masks.json"))
        masks = {}
        for name in set(mask_index.values()):
            arr = np.asarray(Image.open(os.path.join(BASE, scene, name.replace("valid_mask_", "valid_mask_"))).convert("L")) \
                if False else np.asarray(Image.open(os.path.join(gdir, name)).convert("L"))
            masks[name] = torch.from_numpy(arr > 127).to(DEVICE)
        results[scene] = {}
        for method in ["gray", "DFGS", "3dgrut"]:
            renders = method_renders(method, scene)
            if not renders or len(renders) != len(gts):
                print(f"  SKIP {method} {scene}: n_render={len(renders)} n_gt={len(gts)}", flush=True)
                continue
            ps, ss, lp = [], [], []
            for r, g in zip(renders, gts):
                fname = os.path.basename(g)
                mask = masks[mask_index[fname]]
                R, G = load_png(r), load_png(g)
                ps.append(masked_psnr(R, G, mask).item())
                ss.append(masked_ssim(R, G, mask).item())
                lp.append(frame_lpips(R, G, mask, (scene, mask_index[fname])))
            results[scene][method] = {
                "PSNR": float(np.mean(ps)), "SSIM": float(np.mean(ss)),
                "LPIPS": float(np.mean(lp)), "n": len(ps),
                "per_view": {"psnr": [round(x, 3) for x in ps]},
            }
            print(f"  {method:8s} {scene:15s} PSNR {np.mean(ps):6.2f}  SSIM {np.mean(ss):.4f}  "
                  f"LPIPS {np.mean(lp):.4f}  (n={len(ps)})", flush=True)

    with open(f"{BASE}/fiord_masked_metrics.json", "w") as f:
        json.dump(results, f, indent=1)
    with open(f"{BASE}/fiord_results.csv", "w") as f:
        f.write("scene,method,psnr,ssim,lpips,n\n")
        for scene, ms in results.items():
            for m, v in ms.items():
                f.write(f"{scene},{m},{v['PSNR']:.3f},{v['SSIM']:.4f},{v['LPIPS']:.4f},{v['n']}\n")
    print("\nwrote fiord_masked_metrics.json + fiord_results.csv", flush=True)


if __name__ == "__main__":
    main()
