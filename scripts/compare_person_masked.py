"""Compare a person-masked gray run against the stock baseline on one FIORD scene.

Both runs are evaluated over the SAME test views (every-8th-of-sorted-names split,
identical to training) under TWO mask protocols:
  valid        : the shared per-camera fisheye disk mask (r=0.95) — same numbers
                 as scripts/masked_eval_fiord.py
  valid+static : disk mask AND NOT person mask of the test view — measures the
                 static scene only, which is the fair protocol when one model
                 deliberately does not reproduce the photographer.

PSNR/SSIM masked as in gray.utils; LPIPS masked by zeroing both images and
bbox-cropping to the DISK mask bbox (same crop for both protocols so LPIPS
stays comparable).

Usage (pueue, GPU):
  python scripts/compare_person_masked.py <scene> \
      --baseline /workspace/gray/worktrees/fiord-multicam/out/<scene>_fisheye_baseline \
      --masked   out/<scene>_person_masked \
      [--person_mask_root /workspace/dataset/fiord_masks] [--iteration 15000]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gray import colmap  # noqa: E402
from gray.utils import masked_psnr, masked_ssim  # noqa: E402

DEVICE = "cuda"
BASE = "/workspace/dataset/fiord_baselines"

_lpips = None


def lpips_fn():
    global _lpips
    if _lpips is None:
        from piq import LPIPS
        _lpips = LPIPS(reduction="none").to(DEVICE)
    return _lpips


def load_rgb(p):
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    ).permute(2, 0, 1).to(DEVICE)


def load_mask(p):
    return torch.from_numpy(
        np.asarray(Image.open(p).convert("L")) > 127
    ).to(DEVICE)


def bbox_lpips(render, gt, mask, bbox):
    y0, y1, x0, x1 = bbox
    m = mask.unsqueeze(0).float()
    r = (render * m)[:, y0:y1, x0:x1].unsqueeze(0)
    g = (gt * m)[:, y0:y1, x0:x1].unsqueeze(0)
    with torch.no_grad():
        return float(lpips_fn()(r, g).item())


def test_split_names(scene):
    extr = colmap.read_extrinsics_binary(
        os.path.join(BASE, scene, "distorted", "sparse", "0", "images.bin"))
    names = sorted(e.name for e in extr.values())
    return names[::8]


def eval_run(run_dir, iteration, test_names, person_root, scene):
    gdir = os.path.join(run_dir, "test", f"{iteration:05d}", "opencv_fisheye")
    renders = sorted(glob.glob(os.path.join(gdir, "renders", "*.png")))
    gts = sorted(glob.glob(os.path.join(gdir, "gt", "*.png")))
    assert len(renders) == len(gts) == len(test_names), (
        f"{run_dir}: {len(renders)} renders, {len(gts)} gts, "
        f"{len(test_names)} split names")
    valid_masks, bboxes = {}, {}
    rows = []
    for r, g, name in zip(renders, gts, test_names):
        cam = os.path.dirname(name)  # cam1 / cam2
        uid = int(cam.replace("cam", ""))
        if uid not in valid_masks:
            valid_masks[uid] = load_mask(os.path.join(gdir, f"valid_mask_cam{uid}.png"))
            ys, xs = torch.where(valid_masks[uid])
            bboxes[uid] = (ys.min().item(), ys.max().item() + 1,
                           xs.min().item(), xs.max().item() + 1)
        valid = valid_masks[uid]
        stem = os.path.splitext(os.path.basename(name))[0]
        pm_path = os.path.join(person_root, scene, "fisheye", cam, stem + ".png")
        person = load_mask(pm_path) if os.path.exists(pm_path) else None
        static = valid & ~person if person is not None else valid
        R, G = load_rgb(r), load_rgb(g)
        row = {"name": name,
               "person_frac": float(person.float().mean().item()) if person is not None else 0.0}
        for proto, m in (("valid", valid), ("static", static)):
            row[proto] = {
                "psnr": float(masked_psnr(R, G, m).item()),
                "ssim": float(masked_ssim(R, G, m).item()),
                "lpips": bbox_lpips(R, G, m, bboxes[uid]),
            }
        rows.append(row)
    return rows


def summarize(rows, proto):
    return {k: float(np.mean([r[proto][k] for r in rows]))
            for k in ("psnr", "ssim", "lpips")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--masked", required=True)
    ap.add_argument("--person_mask_root", default="/workspace/dataset/fiord_masks")
    ap.add_argument("--iteration", type=int, default=15000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    test_names = test_split_names(args.scene)
    runs = {}
    for label, run_dir in (("baseline", args.baseline), ("person_masked", args.masked)):
        rows = eval_run(run_dir, args.iteration, test_names,
                        args.person_mask_root, args.scene)
        runs[label] = rows
        for proto in ("valid", "static"):
            s = summarize(rows, proto)
            print(f"{label:14s} [{proto:6s}] PSNR {s['psnr']:6.3f}  "
                  f"SSIM {s['ssim']:.4f}  LPIPS {s['lpips']:.4f}  (n={len(rows)})",
                  flush=True)

    mean_pf = float(np.mean([r["person_frac"] for r in runs["baseline"]]))
    print(f"mean person fraction over test views: {mean_pf:.4f}")

    out = args.out or os.path.join(args.masked, "compare_person_masked.json")
    with open(out, "w") as f:
        json.dump({
            "scene": args.scene, "iteration": args.iteration,
            "summary": {label: {proto: summarize(rows, proto)
                                for proto in ("valid", "static")}
                        for label, rows in runs.items()},
            "mean_person_frac_test": mean_pf,
            "per_view": runs,
        }, f, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
