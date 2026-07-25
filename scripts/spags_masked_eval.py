#!/usr/bin/env python
"""Capturer-masked eval for native SPaGS pano renders (FullCircle).

Protocol (shared across runs/methods): masked PSNR/SSIM over valid = NOT person
(the test view's capturer mask from fullcircle_pano/<scene>/masks), LPIPS on the
mask-multiplied full pano (ERP has no disk mask). Test split = natsorted shot
stems [::8], matching RaRPano TEST_STEP=8; render NNNNN.png index = position in
that list.

Usage: spags_masked_eval.py <scene> <run_dir> [<run_dir> ...]
"""

import json
import os
import sys

import torch
from torchvision.io import read_image, ImageReadMode
import torch.nn.functional as F

sys.path.insert(0, "/workspace/gray/worktrees/person-masks")
from gray.utils import masked_psnr, masked_ssim  # noqa: E402


def lpips_fn():
    import piq
    return piq.LPIPS(reduction="none")


def load_rgb(path):
    return read_image(path, ImageReadMode.RGB).float().cuda() / 255.0


def main():
    scene = sys.argv[1]
    run_dirs = sys.argv[2:]
    pano_root = f"/workspace/dataset/fullcircle_pano/{scene}"
    stems = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(os.path.join(pano_root, "images")))
    test_stems = stems[::8]
    lp = lpips_fn()

    for run in run_dirs:
        rgb_dir = os.path.join(run, "test_30000", "rgb")
        gt_dir = os.path.join(run, "test_30000", "rgb_gt")
        files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
        assert len(files) == len(test_stems), (len(files), len(test_stems))
        rows = []
        for i, f in enumerate(files):
            render = load_rgb(os.path.join(rgb_dir, f))
            gt = load_rgb(os.path.join(gt_dir, f))
            mpath = os.path.join(pano_root, "masks", test_stems[i] + ".png")
            person = read_image(mpath, ImageReadMode.GRAY).cuda() > 127
            if person.shape[-2:] != render.shape[-2:]:
                person = F.interpolate(person.float()[None], size=render.shape[-2:],
                                       mode="nearest")[0] > 0.5
            keep = (~person[0]).float()  # [H, W]
            psnr = masked_psnr(render, gt, keep).item()
            ssim = masked_ssim(render, gt, keep).item()
            with torch.no_grad():
                lpips = lp((render * keep)[None], (gt * keep)[None]).item()
            rows.append({"stem": test_stems[i], "file": f, "psnr": psnr,
                         "ssim": ssim, "lpips": lpips,
                         "person_frac": person.float().mean().item()})
        mean = lambda k: sum(r[k] for r in rows) / len(rows)
        summary = {"run": run, "scene": scene, "protocol": "static(=NOT person)",
                   "n": len(rows), "psnr": mean("psnr"), "ssim": mean("ssim"),
                   "lpips": mean("lpips"), "mean_person_frac": mean("person_frac")}
        out = os.path.join(run, "masked_eval_static.json")
        with open(out, "w") as fjson:
            json.dump({"summary": summary, "per_view": rows}, fjson, indent=1)
        print(f"{os.path.basename(run):60s} PSNR {summary['psnr']:.3f} "
              f"SSIM {summary['ssim']:.4f} LPIPS {summary['lpips']:.4f} "
              f"(n={summary['n']}, person {summary['mean_person_frac']:.3f})",
              flush=True)


if __name__ == "__main__":
    main()
