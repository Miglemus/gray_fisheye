"""Complete metric set for the myscenes runs, in the project's required protocol.

The house rule for any cross-method table is PSNR / SSIM / LPIPS / #gaussians / FPS /
train-time, all under ONE shared eval pass -- never a repo's self-reported numbers. This
reproduces exactly the definitions `scripts/masked_eval.py` uses:

  PSNR  : masked, MSE over valid pixels only        (gray.utils.masked_psnr)
  SSIM  : masked, invalid pixels zeroed in both     (gray.utils.masked_ssim)
  LPIPS : invalid pixels zeroed in BOTH images, then bbox-cropped to the lens disk
          (masked_eval.frame_lpips) -- gray does not mask its render output and spills
          content into the invalid ring, so without the zeroing it is penalised for pixels
          that are not even evaluated.

The mask is the `valid_mask.png` render.py saved next to the renders, i.e. built from the
COLMAP intrinsics and identical for every ablation rung.

usage: python scripts/full_metrics.py --runs-root /workspace/gray/tmp/final --suffix _noncentral
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/gray/worktrees/noncentral-camera")
from gray.utils import masked_psnr, masked_ssim  # noqa: E402

SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
DEVICE = "cuda"
_lpips = None


def lpips_fn():
    global _lpips
    if _lpips is None:
        from piq import LPIPS

        _lpips = LPIPS(reduction="none").to(DEVICE)
    return _lpips


def load(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(DEVICE)


def frame_lpips(render, gt, mask, bbox):
    y0, y1, x0, x1 = bbox
    m = mask.unsqueeze(0).float()
    a = (render * m)[:, y0:y1, x0:x1].unsqueeze(0)
    b = (gt * m)[:, y0:y1, x0:x1].unsqueeze(0)
    with torch.no_grad():
        return lpips_fn()(a, b).item()


def last_csv_value(path):
    """Last numeric value of a gray log. Formats differ per file: fps.csv is a bare float,
    num_gaussians.csv is `iteration value`, and time.csv is `iteration HH:MM:SS`."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        rows = [r for r in csv.reader(handle, delimiter=" ") if r]
    if not rows:
        return None
    token = rows[-1][-1]
    if ":" in token:
        parts = [float(x) for x in token.split(":")]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    try:
        return float(token)
    except ValueError:
        return None


def score(run_dir):
    renders = sorted(glob.glob(f"{run_dir}/test/*/rad_tan_thin_prism_fisheye/renders/*.png"))
    if not renders:
        return None
    parent = os.path.dirname(os.path.dirname(renders[0]))
    gts = sorted(glob.glob(f"{parent}/gt/*.png"))
    mask = torch.from_numpy(
        np.asarray(Image.open(f"{parent}/valid_mask.png").convert("L")) > 127
    ).to(DEVICE)
    ys, xs = torch.where(mask)
    bbox = (ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1)

    psnrs, ssims, lpipses = [], [], []
    for render_path, gt_path in zip(renders, gts):
        render, gt = load(render_path), load(gt_path)
        psnrs.append(masked_psnr(render, gt, mask).item())
        ssims.append(masked_ssim(render, gt, mask).item())
        lpipses.append(frame_lpips(render, gt, mask, bbox))

    return {
        "psnr": statistics.mean(psnrs),
        "ssim": statistics.mean(ssims),
        "lpips": statistics.mean(lpipses),
        "fps": last_csv_value(f"{run_dir}/fps.csv"),
        "train_s": last_csv_value(f"{run_dir}/time.csv"),
        "gaussians": last_csv_value(f"{run_dir}/num_gaussians.csv"),
        "views": len(renders),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--suffix", default="_noncentral")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    results = {}
    header = f"{'scene':12s}{'PSNR':>9s}{'SSIM':>9s}{'LPIPS':>9s}{'FPS':>9s}{'train_s':>9s}{'#gauss':>11s}"
    print(header)
    print("-" * len(header))
    for scene in SCENES:
        run = os.path.join(args.runs_root, f"{scene}{args.suffix}")
        entry = score(run)
        if entry is None:
            print(f"{scene:12s}{'-- no rttpf renders --':>47s}")
            continue
        results[scene] = entry
        print(
            f"{scene:12s}{entry['psnr']:9.3f}{entry['ssim']:9.4f}{entry['lpips']:9.4f}"
            f"{(entry['fps'] or float('nan')):9.1f}{(entry['train_s'] or float('nan')):9.0f}"
            f"{(entry['gaussians'] or float('nan')):11.0f}"
        )
    if results:
        print("-" * len(header))
        agg = {}
        for key in ("psnr", "ssim", "lpips", "fps", "train_s", "gaussians"):
            values = [v[key] for v in results.values() if v[key] is not None]
            agg[key] = statistics.mean(values) if values else float("nan")
        print(
            f"{'MEAN':12s}{agg['psnr']:9.3f}{agg['ssim']:9.4f}{agg['lpips']:9.4f}"
            f"{agg['fps']:9.1f}{agg['train_s']:9.0f}{agg['gaussians']:11.0f}"
        )
        print("\nNOTE: FPS and train_s are wall-clock measurements and were taken while "
              "another job shared the GPU;\n      treat them as indicative only and "
              "re-measure on an idle device before publishing.")
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
