"""Radially-binned masked metrics for fisheye runs.

Why this exists: a camera-model residual does not act uniformly over the field. A radial
d(theta) residual vanishes at theta = 0 and the non-central term scales as
sin(theta) * z(theta) / depth, so both are peripheral by construction. A single full-disk
PSNR averages that signal away -- +0.2 dB overall is uninformative, +1.5 dB on the outer
ring is a result. gray had no such tooling.

Comparability rules this script enforces:
  * the mask is whatever render.py saved next to the renders (`valid_mask.png`), which is
    built from the COLMAP intrinsics and is therefore identical for every ablation rung --
    the learned residual never touches `cam_info.intrinsics`;
  * bins are equal-area annuli in *pixel* radius about the COLMAP principal point, i.e. a
    fixed function of the pixel, so ring k means the same pixels in every run;
  * PSNR is pooled over views within a ring (sum of squared error over all valid pixels of
    that ring, then converted once), which is far more stable than averaging per-view dB.

usage:
  python scripts/radial_eval.py --runs out/a out/b --labels baseline noncentral [--rings 6]
"""

import argparse
import glob
import json
import math
import os

import numpy as np
from PIL import Image


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
    return renders, ground_truth, mask_path


def ring_index(mask, num_rings):
    """Equal-area annuli of the valid disk, indexed by radius about the disk centre.

    Equal area keeps every ring statistically comparable; radius is taken about the disk's
    bounding-box centre, which for these calibrations is the principal point to within a
    fraction of a pixel (myscenes has cx, cy pinned at the exact sensor centre).
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
    return np.digitize(normalized, digitize_edges) - 1, edges


def evaluate(run_dir, num_rings):
    renders, ground_truth, mask_path = find_split(run_dir)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    bins, edges = ring_index(mask, num_rings)
    ring_masks = [mask & (bins == k) for k in range(num_rings)]

    squared = np.zeros(num_rings)
    counts = np.zeros(num_rings)
    disk_squared = 0.0
    disk_count = 0
    per_view = []
    for render_path, gt_path in zip(renders, ground_truth):
        render = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float64) / 255.0
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float64) / 255.0
        error = ((render - gt) ** 2).sum(-1)
        for k in range(num_rings):
            squared[k] += error[ring_masks[k]].sum()
            counts[k] += ring_masks[k].sum() * 3
        view_squared = error[mask].sum()
        disk_squared += view_squared
        disk_count += mask.sum() * 3
        per_view.append(-10.0 * math.log10(max(view_squared / (mask.sum() * 3), 1e-12)))

    to_db = lambda s, c: -10.0 * math.log10(max(s / max(c, 1), 1e-12))  # noqa: E731
    return {
        "run": run_dir,
        "views": len(renders),
        "rings": [to_db(squared[k], counts[k]) for k in range(num_rings)],
        "ring_edges": edges[: num_rings + 1],
        "disk_pooled": to_db(disk_squared, disk_count),
        "disk_per_view_mean": float(np.mean(per_view)),
        "valid_fraction": float(mask.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--rings", type=int, default=6)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match --runs")

    results = [evaluate(run, args.rings) for run in args.runs]

    width = max(len(label) for label in labels) + 2
    header = "label".ljust(width) + "".join(f"ring{k}".rjust(9) for k in range(args.rings))
    header += "   disk(pool)  disk(view)   n"
    print(header)
    print("-" * len(header))
    for label, result in zip(labels, results):
        row = label.ljust(width) + "".join(f"{v:9.3f}" for v in result["rings"])
        row += f"   {result['disk_pooled']:10.3f}  {result['disk_per_view_mean']:10.3f} {result['views']:3d}"
        print(row)

    if len(results) > 1:
        print()
        print(f"deltas vs {labels[0]} (dB)")
        print("-" * len(header))
        for label, result in zip(labels[1:], results[1:]):
            row = label.ljust(width)
            row += "".join(f"{v - b:+9.3f}" for v, b in zip(result["rings"], results[0]["rings"]))
            row += f"   {result['disk_pooled'] - results[0]['disk_pooled']:+10.3f}"
            row += f"  {result['disk_per_view_mean'] - results[0]['disk_per_view_mean']:+10.3f}"
            print(row)

    print()
    print("ring edges (fraction of disk radius): " + ", ".join(f"{e:.3f}" for e in results[0]["ring_edges"]))
    print(f"valid fraction of frame: {results[0]['valid_fraction']:.4f}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump({label: r for label, r in zip(labels, results)}, handle, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
