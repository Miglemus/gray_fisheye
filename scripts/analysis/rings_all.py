"""W3.1 -- ring-resolved PSNR / SSIM / LPIPS, `off` vs `noncentral`, every paired dataset.

Runs `scripts/radial_eval.evaluate` over every (off, noncentral) pair that exists on disk
and writes ONE json. No training, no re-rendering: it re-scores PNGs that are already
there.

  python scripts/analysis/rings_all.py [--rings 6] [--only myscenes] [--device cuda]
      -> tmp/w3_radial/rings_all.json

WHICH RUN IS `off` (this is a choice, and it moves the numbers)
---------------------------------------------------------------------------------------
myscenes `off` is the PUBLISHED gray baseline (`/workspace/gray/out/<scene>_fisheye_baseline`)
for six scenes, and `tmp/noncentral/fix15k_workshop` for `workshop` -- the published
workshop baseline was trained without `--vignetting_comp` and at `batch_size 1`, a
configuration slip worth +0.71 dB that is not attributable to the camera model. This is
the convention of IMPLEMENTATION.md's attribution table.

It is NOT the convention of IMPLEMENTATION.md's "Where the gain lands" ring table, which
used `/workspace/gray/tmp/r4/tunnel_off` (28.483) for tunnel. That run is included here as
the extra pair `tunnel_r4off` precisely so the difference is visible: against it ring 0
gains +0.399, against `out/tunnel_fisheye_baseline` only +0.220. A ring profile is a
difference of two runs and inherits the noise of BOTH.

WHAT THE RINGS MEAN, AND WHERE THEY LIE
---------------------------------------------------------------------------------------
Equal-area annuli in pixel radius about the disk centre, so every ring holds the same
number of pixels. Ring k is the same pixels in `off` and in `noncentral` (the mask comes
from the COLMAP intrinsics, which no rung touches). Read `radial_eval.py`'s module
docstring before quoting a ring number -- in particular:

  * SSIM and LPIPS on the OUTERMOST ring are pulled toward the zeroed exterior, because
    their spatial support straddles the rim and scores black-against-black as agreement.
    PSNR has no support and is clean everywhere. Quote PSNR on the outer ring.
  * LPIPS' deepest receptive field (~212 px) is wider than a ring (~70 px on myscenes),
    so a 6-ring LPIPS profile cannot resolve ring-to-ring structure at all. It is
    reported because a FLAT LPIPS profile against a sloped PSNR profile is itself
    information, not because the ring values are independent.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import radial_eval  # noqa: E402

GRAY = "/workspace/gray"
MYSCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
FULLCIRCLE = ["room1", "room2", "room3", "flat1", "flat2", "lab", "lounge", "dark", "persons"]


def pairs():
    """[(dataset, scene, off_dir, nc_dir)] -- only the ones that exist are evaluated."""
    out = []
    for scene in MYSCENES:
        off = (f"{GRAY}/tmp/noncentral/fix15k_workshop" if scene == "workshop"
               else f"{GRAY}/out/{scene}_fisheye_baseline")
        out.append(("myscenes", scene, off, f"{GRAY}/tmp/final/{scene}_noncentral"))
    # * the `off` run the published ring table actually used for tunnel -- kept separate,
    # * never averaged into the myscenes mean.
    out.append(("myscenes_alt", "tunnel_r4off", f"{GRAY}/tmp/r4/tunnel_off",
                f"{GRAY}/tmp/final/tunnel_noncentral"))
    for scene in FULLCIRCLE:
        out.append(("fullcircle", scene,
                    f"{REPO}/out/fullcircle_rttpf/{scene}_refit_rttpf_off",
                    f"{REPO}/out/fullcircle_rttpf/{scene}_refit_rttpf"))
    out.append(("others", "workshop_immervision",
                f"{REPO}/out/workshop_immervision_off",
                f"{REPO}/out/workshop_immervision_noncentral"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rings", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="dataset or scene filter")
    ap.add_argument("--out", default=os.path.join(REPO, "tmp/w3_radial/rings_all.json"))
    args = ap.parse_args()

    selected = [p for p in pairs()
                if args.only is None or p[0] in args.only or p[1] in args.only]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results = {}
    for dataset, scene, off, nc in selected:
        key = f"{dataset}/{scene}"
        if not (os.path.isdir(off) and os.path.isdir(nc)):
            print(f"SKIP {key}: missing run dir", flush=True)
            continue
        entry = {"dataset": dataset, "scene": scene, "off_run": off, "noncentral_run": nc}
        for label, run in (("off", off), ("noncentral", nc)):
            t0 = time.time()
            entry[label] = radial_eval.evaluate(
                run, args.rings, metrics=("psnr", "ssim", "lpips"),
                mask_radius="saved", device=args.device,
            )
            print(f"  {key:34s} {label:11s} "
                  f"psnr {entry[label]['metrics']['psnr']['disk_pooled']:7.3f} "
                  f"ssim {entry[label]['metrics']['ssim']['frame_per_view_mean']:.4f} "
                  f"lpips {entry[label]['metrics']['lpips']['frame_per_view_mean']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        results[key] = entry
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1)
    print(f"wrote {args.out}  ({len(results)} pairs)", flush=True)


if __name__ == "__main__":
    main()
