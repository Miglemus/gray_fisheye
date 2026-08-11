"""Do the central terms carry their weight? -- the two *subtractive* rungs, on 7 scenes.

The cumulative ladder (`off` -> `ana` -> `noncentral`) answers "what does z(theta) add on
top of everything else". It does NOT answer "can z(theta) stand on its own", because the
mean-over-depth part of the non-central shift, z(theta) sin(theta) E[1/t | theta], has
exactly the form of a central radial correction. Remove the central terms and `z` is free
to absorb their job, so a subtractive rung is a different experiment, not the same one
read backwards.

    noncentral_no_ana  = noncentral minus the anamorphic harmonics   (111 -> 31 params)
    z_only             = the non-central profile alone               (111 ->  8 params)

Scoring is the shared protocol: the run's own valid_mask.png (radius 0.95), masked PSNR
per view, then the mean over views -- identical to what produced the published 27.124.
"""

import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect import NUM_RINGS, SCENES, load, mask_for, resolve, ring_map  # noqa: E402

WS = "/workspace"
HERE = os.path.dirname(os.path.abspath(__file__))

# * `off` is the CONFIG-MATCHED baseline, not the published one: workshop's published run
# * is the only one trained without vignetting_comp / batch_size 2, so its re-run stands in.
VARIANTS = ["off", "ana", "central_matched", "noncentral_no_ana", "z_only", "noncentral"]


def run_dir(scene, variant):
    if variant == "off":
        return (
            f"{WS}/gray/tmp/noncentral/fix15k_workshop"
            if scene == "workshop"
            else f"{WS}/gray/out/{scene}_fisheye_baseline"
        )
    # * `tmp/` resolves against the launching shell's cwd, so runs are split across two roots.
    for root in (f"{WS}/gray/tmp/final", f"{WS}/gray/worktrees/noncentral-camera/tmp/final"):
        if os.path.isdir(f"{root}/{scene}_{variant}"):
            return f"{root}/{scene}_{variant}"
    return None


def score(renders, gts, mask, idx):
    per_view, ring_se, ring_n = [], np.zeros(NUM_RINGS), np.zeros(NUM_RINGS)
    valid = mask.sum()
    for rp, gp in zip(renders, gts):
        err = ((load(rp) - load(gp)) ** 2).sum(axis=2)
        masked = np.where(mask, err, 0.0)
        per_view.append(-10.0 * math.log10(max(masked.sum() / (valid * 3.0), 1e-12)))
        for k in range(NUM_RINGS):
            sel = idx == k
            ring_se[k] += masked[sel].sum()
            ring_n[k] += sel.sum()
    return {
        "psnr": float(np.mean(per_view)),
        "rings": [
            -10.0 * math.log10(max(ring_se[k] / (ring_n[k] * 3.0), 1e-12)) for k in range(NUM_RINGS)
        ],
        "views": len(per_view),
    }


def from_dir(base, mask, idx):
    r = sorted(glob.glob(f"{base}/test/*/rad_tan_thin_prism_fisheye/renders/*.png"))
    g = sorted(glob.glob(f"{base}/test/*/rad_tan_thin_prism_fisheye/gt/*.png"))
    if not r or len(r) != len(g):
        return None
    return score(r, g, mask, idx)


def main():
    out = {}
    for scene in SCENES:
        mask = mask_for(scene)
        idx, _, _ = ring_map(mask, NUM_RINGS)
        out[scene] = {}
        for variant in VARIANTS:
            base = run_dir(scene, variant)
            result = from_dir(base, mask, idx) if base else None
            if result is None:
                print(f"  -- {scene}/{variant}: missing", flush=True)
                continue
            out[scene][variant] = result
            print(f"{scene:11s} {variant:20s} {result['psnr']:7.3f}", flush=True)
        # * The published gray row (workshop included as-published) and SPaGS.
        for method, key in (("gray", "published"), ("SPaGS", "spags")):
            hit = resolve(scene, method)
            if hit is None:
                continue
            out[scene][key] = score(hit[0], hit[1], mask, idx)
            print(f"{scene:11s} {key:20s} {out[scene][key]['psnr']:7.3f}", flush=True)

    with open(os.path.join(HERE, "subtractive.json"), "w") as handle:
        json.dump(out, handle)

    print("\n7-scene means (only over scenes where the variant exists):")
    for variant in VARIANTS + ["published", "spags"]:
        vals = [out[s][variant]["psnr"] for s in SCENES if variant in out[s]]
        if vals:
            tag = "" if len(vals) == len(SCENES) else f"   [{len(vals)}/7 scenes]"
            print(f"  {variant:20s} {np.mean(vals):7.3f}{tag}")


if __name__ == "__main__":
    main()
