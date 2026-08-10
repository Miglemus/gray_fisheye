"""Ring-resolved scores for the ablation rungs, so the non-central term can be isolated.

`noncentral - ana` is the *only* difference that is the z(theta) profile alone: the two
rungs share the tilt, radial and anamorphic components and the same knot count, and are
trained with the same schedule from the same initial point cloud. `central_matched` is the
capacity control (183 trained parameters against 111 -- biased in its favour).
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect import NUM_RINGS, load, mask_for, ring_map  # noqa: E402

import math  # noqa: E402

WS = "/workspace"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNGS = ["off", "ana", "central_matched", "noncentral"]


def run_dir(scene, rung):
    if rung == "off":
        # * The config-matched baseline: workshop's published run is the only one trained
        # * without vignetting_comp/batch_size 2, so its re-run stands in for it.
        return (f"{WS}/gray/tmp/noncentral/fix15k_workshop" if scene == "workshop"
                else f"{WS}/gray/out/{scene}_fisheye_baseline")
    # * `tmp/` resolves relative to the cwd of the launching shell, so the ablation runs are
    # * split between the main checkout and the worktree depending on when they were queued.
    for root in (f"{WS}/gray/tmp/final", f"{WS}/gray/worktrees/noncentral-camera/tmp/final"):
        if os.path.isdir(f"{root}/{scene}_{rung}"):
            return f"{root}/{scene}_{rung}"
    return f"{WS}/gray/tmp/final/{scene}_{rung}"


def score(run, mask, idx):
    renders = sorted(glob.glob(f"{run}/test/*/rad_tan_thin_prism_fisheye/renders/*.png"))
    gts = sorted(glob.glob(f"{run}/test/*/rad_tan_thin_prism_fisheye/gt/*.png"))
    if not renders or len(renders) != len(gts):
        return None
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
        "per_view": [float(v) for v in per_view],
        "names": [os.path.basename(p) for p in renders],
        "rings": [-10.0 * math.log10(max(ring_se[k] / (ring_n[k] * 3.0), 1e-12))
                  for k in range(NUM_RINGS)],
    }


def main():
    scenes = sys.argv[1:] or ["tunnel"]
    dest = os.path.join(HERE, "rungs.json")
    out = json.load(open(dest)) if os.path.exists(dest) else {}
    for scene in scenes:
        mask = mask_for(scene)
        idx, edges, _ = ring_map(mask, NUM_RINGS)
        entry = out.setdefault(scene, {"ring_edges": edges.tolist(), "rungs": {}})
        for rung in RUNGS:
            result = score(run_dir(scene, rung), mask, idx)
            if result is None:
                print(f"  -- {scene}/{rung}: not available yet")
                continue
            entry["rungs"][rung] = result
            print(f"{scene:11s} {rung:16s} {result['psnr']:7.3f}   rings "
                  + " ".join(f"{v:6.2f}" for v in result["rings"]), flush=True)
    with open(dest, "w") as handle:
        json.dump(out, handle)
    print("wrote", dest)


if __name__ == "__main__":
    main()
