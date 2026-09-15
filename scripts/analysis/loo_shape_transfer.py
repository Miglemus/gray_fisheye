#!/usr/bin/env python
"""Leave-one-out, corrected: transfer the SHAPE of z(theta), refit its amplitude.

WHY THE FIRST LOO FAILED. `scripts/analysis/loo_camera_model.py` averaged the seven
`noncentral` models as they sit on disk and converted `z` through the scene radius. It lost
0.18 dB on average and 1.43 dB on tunnel. Three candidate explanations were killed first:
the load/freeze machinery reproduces its source run to 0.05 dB (`tunnel_selfinit_from0`),
`--camera_opt_from_iter` moves a frozen model by 0.02 dB, and the tensors were loaded.

WHAT `loo_diagnose.py` FOUND. The rim value of z(theta)-z(0) spans 5.4x across the seven
scenes (4.58e-3 atrium .. 2.49e-2 workshop) and the scene radius does NOT explain it --
normalising by the radius makes the spread WORSE (sd/|mean| 0.75 -> 0.83, max/min 5.4 ->
6.5). That is expected in hindsight: the radius measures how far the camera walked, not how
many world units are in a millimetre, and COLMAP's scale is arbitrary per reconstruction.
Meanwhile the SHAPE of z(theta) agrees across all seven -- pairwise cosine 0.971 mean, 0.903
min, 0/21 negative -- and after one optimal scalar the donor mean explains each held-out
scene's profile to 7.5-17.1% (atrium 34.2%). The amplitude in world units is a nuisance
parameter of the reconstruction; the profile is the lens.

So the transfer to test is: donor SHAPE, one scalar amplitude.

WHAT THIS SCRIPT WRITES, per held-out scene s:

  donorshape_for_s   angular = mean of the other six (radians, unit-free, no conversion);
                     z = unit-norm mean of the other six SHAPES, times the scalar that
                     best fits s's own profile. Nothing about s enters the shape.
  donorang_for_s     the same angular part, z zeroed -- what does the angular residual
                     alone transfer?
  donorz_for_s       the z part alone, angular zeroed -- what does non-centrality alone
                     transfer? This is the one the paper's claim rests on.

THE SCALAR, honestly. alpha_s is read off s's OWN trained camera model, which was fit on
s's TRAIN views only -- the test views were never seen. So there is no test leakage, but
the protocol IS "shape transferred, amplitude refit on the target reconstruction", not
"nothing about s touched the model". The 1-parameter no-leakage version (learn alpha by
SGD against the training loss) is the follow-up; this run establishes whether the shape is
worth that effort.

Because the amplitude is baked into the checkpoint in the TARGET scene's own world units,
these runs use `--camera_model_init_z_scale none`: the radius conversion is the thing under
indictment and must not be reapplied.

usage:
    python scripts/analysis/loo_shape_transfer.py            # build + print the queue
    python scripts/analysis/loo_shape_transfer.py --only tunnel classroom reception
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gray.camera_model import bspline_eval  # noqa: E402
from gray.config import LENS_PARAMETERS  # noqa: E402

SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
# * `noncentral` donors live in the shared tree; `z_only` donors only in this worktree.
SOURCE_TREES = {
    "noncentral": "/workspace/gray/tmp/final/{scene}_noncentral",
    "z_only": "/workspace/gray/worktrees/noncentral-camera/tmp/final/{scene}_z_only",
}
OUT = "/workspace/gray/worktrees/noncentral-camera/tmp/loo/shape"
THETA = np.linspace(0.0, np.pi / 2, 200)
ANGULAR = ("omega", "theta_weights", "phi_weights")


def latest(run):
    hits = sorted(glob.glob(os.path.join(run, "gaussians_*.safetensors")))
    if not hits:
        raise SystemExit(f"{run}: no gaussians_*.safetensors")
    return hits[-1]


def blocks(checkpoint):
    out = {}
    with safe_open(checkpoint, "pt") as handle:
        for key in handle.keys():
            if key.startswith("camera_model.lenses.1."):
                name = key.split(".", 3)[3]
                if name in LENS_PARAMETERS:
                    out[name] = handle.get_tensor(key).float()
    return out


def gauge_free_profile(weights):
    """z(theta) - z(0) on [0, pi/2], the quantity `camera_model.py:746` actually applies."""
    t01 = torch.from_numpy(THETA / (np.pi / 2)).float()
    values = bspline_eval(weights, t01)[0].detach().numpy()
    return values - float(bspline_eval(weights, torch.zeros(()))[0])


def build(scenes, rung="noncentral"):
    own = {s: blocks(latest(SOURCE_TREES[rung].format(scene=s))) for s in SCENES}
    profiles = {s: gauge_free_profile(own[s]["z_weights"]) for s in SCENES}

    written = []
    for held_out in scenes:
        others = [s for s in SCENES if s != held_out]
        # * unit-norm each donor's z BEFORE averaging: the raw tensors are in seven
        # * different world-unit scales, so their plain mean is a mean of seven units.
        shape_weights = torch.stack(
            [own[s]["z_weights"] / np.linalg.norm(profiles[s]) for s in others]).mean(0)
        shape_profile = gauge_free_profile(shape_weights)
        alpha = float(np.dot(shape_profile, profiles[held_out])
                      / np.dot(shape_profile, shape_profile))
        residual = (np.linalg.norm(alpha * shape_profile - profiles[held_out])
                    / np.linalg.norm(profiles[held_out]))
        angular = {n: torch.stack([own[s][n] for s in others]).mean(0) for n in ANGULAR}
        z = shape_weights * alpha

        suffix = "" if rung == "noncentral" else f"_{rung}"
        for tag, tensors in (
                (f"donorshape{suffix}", {**angular, "z_weights": z}),
                (f"donorang{suffix}", {**angular, "z_weights": torch.zeros_like(z)}),
                (f"donorz{suffix}", {**{n: torch.zeros_like(angular[n]) for n in ANGULAR},
                                     "z_weights": z})):
            destination = os.path.join(OUT, f"{tag}_for_{held_out}")
            os.makedirs(destination, exist_ok=True)
            save_file({f"camera_model.lenses.1.{n}": t.contiguous()
                       for n, t in tensors.items()},
                      os.path.join(destination, "gaussians_15000.safetensors"))
            with open(os.path.join(destination, "config.json"), "w") as handle:
                json.dump({"camera_opt": rung, "iterations": 15000,
                           "_note": f"{tag}: donors {others}; z shape = unit-norm mean, "
                                    f"alpha={alpha:.6g} fit on {held_out}'s own TRAIN-fit "
                                    f"profile; z already in {held_out}'s world units"},
                          handle, indent=1)
            written.append((held_out, tag, destination))
        print(f"  {held_out:11s} alpha {alpha:10.4g}   shape residual {residual:6.1%}")
    return written


def queue_lines(written, rung="noncentral"):
    env = ("PATH=/workspace/gray/.venv/bin:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID "
           "CUDA_VISIBLE_DEVICES=1")
    root = "/workspace/gray/worktrees/noncentral-camera"
    # * other sessions run GPU-1 jobs outside the gpu1 group, so the lane alone does not
    # * keep the card to one job; wait for free VRAM as well.
    wait = ('for i in $(seq 1 90); do f=$(nvidia-smi --query-gpu=memory.free '
            '--format=csv,noheader,nounits -i 1); [ "$f" -gt 20000 ] && break; sleep 60; done')
    lines = []
    for held_out, tag, donor in written:
        model_path = f"tmp/loo/shape/{held_out}_{tag}"
        lines.append(
            f"pueue add --group gpu1 --priority 100 --print-task-id -- "
            f"'cd {root} && export {env} && {wait} && "
            f"bash scripts/train_myscenes.sh {held_out} {model_path} -y "
            f"--camera_opt {rung} --camera_opt_from_iter 0 "
            f"--camera_model_init {donor} --camera_model_init_rung {rung} "
            f"--camera_model_init_z_scale none --camera_model_freeze'")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=SCENES)
    parser.add_argument("--rung", default="noncentral", choices=sorted(SOURCE_TREES))
    parser.add_argument("--tags", nargs="*", default=["donorshape", "donorang", "donorz"])
    args = parser.parse_args()
    print(f"building donor shapes from the `{args.rung}` fits:")
    written = [w for w in build(args.only, args.rung) if w[1] in args.tags]
    lines = queue_lines(written, args.rung)
    path = os.path.join(OUT, "queue.sh")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nwrote {len(lines)} queue lines to {path}")


if __name__ == "__main__":
    main()
