#!/usr/bin/env python
"""Build the leave-one-out camera model for each scene, and the pueue lines to test it.

THE QUESTION. `z(theta)` and the angular residual buy +0.33 dB on myscenes. Is that a
property of the LENS -- one physical object, calibrated seven times -- or is each scene's
model absorbing that scene's own calibration error and its own gaussians? Every number so
far is compatible with both readings, because every model was trained on the scene it was
scored on.

THE TEST. For each held-out scene s, average the camera models learned on the OTHER SIX,
install it with `--camera_model_init`, `--camera_model_freeze` it, and train the gaussians
on s. Nothing about s ever touched the camera model. Then:

    transfer - off                   how much of the gain survives a model that never saw s
    ---------------------            = the fraction that is lens, not scene
    self-trained - off

A lens property transfers. Per-scene absorption cannot: the six donors disagree about s's
own calibration error by construction, so their mean cancels it.

WHY AN AVERAGE AND NOT ONE DONOR. A single donor confounds "does it transfer" with "is this
particular pair similar". The mean of six is the closest thing to "the lens" that seven
independent fits can produce, and it is what the shared/scene-specific split of
`calib_consistency.py` (1.56 on myscenes) says should carry.

THE UNIT TRAP, and why this script exists at all rather than a shell loop. `z(theta)` is in
RAW COLMAP WORLD UNITS: the same physical pupil shift is a different number in each scene's
reconstruction. Averaging the seven z tensors as they sit on disk averages seven different
units and is meaningless. So every donor's z is converted into the units of ONE reference
donor before averaging, and the synthetic checkpoint is written next to that reference's
`cameras.json` -- which is what `--camera_model_init_z_scale scene_radius` then reads to
convert into the target's units. The angular tensors (`omega`, `theta_weights`,
`phi_weights`) are angles and need no conversion.

usage:
    python scripts/analysis/loo_camera_model.py                      # build + print the queue
    python scripts/analysis/loo_camera_model.py --rung noncentral    # which rung to transfer
"""

import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gray.config import LENS_PARAMETERS, scene_radius_from_cameras_json  # noqa: E402

SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
# * The myscenes directory layout, same split `scripts/train_myscenes.sh` uses.
UNDISTORTION = {"atrium", "library", "reception", "tunnel"}
SOURCES = "/workspace/gray/tmp/final/{scene}_{rung}"
OUT = "/workspace/gray/worktrees/noncentral-camera/tmp/loo/{rung}"


def latest_checkpoint(run_dir):
    hits = sorted(glob.glob(os.path.join(run_dir, "gaussians_*.safetensors")))
    if not hits:
        raise SystemExit(f"{run_dir}: no gaussians_*.safetensors")
    return hits[-1]


def read_lens_blocks(checkpoint):
    """{uid: {name: tensor}} for the four transferable lens tensors."""
    blocks = {}
    with safe_open(checkpoint, "pt") as handle:
        for key in handle.keys():
            if not key.startswith("camera_model.lenses."):
                continue
            _, _, uid, name = key.split(".", 3)
            if name not in LENS_PARAMETERS:
                continue
            blocks.setdefault(int(uid), {})[name] = handle.get_tensor(key).float()
    return blocks


def build(rung):
    donors = {}
    for scene in SCENES:
        run = SOURCES.format(scene=scene, rung=rung)
        checkpoint = latest_checkpoint(run)
        cameras_json = os.path.join(run, "cameras.json")
        donors[scene] = {
            "run": run,
            "blocks": read_lens_blocks(checkpoint),
            "radius": scene_radius_from_cameras_json(cameras_json),
        }
        uids = sorted(donors[scene]["blocks"])
        if uids != [1]:
            raise SystemExit(f"{scene}: expected a single lens uid 1, found {uids}")

    print(f"donor scene radii ({rung}):")
    for scene, d in donors.items():
        z = d["blocks"][1]["z_weights"]
        print(f"  {scene:10s} radius {d['radius']:9.4f}   |z| max {z.abs().max():.6f}"
              f"   z/radius max {(z.abs().max() / d['radius']):.3e}")

    written = []
    for held_out in SCENES:
        others = [s for s in SCENES if s != held_out]
        # * The reference donor defines the units of the synthetic checkpoint; its
        # * cameras.json is copied next to it so the loader can recover that radius.
        reference = others[0]
        reference_radius = donors[reference]["radius"]

        averaged = {}
        for name in LENS_PARAMETERS:
            stack = []
            for scene in others:
                tensor = donors[scene]["blocks"][1][name]
                if name == "z_weights":
                    # * into the reference scene's world units
                    tensor = tensor * (reference_radius / donors[scene]["radius"])
                stack.append(tensor)
            averaged[name] = torch.stack(stack).mean(0)

        destination = os.path.join(OUT.format(rung=rung), f"donor_for_{held_out}")
        os.makedirs(destination, exist_ok=True)
        save_file({f"camera_model.lenses.1.{n}": t.contiguous() for n, t in averaged.items()},
                  os.path.join(destination, "gaussians_15000.safetensors"))
        # * cameras.json defines the units of the z above; config.json tells the loader the
        # * rung, which is NOT recoverable from the tensors (every rung writes all four).
        with open(os.path.join(donors[reference]["run"], "cameras.json")) as handle:
            cameras = json.load(handle)
        with open(os.path.join(destination, "cameras.json"), "w") as handle:
            json.dump(cameras, handle)
        with open(os.path.join(destination, "config.json"), "w") as handle:
            json.dump({"camera_opt": rung, "iterations": 15000,
                       "_note": f"synthetic: mean of {others}, z in {reference}'s world units"},
                      handle, indent=1)
        written.append((held_out, destination, reference, len(others)))

    print(f"\nwrote {len(written)} donor models under {OUT.format(rung=rung)}")
    return written


def queue_lines(written, rung):
    env = ("PATH=/workspace/gray/.venv/bin:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID "
           "CUDA_VISIBLE_DEVICES=1")
    root = "/workspace/gray/worktrees/noncentral-camera"
    print("\n# --- paste these; they are ONE line each ---")
    for held_out, donor, _, _ in written:
        model_path = f"tmp/loo/{rung}/{held_out}_transfer"
        # * A free-VRAM wait as well as the group: other sessions run GPU-1 jobs in the
        # * `default` group, so the gpu1 lane alone does not keep the card to one job.
        wait = ('for i in $(seq 1 60); do f=$(nvidia-smi --query-gpu=memory.free '
                '--format=csv,noheader,nounits -i 1); [ "$f" -gt 17000 ] && break; sleep 60; done')
        print(f"pueue add --group gpu1 --print-task-id -- 'cd {root} && export {env} && {wait} && "
              f"bash scripts/train_myscenes.sh {held_out} {model_path} -y "
              f"--camera_opt {rung} --camera_opt_from_iter 3000 "
              f"--camera_model_init {donor} --camera_model_init_rung {rung} "
              f"--camera_model_init_z_scale scene_radius --camera_model_freeze'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", default="noncentral")
    args = parser.parse_args()
    queue_lines(build(args.rung), args.rung)


if __name__ == "__main__":
    main()
