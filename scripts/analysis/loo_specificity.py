#!/usr/bin/env python
"""Does the transferred z(theta) measure THIS lens, or would any smooth profile do?

THE OBJECTION THIS ANSWERS. `loo_shape_transfer.py` showed that the non-central profile
averaged over six myscenes scenes is worth +0.250 dB (p=0.007, 7/7) on the seventh, which it
never saw. A reviewer will ask the obvious follow-up: is that because the profile is a
property of the Fujinon fisheye, or because *any* smooth radial ray-origin shift of about the
right magnitude helps a fisheye reconstruction? If the second, the result measures a generic
prior and the paper's claim evaporates.

TWO CONTROLS, both of which must FAIL for the main result to mean what it says:

  wronglens   the z shape averaged over the seven mip-NeRF 360 scenes -- the same rung, the
              same optimiser, the same code, fit on a PINHOLE camera where a pupil shift is
              unphysical by construction -- rescaled to the magnitude the held-out scene's
              own fit chose. Same displacement, wrong shape.
  flip        the CORRECT donor shape with the amplitude negated. If -z helps as much as +z,
              the model is not recovering a direction.

THE AMPLITUDE, and why it is matched by NORM and not by least squares. A least-squares alpha
against a shape that is orthogonal to the target shrinks to ~0, so the control would install
nothing and "pass" for a trivial reason. Matching the norm installs the same magnitude of
ray displacement and varies only the shape, which is the thing under test.

WHAT IS ALREADY KNOWN WITHOUT A GPU, and is arguably the stronger figure. The same code on
seven scenes of one fisheye produces the same profile every time (pairwise cosine 0.979 mean,
0.938 min, 0/21 negative). On seven pinhole scenes it produces mutually orthogonal noise:
cosine 0.142 mean, -0.965 min, 9/21 pairs NEGATIVE. And the two population means are
orthogonal to each other (-0.015). The method finds a consistent lens where there is one and
finds nothing where there is not -- which is exactly the behaviour a measurement should have.
These runs put that in dB.

usage:
    python scripts/analysis/loo_specificity.py             # build + write the queue
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "loo_shape_transfer", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "loo_shape_transfer.py"))
LST = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LST)

MYSCENES = LST.SCENES
MIP360 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
MY_RUNS = "/workspace/gray/worktrees/noncentral-camera/tmp/final/{scene}_z_only"
MIP_RUNS = "/workspace/gray/worktrees/noncentral-camera/tmp/mipnerf360/{scene}_noncentral"
OUT = "/workspace/gray/worktrees/noncentral-camera/tmp/loo/specificity"
ANGULAR = LST.ANGULAR


def unit(profile):
    return profile / np.linalg.norm(profile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", nargs="*", default=["wronglens", "flip"])
    args = parser.parse_args()

    my = {s: LST.blocks(LST.latest(MY_RUNS.format(scene=s))) for s in MYSCENES}
    my_profile = {s: LST.gauge_free_profile(my[s]["z_weights"]) for s in MYSCENES}
    mip = {s: LST.blocks(LST.latest(MIP_RUNS.format(scene=s))) for s in MIP360}
    mip_profile = {s: LST.gauge_free_profile(mip[s]["z_weights"]) for s in MIP360}

    # * one "wrong lens" shape: the mean of the seven pinhole fits, unit-normed first so the
    # * mean is a shape and not a mean of seven arbitrary world-unit scales.
    wrong_weights = torch.stack(
        [mip[s]["z_weights"] / np.linalg.norm(mip_profile[s]) for s in MIP360]).mean(0)
    wrong_profile = LST.gauge_free_profile(wrong_weights)

    written = []
    for held_out in MYSCENES:
        others = [s for s in MYSCENES if s != held_out]
        right_weights = torch.stack(
            [my[s]["z_weights"] / np.linalg.norm(my_profile[s]) for s in others]).mean(0)
        right_profile = LST.gauge_free_profile(right_weights)
        target_norm = np.linalg.norm(my_profile[held_out])
        # * least-squares alpha, the one the main experiment used, for the flip control
        alpha_ls = float(np.dot(right_profile, my_profile[held_out])
                         / np.dot(right_profile, right_profile))

        variants = {
            "wronglens": wrong_weights * (target_norm / np.linalg.norm(wrong_profile)),
            "flip": right_weights * (-alpha_ls),
        }
        cos = float(np.dot(unit(wrong_profile), unit(my_profile[held_out])))
        print(f"  {held_out:11s} cos(pinhole shape, own) {cos:+.3f}   "
              f"alpha_ls {alpha_ls:9.4g}   target |z| {target_norm:.4g}")

        for tag in args.controls:
            destination = os.path.join(OUT, f"{tag}_for_{held_out}")
            os.makedirs(destination, exist_ok=True)
            tensors = {n: torch.zeros_like(my[held_out][n]) for n in ANGULAR}
            tensors["z_weights"] = variants[tag]
            save_file({f"camera_model.lenses.1.{n}": t.contiguous()
                       for n, t in tensors.items()},
                      os.path.join(destination, "gaussians_15000.safetensors"))
            with open(os.path.join(destination, "config.json"), "w") as handle:
                json.dump({"camera_opt": "z_only", "iterations": 15000,
                           "_note": f"{tag} control for {held_out}; "
                                    + ("shape = mean of the 7 mip-NeRF 360 PINHOLE fits, "
                                       "norm-matched to the target's own |z|"
                                       if tag == "wronglens" else
                                       "correct donor shape with the amplitude NEGATED")},
                          handle, indent=1)
            written.append((held_out, tag, destination))

    lines = LST.queue_lines(written, "z_only")
    path = os.path.join(OUT, "queue.sh")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nwrote {len(lines)} queue lines to {path}")


if __name__ == "__main__":
    main()
