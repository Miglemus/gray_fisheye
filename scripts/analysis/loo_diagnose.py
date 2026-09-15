#!/usr/bin/env python
"""Why does the leave-one-out transfer LOSE dB when the donor tensors look consistent?

The LOO result (15k, myscenes, `noncentral`) was negative: mean transfer - off = -0.18 dB,
and tunnel fell 1.43 dB BELOW no camera model at all. Three things are already established
and are NOT the explanation:

  * the machinery is correct -- initialising tunnel from its OWN model, frozen, reproduces
    the source run to 28.78 vs 28.73 (`tunnel_selfinit_from0`, inside the 0.068 dB floor);
  * `--camera_opt_from_iter` is irrelevant for a frozen model -- 0 vs 3000 moved tunnel by
    0.02 dB and classroom by 0.09 dB;
  * the tensors were loaded (4 tensors, 111 frozen parameters, 3 dropped groups).

So the donor ray field is genuinely wrong for the held-out scene. This script asks WHICH
PART of it is wrong, in the only units that matter -- the displacement it puts on a ray.

WHAT IT MEASURES, per scene s:

  z profile        z_s(theta) - z_s(0), the gauge-free pupil shift `camera_model.py:746`
                   subtracts. Reported both in raw world units and normalised by the scene
                   radius, because `--camera_model_init_z_scale scene_radius` ASSUMES the
                   radius is the world-unit-per-metre proxy. If the normalised profiles
                   disagree more than the raw ones, that assumption is the bug.
  angular residual dtheta_s(theta), channel 0, in radians and in rim pixels through the
                   local plate scale (`calib_consistency.radial_poly` / `plate_scale`).
  donor error      the same quantities for the mean of the OTHER SIX, after the exact
                   z_scale the run applied, minus s's own -- i.e. how wrong the installed
                   model was, per part.

The prediction to check: the per-part donor error should ORDER the scenes the way the PSNR
loss does (reception recovered 95% of the gain, tunnel lost 1.43 dB).

usage: python scripts/analysis/loo_diagnose.py
"""

import glob
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import scripts.analysis.calib_consistency as C  # noqa: E402
from gray.camera_model import bspline_eval  # noqa: E402
from gray.config import LENS_PARAMETERS, scene_radius_from_cameras_json  # noqa: E402

SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
RUNS = "/workspace/gray/tmp/final/{scene}_noncentral"
LOO = "/workspace/gray/worktrees/noncentral-camera/tmp/loo/noncentral"
# * test PSNR at 15k, masked protocol, from the LOO batch and the ladder.
OFF = {"atrium": 27.878, "classroom": 28.562, "forest": 19.565, "library": 31.580,
       "reception": 27.277, "tunnel": 28.538, "workshop": 27.198}
SELF = {"atrium": 28.0358, "classroom": 28.9728, "forest": 19.6451, "library": 31.8535,
        "reception": 27.7514, "tunnel": 28.7286, "workshop": 27.9415}
TRANSFER = {"atrium": 27.8025, "classroom": 28.1240, "forest": 19.6179, "library": 31.6919,
            "reception": 27.7292, "tunnel": 27.1032, "workshop": 27.2839}
THETA = np.linspace(0.0, np.pi / 2, 200)  # * the model's own theta01 domain, [0, pi/2]


def blocks(checkpoint):
    out = {}
    with safe_open(checkpoint, "pt") as handle:
        for key in handle.keys():
            if key.startswith("camera_model.lenses.1."):
                name = key.split(".", 3)[3]
                if name in LENS_PARAMETERS:
                    out[name] = handle.get_tensor(key).float()
    return out


def latest(run):
    return sorted(glob.glob(os.path.join(run, "gaussians_*.safetensors")))[-1]


def profile(weights, theta):
    """B-spline evaluated on [0, pi/2] -> theta01, channel 0, gauge removed at theta=0."""
    t01 = torch.from_numpy(theta / (np.pi / 2)).float()
    values = bspline_eval(weights, t01)[0].detach().numpy()
    return values - bspline_eval(weights, torch.zeros(()))[0].item()


def main():
    own, radii = {}, {}
    for scene in SCENES:
        run = RUNS.format(scene=scene)
        own[scene] = blocks(latest(run))
        radii[scene] = scene_radius_from_cameras_json(os.path.join(run, "cameras.json"))

    z_raw = {s: profile(own[s]["z_weights"], THETA) for s in SCENES}
    dtheta = {s: profile(own[s]["theta_weights"], THETA) for s in SCENES}

    print("=" * 96)
    print("PART 1 -- is the scene radius the right unit for z?")
    print("  z(theta)-z(0) at the rim, raw world units vs normalised by the scene radius.")
    print("  If `scene_radius` were the world-units-per-metre proxy the loader assumes,")
    print("  the NORMALISED column would be the tight one.")
    print(f"  {'scene':11s} {'radius':>8s} {'z_rim raw':>12s} {'z_rim/radius':>14s}")
    for s in SCENES:
        print(f"  {s:11s} {radii[s]:8.3f} {z_raw[s][-1]:12.3e} {z_raw[s][-1]/radii[s]:14.3e}")
    for label, values in (("raw", [z_raw[s][-1] for s in SCENES]),
                          ("normalised", [z_raw[s][-1] / radii[s] for s in SCENES])):
        v = np.array(values)
        print(f"  spread {label:11s}: mean {v.mean():+.3e}  sd {v.std():.3e}  "
              f"sd/|mean| {v.std()/abs(v.mean()):.2f}  max/min {v.max()/v.min():.2f}")

    print()
    print("=" * 96)
    print("PART 2 -- how wrong was the model each transfer run actually installed?")
    print("  donor = mean of the other six, z converted through the SAME chain the run used")
    print("  (into the reference donor's units, then x target_radius/reference_radius).")
    header = (f"  {'scene':11s} {'z_rim own':>11s} {'z_rim donor':>12s} {'z err':>10s} "
              f"{'|dth| own':>10s} {'|dth| donor':>12s} {'dth err px':>11s} "
              f"{'lost dB':>8s} {'recovered':>10s}")
    print(header)
    rows = []
    for held_out in SCENES:
        others = [s for s in SCENES if s != held_out]
        reference = others[0]
        stacked = {}
        for name in LENS_PARAMETERS:
            stack = []
            for scene in others:
                tensor = own[scene][name]
                if name == "z_weights":
                    tensor = tensor * (radii[reference] / radii[scene])
                stack.append(tensor)
            stacked[name] = torch.stack(stack).mean(0)
        # * exactly what the loader did: source_scene_radius is the reference donor's.
        z_scale = radii[held_out] / radii[reference]
        donor_z = profile(stacked["z_weights"], THETA) * z_scale
        donor_dtheta = profile(stacked["theta_weights"], THETA)

        # * rim pixels for the angular residual, at the run's -r 4 plate scale.
        px = C.plate_scale(*C.fisheye_intrinsics(held_out)) if hasattr(C, "fisheye_intrinsics") \
            else None
        dth_err_px = float("nan")
        if px is not None:
            dth_err_px = abs(donor_dtheta[-1] - dtheta[held_out][-1]) * px

        z_err = donor_z[-1] - z_raw[held_out][-1]
        gain = SELF[held_out] - OFF[held_out]
        recovered = (TRANSFER[held_out] - OFF[held_out]) / gain if abs(gain) > 1e-9 else 0.0
        rows.append((held_out, z_raw[held_out][-1], donor_z[-1], z_err,
                     abs(dtheta[held_out][-1]), abs(donor_dtheta[-1]),
                     donor_dtheta[-1] - dtheta[held_out][-1],
                     TRANSFER[held_out] - OFF[held_out], recovered))
        print(f"  {held_out:11s} {z_raw[held_out][-1]:11.3e} {donor_z[-1]:12.3e} "
              f"{z_err:+10.3e} {abs(dtheta[held_out][-1]):10.3e} "
              f"{abs(donor_dtheta[-1]):12.3e} {donor_dtheta[-1]-dtheta[held_out][-1]:+11.3e} "
              f"{TRANSFER[held_out]-OFF[held_out]:+8.3f} {recovered:9.0%}")

    print()
    print("  correlations against `transfer - off` (n=7, Pearson):")
    lost = np.array([r[7] for r in rows])
    for label, values in (("|z error|", np.abs([r[3] for r in rows])),
                          ("|dtheta error|", np.abs([r[6] for r in rows])),
                          ("|z error| / |z own|",
                           np.abs([r[3] / r[1] if r[1] else np.nan for r in rows]))):
        v = np.asarray(values, dtype=float)
        ok = np.isfinite(v)
        if ok.sum() > 2:
            r = np.corrcoef(v[ok], lost[ok])[0, 1]
            print(f"    {label:22s} r = {r:+.3f}")

    json_out = os.path.join(LOO, "diagnose.json")
    with open(json_out, "w") as handle:
        json.dump({"scenes": SCENES, "radii": radii,
                   "z_rim_raw": {s: float(z_raw[s][-1]) for s in SCENES},
                   "rows": [list(map(lambda x: float(x) if not isinstance(x, str) else x, r))
                            for r in rows]}, handle, indent=1)
    print(f"\n  wrote {json_out}")


if __name__ == "__main__":
    main()
