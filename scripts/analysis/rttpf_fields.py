"""Do the two rungs move the image the same way? Full 2D displacement fields.

`rttpf_vs_noncentral.py` compares the radial slice only, which is the wrong lens for a
correction that is largely a principal-point / decentring term -- on tunnel the two radial
curves are *anti*-correlated while the two rungs score within 0.03 dB of each other, so the
slice says "different" where the image says "the same". This script compares what actually
matters: the per-pixel displacement each rung applies to the image, as a 2D vector field.

Both fields are produced by the SAME code the renderer uses -- `rttpf_solve` for the
intrinsic rung, `LensResidual` + `harmonics` for the residual rungs -- evaluated on an
analytic (theta, phi) grid rather than a probed pixel grid, so no camera model is
re-implemented here and no raytracer (hence no OptiX) is needed.

What is NOT in the comparison: the non-central `z(theta)` term. It moves the ray *origin*,
so its image-space effect is depth-dependent (`z sin(theta) / t`) and is not a displacement
field at all. It is reported separately as the angular shift it would cause at the scene's
median depth. That term is precisely the part no re-calibration can reach, so keeping it out
of the "did the re-fit find the same correction" comparison is the point, not an omission.

usage:
  python scripts/analysis/rttpf_fields.py --root tmp/r8_control --scenes tunnel workshop
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import safetensors.numpy
import torch

sys.path.insert(0, "/workspace/gray/worktrees/rttpf-intrinsics")

from gray.camera_model import (  # noqa: E402
    LensResidual,
    RTTPF_NAMES,
    bspline_eval,
    rttpf_normalization,
    rttpf_solve,
    skew_rotate,
)

SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
NUM_THETA, NUM_PHI = 128, 64


def grid(cutoff_deg):
    theta = torch.linspace(1e-4, np.deg2rad(cutoff_deg), NUM_THETA, device="cuda")
    phi = torch.linspace(0.0, 2.0 * np.pi, NUM_PHI + 1, device="cuda")[:-1]
    return theta.unsqueeze(1).expand(NUM_THETA, NUM_PHI), phi.unsqueeze(0).expand(NUM_THETA, NUM_PHI)


def plate_scale(theta, params):
    "Local dr/dtheta [px/rad] and r(theta) [px] of the delivered calibration."
    theta2 = theta * theta
    growth = torch.ones_like(theta)
    slope = torch.ones_like(theta)
    power = torch.ones_like(theta)
    for index in range(6):
        power = power * theta2
        growth = growth + params[4 + index] * power
        slope = slope + (2 * index + 3) * params[4 + index] * power
    return params[0] * slope, params[0] * growth * theta


def latest(run_dir, stem, extension="csv"):
    files = sorted(glob.glob(os.path.join(run_dir, f"{stem}_*.{extension}")))
    if not files:
        raise FileNotFoundError(f"no {stem}_*.{extension} in {run_dir}")
    return files[-1]


def colmap_params(rttpf_run):
    base, delta = {}, {}
    with open(latest(rttpf_run, "camera_intrinsics")) as handle:
        for row in csv.DictReader(handle):
            base[row["name"]] = float(row["colmap"])
            delta[row["name"]] = float(row["delta"])
    to_tensor = lambda d: torch.tensor(  # noqa: E731
        [d[name] for name in RTTPF_NAMES], dtype=torch.float32, device="cuda"
    )
    return to_tensor(base), to_tensor(delta)


def rttpf_field(run_dir, theta, phi):
    "(d_theta, d_phi) of the re-calibration, from its learned intrinsic deltas."
    params, deltas = colmap_params(run_dir)
    w0 = torch.stack([theta * phi.cos(), theta * phi.sin()], dim=-1)
    scales = rttpf_normalization(params)
    table = {
        "params": params,
        "w0": w0,
        "norm0": w0.norm(dim=-1),
        # * Rebuilt with the module's own projection, exactly as rttpf_tables does.
        "target": None,
        "scales": scales,
    }
    from gray.camera_model import rttpf_project

    table["target"] = rttpf_project(w0, params)
    # * Invert the coefficient -> delta map so the solve sees the trained camera.
    focal = torch.stack([params[0], params[1], params[0], params[1]])
    coefficients = torch.cat([deltas[:4] / focal, deltas[4:]]) * scales
    delta_theta, delta_phi, _ = rttpf_solve(coefficients, table)
    return delta_theta, delta_phi, params


def residual_field(run_dir, theta, phi, components):
    "(d_theta, d_phi) of a spline rung, from its checkpoint, minus the z(theta) term."
    state = safetensors.numpy.load_file(latest(run_dir, "gaussians", "safetensors"))
    prefix = next(k for k in state if k.endswith("theta_weights")).rsplit(".", 1)[0]
    knots = state[f"{prefix}.theta_weights"].shape[-1]
    lens = LensResidual(components, knots, state[f"{prefix}.z_weights"].shape[-1])
    with torch.no_grad():
        for name in ("omega", "theta_weights", "phi_weights", "z_weights"):
            getattr(lens, name).copy_(torch.from_numpy(state[f"{prefix}.{name}"]).cuda())

        theta01 = (theta / (np.pi / 2.0)).clamp(0.0, 1.0)
        spline_theta = bspline_eval(lens.theta_weights, theta01)
        spline_phi = bspline_eval(lens.phi_weights, theta01)
        delta_theta = lens.harmonics(spline_theta, phi.cos(), phi.sin())
        delta_phi = lens.harmonics(spline_phi, phi.cos(), phi.sin())

        # * The global tilt, converted to the same (d_theta, d_phi) coordinates by rotating
        # * the bearing and reading off the change in its spherical angles.
        bearing = torch.stack(
            [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()], dim=-1
        )
        tilted = skew_rotate(lens.omega, bearing)
        delta_theta = delta_theta + (
            torch.atan2(tilted[..., :2].norm(dim=-1), tilted[..., 2]) - theta
        )
        delta_phi = delta_phi + torch.atan2(
            bearing[..., 0] * tilted[..., 1] - bearing[..., 1] * tilted[..., 0],
            bearing[..., 0] * tilted[..., 0] + bearing[..., 1] * tilted[..., 1],
        )
        z_profile = bspline_eval(lens.z_weights, theta01)[0]
        z_profile = z_profile - bspline_eval(lens.z_weights, theta01.new_zeros(()))[0]
    return delta_theta, delta_phi, z_profile


def displacement(delta_theta, delta_phi, theta, params):
    "Angular residual -> image-plane displacement [px], through the LOCAL plate scale."
    scale, radius = plate_scale(theta, params)
    return torch.stack([delta_theta * scale, delta_phi * radius], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--pattern", default="{scene}_{rung}")
    parser.add_argument("--against", default="noncentral")
    parser.add_argument("--rttpf-rung", default="rttpf", help="name of the re-calibrated run")
    parser.add_argument("--components", nargs="+", default=["tilt", "radial", "ana", "z"])
    parser.add_argument("--cutoff-deg", type=float, default=85.5, help="the 0.95 mask edge")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    theta, phi = grid(args.cutoff_deg)
    print(f"{'scene':11s}{'|rttpf| px':>12s}{'|nc| px':>10s}{'cos':>7s}"
          f"{'explained':>11s}{'|nc-rttpf|':>12s}{'z rim px':>10s}")
    print("-" * 73)
    out = {}
    for scene in args.scenes:
        rttpf_run = os.path.join(args.root, args.pattern.format(scene=scene, rung=args.rttpf_rung))
        other_run = os.path.join(args.root, args.pattern.format(scene=scene, rung=args.against))
        try:
            fitted_theta, fitted_phi, params = rttpf_field(rttpf_run, theta, phi)
            residual_theta, residual_phi, z_profile = residual_field(
                other_run, theta, phi, tuple(args.components)
            )
        except (FileNotFoundError, StopIteration, KeyError) as error:
            print(f"  -- {scene}: {type(error).__name__} {error}")
            continue

        fitted = displacement(fitted_theta, fitted_phi, theta, params)
        residual = displacement(residual_theta, residual_phi, theta, params)
        # * Area weight: an equal-angle grid over-samples the centre of the disk.
        weight = (theta.sin() * plate_scale(theta, params)[0]).unsqueeze(-1)
        inner = float((fitted * residual * weight).sum())
        fitted_norm = float(((fitted**2) * weight).sum()) ** 0.5
        residual_norm = float(((residual**2) * weight).sum()) ** 0.5
        cosine = inner / max(fitted_norm * residual_norm, 1e-30)
        gap = float((((residual - fitted) ** 2) * weight).sum()) ** 0.5
        scale, _ = plate_scale(theta, params)
        # * z is radial, so any azimuth of the last theta row carries the whole profile.
        rim = float((z_profile[-1, 0] * theta[-1, 0].sin() * scale[-1, 0]).abs())
        out[scene] = {
            "rttpf_rms_px": fitted_norm / float(weight.sum()) ** 0.5,
            "noncentral_rms_px": residual_norm / float(weight.sum()) ** 0.5,
            "cosine": cosine,
            "explained": 1.0 - (gap / max(residual_norm, 1e-30)) ** 2,
            "gap_rms_px": gap / float(weight.sum()) ** 0.5,
            "z_rim_over_unit_depth_px": rim,
        }
        print(f"{scene:11s}{out[scene]['rttpf_rms_px']:12.3f}{out[scene]['noncentral_rms_px']:10.3f}"
              f"{cosine:7.2f}{out[scene]['explained']:11.2f}{out[scene]['gap_rms_px']:12.3f}"
              f"{rim:10.3f}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(out, handle, indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
