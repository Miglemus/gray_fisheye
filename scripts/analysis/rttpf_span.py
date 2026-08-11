"""Expressible vs findable: project the learned residual onto the rttpf parameter span.

`residual_expressible.py` reports that 98.6 % of the learned *radial* correction lies in the
span of rttpf's `[t, t^3, t^5, t^7, t^9]` basis, and IMPLEMENTATION.md turns that into
"~60 % of the gain is re-calibration". The direct control (`--camera_opt rttpf`, 7 scenes)
recovers only 25 %. Both cannot be right, and this script says which.

The two claims differ in three ways, each of which is separated here:

  1. RADIAL ONLY vs the full 2D field. The residual has k=1 (decentring) and k=2 (anamorphic)
     orders too; rttpf answers them with `p0,p1` and `s0..s3`, which is a much smaller basis
     than the spline's. Fitting only the radial slice cannot see that.
  2. 5 PARAMETERS vs 16. The radial analysis moved `fx, k1..k4`; the real control moves all
     sixteen, so the span it can reach is strictly larger.
  3. EXPRESSIBLE vs FINDABLE. Even a correction inside the span need not be what photometric
     descent converges to. Comparing the least-squares optimum against the coefficients the
     control actually learned separates the model class from the optimizer.

The rttpf basis is obtained by *linearizing the renderer's own solver*: perturb one
normalized coefficient, run `rttpf_solve`, divide by the perturbation. So the span measured
here is the span the control could actually have reached, not an idealized one.

usage:
  python scripts/analysis/rttpf_span.py --root tmp/r4_paired
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/gray/worktrees/rttpf-intrinsics")
sys.path.insert(0, "/workspace/gray/worktrees/rttpf-intrinsics/scripts/analysis")

from gray.camera_model import NUM_RTTPF_PARAMS, rttpf_normalization, rttpf_project, rttpf_solve  # noqa: E402
from rttpf_fields import SCENES, colmap_params, grid, plate_scale, residual_field  # noqa: E402

PROBE = 1e-3  # * normalized-coefficient step for the linearization


def rttpf_basis(params, theta, phi):
    "[16, ..., 2] displacement fields, one per normalized rttpf coefficient, in px."
    w0 = torch.stack([theta * phi.cos(), theta * phi.sin()], dim=-1)
    table = {
        "params": params,
        "w0": w0,
        "norm0": w0.norm(dim=-1),
        "target": rttpf_project(w0, params),
        "scales": rttpf_normalization(params),
    }
    scale, radius = plate_scale(theta, params)
    columns = []
    for index in range(NUM_RTTPF_PARAMS):
        coefficients = torch.zeros(NUM_RTTPF_PARAMS, device=theta.device)
        coefficients[index] = PROBE
        with torch.no_grad():
            delta_theta, delta_phi, _ = rttpf_solve(coefficients, table)
        columns.append(
            torch.stack([delta_theta * scale, delta_phi * radius], dim=-1) / PROBE
        )
    return torch.stack(columns), table


def weighted_lstsq(basis, target, weight):
    "Least squares in the area-weighted pixel metric; returns (coeffs, explained fraction)."
    root = weight.sqrt()
    design = (basis * root.unsqueeze(-1)).reshape(basis.shape[0], -1).T.double()
    observed = (target * root.unsqueeze(-1)).reshape(-1).double()
    solution = torch.linalg.lstsq(design, observed.unsqueeze(1)).solution.squeeze(1)
    residual = observed - design @ solution
    total = float(observed.dot(observed))
    return solution, 1.0 - float(residual.dot(residual)) / max(total, 1e-30), total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--pattern", default="{scene}_{rung}")
    parser.add_argument("--against", default="noncentral")
    parser.add_argument("--rttpf-rung", default="rttpf")
    parser.add_argument("--components", nargs="+", default=["tilt", "radial", "ana", "z"])
    parser.add_argument("--cutoff-deg", type=float, default=85.5)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    theta, phi = grid(args.cutoff_deg)
    weight = theta.sin() * plate_scale(theta, torch.ones(16, device="cuda"))[0]

    print(f"{'scene':11s}{'|resid| px':>12s}{'expl 2D':>9s}{'expl radial':>13s}"
          f"{'best-fit px':>13s}{'learned px':>12s}{'found/best':>12s}")
    print("-" * 82)
    out = {}
    for scene in args.scenes:
        rttpf_run = os.path.join(args.root, args.pattern.format(scene=scene, rung=args.rttpf_rung))
        other_run = os.path.join(args.root, args.pattern.format(scene=scene, rung=args.against))
        try:
            params, learned_deltas = colmap_params(rttpf_run)
            residual_theta, residual_phi, _ = residual_field(
                other_run, theta, phi, tuple(args.components)
            )
        except (FileNotFoundError, StopIteration, KeyError) as error:
            print(f"  -- {scene}: {type(error).__name__} {error}")
            continue

        scale, radius = plate_scale(theta, params)
        target = torch.stack([residual_theta * scale, residual_phi * radius], dim=-1)
        basis, _ = rttpf_basis(params, theta, phi)

        solution, explained, total = weighted_lstsq(basis, target, weight)
        # * The radial-only question `residual_expressible.py` asked, on the same grid:
        # * fit the theta component alone with the 5 radial channels (fx, k0..k3).
        radial_index = [0, 4, 5, 6, 7]
        _, explained_radial, _ = weighted_lstsq(
            basis[radial_index, ..., 0:1], target[..., 0:1], weight
        )

        # * Where did descent actually land? Convert the learned COLMAP deltas back into
        # * normalized coefficients -- the same coordinates the basis is expressed in.
        focal = torch.stack([params[0], params[1], params[0], params[1]])
        scales = rttpf_normalization(params)
        found = torch.cat([learned_deltas[:4] / focal, learned_deltas[4:]]) * scales

        norm = lambda f: float(((f**2) * weight.unsqueeze(-1)).sum()) ** 0.5 / float(  # noqa: E731
            weight.sum()
        ) ** 0.5
        best_field = torch.einsum("i...,i->...", basis, solution.float())
        found_field = torch.einsum("i...,i->...", basis, found)
        out[scene] = {
            "residual_rms_px": norm(target),
            "explained_2d": explained,
            "explained_radial": explained_radial,
            "best_fit_rms_px": norm(best_field),
            "learned_rms_px": norm(found_field),
        }
        ratio = out[scene]["learned_rms_px"] / max(out[scene]["best_fit_rms_px"], 1e-30)
        out[scene]["found_over_best"] = ratio
        print(f"{scene:11s}{out[scene]['residual_rms_px']:12.3f}{explained:9.2f}"
              f"{explained_radial:13.2f}{out[scene]['best_fit_rms_px']:13.3f}"
              f"{out[scene]['learned_rms_px']:12.3f}{ratio:12.2f}")

    if out:
        print("-" * 82)
        mean = lambda k: float(np.mean([v[k] for v in out.values()]))  # noqa: E731
        print(f"{'mean':11s}{mean('residual_rms_px'):12.3f}{mean('explained_2d'):9.2f}"
              f"{mean('explained_radial'):13.2f}{mean('best_fit_rms_px'):13.3f}"
              f"{mean('learned_rms_px'):12.3f}{mean('found_over_best'):12.2f}")
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(out, handle, indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
