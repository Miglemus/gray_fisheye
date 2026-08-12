"""Do the two rungs learn the same correction?

`residual_expressible.py` established, offline, that 98.6 % of the radial residual the
`noncentral` rung learns on myscenes lies inside the span of rttpf's own coefficients --
i.e. the model class was adequate and the delivered coefficients were wrong. That is a
statement about *reachability*. This script asks the operational question instead: when the
16 rttpf coefficients are actually handed to the photometric optimizer, does it find that
same correction?

For each scene it reconstructs, on a common theta grid:

* `d_theta_rttpf(theta)` -- the angular shift implied by the learned intrinsic deltas,
  obtained by solving `project(theta + d, new_params) = project(theta, colmap_params)`,
  which is the same equation `rttpf_solve` solves per pixel (radial slice, phi = 0);
* `d_theta_noncentral(theta)` -- channel 0 of the trained spline, read from the run's own
  `camera_model_*.csv`.

Both are then converted to pixels through the LOCAL plate scale `dr/dtheta`, never the
paraxial `fx`: the rttpf polynomial compresses the rim to ~0.56 fx, so `fx * angle`
overstates every peripheral figure by ~1.8x (IMPLEMENTATION.md).

usage:
  python scripts/analysis/rttpf_vs_noncentral.py \
      --rttpf tmp/r8_control/tunnel_rttpf --noncentral tmp/r8_control/tunnel_noncentral
  python scripts/analysis/rttpf_vs_noncentral.py --root tmp/r8_control --scenes tunnel workshop
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

NAMES = ("fx", "fy", "cx", "cy", "k0", "k1", "k2", "k3", "k4", "k5",
         "p0", "p1", "s0", "s1", "s2", "s3")
SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]


def latest(run_dir, stem):
    files = sorted(glob.glob(os.path.join(run_dir, f"{stem}_*.csv")))
    if not files:
        raise FileNotFoundError(f"no {stem}_*.csv in {run_dir}")
    return files[-1]


def read_intrinsics(run_dir):
    "COLMAP parameters and the learned delta, from the last checkpoint of a rttpf run."
    base, delta = {}, {}
    with open(latest(run_dir, "camera_intrinsics")) as handle:
        for row in csv.DictReader(handle):
            base[row["name"]] = float(row["colmap"])
            delta[row["name"]] = float(row["delta"])
    return (np.array([base[name] for name in NAMES]),
            np.array([delta[name] for name in NAMES]))


def read_spline(run_dir):
    "theta grid (rad) and channel-0 d(theta) of a residual-rung run."
    theta, radial = [], []
    with open(latest(run_dir, "camera_model")) as handle:
        for row in csv.DictReader(handle):
            theta.append(float(row["theta_deg"]) * np.pi / 180.0)
            radial.append(float(row["delta_theta_rad"]))
    return np.array(theta), np.array(radial)


def radius(theta, params):
    "Radial image coordinate r(theta) of the rttpf model, phi = 0 slice, in pixels."
    growth = np.ones_like(theta)
    power = np.ones_like(theta)
    for index in range(6):
        power = power * theta * theta
        growth = growth + params[4 + index] * power
    x = growth * theta
    # * phi = 0, so y = 0: the tangential and thin-prism terms reduce to their x parts.
    return params[0] * (x + params[10] * 3.0 * x * x + params[12] * x * x + params[13] * x**4)


def plate_scale(theta, params, step=1e-4):
    "Local dr/dtheta in pixels per radian -- the honest angle -> pixel conversion."
    return (radius(theta + step, params) - radius(theta - step, params)) / (2.0 * step)


def implied_shift(theta, params, deltas):
    """d(theta) such that the RE-FITTED model puts the same pixel at theta + d(theta).

    Same equation `rttpf_solve` solves per pixel, restricted to the phi = 0 meridian and
    solved by Newton on the scalar radius.
    """
    fitted = params + deltas
    target = radius(theta, params)
    current = theta.copy()
    for _ in range(50):
        step = (radius(current, fitted) - target) / plate_scale(current, fitted)
        current = current - step
    return current - theta


def compare(rttpf_dir, noncentral_dir, cutoff_deg):
    params, deltas = read_intrinsics(rttpf_dir)
    theta, spline = read_spline(noncentral_dir)
    keep = theta <= np.deg2rad(cutoff_deg)
    theta = theta[keep]
    scale = plate_scale(theta, params)
    fitted = implied_shift(theta, params, deltas) * scale
    residual = spline[keep] * scale
    both = np.vstack([fitted, residual])
    correlation = float(np.corrcoef(both)[0, 1]) if both[0].std() and both[1].std() else float("nan")
    return {
        "theta_deg": np.rad2deg(theta).tolist(),
        "rttpf_px": fitted.tolist(),
        "noncentral_px": residual.tolist(),
        "rttpf_rim_px": float(fitted[-1]),
        "noncentral_rim_px": float(residual[-1]),
        "rttpf_max_px": float(np.abs(fitted).max()),
        "noncentral_max_px": float(np.abs(residual).max()),
        "correlation": correlation,
        # * How much of the residual rung's radial correction the re-fit actually found,
        # * in the least-squares sense: 1 - ||nc - rttpf||^2 / ||nc||^2.
        "explained": float(1.0 - ((residual - fitted) ** 2).sum() / max((residual**2).sum(), 1e-30)),
        "deltas": {name: float(value) for name, value in zip(NAMES, deltas)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--rttpf", default=None, help="single rttpf run dir")
    parser.add_argument("--noncentral", default=None, help="single noncentral run dir")
    parser.add_argument("--pattern", default="{scene}_{rung}")
    parser.add_argument("--cutoff-deg", type=float, default=85.5, help="the 0.95 mask edge")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    pairs = {}
    if args.rttpf and args.noncentral:
        pairs["run"] = (args.rttpf, args.noncentral)
    else:
        for scene in args.scenes:
            pairs[scene] = (
                os.path.join(args.root, args.pattern.format(scene=scene, rung="rttpf")),
                os.path.join(args.root, args.pattern.format(scene=scene, rung="noncentral")),
            )

    print(f"{'scene':11s}{'rttpf rim':>11s}{'nc rim':>9s}{'rttpf max':>11s}"
          f"{'nc max':>9s}{'corr':>8s}{'explained':>11s}")
    print("-" * 70)
    out = {}
    for name, (rttpf_dir, noncentral_dir) in pairs.items():
        try:
            result = compare(rttpf_dir, noncentral_dir, args.cutoff_deg)
        except (FileNotFoundError, IndexError) as error:
            print(f"  -- {name}: {type(error).__name__}")
            continue
        out[name] = result
        print(f"{name:11s}{result['rttpf_rim_px']:11.3f}{result['noncentral_rim_px']:9.3f}"
              f"{result['rttpf_max_px']:11.3f}{result['noncentral_max_px']:9.3f}"
              f"{result['correlation']:8.2f}{result['explained']:11.2f}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(out, handle, indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
