#!/usr/bin/env python
"""Is the learned residual a LENS property, or per-scene calibration error?

The question this answers: the camera model bought +0.44 dB on myscenes and +0.016 dB on
FullCircle refit_rttpf. Does that mean rttpf underfits the myscenes lens and not the
FullCircle one?

The discriminating fact is that in both datasets the SAME physical lens is calibrated
independently on every scene. So:

  * spread of theta_s(rho) across scenes  = how badly determined the calibration is;
  * mean over scenes of the learned dtheta_s = a correction the model applies EVERYWHERE,
    i.e. a genuine lens-model deficiency rttpf could not express;
  * scene-to-scene deviation of dtheta_s   = the model absorbing that scene's own
    calibration error (or anything else scene-specific).

If the residual were fixing an underfit LENS, its cross-scene mean would dominate and
adding it would pull the per-scene curves TOGETHER. If it is mopping up per-scene
calibration noise, the mean is small and the deviation carries everything.

Everything is reported in pixels AT THE RESOLUTION THE RUN WAS TRAINED AT (gray uses -r 4,
so COLMAP fx 1240 -> 310), through the LOCAL plate scale dr/dtheta rather than the paraxial
fx (which overstates the rim by ~1.8x). Those are the pixels the renderer samples.

usage: python scripts/analysis/calib_consistency.py
"""

import glob
import json
import os
import sys

import numpy as np
import pycolmap
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gray.camera_model import bspline_eval  # noqa: E402

MYSCENES = {
    "atrium": "atrium_undistortion", "library": "library_undistortion",
    "reception": "reception_undistortion", "tunnel": "tunnel_undistortion",
    "classroom": "classroom", "forest": "forest", "workshop": "workshop",
}
MY_ROOT = "/workspace/gray/data/myscenes"
MY_RUNS = "/workspace/gray/tmp/final/{scene}_noncentral"
FC = ["room1", "room2", "room3", "flat1", "flat2", "lab", "lounge", "dark", "persons"]
FC_ROOT = "/workspace/dataset/fullcircle_tracks/refit_rttpf"
FC_RUNS = ("/workspace/gray/worktrees/noncentral-camera/out/fullcircle_rttpf/"
           "{scene}_refit_rttpf")
# rttpf params: fx fy cx cy then k1 k2 p1 p2 k3 k4 sx1 sy1 ... -- the radial ones are the
# 4 k's at 4,5,8,9 (p1,p2 tangential and sx1,sy1 thin-prism sit between them).
RADIAL = [4, 5, 8, 9]
GRID = np.linspace(0.02, 1.0, 60)  # * fraction of the common invertible pixel radius


def radial_poly(params):
    return [params[i] for i in RADIAL]


def r_of_theta(fx, radial, theta):
    poly = np.ones_like(theta)
    for order, k in enumerate(radial, start=1):
        poly = poly + k * theta ** (2 * order)
    return fx * theta * poly


def plate_scale(fx, radial, theta):
    poly, dpoly = np.ones_like(theta), np.zeros_like(theta)
    for order, k in enumerate(radial, start=1):
        poly = poly + k * theta ** (2 * order)
        dpoly = dpoly + k * (2 * order) * theta ** (2 * order - 1)
    return fx * (poly + theta * dpoly)


def theta_of_r(fx, radial, r_px, theta_max):
    """Invert r(theta) at a PIXEL radius.

    The independent variable must be a property of the sensor, not of the fit. Normalising
    by each scene's own r(theta_max) pins every scene to the same value at the rim and
    manufactures agreement exactly where the residual acts -- the first version of this
    script did that and reported a spread of 0.000 px at the edge.
    """
    grid = np.linspace(0.0, theta_max, 4000)
    radii = r_of_theta(fx, radial, grid)
    return np.interp(r_px, radii, grid)


def run_downsampling(run):
    """The `-r` the run was trained at; every pixel figure is reported at that scale."""
    try:
        with open(os.path.join(run, "config.json")) as handle:
            return float(json.load(handle).get("downsampling", 1) or 1)
    except (OSError, ValueError):
        return 1.0


def learned_dtheta(run, uid, theta, theta_max_model=np.pi / 2):
    """Channel-0 (azimuth-independent) angular residual, evaluated at `theta` [rad]."""
    checkpoints = sorted(glob.glob(f"{run}/gaussians_*.safetensors"))
    if not checkpoints:
        return None
    with safe_open(checkpoints[-1], "pt") as handle:
        key = f"camera_model.lenses.{uid}.theta_weights"
        if key not in handle.keys():
            return None
        weights = handle.get_tensor(key).float()
    t01 = torch.from_numpy(np.clip(theta / theta_max_model, 0.0, 1.0)).float()
    return bspline_eval(weights, t01)[0].numpy()


def collect(name, scenes, sparse_of, run_of, uid):
    fits = []
    for scene in scenes:
        sparse = sparse_of(scene)
        if not os.path.isdir(sparse):
            continue
        rec = pycolmap.Reconstruction(sparse)
        if uid not in rec.cameras:
            continue
        params = list(rec.cameras[uid].params)
        # * Max field angle the calibration can express: r(theta) must stay monotone.
        grid = np.linspace(0.0, np.pi / 2, 4000)
        radii = r_of_theta(params[0], radial_poly(params), grid)
        fold = np.argmax(np.diff(radii) <= 0) if (np.diff(radii) <= 0).any() else len(grid) - 1
        fits.append({"scene": scene, "params": params, "theta_max": grid[fold],
                     "r_max": radii[fold]})
    if not fits:
        print(f"\n### {name}: no data")
        return None

    # * One pixel-radius grid for every scene, out to the smallest invertible radius, so the
    # * comparison is over the same physical sensor positions.
    r_px = GRID * min(f["r_max"] for f in fits)
    rows = []
    for f in fits:
        theta = theta_of_r(f["params"][0], radial_poly(f["params"]), r_px, f["theta_max"])
        rows.append({**f, "theta": theta,
                     "scale": plate_scale(f["params"][0], radial_poly(f["params"]), theta),
                     "dtheta": learned_dtheta(run_of(f["scene"]), uid, theta)})
    if not rows:
        print(f"\n### {name}: no data")
        return None

    theta = np.stack([r["theta"] for r in rows])          # [S, G] rad, vs normalised radius
    scale = np.stack([r["scale"] for r in rows])
    # * Report at the resolution the model was TRAINED at, not at sensor resolution.
    # * COLMAP's fx is the full-frame one (1240 px on myscenes) while gray runs at -r 4
    # * (310 px), so quoting sensor pixels overstates every figure by 4x -- and the number
    # * that matters is the one in the pixels the renderer actually samples.
    downsample = run_downsampling(run_of(rows[0]["scene"]))
    px = scale.mean(0) / downsample                        # px per rad at working resolution

    print(f"\n### {name}  ({len(rows)} scenes, camera uid {uid})")
    print(f"  fx spread over scenes: {np.std([r['params'][0] for r in rows]):.3f} px "
          f"(mean {np.mean([r['params'][0] for r in rows]):.2f})")

    # -- how much do the per-scene CALIBRATIONS disagree, in pixels at the rim?
    disagree = (theta.std(0) * px)
    print(f"  calibration disagreement across scenes  (px): "
          f"mid-field {disagree[len(GRID) // 2]:.3f}   rim {disagree[-1]:.3f}   "
          f"max {disagree.max():.3f}")

    if any(r["dtheta"] is None for r in rows):
        print("  (no learned residual available)")
        return None
    d = np.stack([r["dtheta"] for r in rows])              # [S, G] rad
    common, deviation = d.mean(0), d.std(0)
    print(f"  learned residual, cross-scene MEAN      (px): "
          f"mid-field {abs(common[len(GRID) // 2]) * px[len(GRID) // 2]:.3f}   "
          f"rim {abs(common[-1]) * px[-1]:.3f}   max {np.abs(common * px).max():.3f}")
    print(f"  learned residual, cross-scene DEVIATION (px): "
          f"mid-field {deviation[len(GRID) // 2] * px[len(GRID) // 2]:.3f}   "
          f"rim {deviation[-1] * px[-1]:.3f}   max {(deviation * px).max():.3f}")
    ratio = np.abs(common).sum() / max(deviation.sum(), 1e-12)
    print(f"  shared / scene-specific  = {ratio:.2f}   "
          f"({'a LENS property' if ratio > 1.5 else 'mostly PER-SCENE'})")

    # -- the decisive one: does calibration + residual agree BETTER across scenes?
    before = (theta.std(0) * px)
    after = ((theta + d).std(0) * px)
    print(f"  cross-scene spread of theta(rho)        (px): "
          f"before {before.mean():.4f} -> after {after.mean():.4f}  "
          f"({'CONVERGES' if after.mean() < before.mean() else 'DIVERGES'}, "
          f"{100 * (after.mean() / before.mean() - 1):+.1f} %)")
    return {"name": name, "before": before.mean(), "after": after.mean(),
            "common_px": float(np.abs(common * px).max()),
            "deviation_px": float((deviation * px).max()),
            "calib_disagreement_px": float(disagree.max())}


def main():
    out = []
    out.append(collect(
        "myscenes (one lens, 7 independent COLMAP fits)", list(MYSCENES),
        lambda s: f"{MY_ROOT}/{MYSCENES[s]}/distorted/sparse/0",
        lambda s: MY_RUNS.format(scene=s), 1))
    for uid in (1, 2):
        out.append(collect(
            f"FullCircle refit_rttpf (lens {uid}, 9 independent re-fits)", FC,
            lambda s: f"{FC_ROOT}/{s}/distorted/sparse/0",
            lambda s: FC_RUNS.format(scene=s), uid))
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_consistency.json")
    with open(dest, "w") as handle:
        json.dump([o for o in out if o], handle, indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
