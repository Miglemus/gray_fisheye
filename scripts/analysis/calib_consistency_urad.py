#!/usr/bin/env python
"""Shared calibration bias in ANGULAR units -- the well-conditioned version.

Why this file exists, and why it does not simply edit `calib_consistency.py`:

`calib_consistency.py` reports everything in pixels, obtained by multiplying the (radian)
residual by the LOCAL plate scale `dr/dtheta` (its line 153, `px = scale.mean(0) /
downsample`). Its docstring justifies that choice against the paraxial `fx`, which does
overstate the rim -- but the local derivative is *badly conditioned*, and that was measured:
across four independent COLMAP fits of the SAME myscenes lens, `S(theta)/fx` at the rim
takes the values 0.905 (atrium), 0.573 (tunnel), 1.262 (workshop), 0.251 (reception) -- a
factor of 5 -- while the rim RADIUS is stable to +-5 %. Differentiating a fitted polynomial
multiplies each coefficient by 2n (x3, x5, x7, x9 on k1..k4) and the amplification peaks at
the edge of the support, which is exactly where the residual acts. So the calibration of
this glass is well constrained in POSITION and not at all in DERIVATIVE, and any figure
whose x axis is built from that derivative inherits the factor-5 dispersion.

Two consequences, both implemented here:

* **Report radians, never pixels.** The residual is native in radians; converting to px and
  back through a fitted derivative can only lose. If a pixel figure is needed for prose,
  use the SECANT scale `r(theta)/theta`, which involves no derivative -- emitted here as
  `secant_px_per_rad_mean` for that purpose only.
* **Aggregate as an area-weighted RMS, not a peak.** `calib_consistency.py` line 188 takes
  `np.abs(common * px).max()`, a rim peak. Measured counter-example: mip-NeRF 360 bicycle
  and stump have near-identical PEAK residuals (492.9 vs 468.2 urad) but RMS in a ratio of
  2.0 and PSNR gains in a ratio of 7 -- a rim-peak x axis collapses those two scenes onto
  one point. The grid is a uniform PIXEL-radius grid, so the disk area element gives weight
  proportional to rho.

`common` (cross-scene mean) is the shared model deficiency; `deviation` (cross-scene std) is
per-scene calibration error the model mops up. That decomposition is unchanged from the
original script -- only the units and the aggregation differ.

Read-only: imports the original module and reuses its helpers rather than duplicating them,
so the two cannot drift apart. Writes one new JSON, touches nothing else.

usage: python scripts/analysis/calib_consistency_urad.py
"""

import json
import os
import sys

import numpy as np
import pycolmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_consistency as cc  # noqa: E402


def area_weighted_rms(values, rho):
    "RMS of `values` over the disk: uniform rho grid => area element weight ~ rho."
    weights = rho / rho.sum()
    return float(np.sqrt((weights * values**2).sum()))


def collect_urad(name, scenes, sparse_of, run_of, uid):
    fits = []
    for scene in scenes:
        sparse = sparse_of(scene)
        if not os.path.isdir(sparse):
            continue
        rec = pycolmap.Reconstruction(sparse)
        if uid not in rec.cameras:
            continue
        params = list(rec.cameras[uid].params)
        grid = np.linspace(0.0, np.pi / 2, 4000)
        radii = cc.r_of_theta(params[0], cc.radial_poly(params), grid)
        folds = np.diff(radii) <= 0
        fold = np.argmax(folds) if folds.any() else len(grid) - 1
        fits.append({"scene": scene, "params": params,
                     "theta_max": grid[fold], "r_max": radii[fold]})
    if not fits:
        print(f"\n### {name}: no data")
        return None

    r_px = cc.GRID * min(f["r_max"] for f in fits)
    rows = []
    for f in fits:
        theta = cc.theta_of_r(f["params"][0], cc.radial_poly(f["params"]),
                              r_px, f["theta_max"])
        dtheta = cc.learned_dtheta(run_of(f["scene"]), uid, theta)
        rows.append({**f, "theta": theta, "dtheta": dtheta,
                     "scale": cc.plate_scale(f["params"][0], cc.radial_poly(f["params"]),
                                             theta)})
    if any(r["dtheta"] is None for r in rows):
        print(f"\n### {name}: no learned residual on some scene, skipped")
        return None

    theta = np.stack([r["theta"] for r in rows])   # [S, G] rad
    d = np.stack([r["dtheta"] for r in rows])      # [S, G] rad
    common, deviation = d.mean(0), d.std(0)
    rho = cc.GRID

    # * Secant scale r(theta)/theta -- derivative-free, for prose only.
    downsample = cc.run_downsampling(run_of(rows[0]["scene"]))
    secant = np.stack([cc.r_of_theta(r["params"][0], cc.radial_poly(r["params"]),
                                     r["theta"]) / np.maximum(r["theta"], 1e-9)
                       for r in rows]).mean(0) / downsample
    # * The badly-conditioned quantity, kept only to reproduce the legacy field.
    legacy_px = cc.plate_scale(rows[0]["params"][0], cc.radial_poly(rows[0]["params"]),
                               theta[0])
    legacy_px = np.stack([r["scale"] for r in rows]).mean(0) / downsample

    res = {
        "name": name,
        "n_scenes": len(rows),
        "common_urad_rms": area_weighted_rms(common, rho) * 1e6,
        "common_urad_peak": float(np.abs(common).max()) * 1e6,
        "deviation_urad_rms": area_weighted_rms(deviation, rho) * 1e6,
        "calib_disagreement_urad_rms": area_weighted_rms(theta.std(0), rho) * 1e6,
        "shared_over_scene_specific": float(np.abs(common).sum()
                                            / max(deviation.sum(), 1e-12)),
        "secant_px_per_rad_mean": float(secant.mean()),
        # * Legacy, DO NOT PUBLISH: rim peak through the local derivative.
        "legacy_peak_via_derivative_px": float(np.abs(common * legacy_px).max()),
    }
    print(f"\n### {name}  ({len(rows)} scenes, uid {uid})")
    print(f"  shared bias, area-weighted RMS      : {res['common_urad_rms']:8.2f} urad")
    print(f"  shared bias, peak (do not use as x) : {res['common_urad_peak']:8.2f} urad")
    print(f"  per-scene deviation, RMS            : {res['deviation_urad_rms']:8.2f} urad")
    print(f"  calibration disagreement, RMS       : "
          f"{res['calib_disagreement_urad_rms']:8.2f} urad")
    print(f"  shared / scene-specific             : "
          f"{res['shared_over_scene_specific']:8.2f}")
    print(f"  [legacy peak via local derivative]  : "
          f"{res['legacy_peak_via_derivative_px']:8.4f} px  <- quarantined")
    return res


def main():
    out = [collect_urad("myscenes (one lens, 7 independent COLMAP fits)",
                        list(cc.MYSCENES),
                        lambda s: f"{cc.MY_ROOT}/{cc.MYSCENES[s]}/distorted/sparse/0",
                        lambda s: cc.MY_RUNS.format(scene=s), 1)]
    for uid in (1, 2):
        out.append(collect_urad(
            f"FullCircle refit_rttpf (lens {uid}, 9 independent re-fits)", cc.FC,
            lambda s: f"{cc.FC_ROOT}/{s}/distorted/sparse/0",
            lambda s: cc.FC_RUNS.format(scene=s), uid))
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "calib_consistency_urad.json")
    with open(dest, "w") as handle:
        json.dump([o for o in out if o], handle, indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
