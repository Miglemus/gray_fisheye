"""What shape is the purely radial (k=0) part of the learned residual?

Two very different things produce a radial term, and they have different shapes:

  * a focal-length error       dr(r) = e * r                (linear, one parameter)
  * residual lens distortion   dr(r) = k * r^3 (+ k5 r^5)   (odd, grows fast at the edge)

COLMAP's `image_undistorter` only removes the distortion its *estimated* model captured, so
an under-parameterised original camera (SIMPLE_RADIAL keeps a single k1) leaves a higher-order
remainder in the "undistorted" images. That remainder is exactly a k=0 term.

Fits both templates to the learned channel-0 profile, in pixels, over the theta range the
scene actually uses. Outputs tmp/mipnerf360/analysis/radial_shape.json.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.mip360_aspect import (  # noqa: E402
    SCENES, bspline_eval, load_lens,
)

ROOT = "/workspace/gray/worktrees/noncentral-camera"
OUT = f"{ROOT}/tmp/mipnerf360/analysis/radial_shape.json"


def main():
    cameras = json.load(open(f"{ROOT}/tmp/mipnerf360/analysis/aspect.json"))["cameras"]
    payload = {}
    print(f"{'scene':9s} {'theta_max':>9s} {'rms dr px':>10s} | "
          f"{'focal only':>10s} {'cubic only':>10s} {'both':>8s} | {'e_focal':>9s} {'k3':>9s}")
    for scene in SCENES:
        run = f"{ROOT}/tmp/mipnerf360/{scene}_noncentral"
        lens = load_lens(run)
        if lens is None:
            continue
        cam = cameras[scene]
        width, height = cam["width"], cam["height"]
        f_impl = height / (2.0 * math.tan(cam["fov_y"] / 2.0))
        r_max = math.hypot(width / 2.0, height / 2.0)
        theta_max = math.atan(r_max / f_impl)

        theta = np.linspace(0.0, theta_max, 400)
        theta01 = np.clip(theta / (math.pi / 2.0), 0.0, 1.0)
        d_theta = bspline_eval(lens["theta_weights"], theta01)[0]  # * channel 0 only
        radius = f_impl * np.tan(theta)
        # * dr = d(f tan theta)/dtheta * dtheta = f sec^2(theta) * dtheta
        dr = f_impl / np.cos(theta) ** 2 * d_theta

        # * Weight by the area element r dr so the fit reflects how many pixels sit at each r.
        weight = np.sqrt(np.maximum(radius, 1e-9))
        design_lin = (radius * weight)[:, None]
        design_cub = (radius**3 * weight)[:, None]
        design_both = np.stack([radius * weight, radius**3 * weight], -1)
        target = dr * weight

        def fit(design):
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
            resid = target - design @ coef
            ss = float((target**2).sum())
            return coef, 1.0 - float((resid**2).sum()) / max(ss, 1e-30)

        c_lin, r2_lin = fit(design_lin)
        c_cub, r2_cub = fit(design_cub)
        c_both, r2_both = fit(design_both)
        rms = float(np.sqrt((dr**2 * weight**2).sum() / (weight**2).sum()))

        payload[scene] = {
            "theta_max_deg": math.degrees(theta_max),
            "theta01_max": float(theta01.max()),
            "rms_dr_px": rms,
            "r2_focal": r2_lin, "r2_cubic": r2_cub, "r2_both": r2_both,
            "e_focal": float(c_both[0]), "k3": float(c_both[1]),
            "radius": [float(v) for v in radius[::8]],
            "dr_px": [float(v) for v in dr[::8]],
        }
        p = payload[scene]
        print(f"{scene:9s} {p['theta_max_deg']:9.2f} {rms:10.3f} | {r2_lin:10.3f} "
              f"{r2_cub:10.3f} {r2_both:8.3f} | {c_both[0]:9.2e} {c_both[1]:9.2e}")

    with open(OUT, "w") as handle:
        json.dump(payload, handle)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
