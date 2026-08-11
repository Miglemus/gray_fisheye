"""Export the *full* learned residual, all five azimuthal channels, per scene.

`camera_model_*.csv` only logs channel 0 (the purely radial term), because that is what the
cross-scene consistency figure needs. The anamorphic channels live in the checkpoint, so we
read them straight out of `gaussians_15000.safetensors` and evaluate the splines on a theta
grid. The report's 2-D field map is then drawn in the browser from these 5 curves:

    dtheta(theta, phi) = s0(theta) + s1 cos(phi) + s2 sin(phi) + s3 cos(2phi) + s4 sin(2phi)

and likewise for the sagittal term. Both are angles; the pixel displacement they cause is
    radial      = dr/dtheta * dtheta
    tangential  = r(theta)  * dphi
with r(theta) the rttpf radial projection actually used at render time.
"""

import json
import math
import os

import numpy as np
import torch
from safetensors import safe_open

WS = "/workspace"
SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
HERE = os.path.dirname(os.path.abspath(__file__))
RUNGS = {"noncentral": "_noncentral", "ana": "_ana", "central_matched": "_central_matched"}


def bspline_eval(weights, t01):
    "Same uniform cubic B-spline as gray/camera_model.py, on CPU numpy."
    num_ctrl = weights.shape[-1]
    segments = max(num_ctrl - 3, 1)
    t = np.clip(t01, 0.0, 1.0) * segments
    idx = np.clip(np.floor(t).astype(int), 0, segments - 1)
    u = t - idx
    u2, u3 = u * u, u * u * u
    taps = [
        (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
        (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
        (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
        u3 / 6.0,
    ]
    out = np.zeros((weights.shape[0], t01.size))
    for offset, tap in enumerate(taps):
        out += tap * weights[:, np.clip(idx + offset, 0, num_ctrl - 1)]
    return out


def load_weights(run):
    path = f"{run}/gaussians_15000.safetensors"
    with safe_open(path, "pt") as handle:
        keys = [k for k in handle.keys() if k.startswith("camera_model.lenses.")]
        uid = sorted({k.split(".")[2] for k in keys})[0]
        get = lambda name: handle.get_tensor(f"camera_model.lenses.{uid}.{name}").float().numpy()
        return {
            "omega": get("omega"),
            "theta_weights": get("theta_weights"),
            "phi_weights": get("phi_weights"),
            "z_weights": get("z_weights"),
        }


def radial_projection(fx, radial, theta):
    "rttpf image radius r(theta) in pixels, and its derivative dr/dtheta."
    poly = np.ones_like(theta)
    dpoly = np.zeros_like(theta)
    for order, k in enumerate(radial, start=1):
        poly = poly + k * theta ** (2 * order)
        dpoly = dpoly + k * (2 * order) * theta ** (2 * order - 1)
    return fx * theta * poly, fx * (poly + theta * dpoly)


def main():
    samples = np.linspace(0.0, 1.0, 91)
    theta = samples * (math.pi / 2.0)
    out = {"theta_deg": (samples * 90.0).tolist(), "scenes": {}}

    for scene in SCENES:
        entry = {}
        cams = json.load(open(f"{WS}/gray/tmp/final/{scene}_noncentral/cameras.json"))
        intrinsics = cams[0]["intrinsics"]
        fx = intrinsics[0]
        radial_k = intrinsics[4:10]
        r_px, drdt = radial_projection(fx, radial_k, theta)
        entry["focal_px"] = fx
        entry["r_px"] = np.round(r_px, 4).tolist()
        entry["drdtheta_px"] = np.round(drdt, 4).tolist()

        for rung, suffix in RUNGS.items():
            run = f"{WS}/gray/tmp/final/{scene}{suffix}"
            if not os.path.exists(f"{run}/gaussians_15000.safetensors"):
                continue
            w = load_weights(run)
            dtheta = bspline_eval(w["theta_weights"], samples)
            dphi = bspline_eval(w["phi_weights"], samples)
            z = bspline_eval(w["z_weights"], samples)[0]
            z = z - z[0]  # * gauge z(0) = 0, exactly as the forward pass imposes it
            entry[rung] = {
                "dtheta_channels": [[float(f"{v:.6g}") for v in row] for row in dtheta],
                "dphi_channels": [[float(f"{v:.6g}") for v in row] for row in dphi],
                "z": [float(f"{v:.6g}") for v in z],
                "omega": [float(v) for v in w["omega"]],
                "n_trained": int(
                    w["theta_weights"].size + w["phi_weights"].size + w["omega"].size
                    + (w["z_weights"].size if rung == "noncentral" else 0)
                ),
            }
        out["scenes"][scene] = entry
        nc = entry["noncentral"]
        ana_amp = np.abs(np.asarray(nc["dtheta_channels"])[1:]).max()
        rad_amp = np.abs(np.asarray(nc["dtheta_channels"])[0]).max()
        print(f"{scene:11s} radial={rad_amp*fx:6.3f}px  anamorphic={ana_amp*fx:6.3f}px  "
              f"|omega|={np.linalg.norm(nc['omega'])*1e3:6.3f}mrad  "
              f"|z|max={max(abs(v) for v in nc['z']):.5f}  params={nc['n_trained']}")

    with open(os.path.join(HERE, "fields.json"), "w") as handle:
        json.dump(out, handle)
    print("wrote fields.json")


if __name__ == "__main__":
    main()
