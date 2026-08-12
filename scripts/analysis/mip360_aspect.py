"""Why does a *learnable* camera help on mip-NeRF 360, which is already pinhole?

gray's pinhole raygen (`cuda/core/camera.h`) builds bearings from `vertical_fov_radians`
plus `aspect_ratio = dim.x / dim.y`. That forces `fx == fy` exactly: the implied focal is
`dim.y / (2 tan(fovy/2))` on *both* axes. `cam_info.fov_x` is computed in `gray/scene.py`
and then never reaches the renderer. mip-NeRF 360's COLMAP cameras are PINHOLE with two
independent focals, and on several scenes `fx != fy` by up to 5e-3 relative.

So gray renders those scenes with a horizontally mis-scaled camera. This script asks
whether the learned residual is simply *undoing that*, by comparing the measured
displacement field of the learned camera against the parameter-free prediction

    D_x(p) = (fy/fx - 1) * (x - cx),    D_y(p) = 0

which is what an fx-vs-fy fix must look like. Nothing here is fitted: eps comes from
cameras.bin, the field comes from the checkpoint.

Outputs tmp/mipnerf360/analysis/aspect.json.
"""

import json
import math
import os
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gray.colmap import read_intrinsics_binary  # noqa: E402

ROOT = "/workspace/gray/worktrees/noncentral-camera"
OUT = f"{ROOT}/tmp/mipnerf360/analysis/aspect.json"
# * Downsampling actually used per scene: outdoor -r 4, indoor -r 2 (user-authorised).
SCENES = {
    "bicycle": 4, "garden": 4, "stump": 4,
    "bonsai": 2, "counter": 2, "kitchen": 2, "room": 2,
}
RUNGS = ["noncentral", "ana", "central_matched", "radial", "tilt", "z_only"]


def bspline_eval(weights, t01):
    "Uniform cubic B-spline, mirroring gray/camera_model.py on numpy."
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
    out = np.zeros((weights.shape[0],) + t01.shape)
    for offset, tap in enumerate(taps):
        out += tap * weights[:, np.clip(idx + offset, 0, num_ctrl - 1)]
    return out


def harmonics(spline, cos_phi, sin_phi):
    "Channel 0 is radial; 1..4 are cos/sin of phi and 2phi. Mirrors LensResidual.harmonics."
    cos2 = cos_phi * cos_phi - sin_phi * sin_phi
    sin2 = 2.0 * cos_phi * sin_phi
    weights = [np.ones_like(cos_phi), cos_phi, sin_phi, cos2, sin2]
    return sum(spline[i] * weights[i] for i in range(spline.shape[0]))


def load_lens(run):
    path = f"{run}/gaussians_15000.safetensors"
    with safe_open(path, "pt") as handle:
        keys = [k for k in handle.keys() if k.startswith("camera_model.lenses.")]
        if not keys:
            return None
        uid = sorted({k.split(".")[2] for k in keys})[0]

        def get(name):
            full = f"camera_model.lenses.{uid}.{name}"
            return handle.get_tensor(full).float().numpy() if full in handle.keys() else None

        return {n: get(n) for n in ("omega", "theta_weights", "phi_weights", "z_weights")}


def skew_rotate(omega, vectors):
    "Rodrigues, matching gray/camera_model.py::skew_rotate."
    angle = np.linalg.norm(omega)
    if angle < 1e-12:
        return vectors
    axis = omega / angle
    cross = np.cross(axis, vectors)
    dot = (vectors * axis).sum(-1, keepdims=True)
    return vectors * math.cos(angle) + cross * math.sin(angle) + axis * dot * (1 - math.cos(angle))


def render_grid(width, height, fov_y, step=8):
    "gray's pinhole bearings, exactly as cuda/core/camera.h emits them (then flipped to OpenCV)."
    view = math.tan(fov_y / 2.0)
    aspect = width / height
    i = np.arange(0.5, width, step)
    j = np.arange(0.5, height, step)
    ii, jj = np.meshgrid(i, j)
    u = aspect * view * (2.0 * ii / width - 1.0)
    v = view * (2.0 * jj / height - 1.0)  # * flip of the raygen's y-up already applied
    bearings = np.stack([u, v, np.ones_like(u)], -1)
    bearings /= np.linalg.norm(bearings, axis=-1, keepdims=True)
    return ii, jj, bearings


CHANNEL_NAMES = ["k0 (radial)", "k1 cos", "k1 sin", "k2 cos", "k2 sin"]


def apply_lens(lens, bearings, components, only_channel=None, only_term=None):
    "Reproduce CameraModel.forward for the bearing (not the origin) path."
    bx, by, bz = bearings[..., 0], bearings[..., 1], bearings[..., 2]
    radius = np.hypot(bx, by)
    safe = np.maximum(radius, 1e-12)
    cos_phi, sin_phi = bx / safe, by / safe
    theta = np.arctan2(radius, bz)
    theta01 = np.clip(theta / (math.pi / 2.0), 0.0, 1.0)

    out = bearings.copy()
    if any(c in components for c in ("radial", "ana", "extra_knots")):
        st = bspline_eval(lens["theta_weights"], theta01)
        sp = bspline_eval(lens["phi_weights"], theta01)
        active = {"radial": [0], "ana": [0, 1, 2, 3, 4], "extra_knots": [0, 1, 2, 3, 4]}
        keep = sorted({i for c in components if c in active for i in active[c]})
        if only_channel is not None:
            keep = [i for i in keep if i == only_channel]
        mask = np.zeros((5, 1, 1))
        mask[keep] = 1.0
        d_theta = harmonics(st * mask, cos_phi, sin_phi)
        d_phi = harmonics(sp * mask, cos_phi, sin_phi)
        if only_term == "theta":
            d_phi = np.zeros_like(d_phi)
        elif only_term == "phi":
            d_theta = np.zeros_like(d_theta)
        meridian = np.stack([-sin_phi, cos_phi, np.zeros_like(sin_phi)], -1)
        cos_d, sin_d = np.cos(d_theta)[..., None], np.sin(d_theta)[..., None]
        out = out * cos_d + np.cross(meridian, out) * sin_d
        cp, sp_ = np.cos(d_phi), np.sin(d_phi)
        ox, oy, oz = out[..., 0], out[..., 1], out[..., 2]
        out = np.stack([ox * cp - oy * sp_, ox * sp_ + oy * cp, oz], -1)
    if "tilt" in components:
        out = skew_rotate(lens["omega"], out)
    return out


def main():
    cameras = {}
    for scene, down in SCENES.items():
        cams = read_intrinsics_binary(f"{ROOT}/data/360_v2/{scene}/sparse/0/cameras.bin")
        cam = next(iter(cams.values()))
        fx, fy, cx, cy = (float(v) for v in cam.params[:4])
        # * gray/scene.py scales each axis by image_dim / colmap_dim; the resized dims come
        # * from the mip-NeRF 360 release (ImageMagick, round-half-up), so recover them the
        # * same way gray does -- from the file on disk.
        from PIL import Image

        sample = sorted(os.listdir(f"{ROOT}/data/360_v2/{scene}/images_{down}"))[0]
        with Image.open(f"{ROOT}/data/360_v2/{scene}/images_{down}/{sample}") as im:
            width, height = im.size
        sx, sy = width / cam.width, height / cam.height
        cameras[scene] = {
            "colmap": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": cam.width, "h": cam.height},
            "width": width, "height": height,
            "fx_scaled": fx * sx, "fy_scaled": fy * sy,
            "fov_y": 2.0 * math.atan(height / (2.0 * fy * sy)),
            "aspect_eps": fy * sy / (fx * sx) - 1.0,
            "aspect_eps_full": fy / fx - 1.0,
        }

    results = {}
    for scene, cam in cameras.items():
        eps = cam["aspect_eps"]
        ii, jj, bearings = render_grid(cam["width"], cam["height"], cam["fov_y"])
        f_impl = cam["height"] / (2.0 * math.tan(cam["fov_y"] / 2.0))
        cx_g, cy_g = cam["width"] / 2.0, cam["height"] / 2.0
        # * The parameter-free prediction of an fx-vs-fy fix, in pixels.
        pred_x = eps * (ii - cx_g)
        pred_y = np.zeros_like(pred_x)

        entry = {"eps": eps, "f_impl": f_impl,
                 "pred_rms_px": float(np.sqrt((pred_x**2).mean())),
                 "rungs": {}}
        for rung in RUNGS:
            run = f"{ROOT}/tmp/mipnerf360/{scene}_{rung}"
            if not os.path.exists(f"{run}/gaussians_15000.safetensors"):
                continue
            lens = load_lens(run)
            if lens is None:
                continue
            comps = {
                "noncentral": ("tilt", "radial", "ana"),
                "ana": ("tilt", "radial", "ana"),
                "central_matched": ("tilt", "radial", "ana", "extra_knots"),
                "radial": ("tilt", "radial"),
                "tilt": ("tilt",),
                "z_only": (),
            }[rung]
            out = apply_lens(lens, bearings, comps)
            # * Where does the moved ray land, in gray's own pinhole projection? That is the
            # * displacement an observer sees.
            px = f_impl * out[..., 0] / out[..., 2] + cx_g
            py = f_impl * out[..., 1] / out[..., 2] + cy_g
            dx, dy = px - ii, py - jj
            # * Free scale of the prediction: how much of the fx-fix is realised.
            denom = float((pred_x * pred_x + pred_y * pred_y).sum())
            alpha = float((dx * pred_x + dy * pred_y).sum() / denom) if denom > 0 else float("nan")
            resid = np.sqrt(((dx - alpha * pred_x) ** 2 + (dy - alpha * pred_y) ** 2).mean())
            total = np.sqrt((dx * dx + dy * dy).mean())
            entry["rungs"][rung] = {
                "rms_px": float(total),
                "max_px": float(np.sqrt(dx * dx + dy * dy).max()),
                "alpha": alpha,
                "residual_rms_px": float(resid),
                "explained": float(1.0 - (resid**2) / max((total**2), 1e-30)),
                "dx_profile": [float(v) for v in dx[dx.shape[0] // 2, :]],
                "dy_profile": [float(v) for v in dy[dy.shape[0] // 2, :]],
            }
            if rung == "noncentral":
                # * Per-harmonic attribution: zero every channel but one and re-measure.
                per = {}
                for ch, name in enumerate(CHANNEL_NAMES):
                    o = apply_lens(lens, bearings, comps, only_channel=ch)
                    qx = f_impl * o[..., 0] / o[..., 2] + cx_g - ii
                    qy = f_impl * o[..., 1] / o[..., 2] + cy_g - jj
                    per[name] = float(np.sqrt((qx * qx + qy * qy).mean()))
                entry["channels"] = per
                # * Full 2-D field, decimated, for the report figure.
                entry["field"] = {
                    "x": [float(v) for v in ii[0, ::4]],
                    "y": [float(v) for v in jj[::4, 0]],
                    "dx": [[float(v) for v in row] for row in dx[::4, ::4]],
                    "dy": [[float(v) for v in row] for row in dy[::4, ::4]],
                }
        results[scene] = entry

    payload = {"cameras": cameras, "fields": results}
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1)

    print(f"{'scene':9s} {'eps(fy/fx-1)':>13s} {'pred_rms':>9s} | "
          f"{'rung':16s} {'rms_px':>8s} {'alpha':>7s} {'explained':>10s}")
    for scene, entry in results.items():
        head = f"{scene:9s} {entry['eps']:13.3e} {entry['pred_rms_px']:9.3f}"
        for rung, r in entry["rungs"].items():
            print(f"{head} | {rung:16s} {r['rms_px']:8.3f} {r['alpha']:7.3f} {r['explained']:10.3f}")
            head = " " * 33
    print("\nper-harmonic rms pixel displacement of the `noncentral` field")
    print(f"{'scene':9s} " + " ".join(f"{n:>12s}" for n in CHANNEL_NAMES))
    for scene, entry in results.items():
        if "channels" in entry:
            row = " ".join(f"{entry['channels'][n]:12.3f}" for n in CHANNEL_NAMES)
            print(f"{scene:9s} {row}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
