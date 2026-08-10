"""Turn the learned camera models into the physical quantities the report argues about.

Three products, all per scene:

1. the learned profiles as logged by train.py (`camera_model_15000.csv`);
2. the caustic of the learned ray field -- the envelope the rays are tangent to, which
   for an axial camera replaces the single pin-hole. Closed form for a meridional ray
   from (0, 0, z(t)) along (sin t, 0, cos t):
       s* = z'(t) sin t,   x_c = z'(t) sin^2 t,   z_c = z(t) + z'(t) sin t cos t;
3. the empirical inverse-depth distribution *per field angle*, from the fisheye COLMAP
   model (`distorted/sparse/0`). This is the term that decides whether a central camera
   can absorb the non-central offset: the induced angular shift is z(t) sin t / depth,
   so a central model can cancel it at exactly one depth and nowhere else.

Everything is in COLMAP units; pixel conversions use the r4 focal length gray actually
rendered with (~310 px/rad), read from the run's own cameras.json.
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

WS = "/workspace"
SCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
HERE = os.path.dirname(os.path.abspath(__file__))


def run_dir(scene):
    return f"{WS}/gray/tmp/final/{scene}_noncentral"


def source_path(scene):
    cfg = json.load(open(f"{run_dir(scene)}/config.json"))
    return os.path.join(WS, "gray", cfg["source_path"])


def read_profile(scene):
    rows = [l.split(",") for l in open(f"{run_dir(scene)}/camera_model_15000.csv").read().splitlines()[1:]]
    rows = [r for r in rows if r and r[0].strip()]
    return {
        "theta_deg": [float(r[1]) for r in rows],
        "dtheta": [float(r[2]) for r in rows],
        "dphi": [float(r[3]) for r in rows],
        "z": [float(r[4]) for r in rows],
    }


def plate_scale(fx, radial, theta):
    """dr/dtheta in px/rad for the rttpf radial polynomial -- the LOCAL image scale.

    An angular ray error of `d` radians displaces the rendered content by `dr/dtheta * d`
    pixels, not `fx * d`: the paraxial focal length is only the plate scale at theta = 0.
    On these calibrations dr/dtheta drops to ~0.56 fx at the rim, so the distinction is
    worth ~1.8x on exactly the field angles the camera model acts on.
    """
    poly = np.ones_like(theta)
    dpoly = np.zeros_like(theta)
    for order, k in enumerate(radial, start=1):
        poly = poly + k * theta ** (2 * order)
        dpoly = dpoly + k * (2 * order) * theta ** (2 * order - 1)
    return fx * (poly + theta * dpoly)


def caustic(theta_deg, z):
    """Envelope of the meridional ray family; returns (x, z) in COLMAP units."""
    t = np.radians(np.asarray(theta_deg))
    zv = np.asarray(z)
    dz = np.gradient(zv, t)
    return (dz * np.sin(t) ** 2).tolist(), (zv + dz * np.sin(t) * np.cos(t)).tolist()


def quat_to_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def radial_inverse_table(fx, radial):
    """theta -> pixel radius for the rttpf radial polynomial, then a table to invert it.

    Only the radial part is used. The tangential/thin-prism terms are <= 7.4e-3 in
    normalized units (~0.4 deg) and they merely re-label which theta bin a point falls in,
    which is irrelevant at 5-degree resolution.
    """
    theta = np.linspace(0.0, math.pi / 2.0 * 1.05, 4096)
    poly = np.ones_like(theta)
    for order, k in enumerate(radial, start=1):
        poly = poly + k * theta ** (2 * order)
    return theta, fx * theta * poly


def observations(scene):
    """(theta_rad, depth) for every 2D-3D observation in the fisheye COLMAP model."""
    root = os.path.join(source_path(scene), "distorted", "sparse", "0")
    cam_lines = [l for l in open(f"{root}/cameras.txt") if not l.startswith("#")]
    parts = cam_lines[0].split()
    fx, cx, cy = float(parts[4]), float(parts[6]), float(parts[7])
    radial = [float(v) for v in parts[8:14]]
    theta_tab, r_tab = radial_inverse_table(fx, radial)

    points = {}
    for line in open(f"{root}/points3D.txt"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        points[int(f[0])] = (float(f[1]), float(f[2]), float(f[3]))

    thetas, depths, centres = [], [], []
    with open(f"{root}/images.txt") as handle:
        lines = [l for l in handle if not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        header = lines[i].split()
        if len(header) < 10:
            continue
        qw, qx, qy, qz = (float(v) for v in header[1:5])
        tvec = np.array([float(v) for v in header[5:8]])
        rot = quat_to_matrix(qw, qx, qy, qz)
        centre = -rot.T @ tvec
        centres.append(centre)
        tokens = lines[i + 1].split()
        for j in range(0, len(tokens), 3):
            pid = int(tokens[j + 2])
            if pid < 0 or pid not in points:
                continue
            u, v = float(tokens[j]), float(tokens[j + 1])
            r = math.hypot(u - cx, v - cy)
            thetas.append(np.interp(r, r_tab, theta_tab))
            depths.append(float(np.linalg.norm(np.asarray(points[pid]) - centre)))
    centres = np.asarray(centres)
    extent = float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max())
    return np.asarray(thetas), np.asarray(depths), extent, len(centres)


def depth_bins(thetas, depths, edges_deg):
    """Per field-angle bin: inverse-depth mean / std / percentiles over observations."""
    out = []
    edges = np.radians(edges_deg)
    inv = 1.0 / np.maximum(depths, 1e-9)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (thetas >= lo) & (thetas < hi)
        n = int(sel.sum())
        if n < 32:
            out.append(None)
            continue
        w = inv[sel]
        out.append({
            "n": n,
            "inv_mean": float(w.mean()),
            "inv_std": float(w.std()),
            "inv_p10": float(np.percentile(w, 10)),
            "inv_p50": float(np.percentile(w, 50)),
            "inv_p90": float(np.percentile(w, 90)),
        })
    return out


def main():
    edges_deg = np.arange(0.0, 95.0, 5.0)
    centres_deg = (edges_deg[:-1] + edges_deg[1:]) / 2.0
    result = {"bin_edges_deg": edges_deg.tolist(), "bin_centres_deg": centres_deg.tolist(),
              "scenes": {}}

    for scene in SCENES:
        cams = json.load(open(f"{run_dir(scene)}/cameras.json"))
        fx = cams[0]["intrinsics"][0]
        width, height = cams[0]["image_width"], cams[0]["image_height"]
        profile = read_profile(scene)
        cx_list, cz_list = caustic(profile["theta_deg"], profile["z"])
        thetas, depths, extent, n_images = observations(scene)
        bins = depth_bins(thetas, depths, edges_deg)

        # * The quantity the whole argument turns on, in rendered pixels:
        # *   shift(theta, t) = dr/dtheta * z(theta) * sin(theta) / t
        # * A central model can subtract dr/dtheta*z*sin(theta)*E[1/t]; what survives is
        # *   dr/dtheta * |z(theta)| * sin(theta) * std(1/t),
        # * i.e. proportional to the spread of inverse depth *inside one field-angle bin*.
        # *
        # * The conversion factor is the LOCAL plate scale dr/dtheta, not the paraxial focal
        # * length. On this lens they differ by a lot where it matters: the rttpf radial
        # * polynomial compresses the rim, so dr/dtheta falls to ~0.56 fx at theta = 90 deg.
        # * Using fx would overstate every peripheral pixel figure by ~1.8x.
        plate = plate_scale(fx, cams[0]["intrinsics"][4:10], np.radians(centres_deg))
        z_at = np.interp(centres_deg, profile["theta_deg"], profile["z"])
        irreducible, mean_shift = [], []
        for value, angle, entry, scale in zip(z_at, centres_deg, bins, plate):
            if entry is None:
                irreducible.append(None)
                mean_shift.append(None)
                continue
            geo = scale * abs(value) * math.sin(math.radians(angle))
            irreducible.append(geo * entry["inv_std"])
            mean_shift.append(geo * entry["inv_mean"])

        # * Joint (field angle, log depth) histogram: the raw evidence that the depth spread
        # * inside a single field-angle bin is wide. Column-normalised in the figure.
        log_edges = np.linspace(math.log10(0.2), math.log10(60.0), 49)
        hist, _, _ = np.histogram2d(np.degrees(thetas), np.log10(np.maximum(depths, 1e-6)),
                                    bins=[edges_deg, log_edges])

        result["scenes"][scene] = {
            "depth_hist": hist.astype(int).tolist(),
            "depth_hist_log_edges": log_edges.tolist(),
            "focal_px": fx,
            "width": width,
            "height": height,
            "n_train_views": len(cams),
            "n_colmap_images": n_images,
            "n_observations": int(thetas.size),
            "camera_extent": extent,
            "depth_p05": float(np.percentile(depths, 5)),
            "depth_p50": float(np.percentile(depths, 50)),
            "depth_p95": float(np.percentile(depths, 95)),
            "profile": profile,
            "caustic_x": cx_list,
            "caustic_z": cz_list,
            "z_max_abs": float(np.max(np.abs(profile["z"]))),
            "plate_scale_rim": float(plate_scale(fx, cams[0]["intrinsics"][4:10],
                                                 np.array([math.pi / 2]))[0]),
            "dtheta_max_px": float(np.max(np.abs(
                np.asarray(profile["dtheta"])
                * plate_scale(fx, cams[0]["intrinsics"][4:10],
                              np.radians(profile["theta_deg"]))))),
            "bins": bins,
            "irreducible_px": irreducible,
            "mean_shift_px": mean_shift,
        }
        print(f"{scene:11s} f={fx:7.2f}px  extent={extent:6.3f}  "
              f"depth p05/p50/p95={result['scenes'][scene]['depth_p05']:.3f}/"
              f"{result['scenes'][scene]['depth_p50']:.3f}/{result['scenes'][scene]['depth_p95']:.3f}  "
              f"|z|max={result['scenes'][scene]['z_max_abs']:.4f}  "
              f"dtheta_max={result['scenes'][scene]['dtheta_max_px']:.3f}px  "
              f"obs={thetas.size}", flush=True)

    with open(os.path.join(HERE, "optics.json"), "w") as handle:
        json.dump(result, handle)
    print("wrote optics.json")


if __name__ == "__main__":
    main()
