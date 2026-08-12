#!/usr/bin/env python
"""Local plate scale S(theta) = dr/dtheta, and the px -> microradian conversion (task A2).

WHY THIS EXISTS
---------------
Every "calibration error" number in this project is quoted in PIXELS, and pixels are not a
comparable unit across the tracks:

  (a) the tracks train at different downsamplings (-r 4, -r 2, -r 1 / native), so the SAME
      angular residual is a different number of pixels depending on the track;
  (b) the plate scale of a fisheye is NOT fx. For a pinhole dr/dtheta = f / cos^2(theta);
      for a rad_tan_thin_prism_fisheye it falls to ~0.56 fx at the rim on the myscenes
      lens, and an ImmerVision panomorph peaks at ~1.6x its own centre near 60 deg.
      Converting px -> rad with fx is therefore wrong EXACTLY at the rim, i.e. exactly
      where the residual acts.

So the dose-response x axis must be in microradians, and the conversion must use the LOCAL
plate scale of the real COLMAP projection function, differentiated NUMERICALLY.

WHAT IT COMPUTES
----------------
For every camera of every track, at the resolution the run was evaluated at:

  p(theta, phi)   = the true COLMAP forward projection of the bearing
                    (sin t cos f, sin t sin f, cos t), taken straight from
                    gray/fisheye_geometry.py (pinhole handled inline)
  S_rad(theta,phi)= |d p / d theta|   central difference, float64, h = 1e-5 rad  [px/rad]
  S_tan(theta,phi)= |d p / d phi|     the azimuthal companion, = r(theta) for a
                                      perfectly equidistant lens                [px/rad]

S_rad is reported as the mean over phi (plus min/max over phi, which is the anamorphism).
No analytic approximation is used anywhere: the derivative is of the actual function.

THE EDGE OF THE EVALUATED DISK
------------------------------
gray's r=0.95 mask (`gray/config.py:fisheye_mask_radius_scale`,
`gray/fisheye_mask.py`) is a cut in ANGLE, not in pixel radius: it keeps
`theta < 0.95 * 90 deg = 85.5 deg` for thin_prism / rad_tan_thin_prism, and
`theta_d < theta_d(85.5 deg)` for opencv_fisheye, which is the same cut because theta_d is
monotone. So for every fisheye track the evaluated disk ends at 85.5 deg -- unless the frame
runs out first, which is why `theta_edge_deg` is min(85.5 deg, the largest field angle that
still lands inside the image). Pinhole tracks carry no mask at all, so their edge is the
frame CORNER.

usage:  python scripts/analysis/plate_scale.py            # writes plate_scale.json
        python scripts/analysis/plate_scale.py --quick    # skip the mask-fraction pass
No GPU.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from gray.fisheye_geometry import (  # noqa: E402
    opencv_fisheye_project,
    rad_tan_thin_prism_distortion,
    rad_tan_thin_prism_project,
    thin_prism_distortion,
    thin_prism_project,
)

# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

DEG = math.pi / 180.0
MASK_RADIUS_SCALE = 0.95                       # gray/config.py:40
MASK_EDGE_DEG = MASK_RADIUS_SCALE * 90.0       # 85.5 deg -- the cut is in ANGLE
THETA_GRID_DEG = np.round(np.arange(0.0, 90.0 + 1e-9, 0.5), 4)
N_PHI = 180
H_DIFF = 1e-5                                  # rad; float64 central difference


def project(model, params, theta, phi):
    """The real COLMAP forward projection, as (u_pix, v_pix), float64.

    `theta` and `phi` broadcast against each other. The fisheye models are written in the
    (theta, phi) parameterisation rather than through cam_xyz purely to remove the
    removable 1/z singularity at theta = 90 deg: gray's own code computes
    `uu = theta * u / r` with `u = tan(theta) cos(phi)`, `r = tan(theta)`, i.e. exactly
    `uu = theta cos(phi)`. `check_projection_identity()` proves the two agree.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if model == "pinhole":
        fx, fy, cx, cy = (float(v) for v in params[:4])
        t = np.tan(theta)
        return fx * t * np.cos(phi) + cx, fy * t * np.sin(phi) + cy
    if model == "opencv_fisheye":
        fx, fy, cx, cy, k1, k2, k3, k4 = (float(v) for v in params)
        t2 = theta * theta
        theta_d = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
        return fx * theta_d * np.cos(phi) + cx, fy * theta_d * np.sin(phi) + cy
    if model == "thin_prism_fisheye":
        fx, fy, cx, cy = (float(v) for v in params[:4])
        extra = tuple(float(v) for v in params[4:])
        uu, vv = theta * np.cos(phi), theta * np.sin(phi)
        du, dv = thin_prism_distortion(uu, vv, extra)
        return fx * (uu + du) + cx, fy * (vv + dv) + cy
    if model == "rad_tan_thin_prism_fisheye":
        fx, fy, cx, cy = (float(v) for v in params[:4])
        extra = tuple(float(v) for v in params[4:])
        uu, vv = theta * np.cos(phi), theta * np.sin(phi)
        du, dv = rad_tan_thin_prism_distortion(uu, vv, extra)
        return fx * (uu + du) + cx, fy * (vv + dv) + cy
    raise ValueError(f"unsupported camera model: {model}")


def check_projection_identity(model, params, max_theta_deg=88.0):
    """Max |p_ours - p_gray| in px, gray's cam_xyz path vs the (theta, phi) rewrite."""
    if model == "pinhole":
        return 0.0
    theta = np.linspace(1e-4, max_theta_deg * DEG, 400)[:, None]
    phi = np.linspace(0.0, 2 * math.pi, 37)[None, :]
    xyz = np.stack([np.sin(theta) * np.cos(phi) * np.ones_like(phi),
                    np.sin(theta) * np.sin(phi) * np.ones_like(theta),
                    np.cos(theta) * np.ones_like(phi)], axis=-1)
    fn = {"opencv_fisheye": opencv_fisheye_project,
          "thin_prism_fisheye": thin_prism_project,
          "rad_tan_thin_prism_fisheye": rad_tan_thin_prism_project}[model]
    gu, gv = fn(xyz, [float(v) for v in params])
    ou, ov = project(model, params, theta, phi)
    return float(np.nanmax(np.hypot(gu - ou, gv - ov)))


def plate_scales(model, params, theta_deg, n_phi=N_PHI):
    """S_rad, S_tan, r, and frame containment, on the (theta x phi) grid. All px/rad."""
    theta = np.asarray(theta_deg, dtype=np.float64)[:, None] * DEG
    phi = np.linspace(0.0, 2 * math.pi, n_phi, endpoint=False)[None, :]
    cx, cy = float(params[2]), float(params[3])

    up, vp = project(model, params, theta + H_DIFF, phi)
    um, vm = project(model, params, theta - H_DIFF, phi)
    s_rad = np.hypot(up - um, vp - vm) / (2.0 * H_DIFF)

    up, vp = project(model, params, theta, phi + H_DIFF)
    um, vm = project(model, params, theta, phi - H_DIFF)
    s_tan = np.hypot(up - um, vp - vm) / (2.0 * H_DIFF)

    u, v = project(model, params, theta, phi)
    r = np.hypot(u - cx, v - cy)
    return s_rad, s_tan, r, u, v


def frame_reach(u, v, width, height):
    """Per-theta: is any / every azimuth inside the image rectangle?"""
    inside = (u >= 0.0) & (u <= width - 1.0) & (v >= 0.0) & (v <= height - 1.0)
    return inside.any(axis=1), inside.all(axis=1)


def fold_angle_deg(model, params, s_mean, theta_deg):
    """First theta where dr/dtheta stops being positive: past it the lens is not invertible.

    A grid-resolution answer is enough because every calibration here either folds far
    outside the evaluated disk or not at all.
    """
    if model == "pinhole":
        return 90.0
    bad = np.nonzero(np.asarray(s_mean) <= 0.0)[0]
    return float(theta_deg[bad[0]]) if bad.size else 90.0


def frame_max_theta_deg(model, params, width, height, upper_deg, n_phi=1440, refines=4):
    """Largest field angle that still lands inside the image, in degrees.

    Pixel-CENTRE convention, [0.5, W-0.5] x [0.5, H-0.5], the same one
    `gray/fisheye_mask.py` samples with (`arange(n) + 0.5`).

    Bisection in theta for each azimuth, then a zoom on the winning azimuth. A plain
    (theta, phi) grid is not good enough: on `bicycle` the frame corner is reachable in a
    ~0.4-deg-wide azimuth window and a single point of it, so a uniform azimuth grid
    reports the corner low by whatever its spacing is.
    """
    lo_phi, hi_phi = 0.0, 2 * math.pi
    best = 0.0
    for step in range(refines):
        phi = np.linspace(lo_phi, hi_phi, n_phi)
        lo = np.zeros(phi.size)
        hi = np.full(phi.size, min(upper_deg, 89.99) * DEG)

        def inside(t, phi=phi):
            u, v = project(model, params, t, phi)
            ok = (u >= 0.5) & (u <= width - 0.5) & (v >= 0.5) & (v <= height - 0.5)
            return ok & np.isfinite(u) & np.isfinite(v)

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            ok = inside(mid)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        j = int(np.argmax(lo))
        best = float(lo[j])
        if step < refines - 1:
            span = (hi_phi - lo_phi) / (phi.size - 1)
            lo_phi, hi_phi = phi[j] - 2.0 * span, phi[j] + 2.0 * span
    return best / DEG


def theta_edge(model, params, width, height, s_mean, theta_deg):
    """Largest field angle actually evaluated, in degrees, and what limits it.

    Fisheye: gray's r=0.95 mask cuts at theta < 85.5 deg (an ANGLE cut, see module
    docstring), so the edge is 85.5 deg unless the frame or the lens' own invertibility
    runs out first. Pinhole: no mask at all, so the edge is the frame corner.
    """
    fold = fold_angle_deg(model, params, s_mean, theta_deg)
    frame = frame_max_theta_deg(model, params, width, height, fold)
    if model == "pinhole":
        return frame, "frame_corner", frame, fold
    candidates = [(MASK_EDGE_DEG, "mask_r0.95"), (frame, "frame_limited"),
                  (fold, "lens_fold_limited")]
    edge, why = min(candidates, key=lambda c: c[0])
    return edge, why, frame, fold


def area_weighted(theta_deg, s_mean, r_mean, edge_deg, n=2000):
    """Image-area-weighted mean plate scale over [0, edge], dA = r(theta) S(theta) dtheta."""
    t = np.linspace(0.0, edge_deg, n)
    s = np.interp(t, theta_deg, s_mean)
    w = np.interp(t, theta_deg, r_mean) * s
    total = np.trapezoid(w, t) if hasattr(np, "trapezoid") else np.trapz(w, t)
    num = (np.trapezoid(w * s, t) if hasattr(np, "trapezoid") else np.trapz(w * s, t))
    return float(num / total) if total > 0 else float(s_mean[0])


def mask_valid_fraction(model, params, width, height):
    """Fraction of the frame gray's own r=0.95 mask keeps (its code, on CPU)."""
    import torch

    from gray import fisheye_mask as fm

    intr = np.asarray([float(v) for v in params], dtype=np.float64)
    device = torch.device("cpu")
    if model == "opencv_fisheye":
        mask = fm.geometric_valid_mask_opencv_fisheye(intr, height, width, device,
                                                      MASK_RADIUS_SCALE)
    elif model == "thin_prism_fisheye":
        mask = fm.geometric_valid_mask_thin_prism_fisheye(intr, height, width, device,
                                                          MASK_RADIUS_SCALE)
    elif model == "rad_tan_thin_prism_fisheye":
        mask = fm.geometric_valid_mask_rad_tan_thin_prism_fisheye(intr, height, width,
                                                                  device,
                                                                  MASK_RADIUS_SCALE)
    else:
        return 1.0
    return float(mask.float().mean())


# --------------------------------------------------------------------------------------
# where the cameras live
# --------------------------------------------------------------------------------------

WT = ROOT
MAIN = "/workspace/gray"
MYSCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
FC = ["room1", "room2", "room3", "flat1", "flat2", "lab", "lounge", "dark", "persons"]


def track_runs():
    """(track, family, scene, run_dir) for every camera-carrying run on disk."""
    runs = []
    for s in MYSCENES:
        runs.append(("myscenes_rttpf", "fisheye_circular", s,
                     f"{MAIN}/tmp/final/{s}_noncentral"))
    for d in sorted(glob.glob(f"{WT}/tmp/mipnerf360/*_noncentral")):
        runs.append(("mipnerf360_pinhole", "pinhole",
                     os.path.basename(d).replace("_noncentral", ""), d))
    for d in sorted(glob.glob(f"{WT}/tmp/mipnerf360/*_off")):
        scene = os.path.basename(d).replace("_off", "")
        if not any(r[2] == scene and r[0] == "mipnerf360_pinhole" for r in runs):
            runs.append(("mipnerf360_pinhole", "pinhole", scene, d))
    runs.append(("workshop_immervision_rttpf", "panomorph", "workshop_immervision",
                 f"{WT}/out/workshop_immervision_noncentral"))
    runs.append(("workshop_immervision_ocv", "panomorph", "workshop_immervision",
                 f"{MAIN}/out/ocv/workshop_immervision_ocv"))
    runs.append(("workshop_fujinon_rttpf", "fisheye_fullframe", "workshop_fujinon",
                 f"{MAIN}/out/workshop_fujinon_fisheye_baseline"))
    for s in FC:
        runs.append(("fullcircle_refit_rttpf", "fisheye_circular", s,
                     f"{WT}/out/fullcircle_rttpf/{s}_refit_rttpf"))
    for d in sorted(glob.glob(f"{MAIN}/out/ocv/*")):
        name = os.path.basename(d)
        if name.startswith("workshop_immervision"):
            continue
        runs.append(("myscenes_ocv", "fisheye_circular", name, d))
    return [r for r in runs if os.path.exists(os.path.join(r[3], "cameras.json"))]


def cameras_of(run_dir):
    """Unique (uid, model, intrinsics, W, H) of a run, at the resolution it was rendered at.

    `cameras.json` already carries the intrinsics SCALED to the working resolution, which is
    the resolution every masked metric in this project is computed at. Pinhole entries carry
    `intrinsics: null`, so fx/fy are recovered from the FOV the renderer actually used.
    """
    with open(os.path.join(run_dir, "cameras.json")) as handle:
        cams = json.load(handle)
    out = {}
    for entry in cams:
        uid = int(entry["uid"])
        if uid in out:
            continue
        width, height = int(entry["image_width"]), int(entry["image_height"])
        model = entry["model"]
        intr = entry.get("intrinsics")
        if intr is None:
            if model != "pinhole":
                raise ValueError(f"{run_dir}: {model} with no intrinsics")
            fx = (width / 2.0) / math.tan(entry["fov_x"] / 2.0)
            fy = (height / 2.0) / math.tan(entry["fov_y"] / 2.0)
            intr = [fx, fy, width / 2.0, height / 2.0]
        out[uid] = {"uid": uid, "model": model, "params": [float(v) for v in intr],
                    "width": width, "height": height}
    down = 1.0
    cfg = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg):
        with open(cfg) as handle:
            down = float(json.load(handle).get("downsampling", 1) or 1)
    for cam in out.values():
        cam["downsampling"] = down
    return out


# --------------------------------------------------------------------------------------
# the learned residual (E_learned) straight out of camera_model_15000.csv
# --------------------------------------------------------------------------------------

def read_camera_model_csv(path):
    """{uid: {theta_deg, delta_theta_rad, delta_phi_rad, z_scene_units} as arrays}."""
    rows = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            uid = int(row["uid"])
            rows.setdefault(uid, []).append(
                (float(row["theta_deg"]), float(row["delta_theta_rad"]),
                 float(row["delta_phi_rad"]), float(row["z_scene_units"])))
    out = {}
    for uid, items in rows.items():
        arr = np.asarray(sorted(items), dtype=np.float64)
        out[uid] = {"theta_deg": arr[:, 0], "dtheta": arr[:, 1],
                    "dphi": arr[:, 2], "z": arr[:, 3]}
    return out


def interp_at(theta_deg_grid, values, theta_deg):
    return float(np.interp(theta_deg, theta_deg_grid, values))


def spline_dtheta_at(run_dir, uid, theta_deg):
    """Channel-0 residual read straight off the checkpoint spline, at an arbitrary theta.

    `camera_model_*.csv` is tabulated on a 1-deg grid; linear interpolation of it at, say,
    a pinhole's 32.57-deg corner is biased where the curve is convex. This is the exact
    value, used only to size that bias.
    """
    try:
        import torch
        from safetensors import safe_open

        from gray.camera_model import bspline_eval
    except Exception:
        return None
    ckpts = sorted(glob.glob(os.path.join(run_dir, "gaussians_*.safetensors")))
    if not ckpts:
        return None
    key = f"camera_model.lenses.{uid}.theta_weights"
    with safe_open(ckpts[-1], "pt") as handle:
        if key not in handle.keys():
            return None
        weights = handle.get_tensor(key).float()
    t01 = torch.tensor([min(max(theta_deg / 90.0, 0.0), 1.0)], dtype=torch.float32)
    return float(bspline_eval(weights, t01)[0, 0])


def monotonicity(theta_deg, values):
    """How close to monotone the residual is over the field actually seen."""
    if values.size < 3:
        return {}
    diffs = np.diff(values)
    dominant = 1.0 if diffs.sum() >= 0 else -1.0
    agree = float(np.mean(np.sign(diffs) == dominant))
    ranks = lambda a: np.argsort(np.argsort(a)).astype(float)  # noqa: E731
    rt, rv = ranks(theta_deg), ranks(values)
    spearman = float(np.corrcoef(rt, rv)[0, 1])
    return {"strictly_monotone": bool(agree == 1.0),
            "fraction_of_steps_in_dominant_direction": agree,
            "spearman_theta_vs_dtheta": spearman,
            "direction": "increasing" if dominant > 0 else "decreasing"}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def build_camera_table(quick=False):
    table, order = {}, []
    for track, family, scene, run in track_runs():
        for uid, cam in sorted(cameras_of(run).items()):
            key = f"{track}/{scene}/cam{uid}"
            model, params = cam["model"], cam["params"]
            width, height = cam["width"], cam["height"]
            s_rad, s_tan, r, u, v = plate_scales(model, params, THETA_GRID_DEG)
            in_any, in_all = frame_reach(u, v, width, height)
            s_mean = s_rad.mean(axis=1)
            edge_deg, edge_src, frame_deg, fold_deg = theta_edge(
                model, params, width, height, s_mean, THETA_GRID_DEG)

            fx = float(params[0])

            def at(deg, arr=s_mean):
                return float(np.interp(deg, THETA_GRID_DEG, arr))

            entry = {
                "track": track, "scene": scene, "uid": uid, "family": family,
                "model": model, "run_dir": run,
                "eval_width": width, "eval_height": height,
                "downsampling": cam["downsampling"],
                "intrinsics_eval": params,
                "fx": fx, "fy": float(params[1]),
                "cx": float(params[2]), "cy": float(params[3]),
                "projection_identity_max_px": check_projection_identity(model, params),
                "S_rad_px_per_rad": [round(x, 4) for x in s_mean],
                "S_rad_min_over_phi": [round(x, 4) for x in s_rad.min(axis=1)],
                "S_rad_max_over_phi": [round(x, 4) for x in s_rad.max(axis=1)],
                "S_tan_px_per_rad": [round(x, 4) for x in s_tan.mean(axis=1)],
                "r_px": [round(x, 4) for x in r.mean(axis=1)],
                "S_over_fx": [round(x / fx, 6) for x in s_mean],
                "in_frame_any": [bool(x) for x in in_any],
                "in_frame_all": [bool(x) for x in in_all],
                "theta_edge_deg": edge_deg,
                "theta_edge_source": edge_src,
                "theta_frame_max_deg": frame_deg,
                "theta_lens_fold_deg": fold_deg,
                "S_paraxial_px_per_rad": at(0.0),
                "S_edge_px_per_rad": at(edge_deg),
                "S_edge_over_fx": at(edge_deg) / fx,
                "S_edge_over_S0": at(edge_deg) / at(0.0),
                "S_at_85p5deg_px_per_rad": at(MASK_EDGE_DEG) if model != "pinhole" else None,
                # * meaningless for a pinhole (tan(90 deg) diverges) and unreachable anyway
                "S_at_90deg_over_fx": at(90.0) / fx if model != "pinhole" else None,
                "S_tan_edge_px_per_rad": at(edge_deg, s_tan.mean(axis=1)),
                "r_edge_px": at(edge_deg, r.mean(axis=1)),
                "anamorphism_at_edge": (at(edge_deg, s_rad.max(axis=1))
                                        / max(at(edge_deg, s_rad.min(axis=1)), 1e-12)),
                # * area-weighted mean of S over the evaluated disk: dA = r dr dphi
                # * = r(theta) S(theta) dtheta dphi. This is the RIGHT conversion for a
                # * quantity pooled over the whole image (an SfM reprojection median, say),
                # * where the edge value is the right one for a rim-located residual.
                "S_area_weighted_px_per_rad": area_weighted(
                    THETA_GRID_DEG, s_mean, r.mean(axis=1), edge_deg),
                "urad_per_px_area_weighted": 1e6 / area_weighted(
                    THETA_GRID_DEG, s_mean, r.mean(axis=1), edge_deg),
                "S_peak_px_per_rad": float(s_mean.max()),
                "S_peak_theta_deg": float(THETA_GRID_DEG[int(np.argmax(s_mean))]),
                "S_peak_over_S0": float(s_mean.max() / s_mean[0]),
                # the conversion the dose-response axis needs
                "urad_per_px_at_edge": 1e6 / at(edge_deg),
                "px_per_urad_at_edge": at(edge_deg) * 1e-6,
            }
            if not quick:
                entry["mask_valid_fraction"] = (
                    mask_valid_fraction(model, params, width, height)
                    if model != "pinhole" else 1.0)
            table[key] = entry
            order.append(key)
    return table, order


def track_reference(table):
    """One reference camera per track: the median S at the edge over that track's cameras."""
    per_track = {}
    for key, entry in table.items():
        per_track.setdefault(entry["track"], []).append((key, entry))
    out = {}
    for track, items in per_track.items():
        edges = np.asarray([e["S_edge_px_per_rad"] for _, e in items])
        pick = int(np.argsort(edges)[len(edges) // 2])
        key, entry = items[pick]
        area = np.asarray([e["S_area_weighted_px_per_rad"] for _, e in items])
        out[track] = {
            "n_cameras": len(items),
            "family": entry["family"],
            "model": entry["model"],
            "downsampling": entry["downsampling"],
            "eval_size": [entry["eval_width"], entry["eval_height"]],
            "median_camera": key,
            # * a track is only summarisable by one number when its cameras share a
            # * resolution AND a downsampling: mip-NeRF 360 does not (outdoor -r 4,
            # * indoor -r 2), so the same angular residual is 2x the pixels on half of it.
            "heterogeneous": bool(
                len({e["downsampling"] for _, e in items}) > 1
                or len({(e["eval_width"], e["eval_height"]) for _, e in items}) > 1),
            "downsamplings_present": sorted({e["downsampling"] for _, e in items}),
            "eval_sizes_present": sorted({(e["eval_width"], e["eval_height"])
                                          for _, e in items}),
            "theta_edge_deg_range": [float(min(e["theta_edge_deg"] for _, e in items)),
                                     float(max(e["theta_edge_deg"] for _, e in items))],
            "theta_edge_deg": entry["theta_edge_deg"],
            "theta_edge_source": entry["theta_edge_source"],
            "S_edge_px_per_rad": float(np.median(edges)),
            "S_edge_spread_px_per_rad": float(edges.max() - edges.min()),
            "S_paraxial_px_per_rad": float(np.median(
                [e["S_paraxial_px_per_rad"] for _, e in items])),
            "S_edge_over_fx": float(np.median([e["S_edge_over_fx"] for _, e in items])),
            "S_area_weighted_px_per_rad": float(np.median(area)),
            "mask_valid_fraction": float(np.median(
                [e["mask_valid_fraction"] for _, e in items if "mask_valid_fraction" in e]))
            if any("mask_valid_fraction" in e for _, e in items) else None,
            # the two conversions W1/W2 need; pick by where the quantity lives
            "urad_per_px_at_edge": 1e6 / float(np.median(edges)),
            "urad_per_px_area_weighted": 1e6 / float(np.median(area)),
        }
    return out


def dose_response_x_axis(refs, angles, learned):
    """One row per track: the estimators of calibration error, all in microradians."""
    by_track = {}
    for row in learned:
        by_track.setdefault(row["track"], []).append(row["dtheta_edge_urad"])
    shared = {(a["track"], a["uid"]): a for a in angles}
    rows = []
    for track, ref in sorted(refs.items()):
        vals = np.abs(np.asarray(by_track.get(track, []), dtype=np.float64))
        lens_rows = [a for (t, _), a in shared.items() if t == track]
        rows.append({
            "track": track, "family": ref["family"], "model": ref["model"],
            "downsampling": ref["downsampling"], "eval_size": ref["eval_size"],
            "theta_edge_deg": ref["theta_edge_deg"],
            "S_edge_px_per_rad": ref["S_edge_px_per_rad"],
            "urad_per_px_at_edge": ref["urad_per_px_at_edge"],
            "urad_per_px_area_weighted": ref["urad_per_px_area_weighted"],
            "E_shared_urad_at_edge": [
                {"uid": a["uid"], "urad": a["common_urad_at_edge"],
                 "px_true_scale": a["common_px_at_edge_true_scale"],
                 "n_scenes": a["n_scenes"]} for a in lens_rows] or None,
            "E_learned_urad_at_edge_mean": float(vals.mean()) if vals.size else None,
            "E_learned_urad_at_edge_median": float(np.median(vals)) if vals.size else None,
            "E_learned_urad_at_edge_n": int(vals.size),
            "E_sfm_note": ("not computed here: convert a median SfM reprojection residual "
                           "with urad_per_px_area_weighted (it is pooled over the frame), "
                           "not with urad_per_px_at_edge"),
        })
    return rows


# ---- conversion of the three estimators -----------------------------------------------

CALIB_TRACK = {
    "myscenes (one lens, 7 independent COLMAP fits)": "myscenes_rttpf",
    "FullCircle refit_rttpf (lens 1, 9 independent re-fits)": "fullcircle_refit_rttpf",
    "FullCircle refit_rttpf (lens 2, 9 independent re-fits)": "fullcircle_refit_rttpf",
    "FullCircle refit_rttpf (lens 1)": "fullcircle_refit_rttpf",
    "FullCircle refit_rttpf (lens 2)": "fullcircle_refit_rttpf",
}


KNOWN_ISSUES = [
    {
        "id": "calib_consistency_radial_layout",
        "where": "scripts/analysis/calib_consistency.py:54 (RADIAL = [4, 5, 8, 9]), inherited "
                 "by scripts/analysis/residual_expressible.py via `import calib_consistency`",
        "what": "RADIAL = [4, 5, 8, 9] is the COLMAP THIN_PRISM_FISHEYE parameter layout "
                "(fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1). Every camera those scripts read is "
                "RAD_TAN_THIN_PRISM_FISHEYE, laid out fx fy cx cy k0 k1 k2 k3 k4 k5 p0 p1 "
                "s0 s1 s2 s3: six radial coefficients at indices 4..9 carrying theta^2 .. "
                "theta^12. The subset therefore drops k2 and k3 and re-labels k4, k5 as the "
                "theta^6 and theta^8 terms.",
        "verified_by": "gray/fisheye_geometry.py:rad_tan_thin_prism_distortion, and "
                       "pycolmap reporting model=RAD_TAN_THIN_PRISM_FISHEYE with 16 params "
                       "on data/myscenes/*/distorted/sparse/0 and on "
                       "/workspace/dataset/fullcircle_tracks/refit_rttpf/*/distorted/sparse/0",
        "size_of_the_error": {
            "plate_scale_at_85.5deg_published_over_true": {
                "myscenes": 1.142, "fullcircle_lens1": 0.637, "fullcircle_lens2": 0.699},
            "calib_disagreement_px_published_vs_true_at_85.5deg": {
                "myscenes": [2.970, 0.360], "fullcircle_lens1": [1.273, 0.069],
                "fullcircle_lens2": [2.393, 0.045]},
            "true_disagreement_is_insensitive_to_the_evaluation_angle": {
                "myscenes_urad_at_85.5_88.7_90_deg": [1834.4, 2236.2, 2469.7]},
        },
        "why_it_matters": "the plan's section 2.3 argues that cross-scene calibration "
                          "DISAGREEMENT predicts nothing, because FullCircle lens 2 (2.393 "
                          "px) sits near myscenes (2.970 px) and gains no PSNR. Recomputed "
                          "through the real projection the two are 0.045 px and 0.360 px, a "
                          "factor 8, ordered exactly like the gains. That argument has to be "
                          "re-checked against corrected numbers before it is written up. "
                          "`common_px` / `deviation_px` / `total_px` move much less (the "
                          "ANGLES they come from are read off the checkpoint and are "
                          "untouched); it is the pixel conversion and the theta grid that "
                          "move.",
        "status": "FIXED 2026-08-12 at the source. `calib_consistency.RADIAL` is now "
                  "`RADIAL_BY_NPARAM` (16 -> 4..9, 12 -> [4,5,8,9], 8 -> 4..7) and raises on "
                  "an unknown layout; `residual_expressible.BASIS` is now built from that "
                  "same layout (`basis_for`), rttpf being seven terms t..t^13, not five. "
                  "Both JSONs were regenerated. This entry stays as the audit trail, and "
                  "plate_scale.py still recomputes independently -- keep it that way.",
        "post_fix_numbers": {
            "calib_disagreement_px": {"myscenes": 0.397, "fullcircle_lens1": 0.095,
                                      "fullcircle_lens2": 0.049},
            "cross_scene_disagreement_urad_rms_via_pycolmap": {
                "myscenes": 868.7, "fullcircle_lens1": 184.4, "fullcircle_lens2": 129.9},
            "agrees_with_this_file": "yes -- plate_scale predicted 0.360 / 0.069 / 0.045 px "
                                     "from the real projection; the fixed polynomial gives "
                                     "0.397 / 0.095 / 0.049 (residual gap = the theta range, "
                                     "fold vs 85.5 deg mask edge, not the layout)",
            "consequence": "the 'disagreement predicts nothing' argument is DEAD. Corrected, "
                           "disagreement orders myscenes 4.7-6.7x above FullCircle, i.e. the "
                           "same order as the gains, and sits within ~10-40 % of the shared "
                           "bias (791/869, 208/184, 197/130 urad). Shared bias and "
                           "disagreement are NOT separable on this evidence -- do not claim "
                           "they are. Nothing load-bearing depends on it: the 2x2 pinhole "
                           "control, the rttpf_z decomposition and the peripheral signature "
                           "never used this script.",
        },
    },
    {
        "id": "published_maximum_outside_the_evaluated_disk",
        "where": "calib_consistency.json common_px for myscenes",
        "what": "its maximum falls at theta = 88.7 deg, outside gray's own r=0.95 mask "
                "(85.5 deg), so the headline 0.393 px is quoted at a field angle no metric "
                "in this project ever scores. `common_urad_at_edge` is the in-disk value.",
    },
    {
        "id": "camera_model_csv_covers_0_to_90_always",
        "where": "camera_model_*.csv written by train.py",
        "what": "the spline is parameterised on theta / (pi/2) and the CSV is dumped over "
                "the full 0..90 deg regardless of the camera's real field, so on a pinhole "
                "(bicycle: 32.6 deg) two thirds of the rows are untrained extrapolation. "
                "`learned_residual` here only ever reads rows at or below theta_edge_deg.",
    },
    {
        "id": "csv_theta_grid_is_1_degree",
        "where": "camera_model_*.csv",
        "what": "interpolating it at a pinhole corner biases |dtheta| slightly where the "
                "curve is convex. `dtheta_edge_urad_exact_spline` reads the checkpoint "
                "spline at the exact angle; on every run here the two agree to < 2 %.",
    },
]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


# --------------------------------------------------------------------------------------
# E_shared re-derived in ANGLE, so the urad axis does not inherit a plate-scale choice
# --------------------------------------------------------------------------------------

CALIB_FAMILIES = [
    ("myscenes (one lens, 7 independent COLMAP fits)", "myscenes_rttpf", 1),
    ("FullCircle refit_rttpf (lens 1, 9 independent re-fits)", "fullcircle_refit_rttpf", 1),
    ("FullCircle refit_rttpf (lens 2, 9 independent re-fits)", "fullcircle_refit_rttpf", 2),
]


def _calib_inputs(uid, which):
    import calib_consistency as C
    if which == "myscenes":
        return (list(C.MYSCENES),
                lambda s: f"{C.MY_ROOT}/{C.MYSCENES[s]}/distorted/sparse/0",
                lambda s: C.MY_RUNS.format(scene=s))
    return (C.FC, lambda s: f"{C.FC_ROOT}/{s}/distorted/sparse/0",
            lambda s: C.FC_RUNS.format(scene=s))


def estimator_angles(refs):
    """Re-derive E_shared / E_deviation / E_calib-disagreement in RADIANS.

    `calib_consistency.py` reports them in pixels through its own plate scale. Two reasons
    to redo it here:

      1. urad is the unit the dose-response axis needs, and dividing a published pixel
         figure by a plate scale evaluated at a DIFFERENT theta than the one the maximum
         came from is not a conversion, it is a guess.
      2. `calib_consistency.RADIAL = [4, 5, 8, 9]` is the THIN_PRISM_FISHEYE radial layout
         (fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1). Every camera it is actually run on is
         RAD_TAN_THIN_PRISM_FISHEYE, whose 16 params are fx fy cx cy k0..k5 p0 p1 s0..s3 --
         six radial coefficients at 4..9 carrying theta^2..theta^12. So it drops k2, k3 and
         re-labels k4, k5 as the theta^6 / theta^8 terms. This function computes both, so
         the size of that is a number rather than an opinion.

    Angles are identical under both (they come from the checkpoint), only the pixel figures
    and the theta grid the residual is sampled on differ.
    """
    sys.path.insert(0, HERE)
    import calib_consistency as C

    out = []
    for name, track, uid in CALIB_FAMILIES:
        which = "myscenes" if track == "myscenes_rttpf" else "fullcircle"
        scenes, sparse_of, run_of = _calib_inputs(uid, which)
        try:
            import pycolmap
        except ImportError:
            return out
        fits = []
        for scene in scenes:
            sparse = sparse_of(scene)
            if not os.path.isdir(sparse):
                continue
            rec = pycolmap.Reconstruction(sparse)
            if uid not in rec.cameras:
                continue
            cam = rec.cameras[uid]
            params = list(cam.params)
            model = cam.model.name.lower()
            # published path: calib_consistency's 4-term radial subset
            grid = np.linspace(0.0, np.pi / 2, 4000)
            radii_pub = C.r_of_theta(params[0], C.radial_poly(params), grid)
            drop = np.diff(radii_pub) <= 0
            fold = np.argmax(drop) if drop.any() else len(grid) - 1
            # true path: the real projection function of this camera model
            s_rad, _, r_true, _, _ = plate_scales(model, params, grid / DEG, n_phi=64)
            fits.append({"scene": scene, "params": params, "model": model,
                         "theta_max_pub": grid[fold], "r_max_pub": radii_pub[fold],
                         "grid": grid, "r_true": r_true.mean(axis=1),
                         "s_true": s_rad.mean(axis=1)})
        if not fits:
            continue
        down = C.run_downsampling(run_of(fits[0]["scene"]))

        # ---- published convention -------------------------------------------------
        r_px = C.GRID * min(f["r_max_pub"] for f in fits)
        th_pub, sc_pub, d_pub = [], [], []
        for f in fits:
            theta = C.theta_of_r(f["params"][0], C.radial_poly(f["params"]), r_px,
                                 f["theta_max_pub"])
            dtheta = C.learned_dtheta(run_of(f["scene"]), uid, theta)
            if dtheta is None:
                continue
            th_pub.append(theta)
            sc_pub.append(C.plate_scale(f["params"][0], C.radial_poly(f["params"]), theta))
            d_pub.append(dtheta)
        if not d_pub:
            continue
        th_pub = np.stack(th_pub)
        px_pub = np.stack(sc_pub).mean(0) / down
        d_pub = np.stack(d_pub)
        common_pub, dev_pub = d_pub.mean(0), d_pub.std(0)
        j = int(np.argmax(np.abs(common_pub * px_pub)))

        # ---- true convention, on a common ANGLE grid out to the evaluated disk edge --
        edge = refs[track]["theta_edge_deg"] * DEG
        theta_common = np.linspace(0.02 * edge, edge, 200)
        d_true, sc_true, r_true_all = [], [], []
        for f in fits:
            dtheta = C.learned_dtheta(run_of(f["scene"]), uid, theta_common)
            if dtheta is None:
                continue
            d_true.append(dtheta)
            sc_true.append(np.interp(theta_common, f["grid"], f["s_true"]) / down)
            r_true_all.append(np.interp(theta_common, f["grid"], f["r_true"]) / down)
        d_true = np.stack(d_true)
        sc_true = np.stack(sc_true).mean(0)
        common_true, dev_true = d_true.mean(0), d_true.std(0)
        # calibration disagreement: spread of theta(rho) across the independent fits,
        # on a common pixel-radius grid built from the TRUE r(theta)
        rho = np.linspace(0.02, 1.0, 60) * min(
            np.interp(edge, f["grid"], f["r_true"]) for f in fits)
        th_true = np.stack([np.interp(rho, f["r_true"], f["grid"]) for f in fits])
        disagree_urad = th_true.std(0) * 1e6
        disagree_px_true = th_true.std(0) * np.stack(
            [np.interp(np.interp(rho, f["r_true"], f["grid"]), f["grid"], f["s_true"])
             for f in fits]).mean(0) / down

        out.append({
            "name": name, "track": track, "uid": uid, "n_scenes": int(d_true.shape[0]),
            "colmap_model": fits[0]["model"], "downsampling": down,
            "theta_edge_deg": refs[track]["theta_edge_deg"],
            # --- as published (validates that this reproduces calib_consistency.json)
            "published_common_px": float(np.abs(common_pub * px_pub).max()),
            "published_deviation_px": float((dev_pub * px_pub).max()),
            "published_argmax_theta_deg": float(th_pub.mean(0)[j] / DEG),
            "published_argmax_inside_evaluated_disk": bool(
                th_pub.mean(0)[j] / DEG <= refs[track]["theta_edge_deg"] + 1e-9),
            "published_plate_scale_at_argmax_px_per_rad": float(px_pub[j]),
            # --- the angle behind it: exact, unit-free of any plate scale
            "common_urad_at_published_argmax": float(abs(common_pub[j]) * 1e6),
            # --- corrected, at the edge of the evaluated disk (what W1 should use)
            "common_urad_at_edge": float(abs(common_true[-1]) * 1e6),
            "common_px_at_edge_true_scale": float(abs(common_true[-1] * sc_true[-1])),
            "common_urad_max": float(np.abs(common_true).max() * 1e6),
            "common_urad_max_theta_deg": float(
                theta_common[int(np.argmax(np.abs(common_true)))] / DEG),
            "deviation_urad_at_edge": float(dev_true[-1] * 1e6),
            "deviation_px_at_edge_true_scale": float(dev_true[-1] * sc_true[-1]),
            "calib_disagreement_urad_at_edge": float(disagree_urad[-1]),
            "calib_disagreement_urad_max": float(disagree_urad.max()),
            "calib_disagreement_px_at_edge_true_scale": float(disagree_px_true[-1]),
            "true_plate_scale_at_edge_px_per_rad": float(sc_true[-1]),
            "plate_scale_ratio_published_over_true_at_edge": float(
                C.plate_scale(np.mean([f["params"][0] for f in fits]),
                              C.radial_poly(fits[0]["params"]),
                              np.array([edge]))[0] / down / sc_true[-1]),
        })
    return out


def convert_px_json(entries, fields, refs, label):
    """px -> urad for an existing analysis JSON, at each track's evaluated-disk edge."""
    out = []
    for entry in entries or []:
        track = CALIB_TRACK.get(entry["name"])
        if track is None or track not in refs:
            continue
        ref = refs[track]
        row = {"source": label, "name": entry["name"], "track": track,
               "uid": entry.get("uid"),
               "theta_edge_deg": ref["theta_edge_deg"],
               "S_edge_px_per_rad": ref["S_edge_px_per_rad"],
               "CAVEAT": ("naive conversion: it divides the PUBLISHED pixel figure by the "
                          "TRUE plate scale at the evaluated-disk edge, while the published "
                          "figure was produced with a different plate scale at a different "
                          "theta (see known_issues). Use estimator_angles_rederived for "
                          "E_shared; these rows exist so the published numbers have a "
                          "microradian label at all.")}
        for field in fields:
            if field not in entry:
                continue
            row[field] = entry[field]
            row[field.replace("_px", "") + "_urad"] = (
                entry[field] / ref["S_edge_px_per_rad"] * 1e6)
        out.append(row)
    return out


def numerical_derivative_check():
    """The numerical dr/dtheta against the two cases where the answer is known in closed form.

    pinhole  : dr/dtheta = f / cos^2(theta), exactly.
    rttpf    : with the tangential and thin-prism coefficients zeroed, the projection is
               r = fx theta P(theta) and dr/dtheta = fx (P + theta P'), so the derivative
               of the 6-term radial polynomial is the answer.
    """
    theta = np.linspace(0.5, 80.0, 300)
    s, _, _, _, _ = plate_scales("pinhole", [1000.0, 1000.0, 0.0, 0.0], theta, n_phi=16)
    analytic = 1000.0 / np.cos(theta * DEG) ** 2
    pin = float(np.max(np.abs(s.mean(axis=1) - analytic) / analytic))

    k = [-0.0340224, -0.00088329, -0.000581494, -0.000599972, 0.000184274, -2.91702e-05]
    params = [1240.52, 1240.52, 0.0, 0.0] + k + [0.0] * 6
    theta = np.linspace(0.5, 90.0, 300)
    s, _, _, _, _ = plate_scales("rad_tan_thin_prism_fisheye", params, theta, n_phi=16)
    t = theta * DEG
    poly = np.ones_like(t)
    dpoly = np.zeros_like(t)
    for order, ki in enumerate(k, start=1):
        poly = poly + ki * t ** (2 * order)
        dpoly = dpoly + ki * (2 * order) * t ** (2 * order - 1)
    analytic = 1240.52 * (poly + t * dpoly)
    rttpf = float(np.max(np.abs(s.mean(axis=1) - analytic) / np.abs(analytic)))
    return {"pinhole_max_relative_error": pin,
            "rttpf_radial_only_max_relative_error": rttpf,
            "S_over_fx_at_90deg_for_the_myscenes_tunnel_radial_terms":
                float(analytic[-1] / 1240.52),
            "note": "both must be ~1e-9; the second also reproduces the 0.56 fx rim figure"}


def learned_residual_rows(table):
    """E_learned, straight from camera_model_15000.csv, in urad AND in px through S."""
    rows = []
    for key, entry in table.items():
        run = entry["run_dir"]
        csv_path = os.path.join(run, "camera_model_15000.csv")
        if not os.path.exists(csv_path):
            continue
        data = read_camera_model_csv(csv_path)
        if entry["uid"] not in data:
            continue
        d = data[entry["uid"]]
        edge = entry["theta_edge_deg"]
        grid = THETA_GRID_DEG
        s_mean = np.asarray(entry["S_rad_px_per_rad"], dtype=np.float64)
        s_tan = np.asarray(entry["S_tan_px_per_rad"], dtype=np.float64)
        # * only the part of the tabulated residual the camera can actually see: the CSV
        # * always runs 0..90 deg because the spline is parameterised on theta/(pi/2),
        # * but a pinhole at 32.6 deg never trains past its own corner.
        inside = d["theta_deg"] <= edge + 1e-9
        dtheta_in = d["dtheta"][inside]
        s_in = np.interp(d["theta_deg"][inside], grid, s_mean)
        st_in = np.interp(d["theta_deg"][inside], grid, s_tan)
        px_in = dtheta_in * s_in
        tan_px_in = d["dphi"][inside] * st_in
        j = int(np.argmax(np.abs(px_in))) if px_in.size else 0
        rows.append({
            "camera": key, "track": entry["track"], "scene": entry["scene"],
            "uid": entry["uid"], "run_dir": run,
            "theta_edge_deg": edge,
            "S_edge_px_per_rad": entry["S_edge_px_per_rad"],
            "dtheta_edge_urad": interp_at(d["theta_deg"], d["dtheta"], edge) * 1e6,
            "dtheta_edge_px": (interp_at(d["theta_deg"], d["dtheta"], edge)
                               * entry["S_edge_px_per_rad"]),
            "dphi_edge_urad": interp_at(d["theta_deg"], d["dphi"], edge) * 1e6,
            "dphi_edge_px": (interp_at(d["theta_deg"], d["dphi"], edge)
                             * entry["S_tan_edge_px_per_rad"]),
            "dtheta_absmax_urad": float(dtheta_in[j] * 1e6) if px_in.size else None,
            "dtheta_absmax_px": float(px_in[j]) if px_in.size else None,
            "dtheta_absmax_theta_deg": (float(d["theta_deg"][inside][j])
                                        if px_in.size else None),
            "dphi_absmax_px": (float(tan_px_in[np.argmax(np.abs(tan_px_in))])
                               if tan_px_in.size else None),
            "radial_over_tangential_px": (
                float(np.abs(px_in).max() / max(np.abs(tan_px_in).max(), 1e-12))
                if px_in.size else None),
            "monotonicity_in_field": monotonicity(d["theta_deg"][inside], dtheta_in),
            "dtheta_edge_over_dtheta_centre": (
                float(interp_at(d["theta_deg"], d["dtheta"], edge) / d["dtheta"][0])
                if abs(d["dtheta"][0]) > 0 else None),
            "dtheta_edge_urad_exact_spline": (
                (lambda x: None if x is None else x * 1e6)(
                    spline_dtheta_at(run, entry["uid"], edge))),
            "csv_grid_deg": float(np.median(np.diff(d["theta_deg"]))),
            "z_absmax_scene_units": float(np.abs(d["z"][inside]).max()) if px_in.size else None,
        })
    return sorted(rows, key=lambda r: (r["track"], r["scene"], r["uid"]))


def bicycle_check(table, learned):
    """The handover's sanity check: bicycle's learned residual ~ -0.80 px at the corner."""
    key = "mipnerf360_pinhole/bicycle/cam1"
    entry, row = table.get(key), next((r for r in learned if r["camera"] == key), None)
    if entry is None or row is None:
        return {"status": "missing"}
    f_expected = 4649.505977743847 / 4.0
    csv_path = os.path.join(entry["run_dir"], "camera_model_15000.csv")
    d = read_camera_model_csv(csv_path)[1]
    s_mean = np.asarray(entry["S_rad_px_per_rad"], dtype=np.float64)
    at32 = float(np.interp(32.0, d["theta_deg"], d["dtheta"])
                 * np.interp(32.0, THETA_GRID_DEG, s_mean))
    at_iso = float(np.interp(32.5545, d["theta_deg"], d["dtheta"]) * f_expected
                   / math.cos(32.5545 * DEG) ** 2)
    return {
        "expected_px_at_corner": -0.80,
        "expected_theta_max_deg": 32.56,
        "expected_f_px": f_expected,
        "measured_theta_edge_deg": entry["theta_edge_deg"],
        "measured_fx_px": entry["fx"],
        "measured_fy_px": entry["fy"],
        "measured_S_edge_px_per_rad": entry["S_edge_px_per_rad"],
        "S_edge_over_fx": entry["S_edge_over_fx"],
        "measured_px_at_edge": row["dtheta_edge_px"],
        "measured_px_at_theta32": at32,
        "measured_px_isotropic_corner_32p5545deg": at_iso,
        "measured_urad_at_edge": row["dtheta_edge_urad"],
        "measured_urad_at_edge_exact_spline": row["dtheta_edge_urad_exact_spline"],
        "tangential_px_at_edge": row["dphi_edge_px"],
        "radial_over_tangential_px": row["radial_over_tangential_px"],
        "monotonicity_in_field": row["monotonicity_in_field"],
        "dtheta_edge_over_dtheta_centre": row["dtheta_edge_over_dtheta_centre"],
        "note": ("the handover's -0.80 px is measured_px_at_theta32, i.e. the CSV read at "
                 "its nearest tabulated row (theta = 32 deg); evaluating at the true corner "
                 "theta = 32.57 deg gives measured_px_at_edge, which is larger because "
                 "|dtheta| is still steepening there"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the exact mask-fraction pass (the slow part)")
    ap.add_argument("--out", default=os.path.join(HERE, "plate_scale.json"))
    args = ap.parse_args()

    table, order = build_camera_table(quick=args.quick)
    refs = track_reference(table)
    learned = learned_residual_rows(table)

    try:
        angles = estimator_angles(refs)
    except Exception as exc:                                  # noqa: BLE001
        print(f"! estimator_angles failed: {type(exc).__name__}: {exc}")
        angles = []

    calib = load_json(os.path.join(HERE, "calib_consistency.json"))
    expressible = load_json(os.path.join(HERE, "residual_expressible.json"))
    conversions = (
        convert_px_json(calib, ["common_px", "deviation_px", "calib_disagreement_px"],
                        refs, "calib_consistency.json")
        + convert_px_json(expressible, ["total_px", "inexpressive_px", "control_rms_px",
                                        "corrected_rms_px"], refs,
                          "residual_expressible.json"))

    doc = {
        "what": "local plate scale S(theta) = dr/dtheta per camera, and px <-> urad",
        "generated_by": "scripts/analysis/plate_scale.py",
        "conventions": {
            "theta_grid_deg": [float(x) for x in THETA_GRID_DEG],
            "S_units": "pixels per radian, at the resolution the run was EVALUATED at",
            "S_rad_definition": "|d p(theta,phi) / d theta|, mean over 180 azimuths, "
                                "central difference h=1e-5 rad in float64 on the real "
                                "COLMAP projection",
            "S_tan_definition": "|d p(theta,phi) / d phi|, same; equals r(theta) for an "
                                "ideal equidistant lens",
            "mask_radius_scale": MASK_RADIUS_SCALE,
            "mask_edge_deg": MASK_EDGE_DEG,
            "mask_is_an_angle_cut": ("gray/fisheye_mask.py cuts at theta < 0.95*90deg, not "
                                     "at 0.95 of the pixel radius"),
            "px_to_urad": "urad = px / S_rad(theta) * 1e6",
            "pinhole_edge": "no mask exists, so the evaluated edge is the frame corner",
        },
        "camera_order": order,
        "cameras": table,
        "tracks": refs,
        "dose_response_x_axis_urad": dose_response_x_axis(refs, angles, learned),
        "estimator_conversions_px_to_urad": conversions,
        "estimator_angles_rederived": angles,
        "learned_residual": learned,
        "checks": {"bicycle_pinhole": bicycle_check(table, learned),
                   "numerical_derivative": numerical_derivative_check(),
                   "protocol_cross_checks": {
                       "note": "the plan's published constants, recomputed here",
                       "myscenes_valid_fraction_expected_0.44": float(np.median(
                           [e.get("mask_valid_fraction", float("nan"))
                            for e in table.values() if e["track"] == "myscenes_rttpf"])),
                       "workshop_fujinon_valid_fraction_expected_0.955": table.get(
                           "workshop_fujinon_rttpf/workshop_fujinon/cam1",
                           {}).get("mask_valid_fraction"),
                       "immervision_rttpf_valid_fraction_expected_0.479": table.get(
                           "workshop_immervision_rttpf/workshop_immervision/cam1",
                           {}).get("mask_valid_fraction"),
                       "immervision_ocv_valid_fraction_expected_0.656": table.get(
                           "workshop_immervision_ocv/workshop_immervision/cam1",
                           {}).get("mask_valid_fraction"),
                       "myscenes_S90_over_fx_expected_0.56": float(np.median(
                           [e["S_at_90deg_over_fx"] for e in table.values()
                            if e["track"] == "myscenes_rttpf"])),
                       "immervision_peak_over_centre_expected_1.65": table.get(
                           "workshop_immervision_rttpf/workshop_immervision/cam1",
                           {}).get("S_peak_over_S0"),
                   }},
        "known_issues": KNOWN_ISSUES,
    }
    with open(args.out, "w") as handle:
        json.dump(doc, handle, indent=1)

    # ---- human-readable summary -------------------------------------------------------
    print(f"{'track':28s} {'model':28s} {'edge':>7s} {'S0':>9s} {'S_edge':>9s} "
          f"{'S/fx':>6s} {'urad/px_edge':>12s} {'urad/px_area':>12s}")
    for track, ref in sorted(refs.items()):
        print(f"{track:28s} {ref['model']:28s} {ref['theta_edge_deg']:6.2f}d "
              f"{ref['S_paraxial_px_per_rad']:9.2f} {ref['S_edge_px_per_rad']:9.2f} "
              f"{ref['S_edge_over_fx']:6.3f} {ref['urad_per_px_at_edge']:12.2f} "
              f"{ref['urad_per_px_area_weighted']:12.2f}")
    print("\ndose-response x axis (microradians at the evaluated-disk edge):")
    for row in doc["dose_response_x_axis_urad"]:
        shared = (", ".join(f"lens{s['uid']}={s['urad']:.0f}"
                            for s in row["E_shared_urad_at_edge"])
                  if row["E_shared_urad_at_edge"] else "n/a")
        learned_mean = row["E_learned_urad_at_edge_mean"]
        print(f"  {row['track']:28s} E_shared[{shared}]  "
              f"E_learned={'n/a' if learned_mean is None else f'{learned_mean:.0f}'} urad "
              f"(n={row['E_learned_urad_at_edge_n']})")
    print("\nE_shared re-derived in angle (published px reproduced as a check):")
    for row in angles:
        print(f"  {row['name'][:46]:46s} pub={row['published_common_px']:.4f}px "
              f"@{row['published_argmax_theta_deg']:.1f}deg -> "
              f"{row['common_urad_at_published_argmax']:8.1f} urad ; "
              f"edge {row['common_urad_at_edge']:8.1f} urad = "
              f"{row['common_px_at_edge_true_scale']:.4f} px (true S) ; "
              f"S_pub/S_true={row['plate_scale_ratio_published_over_true_at_edge']:.3f}")
    print("\nestimators converted at each track's evaluated-disk edge:")
    for row in conversions:
        bits = " ".join(f"{k}={row[k]:.4g}" for k in row
                        if k.endswith("_urad") or k.endswith("_px"))
        print(f"  [{row['source'][:22]:22s}] {row['name'][:46]:46s} {bits}")
    print(f"\nbicycle check: {json.dumps(doc['checks']['bicycle_pinhole'], indent=1)}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
