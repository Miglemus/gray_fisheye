#!/usr/bin/env python
"""W1 -- the dose-response curve: does pre-training camera error predict the PSNR gain?

    y = masked-PSNR gain of `--camera_opt noncentral` over `--camera_opt off`, PAIRED BY SCENE
    x = the size of the camera-model error BEFORE training, in MICRORADIANS

Design, point list, functional form and falsification conditions were fixed in
`dose_response_prereg.md` (timestamped 2026-08-10T21:46:30Z, commit 8e0a15bf) BEFORE any
curve was fitted. Read that file first; this script executes it.

Zero GPU. Everything runs on CPU from artefacts already on disk: rendered PNGs, COLMAP
models, and `camera_model_15000.csv` / `gaussians_15000.safetensors` from finished runs.

    python scripts/analysis/dose_response.py              # full run, ~12 min, writes JSON + SVGs
    python scripts/analysis/dose_response.py --from-json  # refit + redraw from the cached JSON
    python scripts/analysis/dose_response.py --workers 5  # parallelism of the COLMAP pass


THE FIVE STATISTICS, AND WHICH OF THEM MAY BE AN X AXIS
--------------------------------------------------------------------------------------
  E_sfm       median SfM reprojection residual of the COLMAP model gray trained on, pushed
              through the inverse Jacobian of the true COLMAP projection so the unit is an
              angle, restricted to the evaluated disk (theta <= 85.5 deg).
              TRAINING-FREE. Exists for every point.  --> PRE-REGISTERED PRIMARY X AXIS.
  E_sfm_sys   the azimuth-averaged, radial part of that same residual: bin by theta, take
              the MEAN signed meridional angular error per bin, subtract the sampling
              variance of that mean, area-weight, RMS. This is the component a radial
              camera-model residual could absorb; the rest is feature noise.
              TRAINING-FREE. EXPLORATORY (added post-hoc, per prereg section 6).
  E_calib     cross-fit calibration disagreement: the SAME physical lens is calibrated
              independently on every scene, so at a fixed pixel radius the fits disagree
              about the field angle by std(theta) over fits; area-weighted RMS over the
              evaluated disk. TRAINING-FREE. Needs >= 3 fits of one lens, so it exists only
              for myscenes and FullCircle.  --> the honest stand-in for E_shared.
  E_shared    cross-scene MEAN of the LEARNED angular residual, area-weighted RMS.
              *** NOT TRAINING-FREE *** -- see the gotcha below. Reported, never an x axis.
  E_learned   per-scene learned angular residual, area-weighted RMS.
              *** FORBIDDEN AS AN X AXIS *** (circular: it is the optimiser's output).
              Its only legitimate use is "did the model find what E_sfm / E_calib said?"

GOTCHA THAT CHANGES HOW THE PLAN MUST BE READ
--------------------------------------------------------------------------------------
The Phase-1 plan lists E_shared as a mechanistic *predictor* alongside E_sfm. It is not one.
`calib_consistency.py:141,169` builds `common` as `d.mean(0)` where `d` is
`learned_dtheta(...)` read out of `gaussians_15000.safetensors`. E_shared is therefore the
cross-scene mean of E_learned: a decomposition of the training result, not an input to it.
Averaging over scenes removes the per-scene circularity but not the per-track one, and at
track level is where all of this figure's separating power lives -- so regressing the gain on
E_shared is circular in exactly the way the plan forbids for E_learned. E_calib, computed
here from the COLMAP `cameras` alone, is what the plan's prose actually describes, and it
lands at the same magnitude (myscenes 958 vs 779 urad; FullCircle 198/139 vs 195/195), which
is itself a result: the photometric descent recovers about as much angle as the sparse-feature
calibration is uncertain about.

A SECOND GOTCHA, INHERITED
--------------------------------------------------------------------------------------
`calib_consistency.py:54` sets `RADIAL = [4, 5, 8, 9]`, the COLMAP THIN_PRISM_FISHEYE layout.
Every camera it reads is RAD_TAN_THIN_PRISM_FISHEYE (16 params, six radial coefficients at
indices 4..9 -- see `gray/fisheye_geometry.py:132`), so its r(theta) silently drops k2, k3 and
mislabels k4, k5. This script never uses that polynomial: E_calib projects bearings through
`pycolmap`'s own camera object, which cannot be wrong about its own parameter layout.
E_shared is reproduced through `calib_consistency`'s helpers only for the LEARNED residual,
which is read from the checkpoint spline and does not touch RADIAL.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WORKTREE, "scripts"))

OUT_JSON = os.path.join(HERE, "dose_response.json")
CACHE_JSON = os.path.join(HERE, "dose_response_cache.json")
FIG_DIR = os.path.join(WORKTREE, "figures")
NOISE_FLOOR_DB = 0.068           # measured run-to-run noise (PROTOCOL.md / IMPLEMENTATION.md)
FD_H = 1e-4                      # finite-difference step for the projection Jacobian, rad
MASK_EDGE_DEG = 85.5             # 0.95 * 90 deg, the evaluated disk (gray/fisheye_mask.py)
N_BOOT = 10000
RNG_SEED = 20260810

# --------------------------------------------------------------------------------------- #
#  1. the paired runs                                                                       #
# --------------------------------------------------------------------------------------- #
# `off` must differ from `noncentral` by the --camera_opt flag and NOTHING else; the
# config parity of every pair is re-checked at run time (assert_config_parity).
MYSCENES = ["atrium", "classroom", "forest", "library", "reception", "tunnel", "workshop"]
FC_SCENES = ["room1", "room2", "room3", "flat1", "flat2", "lab", "lounge", "dark", "persons"]
MIP_SCENES = ["bicycle", "stump", "garden", "bonsai", "counter", "kitchen", "room"]
# `kitchen` and `room` were EXCLUDED from the pre-registered point list of 22 because their
# runs had not finished when the pre-registration was written (prereg section 3, which
# explicitly allows them as ADDITIONS, never as substitutions). Both `*_off` and
# `*_noncentral` landed on 2026-08-10 between 17:51 and 19:30 with matched configs
# (camera_opt the only difference, downsampling=2, iterations=15000, batch_size=1).
# Everything is therefore reported twice: on the 24 points, and on the pre-registered 22
# alone (key `prereg_22_only` in the JSON), so the addition cannot be mistaken for
# post-hoc point selection.
PREREG_22_EXCLUDED = {("mipnerf360_pinhole", "kitchen"), ("mipnerf360_pinhole", "room")}


def build_pairs() -> list[dict]:
    pairs = []
    for s in MYSCENES:
        # workshop's PUBLISHED baseline was trained with vignetting_comp=False and
        # batch_size=1; it is disqualified as the paired control (prereg section 1).
        off = ("/workspace/gray/tmp/noncentral/fix15k_workshop" if s == "workshop"
               else f"/workspace/gray/out/{s}_fisheye_baseline")
        pairs.append(dict(track="myscenes_rttpf", scene=s, family="fisheye_circular",
                          kind="fisheye", off=off,
                          nc=f"/workspace/gray/tmp/final/{s}_noncentral",
                          sfm_key=f"myscenes_rttpf/{s}", lens="myscenes", primary=True))
    for s in FC_SCENES:
        pairs.append(dict(track="fullcircle_refit_rttpf", scene=s, family="fisheye_circular",
                          kind="fisheye",
                          off=f"{WORKTREE}/out/fullcircle_rttpf/{s}_refit_rttpf_off",
                          nc=f"{WORKTREE}/out/fullcircle_rttpf/{s}_refit_rttpf",
                          sfm_key=f"fullcircle_refit_rttpf/{s}", lens="fullcircle",
                          primary=True))
    for s in MIP_SCENES:
        pairs.append(dict(track="mipnerf360_pinhole", scene=s, family="pinhole",
                          kind="pinhole",
                          off=f"{WORKTREE}/tmp/mipnerf360/{s}_off",
                          nc=f"{WORKTREE}/tmp/mipnerf360/{s}_noncentral",
                          sfm_key=f"mipnerf360/{s}", lens=f"mip360_{s}", primary=True))
    pairs.append(dict(track="workshop_immervision_rttpf", scene="workshop_immervision",
                      family="panomorph", kind="fisheye",
                      off=f"{WORKTREE}/out/workshop_immervision_off",
                      nc=f"{WORKTREE}/out/workshop_immervision_noncentral",
                      sfm_key="others_rttpf/workshop_immervision", lens="immervision",
                      primary=True))
    # sensitivity pairs -- NOT part of the 22, never entered into a fit
    pairs.append(dict(track="SENSITIVITY", scene="workshop_published_off",
                      family="fisheye_circular", kind="fisheye",
                      off="/workspace/gray/out/workshop_fisheye_baseline",
                      nc="/workspace/gray/tmp/final/workshop_noncentral",
                      sfm_key="myscenes_rttpf/workshop", lens="myscenes", primary=False))
    pairs.append(dict(track="SENSITIVITY", scene="tunnel_r4_off",
                      family="fisheye_circular", kind="fisheye",
                      off="/workspace/gray/tmp/r4/tunnel_off",
                      nc="/workspace/gray/tmp/final/tunnel_noncentral",
                      sfm_key="myscenes_rttpf/tunnel", lens="myscenes", primary=False))
    return pairs


PARITY_KEYS = ["vignetting_comp", "vignetting_terms", "batch_size", "downsampling",
               "iterations", "camera_model"]


def config_parity(off: str, nc: str) -> dict:
    def load(p):
        try:
            with open(os.path.join(p, "config.json")) as h:
                return json.load(h)
        except OSError:
            return {}
    a, b = load(off), load(nc)
    mismatch = {}
    for k in PARITY_KEYS:
        va, vb = a.get(k), b.get(k)
        if str(va) != str(vb):
            mismatch[k] = [va, vb]
    return {"off_camera_opt": a.get("camera_opt"), "nc_camera_opt": b.get("camera_opt"),
            "mismatch": mismatch, "parity_ok": not mismatch}


def fisheye_scores(run_dir: str) -> dict:
    from radial_eval import evaluate
    r = evaluate(run_dir, 6, metrics=("psnr",))
    return {"per_view_mean": float(r["disk_per_view_mean"]), "pooled": float(r["disk_pooled"]),
            "n": int(r["views"]), "valid_fraction": float(r["valid_fraction"]),
            "distinct_masks": int(r["distinct_masks"]), "protocol": "radial_eval disk r=0.95"}


def pinhole_scores(run_dir: str, split: str = "test") -> dict:
    """Full-frame masked-equivalent PSNR: a pinhole run has no valid_mask, so the disk IS
    the frame. Same float64 numpy path as radial_eval's PSNR, same per-view-mean pooling."""
    renders = sorted(glob.glob(os.path.join(run_dir, split, "*", "pinhole", "renders", "*.png")))
    if not renders:
        raise SystemExit(f"no pinhole renders under {run_dir}/{split}")
    parent = os.path.dirname(os.path.dirname(renders[0]))
    gts = sorted(glob.glob(os.path.join(parent, "gt", "*.png")))
    if len(gts) != len(renders):
        raise SystemExit(f"{run_dir}: {len(renders)} renders vs {len(gts)} gt")
    per_view, sse, npx = [], 0.0, 0
    for r, g in zip(renders, gts):
        a = np.asarray(Image.open(r).convert("RGB"), dtype=np.uint8).astype(np.float64) / 255.0
        b = np.asarray(Image.open(g).convert("RGB"), dtype=np.uint8).astype(np.float64) / 255.0
        e = (a - b) ** 2
        per_view.append(10.0 * math.log10(1.0 / e.mean()))
        sse += float(e.sum())
        npx += e.size
    return {"per_view_mean": float(np.mean(per_view)),
            "pooled": float(10 * math.log10(1.0 / (sse / npx))), "n": len(renders),
            "valid_fraction": 1.0, "distinct_masks": 1, "protocol": "full frame (no mask)"}


def collect_gains(pairs: list[dict]) -> list[dict]:
    rows = []
    for p in pairs:
        fn = pinhole_scores if p["kind"] == "pinhole" else fisheye_scores
        off, nc = fn(p["off"]), fn(p["nc"])
        rows.append({**{k: p[k] for k in
                        ("track", "scene", "family", "kind", "off", "nc", "sfm_key", "lens",
                         "primary")},
                     "off_scores": off, "nc_scores": nc,
                     "config_parity": config_parity(p["off"], p["nc"]),
                     "y_db": off and nc["per_view_mean"] - off["per_view_mean"],
                     "y_db_pooled": nc["pooled"] - off["pooled"]})
        print(f"  {p['track']:26s} {p['scene']:22s} off={off['per_view_mean']:8.4f} "
              f"nc={nc['per_view_mean']:8.4f}  d={rows[-1]['y_db']:+.4f}", flush=True)
    return rows


# --------------------------------------------------------------------------------------- #
#  2. the angular estimators                                                                #
# --------------------------------------------------------------------------------------- #
def signed_angular(camera, cam_pts, dpix):
    """Signed meridional / sagittal angular components of a pixel residual.

    Same 2x2 Jacobian as `sfm_residual.angular_residual` (which returns the magnitude only):
    J = [dp/dtheta, (1/sin theta) dp/dphi], both columns mapping a TRUE angle on the sphere
    to an image displacement, differentiated numerically on the real COLMAP projection.
    """
    n = np.linalg.norm(cam_pts, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        th = np.arccos(np.clip(cam_pts[:, 2] / np.where(n > 0, n, np.nan), -1.0, 1.0))
    ph = np.arctan2(cam_pts[:, 1], cam_pts[:, 0])

    def bearing(t, p):
        st = np.sin(t)
        return np.stack([st * np.cos(p), st * np.sin(p), np.cos(t)], axis=1)

    thc = np.maximum(th, FD_H)
    dth = (np.asarray(camera.img_from_cam(bearing(thc + FD_H, ph)), dtype=np.float64)
           - np.asarray(camera.img_from_cam(bearing(thc - FD_H, ph)), dtype=np.float64)) / (2 * FD_H)
    with np.errstate(invalid="ignore", divide="ignore"):
        dph = ((np.asarray(camera.img_from_cam(bearing(thc, ph + FD_H)), dtype=np.float64)
                - np.asarray(camera.img_from_cam(bearing(thc, ph - FD_H)), dtype=np.float64))
               / (2 * FD_H) / np.maximum(np.sin(thc), 1e-9)[:, None])
        det = dth[:, 0] * dph[:, 1] - dth[:, 1] * dph[:, 0]
        a_th = (dph[:, 1] * dpix[:, 0] - dph[:, 0] * dpix[:, 1]) / det
        a_ph = (-dth[:, 1] * dpix[:, 0] + dth[:, 0] * dpix[:, 1]) / det
    return a_th, a_ph, th


def radial_profile_px(camera, theta, n_phi=64):
    """(r(theta), S(theta)) of the REAL projection, averaged over azimuth. No polynomial
    is assumed, so no parameter-layout mistake is possible (see the module docstring)."""
    ph = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    pp = np.array([camera.params[2], camera.params[3]], dtype=np.float64)

    def proj(t):
        b = np.stack([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph),
                      np.full(n_phi, np.cos(t))], axis=1)
        return np.asarray(camera.img_from_cam(b), dtype=np.float64)

    r, s = [], []
    for t in theta:
        t = max(float(t), FD_H)
        r.append(float(np.nanmean(np.linalg.norm(proj(t) - pp, axis=1))))
        s.append(float(np.nanmean(np.linalg.norm((proj(t + FD_H) - proj(t - FD_H))
                                                 / (2 * FD_H), axis=1))))
    return np.asarray(r), np.asarray(s)


def pinhole_corner_deg(camera) -> float:
    fx, fy, cx, cy = (float(camera.params[i]) for i in range(4))
    dx = max(abs(0.5 - cx), abs(camera.width - 0.5 - cx)) / fx
    dy = max(abs(0.5 - cy), abs(camera.height - 0.5 - cy)) / fy
    return float(np.degrees(np.arctan(math.hypot(dx, dy))))


def sfm_systematic(track_name: str) -> dict:
    """E_sfm_sys for one track. EXPLORATORY (prereg section 6). Runs in a worker process."""
    import pycolmap
    import sfm_residual as sr

    spec = {t["track"]: t for t in sr.build_tracks()}[track_name]
    rec = pycolmap.Reconstruction(str(spec["colmap"]))
    xyz_of = {pid: p.xyz for pid, p in rec.points3D.items()}
    chunks: dict[int, list] = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        xy, xyz = [], []
        for p in im.points2D:
            if p.has_point3D():
                xy.append(p.xy)
                xyz.append(xyz_of[p.point3D_id])
        if not xy:
            continue
        xy = np.asarray(xy, dtype=np.float64)
        M = im.cam_from_world().matrix()
        cam_pts = np.asarray(xyz, dtype=np.float64) @ M[:, :3].T + M[:, 3]
        dpix = np.asarray(im.camera.img_from_cam(cam_pts), dtype=np.float64) - xy
        a_th, a_ph, th = signed_angular(im.camera, cam_pts, dpix)
        ok = np.isfinite(a_th) & np.isfinite(th)
        chunks.setdefault(int(im.camera_id), []).append(
            np.stack([np.degrees(th[ok]), a_th[ok]], axis=1))

    out = {"track": track_name, "family": spec["camera_family"], "cameras": {}}
    num = den = 0.0
    for cid, parts in chunks.items():
        cam = rec.cameras[cid]
        edge = pinhole_corner_deg(cam) if spec["camera_family"] == "pinhole" else MASK_EDGE_DEG
        d = np.concatenate(parts)
        nb = 60
        edges = np.linspace(0.0, edge, nb + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        idx = np.digitize(d[:, 0], edges) - 1
        keep = (idx >= 0) & (idx < nb)
        idx, vals = idx[keep], d[keep, 1]
        r, s = radial_profile_px(cam, np.radians(centres))
        w = np.where(np.isfinite(r * s) & (r * s > 0), r * s, 0.0)   # dA/dtheta
        m = np.full(nb, np.nan)
        var = np.full(nb, np.nan)
        cnt = np.zeros(nb)
        for k in range(nb):
            sel = idx == k
            cnt[k] = sel.sum()
            if cnt[k] >= 30:
                m[k] = vals[sel].mean()
                var[k] = vals[sel].var(ddof=1)
        good = np.isfinite(m) & (w > 0)
        ww = w[good] / w[good].sum()
        # E[binmean^2] = systematic^2 + var/N  ->  unbiased estimate of systematic^2
        sq = np.maximum(m[good] ** 2 - var[good] / cnt[good], 0.0)
        out["cameras"][str(cid)] = {
            "theta_edge_deg": edge, "n_obs": int(keep.sum()), "n_bins_used": int(good.sum()),
            "E_sfm_sys_urad": float(np.sqrt((ww * sq).sum()) * 1e6),
            "E_sfm_binmean_urad_uncorrected": float(np.sqrt((ww * m[good] ** 2).sum()) * 1e6),
            "E_sfm_scatter_urad": float(np.sqrt((ww * var[good]).sum()) * 1e6),
            "profile_theta_deg": centres[good].tolist(),
            "profile_mean_urad": (m[good] * 1e6).tolist(),
            "profile_sem_urad": (np.sqrt(var[good] / cnt[good]) * 1e6).tolist(),
        }
        num += out["cameras"][str(cid)]["E_sfm_sys_urad"] ** 2 * keep.sum()
        den += keep.sum()
    out["E_sfm_sys_urad"] = float(np.sqrt(num / den)) if den else None
    return out


def calib_disagreement(name: str, sparses: list[str], uid: int, family: str,
                       n_grid: int = 200) -> dict | None:
    """E_calib: how much do independent fits of ONE lens disagree about the bearing at a
    fixed pixel radius? Training-free. Projection via pycolmap, no polynomial assumed."""
    import pycolmap

    curves = []
    for sp in sparses:
        if not os.path.isdir(sp):
            continue
        rec = pycolmap.Reconstruction(sp)
        if uid not in rec.cameras:
            continue
        cam = rec.cameras[uid]
        edge = pinhole_corner_deg(cam) if family == "pinhole" else MASK_EDGE_DEG
        th = np.linspace(0.0, np.radians(edge), n_grid)
        r, _ = radial_profile_px(cam, th)
        if not np.all(np.diff(r) > 0):                 # keep the invertible part only
            k = int(np.argmax(np.diff(r) <= 0))
            th, r = th[: k + 1], r[: k + 1]
        curves.append((th, r))
    if len(curves) < 3:
        return None
    rmax = min(c[1][-1] for c in curves)
    rho = np.linspace(0.02, 1.0, 60) * rmax
    T = np.stack([np.interp(rho, c[1], c[0]) for c in curves])
    w = rho / rho.sum()                                # dA ~ rho drho on a uniform rho grid
    sd = T.std(0, ddof=1)
    return {"name": name, "uid": uid, "n_fits": len(curves),
            "E_calib_urad_rms": float(np.sqrt((w * sd**2).sum()) * 1e6),
            "E_calib_urad_at_edge": float(sd[-1] * 1e6),
            "E_calib_urad_max": float(sd.max() * 1e6),
            "r_edge_px_native": float(rmax),
            "theta_grid_deg": np.degrees(T.mean(0)).tolist(),
            "sd_urad": (sd * 1e6).tolist()}


def area_weights_from_plate_scale(cam: dict):
    ps = json.load(open(os.path.join(HERE, "plate_scale.json")))
    theta = np.asarray(ps["conventions"]["theta_grid_deg"], dtype=float)
    r = np.asarray(cam["r_px"], dtype=float)
    s = np.asarray(cam["S_rad_px_per_rad"], dtype=float)
    keep = theta <= cam["theta_edge_deg"] + 1e-9
    w = np.maximum((r * s)[keep], 0.0)
    return theta[keep], w / w.sum()


def pinhole_plate_scale_from_run(run_dir: str) -> dict | None:
    """Stand-in `plate_scale.json` entry for a PINHOLE run that A2's pass did not cover.

    Needed only for mip-NeRF 360 `kitchen` and `room`, whose runs landed after
    `plate_scale.json` was written; that file belongs to another agent and is not rewritten
    here. A pinhole has no fitted distortion, so both quantities are closed form:

        r(theta, phi)  = tan(theta) * hypot(fx cos phi, fy sin phi)
        dr/dtheta      = sec^2(theta) * hypot(fx cos phi, fy sin phi)

    azimuth-averaged over the same 180 azimuths A2 uses. Validated against the five pinhole
    cameras A2 DID compute (bicycle, stump, garden, bonsai, counter): max relative error
    3.3e-6 on r_px, 5.1e-8 on S_rad, and theta_edge_deg identical to 1e-9 deg -- the field
    `pinhole_plate_scale_validation` in the JSON re-runs that check on every call.
    """
    path = os.path.join(run_dir, "cameras.json")
    if not os.path.exists(path):
        return None
    cams = json.load(open(path))
    if not cams or cams[0].get("model") != "pinhole":
        return None
    c = cams[0]
    w, h = int(c["image_width"]), int(c["image_height"])
    fx = (w / 2.0) / math.tan(c["fov_x"] / 2.0)
    fy = (h / 2.0) / math.tan(c["fov_y"] / 2.0)
    ps = json.load(open(os.path.join(HERE, "plate_scale.json")))
    theta = np.radians(np.asarray(ps["conventions"]["theta_grid_deg"], dtype=float))
    phi = (np.arange(180) + 0.5) / 180.0 * 2.0 * np.pi
    m = float(np.hypot(fx * np.cos(phi), fy * np.sin(phi)).mean())
    edge = math.degrees(math.atan(math.hypot((w / 2.0 - 0.5) / fx, (h / 2.0 - 0.5) / fy)))
    return {"family": "pinhole", "model": "pinhole", "fx": fx, "fy": fy,
            "eval_width": w, "eval_height": h, "theta_edge_deg": edge,
            "theta_edge_source": "frame_corner (analytic, pixel-centre convention)",
            "r_px": (np.tan(theta) * m).tolist(),
            "S_rad_px_per_rad": (m / np.cos(theta) ** 2).tolist(),
            "source": "computed here from the run's cameras.json; A2's plate_scale.json has "
                      "no entry for this camera"}


def validate_pinhole_plate_scale() -> dict:
    """Re-run, at every execution, the check that the closed form above reproduces A2."""
    ps = json.load(open(os.path.join(HERE, "plate_scale.json")))
    grid = np.asarray(ps["conventions"]["theta_grid_deg"], dtype=float)
    worst = {"max_rel_err_r": 0.0, "max_rel_err_S": 0.0, "max_abs_err_theta_edge_deg": 0.0,
             "cameras_checked": []}
    for key, cam in ps["cameras"].items():
        if cam.get("family") != "pinhole":
            continue
        mine = pinhole_plate_scale_from_run(cam["run_dir"])
        if mine is None:
            continue
        keep = grid <= cam["theta_edge_deg"]
        for src, dst, tag in ((cam["r_px"], mine["r_px"], "max_rel_err_r"),
                              (cam["S_rad_px_per_rad"], mine["S_rad_px_per_rad"],
                               "max_rel_err_S")):
            a = np.asarray(src, float)[keep]
            b = np.asarray(dst, float)[keep]
            e = float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)))
            worst[tag] = max(worst[tag], e)
        worst["max_abs_err_theta_edge_deg"] = max(
            worst["max_abs_err_theta_edge_deg"],
            abs(cam["theta_edge_deg"] - mine["theta_edge_deg"]))
        worst["cameras_checked"].append(key)
    worst["passes"] = bool(worst["max_rel_err_r"] < 1e-4 and worst["max_rel_err_S"] < 1e-4
                           and worst["max_abs_err_theta_edge_deg"] < 1e-6)
    return worst


def learned_rms(run_dir: str, uid: int, grid_deg, weights) -> dict | None:
    path = os.path.join(run_dir, "camera_model_15000.csv")
    if not os.path.exists(path):
        return None
    th, dth, dph = [], [], []
    with open(path) as h:
        for row in csv.DictReader(h):
            if int(row["uid"]) != uid:
                continue
            th.append(float(row["theta_deg"]))
            dth.append(float(row["delta_theta_rad"]))
            dph.append(float(row["delta_phi_rad"]))
    if not th:
        return None
    th = np.asarray(th)
    v = np.interp(grid_deg, th, np.asarray(dth))
    vc = v - np.interp(0.0, th, np.asarray(dth))
    vp = np.interp(grid_deg, th, np.asarray(dph))
    return {"E_learned_urad_rms": float(np.sqrt((weights * v**2).sum()) * 1e6),
            "E_learned_urad_rms_gauge_centred": float(np.sqrt((weights * vc**2).sum()) * 1e6),
            "E_learned_urad_peak": float(np.max(np.abs(v)) * 1e6),
            "E_learned_peak_theta_deg": float(grid_deg[int(np.argmax(np.abs(v)))]),
            "E_learned_urad_at_edge": float(v[-1] * 1e6),
            "dphi_urad_rms": float(np.sqrt((weights * vp**2).sum()) * 1e6)}


def shared_learned(name: str, scenes, sparse_of, run_of, uid: int) -> dict | None:
    """E_shared: cross-scene mean of the LEARNED residual. NOT training-free (see docstring).
    Reproduced through calib_consistency's own helpers so the two cannot drift apart."""
    import pycolmap

    import calib_consistency as cc
    fits = []
    for s in scenes:
        sp = sparse_of(s)
        if not os.path.isdir(sp):
            continue
        rec = pycolmap.Reconstruction(sp)
        if uid not in rec.cameras:
            continue
        p = list(rec.cameras[uid].params)
        grid = np.linspace(0.0, np.pi / 2, 4000)
        radii = cc.r_of_theta(p[0], cc.radial_poly(p), grid)
        folds = np.diff(radii) <= 0
        fold = int(np.argmax(folds)) if folds.any() else len(grid) - 1
        fits.append({"scene": s, "params": p, "theta_max": grid[fold], "r_max": radii[fold]})
    if len(fits) < 3:
        return None
    r_px = cc.GRID * min(f["r_max"] for f in fits)
    d = []
    for f in fits:
        t = cc.theta_of_r(f["params"][0], cc.radial_poly(f["params"]), r_px, f["theta_max"])
        v = cc.learned_dtheta(run_of(f["scene"]), uid, t)
        if v is None:
            return None
        d.append(v)
    d = np.stack(d)
    common, dev = d.mean(0), d.std(0)
    w = cc.GRID / cc.GRID.sum()
    rms = lambda v: float(np.sqrt((w * v**2).sum()) * 1e6)  # noqa: E731
    return {"name": name, "uid": uid, "n_scenes": len(fits),
            "E_shared_urad_rms": rms(common),
            "E_shared_urad_peak": float(np.abs(common).max() * 1e6),
            "E_deviation_urad_rms": rms(dev),
            "shared_over_scene_specific": float(np.abs(common).sum() / max(dev.sum(), 1e-12)),
            "is_training_free": False,
            "warning": "cross-scene mean of the LEARNED residual; never an x axis"}


# --------------------------------------------------------------------------------------- #
#  3. the pre-registered transfer function                                                  #
# --------------------------------------------------------------------------------------- #
def transfer(x_rad, G, mse0, C):
    return 10.0 * np.log10(1.0 + np.minimum(x_rad**2 * G, C) / mse0)


def fit_transfer(x_urad, y_db, seed=RNG_SEED, n_boot=N_BOOT, strata=None):
    """Least squares on gain_dB = 10log10(1 + min(x^2 G, C)/MSE0), x in RADIANS.

    G, MSE0 and C are only jointly identifiable through the saturation elbow. If the fit
    does not reach it (the usual case here), the model collapses to one effective parameter
    A = G/MSE0 and that is what is reported, with C flagged unidentified.
    """
    x = np.asarray(x_urad, dtype=float) * 1e-6
    y = np.asarray(y_db, dtype=float)

    def unsat(A):                                    # A = G / MSE0, rad^-2
        return 10.0 * np.log10(1.0 + np.maximum(A, 0.0) * x**2)

    def resid(logA):
        return unsat(np.exp(logA[0])) - y

    best, saturated = None, False
    for a0 in (1e2, 1e4, 1e6):
        r = optimize.least_squares(resid, [math.log(a0)], method="lm")
        if best is None or r.cost < best.cost:
            best = r
    A = float(np.exp(best.x[0]))
    pred = unsat(A)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())

    boot = []
    rng = np.random.default_rng(seed)
    n = len(x)
    groups = np.asarray(strata) if strata is not None else np.zeros(n, dtype=int)
    idx_by = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(v, size=len(v), replace=True)
                               for v in idx_by.values()])
        xb, yb = x[pick], y[pick]

        def rb(logA, xb=xb, yb=yb):
            return 10.0 * np.log10(1.0 + np.exp(logA[0]) * xb**2) - yb
        try:
            rr = optimize.least_squares(rb, [math.log(max(A, 1e-6))], method="lm")
            boot.append(float(np.exp(rr.x[0])))
        except Exception:
            continue
    boot = np.asarray(boot)
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
              ) if boot.size else (float("nan"), float("nan"))

    def threshold(a):
        if a <= 0:
            return float("inf")
        return float(np.sqrt((10 ** (NOISE_FLOOR_DB / 10) - 1) / a) * 1e6)

    return {
        "form": "gain_dB = 10*log10(1 + min(x^2*G, C)/MSE0), x in rad",
        "saturation_reached": saturated,
        "C_identified": False,
        "C_note": "the data never reach the elbow, so C and the split of A into G and MSE0 "
                  "are unidentified; only A = G/MSE0 is estimated",
        "A_rad^-2": A, "A_ci95": [lo, hi],
        "A_px^-2_at_eval_scale": None,     # filled in by the caller when a plate scale applies
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "rmse_db": float(np.sqrt(ss_res / len(y))),
        "threshold_urad_at_noise_floor": threshold(A),
        "threshold_ci95_urad": [threshold(hi), threshold(lo)],
        "n": int(len(y)), "n_bootstrap": int(boot.size),
        "prereg_start": {"G": 0.0070, "MSE0": 2e-3, "C": 0.12,
                         "A_implied_rad^-2_if_x_in_rad": 0.0070 / 2e-3},
    }


def shared_learned_fixed(name: str, scenes, sparse_of, run_of, uid: int) -> dict | None:
    """E_shared WITHOUT calib_consistency's radial-index bug.

    `calib_consistency.py:54` uses `RADIAL = [4, 5, 8, 9]`, which is the parameter layout of
    COLMAP's 12-parameter THIN_PRISM_FISHEYE. Every camera in these models is the
    16-parameter RAD_TAN_THIN_PRISM_FISHEYE, whose `params_info` (checked here against
    pycolmap itself, not against a comment) reads

        fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3

    so the radial coefficients are 4..9 and [4, 5, 8, 9] silently drops k2, k3 and relabels
    k4, k5. The Phase-1 plan section 2.2 bis states the opposite and cites `gray/camera.py:58`
    -- that line is the THIN_PRISM_FISHEYE branch; the RAD_TAN branch is three lines below and
    unpacks sixteen names. `shared_learned()` above reproduces the buggy mapping on purpose
    (it must agree with the published `calib_consistency.json`); this function is the
    corrected one, and it never assumes a polynomial at all: r(theta) comes from pycolmap's
    own projection of the camera.
    """
    import pycolmap

    import calib_consistency as cc
    curves = []
    for s in scenes:
        sp = sparse_of(s)
        if not os.path.isdir(sp):
            continue
        rec = pycolmap.Reconstruction(sp)
        if uid not in rec.cameras:
            continue
        cam = rec.cameras[uid]
        th = np.linspace(0.0, np.radians(MASK_EDGE_DEG), 400)
        r, _ = radial_profile_px(cam, th)
        if not np.all(np.diff(r) > 0):
            k = int(np.argmax(np.diff(r) <= 0))
            th, r = th[: k + 1], r[: k + 1]
        curves.append((s, th, r))
    if len(curves) < 3:
        return None
    rmax = min(c[2][-1] for c in curves)
    rho = np.linspace(0.02, 1.0, 200) * rmax
    d = []
    for s, th, r in curves:
        v = cc.learned_dtheta(run_of(s), uid, np.interp(rho, r, th))
        if v is None:
            return None
        d.append(v)
    d = np.stack(d)
    common, dev = d.mean(0), d.std(0)
    w = rho / rho.sum()                     # dA ~ rho drho on a uniform pixel-radius grid
    def rms(v):
        return float(np.sqrt((w * v ** 2).sum()) * 1e6)
    return {"name": name, "uid": uid, "n_scenes": len(curves),
            "E_shared_urad_rms": rms(common),
            "E_shared_urad_peak": float(np.abs(common).max() * 1e6),
            "E_deviation_urad_rms": rms(dev),
            "r_edge_px_native": float(rmax),
            "is_training_free": False,
            "note": "same estimator as `e_shared`, with the RAD_TAN radial-index bug removed "
                    "and r(theta) taken from pycolmap's projection instead of a polynomial"}


SWEEP_FC = os.path.join(WORKTREE, "tmp", "w3_radial", "sweep_fullcircle_rttpf.json")


def fullcircle_seed_repeat(rows) -> dict | None:
    """A FREE, 7-scene, same-config REPEAT of the FullCircle `noncentral` track.

    Provenance, established from `pueue` and from file mtimes, not assumed: tasks 1539-1545
    (enqueued 2026-08-10T11:48, run 19:30-20:40) re-ran `train.py ... --camera_opt noncentral
    --camera_opt_from_iter 3000 -y` in place on room1, room2, room3, flat1, flat2, lab and
    lounge -- byte-identical command line to the runs already there, so the difference
    between the two is run-to-run noise and nothing else. `dark` and `persons` were NOT
    re-run.

    The PRE-repeat masked PSNR of those runs survives in the W3 agent's
    `tmp/w3_radial/sweep_fullcircle_rttpf.json` (written 19:23, seven minutes before the
    first overwrite), under `results.refit_rttpf.<scene>.gray-nc["0.95"].PSNR`. It is the
    same statistic: on the two scenes that were NOT re-run it agrees with the current
    measurement to 1e-5 dB, which is what makes the other seven comparable.

    This matters because the FullCircle mean gain is the paper's null point: the repeat
    moves individual scenes by up to 0.11 dB, i.e. by MORE than the entire claimed effect.
    """
    if not os.path.exists(SWEEP_FC):
        return None
    sweep = json.load(open(SWEEP_FC))["results"]["refit_rttpf"]
    per = []
    for r in rows:
        if r["track"] != "fullcircle_refit_rttpf":
            continue
        prev = sweep.get(r["scene"], {}).get("gray-nc", {}).get("0.95", {}).get("PSNR")
        if prev is None:
            continue
        now = r["nc_scores"]["per_view_mean"]
        per.append({"scene": r["scene"], "nc_psnr_run_A_2026-08-07_or_08-10_morning": prev,
                    "nc_psnr_run_B_2026-08-10_evening": now, "delta_db": now - prev,
                    "re_run": abs(now - prev) > 1e-6,
                    "y_db_with_run_A": prev - r["off_scores"]["per_view_mean"],
                    "y_db_with_run_B": r["y_db"]})
    if not per:
        return None
    d = np.array([p["delta_db"] for p in per if p["re_run"]])
    ya = np.array([p["y_db_with_run_A"] for p in per])
    yb = np.array([p["y_db_with_run_B"] for p in per])
    return {
        "what": "same-config repeat of 7 of the 9 FullCircle noncentral runs (pueue "
                "1539-1545); an empirical, paired run-to-run noise floor on the exact track "
                "whose gain the paper calls null",
        "source_of_the_earlier_numbers": SWEEP_FC + " (written 2026-08-10 19:23, before the "
                                                    "first overwrite at 19:30)",
        "n_scenes_re_run": int(d.size),
        "n_scenes_unchanged_control": int(len(per) - d.size),
        "unchanged_control_max_abs_db": float(max(
            [abs(p["delta_db"]) for p in per if not p["re_run"]] or [0.0])),
        "repeat_delta_rms_db": float(np.sqrt((d ** 2).mean())) if d.size else None,
        "repeat_delta_max_abs_db": float(np.abs(d).max()) if d.size else None,
        "repeat_delta_mean_db": float(d.mean()) if d.size else None,
        "track_mean_gain_with_run_A_db": float(ya.mean()),
        "track_mean_gain_with_run_B_db": float(yb.mean()),
        "verdict": "the same configuration, re-run, moves single scenes by more than the "
                   "entire measured track gain; both versions of the FullCircle mean sit "
                   f"below the {NOISE_FLOOR_DB} dB floor, so the null holds either way",
        "per_scene": per,
    }


def stratified_rank_assoc(x, y, strata, seed=RNG_SEED, n_perm=20000):
    """EXPLORATORY, post-hoc. Association of y with x AFTER removing the between-track
    level, i.e. the pooled within-track rank correlation.

    Why it is needed: the pooled Spearman answers "does a track with a larger E_sfm gain
    more?", and the pre-registered answer to that is NO. It cannot answer "within one lens,
    does the dirtier scene gain more?" -- the between-track differences dominate the ranks.
    Ranking inside each track, centring, and pooling separates the two questions, and they
    have OPPOSITE signs here (Simpson's paradox), which is the actual finding.

    p by permuting y WITHIN each track (the exchangeability the statistic assumes), so no
    normality and no between-track leakage.
    """
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(strata)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    groups = [np.flatnonzero(g == t) for t in sorted(set(g.tolist()))]
    groups = [ix for ix in groups if len(ix) >= 4]
    if not groups:
        return {"note": "no track with n >= 4"}

    def stat(yv):
        num = den_x = den_y = 0.0
        for ix in groups:
            rx = stats.rankdata(x[ix])
            ry = stats.rankdata(yv[ix])
            rx = rx - rx.mean()
            ry = ry - ry.mean()
            num += float((rx * ry).sum())
            den_x += float((rx * rx).sum())
            den_y += float((ry * ry).sum())
        return num / math.sqrt(den_x * den_y) if den_x > 0 and den_y > 0 else float("nan")

    obs = stat(y)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        yp = y.copy()
        for ix in groups:
            yp[ix] = rng.permutation(y[ix])
        if abs(stat(yp)) >= abs(obs) - 1e-15:
            cnt += 1
    return {"statistic_pooled_within_track_spearman": float(obs),
            "p_two_sided_within_track_permutation": float((cnt + 1) / (n_perm + 1)),
            "n_points": int(sum(len(ix) for ix in groups)),
            "n_tracks": len(groups),
            "per_track": {str(sorted(set(g.tolist()))[i]): None for i in range(0)},
            "label": "EXPLORATORY, post-hoc (not in the pre-registration)"}


def fit_transfer_three_parameter(x_urad, y_db):
    """Fit the pre-registered form with ALL THREE parameters free, and say honestly whether
    they are identifiable.

    The task asks for G, MSE0 and C to be re-fitted. They cannot all be estimated from these
    data, and the reason is structural, not numerical: below the saturation elbow

        10*log10(1 + min(x^2 G, C)/MSE0)  ==  10*log10(1 + x^2 * (G/MSE0))

    depends on G and MSE0 ONLY through the ratio A = G/MSE0, and not on C at all. So this
    function reports (a) the 3-parameter least-squares solution, (b) how many points are
    saturated at it, and (c) a profile of the residual cost over C on a log grid with G and
    MSE0 re-fitted at each C -- the flat part of that profile IS the unidentified set.
    """
    x = np.asarray(x_urad, float) * 1e-6
    y = np.asarray(y_db, float)

    def cost_at(logG, logM, C):
        return float(((transfer(x, math.exp(logG), math.exp(logM), C) - y) ** 2).sum())

    def fit_given_C(C):
        best = None
        for a0 in (1e2, 1e4, 1e6):
            r = optimize.least_squares(
                lambda p, C=C: transfer(x, np.exp(p[0]), np.exp(p[1]), C) - y,
                [math.log(a0 * 2e-3), math.log(2e-3)], method="lm", max_nfev=8000)
            if best is None or r.cost < best.cost:
                best = r
        return best

    Cs = np.geomspace(1e-6, 1e2, 49)
    profile = []
    for C in Cs:
        r = fit_given_C(float(C))
        profile.append({"C": float(C), "sse": float(2 * r.cost),
                        "G": float(np.exp(r.x[0])), "MSE0": float(np.exp(r.x[1])),
                        "A_rad^-2": float(np.exp(r.x[0] - r.x[1]))})
    sse = np.array([p["sse"] for p in profile])
    best_i = int(np.argmin(sse))
    G_best, M_best, C_best = (profile[best_i]["G"], profile[best_i]["MSE0"],
                              profile[best_i]["C"])
    n_sat = int((x ** 2 * G_best >= C_best).sum())
    x2G_max = float((x ** 2).max() * G_best)
    # the flat set: every C whose refitted SSE is within 0.1 % of the best
    flat = Cs[sse <= sse[best_i] * 1.001]
    # unsaturated reference (one effective parameter) for the likelihood-ratio style read
    r1 = optimize.least_squares(
        lambda p: 10 * np.log10(1 + np.exp(p[0]) * x ** 2) - y, [math.log(1e4)], method="lm")
    sse_unsat = float(2 * r1.cost)
    return {
        "form": "gain_dB = 10*log10(1 + min(x^2*G, C)/MSE0), x in rad, all three free",
        "least_squares_solution": {"G": G_best, "MSE0": M_best, "C": C_best,
                                   "A_rad^-2": profile[best_i]["A_rad^-2"],
                                   "plateau_db": float(10 * math.log10(1 + C_best / M_best)),
                                   "sse_db2": float(sse[best_i])},
        "prereg_start": {"G": 0.0070, "MSE0": 2e-3, "C": 0.12,
                         "A_implied_rad^-2": 0.0070 / 2e-3},
        "effective_parameters": ["A = G/MSE0 (slope of the rising branch, rad^-2)",
                                 "plateau_dB = 10*log10(1 + C/MSE0) (height of the flat top)"],
        "identifiable": {"A_ratio_G_over_MSE0": True, "plateau_dB": bool(n_sat > 0),
                         "G_alone": False, "MSE0_alone": False,
                         "C_alone": bool(n_sat > 0 and flat.size == 1)},
        "C_within_0p1pct_of_best": [float(flat.min()), float(flat.max())] if flat.size else None,
        "largest_x2G_in_the_data": x2G_max,
        "n_points_saturated_at_best": n_sat,
        "n_points": int(len(y)),
        "sse_unsaturated_one_parameter_db2": sse_unsat,
        "sse_gain_from_saturation_db2": float(sse_unsat - sse[best_i]),
        "why": ("G, MSE0 and C are NOT three estimates: below the elbow the model depends "
                "only on A = G/MSE0, and above it only on 10*log10(1+C/MSE0). At most two "
                "numbers are identified, and only when some points sit on each side of the "
                "elbow. Here " + (f"{n_sat} of {len(y)} points are on the plateau, so the "
                                  "fit is 'a constant plus a short rising branch', which is "
                                  "what a non-predictive x looks like under this family"
                                  if n_sat else
                                  "no point reaches the elbow, so C is unidentified and the "
                                  "model collapses to the single parameter A")),
        "profile_over_C": profile,
    }


def paired_stats(deltas, seed=RNG_SEED, n_boot=N_BOOT):
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    out = {"n": n, "mean_db": float(d.mean()),
           "median_db": float(np.median(d)),
           "std_db": float(d.std(ddof=1)) if n > 1 else None,
           "sem_db": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
           "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())}
    if n >= 2:
        try:
            w = stats.wilcoxon(d, alternative="two-sided")
            out["wilcoxon_stat"] = float(w.statistic)
            out["wilcoxon_p"] = float(w.pvalue)
        except ValueError as exc:
            out["wilcoxon_p"] = None
            out["wilcoxon_error"] = str(exc)
        rng = np.random.default_rng(seed)
        bs = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
        out["mean_ci95_db"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    else:
        out["wilcoxon_p"] = None
        out["note"] = "n = 1: no test possible"
    out["above_noise_floor"] = bool(abs(out["mean_db"]) > NOISE_FLOOR_DB)
    return out


def rank_assoc(x, y, seed=RNG_SEED, n_boot=N_BOOT, strata=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return {"n": int(len(x)), "note": "too few points"}
    sp = stats.spearmanr(x, y)
    pe = stats.pearsonr(np.log(x), y)
    rng = np.random.default_rng(seed)
    groups = np.asarray(strata)[ok] if strata is not None else np.zeros(len(x), dtype=int)
    idx_by = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    bs = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(v, size=len(v), replace=True)
                               for v in idx_by.values()])
        if len(np.unique(x[pick])) < 3:
            continue
        bs.append(stats.spearmanr(x[pick], y[pick]).statistic)
    bs = np.asarray([b for b in bs if np.isfinite(b)])
    return {"n": int(len(x)), "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "spearman_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            if bs.size else None,
            "pearson_logx_r": float(pe.statistic), "pearson_logx_p": float(pe.pvalue)}


# --------------------------------------------------------------------------------------- #
#  4. figures                                                                               #
# --------------------------------------------------------------------------------------- #
FAMILY_STYLE = {
    "pinhole":          dict(marker="s", color="#1f6feb", label="pinhole (mip-NeRF 360)"),
    "fisheye_circular": dict(marker="o", color="#c9432a", label="circular fisheye"),
    "fisheye_fullframe": dict(marker="^", color="#2a8c4a", label="full-frame fisheye"),
    "panomorph":        dict(marker="D", color="#8250df", label="panomorph (ImmerVision)"),
}
TRACK_EDGE = {"myscenes_rttpf": "#000000", "fullcircle_refit_rttpf": "#ffffff",
              "mipnerf360_pinhole": "#000000", "workshop_immervision_rttpf": "#000000"}


def _panel(ax, rows, xkey, fit, title, subtitle, forbidden=False):
    xs = np.array([r[xkey] for r in rows], float)
    ys = np.array([r["y_db"] for r in rows], float)
    lo, hi = xs.min() / 1.7, xs.max() * 1.7
    grid = np.geomspace(lo, hi, 400)
    if fit and np.isfinite(fit["A_rad^-2"]):
        band_lo = 10 * np.log10(1 + fit["A_ci95"][0] * (grid * 1e-6) ** 2)
        band_hi = 10 * np.log10(1 + fit["A_ci95"][1] * (grid * 1e-6) ** 2)
        ax.fill_between(grid, band_lo, band_hi, color="#888888", alpha=0.18, lw=0,
                        zorder=1, label="fit 95 % CI (stratified bootstrap)")
        ax.plot(grid, 10 * np.log10(1 + fit["A_rad^-2"] * (grid * 1e-6) ** 2),
                color="#333333", lw=1.6, zorder=2,
                label=r"fit $10\log_{10}(1+Ax^2)$, $A$=%.2g rad$^{-2}$" % fit["A_rad^-2"])
        t = fit["threshold_urad_at_noise_floor"]
        if np.isfinite(t) and lo < t < hi:
            ax.axvline(t, color="#333333", ls=":", lw=1.2, zorder=2)
            ax.annotate(f"threshold\n{t:.0f} µrad", (t, ys.max() * 0.82),
                        xytext=(6, 0), textcoords="offset points", fontsize=7.5,
                        color="#333333")
    ax.axhline(NOISE_FLOOR_DB, color="#b0453a", ls="--", lw=1.1, zorder=2,
               label=f"measured noise floor {NOISE_FLOOR_DB:.3f} dB")
    ax.axhline(-NOISE_FLOOR_DB, color="#b0453a", ls="--", lw=0.7, alpha=0.5, zorder=2)
    ax.axhline(0.0, color="#999999", lw=0.6, zorder=1)
    seen = set()
    for r in rows:
        st = FAMILY_STYLE[r["family"]]
        ax.scatter(r[xkey], r["y_db"], marker=st["marker"], s=46, c=st["color"],
                   edgecolors=TRACK_EDGE.get(r["track"], "#000000"), linewidths=0.7,
                   zorder=4, label=st["label"] if st["label"] not in seen else None)
        seen.add(st["label"])
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"{xkey}  [µrad]")
    ax.set_ylabel("masked-PSNR gain, noncentral − off  [dB]")
    import textwrap
    ax.set_title(title, fontsize=10, loc="left", pad=34)
    ax.text(0.0, 1.008, "\n".join(textwrap.wrap(subtitle, 74)), transform=ax.transAxes,
            fontsize=7.2, va="bottom", linespacing=1.35,
            color="#b0453a" if forbidden else "#555555")
    ax.grid(True, which="both", alpha=0.18, lw=0.5)
    ax.legend(fontsize=6.6, loc="upper left", framealpha=0.92, borderpad=0.35,
              labelspacing=0.3)


def figure_dose_response(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in data["points"] if r["primary"]]
    a = data["associations"]["E_sfm_urad"]
    falsified = not (a["spearman_rho"] > 0 and a["spearman_p"] < 0.05)
    wi = data.get("associations_within_track", {}).get("POOLED_WITHIN_TRACK_exploratory", {})
    wi = wi.get("E_sfm_urad", {}) if wi else {}
    sub_a = ("FALSIFIED (prereg cond. 1): rho = %+.2f (p = %.3f) across tracks, "
             "the sign is %s." % (a["spearman_rho"], a["spearman_p"],
                                  "NEGATIVE" if a["spearman_rho"] < 0 else "positive"))
    if wi.get("statistic_pooled_within_track_spearman") is not None:
        sub_a += ("  WITHIN track (exploratory): rho = %+.2f (p = %.3f) — opposite sign."
                  % (wi["statistic_pooled_within_track_spearman"],
                     wi["p_two_sided_within_track_permutation"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    _panel(axes[0], rows, "E_sfm_urad", data["fits"]["E_sfm_urad"],
           "A. PRE-REGISTERED axis: E_sfm (training-free)", sub_a, forbidden=falsified)
    _panel(axes[1], rows, "E_learned_urad", data["fits"]["E_learned_urad"],
           "B. E_learned — SHAPE ONLY, NOT A PREDICTOR",
           "CIRCULAR: E_learned is the optimiser's own output (plan §2.3). "
           "rho = %+.2f (p = %.4f)."
           % (data["associations"]["E_learned_urad"]["spearman_rho"],
              data["associations"]["E_learned_urad"]["spearman_p"]),
           forbidden=True)
    fig.suptitle("W1 dose-response: PSNR gain of the learnable camera model vs pre-training "
                 "angular camera error\n"
                 f"{len(rows)} paired scenes, {len({r['family'] for r in rows})} camera "
                 "families, one shared masked eval pass "
                 "(r = 0.95 on fisheye, full frame on pinhole)", fontsize=9.4, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5)
    fig.savefig(path, format=os.path.splitext(path)[1].lstrip(".") or "svg")
    plt.close(fig)


def figure_estimator_agreement(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in data["points"] if r["primary"]]
    fig, axgrid = plt.subplots(2, 2, figsize=(11.6, 8.6))
    axes = [axgrid[0][0], axgrid[0][1], axgrid[1][0]]

    pairs = [("E_sfm_urad", "E_learned_urad"), ("E_sfm_urad", "E_sfm_sys_urad"),
             ("E_sfm_sys_urad", "E_learned_urad")]
    for ax, (kx, ky) in zip(axes[:3], pairs):
        for r in rows:
            if not (np.isfinite(r.get(kx) or np.nan) and np.isfinite(r.get(ky) or np.nan)):
                continue
            st = FAMILY_STYLE[r["family"]]
            ax.scatter(r[kx], r[ky], marker=st["marker"], s=44, c=st["color"],
                       edgecolors="#000000", linewidths=0.6, zorder=3)
        xs = [r[kx] for r in rows if np.isfinite(r.get(kx) or np.nan)]
        ys = [r[ky] for r in rows if np.isfinite(r.get(ky) or np.nan)]
        lim = [min(min(xs), min(ys)) / 2, max(max(xs), max(ys)) * 2]
        ax.plot(lim, lim, color="#999999", ls="--", lw=0.8, zorder=1, label="y = x")
        a = rank_assoc([r[kx] for r in rows], [r[ky] for r in rows], n_boot=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel(kx + "  [µrad]")
        ax.set_ylabel(ky + "  [µrad]")
        ax.set_title("%s. %s vs %s   $\\rho$ = %+.2f  (p = %.3f, n = %d)"
                     % ("ABC"[axes.index(ax)], kx.replace("_urad", ""),
                        ky.replace("_urad", ""), a["spearman_rho"], a["spearman_p"], a["n"]),
                     fontsize=9, loc="left")
        ax.grid(True, which="both", alpha=0.18, lw=0.5)

    # D. the only place where E_shared exists at all: the three lenses calibrated >= 3 times
    ax = axgrid[1][1]
    lens = data["lens_level"]
    track_of = {"myscenes": "myscenes_rttpf", "fullcircle_1": "fullcircle_refit_rttpf",
                "fullcircle_2": "fullcircle_refit_rttpf"}
    names, series = [], {"E_sfm (track median)": [], "E_calib (training-free)": [],
                         "E_shared (NOT training-free)": [], "E_learned (mean, circular)": []}
    for v in lens:
        sub = [r["E_sfm_urad"] for r in rows if r["track"] == track_of[v["key"]]]
        names.append(f"{v['label']}\ngain {v['mean_gain_db']:+.3f} dB")
        series["E_sfm (track median)"].append(float(np.median(sub)) if sub else np.nan)
        series["E_calib (training-free)"].append(v.get("E_calib_urad_rms") or np.nan)
        series["E_shared (NOT training-free)"].append(v.get("E_shared_urad_rms") or np.nan)
        series["E_learned (mean, circular)"].append(v.get("E_learned_urad_rms_mean") or np.nan)
    idx = np.arange(len(names))
    colours = ["#555555", "#1f6feb", "#c9432a", "#8250df"]
    for k, (lab, vals) in enumerate(series.items()):
        ax.bar(idx + (k - 1.5) * 0.2, vals, 0.2, label=lab, color=colours[k])
    ax.set_yscale("log")
    ax.set_ylim(50, 1600)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=7.6)
    ax.set_ylabel("angular error  [µrad]")
    ax.set_title("D. The three lenses where E_shared exists at all (>= 3 fits of one glass).\n"
                 "E_sfm orders them BACKWARDS vs the gain; E_calib / E_shared order them "
                 "correctly.", fontsize=8.4, loc="left")
    ax.legend(fontsize=6.8)
    ax.grid(True, axis="y", which="both", alpha=0.2, lw=0.5)

    present = {r["family"] for r in rows}
    handles = [plt.Line2D([], [], ls="none", marker=v["marker"], color=v["color"],
                          markeredgecolor="k", markersize=7, label=v["label"])
               for k, v in FAMILY_STYLE.items() if k in present]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    import textwrap
    fig.suptitle("\n".join(textwrap.wrap(
        "Estimator agreement. E_sfm (training-free, observation-weighted median), "
        "E_sfm_sys (its azimuth-averaged systematic part), E_shared (cross-scene mean of "
        "the learned residual), E_learned (the optimiser's output). Panels A-C are per "
        "scene; E_shared has no per-scene value, so panel D is the lens level -- the only "
        "level at which all four coexist.", 118)), fontsize=8.6, y=0.985)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(path, format=os.path.splitext(path)[1].lstrip(".") or "svg")
    plt.close(fig)


def figure_lens_level(data, path):
    """The lens-level comparison E_calib (training-free) vs E_shared (not) vs the gain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lens = data["lens_level"]
    names = [v["label"] for v in lens]
    idx = np.arange(len(names))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    w = 0.26
    ax.bar(idx - w, [v.get("E_calib_urad_rms") or 0 for v in lens], w,
           label="E_calib (training-free)", color="#1f6feb")
    ax.bar(idx, [v.get("E_shared_urad_rms") or 0 for v in lens], w,
           label="E_shared (NOT training-free)", color="#c9432a")
    ax.bar(idx + w, [v.get("E_learned_urad_rms_mean") or 0 for v in lens], w,
           label="mean E_learned (circular)", color="#8250df")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("angular error  [µrad]")
    ax.set_title("A. Three ways to size the same lens's calibration error", fontsize=10,
                 loc="left")
    ax.legend(fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.2, lw=0.5)

    for v in lens:
        if v.get("E_calib_urad_rms") is None or v.get("mean_gain_db") is None:
            continue
        ax2.errorbar(v["E_calib_urad_rms"], v["mean_gain_db"],
                     yerr=[[v["mean_gain_db"] - v["gain_ci95"][0]],
                           [v["gain_ci95"][1] - v["mean_gain_db"]]],
                     fmt="o", ms=7, color="#1f6feb", ecolor="#1f6feb", capsize=3)
        ax2.annotate(v["label"], (v["E_calib_urad_rms"], v["mean_gain_db"]),
                     xytext=(7, -3), textcoords="offset points", fontsize=8)
    ax2.axhline(NOISE_FLOOR_DB, color="#b0453a", ls="--", lw=1.1,
                label=f"noise floor {NOISE_FLOOR_DB:.3f} dB")
    ax2.axhline(0, color="#999", lw=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("E_calib, cross-fit calibration disagreement  [µrad]")
    ax2.set_ylabel("mean paired gain  [dB]")
    ax2.set_title("B. The only training-free statistic that orders the tracks (n = 3 lenses)",
                  fontsize=10, loc="left")
    ax2.legend(fontsize=7.5)
    ax2.grid(True, which="both", alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, format=os.path.splitext(path)[1].lstrip(".") or "svg")
    plt.close(fig)


# --------------------------------------------------------------------------------------- #
#  5. main                                                                                  #
# --------------------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", action="store_true",
                    help="skip the image/COLMAP passes, refit and redraw from dose_response.json")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    args = ap.parse_args()

    t0 = time.time()
    if args.from_json:
        src = CACHE_JSON if os.path.exists(CACHE_JSON) else OUT_JSON
        data = json.load(open(src))
        rows = data["points"]
        print(f"  reusing the cached heavy passes from {src}")
    else:
        print("[1/4] paired masked PSNR, one shared eval pass ...", flush=True)
        rows = collect_gains(build_pairs())

        print("[2/4] E_sfm from sfm_residual.json ...", flush=True)
        sfm = json.load(open(os.path.join(HERE, "sfm_residual.json")))["tracks"]
        for r in rows:
            t = sfm[r["sfm_key"]]
            r["E_sfm_urad"] = t["residual_urad_theta_le_85_5"]["median"]
            r["E_sfm_rms_urad"] = t["residual_urad_theta_le_85_5"]["rms"]
            r["E_sfm_eval_px"] = t["residual_eval_px_theta_le_85_5"]["median"]
            r["eval_resolution"] = t["eval_resolution"]
            r["colmap_model"] = t["colmap_model"]

        print("[3/4] E_sfm_sys (EXPLORATORY) + E_calib + E_shared + E_learned ...", flush=True)
        keys = sorted({r["sfm_key"] for r in rows})
        sys_out = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(sfm_systematic, keys):
                sys_out[res["track"]] = res
                print(f"    E_sfm_sys {res['track']:40s} {res['E_sfm_sys_urad']:8.1f} µrad",
                      flush=True)
        for r in rows:
            r["E_sfm_sys_urad"] = sys_out[r["sfm_key"]]["E_sfm_sys_urad"]

        ps = json.load(open(os.path.join(HERE, "plate_scale.json")))["cameras"]
        for r in rows:
            uids = [1, 2] if r["track"] == "fullcircle_refit_rttpf" else [1]
            vals, det = [], {}
            for uid in uids:
                cam = ps.get(f"{r['track']}/{r['scene']}/cam{uid}")
                if cam is None and r["kind"] == "pinhole" and uid == 1:
                    # kitchen / room: their runs post-date plate_scale.json (another agent's
                    # file, not rewritten here). Closed form, validated against A2's five.
                    cam = pinhole_plate_scale_from_run(r["off"])
                if cam is None:
                    continue
                grid, w = area_weights_from_plate_scale(cam)
                lr = learned_rms(r["nc"], uid, grid, w)
                if lr:
                    det[str(uid)] = lr
                    vals.append(lr["E_learned_urad_rms"])
            r["E_learned_detail"] = det
            r["E_learned_urad"] = float(np.sqrt(np.mean(np.square(vals)))) if vals else None

        import calib_consistency as cc
        lens_level = []
        e_calib = {
            "myscenes": calib_disagreement(
                "myscenes rttpf, 7 independent fits",
                [f"{cc.MY_ROOT}/{cc.MYSCENES[s]}/distorted/sparse/0" for s in cc.MYSCENES],
                1, "fisheye_circular"),
            "fullcircle_1": calib_disagreement(
                "FullCircle refit_rttpf lens 1, 9 re-fits",
                [f"{cc.FC_ROOT}/{s}/distorted/sparse/0" for s in cc.FC], 1, "fisheye_circular"),
            "fullcircle_2": calib_disagreement(
                "FullCircle refit_rttpf lens 2, 9 re-fits",
                [f"{cc.FC_ROOT}/{s}/distorted/sparse/0" for s in cc.FC], 2, "fisheye_circular"),
        }
        ocv = calib_disagreement(
            "myscenes OPENCV_FISHEYE warmstart, 7 fits (no paired y)",
            [f"/workspace/gray/data/myscenes_ocv/{s}_warmstart/distorted/sparse/0"
             for s in cc.MYSCENES], 1, "fisheye_circular")
        if ocv:
            e_calib["myscenes_ocv_context"] = ocv

        e_shared = [
            shared_learned("myscenes (1 lens, 7 fits)", list(cc.MYSCENES),
                           lambda s: f"{cc.MY_ROOT}/{cc.MYSCENES[s]}/distorted/sparse/0",
                           lambda s: cc.MY_RUNS.format(scene=s), 1),
            shared_learned("FullCircle refit_rttpf lens 1 (9 re-fits)", cc.FC,
                           lambda s: f"{cc.FC_ROOT}/{s}/distorted/sparse/0",
                           lambda s: cc.FC_RUNS.format(scene=s), 1),
            shared_learned("FullCircle refit_rttpf lens 2 (9 re-fits)", cc.FC,
                           lambda s: f"{cc.FC_ROOT}/{s}/distorted/sparse/0",
                           lambda s: cc.FC_RUNS.format(scene=s), 2),
        ]
        data = {"e_calib": e_calib, "e_shared": e_shared, "sfm_sys_detail": sys_out,
                "lens_level": lens_level}
        # cache the two expensive passes so --from-json can refit and redraw in seconds
        with open(CACHE_JSON, "w") as h:
            json.dump({"points": rows, **data}, h, indent=1)
        print(f"  cached the heavy passes to {CACHE_JSON}")

    # ---------------- statistics (always recomputed) --------------------------------- #
    print("[4/4] statistics, fits, figures ...", flush=True)
    primary = [r for r in rows if r["primary"]]
    tracks = sorted({r["track"] for r in primary})
    per_track = {}
    for t in tracks:
        sub = [r for r in primary if r["track"] == t]
        per_track[t] = {
            "scenes": [r["scene"] for r in sub],
            "gain": paired_stats([r["y_db"] for r in sub], n_boot=args.bootstrap),
            "gain_pooled_mean_db": float(np.mean([r["y_db_pooled"] for r in sub])),
            "E_sfm_urad_range": [min(r["E_sfm_urad"] for r in sub),
                                 max(r["E_sfm_urad"] for r in sub)],
            "E_sfm_sys_urad_range": [min(r["E_sfm_sys_urad"] for r in sub),
                                     max(r["E_sfm_sys_urad"] for r in sub)],
            "E_learned_urad_range": [min(r["E_learned_urad"] for r in sub),
                                     max(r["E_learned_urad"] for r in sub)],
            "eval_resolution": sub[0].get("eval_resolution"),
            "family": sub[0]["family"],
        }
    per_track["ALL_POOLED_do_not_publish_as_one_number"] = {
        "gain": paired_stats([r["y_db"] for r in primary], n_boot=args.bootstrap),
        "warning": "pools four camera families and three frame regimes; the plan forbids "
                   "averaging across lenses. Reported only so the pooled Wilcoxon is on record.",
    }

    strata = [r["track"] for r in primary]
    assoc, fits = {}, {}
    for key in ("E_sfm_urad", "E_sfm_rms_urad", "E_sfm_sys_urad", "E_learned_urad"):
        xs = [r[key] for r in primary]
        ys = [r["y_db"] for r in primary]
        assoc[key] = rank_assoc(xs, ys, n_boot=args.bootstrap, strata=strata)
        fits[key] = fit_transfer(xs, ys, n_boot=args.bootstrap, strata=strata)
        fits[key]["three_parameter"] = fit_transfer_three_parameter(xs, ys)
        fits[key]["x_is_a_legitimate_predictor"] = key != "E_learned_urad"
        if key == "E_learned_urad":
            fits[key]["warning"] = ("CIRCULAR: E_learned is the training result. This fit "
                                    "describes the SHAPE of the law, it does not predict.")
    # within-track associations (removes the between-track confound)
    assoc_within = {}
    for t in tracks:
        sub = [r for r in primary if r["track"] == t]
        if len(sub) < 4:
            continue
        assoc_within[t] = {k: rank_assoc([r[k] for r in sub], [r["y_db"] for r in sub],
                                         n_boot=args.bootstrap)
                           for k in ("E_sfm_urad", "E_sfm_sys_urad", "E_learned_urad")}
    assoc_within["POOLED_WITHIN_TRACK_exploratory"] = {
        k: stratified_rank_assoc([r[k] for r in primary], [r["y_db"] for r in primary], strata)
        for k in ("E_sfm_urad", "E_sfm_sys_urad", "E_learned_urad")}

    # lens-level table (E_calib / E_shared / gain)
    lens_gain = {
        "myscenes": [r["y_db"] for r in primary if r["track"] == "myscenes_rttpf"],
        "fullcircle_1": [r["y_db"] for r in primary if r["track"] == "fullcircle_refit_rttpf"],
        "fullcircle_2": [r["y_db"] for r in primary if r["track"] == "fullcircle_refit_rttpf"],
    }
    lens_learned = {
        "myscenes": [v["E_learned_urad"] for v in primary if v["track"] == "myscenes_rttpf"],
        "fullcircle_1": [d["1"]["E_learned_urad_rms"] for v in primary
                         if v["track"] == "fullcircle_refit_rttpf"
                         for d in [v["E_learned_detail"]] if "1" in d],
        "fullcircle_2": [d["2"]["E_learned_urad_rms"] for v in primary
                         if v["track"] == "fullcircle_refit_rttpf"
                         for d in [v["E_learned_detail"]] if "2" in d],
    }
    labels = {"myscenes": "myscenes lens (7 fits)",
              "fullcircle_1": "FullCircle lens 1 (9)",
              "fullcircle_2": "FullCircle lens 2 (9)"}
    e_shared_by = {"myscenes": data["e_shared"][0], "fullcircle_1": data["e_shared"][1],
                   "fullcircle_2": data["e_shared"][2]}
    import calib_consistency as cc
    e_shared_fixed = data.get("e_shared_bugfixed") or {
        "myscenes": shared_learned_fixed(
            "myscenes (1 lens, 7 fits)", list(cc.MYSCENES),
            lambda s: f"{cc.MY_ROOT}/{cc.MYSCENES[s]}/distorted/sparse/0",
            lambda s: cc.MY_RUNS.format(scene=s), 1),
        "fullcircle_1": shared_learned_fixed(
            "FullCircle refit_rttpf lens 1 (9 re-fits)", cc.FC,
            lambda s: f"{cc.FC_ROOT}/{s}/distorted/sparse/0",
            lambda s: cc.FC_RUNS.format(scene=s), 1),
        "fullcircle_2": shared_learned_fixed(
            "FullCircle refit_rttpf lens 2 (9 re-fits)", cc.FC,
            lambda s: f"{cc.FC_ROOT}/{s}/distorted/sparse/0",
            lambda s: cc.FC_RUNS.format(scene=s), 2),
    }
    if data.get("e_shared_bugfixed") is None:
        data["e_shared_bugfixed"] = e_shared_fixed
        cached = json.load(open(CACHE_JSON)) if os.path.exists(CACHE_JSON) else {}
        cached["e_shared_bugfixed"] = e_shared_fixed
        with open(CACHE_JSON, "w") as h:
            json.dump(cached, h, indent=1)
    lens_level = []
    for k, label in labels.items():
        st = paired_stats(lens_gain[k], n_boot=args.bootstrap)
        lens_level.append({
            "key": k, "label": label,
            "E_calib_urad_rms": (data["e_calib"][k] or {}).get("E_calib_urad_rms"),
            "E_calib_urad_at_edge": (data["e_calib"][k] or {}).get("E_calib_urad_at_edge"),
            "E_shared_urad_rms": (e_shared_fixed[k] or {}).get("E_shared_urad_rms"),
            "E_shared_urad_rms_as_calib_consistency_computes_it":
                (e_shared_by[k] or {}).get("E_shared_urad_rms"),
            "E_learned_urad_rms_mean": float(np.mean(lens_learned[k])) if lens_learned[k] else None,
            "mean_gain_db": st["mean_db"], "gain_ci95": st["mean_ci95_db"],
            "wilcoxon_p": st["wilcoxon_p"], "n_scenes": st["n"],
        })
    data["lens_level"] = lens_level

    # ---- the practical threshold on the ONLY training-free axis that orders the tracks ----
    # n = 3 lenses. Descriptive by construction: the plan itself says no fit on a lens-level
    # statistic can be more than descriptive. Reported because the deliverable asks for a
    # threshold and E_sfm cannot supply a meaningful one (its fit has r2 < 0).
    pts = [(v["E_calib_urad_rms"], v["mean_gain_db"], v["label"]) for v in lens_level
           if v.get("E_calib_urad_rms")]
    xs = np.array([p[0] for p in pts]) * 1e-6
    ys = np.array([p[1] for p in pts])
    fit3 = optimize.least_squares(
        lambda p: 10 * np.log10(1 + np.exp(p[0]) * xs ** 2) - ys, [math.log(1e4)],
        method="lm")
    A3 = float(np.exp(fit3.x[0]))
    per_lens_A = [float((10 ** (y / 10) - 1) / x ** 2) for x, y in zip(xs, ys) if y > 0]
    thr = lambda a: float(np.sqrt((10 ** (NOISE_FLOOR_DB / 10) - 1) / a) * 1e6)  # noqa: E731
    data["threshold_on_E_calib"] = {
        "status": "DESCRIPTIVE ONLY, n = 3 lenses. No confidence interval is quotable at "
                  "n = 3; the spread of the three per-lens implied A values is given "
                  "instead, and it is the honest uncertainty statement.",
        "points": [{"lens": p[2], "E_calib_urad": p[0], "mean_gain_db": p[1]} for p in pts],
        "A_rad^-2_joint_least_squares": A3,
        "A_rad^-2_implied_per_lens": per_lens_A,
        "threshold_urad_at_noise_floor_joint": thr(A3),
        "threshold_urad_range_from_per_lens_A": [thr(max(per_lens_A)), thr(min(per_lens_A))]
        if per_lens_A else None,
        "noise_floor_db": NOISE_FLOOR_DB,
        "reading": "below roughly 0.4-0.9 mrad of cross-fit calibration disagreement, the "
                   "learnable camera model buys less than the run-to-run noise floor",
    }

    out = {
        "what": "W1 dose-response: masked-PSNR gain of --camera_opt noncentral vs off, "
                "paired by scene, against pre-training angular camera error.",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": "scripts/analysis/dose_response_prereg.md "
                           "(2026-08-10T21:46:30Z, commit 8e0a15bf)",
        "gpu_used": False,
        "noise_floor_db": NOISE_FLOOR_DB,
        "conventions": {
            "y": "disk_per_view_mean PSNR difference, one shared eval pass over the raw "
                 "renders; mask r=0.95 on fisheye tracks, full frame on pinhole",
            "x_units": "microradians; never pixels (heterogeneous -r, and a fisheye plate "
                       "scale that is not fx)",
            "E_sfm": "median |angular reprojection residual| inside theta<=85.5 deg, from "
                     "sfm_residual.json",
            "E_sfm_sys": "EXPLORATORY: area-weighted RMS of the per-theta MEAN meridional "
                         "residual, sampling-variance corrected. The part a radial camera "
                         "residual could absorb.",
            "E_calib": "training-free cross-fit disagreement about theta at a fixed pixel "
                       "radius, area-weighted RMS over the evaluated disk",
            "E_shared": "cross-scene mean of the LEARNED residual -- NOT training-free",
            "E_learned": "the optimiser's own output -- FORBIDDEN as an x axis",
        },
        "points": rows,
        "per_track": per_track,
        "lens_level": lens_level,
        "threshold_on_E_calib": data["threshold_on_E_calib"],
        "associations": assoc,
        "associations_within_track": assoc_within,
        "fits": fits,
        "e_calib": data["e_calib"],
        "e_shared": data["e_shared"],
        "e_shared_bugfixed": data.get("e_shared_bugfixed"),
        "known_issues": [
            {"id": "radial_index_layout",
             "what": "calib_consistency.py:54 `RADIAL = [4, 5, 8, 9]` is the 12-parameter "
                     "THIN_PRISM_FISHEYE layout, applied to 16-parameter "
                     "RAD_TAN_THIN_PRISM_FISHEYE cameras, where the six radial coefficients "
                     "are at 4..9. Verified against pycolmap's own `params_info`: "
                     "'fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3'.",
             "plan_says": "Phase-1 plan section 2.2 bis asserts [4, 5, 8, 9] is CORRECT and "
                          "cites gray/camera.py:58. That line is the THIN_PRISM_FISHEYE "
                          "branch; the RAD_TAN branch is at :61 and unpacks 16 names. The "
                          "plan is wrong on this point.",
             "effect_here": "none on any x axis of this file. E_sfm comes from pycolmap "
                            "projections, E_calib from pycolmap projections, E_learned from "
                            "the checkpoint spline. Only `e_shared` inherits it, and "
                            "`e_shared_bugfixed` is published beside it."},
            {"id": "fullcircle_checkpoints_moved",
             "what": "pueue 1539-1545 re-ran 7 of the 9 FullCircle `noncentral` runs in "
                     "place on 2026-08-10 19:30-20:40 with a byte-identical command line. "
                     "Everything in this file post-dates that; calib_consistency.json "
                     "(13:16) and plate_scale.json (16:58) pre-date it, so their FullCircle "
                     "learned-residual numbers come from the earlier training instance.",
             "effect_here": "quantified in `fullcircle_seed_repeat`; the FullCircle mean "
                            "gain is +0.0156 dB on the earlier instance and +0.0031 dB on "
                            "the current one. Both are null against the 0.068 dB floor."},
            {"id": "e_shared_is_not_training_free",
             "what": "E_shared is the cross-scene mean of the LEARNED residual, i.e. a "
                     "decomposition of the training result, not an input to it. The plan "
                     "lists it as a mechanistic predictor; at track level, which is where "
                     "all the separating power lives, regressing the gain on it is circular "
                     "in the same way the plan forbids for E_learned.",
             "effect_here": "E_shared is reported and never used as an x axis. E_calib, "
                            "computed from the COLMAP cameras alone, is the training-free "
                            "statistic that the plan's prose actually describes."},
        ],
        "sfm_sys_detail": data.get("sfm_sys_detail", {}),
        "prereg_falsification": {},
    }
    a = assoc["E_sfm_urad"]
    f = fits["E_sfm_urad"]
    out["prereg_falsification"] = {
        "condition_1_spearman_not_significantly_positive": bool(
            not (a["spearman_rho"] > 0 and a["spearman_p"] < 0.05)),
        "condition_2_track_order_violated": None,   # filled just below
        "condition_3_A_ci_contains_zero": bool(f["A_ci95"][0] <= 0 <= f["A_ci95"][1]
                                               or f["A_ci95"][0] < 1e-9),
        "spearman": a, "fit": f,
    }
    ranked = sorted([kv for kv in per_track.items() if "E_sfm_urad_range" in kv[1]],
                    key=lambda kv: kv[1]["E_sfm_urad_range"][0])
    viol = []
    for i in range(len(ranked)):
        for j in range(len(ranked)):
            ki, kj = ranked[i], ranked[j]
            if ki[1]["E_sfm_urad_range"][0] > kj[1]["E_sfm_urad_range"][1] and \
               ki[1]["gain"]["mean_db"] + NOISE_FLOOR_DB < kj[1]["gain"]["mean_db"]:
                viol.append({"higher_E_sfm": ki[0], "lower_E_sfm": kj[0],
                             "gain_higher": ki[1]["gain"]["mean_db"],
                             "gain_lower": kj[1]["gain"]["mean_db"]})
    out["prereg_falsification"]["condition_2_track_order_violated"] = bool(viol)
    out["prereg_falsification"]["condition_2_violations"] = viol

    # ---- the pre-registered 22 alone, so the two additions cannot look like point picking --
    p22 = [r for r in primary if (r["track"], r["scene"]) not in PREREG_22_EXCLUDED]
    s22 = [r["track"] for r in p22]
    out["prereg_22_only"] = {
        "note": "identical analysis restricted to the 22 scenes listed in the "
                "pre-registration; kitchen and room are the only additions and they are "
                "additions, not substitutions (prereg section 3).",
        "n": len(p22),
        "excluded": sorted(f"{t}/{s}" for t, s in PREREG_22_EXCLUDED),
        "associations": {k: rank_assoc([r[k] for r in p22], [r["y_db"] for r in p22],
                                       n_boot=args.bootstrap, strata=s22)
                         for k in ("E_sfm_urad", "E_sfm_rms_urad", "E_sfm_sys_urad",
                                   "E_learned_urad")},
        "fit_E_sfm": fit_transfer([r["E_sfm_urad"] for r in p22], [r["y_db"] for r in p22],
                                  n_boot=args.bootstrap, strata=s22),
        "fit_E_learned": fit_transfer([r["E_learned_urad"] for r in p22],
                                      [r["y_db"] for r in p22],
                                      n_boot=args.bootstrap, strata=s22),
        "mipnerf360_gain": paired_stats(
            [r["y_db"] for r in p22 if r["track"] == "mipnerf360_pinhole"],
            n_boot=args.bootstrap),
    }
    out["pinhole_plate_scale_validation"] = validate_pinhole_plate_scale()
    out["fullcircle_seed_repeat"] = fullcircle_seed_repeat(primary)

    with open(OUT_JSON, "w") as h:
        json.dump(out, h, indent=1)
    print(f"wrote {OUT_JSON}")

    os.makedirs(FIG_DIR, exist_ok=True)
    figure_dose_response(out, os.path.join(FIG_DIR, "dose_response.svg"))
    figure_estimator_agreement(out, os.path.join(FIG_DIR, "estimator_agreement.svg"))
    figure_lens_level(out, os.path.join(FIG_DIR, "estimator_lens_level.svg"))
    print(f"wrote {FIG_DIR}/dose_response.svg, estimator_agreement.svg, "
          f"estimator_lens_level.svg")

    # ------------- console summary ---------------------------------------------------- #
    print("\n=== paired gains, per track (Wilcoxon signed-rank) ===")
    for t, v in per_track.items():
        g = v["gain"]
        ci = g.get("mean_ci95_db") or [float("nan")] * 2
        print(f"  {t:46s} n={g['n']:2d}  mean={g['mean_db']:+.4f} dB "
              f"CI95=[{ci[0]:+.4f},{ci[1]:+.4f}]  p={g['wilcoxon_p']}")
    print(f"\n=== association of the gain with each x estimator (n = {len(primary)}) ===")
    for k, v in assoc.items():
        ci = v.get("spearman_ci95") or [float("nan")] * 2
        print(f"  {k:20s} rho={v['spearman_rho']:+.3f} p={v['spearman_p']:.4f} "
              f"CI95=[{ci[0]:+.2f},{ci[1]:+.2f}]  A={fits[k]['A_rad^-2']:.3g} rad^-2 "
              f"CI=[{fits[k]['A_ci95'][0]:.3g},{fits[k]['A_ci95'][1]:.3g}]")
    print("\n=== pre-registered falsification conditions (E_sfm, the primary axis) ===")
    for k, v in out["prereg_falsification"].items():
        if isinstance(v, bool):
            print(f"  {k:52s} {v}")
    print(f"\ndone in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
