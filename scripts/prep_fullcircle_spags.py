#!/usr/bin/env python
"""Prepare FullCircle omni panoramas for native SPaGS (RaRPano dataloader).

Per scene, builds /workspace/dataset/fullcircle_pano/<scene>/:
    images/frame_NNNN.png        2048x1024 ERP (downscaled from 2880x5760)
    masks/frame_NNNN.png         person/capturer mask, 255 = transient (grayscale)
    reconstruction.json          OpenSfM-style: spherical shots + colmap points

Pano pose convention (verified by fisheye->pano reprojection, corr 0.994/0.987 on
room2): the released omni ERP uses the same lon/lat mapping as make_fiord_panos'
pano_directions, in the frame R_front @ cam1, where R_front is the fixed rig
rotation from the authors' masking/mapping.txt. Hence per omni frame:
    R_pano_from_world = R_front @ R_cam1_from_world
    t_pano            = R_front @ t_cam1
"""

import json
import os
import sys
import concurrent.futures as futures

import cv2
import numpy as np
from PIL import Image

SRC_ROOT = "/workspace/dataset/fullcircle"
DEST_ROOT = "/workspace/dataset/fullcircle_pano"
PANO_W, PANO_H = 2048, 1024
RVEC_FRONT = np.array([-0.0371781, -0.00746628, 0.00398891])  # authors' mapping.txt
CORR_MIN = 0.97  # per-scene safety check on the pose convention


def resize_one(job):
    kind, src, dst = job
    if os.path.exists(dst):
        return
    if kind == "image":
        im = Image.open(src).convert("RGB").resize((PANO_W, PANO_H), Image.LANCZOS)
        im.save(dst)
    else:  # mask: grayscale, conservative (BOX downscale keeps soft edges, low thr)
        im = Image.open(src).convert("L").resize((PANO_W, PANO_H), Image.BOX)
        arr = (np.array(im) > 32).astype(np.uint8) * 255
        Image.fromarray(arr).save(dst)


def pano_dirs(w, h):
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    lon = (u - 0.5) * 2 * np.pi
    lat = (0.5 - v) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat),
                     np.cos(lat) * np.cos(lon)], -1)


def check_convention(scene_dir, rec, R_front, stem):
    """Reproject camera1's x4 fisheye into the pano frame; corr vs provided omni."""
    by_name = {img.name: img for img in rec.images.values()}
    img1 = by_name[f"camera1/{stem}.png"]
    cam = rec.cameras[img1.camera_id]
    R1 = img1.cam_from_world().rotation.matrix()
    w, h = 1024, 512
    d_local = pano_dirs(w, h)
    d_cam = (d_local @ (R_front @ R1)) @ R1.T  # world dirs @ R1.T = cam1 dirs
    theta = np.arctan2(np.hypot(d_cam[..., 0], d_cam[..., 1]), d_cam[..., 2])
    fx, fy, cx, cy, k1, k2, k3, k4 = cam.params
    t2 = theta * theta
    td = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    phi = np.arctan2(d_cam[..., 1], d_cam[..., 0])
    x = (fx * td * np.cos(phi) + cx) / 4
    y = (fy * td * np.sin(phi) + cy) / 4
    im = np.asarray(Image.open(
        os.path.join(scene_dir, "images_4", "camera1", stem + ".png")).convert("RGB"),
        dtype=float)
    side = im.shape[0]
    valid = (theta < np.radians(80)) & (x >= 0) & (x < side - 1) & (y >= 0) & (y < side - 1)
    omni = np.asarray(Image.open(
        os.path.join(scene_dir, "omni", "images", stem + ".png")).convert("RGB")
        .resize((w, h), Image.LANCZOS), dtype=float)
    proj = im[np.clip(y, 0, side - 1).astype(int)[valid].ravel(),
              np.clip(x, 0, side - 1).astype(int)[valid].ravel()]
    return float(np.corrcoef(proj.ravel(), omni[valid].ravel())[0, 1])


def prep_scene(scene, workers=12):
    import pycolmap

    src = os.path.join(SRC_ROOT, scene)
    dest = os.path.join(DEST_ROOT, scene)
    os.makedirs(os.path.join(dest, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest, "masks"), exist_ok=True)

    rec = pycolmap.Reconstruction(os.path.join(src, "sparse", "0"))
    R_front, _ = cv2.Rodrigues(RVEC_FRONT)
    by_name = {img.name: img for img in rec.images.values()}

    omni_dir = os.path.join(src, "omni", "images")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(omni_dir)
                   if f.endswith(".png"))
    stems = [s for s in stems if f"camera1/{s}.png" in by_name]
    missing = len(os.listdir(omni_dir)) - len(stems)

    corr = check_convention(src, rec, R_front, stems[len(stems) // 2])
    if corr < CORR_MIN:
        raise RuntimeError(f"{scene}: pano convention check corr={corr:.4f} < {CORR_MIN}")

    jobs = []
    for s in stems:
        jobs.append(("image", os.path.join(omni_dir, s + ".png"),
                     os.path.join(dest, "images", s + ".png")))
        msrc = os.path.join(src, "omni", "masks", s + "_mask.png")
        if os.path.exists(msrc):
            jobs.append(("mask", msrc, os.path.join(dest, "masks", s + ".png")))
    with futures.ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(resize_one, jobs, chunksize=4))

    shots = {}
    for s in stems:
        img1 = by_name[f"camera1/{s}.png"]
        R1 = img1.cam_from_world().rotation.matrix()
        t1 = np.asarray(img1.cam_from_world().translation)
        Rp = R_front @ R1
        rvec, _ = cv2.Rodrigues(Rp)
        rvec = rvec.ravel()
        if np.linalg.norm(rvec) < 1e-8:  # identity nudge (dataloader PCA guard)
            rvec = np.array([1e-7, 0.0, 0.0])
        shots[s + ".png"] = {
            "rotation": [float(x) for x in rvec],
            "translation": [float(x) for x in (R_front @ t1)],
            "camera": "insta360",
        }

    points = {}
    for pid, p in rec.points3D.items():
        points[str(pid)] = {"coordinates": [float(x) for x in p.xyz],
                            "color": [int(c) for c in p.color]}

    recon = [{
        "cameras": {"insta360": {"projection_type": "spherical",
                                 "width": PANO_W, "height": PANO_H}},
        "shots": shots,
        "points": points,
    }]
    with open(os.path.join(dest, "reconstruction.json"), "w") as f:
        json.dump(recon, f)

    print(f"[{scene}] panos={len(stems)} (skipped {missing} unregistered) "
          f"masks={len(os.listdir(os.path.join(dest, 'masks')))} "
          f"points={len(points)} conv_corr={corr:.4f}", flush=True)


def main():
    scenes = sys.argv[1:] or sorted(
        s for s in os.listdir(SRC_ROOT)
        if os.path.exists(os.path.join(SRC_ROOT, s, ".done"))
    )
    for scene in scenes:
        try:
            prep_scene(scene)
        except Exception as e:
            print(f"[{scene}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
