#!/usr/bin/env python
"""Resample SPaGS golden-pose ERP renders into the golden fisheye test views.

For each view in the scene's test.txt (gray's exact test order), samples the
rendered pano (camera "insta360", make_fiord_panos ERP convention, pose from
golden_poses.json) into that view's OPENCV_FISHEYE camera at x4 resolution.
Rotation-only per pixel: d_cam (undistort) -> world (view pose) -> pano frame
(golden pose) -> ERP uv -> bilinear sample with horizontal wrap.

Output mimics a gray run dir so gray's metrics.py runs unchanged:
    <out>/<scene>/test/30000/opencv_fisheye/{renders/NNNNN.png, gt/ (links to the
    gray-masked run's gt), masks.json, valid_mask_cam{1,2}.png}

Usage: resample_spags_golden_fisheye.py <scene> <golden_rgb_dir> [<out_root>]
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np

SRC_ROOT = "/workspace/dataset/fullcircle"
BASE_ROOT = "/workspace/dataset/fullcircle_baselines"
PANO_ROOT = "/workspace/dataset/fullcircle_pano"
GRAY_OUT = "/workspace/gray/worktrees/person-masks/out/fullcircle"
DOWNSCALE = 4


def fisheye_dirs(cam, side):
    """Unit ray directions for every pixel of the x4-scaled OPENCV_FISHEYE camera."""
    fx, fy, cx, cy, k1, k2, k3, k4 = cam.params
    K = np.array([[fx / DOWNSCALE, 0, cx / DOWNSCALE],
                  [0, fy / DOWNSCALE, cy / DOWNSCALE], [0, 0, 1]])
    D = np.array([k1, k2, k3, k4])
    xs, ys = np.meshgrid(np.arange(side) + 0.5, np.arange(side) + 0.5)
    pts = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float64)[:, None, :]
    und = cv2.fisheye.undistortPoints(pts, K, D, criteria=(
        cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 50, 1e-10))[:, 0, :]
    d = np.concatenate([und, np.ones((len(und), 1))], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.reshape(side, side, 3)


def main():
    import pycolmap
    scene, golden_dir = sys.argv[1], sys.argv[2]
    out_root = sys.argv[3] if len(sys.argv) > 3 else \
        "/workspace/nerficg-native/output/SPaGS/golden_fisheye"
    side = 720

    rec = pycolmap.Reconstruction(os.path.join(SRC_ROOT, scene, "sparse", "0"))
    by_name = {img.name: img for img in rec.images.values()}
    poses = json.load(open(os.path.join(PANO_ROOT, scene, "golden_poses.json")))["shots"]

    test_names = [l.strip() for l in
                  open(f"{BASE_ROOT}/{scene}/distorted/sparse/0/test.txt") if l.strip()]
    test_names.sort()

    mode_dir = os.path.join(out_root, scene, "test", "30000", "opencv_fisheye")
    rdir = os.path.join(mode_dir, "renders")
    os.makedirs(rdir, exist_ok=True)

    # per-camera pixel ray grids (2 cameras)
    dirs_by_cam = {cid: fisheye_dirs(cam, side) for cid, cam in rec.cameras.items()}

    masks_index, missing = {}, []
    for i, name in enumerate(test_names):
        stem = os.path.splitext(os.path.basename(name))[0]
        img = by_name[name]
        shot = poses.get(stem + ".png")
        ppath = os.path.join(golden_dir, stem + ".png")
        out_name = f"{i:05d}.png"
        masks_index[out_name] = f"valid_mask_cam{img.camera_id}.png"
        if shot is None or not os.path.exists(ppath):
            missing.append(name)
            continue
        pano = cv2.imread(ppath, cv2.IMREAD_COLOR)
        Hp, Wp = pano.shape[:2]
        R_pano, _ = cv2.Rodrigues(np.array(shot["rotation"]))
        R = img.cam_from_world().rotation.matrix()
        d_world = dirs_by_cam[img.camera_id] @ R  # (R^T @ d)^T rows
        d_p = d_world @ R_pano.T
        lon = np.arctan2(d_p[..., 0], d_p[..., 2])
        lat = np.arcsin(np.clip(-d_p[..., 1], -1, 1))
        u = (lon / (2 * np.pi) + 0.5) * Wp - 0.5
        v = (0.5 - lat / np.pi) * Hp - 0.5
        out = cv2.remap(pano, u.astype(np.float32), v.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        cv2.imwrite(os.path.join(rdir, out_name), out)

    # gt + valid masks: reuse the gray-masked run's (identical split/order)
    gray_mode = None
    for it in ("15000",):
        cand = f"{GRAY_OUT}/{scene}_masked/test/{it}/opencv_fisheye"
        if os.path.isdir(cand):
            gray_mode = cand
    assert gray_mode, scene
    gtdir = os.path.join(mode_dir, "gt")
    if os.path.islink(gtdir) or os.path.isfile(gtdir):
        os.unlink(gtdir)
    elif os.path.isdir(gtdir):
        shutil.rmtree(gtdir)
    os.symlink(os.path.join(gray_mode, "gt"), gtdir)
    for m in ("valid_mask_cam1.png", "valid_mask_cam2.png", "valid_mask.png"):
        src = os.path.join(gray_mode, m)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(mode_dir, m))
    with open(os.path.join(mode_dir, "masks.json"), "w") as f:
        json.dump(masks_index, f, indent=1)

    n = len(test_names) - len(missing)
    print(f"[{scene}] resampled {n}/{len(test_names)} golden fisheye views -> {rdir}"
          + (f"  MISSING: {missing}" if missing else ""), flush=True)


if __name__ == "__main__":
    main()
