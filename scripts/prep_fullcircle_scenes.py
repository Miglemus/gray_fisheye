#!/usr/bin/env python
"""Prepare FullCircle scenes (theialab, HF youlenda/FullCircle) for gray.

Source layout (per scene, /workspace/dataset/fullcircle/<scene>/):
    images/camera{1,2}/frame_NNNN[_test].png       2880x2880 dual fisheye
    images_4/camera{1,2}/...                       720x720 (provided x4 downscale)
    masks/masks-{1,20}/camera{1,2}/frame_NNNN[_test]_mask.png   255 = person
    masks_4/masks-{1,20}_4/camera{1,2}/...         720x720
    omni/{images,masks}/frame_NNNN[_mask].png      2880x5760 ERP panos (train frames only)
    sparse/0/{cameras,images,points3D}.bin         OPENCV_FISHEYE, names camera{1,2}/...

Dest (gray-ready, mirrors fiord_baselines layout):
    /workspace/dataset/fullcircle_baselines/<scene>/
        distorted/sparse/0/{cameras,images,points3D}.bin
        distorted/sparse/0/test.txt              names of *_test frames (golden split)
        input_4/camera{1,2} -> symlinks into source images_4
        person_masks_4/camera{1,2}/frame_NNNN.png -> symlinks to masks_4/masks-20_4 (renamed stem)
        point_cloud.safetensors                   gray init cloud from colmap points
        valid_mask_cam{1,2}.png                    r=0.95 disk masks @720px
"""

import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/gray/worktrees/person-masks")
sys.path.insert(0, "/workspace/gray/scripts")
from prep_fiord_scenes import export_point_cloud, write_masks  # noqa: E402

SRC_ROOT = "/workspace/dataset/fullcircle"
DEST_ROOT = "/workspace/dataset/fullcircle_baselines"
MASK_VARIANT = "masks-20_4"  # 20px dilation at full res, x4-downscaled
DOWNSCALE = 4


def prep_scene(scene):
    import pycolmap

    src = os.path.join(SRC_ROOT, scene)
    dest = os.path.join(DEST_ROOT, scene)
    os.makedirs(dest, exist_ok=True)

    sparse_src = os.path.join(src, "sparse", "0")
    sparse_dst = os.path.join(dest, "distorted", "sparse", "0")
    os.makedirs(sparse_dst, exist_ok=True)
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        dst = os.path.join(sparse_dst, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(sparse_src, f), dst)

    rec = pycolmap.Reconstruction(sparse_dst)
    models = {cam.model.name for cam in rec.cameras.values()}
    assert models == {"OPENCV_FISHEYE"}, models

    names = sorted(img.name for img in rec.images.values())
    test_names = [n for n in names if "_test" in n]
    with open(os.path.join(sparse_dst, "test.txt"), "w") as f:
        f.write("\n".join(test_names) + "\n")

    # images: symlink the provided x4 camera dirs
    input_dir = os.path.join(dest, f"input_{DOWNSCALE}")
    os.makedirs(input_dir, exist_ok=True)
    for cam in ("camera1", "camera2"):
        link = os.path.join(input_dir, cam)
        target = os.path.join(src, f"images_{DOWNSCALE}", cam)
        assert os.path.isdir(target), target
        if not os.path.islink(link):
            os.symlink(target, link)

    missing = [n for n in names
               if not os.path.exists(os.path.join(input_dir, n))]
    if missing:
        raise RuntimeError(f"{scene}: {len(missing)} colmap names missing under "
                           f"{input_dir}, e.g. {missing[:3]}")

    # person masks: symlink with the stem gray expects (<image stem>.png)
    n_masks = 0
    for cam in ("camera1", "camera2"):
        mdir = os.path.join(src, f"masks_{DOWNSCALE}", MASK_VARIANT, cam)
        out = os.path.join(dest, f"person_masks_{DOWNSCALE}", cam)
        os.makedirs(out, exist_ok=True)
        for m in sorted(os.listdir(mdir)):
            assert m.endswith("_mask.png"), m
            link = os.path.join(out, m.replace("_mask.png", ".png"))
            if not os.path.islink(link):
                os.symlink(os.path.join(mdir, m), link)
            n_masks += 1

    pc_path = os.path.join(dest, "point_cloud.safetensors")
    n_pts = export_point_cloud(rec, pc_path) if not os.path.exists(pc_path) else "cached"

    sample = Image.open(os.path.join(input_dir, names[0]))
    fracs = write_masks(rec, dest, sample.size[0], scene)

    print(f"[{scene}] n_colmap={len(names)} test={len(test_names)} masks={n_masks} "
          f"({sample.size[0]}px) points3D={n_pts} valid_mask_frac={fracs}", flush=True)


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
