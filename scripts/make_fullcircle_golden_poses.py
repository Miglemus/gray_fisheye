#!/usr/bin/env python
"""Golden-test pano poses for FullCircle SPaGS renders.

For each scene, writes /workspace/dataset/fullcircle_pano/<scene>/golden_poses.json:
OpenSfM-style shots for every golden tripod capture (stem of camera1/<stem>_...png
test frames), in the SAME convention as the training reconstruction.json:
    R_pano_from_world = R_front @ R_cam1_from_world (rodrigues), t = R_front @ t_cam1
so nerficg/SPaGS can render ERP panos at the golden test positions.
"""
import json
import os
import sys

import cv2
import numpy as np

SRC_ROOT = "/workspace/dataset/fullcircle"
PANO_ROOT = "/workspace/dataset/fullcircle_pano"
RVEC_FRONT = np.array([-0.0371781, -0.00746628, 0.00398891])
SCENES = ["room1", "room2", "room3", "flat1", "flat2", "lab", "lounge", "dark", "persons"]


def main():
    import pycolmap
    R_front, _ = cv2.Rodrigues(RVEC_FRONT)
    for scene in sys.argv[1:] or SCENES:
        rec = pycolmap.Reconstruction(os.path.join(SRC_ROOT, scene, "sparse", "0"))
        by_name = {img.name: img for img in rec.images.values()}
        stems = sorted({os.path.splitext(os.path.basename(n))[0]
                        for n in by_name if "_test" in n})
        shots = {}
        for stem in stems:
            # Prefer camera1 (the frame the pano convention is defined in); fall back
            # to camera2 for captures whose camera1 half failed registration — the
            # pano pose only needs to be a consistent world-frame pose to resample
            # from, not literally camera1's.
            img = by_name.get(f"camera1/{stem}.png") or by_name.get(f"camera2/{stem}.png")
            if img is None:
                continue
            R1 = img.cam_from_world().rotation.matrix()
            t1 = np.asarray(img.cam_from_world().translation)
            rvec, _ = cv2.Rodrigues(R_front @ R1)
            shots[stem + ".png"] = {
                "rotation": [float(x) for x in rvec.ravel()],
                "translation": [float(x) for x in (R_front @ t1)],
                "camera": "insta360",
            }
        out = os.path.join(PANO_ROOT, scene, "golden_poses.json")
        with open(out, "w") as f:
            json.dump({"cameras": {"insta360": {"projection_type": "spherical",
                                                "width": 2048, "height": 1024}},
                       "shots": shots}, f, indent=1)
        print(f"[{scene}] {len(shots)} golden pano poses -> {out}", flush=True)


if __name__ == "__main__":
    main()
