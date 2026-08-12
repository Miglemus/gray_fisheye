"""Is the residual distortion visible in the *data*, with gray taken out of the loop?

mip-NeRF 360 ships `sparse/0` already undistorted: PINHOLE cameras, and 2-D observations in
undistorted pixel coordinates. So we can reproject every track with COLMAP's own pose and
intrinsics and look at the residual as a function of image radius. Bundle adjustment removes
whatever its model could express; a *systematic* radial trend that survives means the
undistorted images are not actually rectilinear -- higher-order distortion the original
camera model could not capture.

This is an independent check on the two mechanisms found in the learned camera:

  * an fx-vs-fy error is gray's alone (COLMAP fits both focals), so it must NOT show up here;
  * leftover radial distortion is a property of the dataset, so it MUST show up here.

Outputs tmp/mipnerf360/analysis/sfm_residual.json.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gray.colmap import (  # noqa: E402
    qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary, read_next_bytes,
)


def read_points3D_with_ids(path):
    "gray's own reader drops the COLMAP point ids, which are exactly what we need to join on."
    out = {}
    with open(path, "rb") as handle:
        for _ in range(read_next_bytes(handle, 8, "Q")[0]):
            fields = read_next_bytes(handle, num_bytes=43, format_char_sequence="QdddBBBd")
            out[int(fields[0])] = np.array(fields[1:4], dtype=np.float64)
            length = read_next_bytes(handle, num_bytes=8, format_char_sequence="Q")[0]
            read_next_bytes(handle, num_bytes=8 * length, format_char_sequence="ii" * length)
    return out

ROOT = "/workspace/gray/worktrees/noncentral-camera"
OUT = f"{ROOT}/tmp/mipnerf360/analysis/sfm_residual.json"
SCENES = ["bicycle", "garden", "stump", "bonsai", "counter", "kitchen", "room"]
BINS = 12


def main():
    payload = {}
    print(f"{'scene':9s} {'n_obs':>9s} {'rms px':>7s} | radial mean residual (px), centre -> edge")
    for scene in SCENES:
        path = f"{ROOT}/data/360_v2/{scene}/sparse/0"
        cam = next(iter(read_intrinsics_binary(f"{path}/cameras.bin").values()))
        fx, fy, cx, cy = (float(v) for v in cam.params[:4])
        images = read_extrinsics_binary(f"{path}/images.bin")
        xyz = read_points3D_with_ids(f"{path}/points3D.bin")

        radii, d_rad, d_tan, d_x, d_y = [], [], [], [], []
        for image in images.values():
            rotation = qvec2rotmat(image.qvec)
            tvec = np.asarray(image.tvec, dtype=np.float64)
            ids = np.asarray(image.point3D_ids)
            keep = ids > -1
            if not keep.any():
                continue
            observed = np.asarray(image.xys, dtype=np.float64)[keep]
            world = np.stack([xyz[i] for i in ids[keep]])
            camera_pts = world @ rotation.T + tvec
            in_front = camera_pts[:, 2] > 1e-6
            camera_pts, observed = camera_pts[in_front], observed[in_front]
            projected = np.stack([
                fx * camera_pts[:, 0] / camera_pts[:, 2] + cx,
                fy * camera_pts[:, 1] / camera_pts[:, 2] + cy,
            ], -1)
            delta = observed - projected
            offset = observed - np.array([cx, cy])
            radius = np.linalg.norm(offset, axis=-1)
            unit = offset / np.maximum(radius, 1e-9)[:, None]
            radii.append(radius)
            d_rad.append((delta * unit).sum(-1))
            d_tan.append(delta[:, 0] * -unit[:, 1] + delta[:, 1] * unit[:, 0])
            d_x.append(delta[:, 0])
            d_y.append(delta[:, 1])

        radius = np.concatenate(radii)
        radial = np.concatenate(d_rad)
        tangential = np.concatenate(d_tan)
        dx, dy = np.concatenate(d_x), np.concatenate(d_y)
        # * Robust: outlier tracks would swamp a plain mean.
        keep = np.abs(radial) < 5.0
        edges = np.linspace(0.0, radius.max(), BINS + 1)
        which = np.clip(np.digitize(radius, edges) - 1, 0, BINS - 1)
        profile = [float(np.median(radial[keep & (which == b)])) for b in range(BINS)]
        centres = [float((edges[b] + edges[b + 1]) / 2) for b in range(BINS)]

        payload[scene] = {
            "n_obs": int(radius.size),
            "rms_px": float(np.sqrt((radial[keep] ** 2).mean())),
            "radius": centres,
            "radial_median": profile,
            "tangential_median": [
                float(np.median(tangential[keep & (which == b)])) for b in range(BINS)
            ],
            "r_max": float(radius.max()),
            "median_dx_by_x": None,
            "fx": fx, "fy": fy,
        }
        # * An fx error would show as a median dx growing linearly with (x - cx); measure it.
        x_offset = np.concatenate([
            np.asarray(im.xys, dtype=np.float64)[np.asarray(im.point3D_ids) > -1][:, 0] - cx
            for im in images.values() if (np.asarray(im.point3D_ids) > -1).any()
        ])
        x_offset = x_offset[: dx.size]
        bins_x = np.clip(np.digitize(x_offset, np.linspace(-cx, cx, BINS + 1)) - 1, 0, BINS - 1)
        payload[scene]["median_dx_by_x"] = [
            float(np.median(dx[keep & (bins_x == b)])) if (keep & (bins_x == b)).any() else 0.0
            for b in range(BINS)
        ]
        row = " ".join(f"{v:+5.2f}" for v in profile)
        print(f"{scene:9s} {radius.size:9d} {payload[scene]['rms_px']:7.3f} | {row}")

    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
