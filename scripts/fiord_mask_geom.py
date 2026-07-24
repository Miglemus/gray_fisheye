"""Geometry utilities to move person masks between FIORD fisheye halves and
the stitched equirectangular panos (fiord_pano), using only colmap poses.

Pano frame convention (from gray/scripts/make_fiord_panos.py):
  - one pano per capture stem, rendered in the camera frame of the FIRST
    registered half (its cam_from_world rotation is stored in
    reconstruction.json shots[<stem>.png]['rotation'] as a rodrigues vector);
  - ERP dirs: x = cos(lat) sin(lon), y = -sin(lat), z = cos(lat) cos(lon),
    lon = (u - 0.5) * 2pi, lat = (0.5 - v) * pi.
Rotation-only transfer between halves of one capture (near-concentric lenses),
verified: RGB corr 0.996 (owner half) / 0.969 (opposite half) on kitchen_in.
"""

import json
import os

import cv2
import numpy as np

DOWNSCALE = 4  # images_4


def pano_dirs(w, h):
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    lon = (u - 0.5) * 2 * np.pi
    lat = (0.5 - v) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat),
                     np.cos(lat) * np.cos(lon)], -1)


def fisheye_dirs(w, h, params):
    """Unit bearing for every fisheye pixel via iterative OPENCV_FISHEYE inversion."""
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    D = np.array([k1, k2, k3, k4], np.float64)
    px = np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5), -1)
    und = cv2.fisheye.undistortPoints(
        px.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(h, w, 2)
    d = np.concatenate([und, np.ones((h, w, 1))], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def project_fisheye(d, params):
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    t2 = theta * theta
    td = theta * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
    s = np.where(rho > 1e-9, td / np.maximum(rho, 1e-9), 0.0)
    return ((fx * x * s + cx).astype(np.float32),
            (fy * y * s + cy).astype(np.float32), theta)


def dir_to_pano_uv(d, w, h):
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(-d[..., 1], -1, 1))
    u = (lon / (2 * np.pi) + 0.5) * w - 0.5
    v = (0.5 - lat / np.pi) * h - 0.5
    return u.astype(np.float32), v.astype(np.float32)


class CaptureIndex:
    """Per-scene index: capture stem -> pano shot rotation + registered images."""

    def __init__(self, scene, fiord_base="/workspace/dataset/fiord_baselines",
                 pano_root="/workspace/dataset/fiord_pano"):
        import pycolmap
        self.scene = scene
        self.base = os.path.join(fiord_base, scene)
        self.pano_dir = os.path.join(pano_root, scene)
        self.rec = pycolmap.Reconstruction(
            os.path.join(self.base, "distorted", "sparse", "0"))
        recon = json.load(open(os.path.join(self.pano_dir, "reconstruction.json")))[0]
        self.shots = recon["shots"]
        cam = recon["cameras"]["insta360"]
        self.pano_w, self.pano_h = cam["width"], cam["height"]
        self.params4 = {cid: c.params / np.array([DOWNSCALE] * 4 + [1] * 4)
                        for cid, c in self.rec.cameras.items()}
        self.by_stem = {}
        for img in self.rec.images.values():
            stem = os.path.splitext(os.path.basename(img.name)
                                    .replace("_fisheye1", "").replace("_fisheye2", ""))[0]
            self.by_stem.setdefault(stem, []).append(img)
        for v in self.by_stem.values():
            v.sort(key=lambda i: i.name)
        self._pdirs = None

    def stems(self):
        return sorted(s for s in self.by_stem if s + ".png" in self.shots)

    def R_pano(self, stem):
        R, _ = cv2.Rodrigues(np.array(self.shots[stem + ".png"]["rotation"]))
        return R

    def pano_mask_to_fisheye(self, stem, pano_mask, img):
        """Sample an ERP mask (uint8) into one registered fisheye image's frame."""
        params = self.params4[img.camera_id]
        cam = self.rec.cameras[img.camera_id]
        w, h = cam.width // DOWNSCALE, cam.height // DOWNSCALE
        d = fisheye_dirs(w, h, params)
        R_rel = self.R_pano(stem) @ img.cam_from_world().rotation.matrix().T
        d_pano = d @ R_rel.T
        u, v = dir_to_pano_uv(d_pano, self.pano_w, self.pano_h)
        return cv2.remap(pano_mask, u, v, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)

    def fisheye_mask_to_pano(self, stem, fisheye_masks):
        """Union of registered-half masks {image_name: uint8 mask} in ERP space."""
        if self._pdirs is None:
            self._pdirs = pano_dirs(self.pano_w, self.pano_h)
        out = np.zeros((self.pano_h, self.pano_w), np.float32)
        R_pano = self.R_pano(stem)
        for img in self.by_stem[stem]:
            m = fisheye_masks.get(os.path.basename(img.name))
            if m is None:
                continue
            params = self.params4[img.camera_id]
            R_rel = img.cam_from_world().rotation.matrix() @ R_pano.T
            d_img = self._pdirs @ R_rel.T
            u, v, theta = project_fisheye(d_img, params)
            samp = cv2.remap(m, u, v, cv2.INTER_LINEAR)
            inside = (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0]) & \
                     (theta < np.radians(97.0))
            out = np.maximum(out, samp * inside)
        return out.astype(np.uint8)
