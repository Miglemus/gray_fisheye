from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from gray.camera import CameraInfo


# * Off-axis angle bounding the OpenCV fisheye imaged disk; pixels past this lie
# * outside the lens' valid field of view (matches cuda/core/fisheye.cuh).
FISHEYE_MAX_THETA = math.pi / 2  # 90 degrees


def _theta_d_at_max(k1: float, k2: float, k3: float, k4: float) -> float:
    """Distorted radius theta_d corresponding to theta = 90 deg for the given distortion coeffs."""
    t = FISHEYE_MAX_THETA
    t2 = t * t
    return t * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)


def intrinsics_at_resolution(cam, height: int, width: int) -> np.ndarray:
    """Fisheye intrinsics (fx, fy, cx, cy, k1..k4) scaled to ``height`` x ``width``."""
    intr = cam.intrinsics.copy()
    scale_x = width / cam.image_width
    scale_y = height / cam.image_height
    intr[0] *= scale_x  # fx
    intr[2] *= scale_x  # cx
    intr[1] *= scale_y  # fy
    intr[3] *= scale_y  # cy
    return intr


def geometric_valid_mask_opencv_fisheye(intrinsics, height: int, width: int, device, radius_scale: float = 1.0):
    """Boolean [H, W] mask of pixels inside the fisheye lens disk.

    ``intrinsics`` are (fx, fy, cx, cy, k1, k2, k3, k4) scaled to the given image
    resolution. The baseline (``radius_scale == 1.0``) keeps pixels whose off-axis
    angle is below 90 deg, exactly matching the ray-tracer's cutoff. ``radius_scale``
    tunes the aggressivity: values < 1 shrink the valid disk radius (masking more of
    the vignetted rim), values > 1 grow it. The masked surface grows by roughly
    ``1 - radius_scale**2`` relative to the 90 deg disk.
    """
    fx, fy, cx, cy, k1, k2, k3, k4 = (float(v) for v in intrinsics)

    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # * Distorted normalized radius (theta_d); monotonic in theta over the valid range,
    # * so thresholding theta_d is equivalent to thresholding theta but stays in pixel space.
    xd = (grid_x - cx) / fx
    yd = (grid_y - cy) / fy
    theta_d = torch.sqrt(xd * xd + yd * yd)

    theta_d_max = radius_scale * _theta_d_at_max(k1, k2, k3, k4)
    return theta_d < theta_d_max


def geometric_valid_mask_thin_prism_fisheye(intrinsics, height: int, width: int, device, radius_scale: float = 1.0):
    """Boolean [H, W] mask of pixels inside the thin prism fisheye lens disk.

    ``intrinsics`` are (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1) scaled to the
    given image resolution. The baseline (``radius_scale == 1.0``) keeps pixels whose
    equidistant fisheye angle is below 90 deg, matching the ray-tracer cutoff.
    """
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = (float(v) for v in intrinsics)

    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    uu0 = (grid_x - cx) / fx
    vv0 = (grid_y - cy) / fy
    uu = uu0.clone()
    vv = vv0.clone()

    for _ in range(100):
        r2 = uu * uu + vv * vv
        radial = k1 * r2 + k2 * r2**2 + k3 * r2**3 + k4 * r2**4
        du = uu * radial + 2.0 * p1 * uu * vv + p2 * (r2 + 2.0 * uu * uu) + sx1 * r2
        dv = vv * radial + 2.0 * p2 * uu * vv + p1 * (r2 + 2.0 * vv * vv) + sy1 * r2
        uu = uu0 - du
        vv = vv0 - dv

    theta = torch.sqrt(uu * uu + vv * vv)
    max_theta = radius_scale * (math.pi / 2)
    mask = theta < max_theta
    mask = mask & (~torch.isnan(theta))
    return mask


def geometric_valid_mask_rad_tan_thin_prism_fisheye(
    intrinsics, height: int, width: int, device, radius_scale: float = 1.0
):
    """Boolean [H, W] mask for COLMAP RAD_TAN_THIN_PRISM_FISHEYE."""
    fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 = (float(v) for v in intrinsics)

    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    uu0 = (grid_x - cx) / fx
    vv0 = (grid_y - cy) / fy
    uu = uu0.clone()
    vv = vv0.clone()

    for _ in range(100):
        theta2 = uu * uu + vv * vv
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        theta10 = theta8 * theta2
        theta12 = theta6 * theta6
        th_radial = 1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8 + k4 * theta10 + k5 * theta12

        x = th_radial * uu
        y = th_radial * vv
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        r4 = r2 * r2

        dx_tang = 2.0 * p1 * xy + p0 * (r2 + 2.0 * x2)
        dy_tang = 2.0 * p0 * xy + p1 * (r2 + 2.0 * y2)
        dx_tp = s0 * r2 + s1 * r4
        dy_tp = s2 * r2 + s3 * r4

        uu = uu0 - (x + dx_tang + dx_tp - uu)
        vv = vv0 - (y + dy_tang + dy_tp - vv)

    theta = torch.sqrt(uu * uu + vv * vv)
    max_theta = radius_scale * (math.pi / 2)
    mask = theta < max_theta
    mask = mask & (~torch.isnan(theta))
    return mask


def build_fisheye_mask(cam: CameraInfo, height: int, width: int, device, cfg) -> Optional[torch.Tensor]:
    """Build the shared radial fisheye validity mask as a [H, W] bool tensor.

    One mask suffices for every view when intrinsics and resolution are shared.
    Returns ``None`` when fisheye masking is disabled.
    """
    if not cfg.fisheye_mask_geometric:
        return None

    if cam.intrinsics is None:
        raise ValueError(
            "fisheye_mask_geometric requires fisheye intrinsics on the camera"
        )

    intr = intrinsics_at_resolution(cam, height, width)
    from gray.camera_models import normalize_gray_model

    model = normalize_gray_model(cam.model)
    if model == "opencv_fisheye":
        return geometric_valid_mask_opencv_fisheye(intr, height, width, device, cfg.fisheye_mask_radius_scale)
    elif model == "thin_prism_fisheye":
        return geometric_valid_mask_thin_prism_fisheye(intr, height, width, device, cfg.fisheye_mask_radius_scale)
    elif model == "rad_tan_thin_prism_fisheye":
        return geometric_valid_mask_rad_tan_thin_prism_fisheye(
            intr, height, width, device, cfg.fisheye_mask_radius_scale
        )
    else:
        raise ValueError(f"Unsupported camera model: {cam.model}")
