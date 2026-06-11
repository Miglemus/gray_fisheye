from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


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

    ``intrinsics`` are (fx, fy, cx, cy, k1..k4, p1, p2, sx1, sy1) scaled to the given image
    resolution. The baseline (``radius_scale == 1.0``) keeps pixels whose off-axis
    angle is below 90 deg, exactly matching the ray-tracer's cutoff. ``radius_scale``
    tunes the aggressivity: values < 1 shrink the valid disk radius (masking more of
    the vignetted rim), values > 1 grow it. The masked surface grows by roughly
    ``1 - radius_scale**2`` relative to the 90 deg disk.
    """
    fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1 = (float(v) for v in intrinsics)

    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # * Coordonnées cibles distordues sur le capteur (normalisées par la focale)
    u_target = (grid_x - cx) / fx
    v_target = (grid_y - cy) / fy

    # * Initialisation du point fixe (copie des cibles pour la vectorisation)
    u_final = u_target.clone()
    v_final = v_target.clone()

    MAX_ITERATIONS = 20

    for _ in range(MAX_ITERATIONS):
        r2 = u_final * u_final + v_final * v_final
        r = torch.sqrt(r2)
        
        # * Évite les divisions par zéro au centre optique (cx, cy)
        r_safe = torch.where(r > 0, r, torch.ones_like(r))
        
        theta = torch.atan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        
        theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        
        u_fe = torch.where(r > 0, (theta_d / r_safe) * u_final, u_final)
        v_fe = torch.where(r > 0, (theta_d / r_safe) * v_final, v_final)
        
        # * Application du modèle Thin Prism + Tangentiel complet (avec correction p2)
        u_estimated = u_fe + 2.0 * p1 * u_final * v_final + p2 * (r2 + 2.0 * u_final * u_final) + sx1 * r2
        v_estimated = v_fe + p1 * (r2 + 2.0 * v_final * v_final) + 2.0 * p2 * u_final * v_final + sy1 * r2

        # * Calcul du résidu et mise à jour
        delta_u = u_target - u_estimated
        delta_v = v_target - v_estimated
        
        u_final = u_final + delta_u
        v_final = v_final + delta_v

    # * Une fois que (u_final, v_final) ont convergé vers l'espace pinhole non-distordu,
    # * on extrait l'angle radial pur pour calculer son theta_d équivalent.
    r_final = torch.sqrt(u_final * u_final + v_final * v_final)
    theta_final = torch.atan(r_final)
    
    theta2_f = theta_final * theta_final
    theta4_f = theta2_f * theta2_f
    theta6_f = theta4_f * theta2_f
    theta8_f = theta4_f * theta4_f
    
    theta_d_final = theta_final * (1.0 + k1 * theta2_f + k2 * theta4_f + k3 * theta6_f + k4 * theta8_f)

    # * Seuil basé sur la fonction de coupure à 90° (ou selon le radius_scale)
    theta_d_max = radius_scale * _theta_d_at_max(k1, k2, k3, k4)
    mask = theta_d_final < theta_d_max
    
    # * Sécurité : si l'inversion a divergé mathématiquement à l'extérieur extrême du FOV,
    # * les valeurs deviennent NaN. On s'assure de les invalider (False).
    mask = mask & (~torch.isnan(theta_d_final))
    
    return mask


def build_fisheye_mask(cam, height: int, width: int, device, cfg) -> Optional[torch.Tensor]:
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
    if cam.model.lower() == "opencv_fisheye":
        return geometric_valid_mask_opencv_fisheye(intr, height, width, device, cfg.fisheye_mask_radius_scale)
    elif cam.model.lower() == "thin_prism_fisheye":
        return geometric_valid_mask_thin_prism_fisheye(intr, height, width, device, cfg.fisheye_mask_radius_scale)
    else:
        raise ValueError(f"Unsupported camera model: {cam.model}")
