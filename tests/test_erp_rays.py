import numpy as np
import torch

from gray.imports import *
from gray.prelude import *
from gray.camera import CameraInfo


def _equirectangular_unproject(u_pix, v_pix, width, height):
    """Reference ERP unprojection, in the OpenCV camera frame (x right, y down, z forward)."""
    lon = (u_pix / width - 0.5) * 2.0 * np.pi
    lat = (0.5 - v_pix / height) * np.pi
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], axis=-1)


def _erp_rays(W, H, R, origin):
    """Render one frame with the equirectangular model and return the CUDA primary rays."""
    cam = CameraInfo(
        uid=0,
        R=R,
        T=np.zeros(3),
        origin=origin,
        fov_y=1.0,
        fov_x=1.0,
        image_path="",
        image_name="equirectangular",
        image_width=W,
        image_height=H,
        is_test=False,
        model="equirectangular",
        intrinsics=None,
    )

    cfg = RaytracerConfig(sh=False)
    raytracer = Raytracer(cfg, 1, W, H)
    gaussians = raytracer.cuda_module.get_gaussians()
    gaussians.mean.copy_(torch.tensor([[0.0, 0.0, -2.0]], device="cuda"))
    gaussians.scale.copy_(torch.tensor([[1.0, 1.0, 1.0]], device="cuda").log())
    gaussians.channels.copy_(torch.tensor([[1.0, 1.0, 1.0]], device="cuda"))
    gaussians.rotation.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
    gaussians.opacity.copy_(torch.tensor([[0.5]], device="cuda").logit())
    torch.cuda.synchronize()
    raytracer.cuda_module.rebuild_bvh()

    config = raytracer.cuda_module.get_config()
    config.needs_ray_output.fill_(True)
    config.jitter_primary_rays.fill_(False)

    with torch.no_grad():
        raytracer(cam)

    fb = raytracer.cuda_module.get_framebuffer()
    return fb.ray_direction.detach()[:H, :W].cpu().numpy().reshape(-1, 3).astype(np.float64)


def _rotation():
    # * Non-trivial proper rotation to exercise the world transform
    axis = np.array([0.3, -0.7, 0.5])
    axis = axis / np.linalg.norm(axis)
    angle = 0.6
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def test_equirectangular_rays_match_reference():
    W, H = 80, 40
    R = _rotation()
    cuda_dirs = _erp_rays(W, H, R, np.array([0.2, -0.1, 0.4]))

    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    u_pix, v_pix = us.ravel(), vs.ravel()

    c2w_blender = -R.astype(np.float64).copy()
    c2w_blender[:, 0] *= -1

    ref = _equirectangular_unproject(u_pix, v_pix, W, H)
    ref_gray = ref.copy()
    ref_gray[:, 1] *= -1.0
    ref_gray[:, 2] *= -1.0
    ref_world = ref_gray @ c2w_blender.T
    ref_world = ref_world / np.linalg.norm(ref_world, axis=-1, keepdims=True)

    ang_err = np.arccos(np.clip((cuda_dirs * ref_world).sum(-1), -1.0, 1.0))
    print(f"max angular error vs Python ERP: {float(np.max(ang_err)):.3e} rad")
    assert float(np.max(ang_err)) < 2e-3, "equirectangular rays disagree with the reference"

    # * Gold standard: forward-project the CUDA bearing and check it lands back on its pixel.
    gray_cam = cuda_dirs @ c2w_blender
    ocv = np.stack([gray_cam[:, 0], -gray_cam[:, 1], -gray_cam[:, 2]], axis=-1)
    lon = np.arctan2(ocv[:, 0], ocv[:, 2])
    lat = np.arcsin(np.clip(-ocv[:, 1], -1.0, 1.0))
    u_reproj = (lon / (2.0 * np.pi) + 0.5) * W
    v_reproj = (0.5 - lat / np.pi) * H
    pix_err = np.sqrt((u_reproj - u_pix) ** 2 + (v_reproj - v_pix) ** 2)
    print(f"max reprojection error: {float(np.max(pix_err)):.3e} px")
    assert float(np.max(pix_err)) < 0.02, "equirectangular rays do not reproject to their pixels"

    norms = np.linalg.norm(cuda_dirs, axis=-1)
    assert np.all(np.abs(norms - 1.0) < 1e-5), "equirectangular rays are not unit length"


def test_equirectangular_covers_the_full_sphere_without_a_seam():
    """The properties that make ERP different from every other model in gray."""
    W, H = 64, 32
    R = _rotation()
    dirs = _erp_rays(W, H, R, np.zeros(3)).reshape(H, W, 3)

    # * No pixel is ever inactive: fisheye models return a zero direction outside their FOV.
    assert np.all(np.linalg.norm(dirs, axis=-1) > 0.5), (
        "equirectangular must have no invalid pixels"
    )

    # * Full 4-pi coverage: some ray points opposite to any given ray, which no other model does.
    forward = dirs[H // 2, W // 2]
    cosines = dirs.reshape(-1, 3) @ forward
    assert cosines.min() < -0.99, "equirectangular does not look backwards"
    assert cosines.max() > 0.99, "equirectangular does not cover its own forward axis"

    # * No seam: the +/-180 deg wrap is continuous, so the first and last columns are separated
    # * by exactly the W-1 columns of longitude between them, on their own parallel. That angle
    # * is latitude dependent -- two points a fixed longitude apart converge towards the poles.
    wrap = np.arccos(np.clip((dirs[:, 0] * dirs[:, -1]).sum(-1), -1.0, 1.0))
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    delta_lon = 2.0 * np.pi / W  # * the wrap-around gap, i.e. one column
    expected = np.arccos(np.sin(lat) ** 2 + np.cos(lat) ** 2 * np.cos(delta_lon))
    assert np.all(np.abs(wrap - expected) < 1e-3), "seam is discontinuous"

    # * Poles are single points, not a singularity: the top row all maps near the zenith.
    top = dirs[0]
    assert np.all((top @ top[0]) > np.cos(2.0 * np.pi / H)), "top row does not converge to a pole"
