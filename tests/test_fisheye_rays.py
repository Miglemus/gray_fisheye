import numpy as np
import torch
import cv2

from gray.imports import *
from gray.prelude import *
from gray.camera import CameraInfo


def test_fisheye_rays_match_opencv():
    W, H = 80, 60

    # * OPENCV_FISHEYE intrinsics (params already at render resolution)
    fx, fy = 95.0, 96.5
    cx, cy = W / 2 - 1.7, H / 2 + 2.3
    k1, k2, k3, k4 = -0.034688, 0.002964, -0.003554, 0.000424
    intrinsics = np.array([fx, fy, cx, cy, k1, k2, k3, k4], dtype=np.float64)

    # * Non-trivial proper rotation to exercise the world transform
    axis = np.array([0.3, -0.7, 0.5])
    axis = axis / np.linalg.norm(axis)
    angle = 0.6
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    origin = np.array([0.2, -0.1, 0.4])

    cam = CameraInfo(
        uid=0,
        R=R,
        T=np.zeros(3),
        origin=origin,
        fov_y=1.0,
        fov_x=1.0,
        image_path="",
        image_name="fisheye",
        image_width=W,
        image_height=H,
        is_test=False,
        model="opencv_fisheye",
        intrinsics=intrinsics,
    )

    # * Minimal raytracer with one gaussian so the BVH is valid; we only read primary rays
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
    cuda_dirs = fb.ray_direction.detach()[:H, :W].cpu().numpy().reshape(-1, 3).astype(np.float64)  # (N, 3)

    # * Pixel centers
    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    u_pix = us.ravel()
    v_pix = vs.ravel()

    # * Reference 1: OpenCV fisheye unprojection (sanity, OpenCV uses a fixed-point iteration)
    pix = np.stack([u_pix, v_pix], axis=-1).astype(np.float64)[:, None, :]
    Kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)
    undist = cv2.fisheye.undistortPoints(pix, Kmat, D).reshape(-1, 2)
    c2w_blender = -R.astype(np.float64).copy()
    c2w_blender[:, 0] *= -1
    cam_dir = np.stack([undist[:, 0], -undist[:, 1], -np.ones(undist.shape[0])], axis=-1)
    ref = cam_dir @ c2w_blender.T
    ref = ref / np.linalg.norm(ref, axis=-1, keepdims=True)
    ang_err = np.arccos(np.clip((cuda_dirs * ref).sum(-1), -1.0, 1.0))
    print(f"max angular error vs OpenCV: {float(np.max(ang_err)):.3e} rad")
    assert float(np.max(ang_err)) < 2e-3, "fisheye rays disagree with OpenCV unprojection"

    # * Reference 2 (gold standard): forward-project the CUDA bearing and check it reproduces the pixel
    # * Transform world direction back into the OpenCV camera frame
    gray_cam = cuda_dirs @ c2w_blender  # (N, 3), inverse of orthonormal c2w_blender
    ocv = np.stack([gray_cam[:, 0], -gray_cam[:, 1], -gray_cam[:, 2]], axis=-1)
    a = ocv[:, 0] / ocv[:, 2]
    b = ocv[:, 1] / ocv[:, 2]
    r = np.sqrt(a * a + b * b)
    theta = np.arctan(r)
    theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    scale = np.where(r > 1e-12, theta_d / r, 1.0)
    u_reproj = fx * (a * scale) + cx
    v_reproj = fy * (b * scale) + cy
    pix_err = np.sqrt((u_reproj - u_pix) ** 2 + (v_reproj - v_pix) ** 2)
    print(f"max reprojection error: {float(np.max(pix_err)):.3e} px")
    assert float(np.max(pix_err)) < 0.02, "fisheye rays do not reproject to their pixels"
