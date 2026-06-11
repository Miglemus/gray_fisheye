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


def _thin_prism_unproject(u_pix, v_pix, params):
    fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1 = params
    u_target = (u_pix - cx) / fx
    v_target = (v_pix - cy) / fy
    u_final, v_final = u_target, v_target
    for _ in range(20):
        r2 = u_final * u_final + v_final * v_final
        r = np.sqrt(r2)
        if r > 0.0:
            theta = np.arctan(r)
            theta2 = theta * theta
            theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
            u_fe = (theta_d / r) * u_final
            v_fe = (theta_d / r) * v_final
        else:
            u_fe, v_fe = u_final, v_final
        u_est = u_fe + 2.0 * p1 * u_final * v_final + p2 * (r2 + 2.0 * u_final * u_final) + sx1 * r2
        v_est = v_fe + p1 * (r2 + 2.0 * v_final * v_final) + 2.0 * p2 * u_final * v_final + sy1 * r2
        u_final += u_target - u_est
        v_final += v_target - v_est

    max_theta = np.pi / 2
    r_final = np.sqrt(u_final * u_final + v_final * v_final)
    theta_final = np.arctan(r_final)
    t2 = max_theta * max_theta
    theta_d_max = max_theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    theta2_f = theta_final * theta_final
    theta_d_final = theta_final * (
        1.0 + k1 * theta2_f + k2 * theta2_f**2 + k3 * theta2_f**3 + k4 * theta2_f**4
    )
    if theta_final >= max_theta or theta_d_final >= theta_d_max:
        return np.zeros(3, dtype=np.float64)
    inv_norm = 1.0 / np.sqrt(u_final * u_final + v_final * v_final + 1.0)
    return np.array([u_final * inv_norm, v_final * inv_norm, inv_norm], dtype=np.float64)


def _thin_prism_project(cam_xyz, params):
    fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1 = params
    u = cam_xyz[0] / cam_xyz[2]
    v = cam_xyz[1] / cam_xyz[2]
    r2 = u * u + v * v
    r = np.sqrt(r2)
    if r > 0.0:
        theta = np.arctan(r)
        theta2 = theta * theta
        theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
        u_fe = (theta_d / r) * u
        v_fe = (theta_d / r) * v
    else:
        u_fe, v_fe = u, v
    u_dist = u_fe + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u) + sx1 * r2
    v_dist = v_fe + p1 * (r2 + 2.0 * v * v) + 2.0 * p2 * u * v + sy1 * r2
    return fx * u_dist + cx, fy * v_dist + cy


def test_thin_prism_fisheye_rays_match_reference():
    W, H = 80, 60
    fx, fy = 95.0, 96.5
    cx, cy = W / 2 - 1.7, H / 2 + 2.3
    k1, k2, k3, k4 = -0.034688, 0.002964, -0.003554, 0.000424
    p1, p2, sx1, sy1 = 1e-4, -2e-4, 3e-4, -1e-4
    intrinsics = np.array(
        [fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1], dtype=np.float64
    )

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
        image_name="thin_prism",
        image_width=W,
        image_height=H,
        is_test=False,
        model="thin_prism_fisheye",
        intrinsics=intrinsics,
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
    cuda_dirs = fb.ray_direction.detach()[:H, :W].cpu().numpy().reshape(-1, 3).astype(np.float64)

    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    u_pix = us.ravel()
    v_pix = vs.ravel()

    c2w_blender = -R.astype(np.float64).copy()
    c2w_blender[:, 0] *= -1

    ref_dirs = np.stack(
        [_thin_prism_unproject(u, v, intrinsics) for u, v in zip(u_pix, v_pix)], axis=0
    )
    valid = np.linalg.norm(ref_dirs, axis=-1) > 0.5
    ref_world = np.stack([d @ c2w_blender.T for d in ref_dirs[valid]], axis=0)
    ref_world = ref_world / np.linalg.norm(ref_world, axis=-1, keepdims=True)
    ang_err = np.arccos(np.clip((cuda_dirs[valid] * ref_world).sum(-1), -1.0, 1.0))
    print(f"max angular error vs Python TPF: {float(np.max(ang_err)):.3e} rad")
    assert float(np.max(ang_err)) < 2e-3, "thin prism fisheye rays disagree with Python unprojection"

    gray_cam = cuda_dirs[valid] @ c2w_blender
    ocv = np.stack([gray_cam[:, 0], -gray_cam[:, 1], -gray_cam[:, 2]], axis=-1)
    reproj = np.stack(
        [_thin_prism_project(d, intrinsics) for d in ocv], axis=0
    )
    pix_err = np.sqrt((reproj[:, 0] - u_pix[valid]) ** 2 + (reproj[:, 1] - v_pix[valid]) ** 2)
    print(f"max reprojection error: {float(np.max(pix_err)):.3e} px")
    assert float(np.max(pix_err)) < 0.05, "thin prism fisheye rays do not reproject to their pixels"
