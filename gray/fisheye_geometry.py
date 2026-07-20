from __future__ import annotations

import numpy as np


def opencv_fisheye_project(cam_xyz, params):
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    cam_xyz = np.asarray(cam_xyz, dtype=np.float64)
    u = cam_xyz[..., 0] / cam_xyz[..., 2]
    v = cam_xyz[..., 1] / cam_xyz[..., 2]
    r = np.sqrt(u * u + v * v)
    theta = np.arctan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
    scale = np.divide(theta_d, r, out=np.ones_like(theta), where=r > 0.0)
    return fx * (u * scale) + cx, fy * (v * scale) + cy


def opencv_fisheye_unproject(u_pix, v_pix, params, max_iters: int = 100, tolerance: float = 1e-10):
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    xd = (np.asarray(u_pix, dtype=np.float64) - cx) / fx
    yd = (np.asarray(v_pix, dtype=np.float64) - cy) / fy
    theta_d = np.sqrt(xd * xd + yd * yd)
    theta = theta_d.copy()
    converged = np.zeros_like(theta, dtype=bool)

    for _ in range(max_iters):
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        f = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) - theta_d
        df = 1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8
        step = np.divide(f, df, out=np.zeros_like(f), where=np.abs(df) > 1e-12)
        theta_next = theta - step
        converged |= np.abs(step) < tolerance
        theta = theta_next

    scale = np.divide(np.tan(theta), theta_d, out=np.ones_like(theta), where=theta_d > 0.0)
    x = xd * scale
    y = yd * scale
    inv_norm = 1.0 / np.sqrt(x * x + y * y + 1.0)
    dirs = np.stack([x * inv_norm, y * inv_norm, inv_norm], axis=-1)
    valid = converged & (theta < np.pi / 2) & np.isfinite(dirs).all(axis=-1)
    return dirs, valid


def thin_prism_distortion(u, v, extra):
    k1, k2, p1, p2, k3, k4, sx1, sy1 = extra
    r2 = u * u + v * v
    radial = k1 * r2 + k2 * r2**2 + k3 * r2**3 + k4 * r2**4
    du = u * radial + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u) + sx1 * r2
    dv = v * radial + 2.0 * p2 * u * v + p1 * (r2 + 2.0 * v * v) + sy1 * r2
    return du, dv


def thin_prism_project(cam_xyz, params):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
    extra = (k1, k2, p1, p2, k3, k4, sx1, sy1)
    cam_xyz = np.asarray(cam_xyz, dtype=np.float64)
    u = cam_xyz[..., 0] / cam_xyz[..., 2]
    v = cam_xyz[..., 1] / cam_xyz[..., 2]
    r = np.sqrt(u * u + v * v)
    theta = np.arctan(r)
    scale = np.divide(theta, r, out=np.ones_like(theta), where=r > 0.0)
    uu = scale * u
    vv = scale * v
    du, dv = thin_prism_distortion(uu, vv, extra)
    return fx * (uu + du) + cx, fy * (vv + dv) + cy


def thin_prism_unproject(u_pix, v_pix, params, max_iters: int = 100, tolerance: float = 1e-10):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
    extra = (k1, k2, p1, p2, k3, k4, sx1, sy1)
    uu0 = (np.asarray(u_pix, dtype=np.float64) - cx) / fx
    vv0 = (np.asarray(v_pix, dtype=np.float64) - cy) / fy
    uu = uu0.copy()
    vv = vv0.copy()
    converged = np.zeros_like(uu, dtype=bool)

    for _ in range(max_iters):
        du, dv = thin_prism_distortion(uu, vv, extra)
        next_uu = uu0 - du
        next_vv = vv0 - dv
        step_sq = (next_uu - uu) ** 2 + (next_vv - vv) ** 2
        converged |= step_sq < tolerance
        uu, vv = next_uu, next_vv

    theta = np.sqrt(uu * uu + vv * vv)
    theta_cos_theta = theta * np.cos(theta)
    scale = np.divide(np.sin(theta), theta_cos_theta, out=np.ones_like(theta), where=theta_cos_theta > 1e-8)
    x = uu * scale
    y = vv * scale
    inv_norm = 1.0 / np.sqrt(x * x + y * y + 1.0)
    dirs = np.stack([x * inv_norm, y * inv_norm, inv_norm], axis=-1)
    valid = converged & (theta < np.pi / 2) & np.isfinite(dirs).all(axis=-1)
    return dirs, valid


def rad_tan_thin_prism_distortion(u, v, extra):
    k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 = extra
    theta2 = u * u + v * v
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta10 = theta8 * theta2
    theta12 = theta6 * theta6
    th_radial = 1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8 + k4 * theta10 + k5 * theta12

    x = th_radial * u
    y = th_radial * v
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    r4 = r2 * r2

    dx_tang = 2.0 * p1 * xy + p0 * (r2 + 2.0 * x2)
    dy_tang = 2.0 * p0 * xy + p1 * (r2 + 2.0 * y2)
    dx_tp = s0 * r2 + s1 * r4
    dy_tp = s2 * r2 + s3 * r4

    du = x + dx_tang + dx_tp - u
    dv = y + dy_tang + dy_tp - v
    return du, dv


def rad_tan_thin_prism_project(cam_xyz, params):
    fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 = params
    extra = (k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3)
    cam_xyz = np.asarray(cam_xyz, dtype=np.float64)
    u = cam_xyz[..., 0] / cam_xyz[..., 2]
    v = cam_xyz[..., 1] / cam_xyz[..., 2]
    r = np.sqrt(u * u + v * v)
    theta = np.arctan(r)
    scale = np.divide(theta, r, out=np.ones_like(theta), where=r > 0.0)
    uu = scale * u
    vv = scale * v
    du, dv = rad_tan_thin_prism_distortion(uu, vv, extra)
    return fx * (uu + du) + cx, fy * (vv + dv) + cy


def rad_tan_thin_prism_unproject(u_pix, v_pix, params, max_iters: int = 100, tolerance: float = 1e-10):
    fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 = params
    extra = (k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3)
    uu0 = (np.asarray(u_pix, dtype=np.float64) - cx) / fx
    vv0 = (np.asarray(v_pix, dtype=np.float64) - cy) / fy
    uu = uu0.copy()
    vv = vv0.copy()
    converged = np.zeros_like(uu, dtype=bool)

    for _ in range(max_iters):
        du, dv = rad_tan_thin_prism_distortion(uu, vv, extra)
        next_uu = uu0 - du
        next_vv = vv0 - dv
        step_sq = (next_uu - uu) ** 2 + (next_vv - vv) ** 2
        converged |= step_sq < tolerance
        uu, vv = next_uu, next_vv

    theta = np.sqrt(uu * uu + vv * vv)
    theta_cos_theta = theta * np.cos(theta)
    scale = np.divide(np.sin(theta), theta_cos_theta, out=np.ones_like(theta), where=theta_cos_theta > 1e-8)
    x = uu * scale
    y = vv * scale
    inv_norm = 1.0 / np.sqrt(x * x + y * y + 1.0)
    dirs = np.stack([x * inv_norm, y * inv_norm, inv_norm], axis=-1)
    valid = converged & (theta < np.pi / 2) & np.isfinite(dirs).all(axis=-1)
    return dirs, valid


def project_fisheye(model: str, cam_xyz, params):
    if model == "OPENCV_FISHEYE":
        return opencv_fisheye_project(cam_xyz, params)
    if model == "THIN_PRISM_FISHEYE":
        return thin_prism_project(cam_xyz, params)
    if model == "RAD_TAN_THIN_PRISM_FISHEYE":
        return rad_tan_thin_prism_project(cam_xyz, params)
    raise ValueError(f"Unsupported custom undistort source camera model: {model}")


def unproject_fisheye(model: str, u_pix, v_pix, params):
    if model == "OPENCV_FISHEYE":
        return opencv_fisheye_unproject(u_pix, v_pix, params)
    if model == "THIN_PRISM_FISHEYE":
        return thin_prism_unproject(u_pix, v_pix, params)
    if model == "RAD_TAN_THIN_PRISM_FISHEYE":
        return rad_tan_thin_prism_unproject(u_pix, v_pix, params)
    raise ValueError(f"Unsupported custom undistort source camera model: {model}")
