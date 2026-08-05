"""Render a smooth fly-through video that retraces the capture trajectory.

The scene photos were shot continuously, so sorting the training cameras by
filename recovers the order in which they were taken. This script:

  1. orders the cameras by capture order (filename),
  2. smooths their poses to remove hand-held jitter (Gaussian on positions +
     quaternion pre-smoothing on rotations),
  3. fits an arc-length-parameterised cubic spline through the positions and a
     Slerp through the rotations, then samples both at a constant speed to N
     frames = duration * 30fps,
  4. renders each interpolated pose through gray's raytracer in the requested
     projection (rttpf fisheye OR pinhole), and
  5. encodes a smooth 30fps mp4.

Only gray's public raytracer/camera API is used — no CUDA changes.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, "/workspace/gray")
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from render import (  # noqa: E402
    load_config,
    load_render_views,
    model_dir_from_path,
    resolve_checkpoint,
    RenderCLI,
)
from gray.camera import CameraInfo  # noqa: E402
from gray.camera_models import GrayCameraModelClass  # noqa: E402
from gray.prelude import Raytracer  # noqa: E402


@dataclasses.dataclass
class CLI:
    model_path: str  # trained gray model dir
    out: str  # output mp4 path
    projection: str = "rttpf"  # "rttpf" (fisheye) or "pinhole"
    fps: int = 30
    seconds: float = 45.0  # video duration; frames = seconds * fps
    pinhole_hfov_deg: float = 90.0  # horizontal FOV for pinhole projection
    pinhole_width: int = 1600
    pinhole_height: int = 900
    pos_smooth: float = 2.0  # Gaussian sigma (in keyframes) for position de-jitter
    rot_smooth: int = 2  # half-window for quaternion pre-smoothing (0 disables)
    spline_smooth: float = 0.0  # splprep smoothing factor (0 = interpolate)
    znear: float = 0.01


def ordered_pose_cameras(model_dir: Path, cfg):
    """All cameras (train+test) in capture order, from the rttpf training frame."""
    mode = GrayCameraModelClass("rad_tan_thin_prism_fisheye")
    views = load_render_views(
        mode, cfg=cfg, cli_intrinsics=None, model_dir=model_dir, load_images=False
    )
    cams = list(views.train_cameras or []) + list(views.test_cameras or [])
    cams.sort(key=lambda c: c.image_name)  # IMG_5693, 5694, ... == capture order
    return cams


def smooth_quaternions(quats: np.ndarray, half_window: int) -> np.ndarray:
    """Enforce sign continuity, then optionally average each quat with neighbours."""
    q = quats.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:  # antipodal → flip so slerp path is short
            q[i] = -q[i]
    if half_window <= 0:
        return q
    out = np.empty_like(q)
    for i in range(len(q)):
        lo, hi = max(0, i - half_window), min(len(q), i + half_window + 1)
        m = q[lo:hi].mean(axis=0)
        out[i] = m / np.linalg.norm(m)
    return out


def build_trajectory(cams, cli: CLI):
    """Return (positions[M,3], rotations[M] as Rotation) sampled at constant speed."""
    origins = np.array([np.asarray(c.origin, dtype=np.float64) for c in cams])  # [N,3]
    Rs = np.array([np.asarray(c.R, dtype=np.float64) for c in cams])  # [N,3,3] world->cam

    # de-jitter
    origins_s = np.stack(
        [gaussian_filter1d(origins[:, k], cli.pos_smooth, mode="nearest") for k in range(3)],
        axis=1,
    )
    quats = Rotation.from_matrix(Rs).as_quat()  # [N,4] xyzw
    quats = smooth_quaternions(quats, cli.rot_smooth)

    # cubic spline through positions; u in [0,1] over keyframes
    tck, u = splprep(origins_s.T, u=None, s=cli.spline_smooth, k=min(3, len(cams) - 1))
    # arc-length reparam for constant visual speed
    fine = np.linspace(0, 1, 20000)
    pts = np.array(splev(fine, tck)).T
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    M = int(round(cli.seconds * cli.fps))
    targets = np.linspace(0, arc[-1], M)
    u_samples = np.interp(targets, arc, fine)  # keyframe-param for each frame

    positions = np.array(splev(u_samples, tck)).T  # [M,3]
    slerp = Slerp(u, Rotation.from_quat(quats))
    rotations = slerp(np.clip(u_samples, u[0], u[-1]))  # [M] Rotation (world->cam)
    return positions, rotations


def make_camera(template: CameraInfo, R: np.ndarray, origin: np.ndarray,
                cli: CLI, uid: int) -> CameraInfo:
    T = (-R.T @ origin).astype(np.float64)  # origin = -R@T  ⇒  T = -Rᵀ·origin
    if cli.projection == "pinhole":
        hfov = np.deg2rad(cli.pinhole_hfov_deg)
        vfov = 2 * np.arctan(np.tan(hfov / 2) * cli.pinhole_height / cli.pinhole_width)
        return CameraInfo(
            uid=uid, R=R.astype(np.float64), T=T, origin=origin.astype(np.float64),
            fov_y=np.float64(vfov), fov_x=np.float64(hfov),
            image_path="", image_name=f"{uid:05d}.png",
            image_width=cli.pinhole_width, image_height=cli.pinhole_height,
            is_test=False, model="pinhole", intrinsics=None,
        )
    # rttpf: keep the template's fisheye intrinsics/size, swap pose only
    return dataclasses.replace(
        template, uid=uid, R=R.astype(np.float64), T=T,
        origin=origin.astype(np.float64), image_name=f"{uid:05d}.png",
    )


def main():
    cli = tyro.cli(CLI)
    model_dir = model_dir_from_path(cli.model_path)
    cfg = load_config(model_dir, [])
    dummy = RenderCLI(model_path=cli.model_path)
    iteration, checkpoint_path = resolve_checkpoint(dummy, cfg, model_dir)

    cams = ordered_pose_cameras(model_dir, cfg)
    print(f"{len(cams)} cameras in capture order ({cams[0].image_name} → {cams[-1].image_name})")
    template = cams[0]

    if cli.projection == "pinhole":
        W, H = cli.pinhole_width, cli.pinhole_height
    else:
        W, H = template.image_width, template.image_height
    print(f"projection={cli.projection}  render {W}x{H}  iteration {iteration}")

    raytracer = Raytracer.from_safetensors(cfg, checkpoint_path, W, H, inference_only=True)
    raytracer.set_render_resolution(W, H)

    positions, rotations = build_trajectory(cams, cli)
    print(f"rendering {len(positions)} frames → {cli.seconds:.0f}s @ {cli.fps}fps")

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(cli.out, fps=cli.fps, codec="libx264", quality=8,
                                macro_block_size=None, ffmpeg_log_level="error")
    with torch.no_grad():
        for i in range(len(positions)):
            cam = make_camera(template, rotations[i].as_matrix(), positions[i], cli, i)
            img = raytracer(cam, znear=cli.znear).clamp(0, 1)  # [3,H,W]
            frame = (img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            writer.append_data(frame)
            if i % 100 == 0:
                print(f"  frame {i}/{len(positions)}", flush=True)
    writer.close()
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
