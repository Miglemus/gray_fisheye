"""Depth-peeling diagnostic: re-render one view from a trained gray model in distance slabs.

    python diag_slabs.py -m out/fullcircle/lounge_masked --view 0 --out /tmp/slabs

The raytracer already accepts a near/far plane per call (gray/raytracer.py, `znear`/`zfar`),
so rendering the same camera with a sequence of [znear, zfar] windows shows *where along the
ray* the gaussians that paint each pixel actually live. That answers a question a plain render
cannot: whether a blurry region is under-modelled or modelled at the wrong depth.

Also dumps, for the same camera, the per-gaussian distance distribution restricted to an
angular cone (--cone-uv / --cone-deg) so a specific object can be interrogated numerically.
"""

from __future__ import annotations

import argparse
import copy
import json as _json
import os
from pathlib import Path

import numpy as np
import torch
import tyro

from gray.camera_models import GrayCameraModelClass
from gray.imports import save_image
from gray.prelude import Config, Raytracer, search_for_max_iteration

import render as render_mod


def load(model_path: str, split: str):
    model_dir = Path(model_path)
    cfg = Config(**_json.loads((model_dir / "config.json").read_text()))
    cfg.model_path = str(model_dir)
    mode = render_mod.stored_camera_model(model_dir)
    iteration = search_for_max_iteration(str(model_dir))
    ckpt = str(model_dir / f"gaussians_{iteration:05d}.safetensors")

    views = render_mod.load_render_views(
        mode, cfg=cfg, cli_intrinsics=None, model_dir=model_dir, load_images=True
    )
    cameras = getattr(views, f"{split}_cameras")
    images = getattr(views, f"{split}_images")
    w, h = cameras[0].image_width, cameras[0].image_height
    rt = Raytracer.from_safetensors(cfg, ckpt, w, h, inference_only=True)
    return cfg, mode, iteration, views, cameras, images, rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--slabs",
        default="0:1e9,0:5,5:15,15:40,40:100,100:1e9",
        help="comma-separated znear:zfar windows in world units",
    )
    ap.add_argument("--cone-px", default="", help="'x,y' pixel to interrogate as a cone")
    ap.add_argument("--cone-deg", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg, mode, iteration, views, cameras, images, rt = load(args.model_path, args.split)
    cam = cameras[args.view]
    gt = images.get(cam.image_name)
    w, h = (gt.shape[2], gt.shape[1]) if gt is not None else (cam.image_width, cam.image_height)
    rt.set_render_resolution(w, h)
    print(f"model={args.model_path} mode={mode} iter={iteration} "
          f"view={args.view} name={cam.image_name} {w}x{h}")

    if gt is not None:
        save_image(gt, Path(args.out) / "gt.png")

    for spec in args.slabs.split(","):
        a, b = (float(x) for x in spec.split(":"))
        with torch.no_grad():
            img = rt(cam, znear=a, zfar=b).clamp(0, 1)
        name = f"slab_{a:g}_{b:g}.png".replace("+", "")
        save_image(img, Path(args.out) / name)
        print("  wrote", name)

    # ---- numeric side: where are the gaussians, seen from this camera ----
    means = rt.gaussians.means.detach() if hasattr(rt, "gaussians") else None
    if means is None:
        for attr in ("means", "positions", "xyz", "_means"):
            if hasattr(rt, attr):
                means = getattr(rt, attr).detach()
                break
    if means is None:
        print("  (no means tensor found on raytracer; skipping numeric dump)")
        return
    origin = cam.origin_cuda().reshape(1, 3).to(means.device)
    d = torch.linalg.norm(means - origin, dim=1)
    q = torch.tensor([1, 5, 25, 50, 75, 90, 95, 99, 99.9], device=means.device)
    print("  gaussian distance percentiles:",
          {float(qq): round(float(torch.quantile(d.float(), qq / 100)), 2) for qq in q})


if __name__ == "__main__":
    main()
