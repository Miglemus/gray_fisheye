#!/usr/bin/env python3
"""Convert FishEyeNeRF train/intrinsics (4x4 K + k1,k2) to COLMAP cameras.txt / params.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import tyro
from PIL import Image
from tyro.conf import arg

from colmap_bin_to_txt import write_cameras_text
from gray.camera_models import CAMERA_PARAM_KEYS, param_key_to_colmap_model
from gray.colmap import Camera

_MODEL = {"_ocv": "opencv_fisheye", "_tpf": "thin_prism_fisheye"}


def parse_train_intrinsics(path: Path) -> tuple[float, float, float, float, float, float]:
    v = np.loadtxt(path, dtype=np.float64)
    m = v[:16].reshape(4, 4)
    return float(m[0, 0]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2]), float(v[16]), float(v[17])


def scene_model(name: str) -> str:
    for suffix, model in _MODEL.items():
        if name.endswith(suffix):
            return model
    raise ValueError(f"Unknown scene suffix in '{name}' (expected _ocv or _tpf)")


def dir_size(path: Path) -> tuple[int, int]:
    imgs = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"No images in {path}")
    with Image.open(imgs[0]) as im:
        return im.size


def to_colmap_params(model: str, fx: float, fy: float, cx: float, cy: float, k1: float, k2: float) -> list[float]:
    values = dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2)
    return [float(values.get(k, 0.0)) for k in CAMERA_PARAM_KEYS[model]]


def convert_scene(scene: Path, *, full_res: bool) -> tuple[Camera, dict]:
    intr_dir = scene / "train" / "intrinsics"
    paths = sorted(intr_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No intrinsics in {intr_dir}")

    model = scene_model(scene.name)
    fx, fy, cx, cy, k1, k2 = parse_train_intrinsics(paths[0])
    train_w, train_h = dir_size(scene / "train" / "rgb")
    if full_res and (scene / "input").is_dir():
        out_w, out_h = dir_size(scene / "input")
    else:
        out_w, out_h = train_w, train_h
    sx, sy = out_w / train_w, out_h / train_h
    params = to_colmap_params(model, fx * sx, fy * sy, cx * sx, cy * sy, k1, k2)
    colmap_model = param_key_to_colmap_model(model)
    camera = Camera(id=1, model=colmap_model, width=out_w, height=out_h, params=params)
    params_json = {
        "model": model,
        "width": out_w,
        "height": out_h,
        **{k: v for k, v in zip(CAMERA_PARAM_KEYS[model], params)},
    }
    return camera, params_json


@dataclass
class CLI:
    root: Annotated[str, arg(help="FishEyeNeRF root directory")] = "data/FishEyeNeRF"
    scene: Annotated[Optional[str], arg(help="Single scene name; default: all *_ocv|*_tpf scenes")] = None
    full_res: Annotated[
        bool,
        arg(help="Scale intrinsics to input/ resolution (4240x2384) instead of train/rgb"),
    ] = True


def main() -> int:
    cli = tyro.cli(CLI)
    root = Path(cli.root)
    scenes = [root / cli.scene] if cli.scene else sorted(p for p in root.iterdir() if p.is_dir())
    for scene in scenes:
        try:
            scene_model(scene.name)
        except ValueError:
            continue
        camera, params_json = convert_scene(scene, full_res=cli.full_res)
        write_cameras_text(scene / "intrinsics.txt", {1: camera})
        (scene / "params.json").write_text(json.dumps(params_json, indent=2) + "\n")
        print(f"{scene.name}: {camera.model} {camera.width}x{camera.height}")
    return 0


if __name__ == "__main__":
    main()
