from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import tyro
from PIL import Image, ImageOps
from tqdm import tqdm
from tyro.conf import arg

import gray.colmap as colmap
from gray.fisheye_geometry import project_fisheye


@dataclass
class CLI:
    source_path: Annotated[str, arg(aliases=["-s"])]
    input_dir: Annotated[str, arg(aliases=["-i"])] = "input"
    output_images_dir: str = "images_consistent"
    output_sparse_dir: str = "sparse_consistent/0"
    point_cloud_file: str = "point_cloud_consistent.safetensors"
    width: int = 1600
    height: int = 1066
    hfov_deg: float = 90.0
    report_assets_dir: str = "ai_reports/consistent_undistort_poc_images"
    yes: Annotated[bool, arg(aliases=["-y"])] = False


def fmt_float(x: float) -> str:
    return f"{float(x):.17g}"


def write_cameras_text(path: Path, camera_id: int, width: int, height: int, params: np.ndarray) -> None:
    path.write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"{camera_id} PINHOLE {width} {height} {' '.join(fmt_float(p) for p in params)}\n",
        encoding="utf-8",
    )


def write_images_text(path: Path, images: dict, camera_id: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
        for image_id in sorted(images.keys()):
            img = images[image_id]
            qvec = " ".join(fmt_float(x) for x in img.qvec)
            tvec = " ".join(fmt_float(x) for x in img.tvec)
            f.write(f"{img.id} {qvec} {tvec} {camera_id} {img.name}\n")
            f.write("\n")


if __name__ == "__main__":
    cfg = tyro.cli(CLI)
    scene_dir = Path(cfg.source_path)
    input_dir = scene_dir / cfg.input_dir
    output_images_dir = scene_dir / cfg.output_images_dir
    output_sparse_dir = scene_dir / cfg.output_sparse_dir

    if output_images_dir.exists() or output_sparse_dir.exists():
        if not cfg.yes:
            response = input(
                f"Output exists ({output_images_dir} or {output_sparse_dir}). Overwrite? [y/N]: "
            ).strip().lower()
            if response not in ("y", "yes"):
                raise SystemExit(0)
        shutil.rmtree(output_images_dir, ignore_errors=True)
        shutil.rmtree(output_sparse_dir, ignore_errors=True)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_sparse_dir.mkdir(parents=True, exist_ok=True)

    distorted_sparse_dir = scene_dir / "distorted" / "sparse" / "0"
    cameras = colmap.read_intrinsics_binary(str(distorted_sparse_dir / "cameras.bin"))
    images = colmap.read_extrinsics_binary(str(distorted_sparse_dir / "images.bin"))
    camera = next(iter(cameras.values()))
    if camera.model not in ("OPENCV_FISHEYE", "THIN_PRISM_FISHEYE", "RAD_TAN_THIN_PRISM_FISHEYE"):
        raise ValueError(f"Unsupported custom undistort source camera model: {camera.model}")

    fx = cfg.width / (2.0 * np.tan(np.deg2rad(cfg.hfov_deg) / 2.0))
    fy = fx
    cx = cfg.width / 2.0
    cy = cfg.height / 2.0
    pinhole_params = np.array([fx, fy, cx, cy], dtype=np.float64)
    fov_tag = str(cfg.hfov_deg).replace(".", "p")

    xs = np.arange(cfg.width, dtype=np.float64) + 0.5
    ys = np.arange(cfg.height, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    rays = np.stack([(grid_x - cx) / fx, (grid_y - cy) / fy, np.ones_like(grid_x)], axis=-1)
    map_x, map_y = project_fisheye(camera.model, rays, camera.params)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    valid = (map_x >= 0) & (map_x <= camera.width - 1) & (map_y >= 0) & (map_y <= camera.height - 1)

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for image_path in tqdm(image_paths, desc=f"Undistorting {scene_dir.name}"):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read input image: {image_path}")
        undistorted = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        undistorted[~valid] = 0
        undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        Image.fromarray(undistorted_rgb).save((output_images_dir / image_path.name).with_suffix(".png"))

    write_cameras_text(output_sparse_dir / "cameras.txt", 1, cfg.width, cfg.height, pinhole_params)
    write_images_text(output_sparse_dir / "images.txt", images, 1)
    (output_sparse_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )

    stats = {
        "source_path": str(scene_dir),
        "input_dir": cfg.input_dir,
        "output_images_dir": cfg.output_images_dir,
        "output_sparse_dir": cfg.output_sparse_dir,
        "point_cloud_file": cfg.point_cloud_file,
        "width": cfg.width,
        "height": cfg.height,
        "hfov_deg": cfg.hfov_deg,
        "pinhole_params": pinhole_params.tolist(),
        "source_camera_model": camera.model,
        "source_width": int(camera.width),
        "source_height": int(camera.height),
        "valid_fraction": float(valid.mean()),
        "num_images": len(image_paths),
    }
    (scene_dir / f"consistent_undistort_stats_fov{fov_tag}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    if image_paths:
        assets_dir = Path(cfg.report_assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)
        custom = Image.open((output_images_dir / image_paths[0].name).with_suffix(".png")).convert("RGB")
        colmap_path = (scene_dir / "images" / image_paths[0].name).with_suffix(".png")
        if colmap_path.exists():
            colmap_img = Image.open(colmap_path).convert("RGB")
            preview_h = 400
            custom_preview = ImageOps.contain(custom, (600, preview_h))
            colmap_preview = ImageOps.contain(colmap_img, (600, preview_h))
            preview = Image.new("RGB", (custom_preview.width + colmap_preview.width, preview_h), (0, 0, 0))
            preview.paste(colmap_preview, (0, (preview_h - colmap_preview.height) // 2))
            preview.paste(custom_preview, (colmap_preview.width, (preview_h - custom_preview.height) // 2))
            preview.save(assets_dir / f"{scene_dir.name}_colmap_vs_consistent_fov{fov_tag}.png")

    print(f"Wrote {len(image_paths)} images to {output_images_dir}")
    print(f"Wrote sparse model to {output_sparse_dir}")
    print(f"Valid remap fraction: {valid.mean():.4f}")
