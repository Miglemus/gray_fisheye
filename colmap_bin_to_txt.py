#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gray.colmap as colmap


def read_points3d_binary_full(path: Path) -> dict[int, tuple[tuple[float, float, float], tuple[int, int, int], float, tuple[int, ...]]]:
    points: dict[int, tuple[tuple[float, float, float], tuple[int, int, int], float, tuple[int, ...]]] = {}
    with path.open("rb") as fid:
        num_points = colmap.read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            data = colmap.read_next_bytes(fid, 43, "QdddBBBd")
            point_id = int(data[0])
            xyz = (float(data[1]), float(data[2]), float(data[3]))
            rgb = (int(data[4]), int(data[5]), int(data[6]))
            error = float(data[7])
            track_len = colmap.read_next_bytes(fid, 8, "Q")[0]
            if track_len > 0:
                track = tuple(
                    int(v)
                    for v in colmap.read_next_bytes(fid, 8 * track_len, "ii" * track_len)
                )
            else:
                track = ()
            points[point_id] = (xyz, rgb, error, track)
    return points


def fmt_float(x: float) -> str:
    return f"{float(x):.17g}"


def write_cameras_text(path: Path, cameras: dict) -> None:
    num_cameras = len(cameras)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {num_cameras}\n")
        for cam_id in sorted(cameras.keys()):
            cam = cameras[cam_id]
            params = " ".join(fmt_float(p) for p in cam.params)
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params}\n")


def write_images_text(path: Path, images: dict) -> None:
    num_images = len(images)
    num_obs = sum(img.xys.shape[0] for img in images.values())
    mean_obs = num_obs / num_images if num_images else 0.0
    with path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_images}, mean observations per image: {mean_obs:.2f}\n")
        for image_id in sorted(images.keys()):
            img = images[image_id]
            qvec = " ".join(fmt_float(x) for x in img.qvec)
            tvec = " ".join(fmt_float(x) for x in img.tvec)
            f.write(f"{img.id} {qvec} {tvec} {img.camera_id} {img.name}\n")
            if img.xys.size == 0:
                f.write("\n")
            else:
                parts = []
                for (x, y), pid in zip(img.xys, img.point3D_ids):
                    parts.append(f"{fmt_float(x)} {fmt_float(y)} {int(pid)}")
                f.write(" ".join(parts) + "\n")


def write_points3d_text(path: Path, points: dict) -> None:
    num_points = len(points)
    if num_points:
        total_track = sum(len(track) // 2 for (_, _, _, track) in points.values())
        mean_track = total_track / num_points
    else:
        mean_track = 0.0
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {num_points}, mean track length: {mean_track:.2f}\n")
        for point_id in sorted(points.keys()):
            xyz, rgb, error, track = points[point_id]
            xyz_str = " ".join(fmt_float(v) for v in xyz)
            rgb_str = " ".join(str(int(v)) for v in rgb)
            track_pairs = []
            for image_id, point2d_idx in zip(track[0::2], track[1::2]):
                track_pairs.append(f"{int(image_id)} {int(point2d_idx)}")
            track_str = " ".join(track_pairs)
            if track_str:
                f.write(f"{point_id} {xyz_str} {rgb_str} {fmt_float(error)} {track_str}\n")
            else:
                f.write(f"{point_id} {xyz_str} {rgb_str} {fmt_float(error)}\n")


def find_model_dir(scene_dir: Path, model_dir: Path | None) -> Path:
    if model_dir is not None:
        return model_dir

    sparse0 = scene_dir / "sparse" / "0"
    if (sparse0 / "cameras.bin").is_file() and (sparse0 / "images.bin").is_file():
        return sparse0

    if (scene_dir / "cameras.bin").is_file() and (scene_dir / "images.bin").is_file():
        return scene_dir

    raise FileNotFoundError(
        "Could not find cameras.bin/images.bin. Pass --model-dir explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COLMAP .bin model files to .txt format."
    )
    parser.add_argument("-s", "--scene", required=True, help="Scene directory")
    parser.add_argument(
        "--model-dir",
        help="Path containing cameras.bin/images.bin/points3D.bin",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to write cameras.txt/images.txt/points3D.txt (default: model dir)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene)
    model_dir = find_model_dir(scene_dir, Path(args.model_dir) if args.model_dir else None)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir

    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"
    points_bin = model_dir / "points3D.bin"

    if not cameras_bin.is_file() or not images_bin.is_file() or not points_bin.is_file():
        raise FileNotFoundError(
            f"Missing .bin files in {model_dir}. Expected cameras.bin, images.bin, points3D.bin."
        )

    cameras = colmap.read_intrinsics_binary(str(cameras_bin))
    images = colmap.read_extrinsics_binary(str(images_bin))
    points = read_points3d_binary_full(points_bin)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_text(output_dir / "cameras.txt", cameras)
    write_images_text(output_dir / "images.txt", images)
    write_points3d_text(output_dir / "points3D.txt", points)

    print("Wrote:")
    print(output_dir / "cameras.txt")
    print(output_dir / "images.txt")
    print(output_dir / "points3D.txt")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
