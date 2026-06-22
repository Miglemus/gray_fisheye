#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import tyro
from tyro.conf import arg

import gray.colmap as colmap
from colmap_bin_to_txt import write_images_text


def load_c2w(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.shape != (4, 4):
        values = values.reshape(4, 4)
    return values


def c2w_to_colmap(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 camera-to-world matrix to COLMAP qvec/tvec (world-to-camera)."""
    rotation_c2w = c2w[:3, :3]
    camera_center = c2w[:3, 3]
    rotation_w2c = rotation_c2w.T
    tvec = -rotation_w2c @ camera_center
    qvec = colmap.rotmat2qvec(rotation_w2c)
    return qvec, tvec


def default_image_name(pose_path: Path, image_name_format: str) -> str:
    image_id = int(pose_path.stem)
    return image_name_format.format(id=image_id, stem=pose_path.stem)


def load_reference_metadata(
    reference_sparse: Path,
) -> dict[str, tuple[str, int]]:
    images_bin = reference_sparse / "images.bin"
    images_txt = reference_sparse / "images.txt"
    if images_bin.is_file():
        images = colmap.read_extrinsics_binary(str(images_bin))
    elif images_txt.is_file():
        images = colmap.read_extrinsics_text(str(images_txt))
    else:
        raise FileNotFoundError(
            f"Could not find images.bin or images.txt in {reference_sparse}"
        )

    metadata: dict[str, tuple[str, int]] = {}
    for extr in images.values():
        stem = Path(extr.name).stem
        metadata[stem] = (extr.name, extr.camera_id)
    return metadata


def build_colmap_images(
    pose_paths: list[Path],
    *,
    image_name_format: str,
    reference_metadata: Optional[dict[str, tuple[str, int]]] = None,
    default_camera_id: int = 1,
) -> dict[int, colmap.Image]:
    images: dict[int, colmap.Image] = {}
    for image_id, pose_path in enumerate(pose_paths, start=1):
        c2w = load_c2w(pose_path)
        qvec, tvec = c2w_to_colmap(c2w)

        stem = pose_path.stem
        if reference_metadata is not None:
            ref_key = str(int(stem))
            if ref_key not in reference_metadata:
                raise KeyError(
                    f"No reference COLMAP image found for pose '{pose_path.name}' "
                    f"(expected image stem '{ref_key}')"
                )
            image_name, camera_id = reference_metadata[ref_key]
        else:
            image_name = default_image_name(pose_path, image_name_format)
            camera_id = default_camera_id

        images[image_id] = colmap.Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )
    return images


@dataclass
class CLI:
    poses_dir: Annotated[
        str,
        arg(aliases=["-p"], help="Directory containing one 4x4 c2w matrix per *.txt file"),
    ]
    output_dir: Annotated[
        str,
        arg(aliases=["-o"], help="Output COLMAP sparse directory (images.txt written here)"),
    ]
    reference_sparse: Annotated[
        Optional[str],
        arg(
            aliases=["-r"],
            help=(
                "Existing COLMAP sparse folder used to copy cameras.txt and reuse "
                "image names / camera_id assignments"
            ),
        ),
    ] = None
    image_name_format: Annotated[
        str,
        arg(
            help=(
                "Python format string for image names when --reference-sparse is not set. "
                "Available fields: {id}, {stem}"
            ),
        ),
    ] = "{id}.jpg"
    copy_cameras: Annotated[
        bool,
        arg(help="Copy cameras.txt from --reference-sparse when provided"),
    ] = True


def main() -> int:
    cli = tyro.cli(CLI)
    poses_dir = Path(cli.poses_dir)
    output_dir = Path(cli.output_dir)

    if not poses_dir.is_dir():
        raise FileNotFoundError(f"Pose directory does not exist: {poses_dir}")

    pose_paths = sorted(poses_dir.glob("*.txt"))
    if not pose_paths:
        raise FileNotFoundError(f"No *.txt pose files found in {poses_dir}")

    reference_sparse = Path(cli.reference_sparse) if cli.reference_sparse else None
    reference_metadata = None
    if reference_sparse is not None:
        if not reference_sparse.is_dir():
            raise FileNotFoundError(f"Reference sparse directory does not exist: {reference_sparse}")
        reference_metadata = load_reference_metadata(reference_sparse)

    output_dir.mkdir(parents=True, exist_ok=True)
    images = build_colmap_images(
        pose_paths,
        image_name_format=cli.image_name_format,
        reference_metadata=reference_metadata,
    )
    images_path = output_dir / "images.txt"
    write_images_text(images_path, images)

    cameras_path = output_dir / "cameras.txt"
    if reference_sparse is not None and cli.copy_cameras:
        for source_name in ("cameras.txt", "cameras.bin"):
            source_path = reference_sparse / source_name
            if source_path.is_file():
                if source_name.endswith(".txt"):
                    shutil.copy2(source_path, cameras_path)
                else:
                    cameras = colmap.read_intrinsics_binary(str(source_path))
                    from colmap_bin_to_txt import write_cameras_text

                    write_cameras_text(cameras_path, cameras)
                break
        else:
            raise FileNotFoundError(
                f"Could not find cameras.txt or cameras.bin in {reference_sparse}"
            )

    print(f"Wrote {len(images)} poses to {images_path}")
    if cameras_path.is_file():
        print(f"Wrote {cameras_path}")
    elif reference_sparse is None:
        print(
            "Note: cameras.txt was not written. Provide --reference-sparse to copy intrinsics "
            "from an existing COLMAP reconstruction."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
