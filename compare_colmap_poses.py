#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import tyro
from tyro.conf import arg

import gray.colmap as colmap


@dataclass(frozen=True)
class CameraPose:
    origin: np.ndarray
    c2w: np.ndarray


def read_colmap_extrinsics(sparse_dir: Path) -> dict[int, colmap.Image]:
    images_bin = sparse_dir / "images.bin"
    images_txt = sparse_dir / "images.txt"
    if images_bin.is_file():
        return colmap.read_extrinsics_binary(str(images_bin))
    if images_txt.is_file():
        return colmap.read_extrinsics_text(str(images_txt))
    raise FileNotFoundError(
        f"Could not find images.bin or images.txt in {sparse_dir}"
    )


def image_to_pose(extr: colmap.Image) -> CameraPose:
    w2c = colmap.qvec2rotmat(extr.qvec)
    c2w = w2c.T
    origin = -c2w @ extr.tvec
    return CameraPose(origin=origin.astype(np.float64), c2w=c2w.astype(np.float64))


def poses_by_name(extrinsics: dict[int, colmap.Image]) -> dict[str, CameraPose]:
    return {extr.name: image_to_pose(extr) for extr in extrinsics.values()}


def umeyama_similarity(
    src: np.ndarray, dst: np.ndarray, *, estimate_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping src -> dst."""
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 matched poses for Sim(3) alignment")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = dst_centered.T @ src_centered / src.shape[0]
    u, singular_values, vt = np.linalg.svd(cov)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    if estimate_scale:
        src_var = np.mean(np.sum(src_centered**2, axis=1))
        scale = float(np.sum(singular_values) / src_var) if src_var > 0 else 1.0
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(
    pose: CameraPose, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> CameraPose:
    origin = scale * (rotation @ pose.origin) + translation
    c2w = rotation @ pose.c2w
    return CameraPose(origin=origin, c2w=c2w)


def rotation_geodesic_deg(c2w_a: np.ndarray, c2w_b: np.ndarray) -> float:
    relative = c2w_a.T @ c2w_b
    cos_angle = (np.trace(relative) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def pairwise_distance_matrix(origins: np.ndarray) -> np.ndarray:
    diff = origins[:, None, :] - origins[None, :, :]
    return np.linalg.norm(diff, axis=2)


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


@dataclass
class ImageComparison:
    name: str
    position_error: float
    orientation_error_deg: float
    pairwise_distance_error: float


@dataclass
class ComparisonSummary:
    ref_dir: Path
    other_dir: Path
    matched_count: int
    ref_only_count: int
    other_only_count: int
    aligned: bool
    scale: float
    per_image: list[ImageComparison]
    position_rmse: float
    position_mean: float
    position_max: float
    orientation_rmse_deg: float
    orientation_mean_deg: float
    orientation_max_deg: float
    pairwise_distance_rmse: float
    pairwise_distance_mean: float
    pairwise_distance_max: float


def compare_poses(
    ref_poses: dict[str, CameraPose],
    other_poses: dict[str, CameraPose],
    *,
    align: bool = True,
) -> ComparisonSummary:
    ref_names = set(ref_poses)
    other_names = set(other_poses)
    common_names = sorted(ref_names & other_names)
    if len(common_names) < 2:
        raise ValueError(
            f"Need at least 2 images present in both reconstructions, found {len(common_names)}"
        )

    ref_origins = np.stack([ref_poses[name].origin for name in common_names], axis=0)
    other_origins = np.stack([other_poses[name].origin for name in common_names], axis=0)

    scale = 1.0
    rotation = np.eye(3)
    translation = np.zeros(3)
    if align:
        scale, rotation, translation = umeyama_similarity(ref_origins, other_origins)

    aligned_ref_origins = np.stack(
        [
            apply_similarity(ref_poses[name], scale, rotation, translation).origin
            for name in common_names
        ],
        axis=0,
    )
    aligned_ref_c2w = {
        name: apply_similarity(ref_poses[name], scale, rotation, translation).c2w
        for name in common_names
    }

    position_errors = np.linalg.norm(aligned_ref_origins - other_origins, axis=1)
    orientation_errors = np.array(
        [
            rotation_geodesic_deg(aligned_ref_c2w[name], other_poses[name].c2w)
            for name in common_names
        ],
        dtype=np.float64,
    )

    ref_pairwise = pairwise_distance_matrix(aligned_ref_origins)
    other_pairwise = pairwise_distance_matrix(other_origins)
    pair_indices = np.triu_indices(len(common_names), k=1)
    pairwise_errors = np.abs(ref_pairwise - other_pairwise)[pair_indices]

    per_image_pairwise = []
    for idx, name in enumerate(common_names):
        row_errors = np.abs(ref_pairwise[idx] - other_pairwise[idx])
        row_errors = np.delete(row_errors, idx)
        per_image_pairwise.append(float(np.mean(row_errors)))

    per_image = [
        ImageComparison(
            name=name,
            position_error=float(position_errors[i]),
            orientation_error_deg=float(orientation_errors[i]),
            pairwise_distance_error=per_image_pairwise[i],
        )
        for i, name in enumerate(common_names)
    ]

    return ComparisonSummary(
        ref_dir=Path("."),
        other_dir=Path("."),
        matched_count=len(common_names),
        ref_only_count=len(ref_names - other_names),
        other_only_count=len(other_names - ref_names),
        aligned=align,
        scale=scale,
        per_image=per_image,
        position_rmse=rms(position_errors),
        position_mean=float(np.mean(position_errors)),
        position_max=float(np.max(position_errors)),
        orientation_rmse_deg=rms(orientation_errors),
        orientation_mean_deg=float(np.mean(orientation_errors)),
        orientation_max_deg=float(np.max(orientation_errors)),
        pairwise_distance_rmse=rms(pairwise_errors),
        pairwise_distance_mean=float(np.mean(pairwise_errors)),
        pairwise_distance_max=float(np.max(pairwise_errors)),
    )


def print_summary(summary: ComparisonSummary, ref_dir: Path, other_dir: Path) -> None:
    print("=== COLMAP pose comparison ===")
    print(f"Reference: {ref_dir}")
    print(f"Other:     {other_dir}")
    print(f"Matched images: {summary.matched_count}")
    if summary.ref_only_count:
        print(f"Only in reference: {summary.ref_only_count}")
    if summary.other_only_count:
        print(f"Only in other: {summary.other_only_count}")
    if summary.aligned:
        print(f"Applied Sim(3) alignment (scale={summary.scale:.6f})")
    else:
        print("Alignment disabled (raw COLMAP coordinates)")
    print()

    print(
        "Per-image metrics "
        "(position error in COLMAP units, orientation in degrees, "
        "pairwise distance error = mean |Δ distance| to other views):"
    )
    print(
        f"{'image':40} {'pos_err':>12} {'orient_deg':>12} {'pair_dist_err':>15}"
    )
    print("-" * 83)
    for item in summary.per_image:
        print(
            f"{item.name:40} "
            f"{item.position_error:12.6f} "
            f"{item.orientation_error_deg:12.6f} "
            f"{item.pairwise_distance_error:15.6f}"
        )
    print()

    print("Summary:")
    print(f"  Position RMSE:            {summary.position_rmse:.6f}")
    print(f"  Position mean / max:      {summary.position_mean:.6f} / {summary.position_max:.6f}")
    print(
        f"  Orientation RMSE (deg):   {summary.orientation_rmse_deg:.6f}"
    )
    print(
        f"  Orientation mean / max:   "
        f"{summary.orientation_mean_deg:.6f} / {summary.orientation_max_deg:.6f}"
    )
    print(
        f"  Pairwise distance RMSE:   {summary.pairwise_distance_rmse:.6f}"
    )
    print(
        f"  Pairwise distance mean / max: "
        f"{summary.pairwise_distance_mean:.6f} / {summary.pairwise_distance_max:.6f}"
    )


@dataclass
class CLI:
    ref: Annotated[
        str,
        arg(aliases=["-r"], help="Reference COLMAP sparse folder (images.bin/txt)"),
    ]
    other: Annotated[
        str,
        arg(aliases=["-o"], help="Other COLMAP sparse folder to compare against"),
    ]
    no_align: Annotated[
        bool,
        arg(help="Skip Sim(3) alignment and compare raw COLMAP coordinates"),
    ] = False


def main() -> int:
    cli = tyro.cli(CLI)
    ref_dir = Path(cli.ref)
    other_dir = Path(cli.other)
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference folder does not exist: {ref_dir}")
    if not other_dir.is_dir():
        raise FileNotFoundError(f"Other folder does not exist: {other_dir}")

    ref_poses = poses_by_name(read_colmap_extrinsics(ref_dir))
    other_poses = poses_by_name(read_colmap_extrinsics(other_dir))
    summary = compare_poses(ref_poses, other_poses, align=not cli.no_align)
    print_summary(summary, ref_dir, other_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
