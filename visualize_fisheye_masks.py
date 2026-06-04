#!/usr/bin/env python3
"""Visualize the radial fisheye vignette mask for one scene view."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import tyro
from tyro.conf import arg

from gray.config import Config
from gray.fisheye_mask import geometric_valid_mask, intrinsics_at_resolution
from gray.scene import SceneInfo


@dataclass
class CLI:
    source_path: Annotated[str, arg(aliases=["-s"])]
    downsampling: Annotated[int, arg(aliases=["-r"])] = 4
    image_name: Optional[str] = None  # * default: first training view
    output: Annotated[str, arg(aliases=["-o"])] = "fisheye_mask_preview.png"
    radius_scale: float = 0.95  # * aggressivity of the radial mask (1.0 == exact 90 deg)


def _mask_to_rgb(mask) -> np.ndarray:
    """[H, W] bool -> [H, W, 3] float in [0, 1] (white = valid)."""
    return mask.cpu().numpy()[..., None].repeat(3, axis=-1).astype(np.float32)


def main():
    cli = tyro.cli(CLI)

    cfg = Config(
        source_path=cli.source_path,
        model_path=os.devnull,
        downsampling=str(cli.downsampling),
        fisheye=True,
        eval=False,
    )
    scene = SceneInfo.from_colmap(cfg, parse_point_cloud=False)

    if cli.image_name:
        cam = next(c for c in scene.train_cameras if c.image_name == cli.image_name)
    else:
        cam = scene.train_cameras[0]
    image = scene.train_images[cam.image_name]  # [C, H, W] on CUDA
    height, width = image.shape[-2], image.shape[-1]

    if cam.intrinsics is None:
        raise ValueError(f"Camera {cam.image_name} has no OPENCV_FISHEYE intrinsics")

    intr = intrinsics_at_resolution(cam, height, width)
    baseline_mask = geometric_valid_mask(intr, height, width, image.device, radius_scale=1.0)
    scaled_mask = geometric_valid_mask(
        intr, height, width, image.device, radius_scale=cli.radius_scale
    )

    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    masked_rgb = rgb * scaled_mask.cpu().numpy()[..., None]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = [
        "Original",
        r"Baseline mask ($\theta < 90°$)",
        rf"Scaled mask (radius_scale={cli.radius_scale:g})",
        "Image masked by scaled mask",
    ]
    panels = [rgb, _mask_to_rgb(baseline_mask), _mask_to_rgb(scaled_mask), masked_rgb]

    for ax, title, panel in zip(axes, titles, panels):
        ax.imshow(np.clip(panel, 0, 1))
        ax.set_title(title)
        ax.axis("off")

    valid = scaled_mask.sum().item()
    total = scaled_mask.numel()
    extra = (baseline_mask.sum().item() - valid) / max(baseline_mask.sum().item(), 1) * 100
    fig.suptitle(
        f"{cam.image_name}  |  scaled valid {valid:,} / {total:,} px  "
        f"|  {extra:.1f}% more disk masked vs 90°"
    )
    fig.tight_layout()
    fig.savefig(cli.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {cli.output}")


if __name__ == "__main__":
    main()
