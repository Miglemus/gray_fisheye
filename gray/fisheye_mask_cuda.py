from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

CUDA_MODEL_IDS = {
    "opencv_fisheye": 1,
    "thin_prism_fisheye": 2,
    "rad_tan_thin_prism_fisheye": 3,
}


@lru_cache(maxsize=1)
def _load_cuda_extension():
    from torch.utils.cpp_extension import load

    cuda_dir = Path(__file__).resolve().parent.parent / "cuda"
    return load(
        name="gray_fisheye_mask_cuda",
        sources=[
            str(cuda_dir / "fisheye_mask.cpp"),
            str(cuda_dir / "fisheye_mask.cu"),
        ],
        extra_include_paths=[str(cuda_dir)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=os.environ.get("GRAY_FISHEYE_MASK_CUDA_VERBOSE", "0") == "1",
    )


def cuda_masks_available() -> bool:
    return torch.cuda.is_available()


def geometric_valid_mask_cuda(
    model: str,
    intrinsics,
    height: int,
    width: int,
    radius_scale: float = 1.0,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Generate a fisheye validity mask using the same CUDA unproject code as the raytracer."""
    if not cuda_masks_available():
        raise RuntimeError("CUDA is required for fisheye mask generation")

    model_id = CUDA_MODEL_IDS[model]
    cuda_device = torch.device("cuda")
    params = torch.as_tensor(intrinsics, dtype=torch.float32, device=cuda_device)
    ext = _load_cuda_extension()
    mask = ext.generate_fisheye_valid_mask_cuda(model_id, params, height, width, float(radius_scale))
    if device is not None:
        mask = mask.to(device)
    return mask
