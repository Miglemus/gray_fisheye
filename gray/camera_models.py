"""Canonical camera model names and case-insensitive normalization."""

from __future__ import annotations

from typing import Dict, Literal, Tuple

GrayCameraModel = Literal["pinhole", "opencv_fisheye", "thin_prism_fisheye"]

GRAY_CAMERA_MODELS: Tuple[GrayCameraModel, ...] = (
    "pinhole",
    "opencv_fisheye",
    "thin_prism_fisheye",
)

# * COLMAP parameter order for each supported camera model (lowercase keys).
CAMERA_PARAM_KEYS: Dict[str, Tuple[str, ...]] = {
    "simple_pinhole": ("f", "cx", "cy"),
    "pinhole": ("fx", "fy", "cx", "cy"),
    "simple_radial": ("f", "cx", "cy", "k"),
    "radial": ("f", "cx", "cy", "k1", "k2"),
    "opencv": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "opencv_fisheye": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
    "thin_prism_fisheye": (
        "fx",
        "fy",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
        "k3",
        "k4",
        "sx1",
        "sy1",
    ),
}

_PARAM_KEY_TO_COLMAP: Dict[str, str] = {
    "simple_pinhole": "SIMPLE_PINHOLE",
    "pinhole": "PINHOLE",
    "simple_radial": "SIMPLE_RADIAL",
    "radial": "RADIAL",
    "opencv": "OPENCV",
    "opencv_fisheye": "OPENCV_FISHEYE",
    "thin_prism_fisheye": "THIN_PRISM_FISHEYE",
}

_COLMAP_TO_PARAM_KEY: Dict[str, str] = {v: k for k, v in _PARAM_KEY_TO_COLMAP.items()}

COLMAP_TO_GRAY: Dict[str, GrayCameraModel] = {
    "PINHOLE": "pinhole",
    "SIMPLE_PINHOLE": "pinhole",
    "OPENCV_FISHEYE": "opencv_fisheye",
    "THIN_PRISM_FISHEYE": "thin_prism_fisheye",
}

_GRAY_TO_COLMAP: Dict[GrayCameraModel, str] = {
    "pinhole": "PINHOLE",
    "opencv_fisheye": "OPENCV_FISHEYE",
    "thin_prism_fisheye": "THIN_PRISM_FISHEYE",
}


def normalize_param_key(name: str) -> str:
    """Normalize any alias to the lowercase key used in CAMERA_PARAM_KEYS."""
    if name in _COLMAP_TO_PARAM_KEY:
        return _COLMAP_TO_PARAM_KEY[name]
    key = name.strip().lower()
    if key in CAMERA_PARAM_KEYS:
        return key
    raise ValueError(
        f"Unknown camera model '{name}'. "
        f"Supported models: {', '.join(sorted(CAMERA_PARAM_KEYS))}"
    )


def normalize_gray_model(name: str) -> GrayCameraModel:
    """Normalize any alias to the canonical gray pipeline model name."""
    if name in COLMAP_TO_GRAY:
        return COLMAP_TO_GRAY[name]
    key = normalize_param_key(name)
    gray = COLMAP_TO_GRAY.get(_PARAM_KEY_TO_COLMAP[key])
    if gray is not None:
        return gray
    raise ValueError(
        f"Camera model '{name}' is not supported by the gray pipeline. "
        f"Use one of: {', '.join(GRAY_CAMERA_MODELS)}"
    )


def gray_model_from_colmap(colmap_model: str) -> GrayCameraModel:
    return normalize_gray_model(colmap_model)


def gray_to_colmap_model(gray_model: str) -> str:
    gray = normalize_gray_model(gray_model)
    return _GRAY_TO_COLMAP[gray]


def param_key_to_colmap_model(param_key: str) -> str:
    key = normalize_param_key(param_key)
    return _PARAM_KEY_TO_COLMAP[key]


def is_fisheye_gray_model(model: str) -> bool:
    return normalize_gray_model(model) in ("opencv_fisheye", "thin_prism_fisheye")


def gray_models_equal(left: str, right: str) -> bool:
    return normalize_gray_model(left) == normalize_gray_model(right)

