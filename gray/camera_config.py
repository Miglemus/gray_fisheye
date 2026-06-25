from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pycolmap

import gray.colmap as colmap
from gray.camera_models import (
    CAMERA_PARAM_KEYS,
    normalize_gray_model,
    normalize_param_key,
    param_key_to_colmap_model,
)
from gray.colmap import CAMERA_MODEL_NAMES


@dataclass
class CameraConfig:
    model: str
    intrinsics: List[float]
    width: int
    height: int


def load_config(path: Path, default_model: Optional[str] = None) -> CameraConfig:
    data = json.loads(path.read_text())

    if isinstance(data, list):
        model = default_model
        if model is None:
            raise ValueError(
                f"params.json is a bare list; set 'model' in the json or pass --camera ({path})"
            )
        model = normalize_param_key(model)
        params = [float(v) for v in data]
    elif isinstance(data, dict):
        model = _resolve_model(data, default_model)
        params = _params_from_dict(data, model)
        width, height = int(data["width"]), int(data["height"])
    else:
        raise ValueError(f"Expected a JSON object or list in {path}")

    _validate_params(model, params, path)
    try:
        gray_model = normalize_gray_model(model)
    except ValueError:
        gray_model = model
    return CameraConfig(model=gray_model, intrinsics=params, width=width, height=height)


def _resolve_model(data: dict, default_model: Optional[str]) -> str:
    model = data.get("model", default_model)
    if model is None:
        raise ValueError(
            "Camera model required: set 'model' in params.json or pass --camera on the command line"
        )
    return normalize_param_key(model)


def _params_from_dict(data: dict, model: str) -> List[float]:
    if "params" in data:
        return [float(v) for v in data["params"]]

    keys = CAMERA_PARAM_KEYS[model]
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing camera parameters {missing} for model {model}")
    return [float(data[k]) for k in keys]


def _validate_params(model: str, params: List[float], path: Path) -> None:
    colmap_model = param_key_to_colmap_model(model)
    expected = CAMERA_MODEL_NAMES[colmap_model].num_params
    if len(params) != expected:
        raise ValueError(
            f"{model} expects {expected} parameters, got {len(params)} in {path}"
        )
    pycolmap.Camera(model=colmap_model, width=1, height=1, params=params)


def normalize_intrinsics_file(intrinsics_path: os.PathLike) -> CameraConfig:
    intrinsics_path = Path(intrinsics_path)
    if intrinsics_path.suffix == ".json":
        return load_config(intrinsics_path)
    elif intrinsics_path.suffix == ".bin":
        return colmap.read_intrinsics_binary(intrinsics_path)
    elif intrinsics_path.suffix == ".txt":
        return colmap.read_intrinsics_text(intrinsics_path)
    else:
        raise ValueError(f"Unsupported intrinsics file format: {intrinsics_path}")
