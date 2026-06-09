"""Run COLMAP with fixed intrinsics."""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Optional

import pycolmap
import tyro
from tyro.conf import arg

from gray.colmap import CAMERA_MODEL_NAMES, best_reconstruction_model

# * COLMAP parameter order for each supported camera model.
CAMERA_PARAM_KEYS = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
    "FULL_OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
    "FOV": ("fx", "fy", "cx", "cy", "omega"),
    "SIMPLE_RADIAL_FISHEYE": ("f", "cx", "cy", "k"),
    "RADIAL_FISHEYE": ("f", "cx", "cy", "k1", "k2"),
    "THIN_PRISM_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2", "sx1", "sy1"),
}


@dataclass
class CameraConfig:
    model: str
    params: List[float]


@dataclass
class CLI:
    source_path: Annotated[str, arg(aliases=["-s"])]
    config_path: Annotated[
        Optional[str],
        arg(aliases=["-c"], help="JSON file with camera parameters; defaults to <source>/params.json"),
    ] = None
    camera: Annotated[
        Optional[str],
        arg(help="Camera model fallback when params.json does not specify one"),
    ] = None
    gpu: bool = True


def _resolve_model(data: dict, default_model: Optional[str]) -> str:
    model = data.get("model", default_model)
    if model is None:
        raise ValueError(
            "Camera model required: set 'model' in params.json or pass --camera on the command line"
        )
    if model not in CAMERA_MODEL_NAMES:
        known = ", ".join(sorted(CAMERA_MODEL_NAMES))
        raise ValueError(f"Unknown camera model '{model}'. Supported models: {known}")
    return model


def _params_from_dict(data: dict, model: str) -> List[float]:
    if "params" in data:
        return [float(v) for v in data["params"]]

    keys = CAMERA_PARAM_KEYS[model]
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing camera parameters {missing} for model {model}")
    return [float(data[k]) for k in keys]


def _validate_params(model: str, params: List[float], path: Path) -> None:
    expected = CAMERA_MODEL_NAMES[model].num_params
    if len(params) != expected:
        raise ValueError(
            f"{model} expects {expected} parameters, got {len(params)} in {path}"
        )
    pycolmap.Camera(model=model, width=1, height=1, params=params)


def load_config(path: Path, default_model: Optional[str] = None) -> CameraConfig:
    data = json.loads(path.read_text())

    if isinstance(data, list):
        model = default_model
        if model is None:
            raise ValueError(
                f"params.json is a bare list; set 'model' in the json or pass --camera ({path})"
            )
        params = [float(v) for v in data]
    elif isinstance(data, dict):
        model = _resolve_model(data, default_model)
        params = _params_from_dict(data, model)
    else:
        raise ValueError(f"Expected a JSON object or list in {path}")

    _validate_params(model, params, path)
    return CameraConfig(model=model, params=params)


def main():
    cli = tyro.cli(CLI)
    config_path = Path(cli.config_path or Path(cli.source_path) / "params.json")
    camera = load_config(config_path, default_model=cli.camera)
    src = Path(cli.source_path)
    assert (src / "input").is_dir(), f"Input directory not found: {src / 'input'}"
    (src / "distorted" / "sparse").mkdir(parents=True, exist_ok=True)

    params_str = ",".join(str(p) for p in camera.params)
    print(f"Using fixed {camera.model}: params={params_str}")

    device = pycolmap.Device.cuda if cli.gpu else pycolmap.Device.cpu
    reader_options = pycolmap.ImageReaderOptions(
        camera_model=camera.model,
        camera_params=params_str,
    )

    database_path = src / "distorted" / "database.db"
    if database_path.exists():
        database_path.unlink()

    pycolmap.extract_features(
        database_path=database_path,
        image_path=src / "input",
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=pycolmap.FeatureExtractionOptions(use_gpu=cli.gpu),
        device=device,
    )

    pycolmap.match_exhaustive(database_path=database_path, device=device)

    map_options = pycolmap.IncrementalPipelineOptions(ba_global_function_tolerance=1e-6)
    map_options.ba_refine_focal_length = False
    map_options.ba_refine_extra_params = False
    map_options.ba_refine_principal_point = False

    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=src / "input",
        output_path=src / "distorted" / "sparse",
        options=map_options,
    )
    if not maps:
        logging.error("Incremental mapping failed. Exiting.")
        raise SystemExit(1)

    best_idx, rec = best_reconstruction_model(maps)
    if len(maps) > 1:
        sizes = {i: r.num_reg_images() for i, r in maps.items()}
        print(f"Multiple reconstructions {sizes}; using model {best_idx} ({rec.num_reg_images()} images)")

    cam = next(iter(rec.cameras.values()))
    print(f"Reconstruction camera: {cam}")

    pycolmap.undistort_images(
        output_path=src,
        input_path=src / "distorted" / "sparse" / str(best_idx),
        image_path=src / "input",
        output_type="COLMAP",
    )

    (src / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for f in (src / "sparse").iterdir():
        if f.name == "0":
            continue
        shutil.move(str(f), str(src / "sparse" / "0" / f.name))

    print(f"Done. Distorted sparse: {src / 'distorted' / 'sparse' / best_idx}")
    print(f"Undistorted sparse: {src / 'sparse' / '0'}")


if __name__ == "__main__":
    main()
