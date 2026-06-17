import tyro
from tyro.conf import arg
from dataclasses import dataclass
import json
import gray.colmap as colmap
from typing import Annotated
from pathlib import Path
from gray.camera_models import CAMERA_PARAM_KEYS, GrayCameraModelClass


@dataclass
class CLI: # All args are obligatory
    intrinsics: Annotated[str, arg(aliases=["-i"], help="input colmap file with camera intrinsics (and model name, e.g. 'opencv_fisheye')")] = None
    params: Annotated[str, arg(aliases=["-p"], help="Output JSON file with camera parameters for gray pipeline; defaults to intrinsics path with '_params' suffix")] = None


def main():
    cli = tyro.cli(CLI)

    intrinsics_path = Path(cli.intrinsics)

    if intrinsics_path.suffix == ".bin":
        intrinsics = colmap.read_intrinsics_binary(intrinsics_path)
    elif intrinsics_path.suffix == ".txt":
        intrinsics = colmap.read_intrinsics_text(intrinsics_path)
    else:
        raise ValueError(f"Unsupported intrinsics file format: {intrinsics_path.suffix}. Expected .bin or .txt.")

    assert Path(cli.params).suffix == ".json", "Output params file must have .json extension"

    camera = intrinsics[1]  # Assuming one camera; extend as needed for multiple cameras

    json_data = {param: value for param, value in zip(CAMERA_PARAM_KEYS[GrayCameraModelClass(camera.model).name], camera.params)}
    json_data["model"] = camera.model
    json_data["width"] = camera.width
    json_data["height"] = camera.height

    output_path = Path(cli.params)

    try:
        with output_path.open("w") as f:
                json.dump(json_data, f, indent=4)
    except Exception as e:
        raise ValueError(f"Error writing to output file: {e}")


if __name__ == "__main__":
    main()
