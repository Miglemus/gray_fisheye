import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import pycolmap
import tyro
from tyro.conf import arg

from gray.colmap import best_reconstruction_model


@dataclass
class CLI:
    source_path: Annotated[str, arg(aliases=["-s"])]
    camera: Literal["OPENCV", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE", "RAD_TAN_THIN_PRISM_FISHEYE"] = "OPENCV"
    gpu: bool = True
    delete_input: bool = False
    undistort: bool = False

cli = tyro.cli(CLI)

src = Path(cli.source_path)

assert (src / "input").is_dir(), f"Input directory not found: {src / 'input'}"
distorted_dir = src / "distorted"
database_path = distorted_dir / "database.db"
sparse_root = distorted_dir / "sparse"

if database_path.exists():
    database_path.unlink()
if sparse_root.exists():
    shutil.rmtree(sparse_root)
sparse_root.mkdir(parents=True, exist_ok=True)

# * Feature extraction
device = pycolmap.Device.cuda if cli.gpu else pycolmap.Device.cpu
pycolmap.extract_features(
    database_path=database_path,
    image_path=src / "input",
    camera_mode=pycolmap.CameraMode.SINGLE,
    reader_options=pycolmap.ImageReaderOptions(camera_model=cli.camera),
    extraction_options=pycolmap.FeatureExtractionOptions(use_gpu=cli.gpu),
    device=device,
)

# * Feature matching
pycolmap.match_exhaustive(
    database_path=database_path,
    device=device,
)

# * Incremental mapping (bundle adjustment)
maps = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=src / "input",
    output_path=sparse_root,
    options=pycolmap.IncrementalPipelineOptions(
        ba_global_function_tolerance=1e-6 # * speeds up bundle adjustment 
    ),
)
if not maps:
    logging.error("Incremental mapping failed. Exiting.")
    raise SystemExit(1)

best_idx, rec = best_reconstruction_model(maps)
if len(maps) > 1:
    sizes = {i: r.num_reg_images() for i, r in maps.items()}
    print(f"Multiple reconstructions {sizes}; using model {best_idx} ({rec.num_reg_images()} images)")

# * Promote the selected reconstruction to the canonical path used by fisheye training/eval.
best_sparse_dir = sparse_root / str(best_idx)
canonical_sparse_dir = sparse_root / "0"
if best_sparse_dir != canonical_sparse_dir:
    if canonical_sparse_dir.exists():
        shutil.rmtree(canonical_sparse_dir)
    shutil.copytree(best_sparse_dir, canonical_sparse_dir)
    print(f"Promoted best reconstruction {best_idx} to {canonical_sparse_dir}")

# * Image undistortion
if cli.undistort:
    shutil.rmtree(src / "sparse", ignore_errors=True)
    shutil.rmtree(src / "images", ignore_errors=True)
    pycolmap.undistort_images(
        output_path=src,
        input_path=canonical_sparse_dir,
        image_path=src / "input",
        output_type="COLMAP",
    )

    # * Flatten COLMAP undistortion output into sparse/0.
    (src / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for f in (src / "sparse").iterdir():
        if f.name == "0":
            continue
        shutil.move(str(f), str(src / "sparse" / "0" / f.name))

# * Cleanup
# shutil.rmtree(src / "distorted")
# shutil.rmtree(src / "stereo", ignore_errors=True)
# for f in src.glob("*.sh"):
#     f.unlink()
# if cli.delete_input:
#     shutil.rmtree(src / "input")
