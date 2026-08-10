from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple

import tyro
from tyro.conf import arg

from gray.camera_config import CameraConfig, normalize_intrinsics_file
from gray.camera_models import GrayCameraModelClass, normalize_gray_model
from gray.eval import load_eval_views, scene_to_views, validate_eval_modes
from gray.fisheye_mask import build_fisheye_mask
from gray.imports import json, save_image, torch, tqdm
from gray.prelude import Config, Raytracer, SceneInfo, search_for_max_iteration


@dataclass
class RenderCLI:
    model_path: Annotated[str, arg(aliases=["-m"])]

    iteration: Annotated[int, arg(aliases=["-t"])] = -1
    splits: List[Literal["train", "test"]] = field(default_factory=lambda: ["test"])
    eval_models: List[
        Literal[
            "pinhole", "opencv_fisheye", "thin_prism_fisheye", "rad_tan_thin_prism_fisheye", "equirectangular"
        ]
    ] = field(default_factory=lambda: ["pinhole"])

    intrinsics: Annotated[
        Optional[os.PathLike],
        arg(
            help=(
                "JSON or COLMAP cameras.bin/.txt with camera intrinsics. "
                "Defaults to the matching source sparse path when it exists."
            )
        ),
    ] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fov_y: Optional[float] = None

    znear_list: Optional[List[float]] = None
    znear: float = 0.0

    def __post_init__(self):
        self.eval_models = [GrayCameraModelClass(mode) for mode in self.eval_models]
        if self.intrinsics:
            camera_config = normalized_camera_config(Path(self.intrinsics))
            camera_model = GrayCameraModelClass(camera_config.model)
            if not any(camera_model == eval_mode for eval_mode in self.eval_models):
                self.eval_models.append(camera_model)


def normalized_camera_config(path: Path) -> CameraConfig:
    camera_config = normalize_intrinsics_file(path)
    camera_config.model = normalize_gray_model(camera_config.model)
    return camera_config


def model_dir_from_path(model_path: str) -> Path:
    path = Path(model_path)
    return path.parent if path.suffix == ".safetensors" else path


def load_config(model_dir: Path, unknown_args) -> Config:
    saved_config_path = model_dir / "config.json"
    json_configuration = json.loads(saved_config_path.read_text())
    try:
        default = Config(**json_configuration)
    except TypeError as error:
        print(f"Error loading config: {error}. Using compatible config.")
        json_configuration.pop("fisheye", None)
        default = Config(**json_configuration)
    cfg = tyro.cli(Config, args=unknown_args, default=default)
    cfg.model_path = str(model_dir)
    return cfg


def resolve_checkpoint(cli: RenderCLI, cfg: Config, model_dir: Path) -> Tuple[int, str]:
    model_path = Path(cli.model_path)
    if model_path.suffix == ".safetensors":
        stem = model_path.stem
        if stem.startswith("gaussians_"):
            return int(stem.removeprefix("gaussians_")), str(model_path)
        return cfg.iterations, str(model_path)
    if cli.iteration != -1:
        iteration = cli.iteration
    else:
        iteration = search_for_max_iteration(str(model_dir))
    return iteration, str(model_dir / f"gaussians_{iteration:05d}.safetensors")


def first_existing_intrinsics(source_path: Path, sparse_subdir: str) -> Optional[Path]:
    sparse_dir = source_path / sparse_subdir
    for filename in ("cameras.bin", "cameras.txt"):
        path = sparse_dir / filename
        if path.is_file():
            return path
    return None


def source_intrinsics_for_mode(cfg: Config, mode: GrayCameraModelClass) -> Optional[CameraConfig]:
    source_path = Path(cfg.source_path)
    if not source_path.exists():
        return None
    intrinsics_path = first_existing_intrinsics(source_path, mode.sparse_subdir())
    if intrinsics_path is None:
        return None
    camera_config = normalized_camera_config(intrinsics_path)
    return camera_config if GrayCameraModelClass(camera_config.model) == mode else None


def stored_camera_model(model_dir: Path) -> Optional[GrayCameraModelClass]:
    cameras_path = model_dir / "cameras.json"
    if not cameras_path.is_file():
        return None
    cameras = json.loads(cameras_path.read_text())
    if not cameras:
        return None
    return GrayCameraModelClass(cameras[0].get("model", "pinhole"))


def saved_camera_sizes(model_dir: Path) -> Dict[str, Tuple[int, int]]:
    cameras_path = model_dir / "cameras.json"
    if not cameras_path.is_file():
        return {}
    return {
        entry["image_name"]: (entry["image_width"], entry["image_height"])
        for entry in json.loads(cameras_path.read_text())
    }


def render_size(gt, cam, cli: RenderCLI, sizes: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    if cli.width is not None and cli.height is not None:
        return cli.width, cli.height
    if gt is not None:
        return gt.shape[2], gt.shape[1]
    if cam.image_width and cam.image_height:
        return cam.image_width, cam.image_height
    if cam.image_name in sizes:
        return sizes[cam.image_name]
    raise ValueError("Cannot determine render size; pass --width and --height.")


def explicit_intrinsics_for_mode(
    cli_intrinsics: Optional[CameraConfig],
    mode: GrayCameraModelClass,
) -> Optional[CameraConfig]:
    if cli_intrinsics is None:
        return None
    return cli_intrinsics if GrayCameraModelClass(cli_intrinsics.model) == mode else None


def intrinsics_for_json_override(
    cli_intrinsics: Optional[CameraConfig],
    cfg: Config,
    mode: GrayCameraModelClass,
) -> Optional[CameraConfig]:
    return explicit_intrinsics_for_mode(cli_intrinsics, mode) or source_intrinsics_for_mode(cfg, mode)


def without_prebuilt_mask(cfg: Config) -> Config:
    view_cfg = copy.copy(cfg)
    view_cfg.fisheye_mask_geometric = False
    return view_cfg


def validate_modes(
    modes: List[GrayCameraModelClass],
    cfg: Config,
    cli_intrinsics: Optional[CameraConfig],
) -> None:
    colmap_model = GrayCameraModelClass(cfg.camera_model)
    intrinsics_model = None if cli_intrinsics is None else GrayCameraModelClass(cli_intrinsics.model)
    try:
        validate_eval_modes(modes, colmap_model, intrinsics_model=intrinsics_model)
    except ValueError:
        for mode in modes:
            if mode == GrayCameraModelClass("pinhole") or mode == colmap_model:
                continue
            if explicit_intrinsics_for_mode(cli_intrinsics, mode) or source_intrinsics_for_mode(cfg, mode):
                continue
            raise


def load_render_views(
    mode: GrayCameraModelClass,
    *,
    cfg: Config,
    cli_intrinsics: Optional[CameraConfig],
    model_dir: Path,
    load_images: bool,
):
    if source_intrinsics_for_mode(cfg, mode) is not None:
        return load_eval_views(
            without_prebuilt_mask(cfg) if mode.is_fisheye() else cfg,
            mode,
            load_images=load_images,
        )

    camera_config = intrinsics_for_json_override(cli_intrinsics, cfg, mode)
    stored_model = stored_camera_model(model_dir)
    if camera_config is None and stored_model != mode:
        if mode == GrayCameraModelClass("pinhole"):
            raise ValueError(
                "pinhole rendering without source requires --intrinsics pointing to "
                "undistorted sparse/0 cameras or a pinhole intrinsics JSON."
            )
        raise ValueError(
            f"Camera model '{mode}' is not stored in cameras.json and no matching intrinsics were found."
        )

    return scene_to_views(
        SceneInfo.from_cameras_json(
            str(model_dir),
            camera_model=camera_config,
            parse_images=load_images,
            cfg=without_prebuilt_mask(cfg) if mode.is_fisheye() else cfg,
        )
    )


def fisheye_masks_for_split(mode, views, cameras, gt_images, cli, cfg, sizes):
    """Per-colmap-camera masks {uid: mask} for a split, or None for non-fisheye modes."""
    if not mode.is_fisheye() or not cameras:
        return None
    if getattr(views, "valid_masks", None):
        return views.valid_masks
    if views.valid_mask is not None:
        return {cam.uid: views.valid_mask for cam in cameras}

    masks = {}
    mask_cfg = copy.copy(cfg)
    mask_cfg.fisheye_mask_geometric = True
    ref_cams = {}
    for cam in cameras:
        ref_cams.setdefault(cam.uid, cam)
    for uid, cam in sorted(ref_cams.items()):
        ref_gt = gt_images.get(cam.image_name)
        width, height = render_size(ref_gt, cam, cli, sizes)
        device = ref_gt.device if ref_gt is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Building {mode} fisheye mask for camera {uid} at {width}x{height} on {device}...")
        masks[uid] = build_fisheye_mask(cam, height, width, device=device, cfg=mask_cfg)
    return masks


def main() -> None:
    cli, unknown_args = tyro.cli(RenderCLI, return_unknown_args=True)
    model_dir = model_dir_from_path(cli.model_path)
    cfg = load_config(model_dir, unknown_args)
    cli_intrinsics = (
        normalized_camera_config(Path(cli.intrinsics)) if cli.intrinsics is not None else None
    )

    validate_modes(cli.eval_models, cfg, cli_intrinsics)
    iteration, checkpoint_path = resolve_checkpoint(cli, cfg, model_dir)
    sizes = saved_camera_sizes(model_dir)

    probe_views = {
        mode: load_render_views(
            mode,
            cfg=cfg,
            cli_intrinsics=cli_intrinsics,
            model_dir=model_dir,
            load_images=False,
        )
        for mode in cli.eval_models
    }
    if cli.width is not None and cli.height is not None:
        init_width, init_height = cli.width, cli.height
    else:
        init_sizes = [
            render_size(None, cam, cli, sizes)
            for views in probe_views.values()
            for cams in (views.train_cameras, views.test_cameras)
            for cam in cams
        ]
        if not init_sizes:
            raise ValueError("No cameras found for rendering")
        init_width = max(width for width, _ in init_sizes)
        init_height = max(height for _, height in init_sizes)

    raytracer = Raytracer.from_safetensors(
        cfg,
        checkpoint_path,
        init_width,
        init_height,
        inference_only=True,
    )

    print("Rendering iteration", iteration)
    executor = ThreadPoolExecutor()
    for camera_model in cli.eval_models:
        views = load_render_views(
            camera_model,
            cfg=cfg,
            cli_intrinsics=cli_intrinsics,
            model_dir=model_dir,
            load_images=True,
        )

        for split in cli.splits:
            dir_name = model_dir / split / f"{iteration:05d}" / str(camera_model)
            renders_dir = dir_name / "renders"
            gt_dir = dir_name / "gt"
            renders_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)

            cameras = getattr(views, f"{split}_cameras")
            gt_images = getattr(views, f"{split}_images")
            if cli.znear_list is not None and len(cli.znear_list) != len(cameras):
                raise ValueError(
                    f"Expected {len(cameras)} znear values for split '{split}', got {len(cli.znear_list)}"
                )

            valid_masks = fisheye_masks_for_split(
                camera_model, views, cameras, gt_images, cli, cfg, sizes
            )
            if valid_masks is not None:
                # * Keep valid_mask.png (first camera) for single-camera tooling, and
                # * write per-camera masks plus a per-render-file index for rigs.
                first_uid = cameras[0].uid
                save_image(valid_masks[first_uid].float()[None], dir_name / "valid_mask.png")
                for uid, m in sorted(valid_masks.items()):
                    save_image(m.float()[None], dir_name / f"valid_mask_cam{uid}.png")
                mask_index = {
                    f"{i:05d}.png": f"valid_mask_cam{cam.uid}.png"
                    for i, cam in enumerate(cameras)
                }
                with open(dir_name / "masks.json", "w") as f:
                    import json

                    json.dump(mask_index, f, indent=1)

            futures = []
            for i, cam in enumerate(cameras):
                gt = gt_images.get(cam.image_name)
                if cli.fov_y is not None:
                    cam.fov_y = cli.fov_y

                with torch.no_grad():
                    znear = cli.znear_list[i] if cli.znear_list is not None else cli.znear
                    render_width, render_height = render_size(gt, cam, cli, sizes)
                    raytracer.set_render_resolution(render_width, render_height)
                    render = raytracer(cam, znear=znear).clamp(0, 1)

                futures.append(executor.submit(save_image, render, renders_dir / f"{i:05d}.png"))
                if gt is not None:
                    futures.append(executor.submit(save_image, gt, gt_dir / f"{i:05d}.png"))

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Saving {camera_model} {split} images",
            ):
                future.result()
        del views
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
