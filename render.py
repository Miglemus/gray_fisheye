from gray.imports import *
from gray.prelude import *
from gray.eval import (
    load_eval_views,
    load_eval_gt_images,
    normalize_intrinsics_file,
    scene_to_views,
    validate_eval_modes,
)

from concurrent.futures import ThreadPoolExecutor, as_completed

from gray.camera_models import GrayCameraModelClass, CAMERA_PARAM_KEYS
from run_colmap_fixed import load_config
import os


@dataclass
class RenderCLI:
    model_path: Annotated[str, arg(aliases=["-m"])]

    iteration: Annotated[int, arg(aliases=["-t"])] = -1
    splits: List[Literal["train", "test"]] = field(default_factory=lambda: ["test"])
    eval_models: List[Literal["pinhole", "opencv_fisheye", "thin_prism_fisheye"]] = field(default_factory=lambda: ["pinhole"])

    # * Optional changes to this image size
    intrinsics: Annotated[Optional[os.PathLike], arg(help="JSON or colmap bin/txt file with camera intrinsics (and model name, e.g. 'opencv_fisheye'); defaults to source_path parameters")] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fov_y: Optional[float] = None

    # Optional per-frame znear overrides
    znear_list: Optional[List[float]] = None
    znear: float = 0.0

    def __post_init__(self):
        self.eval_models = [GrayCameraModelClass(mode) for mode in self.eval_models]
        if self.intrinsics:
            camera_config = normalize_intrinsics_file(self.intrinsics)
            camera_model = GrayCameraModelClass(camera_config.model)

            if not any(camera_model == eval_mode for eval_mode in self.eval_models):
                self.eval_models.append(camera_model)


# * Parse Config
cli, unknown_args = tyro.cli(RenderCLI, return_unknown_args=True)

# * Load the config from JSON and allow for Config overrides
saved_cli_path = os.path.join(cli.model_path, "config.json")
try:
    json_configuration = json.load(open(saved_cli_path, "r"))
    default = Config(**json_configuration)
except TypeError as e:
    print(f"Error loading config: {e}. Using default config.")
    if json_configuration.get("fisheye"):
        del json_configuration["fisheye"]
    default = Config(**json_configuration)

cfg = tyro.cli(Config, args=unknown_args, default=default)

camera_config = None
if cli.intrinsics is not None:
    camera_config = normalize_intrinsics_file(Path(cli.intrinsics))

# * Make it possible to point directly to a gaussians file
if cli.model_path.endswith(".safetensors"):
    iteration = cfg.iteration
    save_path = cli.model_path
elif cli.iteration != -1:
    iteration = cli.iteration
    save_path = os.path.join(cli.model_path, f"gaussians_{iteration:05d}.safetensors")
else:
    iteration = search_for_max_iteration(cli.model_path)
    save_path = os.path.join(cli.model_path, f"gaussians_{iteration:05d}.safetensors")

# * Manage camera models
# * we want to find the colmap camera model used during colmap
# * then validate if all eval modes are either colmap model or pinhole
# * if not, make sure an intrinsics model is provided and use that for rendering
colmap_model = GrayCameraModelClass(cfg.camera_model)
intrinsics_model = None if camera_config is None else GrayCameraModelClass(camera_config.model)
validate_eval_modes(cli.eval_models, colmap_model, intrinsics_model=intrinsics_model)
eval_modes = cli.eval_models

def load_render_views(mode, *, load_images=True):
    if mode == colmap_model or mode == GrayCameraModelClass("pinhole"):
        return load_eval_views(
            cfg,
            mode,
            load_images=load_images,
        )
    else:
        return scene_to_views(
            SceneInfo.from_cameras_json(
                cfg.model_path,
                camera_model=camera_config,
                parse_images=load_images,
            )
        )
    

probe_views = {mode: load_render_views(mode, load_images=False) for mode in eval_modes}
all_cameras = [
    cam
    for views in probe_views.values()
    for cams in [views.train_cameras, views.test_cameras]
    for cam in cams[:1]
]
if not all_cameras:
    raise ValueError("No cameras found for rendering")
max_width = max(cam.image_width for cam in all_cameras)
max_height = max(cam.image_height for cam in all_cameras)
raytracer = Raytracer.from_safetensors(
    cfg,
    save_path,
    cli.width or max_width,
    cli.height or max_height,
    inference_only=True,
)

# * Render images
print("Rendering iteration", iteration)
executor = ThreadPoolExecutor()
for camera_model in eval_modes:
    views = load_render_views(camera_model)

    for split in cli.splits:
        dir_name = os.path.join(cli.model_path, split, f"{iteration:05d}", camera_model)
        os.makedirs(os.path.join(dir_name, "renders"), exist_ok=True)
        os.makedirs(os.path.join(dir_name, "gt"), exist_ok=True)

        if views.valid_mask is not None:
            save_image(views.valid_mask.float()[None], os.path.join(dir_name, "valid_mask.png"))

        if split == "train":
            cameras = views.train_cameras
            gt_images = (
                load_eval_gt_images(cfg, camera_model, cameras)
                if camera_model == intrinsics_model
                else views.train_images
            )
        elif split == "test":
            cameras = views.test_cameras
            gt_images = (
                load_eval_gt_images(cfg, camera_model, cameras)
                if camera_model == intrinsics_model
                else views.test_images
            )

        if cli.znear_list is not None and len(cli.znear_list) != len(cameras):
            raise ValueError(
                f"Expected {len(cameras)} znear values for split '{split}', got {len(cli.znear_list)}"
            )

        cam_intrinsics = None
        if camera_config is not None and camera_model == intrinsics_model:
            param_key = GrayCameraModelClass(camera_config.model)
            params = list(camera_config.intrinsics)
            for i, param in enumerate(CAMERA_PARAM_KEYS[param_key]):
                if param in ["fx", "fy", "cx", "cy"]:
                    params[i] /= int(cfg.downsampling)
            cam_intrinsics = np.array(params, dtype=np.float64)

        futures = []

        for i, cam in enumerate(cameras):
            gt = gt_images.get(cam.image_name)
            if cli.fov_y is not None:
                cam.fov_y = cli.fov_y

            with torch.no_grad():
                znear = cli.znear_list[i] if cli.znear_list is not None else cli.znear
                raytracer.set_render_resolution(
                    cli.width or cam.image_width,
                    cli.height or cam.image_height,
                )
                
                if cam_intrinsics is not None:
                    cam.intrinsics = cam_intrinsics
                    cam.model = camera_model
                render = raytracer(cam, znear=znear).clamp(0, 1)

            futures.append(
                executor.submit(save_image, render, os.path.join(dir_name, "renders", f"{i:05d}.png"))
            )
            # * Ground truth is only available when the dataset images are present.
            if gt is not None:
                futures.append(
                    executor.submit(save_image, gt, os.path.join(dir_name, "gt", f"{i:05d}.png"))
                )

        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Saving {camera_model} {split} images"):
            pass
    del views
    torch.cuda.empty_cache()
