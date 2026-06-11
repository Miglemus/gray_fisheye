from gray.imports import *
from gray.prelude import *
from gray.eval import load_eval_views, scene_to_views

from concurrent.futures import ThreadPoolExecutor, as_completed

from run_colmap_fixed import CameraConfig, load_config, CAMERA_PARAM_KEYS
import os


@dataclass
class RenderCLI:
    model_path: Annotated[str, arg(aliases=["-m"])]

    iteration: Annotated[int, arg(aliases=["-t"])] = -1
    splits: List[Literal["train", "test"]] = field(default_factory=lambda: ["test"])
    eval_models: List[Literal["pinhole", "opencv_fisheye", "thin_prism_fisheye"]] = field(default_factory=lambda: ["pinhole"])

    # * Optional changes to this image size
    intrinsics: Annotated[Optional[os.PathLike], arg(help="JSON file with camera intrinsics (and model name, e.g. 'opencv_fisheye'); defaults to source_path parameters")] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fov_y: Optional[float] = None

    # Optional per-frame znear overrides
    znear_list: Optional[List[float]] = None
    znear: float = 0.0


# * Parse Config
cli, unknown_args = tyro.cli(RenderCLI, return_unknown_args=True)

# * Load the config from JSON and allow for Config overrides
saved_cli_path = os.path.join(cli.model_path, "config.json")
cfg = tyro.cli(Config, args=unknown_args, default=Config(**json.load(open(saved_cli_path, "r"))))

camera_config = None
if cli.intrinsics is not None:
    camera_config = load_config(Path(cli.intrinsics))

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

eval_modes = cli.eval_models or cfg.resolved_eval_modes()


def load_render_views(mode, *, load_images=True):
    try:
        return load_eval_views(cfg, mode, load_images=load_images)
    except FileNotFoundError:
        if cli.eval_models:
            raise
        print(
            f"Colmap dataset not found at '{cfg.source_path}'; "
            f"falling back to cameras saved in the model's cameras.json"
        )
        return scene_to_views(SceneInfo.from_cameras_json(cli.model_path, parse_images=load_images))


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
            cameras, images = views.train_cameras, views.train_images
        elif split == "test":
            cameras, images = views.test_cameras, views.test_images

        if cli.znear_list is not None and len(cli.znear_list) != len(cameras):
            raise ValueError(
                f"Expected {len(cameras)} znear values for split '{split}', got {len(cli.znear_list)}"
            )

        cam_intrinsics = None
        if camera_config is not None and camera_model.lower() == camera_config.model.lower():
            for i, param in enumerate(CAMERA_PARAM_KEYS[camera_config.model]):
                if param in ["fx", "fy", "cx", "cy"]:
                    camera_config.params[i] /= int(cfg.downsampling)
            cam_intrinsics = np.array(camera_config.params)

        futures = []

        for i, cam in enumerate(cameras):
            gt = images.get(cam.image_name)
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
