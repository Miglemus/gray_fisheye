from __future__ import annotations

from gray.fisheye_mask import build_fisheye_mask
from gray.imports import *
from gray.utils import *
from gray.camera_models import GrayCameraModelClass
from gray.config import Config
from gray.camera import CameraInfo
import gray.colmap as colmap

from torchvision.io import read_image, ImageReadMode
from concurrent.futures import ThreadPoolExecutor
import torch
import json
import copy
from PIL import Image

from run_colmap_fixed import CameraConfig

executor = ThreadPoolExecutor()


@dataclass
class BasicPointCloud:
    points: np.array
    colors: np.array
    radius: float
    normals: Optional[np.array] = None
    distances_to_cam: Optional[np.array] = None


@dataclass
class ColmapViews:
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    train_images: Dict[str, torch.Tensor]
    test_images: Dict[str, torch.Tensor]
    valid_mask: Optional[torch.Tensor] = None
    valid_mask_halfres: Optional[torch.Tensor] = None
    train_images_halfres: Dict[str, torch.Tensor] = field(default_factory=dict)


def _read_colmap_cameras(sparse_dir):
    try:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
        return (
            colmap.read_extrinsics_binary(cameras_extrinsic_file),
            colmap.read_intrinsics_binary(cameras_intrinsic_file),
        )
    except FileNotFoundError:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.txt")
        return (
            colmap.read_extrinsics_text(cameras_extrinsic_file),
            colmap.read_intrinsics_text(cameras_intrinsic_file),
        )


def load_colmap_views(
    cfg: Config,
    *,
    sparse_subdir: str,
    images_dir: str,
    apply_fisheye_mask: bool = False,
    llffhold=8,
    load_images=True,
    build_halfres=False,
    expected_camera_model: Optional[GrayCameraModelClass] = None,
) -> ColmapViews:
    path = cfg.source_path
    sparse_dir = os.path.join(path, sparse_subdir)
    cam_extrinsics, cam_intrinsics = _read_colmap_cameras(sparse_dir)

    # * Select views for eval
    if cfg.eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [
                name for idx, name in enumerate(cam_names) if idx % llffhold == 0
            ]
        else:
            with open(os.path.join(sparse_dir, "test.txt"), "r") as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    view_cfg = copy.copy(cfg)
    view_cfg.colmap_sparse_subdir = sparse_subdir
    view_cfg.images_dir = images_dir

    cam_infos_unsorted = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        cam_info = CameraInfo.from_colmap(
            view_cfg, key, extr, intr, extr.name in test_cam_names_list
        )
        cam_infos_unsorted.append(cam_info)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    if expected_camera_model is not None and cam_infos:
        actual_model = GrayCameraModelClass(cam_infos[0].model)
        expected_camera_model = expected_camera_model
        if not actual_model == expected_camera_model:
            raise ValueError(
                f"Expected camera model '{expected_camera_model}' but COLMAP sparse "
                f"reconstruction at '{sparse_dir}' uses '{actual_model}'"
            )
    train_cam_infos = [c for c in cam_infos if not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    def load_image(cam):
        image = read_image(cam.image_path, ImageReadMode.RGB).cuda() / 255
        return cam.image_name, image

    if load_images:
        train_images = dict(executor.map(load_image, train_cam_infos))
        test_images = dict(executor.map(load_image, test_cam_infos))
    else:
        train_images = {}
        test_images = {}

    train_images_halfres = {}
    if build_halfres and cfg.half_res_iters > 0:
        train_images_halfres = {
            name: F.interpolate(image[None], scale_factor=0.5, mode="area")[0]
            for name, image in train_images.items()
        }

    valid_mask = None
    valid_mask_halfres = None
    if apply_fisheye_mask and cfg.fisheye_mask_geometric:
        from gray.fisheye_mask import build_fisheye_mask

        ref_cam = train_cam_infos[0] if train_cam_infos else test_cam_infos[0]
        if load_images:
            ref_image = train_images[ref_cam.image_name] if train_cam_infos else test_images[ref_cam.image_name]
            height, width = ref_image.shape[-2], ref_image.shape[-1]
            device = ref_image.device
        else:
            height, width = ref_cam.image_height, ref_cam.image_width
            device = "cpu"
        valid_mask = build_fisheye_mask(ref_cam, height, width, device, cfg)
        if build_halfres and cfg.half_res_iters > 0 and load_images:
            valid_mask_halfres = (
                F.interpolate(valid_mask[None, None].float(), scale_factor=0.5, mode="nearest")[
                    0, 0
                ]
                > 0.5
            )

    return ColmapViews(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        train_images=train_images,
        test_images=test_images,
        valid_mask=valid_mask,
        valid_mask_halfres=valid_mask_halfres,
        train_images_halfres=train_images_halfres,
    )


def peek_max_resolution(cfg: Config, mode_specs) -> Tuple[int, int]:
    max_width = 0
    max_height = 0
    for sparse_subdir, images_dir in mode_specs:
        views = load_colmap_views(
            cfg,
            sparse_subdir=sparse_subdir,
            images_dir=images_dir,
            load_images=False,
        )
        cams = views.train_cameras or views.test_cameras
        if not cams:
            continue
        cam = cams[0]
        max_width = max(max_width, cam.image_width)
        max_height = max(max_height, cam.image_height)
    return max_width, max_height


@dataclass
class SceneInfo:
    point_cloud: Optional[BasicPointCloud]
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    train_images: Dict[str, torch.Tensor]
    train_images_halfres: Dict[str, torch.Tensor]
    test_images: Dict[str, torch.Tensor]
    pc_path: Optional[str]
    is_nerf_synthetic: bool
    # * Shared radial mask for fisheye vignette (same camera / resolution for all views)
    valid_mask: Optional[torch.Tensor] = None
    valid_mask_halfres: Optional[torch.Tensor] = None

    @staticmethod
    def from_colmap(cfg: Config, llffhold=8, parse_point_cloud=True) -> SceneInfo:
        path = cfg.source_path
        views = load_colmap_views(
            cfg,
            sparse_subdir=cfg.colmap_sparse_subdir,
            images_dir=cfg.images_dir,
            apply_fisheye_mask=GrayCameraModelClass(cfg.camera_model).is_fisheye(),
            llffhold=llffhold,
            build_halfres=True,
            expected_camera_model=cfg.camera_model,
        )
        train_cam_infos = views.train_cameras
        test_cam_infos = views.test_cameras
        radius = get_nerf_pp_norm(train_cam_infos)["radius"]

        # * Parse point cloud, cache to safetensors for fast loading
        if parse_point_cloud:
            if os.path.isabs(cfg.point_cloud_file):
                pc_path = cfg.point_cloud_file
            else:
                pc_path = os.path.join(path, cfg.point_cloud_file)
            safetensor_path = pc_path.replace(".ply", ".safetensors")
            import safetensors.numpy

            if os.path.exists(safetensor_path) and (
                not os.path.exists(pc_path)
                or os.path.getmtime(safetensor_path) >= os.path.getmtime(pc_path)
            ):
                data = safetensors.numpy.load_file(safetensor_path)
                positions = data["positions"]
                colors = data["colors"]
                if "distances_to_cam" in data:
                    distances_to_cam = data["distances_to_cam"]
                else:
                    distances_to_cam = None
            else:
                plydata = PlyData.read(pc_path)
                vertices = plydata["vertex"]
                positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
                colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
                distances_to_cam = None
                safetensors.numpy.save_file(
                    {"positions": positions, "colors": colors}, safetensor_path
                )
            pcd = BasicPointCloud(
                points=positions,
                colors=colors,
                normals=None,
                radius=radius,
                distances_to_cam=distances_to_cam,
            )
            pc_path = safetensor_path
        else:
            pcd = None
            pc_path = None

        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            train_images=views.train_images,
            test_images=views.test_images,
            pc_path=pc_path,
            is_nerf_synthetic=False,
            train_images_halfres=views.train_images_halfres,
            valid_mask=views.valid_mask,
            valid_mask_halfres=views.valid_mask_halfres,
        )

    @staticmethod
    def from_cameras_json(model_path: str, camera_model: CameraConfig=None, parse_images=True) -> SceneInfo:
        model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
        cameras_path = os.path.join(model_dir, "cameras.json")
        if not os.path.exists(cameras_path):
            raise FileNotFoundError(
                f"No colmap dataset and no cameras.json found at '{cameras_path}'"
            )

        with open(cameras_path, "r") as file:
            payload = json.load(file)
        cam_infos = sorted(
            (CameraInfo.from_json(entry) for entry in payload),
            key=lambda x: x.image_name,
        )
        train_cam_infos = [c for c in cam_infos if not c.is_test]
        test_cam_infos = [c for c in cam_infos if c.is_test]

        if camera_model is None:
            model = GrayCameraModelClass(cam_infos[0].model)
        else:  # Apply the provided camera model to every view, not just the first one.
            model = GrayCameraModelClass(camera_model.model)

            for cam_info in cam_infos:
                # If pinhole, this function is never called, so necessarily fisheye.
                # Replace "images" with "input" for all cameras so the dataset path matches.
                cam_info.image_path = cam_info.image_path.replace("images", "input")
                cam_info.image_name = os.path.basename(cam_info.image_path)
                cam_info.image_width, cam_info.image_height = camera_model.width, camera_model.height

                cam_info.model = model
                cam_info.intrinsics = camera_model.intrinsics

            with Image.open(cam_infos[0].image_path) as image:
                height, width = image.size[1], image.size[0]

        if model.is_fisheye():
            single_cam_info = cam_infos[0]
            valid_mask = build_fisheye_mask(
                single_cam_info,
                height,
                width,
                device="cpu",
                cfg=Config(
                    source_path=model_dir,
                    model_path=model_dir,
                    camera_model=model,
                    fisheye_mask_geometric=True,
                    fisheye_mask_radius_scale=0.95,
                ),
            )

        def load_images(cams):
            images = {}
            if parse_images:
                for cam in cams:
                    if cam.image_path and os.path.exists(cam.image_path):
                        images[cam.image_name] = (
                            read_image(cam.image_path, ImageReadMode.RGB).cuda() / 255
                        )
            return images

        return SceneInfo(
            point_cloud=None,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            train_images=load_images(train_cam_infos),
            train_images_halfres={},
            test_images=load_images(test_cam_infos),
            pc_path=None,
            is_nerf_synthetic=False,
            valid_mask=valid_mask if model.is_fisheye() else None,
        )


