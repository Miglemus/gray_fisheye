from __future__ import annotations

from gray.imports import *
from gray.camera_models import (
    GrayCameraModel,
    gray_models_equal,
    gray_model_from_colmap,
    is_fisheye_gray_model,
    normalize_gray_model,
)
from gray.config import Config
from gray.scene import ColmapViews, SceneInfo, load_colmap_views
from gray.utils import masked_psnr, masked_ssim


EvalMode = GrayCameraModel


def uses_custom_intrinsics(
    mode: EvalMode, colmap_camera_model: EvalMode, intrinsics_model: Optional[str]
) -> bool:
    if gray_models_equal(mode, "pinhole") or gray_models_equal(mode, colmap_camera_model):
        return False
    return intrinsics_model is not None and gray_models_equal(mode, intrinsics_model)


def validate_eval_modes(
    eval_modes: List[EvalMode],
    colmap_camera_model: EvalMode,
    *,
    intrinsics_model: Optional[str] = None,
) -> None:
    """Each eval mode must be pinhole, the COLMAP model, or covered by --intrinsics."""
    colmap_camera_model = normalize_gray_model(colmap_camera_model)
    normalized_intrinsics = (
        normalize_gray_model(intrinsics_model) if intrinsics_model is not None else None
    )
    for mode in eval_modes:
        mode = normalize_gray_model(mode)
        if gray_models_equal(mode, "pinhole") or gray_models_equal(mode, colmap_camera_model):
            continue
        if uses_custom_intrinsics(mode, colmap_camera_model, normalized_intrinsics):
            continue
        raise ValueError(
            f"Camera model '{mode}' is neither pinhole nor the COLMAP model "
            f"('{colmap_camera_model}'). Provide --intrinsics with a matching model."
        )


def source_mode_for_eval(
    mode: EvalMode,
    colmap_camera_model: EvalMode,
    *,
    intrinsics_model: Optional[str] = None,
) -> EvalMode:
    """COLMAP dataset to load poses/images from for a given eval mode."""
    mode = normalize_gray_model(mode)
    colmap_camera_model = normalize_gray_model(colmap_camera_model)
    normalized_intrinsics = (
        normalize_gray_model(intrinsics_model) if intrinsics_model is not None else None
    )
    validate_eval_modes([mode], colmap_camera_model, intrinsics_model=normalized_intrinsics)
    if gray_models_equal(mode, "pinhole") or gray_models_equal(mode, colmap_camera_model):
        return mode
    return colmap_camera_model


EVAL_MODEL_PRESETS: Dict[EvalMode, Tuple[str, str]] = {
    "pinhole": ("sparse/0", "images_{downsampling}"),
    "opencv_fisheye": ("distorted/sparse/0", "input_{downsampling}"),
    "thin_prism_fisheye": ("distorted/sparse/0", "input_{downsampling}"),
}


def format_eval_images_dir(cfg: Config, mode: EvalMode) -> str:
    return EVAL_MODEL_PRESETS[mode][1].format(
        downsampling=cfg.downsampling,
        source_path=cfg.source_path,
    )


def mode_sparse_subdir(mode: EvalMode) -> str:
    return EVAL_MODEL_PRESETS[mode][0]


def load_eval_views(
    cfg: Config,
    mode: EvalMode,
    *,
    load_images=True,
    expected_camera_model: Optional[EvalMode] = None,
) -> ColmapViews:
    return load_colmap_views(
        cfg,
        sparse_subdir=mode_sparse_subdir(mode),
        images_dir=format_eval_images_dir(cfg, mode),
        apply_fisheye_mask=is_fisheye_gray_model(mode),
        load_images=load_images,
        expected_camera_model=normalize_gray_model(expected_camera_model or mode),
    )


def max_framebuffer_size(scene: SceneInfo, eval_views: Dict[EvalMode, ColmapViews]) -> Tuple[int, int]:
    max_width = 0
    max_height = 0
    for cams in [scene.train_cameras, scene.test_cameras]:
        if cams:
            max_width = max(max_width, cams[0].image_width)
            max_height = max(max_height, cams[0].image_height)
    for views in eval_views.values():
        for cams in [views.train_cameras, views.test_cameras]:
            if cams:
                max_width = max(max_width, cams[0].image_width)
                max_height = max(max_height, cams[0].image_height)
    return max_width, max_height


def scene_to_views(scene: SceneInfo) -> ColmapViews:
    return ColmapViews(
        train_cameras=scene.train_cameras,
        test_cameras=scene.test_cameras,
        train_images=scene.train_images,
        test_images=scene.test_images,
        valid_mask=scene.valid_mask,
        valid_mask_halfres=scene.valid_mask_halfres,
        train_images_halfres=scene.train_images_halfres,
    )


def _metric_pair(render, gt, mask):
    if mask is not None:
        return (
            masked_psnr(render.cuda(), gt.cuda(), mask).item(),
            masked_ssim(render.cuda(), gt.cuda(), mask).item(),
        )
    return (
        psnr(render[None].cuda(), gt[None].cuda()).item(),
        ssim(render[None].cuda(), gt[None].cuda(), downsample=False).item(),
    )


def compute_split_metrics(raytracer, cameras, images, mask=None, znear=0.0):
    if not cameras:
        return None

    raytracer.set_render_resolution(cameras[0].image_width, cameras[0].image_height)
    with torch.no_grad():
        renders = [
            (raytracer(cam, znear=znear).clamp(0, 1).cpu()[None] * 255).floor() / 255
            for cam in cameras
        ]
    gts = [images[cam.image_name].cpu()[None] for cam in cameras]
    renders = torch.cat(renders, dim=0)
    gts = torch.cat(gts, dim=0)

    psnr_scores = []
    ssim_scores = []
    for idx in range(len(cameras)):
        psnr_score, ssim_score = _metric_pair(renders[idx], gts[idx], mask)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)

    return mean(psnr_scores), mean(ssim_scores)


def compute_view_metrics(raytracer, views: ColmapViews, *, znear=0.0):
    metrics = {}
    for split, cameras, images in [
        ("train", views.train_cameras, views.train_images),
        ("test", views.test_cameras, views.test_images),
    ]:
        split_metrics = compute_split_metrics(
            raytracer,
            cameras,
            images,
            views.valid_mask,
            znear=znear,
        )
        if split_metrics is not None:
            metrics[split] = split_metrics
    return metrics
