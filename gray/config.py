from dataclasses import dataclass, field
from tyro.conf import arg
from typing import Annotated, List, Optional, Literal

from gray.camera_models import GrayCameraModel, GrayCameraModelClass, is_fisheye_gray_model, normalize_gray_model


@dataclass
class DatasetConfig:
    source_path: Annotated[str, arg(aliases=["-s"])]  
    model_path: Annotated[str, arg(aliases=["-m"])]  

    downsampling: Annotated[str, arg(aliases=["-r"])] = 4  # * Integer downsampling factor
    images_dir: Annotated[str, arg(aliases=["-i"])] = "images_{downsampling}" # * Relative to source_path or absolute
    
    point_cloud_file: Annotated[str, arg(aliases=["-p"])] = "point_cloud.safetensors" # * Relative to source_path or absolute
    
    eval: bool = True  

    colmap_sparse_subdir: str = "sparse/0"  # * Overridden for fisheye camera models
    eval_modes: List[GrayCameraModel] = field(default_factory=list)

    camera_model: Annotated[GrayCameraModel, arg(aliases=["-c"])] = "pinhole"
    # * Fisheye vignette masking (only applied for fisheye camera models)
    fisheye_mask_geometric: bool = True  # * Mask pixels outside the lens disk (radial mask)
    # * Aggressivity of the radial mask: 1.0 == exact 90 deg disk (baseline); values < 1 shrink the
    # * valid radius to also cover the vignetted rim. Masked surface grows ~ (1 - radius_scale**2).
    fisheye_mask_radius_scale: float = 0.95
    # * Directory holding prebuilt per-camera masks (valid_mask_cam<uid>.png). When set, these
    # * override the geometric masks so every method can share the exact same mask files.
    fisheye_mask_dir: Optional[str] = None
    # * Directory holding per-image transient masks (<image_name stem>.png, 255 = transient,
    # * e.g. the photographer). Masked pixels are excluded from the training loss only; eval
    # * and previews keep using the per-camera valid masks.
    person_mask_dir: Optional[str] = None
    # * Zero out non-finite gaussian gradients before each optimizer step. Guard for scenes
    # * (FIORD night_out) where the backward kernel emits a few Inf/NaN rotation gradients
    # * that would otherwise poison the Adam moments and the whole model within 2 steps.
    nan_grad_guard: bool = False

    def __post_init__(self):
        self.camera_model = normalize_gray_model(self.camera_model)
        self.eval_modes = [normalize_gray_model(mode) for mode in self.eval_modes]
        # * Fisheye uses the distorted COLMAP reconstruction and the raw (resized) images
        if is_fisheye_gray_model(self.camera_model):
            self.colmap_sparse_subdir = "distorted/sparse/0"
            if self.images_dir == "images_{downsampling}":
                self.images_dir = "input_{downsampling}"
        # * Allow using other settings when specifying paths e.g. {downsampling} in images_dir
        self.images_dir = self.images_dir.format(
            downsampling=self.downsampling, source_path=self.source_path
        )
        self.point_cloud_file = self.point_cloud_file.format(
            downsampling=self.downsampling, source_path=self.source_path, images_dir=self.images_dir
        )


@dataclass
class RaytracerConfig:
    # * Render settings
    render_depth: bool = False  
    preview_train_image_name: Optional[str] = None 
    preview_test_image_name: Optional[str] = None

    # * Logging
    preview_iters: List[int] = field(default_factory=lambda: [1, 1_000, 2_500, 7_500, 15_000])
    test_iters: List[int] = field(default_factory=lambda: [7_500, 15_000])
    save_iters: List[int] = field(default_factory=lambda: [7_500, 15_000])
    log_loss_interval: int = 1000
    log_stats_interval: int = 1000
    viewer: bool = False  # * Open the viewer during training
    yes: Annotated[bool, arg(aliases=["-y"])] = False # * Allows overwriting existing directories without prompt

    # * Memory use 
    ppll_forward_size: int = 300_000_000
    ppll_backward_size: int = 120_000_000

    # * Background color
    bg_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # * Raytracing quality
    exp_power: float = 2.0
    alpha_threshold: float = 0.01
    t_threshold: float = 0.03

    # * Init
    init_scale: float = 0.0005  # * Higher values may work better at low resolution
    init_opacity: float = 0.1
    init_binning: bool = True
    init_bin_size: float =  0.0015  # * Post-bugfix default; tuned to roughly match old behavior at 0.04

    # * Low-resolution higher batch size warmup
    half_res_iters: int = 0  
    half_res_batch_size: int = 1  

    # * Loss
    lambda_ssim: float = 0.2

    # * Optimization
    iterations: Annotated[int, arg(aliases=["-t"])]  = 15_000
    batch_size: int = 1  # * Cameras per optimization step (gradient accumulation)
    lr_mean_init: float = 0.00016
    lr_mean_final: float = 0.0000016
    lr_channels: float = 0.0025  # * Only used when SH are disabled
    lr_opacity_init: float = 0.02
    lr_opacity_final: float = 0.005
    lr_scale_init: float = 0.02
    lr_scale_final: float = 0.005
    lr_rotation_init: float = 0.004
    lr_rotation_final: float = 0.001
    lr_sh_dc_init: float = 0.04
    lr_sh_dc_final: float = 0.0025
    lr_sh_rest: float = 0.000625
    tiling: int = 1  # * Legacy setting retained for backwards compatibility
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-15
    sh_update_laziness: int = 1 # * Only step the SH non-dc coefficients every `sh_update_laziness` iterations
    lr_schedule_delay_mult: float = 0.01

    # * Pruning
    pruning: bool = True
    pruning_interval: int = 500  # * Was 100 in 3DGS
    pruning_from_iter: int = 500
    pruning_min_weight: float = 1e-7

    # * Performance settings
    rebuild_interval: int = 500

    # * Scale decay
    scale_decay: float = 0.999875

    # * SH settings
    sh: bool = True
    sh_init_degree: int = 0
    sh_max_degree: int = 3
    sh_increment_interval: int = 1000

    # * Exposure compensation
    exposure_comp_enabled: bool = False
    exposure_comp_lr_init: float = 0.001
    exposure_comp_lr_final: float = 0.0001
    exposure_comp_lr_delay_mult: float = 0.001
    exposure_comp_lr_max_steps: int = 5000

    # * Vignetting compensation (one global set of parameters shared by all views)
    vignetting_comp: bool = False
    vignetting_coeff_lr: float = 0.001
    vignetting_pp_lr: float = 0.001
    vignetting_activation: Literal["exp", "relu"] = "relu"
    vignetting_terms: int = 0  # * Number of even-power radial terms (r^2, r^4, ...)
    vignetting_include_linear_term: bool = True
    vignetting_srgb_comp: bool = False  # * Apply the vignette in (approximate) linear space
    load_vignetting: Optional[str] = None  # * Load fixed vignetting parameters from a safetensors file

    # * MLP settings
    pre_mlp: bool = False
    pre_mlp_feature_size: int = 8
    pre_mlp_width: int = 128
    pre_mlp_layers: int = 4
    pre_mlp_lr: float = 1e-3
    pre_mlp_feature_lr: float = 3e-2
    pre_mlp_freq_bands: Optional[int] = 4
    post_mlp: bool = False
    post_mlp_width: int = 128  # * Wider can improve PSNR a bit
    post_mlp_layers: int = 5  # * Deeper may improve PSNR a bit
    post_mlp_lr: float = 1e-3
    post_mlp_freq_bands: Optional[int] = 1  # * Optimal value may be scene-dependent
    tcnn: bool = False

    @property
    def num_vignetting_coefficients(self) -> int:
        return self.vignetting_terms + int(self.vignetting_include_linear_term)

    def __post_init__(self):
        # * Ensure save_iters includes the final iteration
        if self.iterations not in self.save_iters:
            self.save_iters.append(self.iterations)
        if self.iterations not in self.test_iters:
            self.test_iters.append(self.iterations)
        if self.iterations not in self.preview_iters:
            self.preview_iters.append(self.iterations)

        # * Enforce valid configurations
        assert self.batch_size >= 1
        assert self.vignetting_terms >= 0
        if self.vignetting_comp:
            assert self.num_vignetting_coefficients >= 1, (
                "Vignetting requires at least one coefficient (linear term or vignetting_terms)"
            )
        assert self.sh_init_degree <= self.sh_max_degree
        assert 0 <= self.sh_max_degree <= 3
        if self.sh:
            assert not self.pre_mlp, (
                "Spherical harmonics cannot be used with pre-MLP (choose either one)"
            )
            assert not self.post_mlp, "Spherical harmonics cannot be used with post-MLP"
        if not self.sh:
            self.sh_max_degree = 0

        assert len(self.bg_color) == 3, "bg_color must contain exactly 3 channels"


@dataclass
class Config(RaytracerConfig, DatasetConfig):
    def __post_init__(self):
        DatasetConfig.__post_init__(self)
        RaytracerConfig.__post_init__(self)

    def resolved_eval_modes(self) -> List[GrayCameraModelClass]:
        if self.eval_modes:
            return self.eval_modes
        return [self.camera_model]
