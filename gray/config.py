import json
import os
from dataclasses import dataclass, field
from tyro.conf import arg
from typing import Annotated, Dict, List, Optional, Literal, Tuple

from gray.camera_models import (
    GrayCameraModel,
    GrayCameraModelClass,
    is_fisheye_gray_model,
    normalize_gray_model,
)

# * The ablation rungs of the learnable camera model. Named once so that
# * `camera_model_init_rung` cannot drift away from `camera_opt`.
CameraOptRung = Literal[
    "off",
    "passthrough",
    "tilt",
    "radial",
    "ana",
    "noncentral",
    "noncentral_no_ana",
    "z_only",
    "central_matched",
    "raxel",
    # * The two rungs merged from `rttpf-intrinsics`. They do NOT add a residual on top of
    # * COLMAP's calibration -- they re-fit the calibration itself, which is what makes
    # * `rttpf_z` the structural twin of `aspect_noncentral` on pinhole.
    "rttpf",
    "rttpf_z",
]

# * The four per-lens tensors of `gray.camera_model.LensResidual`, in checkpoint order.
LENS_PARAMETERS: Tuple[str, ...] = ("omega", "theta_weights", "phi_weights", "z_weights")


@dataclass
class DatasetConfig:
    source_path: Annotated[str, arg(aliases=["-s"])]
    model_path: Annotated[str, arg(aliases=["-m"])]

    downsampling: Annotated[str, arg(aliases=["-r"])] = 4  # * Integer downsampling factor
    images_dir: Annotated[str, arg(aliases=["-i"])] = (
        "images_{downsampling}"  # * Relative to source_path or absolute
    )

    point_cloud_file: Annotated[str, arg(aliases=["-p"])] = (
        "point_cloud.safetensors"  # * Relative to source_path or absolute
    )

    eval: bool = True
    # * Every-Nth test split (default 8). Set to 0 to read explicit test image names from
    # * <sparse_dir>/test.txt instead (one colmap image name per line).
    llffhold: int = 8

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
    # * Random seed, wired to python `random`, numpy and torch at the very top of train.py
    # * (`set_seeds`), before the scene, the point cloud or the raytracer are touched. It is
    # * recorded verbatim in the run's config.json.
    # *
    # * WHAT IT DOES NOT BUY: bit-exact determinism. gray accumulates gaussian gradients with
    # * CUDA atomics (`cuda/utils/misc.cu`, `cuda/core/per_pixel_linked_list.h`), so the
    # * backward sums float32 in a non-deterministic order; BVH rebuild order, cuBLAS/cuDNN
    # * reduction order and `torch.use_deterministic_algorithms` (never enabled, and not
    # * available for these kernels) all stay free. Two runs at the same seed land ~0.05-0.06
    # * dB apart, which is exactly the measured noise floor. The point of the flag is to get
    # * INDEPENDENT, LABELLED draws so a variance can be computed at all -- not to replay one.
    # *
    # * WHAT IT ACTUALLY MOVES, in a default build. Exactly one draw depends on it: the
    # * per-epoch `random.shuffle(camera_pool)` of train.py, i.e. the ORDER in which views are
    # * visited. The gaussian initialization is deterministic given the point cloud -- the only
    # * `torch.randn` on that path (`raytracer.py:533`) fills `CHANNELS - 3` extra channels and
    # * `CHANNELS` defaults to 3 (`CMakeLists.txt:68`), so it draws a zero-width tensor. So a
    # * seed sweep measures the variance of the SGD trajectory (+ the atomics floor), not of
    # * the initialization. That is the right variance for the 2x2 replication, but say which
    # * one it is: it does NOT cover init sensitivity, and it does not cover COLMAP.
    seed: int = 0

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
    yes: Annotated[bool, arg(aliases=["-y"])] = (
        False  # * Allows overwriting existing directories without prompt
    )

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
    init_bin_size: float = (
        0.0015  # * Post-bugfix default; tuned to roughly match old behavior at 0.04
    )

    # * Low-resolution higher batch size warmup
    half_res_iters: int = 0
    half_res_batch_size: int = 1

    # * Loss
    lambda_ssim: float = 0.2

    # * Optimization
    iterations: Annotated[int, arg(aliases=["-t"])] = 15_000
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
    sh_update_laziness: int = (
        1  # * Only step the SH non-dc coefficients every `sh_update_laziness` iterations
    )
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
    load_vignetting: Optional[str] = (
        None  # * Load fixed vignetting parameters from a safetensors file
    )

    # * Learnable residual camera model (see gray/camera_model.py). The rungs are cumulative:
    # *   off             native ray generation, nothing changes
    # *   passthrough     rays come from Python with the residual pinned to zero -- must
    # *                   reproduce `off`; this is the control for the whole ladder
    # *   tilt            3-DoF bearing rotation (~ principal point + roll)
    # *   radial          + radial residual d(theta)(theta), a cubic B-spline
    # *   ana             + anamorphic cos/sin(k phi), k = 1, 2, on both d(theta) and d(phi)
    # *   noncentral      + on-axis entrance-pupil profile z(theta), gauged to z(0) = 0
    # *   central_matched same parameter count as `noncentral`, all of it central
    # *   raxel           dense generic ray field, the upper bound of the ladder
    # *   noncentral_no_ana  `noncentral` minus the anamorphic harmonics (subtractive)
    # *   z_only          the non-central profile alone, nothing central (subtractive)
    # *   rttpf           re-fit COLMAP's own 16 rttpf parameters photometrically (control)
    # *   rttpf_z         + the non-central profile, on top of the re-fitted calibration
    camera_opt: CameraOptRung = "off"
    camera_opt_from_iter: int = 8000  # * Phase A / phase B boundary; frozen before this
    camera_opt_knots: int = 10  # * Control points of the angular residual splines
    camera_opt_knots_z: int = 8  # * Control points of the non-central profile
    # * 1e-4 is the measured optimum. Sweep on tunnel (-r 8, 3000 it, camera_opt=noncentral),
    # * test PSNR against 26.10 for `off`: 1e-5 -> 26.15, 1e-4 -> 26.17, 1e-3 -> 26.15,
    # * 1e-2 -> 24.76 (diverges). The regularizer barely moves it (26.17 vs 26.16 at 1e-4).
    camera_opt_lr_tilt: float = 1e-4
    camera_opt_lr_angular: float = 1e-4
    camera_opt_lr_z: float = 1e-4  # * In units of the scene radius, so scene-scale free
    camera_opt_lr_raxel: float = 1e-4
    # * In units of the NORMALIZED image plane (~ fx pixels), so it is resolution-free.
    # * The default IS the swept optimum: 7 points on tunnel (-r 8, 7500 it) give a plateau
    # * 1e-6..1e-2 only 0.07 dB wide, peaking here and diverging at 3e-2 (PROTOCOL.md).
    # * Keep it a default rather than a flag -- this rung is a fairness control, and it must
    # * not be possible to handicap it by forgetting to pass its learning rate.
    camera_opt_lr_intrinsics: float = 1e-3
    camera_opt_lr_final_mult: float = 0.1  # * Exponential decay applied over phase B
    camera_opt_reg_l2: float = 1e-2
    camera_opt_reg_curvature: float = 1e-2
    # * Off by default: the 16 rttpf parameters are the model the baseline already trusts,
    # * so pulling them back towards the COLMAP fit would handicap the control.
    camera_opt_reg_intrinsics: float = 0.0
    camera_opt_raxel_stride: int = 8  # * Ray-field grid is (H // stride, W // stride)

    # * ---------------------------------------------------------------- camera-model transfer
    # * Initialize the learnable camera model from ANOTHER run's checkpoint, instead of from
    # * zero. Accepts a `gaussians_*.safetensors` file or a run directory (the highest
    # * iteration in it is used). Every consistency check is loud: mismatched uids, knot
    # * counts or rungs raise instead of being silently truncated / partially initialized.
    camera_model_init: Optional[str] = None
    # * Explicit `source_uid:target_uid` correspondence, comma separated (e.g. "1:2,2:1").
    # * REQUIRED whenever the source and target uid sets differ (a 2-lens rig against a
    # * mono-lens capture). Positional pairing is never done implicitly.
    camera_model_init_uid_map: Optional[str] = None
    # * The rung the SOURCE checkpoint was trained with. Normally read from the source run's
    # * config.json; give it explicitly when that file is absent. Needed because the rung is
    # * NOT recorded in the checkpoint: every rung writes the same four tensors, so loading
    # * `noncentral` weights under `z_only` (or vice versa) is undetectable from the tensors
    # * alone and would leave channels uninitialized / silently dropped.
    camera_model_init_rung: Optional[CameraOptRung] = None
    # * How to reinterpret `z_weights` across scenes. z(theta) is in RAW COLMAP WORLD UNITS
    # * (scene scale enters only through the learning rate, `camera_opt_lr_z`), so a transfer
    # * between two reconstructions is only meaningful if their COLMAP scales agree.
    # *   none          copy verbatim. Correct for a same-scene transfer (seed replication,
    # *                 frozen-camera controls) and WRONG in general across scenes.
    # *   scene_radius  multiply by target_radius / source_radius, i.e. keep z as a fraction
    # *                 of the scene radius. The defensible convention across scenes, and the
    # *                 one consistent with how the z learning rate is already scaled.
    camera_model_init_z_scale: Literal["none", "scene_radius"] = "none"
    # * Exclude every lens parameter from the optimizer for the whole run (they keep whatever
    # * value they were initialized with -- zero, or `camera_model_init`'s). `--pose_opt` is
    # * independent and stays trainable.
    camera_model_freeze: bool = False

    # * Per-view SE(3) pose residual. Reported separately: it does NOT transfer to held-out
    # * views, whose poses stay at COLMAP, so it can cost test PSNR even when train improves.
    pose_opt: bool = False
    pose_opt_lr_rotation: float = 1e-5
    pose_opt_lr_translation: float = 1e-5

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

        # * Camera-model transfer / freeze: fail at parse time, not 3000 iterations in.
        if self.camera_model_init is not None or self.camera_model_freeze:
            assert self.camera_opt != "off", (
                "--camera_model_init / --camera_model_freeze need a camera model to act on; "
                "`--camera_opt off` has none (use `passthrough` for a zero residual)."
            )
            assert self.camera_opt != "raxel", (
                "--camera_model_init / --camera_model_freeze do not support the `raxel` rung: "
                "its ray field is created lazily per (uid, resolution), so it is neither in "
                "the lens blocks this loads nor in the parameter groups this freezes."
            )
        if self.camera_model_init is None:
            assert self.camera_model_init_uid_map is None, (
                "--camera_model_init_uid_map is meaningless without --camera_model_init"
            )
            assert self.camera_model_init_rung is None, (
                "--camera_model_init_rung is meaningless without --camera_model_init"
            )
            assert self.camera_model_init_z_scale == "none", (
                "--camera_model_init_z_scale is meaningless without --camera_model_init"
            )
        if self.camera_model_init_uid_map is not None:
            parse_uid_map(self.camera_model_init_uid_map)  # * validate the syntax now


@dataclass
class Config(RaytracerConfig, DatasetConfig):
    def __post_init__(self):
        DatasetConfig.__post_init__(self)
        RaytracerConfig.__post_init__(self)

    def resolved_eval_modes(self) -> List[GrayCameraModelClass]:
        if self.eval_modes:
            return self.eval_modes
        return [self.camera_model]


# ======================================================================================
# Camera-model transfer (`--camera_model_init` / `--camera_model_freeze`)
# ======================================================================================
#
# PLACEMENT NOTE. Conceptually this belongs next to `CameraModel` in gray/camera_model.py.
# It lives here so that the whole feature is contained in the two files train.py already
# depends on before it imports torch, and so that it stays importable (and unit-testable)
# without CUDA. Every heavy import below is therefore FUNCTION-LOCAL: `gray.config` must
# stay torch-free at import time, because train.py parses the CLI before importing torch.
#
# The traps this code exists to make loud, all of them real and all of them silent by
# default (see IMPLEMENTATION.md, limitation 10):
#
#   * uid correspondence  -- parameters are keyed by `cam_info.uid`, one block per physical
#     lens. A source run and a target run need not share uids (myscenes uses uid 1;
#     FullCircle's back-to-back rig uses uids 1 and 2). Positional pairing is NEVER done.
#   * knot count          -- `central_matched` allocates 18 knots (`extra_knots`) against 10
#     everywhere else. Loading 18 into 10 must fail, not truncate.
#   * rung                -- every rung writes the SAME four tensors, so the checkpoint alone
#     cannot say which rung produced it. The source rung is read from the source run's
#     config.json (or declared with --camera_model_init_rung) and its component set must
#     equal the target's, otherwise channels are silently dropped or left at zero.
#   * z units             -- z(theta) is in RAW COLMAP WORLD UNITS. See
#     `camera_model_init_z_scale` above.


def parse_uid_map(spec: Optional[str]) -> Optional[Dict[int, int]]:
    """Parse a `source_uid:target_uid` comma-separated correspondence.

    Raises on anything ambiguous: a repeated source, a repeated target, or a non-integer.
    """
    if spec is None:
        return None
    mapping: Dict[int, int] = {}
    targets = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.count(":") != 1:
            raise ValueError(
                f"camera_model_init_uid_map: expected 'source:target' pairs, got {chunk!r}"
            )
        left, right = chunk.split(":")
        try:
            source, target = int(left), int(right)
        except ValueError:
            raise ValueError(
                f"camera_model_init_uid_map: uids must be integers, got {chunk!r}"
            ) from None
        if source in mapping:
            raise ValueError(f"camera_model_init_uid_map: source uid {source} listed twice")
        if target in targets:
            raise ValueError(f"camera_model_init_uid_map: target uid {target} listed twice")
        mapping[source] = target
        targets.add(target)
    if not mapping:
        raise ValueError("camera_model_init_uid_map is empty")
    return mapping


def resolve_checkpoint_path(path: str) -> str:
    "Accept either a `gaussians_*.safetensors` file or a run directory (highest iteration)."
    if os.path.isdir(path):
        candidates = sorted(
            (int(name.split("_")[-1].split(".")[0]), name)
            for name in os.listdir(path)
            if name.startswith("gaussians_") and name.endswith(".safetensors")
        )
        if not candidates:
            raise FileNotFoundError(f"camera_model_init: no gaussians_*.safetensors in {path}")
        return os.path.join(path, candidates[-1][1])
    if not os.path.exists(path):
        raise FileNotFoundError(f"camera_model_init: {path} does not exist")
    return path


def source_rung_of(checkpoint_path: str) -> str:
    """Read `camera_opt` from the run's config.json, next to the checkpoint.

    The rung is NOT in the checkpoint (every rung writes the same tensors), so this is the
    only honest source of truth. Missing file -> raise, telling the caller to declare it.
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"camera_model_init: no config.json next to {checkpoint_path}, so the source rung "
            "cannot be determined (the checkpoint does not record it -- every rung writes the "
            "same four tensors). Pass --camera_model_init_rung explicitly."
        )
    with open(config_path, "r") as handle:
        source = json.load(handle)
    if "camera_opt" not in source:
        raise KeyError(f"camera_model_init: {config_path} has no `camera_opt` field")
    return source["camera_opt"]


def scene_radius_from_cameras_json(cameras_json_path: str) -> float:
    """Recompute a run's scene radius from the cameras.json it wrote.

    This is exactly `get_nerf_pp_norm(train_cameras)["radius"]` -- 1.1x the largest distance
    from the mean TRAIN camera centre -- which is what train.py feeds to
    `camera_model.scene_scale`. Recomputing it (rather than storing it) means the conversion
    works on every run already on disk, none of which recorded a radius.
    """
    import numpy as np

    with open(cameras_json_path, "r") as handle:
        cameras = json.load(handle)
    centres = [np.asarray(cam["origin"], dtype=np.float64) for cam in cameras if not cam["is_test"]]
    if not centres:
        raise ValueError(f"{cameras_json_path}: no train cameras (is_test == false)")
    stacked = np.stack(centres, axis=0)
    return float(np.linalg.norm(stacked - stacked.mean(axis=0), axis=1).max() * 1.1)


def read_source_lens_blocks(checkpoint_path: str) -> Dict[int, Dict[str, "object"]]:
    "Load the `camera_model.lenses.<uid>.<param>` tensors of a checkpoint, on CPU."
    import safetensors.torch

    state = safetensors.torch.load_file(checkpoint_path, device="cpu")
    blocks: Dict[int, Dict[str, object]] = {}
    for key, value in state.items():
        parts = key.split(".")
        if len(parts) == 4 and parts[0] == "camera_model" and parts[1] == "lenses":
            blocks.setdefault(int(parts[2]), {})[parts[3]] = value
    return blocks


def plan_camera_model_transfer(
    source_blocks: Dict[int, Dict[str, "object"]],
    target_shapes: Dict[int, Dict[str, Tuple[int, ...]]],
    source_rung: str,
    target_rung: str,
    uid_map: Optional[Dict[int, int]] = None,
    z_scale: float = 1.0,
) -> Dict[Tuple[int, str], "object"]:
    """Decide, and validate, what goes where. Pure: no CUDA, no side effects.

    `source_blocks`  {source_uid: {param_name: tensor}}   (from the source checkpoint)
    `target_shapes`  {target_uid: {param_name: shape}}    (from the live camera model)
    Returns          {(target_uid, param_name): tensor}   already z-scaled.

    Raises ValueError on ANY mismatch. Silence is the failure mode this whole function
    exists to prevent, so nothing here is best-effort.
    """
    from gray.camera_model import RUNGS

    for name, rung in (("source", source_rung), ("target", target_rung)):
        if rung not in RUNGS:
            raise ValueError(f"unknown {name} rung {rung!r}; known rungs: {sorted(RUNGS)}")
        if rung == "off":
            raise ValueError(f"{name} rung is `off`: there is no camera model to transfer")

    source_components = set(RUNGS[source_rung])
    target_components = set(RUNGS[target_rung])
    if source_components != target_components:
        raise ValueError(
            f"camera_model_init: rung mismatch. Source {source_rung!r} has components "
            f"{sorted(source_components)}, target {target_rung!r} has "
            f"{sorted(target_components)}. Transferring between different rungs would leave "
            "channels uninitialized (or silently drop trained ones): every rung writes the "
            "same four tensors, so nothing downstream would notice. Refusing."
        )

    if not source_blocks:
        raise ValueError(
            "camera_model_init: the checkpoint contains no `camera_model.lenses.*` tensors "
            "(was it trained with --camera_opt off?)"
        )
    if not target_shapes:
        raise ValueError("camera_model_init: the target model has no lens blocks")

    source_uids = set(source_blocks)
    target_uids = set(target_shapes)
    if uid_map is None:
        if source_uids != target_uids:
            raise ValueError(
                f"camera_model_init: uid mismatch. Source checkpoint has lens uids "
                f"{sorted(source_uids)}, this scene has {sorted(target_uids)}. uids identify "
                "PHYSICAL LENSES and are never paired by position. Give the correspondence "
                "explicitly, e.g. --camera_model_init_uid_map "
                f"{','.join(f'{s}:{t}' for s, t in zip(sorted(source_uids), sorted(target_uids)))}"
            )
        uid_map = {uid: uid for uid in source_uids}
    else:
        unknown_sources = sorted(set(uid_map) - source_uids)
        unknown_targets = sorted(set(uid_map.values()) - target_uids)
        if unknown_sources:
            raise ValueError(
                f"camera_model_init_uid_map: source uids {unknown_sources} are not in the "
                f"checkpoint (it has {sorted(source_uids)})"
            )
        if unknown_targets:
            raise ValueError(
                f"camera_model_init_uid_map: target uids {unknown_targets} are not in this "
                f"scene (it has {sorted(target_uids)})"
            )
        uncovered = sorted(target_uids - set(uid_map.values()))
        if uncovered:
            raise ValueError(
                f"camera_model_init_uid_map: target uids {uncovered} would be left at zero "
                "while the others are initialized from the checkpoint. Map every lens of the "
                "target scene, or none."
            )

    plan: Dict[Tuple[int, str], object] = {}
    for source_uid, target_uid in sorted(uid_map.items()):
        source_lens = source_blocks[source_uid]
        target_lens = target_shapes[target_uid]
        for name in LENS_PARAMETERS:
            if name not in source_lens:
                raise ValueError(
                    f"camera_model_init: source lens {source_uid} has no `{name}` "
                    f"(it has {sorted(source_lens)})"
                )
            if name not in target_lens:
                raise ValueError(f"camera_model_init: target lens {target_uid} has no `{name}`")
            source_shape = tuple(source_lens[name].shape)
            target_shape = tuple(target_lens[name])
            if source_shape != target_shape:
                hint = ""
                if name in ("theta_weights", "phi_weights"):
                    hint = (
                        " -- this is the knot count: `central_matched` allocates "
                        "camera_opt_knots + camera_opt_knots_z (18 by default), every other "
                        "rung allocates camera_opt_knots (10). Truncating would change the "
                        "function the spline represents, so this is refused."
                    )
                elif name == "z_weights":
                    hint = " -- check --camera_opt_knots_z on both runs."
                raise ValueError(
                    f"camera_model_init: shape mismatch on lens {source_uid}->{target_uid} "
                    f"`{name}`: checkpoint {source_shape}, target {target_shape}{hint}"
                )
            tensor = source_lens[name].clone()
            if name == "z_weights" and z_scale != 1.0:
                tensor = tensor * z_scale
            plan[(target_uid, name)] = tensor
    return plan


def apply_camera_model_transfer(camera_model, cfg, uids, target_scene_radius: float) -> dict:
    """Load `cfg.camera_model_init` into `camera_model`. Returns a summary for logging.

    `uids` is every camera uid of the scene (train AND test): the lens blocks are created
    eagerly here, which is also what makes `--camera_model_freeze` total -- a block created
    later would have escaped the freeze.

    Call this AFTER `camera_model.scene_scale` has been set: `CameraModel.lens()` bakes
    `camera_opt_lr_z * scene_scale` into the optimizer group at creation time.
    """
    import torch

    checkpoint_path = resolve_checkpoint_path(cfg.camera_model_init)
    source_rung = cfg.camera_model_init_rung or source_rung_of(checkpoint_path)
    uid_map = parse_uid_map(cfg.camera_model_init_uid_map)
    source_blocks = read_source_lens_blocks(checkpoint_path)

    z_scale = 1.0
    source_radius = None
    if cfg.camera_model_init_z_scale == "scene_radius":
        cameras_json = os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_path)), "cameras.json"
        )
        if not os.path.exists(cameras_json):
            raise FileNotFoundError(
                f"camera_model_init_z_scale=scene_radius needs {cameras_json} to recover the "
                "source scene radius"
            )
        source_radius = scene_radius_from_cameras_json(cameras_json)
        if source_radius <= 0.0:
            raise ValueError(f"source scene radius is {source_radius}, cannot rescale z")
        z_scale = float(target_scene_radius) / source_radius

    # * Create every lens block now, so shapes exist and nothing is created later.
    target_shapes = {}
    for uid in sorted(set(int(uid) for uid in uids)):
        lens = camera_model.lens(uid)
        target_shapes[uid] = {name: tuple(getattr(lens, name).shape) for name in LENS_PARAMETERS}

    plan = plan_camera_model_transfer(
        source_blocks=source_blocks,
        target_shapes=target_shapes,
        source_rung=source_rung,
        target_rung=cfg.camera_opt,
        uid_map=uid_map,
        z_scale=z_scale,
    )

    with torch.no_grad():
        for (target_uid, name), tensor in plan.items():
            parameter = getattr(camera_model.lens(target_uid), name)
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))

    # * MANDATORY, and the one silent failure this function could still have caused.
    # * `CameraModel.forward()` caches the pose-independent half of the ray synthesis per
    # * (camera, resolution) and only invalidates on `step()`, `set_frozen()` and the two
    # * state_dict hooks -- its own docstring says "anything that pokes a parameter tensor by
    # * hand must call invalidate_ray_cache()". The copies above are exactly that poke. In
    # * train.py the cache is provably empty here (nothing has rendered yet), so this is free;
    # * it is what keeps the function correct when called from a script, a test or a later
    # * warm-restart path, where a stale cache would render the PREVIOUS camera and nothing
    # * downstream would notice.
    invalidate = getattr(camera_model, "invalidate_ray_cache", None)
    if callable(invalidate):
        invalidate()

    return {
        "checkpoint": checkpoint_path,
        "source_rung": source_rung,
        "target_rung": cfg.camera_opt,
        "uid_map": {int(s): int(t) for s, t in (uid_map or {u: u for u in target_shapes}).items()},
        "z_scale_mode": cfg.camera_model_init_z_scale,
        "z_scale": z_scale,
        "source_scene_radius": source_radius,
        "target_scene_radius": float(target_scene_radius),
        "tensors_loaded": len(plan),
    }


def freeze_camera_model(camera_model, uids) -> dict:
    """Take every lens parameter out of the optimizer and off autograd, for the whole run.

    Two mechanisms, on purpose: `requires_grad_(False)` stops the gradient (so the third
    backward stage skips the ray tensors entirely, and `regularization()` produces no graph,
    which train.py already tests for), and dropping the parameter groups stops Adam from
    moving anything even if a gradient ever reappeared.

    `--pose_opt` blocks (`poseR:` / `poseT:`) are deliberately left alone: the pose residual
    is selected by its own flag and is not part of the lens model.

    No `invalidate_ray_cache()` here on purpose, and the asymmetry with
    `apply_camera_model_transfer` is deliberate: freezing changes no parameter VALUE, and the
    camera-frame cache is only ever populated under `no_grad` (see `CameraModel.forward`), so
    nothing it holds can become wrong. Adding a lens block can only ADD a key.
    """
    for uid in sorted(set(int(uid) for uid in uids)):
        camera_model.lens(uid)  # * materialize before freezing; nothing may appear later

    frozen_parameters = 0
    for lens in camera_model.lenses.values():
        for parameter in lens.parameters():
            parameter.requires_grad_(False)
            frozen_parameters += parameter.numel()

    lens_group_prefixes = ("tilt:", "angular:", "z:", "raxel:")
    kept = [
        group
        for group in camera_model.optimizer.param_groups
        if not str(group.get("name", "")).startswith(lens_group_prefixes)
    ]
    dropped = len(camera_model.optimizer.param_groups) - len(kept)
    camera_model.optimizer.param_groups = kept
    return {"frozen_parameters": frozen_parameters, "dropped_param_groups": dropped}
