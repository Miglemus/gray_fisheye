from gray.imports import *
from gray.config import RaytracerConfig
from gray.camera import CameraInfo
from gray.scene import SceneInfo, BasicPointCloud
from gray.mlp import PreMLP, PostMLP
from gray.exposure_comp import ExposureComp
from gray.vignetting import Vignetting, should_apply_vignetting
from gray.camera_model import CameraModel


def _find_library_path():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(parent_dir)
    search_dirs = [
        # * In a pip install, the shared library is dumped next to this file.
        parent_dir,
        # * During development, the shared library lives in the project build folder.
        os.path.join(project_dir, "build"),
        os.path.join(project_dir, "build", "Release"),
    ]
    lib_names = ["libgray.so", "gray.dll", "libgray.dylib"]
    candidates = [
        os.path.join(directory, lib_name) for directory in search_dirs for lib_name in lib_names
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    tried_paths = "\n - ".join(candidates)
    raise FileNotFoundError(f"Unable to locate the raytracer library. Tried:\n - {tried_paths}")


class Raytracer(torch.nn.Module):
    LIB_PATH = _find_library_path()
    LOADED = False

    def __init__(
        self,
        cfg: RaytracerConfig,
        num_points: int,
        image_width: int,
        image_height: int,
        inference_only: bool = False,
    ):
        "Note that you should call `from_safetensors` or `from_point_cloud` to initialize the raytracer with actual data."
        super().__init__()

        self.cfg = cfg
        self.image_width = image_width
        self.image_height = image_height

        # * Active render resolution, lower during warmup
        self.render_width = image_width
        self.render_height = image_height

        # * Load the CUDA module
        torch.classes.load_library(Raytracer.LIB_PATH)
        Raytracer.LOADED = True

        self.cuda_module = torch.classes.gray.Raytracer(
            image_width,
            image_height,
            num_points,
            cfg.sh_max_degree,
            cfg.ppll_forward_size,
            cfg.ppll_backward_size,
            inference_only,
        )

        # * Decide background color
        self.bg_color = torch.tensor(cfg.bg_color)

        # * Setup config
        config = self.cuda_module.get_config()
        config.alpha_threshold.fill_(cfg.alpha_threshold)
        config.t_threshold.fill_(cfg.t_threshold)
        config.exp_power.fill_(cfg.exp_power)
        config.render_depth.fill_(cfg.render_depth)
        config.background_channels.copy_(self.bg_color)
        config.enable_sh.fill_(cfg.sh)
        config.needs_ray_output.fill_(cfg.post_mlp)

        # * Only optimize the channels if they aren't a viewpoint-depenent output color
        config.update_channels.fill_(not cfg.pre_mlp and not cfg.sh)

        # * Zero the Adam moment and gradient buffers: they come from raw CUDA
        # * allocations and can contain garbage (incl. NaN), which poisons the
        # * first optimizer steps (observed deterministically on FIORD night_out).
        gaussians = self.cuda_module.get_gaussians()
        for _field in (
            "mean",
            "scale",
            "rotation",
            "opacity",
            "channels",
            "sh_coeffs_dc",
            "sh_coeffs_rest",
        ):
            for _prefix in ("grad_", "first_moment_", "second_moment_"):
                _buf = getattr(gaussians, _prefix + _field, None)
                if _buf is not None:
                    _buf.zero_()

        # * Set learning rates
        gaussians.lr_rotation.fill_(cfg.lr_rotation_init)
        gaussians.lr_scale.fill_(cfg.lr_scale_init)
        gaussians.lr_mean.fill_(cfg.lr_mean_init)
        gaussians.lr_opacity.fill_(cfg.lr_opacity_init)
        gaussians.lr_channels.fill_(cfg.lr_channels)
        gaussians.lr_sh_dc.fill_(cfg.lr_sh_dc_init)
        gaussians.lr_sh_rest.fill_(cfg.lr_sh_rest)

        # * Set Adam parameters
        gaussians.beta_1.fill_(cfg.beta_1)
        gaussians.beta_2.fill_(cfg.beta_2)
        gaussians.epsilon.fill_(cfg.epsilon)
        gaussians.sh_update_laziness.fill_(cfg.sh_update_laziness)

        # * Setup MLP
        num_channels = self.cuda_module.get_num_channels()
        if num_channels != 3:
            assert cfg.post_mlp, (
                "Post-processing MLP must be enabled if the number of output channels is not 3"
            )
        if cfg.pre_mlp:
            self.pre_mlp = PreMLP(cfg, gaussians).cuda()
        if cfg.post_mlp:
            self.post_mlp = PostMLP(cfg, num_channels).cuda()

        # * Setup global vignetting compensation (registered as a submodule so its
        # * parameters are saved/restored with the safetensors state_dict)
        if cfg.vignetting_comp:
            self.vignetting = Vignetting(cfg)
            if cfg.load_vignetting is not None and not inference_only:
                self.vignetting.load_parameters(cfg.load_vignetting)
                self.vignetting.set_lrs(0.0, 0.0)

        # * Setup the learnable residual camera model. Only instantiated when enabled so the
        # * state_dict (and therefore every existing checkpoint) is untouched by default.
        self.camera_model = None
        if cfg.camera_opt != "off":
            self.camera_model = CameraModel(cfg)
        self._base_bearing_cache = {}

        # * Last outputs, kept for backward pass
        self.output_channels = None
        self._ray_origin = None
        self._ray_direction = None

    def init_exposure_comp(self, scene_info: SceneInfo):
        self.exposure_comp = ExposureComp(self.cfg, scene_info)

    def scaled_intrinsics(self, cam_info: CameraInfo):
        """(model, intrinsics at the active render resolution) -- None for a pinhole camera.

        Intrinsics are stored for the full image; only the focal length and the principal
        point rescale, the distortion coefficients act on normalized coordinates.
        """
        from gray.camera_models import normalize_gray_model

        model = normalize_gray_model(getattr(cam_info, "model", "pinhole"))
        if model not in ["opencv_fisheye", "thin_prism_fisheye", "rad_tan_thin_prism_fisheye"]:
            return model, None
        intrinsics = cam_info.intrinsics_cuda().clone()
        scale_x = self.render_width / cam_info.image_width
        scale_y = self.render_height / cam_info.image_height
        intrinsics[0] *= scale_x  # fx
        intrinsics[2] *= scale_x  # cx
        intrinsics[1] *= scale_y  # fy
        intrinsics[3] *= scale_y  # cy
        return model, intrinsics

    def upload_camera_intrinsics(self, cam_info: CameraInfo):
        "Push the camera model and its intrinsics, rescaled to the active render resolution."
        camera = self.cuda_module.get_camera()
        camera.vertical_fov_radians.fill_(cam_info.fov_y)
        model, intrinsics = self.scaled_intrinsics(cam_info)
        if intrinsics is not None:
            if model == "opencv_fisheye":
                camera.set_opencv_fisheye(intrinsics)
            elif model == "thin_prism_fisheye":
                camera.set_thin_prism_fisheye(intrinsics)
            else:
                camera.set_rad_tan_thin_prism_fisheye(intrinsics)
        else:
            camera.set_pinhole()

    def base_bearings(self, cam_info: CameraInfo):
        """Base per-pixel bearings in the OpenCV camera frame, probed from the raygen itself.

        Bit-exactness matters: the whole ablation ladder is only meaningful if the
        zero-residual rung reproduces the native path. Rather than re-implementing the
        COLMAP unprojection in torch (and inheriting the `converged` latch bug in
        gray/fisheye_geometry.py), we render one throw-away frame with an identity pose and
        read the bearings the raygen actually produced.

        Cached on everything `upload_camera_intrinsics()` pushes -- never on the pose, which
        a bearing does not depend on.

        The key must include the camera MODEL and its intrinsics, not just the uid and the
        render resolution. `render.py --eval-models pinhole rad_tan_thin_prism_fisheye`
        renders several camera models for the same camera uid in one process, and on a scene
        whose pinhole copy happens to have the same pixel size as its fisheye originals
        (`undistort_consistent.py --width 1440 --height 1080`) a uid+resolution key silently
        hands the second eval mode the FIRST one's bearings. Measured on
        `workshop_immervision`, where both are 1440x1080: the fisheye pass re-rendered at
        13.31 dB against the 25.37 the same model reached live during training. It went
        unnoticed on myscenes only because there the pinhole copy is 400x266 while the
        fisheye is 1368x912, so the resolution alone happened to separate them.
        """
        height, width = self.render_height, self.render_width
        intrinsics = getattr(cam_info, "intrinsics", None)
        cache_key = (
            cam_info.uid, height, width,
            getattr(cam_info, "model", "pinhole"),
            float(cam_info.fov_y), cam_info.image_width, cam_info.image_height,
            None if intrinsics is None else tuple(np.asarray(intrinsics).ravel().tolist()),
        )
        cached = self._base_bearing_cache.get(cache_key)
        if cached is not None:
            return cached

        camera = self.cuda_module.get_camera()
        config = self.cuda_module.get_config()
        framebuffer = self.cuda_module.get_framebuffer()
        previous_ray_output = bool(config.needs_ray_output.item())
        previous_zfar = float(camera.zfar.item())

        # * Self-sufficient on purpose. Depending on a prior __call__ to have uploaded the
        # * intrinsics is a silent trap: an unconfigured camera is PINHOLE with fov_y = 0,
        # * which probes as theta == 0 for every pixel instead of failing.
        self.upload_camera_intrinsics(cam_info)

        with torch.no_grad():
            config.needs_ray_output.fill_(True)
            config.rays_from_python.fill_(False)
            # * Nothing has to be hit; we only want the raygen to emit its bearings.
            camera.zfar.fill_(1e-6)
            camera.set_pose(torch.zeros(3, device="cuda"), torch.eye(3, device="cuda"))
            # * Outside the lens disk the no-grad path never writes ray_direction, so the
            # * zero sentinel has to come from the buffer itself.
            framebuffer.ray_direction.zero_()
            self.cuda_module.forward_pass()
            probed = framebuffer.ray_direction[:height, :width].detach().clone()

        config.needs_ray_output.fill_(previous_ray_output)
        camera.zfar.fill_(previous_zfar)

        # * The raygen emits (x, -y, -z) of the OpenCV bearing, then rotates by c2w_blender.
        # * With an identity pose that rotation is a no-op, so undoing the flip recovers the
        # * OpenCV-frame bearing. (The same two flips cancel in the forward direction, which
        # * is why `d_world = cam_info.R @ bearing` below has no sign fixup.)
        bearings = probed * torch.tensor([1.0, -1.0, -1.0], device="cuda")

        valid = bearings.norm(dim=-1) > 0.5
        bearing_x, bearing_y, bearing_z = bearings.unbind(-1)
        radius = torch.sqrt(bearing_x * bearing_x + bearing_y * bearing_y)
        safe_radius = radius.clamp_min(1e-12)
        zeros = torch.zeros_like(radius)
        cos_phi = torch.where(valid, bearing_x / safe_radius, zeros)
        sin_phi = torch.where(valid, bearing_y / safe_radius, zeros)
        theta = torch.atan2(radius, bearing_z)
        model, scaled = self.scaled_intrinsics(cam_info)
        base = {
            "bearings": bearings,
            # * The calibration these bearings came from, at THIS render resolution: what
            # * the rttpf rung re-fits. Part of the base dict rather than re-derived in the
            # * camera model so that the two can never disagree about the scaling.
            "model": model,
            "intrinsics": scaled,
            # * Axis of the meridional rotation; orthogonal to the bearing by construction.
            "meridian": torch.stack([-sin_phi, cos_phi, zeros], dim=-1),
            "theta01": (theta / (math.pi / 2.0)).clamp(0.0, 1.0),
            "cos_phi": cos_phi,
            "sin_phi": sin_phi,
            "valid": valid.to(bearings.dtype),
            # * Identifies this (camera, resolution); lets the camera model cache anything
            # * derived from theta_base, which never changes for a given key.
            "cache_key": cache_key,
        }
        self._base_bearing_cache[cache_key] = base
        return base

    @staticmethod
    def _rotation_c2w_cuda(cam_info: CameraInfo):
        "COLMAP c2w rotation (OpenCV convention) as a cached CUDA tensor."
        tensor = getattr(cam_info, "_rotation_c2w_opencv_cuda", None)
        if tensor is None:
            tensor = torch.from_numpy(np.asarray(cam_info.R, dtype=np.float32)).cuda()
            cam_info._rotation_c2w_opencv_cuda = tensor
        return tensor

    def __call__(
        self,
        cam_info: CameraInfo,
        znear=0.0,
        zfar=99999.9,
        skip_copy=False,
    ):
        "Render the scene and takes an optimization step if a target is provided."

        # * Set camera parameters
        camera = self.cuda_module.get_camera()
        camera.znear.fill_(znear)
        camera.zfar.fill_(zfar)
        config = self.cuda_module.get_config()
        config.rays_from_python.fill_(False)
        self.upload_camera_intrinsics(cam_info)

        # * Probe the base bearings *before* setting the real pose: the probe needs an
        # * identity pose, and it needs the intrinsics that were just uploaded.
        base = self.base_bearings(cam_info) if self.camera_model is not None else None

        camera.set_pose(cam_info.origin_cuda(), cam_info.rotation_c2w_blender_cuda())

        # * Learnable camera model: synthesize the per-pixel rays in torch and hand them to
        # * the raygen through the framebuffer. Everything upstream of the ray -- intrinsic
        # * residual, non-central origin, per-view pose -- is then plain autograd, closed by
        # * the dL/d(ray) that cuda/backward_pass.cu now writes back.
        self._ray_origin = None
        self._ray_direction = None
        if self.camera_model is not None:
            base = dict(base)
            base["rotation"] = Raytracer._rotation_c2w_cuda(cam_info)
            base["origin"] = cam_info.origin_cuda()
            ray_origin, ray_direction = self.camera_model(
                cam_info, base, self.render_height, self.render_width
            )
            height, width = self.render_height, self.render_width
            framebuffer = self.cuda_module.get_framebuffer()
            with torch.no_grad():
                framebuffer.ray_origin[:height, :width].copy_(ray_origin.detach())
                framebuffer.ray_direction[:height, :width].copy_(ray_direction.detach())
            config.rays_from_python.fill_(True)
            if torch.is_grad_enabled():
                self._ray_origin = ray_origin
                self._ray_direction = ray_direction

        # * Set gaussian colors from view direction MLP
        if self.cfg.pre_mlp:
            self.pre_mlp(cam_info)

        # * Render and step if required
        framebuffer = self.cuda_module.get_framebuffer()
        grad_enabled = torch.is_grad_enabled()
        assert not (skip_copy and grad_enabled), (
            "skip_copy=True is not supported with gradients enabled"
        )
        assert not (config.render_ellipsoids.item() and grad_enabled), (
            "render_ellipsoids=True is only supported for no-grad display renders"
        )

        self.cuda_module.forward_pass()

        # * Slice out the active top-left rectangle (the whole buffer when at full resolution)
        h, w = self.render_height, self.render_width
        output_channels = framebuffer.output_channels.detach()[:h, :w]
        if not skip_copy or grad_enabled:
            output_channels = output_channels.clone()
        output_channels = output_channels.moveaxis(-1, 0)

        if grad_enabled:
            assert self.output_channels is None, (
                "Called the forward pass multiple times without a backward pass"
            )
            output_channels.requires_grad_()
            self.output_channels = output_channels

        # * Apply post-processing MLP
        if self.cfg.post_mlp:
            ray_direction = framebuffer.ray_direction.detach()[:h, :w].moveaxis(-1, 0)
            depth = framebuffer.output_depth.detach()[:h, :w].moveaxis(-1, 0)
            hit_point = cam_info.origin_cuda()[:, None, None] + depth * ray_direction
            render = self.post_mlp(output_channels, hit_point, ray_direction)
        else:
            render = output_channels

        # * Apply the global vignetting model only for the training camera model
        if should_apply_vignetting(self.cfg, getattr(cam_info, "model", "pinhole")):
            render = self.vignetting(render)

        return render

    def backward(self, loss):
        # * Backprop from loss to raytracer (and other parameters forming the loss)
        loss.backward()
        with torch.no_grad():
            framebuffer = self.cuda_module.get_framebuffer()
            h, w = self.render_height, self.render_width
            framebuffer.grad_output_channels[:h, :w].copy_(
                self.output_channels.grad.moveaxis(0, -1)
            )
            self.output_channels = None

        # * Backprop raytracer
        self.cuda_module.backward_pass()

        # * Third stage: dL/d(ray) -> the camera-model parameters. Deliberately a plain
        # * autograd.backward rather than an autograd.Function, so gray's documented
        # * invariant (forward-pass state stays intact until backward()) still holds, and
        # * so the gradients accumulate across the cameras of a batch for free.
        if self._ray_origin is not None:
            with torch.no_grad():
                grad_ray_origin = framebuffer.grad_ray_origin[:h, :w].clone()
                grad_ray_direction = framebuffer.grad_ray_direction[:h, :w].clone()
            # * Only the tensors that actually carry a graph. The ray origin is a plain
            # * broadcast of the camera centre unless the non-central or pose rungs are on,
            # * and `passthrough` gives neither a graph -- autograd.backward raises on any
            # * input without a grad_fn, so the rungs must be filtered, not assumed.
            tensors, gradients = [], []
            for tensor, gradient in (
                (self._ray_origin, grad_ray_origin),
                (self._ray_direction, grad_ray_direction),
            ):
                if tensor.requires_grad:
                    tensors.append(tensor)
                    gradients.append(gradient)
            if tensors:
                torch.autograd.backward(tensors, gradients)
        self._ray_origin = None
        self._ray_direction = None

    def step(self):
        # * Update stats
        self.update_pruning_stats()

        # * Optimization steps
        if self.cfg.pre_mlp:
            self.pre_mlp.step()
        self.cuda_module.step()
        self.cuda_module.update_bvh()
        if self.cfg.post_mlp:
            self.post_mlp.step()
        if self.cfg.exposure_comp_enabled:
            self.exposure_comp.step()
        if self.cfg.vignetting_comp:
            self.vignetting.step()
        if self.camera_model is not None:
            self.camera_model.step()

    def set_render_resolution(self, width: int, height: int):
        "Render at a reduced resolution (must not exceed the allocated framebuffer size)."
        self.render_width = width
        self.render_height = height
        self.cuda_module.set_render_resolution(width, height)

    @staticmethod
    def from_point_cloud(
        cfg: RaytracerConfig,
        point_cloud: BasicPointCloud,
        image_width: int,
        image_height: int,
        inference_only: bool = False,
    ):
        print(f"Initializing {point_cloud.points.shape[0]} points")
        if cfg.init_binning:
            points = torch.from_numpy(point_cloud.points).cuda()
            colors = torch.from_numpy(point_cloud.colors).cuda()
            distances = torch.from_numpy(point_cloud.distances_to_cam).cuda().unsqueeze(1)

            rounded_points = (points / (cfg.init_bin_size * point_cloud.radius)).round()
            unique_pts, inverse, counts = rounded_points.unique(
                dim=0, sorted=True, return_counts=True, return_inverse=True
            )
            num_points = unique_pts.shape[0]

            color_sums = torch.zeros((num_points, 3), device=colors.device)
            color_sums.index_add_(0, inverse, colors)

            position_sums = torch.zeros((num_points, 3), device=points.device)
            position_sums.index_add_(0, inverse, points)

            distance_sums = torch.zeros((num_points, 1), device=points.device)
            distance_sums.index_add_(0, inverse, distances)

            avg_colors = color_sums / counts.unsqueeze(1)
            avg_positions = position_sums / counts.unsqueeze(1)
            avg_distances = distance_sums / counts.unsqueeze(1)

            point_cloud = BasicPointCloud(
                points=avg_positions.cpu().numpy(),
                colors=avg_colors.cpu().numpy(),
                distances_to_cam=avg_distances.squeeze(1).cpu().numpy(),
                radius=point_cloud.radius,
                normals=None,
            )
            print(f"Binning down to {num_points} points")

        torch.cuda.synchronize()  # *** Important for some reason

        num_points = point_cloud.points.shape[0]
        num_orig_points = num_points

        raytracer = Raytracer(
            cfg,
            num_points,
            image_width,
            image_height,
            inference_only=inference_only,
        )
        gaussians = raytracer.cuda_module.get_gaussians()

        rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_points, 1)
        scale = (
            torch.from_numpy(point_cloud.distances_to_cam * cfg.init_scale)
            .cuda()
            .unsqueeze(1)
            .log()
        )
        mean = torch.from_numpy(point_cloud.points).cuda()
        opacity = torch.logit(cfg.init_opacity * torch.ones(num_points, 1))
        channels = torch.cat(
            [
                torch.from_numpy(point_cloud.colors).cuda(),
                torch.randn(num_orig_points, raytracer.cuda_module.get_num_channels() - 3).cuda()
                / 2
                + 0.5,
            ],
            dim=1,
        )
        sh_coeffs_dc = (
            torch.from_numpy(point_cloud.colors).cuda().unsqueeze(1) - 0.5
        ) / 0.28209479177387814

        gaussians.rotation.copy_(rotation)
        gaussians.scale.copy_(scale)
        gaussians.mean.copy_(mean)
        gaussians.opacity.copy_(opacity)
        gaussians.channels.copy_(channels)
        gaussians.sh_coeffs_dc.copy_(sh_coeffs_dc)
        raytracer.cuda_module.rebuild_bvh()

        if cfg.pre_mlp:
            raytracer.pre_mlp.initialize()

        return raytracer

    @torch.no_grad()
    def prune(self, iteration: int, mask: Optional[torch.Tensor] = None):
        gaussians = self.cuda_module.get_gaussians()

        mean = gaussians.mean.clone()
        channels = gaussians.channels.clone()
        opacity = gaussians.opacity.clone()
        rotation = gaussians.rotation.clone()
        scale = gaussians.scale.clone()
        sh_coeffs_dc = gaussians.sh_coeffs_dc.clone()
        sh_coeffs_rest = gaussians.sh_coeffs_rest.clone()

        first_moment_mean = gaussians.first_moment_mean.clone()
        first_moment_rotation = gaussians.first_moment_rotation.clone()
        first_moment_scale = gaussians.first_moment_scale.clone()
        first_moment_opacity = gaussians.first_moment_opacity.clone()
        first_moment_channels = gaussians.first_moment_channels.clone()
        first_moment_sh_coeffs_dc = gaussians.first_moment_sh_coeffs_dc.clone()
        first_moment_sh_coeffs_rest = gaussians.first_moment_sh_coeffs_rest.clone()

        second_moment_mean = gaussians.second_moment_mean.clone()
        second_moment_rotation = gaussians.second_moment_rotation.clone()
        second_moment_scale = gaussians.second_moment_scale.clone()
        second_moment_opacity = gaussians.second_moment_opacity.clone()
        second_moment_channels = gaussians.second_moment_channels.clone()
        second_moment_sh_coeffs_dc = gaussians.second_moment_sh_coeffs_dc.clone()
        second_moment_sh_coeffs_rest = gaussians.second_moment_sh_coeffs_rest.clone()

        if mask is None:
            denom = gaussians.pruning_counter.squeeze(1).clamp(min=1)
            average_weight = gaussians.pruning_weight.squeeze(1) / denom
            mask = average_weight >= self.cfg.pruning_min_weight

        self.cuda_module.resize(mask.sum().item())

        gaussians.mean.copy_(mean[mask])
        gaussians.channels.copy_(channels[mask])
        gaussians.opacity.copy_(opacity[mask])
        gaussians.rotation.copy_(rotation[mask])
        gaussians.scale.copy_(scale[mask])
        gaussians.sh_coeffs_dc.copy_(sh_coeffs_dc[mask])
        gaussians.sh_coeffs_rest.copy_(sh_coeffs_rest[mask])

        gaussians.first_moment_mean.copy_(first_moment_mean[mask])
        gaussians.first_moment_rotation.copy_(first_moment_rotation[mask])
        gaussians.first_moment_scale.copy_(first_moment_scale[mask])
        gaussians.first_moment_opacity.copy_(first_moment_opacity[mask])
        gaussians.first_moment_channels.copy_(first_moment_channels[mask])
        gaussians.first_moment_sh_coeffs_dc.copy_(first_moment_sh_coeffs_dc[mask])
        gaussians.first_moment_sh_coeffs_rest.copy_(first_moment_sh_coeffs_rest[mask])

        gaussians.second_moment_mean.copy_(second_moment_mean[mask])
        gaussians.second_moment_rotation.copy_(second_moment_rotation[mask])
        gaussians.second_moment_scale.copy_(second_moment_scale[mask])
        gaussians.second_moment_opacity.copy_(second_moment_opacity[mask])
        gaussians.second_moment_channels.copy_(second_moment_channels[mask])
        gaussians.second_moment_sh_coeffs_dc.copy_(second_moment_sh_coeffs_dc[mask])
        gaussians.second_moment_sh_coeffs_rest.copy_(second_moment_sh_coeffs_rest[mask])

        gaussians.pruning_weight.zero_()
        gaussians.pruning_counter.zero_()

        if self.cfg.pre_mlp:
            self.pre_mlp.prune(mask)

    @torch.no_grad()
    def update_pruning_stats(self):
        gaussians = self.cuda_module.get_gaussians()
        mask = gaussians.was_visible.squeeze(1)

        gaussians.pruning_counter[mask] += 1

    @staticmethod
    def from_safetensors(
        cfg: RaytracerConfig,
        path: str,
        image_width: int,
        image_height: int,
        inference_only: bool = False,
    ):
        state_dict = safetensors.torch.load_file(path)
        # * Legacy support, values now stored in config.json and cameras.json
        state_dict.pop("bg_color", None)
        state_dict.pop("image_width", None)
        state_dict.pop("image_height", None)

        num_points = state_dict["mean"].shape[0]
        raytracer = Raytracer(
            cfg,
            num_points,
            image_width,
            image_height,
            inference_only=inference_only,
        )
        gaussians = raytracer.cuda_module.get_gaussians()
        gaussians.mean.copy_(state_dict["mean"])
        gaussians.rotation.copy_(state_dict["rotation"])
        gaussians.scale.copy_(state_dict["scale"])
        gaussians.opacity.copy_(state_dict["opacity"])
        if cfg.sh:
            gaussians.channels.zero_()
            gaussians.sh_coeffs_dc.copy_(state_dict["sh_coeffs_dc"])
            gaussians.sh_coeffs_rest.copy_(state_dict["sh_coeffs_rest"])
            gaussians.current_sh_degree.copy_(state_dict["current_sh_degree"])
        else:
            gaussians.channels.copy_(state_dict["channels"])
            gaussians.sh_coeffs_dc.zero_()
            gaussians.sh_coeffs_rest.zero_()
            gaussians.current_sh_degree.zero_()
        raytracer.cuda_module.rebuild_bvh()

        del state_dict["mean"]
        del state_dict["rotation"]
        del state_dict["scale"]
        del state_dict["opacity"]
        state_dict.pop("channels", None)
        state_dict.pop("sh_coeffs_dc", None)
        state_dict.pop("sh_coeffs_rest", None)
        state_dict.pop("current_sh_degree", None)

        if cfg.pre_mlp:
            raytracer.pre_mlp.initialize()

        if raytracer.camera_model is not None:
            raytracer.camera_model.materialize_from_state_dict(state_dict)

        raytracer.load_state_dict(state_dict)
        return raytracer

    def save_safetensors(self, model_path: str, iteration: int):
        gaussians = self.cuda_module.get_gaussians()
        tensors = {
            "mean": gaussians.mean,
            "rotation": gaussians.rotation,
            "scale": gaussians.scale,
            "opacity": gaussians.opacity,
        }
        if self.cfg.sh:
            tensors["sh_coeffs_dc"] = gaussians.sh_coeffs_dc
            tensors["sh_coeffs_rest"] = gaussians.sh_coeffs_rest
            tensors["current_sh_degree"] = gaussians.current_sh_degree
        else:
            tensors["channels"] = gaussians.channels

        path = os.path.join(model_path, f"gaussians_{iteration:05d}.safetensors")
        safetensors.torch.save_file({**self.state_dict(), **tensors}, path)
