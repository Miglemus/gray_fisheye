"""Learnable residual camera model for gray.

gray is a ray tracer: a camera model is nothing but a `pixel -> (origin, direction)` map.
That makes an arbitrary -- even *non-central* -- camera free at trace time, because the BVH
lives in world space and `__intersection__is` reads `optixGetObjectRayOrigin()` per ray.
The only thing gray was missing to *learn* such a model was `dL/d(ray)`, which
`cuda/backward_pass.cu` now produces.

Design notes that are easy to get wrong and are load-bearing here:

* **The base bearings are probed from the raygen itself.** We never re-implement the COLMAP
  unprojection in torch. `Raytracer.base_bearings()` renders one throw-away frame with an
  identity pose and reads `framebuffer.ray_direction`, so the base field is bit-for-bit the
  one the native path traces. Ablation rung 0 -> 1 is therefore rigorously controlled.

* **The residual is applied as rotations of the base bearing, not as a (theta, phi)
  round-trip.** `b = b0 cos(d) + (n x b0) sin(d)` is *exactly* `b0` when `d == 0`, whereas
  `b -> (theta, phi) -> b` loses ~1e-7 in float32. `theta_base` is then only ever used as a
  (detached) spline coordinate, so its precision does not matter at all.

* **Everything is a rotation**, so bearings stay unit-norm by construction and the invalid
  sentinel (a zero bearing outside the lens disk) is preserved automatically: rotating the
  zero vector gives the zero vector.

* **The gauge `z(0) = 0`** is imposed exactly as `z(t) = B(t).w - B(0).w`. Without it the
  non-central profile absorbs a per-camera axial translation and is degenerate with the pose.

Coordinates: `b` is a unit bearing in the OpenCV camera frame (x right, y down, z forward).
gray's raygen flips it to (x, -y, -z) and rotates by `rotation_c2w_blender`; those two flips
cancel exactly, so `d_world = cam_info.R @ b` with `cam_info.R` the COLMAP c2w rotation.
`tests/test_camera_model_passthrough.py` pins that down numerically rather than on faith.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

# * Escape hatch for the pose-independent ray cache (see `CameraModel.camera_frame`).
# * Deliberately an environment variable rather than a config field: `gray/config.py` is
# * edited by several parallel sessions and a new flag there would collide. Set
# * GRAY_NO_RAY_CACHE=1 to force the un-cached path; `tests/test_camera_model_cache.py`
# * toggles `CameraModel.ray_cache_enabled` directly instead.
RAY_CACHE_DEFAULT = os.environ.get("GRAY_NO_RAY_CACHE", "") not in ("1", "true", "True")

# * Which residual components each ablation rung switches on.
RUNGS: dict[str, tuple[str, ...]] = {
    "off": (),
    "passthrough": (),
    "tilt": ("tilt",),
    "radial": ("tilt", "radial"),
    "ana": ("tilt", "radial", "ana"),
    "noncentral": ("tilt", "radial", "ana", "z"),
    # * Two *subtractive* rungs, asking whether the non-central term can carry the model on
    # * its own. They are not part of the cumulative ladder above: they remove capacity that
    # * `noncentral` has. The point is that z(theta) sin(theta) * E[1/t | theta] -- the mean
    # * over depth of the shift a non-central pupil induces -- has exactly the form of a
    # * central radial correction, so `z` and `radial` are NOT orthogonal and `z` can be
    # * pulled into doing a job it was not meant to do. See IMPLEMENTATION.md.
    "noncentral_no_ana": ("tilt", "radial", "z"),  # * drop only the anamorphic harmonics
    "z_only": ("z",),  # * the non-central profile alone, 8 parameters
    # * Parameter-matched central control: same budget as `noncentral`, spent entirely on
    # * central degrees of freedom. Without this the non-central gain is not attributable.
    "central_matched": ("tilt", "radial", "ana", "extra_knots"),
    "raxel": ("tilt", "raxel"),
}

# * Harmonic channels for the angular residuals: k=0 (radial), then cos/sin of phi and 2phi.
RADIAL_CHANNELS = 1
ANA_CHANNELS = 4
TOTAL_CHANNELS = RADIAL_CHANNELS + ANA_CHANNELS


def bspline_basis(u: torch.Tensor) -> tuple[torch.Tensor, ...]:
    "Uniform cubic B-spline basis functions on a unit segment."
    u2 = u * u
    u3 = u2 * u
    return (
        (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
        (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
        (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
        u3 / 6.0,
    )


def build_bspline_basis(t01: torch.Tensor, num_ctrl: int) -> torch.Tensor:
    """Dense per-sample basis matrix [N, K] for a uniform cubic B-spline.

    Built once and cached, because it depends only on theta_base, which is fixed for a
    given (camera, resolution). This exists purely for speed, and the speed matters a lot:
    evaluating the spline by advanced indexing (`weights[:, idx]`) makes its backward a
    scatter-add of ~5M values into ~50 addresses, i.e. near-total atomic contention.
    Measured at 512x512: 130 ms/iteration with gathers versus 3.4 ms for the native path.
    Contracting against this matrix turns the same evaluation into a dense GEMM.
    """
    num_segments = max(num_ctrl - 3, 1)
    flat = t01.reshape(-1).clamp(0.0, 1.0) * num_segments
    idx = flat.floor().long().clamp(0, num_segments - 1)
    taps = bspline_basis(flat - idx.to(flat.dtype))
    basis = t01.new_zeros(flat.numel(), num_ctrl)
    for offset, tap in enumerate(taps):
        column = (idx + offset).clamp(max=num_ctrl - 1).unsqueeze(1)
        basis.scatter_add_(1, column, tap.unsqueeze(1))
    return basis


def bspline_eval_basis(weights: torch.Tensor, basis: torch.Tensor, shape) -> torch.Tensor:
    "Evaluate [C, K] control points against a cached [N, K] basis; returns [C, *shape]."
    return (basis @ weights.transpose(0, 1)).transpose(0, 1).reshape(weights.shape[0], *shape)


def bspline_eval(weights: torch.Tensor, t01: torch.Tensor) -> torch.Tensor:
    """Evaluate a uniform cubic B-spline.

    weights: [C, K] control points, t01: [...] in [0, 1]. Returns [C, ...].
    Zero control points give exactly zero, which is what makes the residual init exact.
    """
    num_ctrl = weights.shape[-1]
    num_segments = max(num_ctrl - 3, 1)
    t = t01.clamp(0.0, 1.0) * num_segments
    idx = t.floor().long().clamp(0, num_segments - 1)
    u = t - idx.to(t.dtype)
    b0, b1, b2, b3 = bspline_basis(u)
    return (
        b0 * weights[:, idx]
        + b1 * weights[:, (idx + 1).clamp(max=num_ctrl - 1)]
        + b2 * weights[:, (idx + 2).clamp(max=num_ctrl - 1)]
        + b3 * weights[:, (idx + 3).clamp(max=num_ctrl - 1)]
    )


def bspline_subdivide(weights: torch.Tensor) -> torch.Tensor:
    """Lane-Riesenfeld subdivision for uniform cubic B-splines.

    Doubles the control-point resolution while representing *exactly* the same function,
    which is what lets coarse-to-fine unlock high frequencies without a discontinuity in
    the training curve. weights: [C, K] -> [C, 2K - 3].
    """
    num_ctrl = weights.shape[-1]
    assert num_ctrl >= 4, "cubic B-spline needs at least 4 control points"
    left, mid, right = weights[:, :-2], weights[:, 1:-1], weights[:, 2:]
    even = (left + 6.0 * mid + right) / 8.0  # [C, K-2]
    odd = (weights[:, :-1] + weights[:, 1:]) / 2.0  # [C, K-1]
    out = weights.new_zeros(weights.shape[0], 2 * num_ctrl - 3)
    out[:, 0::2] = odd
    out[:, 1::2] = even
    return out


def skew_rotate(omega: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation of [..., 3] vectors by an axis-angle vector.

    R v = v + k1 (w x v) + k2 (w x (w x v)), with k1 = sin|w|/|w| and k2 = (1-cos|w|)/|w|^2
    evaluated by their Taylor series in |w|^2. That is deliberate on three counts: it is
    exactly the identity at w = 0 (k1 = 1, k2 = 1/2) so the residual init is exact; it has
    no branch on the magnitude, hence **no GPU->CPU sync** -- the naive `if float(|w|) < eps`
    version cost 4 ms per training iteration, more than the whole native ray generation;
    and it stays well conditioned as |w| -> 0 where axis = w/|w| is undefined.
    Camera tilts here are milliradians; the series is good to ~1e-8 up to |w| ~ 0.5 rad.
    """
    squared = (omega * omega).sum()
    k1 = 1.0 - squared / 6.0 + squared * squared / 120.0
    k2 = 0.5 - squared / 24.0 + squared * squared / 720.0
    expanded = omega.expand_as(vectors)
    first = torch.cross(expanded, vectors, dim=-1)
    second = torch.cross(expanded, first, dim=-1)
    return vectors + k1 * first + k2 * second


class LensResidual(nn.Module):
    """Shared residual parameters for one physical lens (one COLMAP camera uid)."""

    def __init__(
        self,
        components: tuple[str, ...],
        num_knots: int,
        num_knots_z: int,
        device: str = "cuda",
    ):
        super().__init__()
        self.components = set(components)
        knots = num_knots + (num_knots_z if "extra_knots" in self.components else 0)
        self.num_knots = knots

        self.omega = nn.Parameter(torch.zeros(3, device=device))
        self.theta_weights = nn.Parameter(torch.zeros(TOTAL_CHANNELS, knots, device=device))
        self.phi_weights = nn.Parameter(torch.zeros(TOTAL_CHANNELS, knots, device=device))
        self.z_weights = nn.Parameter(torch.zeros(1, num_knots_z, device=device))

    def active_channels(self) -> list[int]:
        channels = []
        if "radial" in self.components or "extra_knots" in self.components:
            channels.append(0)
        if "ana" in self.components:
            channels.extend(range(RADIAL_CHANNELS, TOTAL_CHANNELS))
        return channels

    def parameter_groups(self, lrs: dict[str, float]) -> list[dict]:
        groups = []
        if "tilt" in self.components:
            groups.append({"params": [self.omega], "lr": lrs["tilt"], "name": "tilt"})
        if self.active_channels():
            groups.append(
                {
                    "params": [self.theta_weights, self.phi_weights],
                    "lr": lrs["angular"],
                    "name": "angular",
                }
            )
        if "z" in self.components:
            groups.append({"params": [self.z_weights], "lr": lrs["z"], "name": "z"})
        return groups

    def harmonics(self, spline_values: torch.Tensor, cos_phi, sin_phi) -> torch.Tensor:
        """Combine per-channel spline values into an angular residual field [H, W]."""
        channels = self.active_channels()
        cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi
        sin_2phi = 2.0 * sin_phi * cos_phi
        weights = [None, cos_phi, sin_phi, cos_2phi, sin_2phi]
        total = None
        for channel in channels:
            term = spline_values[channel]
            if weights[channel] is not None:
                term = term * weights[channel]
            total = term if total is None else total + term
        return total

    def regularization(self, l2: float, curvature: float) -> torch.Tensor:
        """L2 towards zero plus a second-difference (curvature) penalty on the splines."""
        loss = self.omega.new_zeros(())
        tensors = []
        if self.active_channels():
            channels = self.active_channels()
            tensors += [self.theta_weights[channels], self.phi_weights[channels]]
        if "z" in self.components:
            tensors.append(self.z_weights)
        for tensor in tensors:
            loss = loss + l2 * tensor.square().sum()
            if tensor.shape[-1] >= 3 and curvature > 0.0:
                second = tensor[..., 2:] - 2.0 * tensor[..., 1:-1] + tensor[..., :-2]
                loss = loss + curvature * second.square().sum()
        return loss


class PoseResidual(nn.Module):
    """Per-view SE(3) residual, applied in the camera frame.

    Unlike the lens parameters this does *not* transfer to held-out views: test poses stay
    at COLMAP, so a drift of the scene frame shows up as a test-PSNR loss. Reported as a
    separate ablation rung for exactly that reason.
    """

    def __init__(self, lr_rotation: float, lr_translation: float, device: str = "cuda"):
        super().__init__()
        self.rotation = nn.ParameterDict()
        self.translation = nn.ParameterDict()
        self.lr_rotation = lr_rotation
        self.lr_translation = lr_translation
        self.device = device

    @staticmethod
    def _key(image_name: str) -> str:
        return image_name.replace(".", "_").replace("/", "_")

    def ensure(self, key: str) -> bool:
        "Create the per-view block if missing; returns True when it was just created."
        if key in self.rotation:
            return False
        self.rotation[key] = nn.Parameter(torch.zeros(3, device=self.device))
        self.translation[key] = nn.Parameter(torch.zeros(3, device=self.device))
        return True

    def block(self, image_name: str):
        key = self._key(image_name)
        return key, self.ensure(key)


class CameraModel(nn.Module):
    """Owns the residual lens parameters, their optimizer, and the ray synthesis."""

    def __init__(self, cfg, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.rung = cfg.camera_opt
        self.components = RUNGS[self.rung]
        # * Scene scale enters through the LEARNING RATE only, never through forward().
        # * z(theta) and the pose translation are therefore plain COLMAP units and the
        # * checkpoint is self-contained. The alternative -- scaling inside forward() --
        # * silently produced wrong renders, because train.py set scene_scale but
        # * render.py had no reason to know about it and left it at 1.0.
        self.scene_scale = 1.0
        self.frozen = True
        self.lenses = nn.ModuleDict()
        self.raxels = nn.ParameterDict()
        self.pose = (
            PoseResidual(cfg.pose_opt_lr_rotation, cfg.pose_opt_lr_translation, device=device)
            if cfg.pose_opt
            else None
        )
        self.optimizer = torch.optim.Adam([{"params": [], "lr": 0.0, "name": "placeholder"}])
        self._theta_max = math.pi / 2.0
        self._basis_cache = {}
        # * Pose-independent half of the ray synthesis, keyed by (base cache key, lens uid).
        # * See `camera_frame()`; invalidated by `invalidate_ray_cache()` on every event that
        # * can move a parameter.
        self.ray_cache_enabled = RAY_CACHE_DEFAULT
        self._ray_cache = {}

    def spline_basis(self, base: dict, num_ctrl: int) -> torch.Tensor:
        "Cached [N, K] B-spline basis for this camera/resolution and control-point count."
        cache_key = (base["cache_key"], num_ctrl)
        if cache_key not in self._basis_cache:
            self._basis_cache[cache_key] = build_bspline_basis(base["theta01"], num_ctrl)
        return self._basis_cache[cache_key]

    # ------------------------------------------------------------------ parameter blocks

    def invalidate_ray_cache(self):
        """Drop the pose-independent ray cache.

        Must be called on every event that can move a residual parameter, otherwise a render
        would silently reuse the previous parameters -- the same failure mode as the
        stale-bearing collision documented in IMPLEMENTATION.md, one level up.
        """
        self._ray_cache.clear()

    def lens(self, uid: int) -> LensResidual:
        key = str(uid)
        if key not in self.lenses:
            lens = LensResidual(
                self.components,
                self.cfg.camera_opt_knots,
                self.cfg.camera_opt_knots_z,
                device=self.device,
            )
            self.lenses[key] = lens
            lrs = {
                "tilt": self.cfg.camera_opt_lr_tilt,
                "angular": self.cfg.camera_opt_lr_angular,
                "z": self.cfg.camera_opt_lr_z * self.scene_scale,
            }
            for group in lens.parameter_groups(lrs):
                group["name"] = f"{group['name']}:{key}"
                self.optimizer.add_param_group(group)
        return self.lenses[key]

    def register_pose_block(self, key: str):
        "Create a per-view SE(3) block and its optimizer groups if they do not exist yet."
        if self.pose is not None and self.pose.ensure(key):
            self.optimizer.add_param_group(
                {
                    "params": [self.pose.rotation[key]],
                    "lr": self.pose.lr_rotation,
                    "name": f"poseR:{key}",
                }
            )
            self.optimizer.add_param_group(
                {
                    "params": [self.pose.translation[key]],
                    "lr": self.pose.lr_translation * self.scene_scale,
                    "name": f"poseT:{key}",
                }
            )

    def materialize_from_state_dict(self, state_dict):
        """Create the lazily-built blocks a checkpoint expects, before a strict load.

        Lens, raxel and pose blocks are all created on first render, so a freshly
        constructed model owns none of them and `load_state_dict` would reject every saved
        `camera_model.*` key. render.py / metrics.py hit this on the first reload.
        """
        self.invalidate_ray_cache()
        for key, value in state_dict.items():
            if not key.startswith("camera_model."):
                continue
            parts = key.split(".")
            if parts[1] == "lenses":
                self.lens(int(parts[2]))
            elif parts[1] == "raxels":
                name = parts[2]
                if name not in self.raxels:
                    self.raxels[name] = nn.Parameter(torch.zeros_like(value, device=self.device))
                    self.optimizer.add_param_group(
                        {
                            "params": [self.raxels[name]],
                            "lr": self.cfg.camera_opt_lr_raxel,
                            "name": f"raxel:{name}",
                        }
                    )
            elif parts[1] == "pose":
                self.register_pose_block(parts[3])

    def raxel_field(self, uid: int, height: int, width: int) -> torch.Tensor:
        "Dense generic ray field: the upper bound of the ablation ladder (~1e4 params)."
        key = f"{uid}_{height}_{width}"
        if key not in self.raxels:
            stride = self.cfg.camera_opt_raxel_stride
            grid_h = max(height // stride, 2)
            grid_w = max(width // stride, 2)
            # * 5 channels: 2 bearing offsets (meridional, sagittal) + 3 origin offsets.
            self.raxels[key] = nn.Parameter(torch.zeros(5, grid_h, grid_w, device=self.device))
            self.optimizer.add_param_group(
                {
                    "params": [self.raxels[key]],
                    "lr": self.cfg.camera_opt_lr_raxel,
                    "name": f"raxel:{key}",
                }
            )
        return self.raxels[key]

    # ------------------------------------------------------------------ ray synthesis

    def camera_frame(self, cam_info, base: dict, height: int, width: int) -> dict:
        """The POSE-INDEPENDENT half of the ray synthesis, in the OpenCV camera frame.

        Everything here is a function of (lens parameters, base bearings) only, and the base
        bearings are themselves a function of (camera model, intrinsics, render resolution) --
        `Raytracer.base_bearings()` caches them on exactly that key. Nothing below reads the
        pose, so the result is shared by every view taken with a given camera, which is what
        makes it cacheable across a whole eval pass.

        Returns a dict with:
          `bearings`      [H, W, 3] residual-rotated unit bearings, camera frame, 0 = invalid
          `z_profile`     [H, W] axial pupil offset `z(theta) - z(0)`, or None (central rungs)
          `raxel_offset`  [H, W, 3] the raxel rung's post-rotation offset, or None

        Those two tables are exactly what a CUDA-side `Camera::bearing_table` /
        `origin_offset_table` would hold, so this split is also the shape the eventual native
        port needs -- see IMPLEMENTATION.md, "the FPS tax is the Python path".
        """
        bearings = base["bearings"]  # [H, W, 3], unit in the OpenCV camera frame, 0 = invalid
        lens = self.lens(cam_info.uid)
        sampled = None

        delta_theta = None
        delta_phi = None
        if lens.active_channels():
            shape = base["theta01"].shape
            basis = self.spline_basis(base, lens.num_knots)
            spline_theta = bspline_eval_basis(lens.theta_weights, basis, shape)
            spline_phi = bspline_eval_basis(lens.phi_weights, basis, shape)
            delta_theta = lens.harmonics(spline_theta, base["cos_phi"], base["sin_phi"])
            delta_phi = lens.harmonics(spline_phi, base["cos_phi"], base["sin_phi"])

        if "raxel" in lens.components:
            field = self.raxel_field(cam_info.uid, height, width)
            sampled = torch.nn.functional.interpolate(
                field.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
            )[0]
            delta_theta = sampled[0] if delta_theta is None else delta_theta + sampled[0]
            delta_phi = sampled[1] if delta_phi is None else delta_phi + sampled[1]

        # * Meridional rotation: exactly the identity when delta_theta == 0.
        if delta_theta is not None:
            cos_d = torch.cos(delta_theta).unsqueeze(-1)
            sin_d = torch.sin(delta_theta).unsqueeze(-1)
            bearings = bearings * cos_d + torch.cross(base["meridian"], bearings, dim=-1) * sin_d

        # * Sagittal rotation about the optical axis, also exact at zero.
        if delta_phi is not None:
            cos_p = torch.cos(delta_phi)
            sin_p = torch.sin(delta_phi)
            bx, by, bz = bearings.unbind(-1)
            bearings = torch.stack([bx * cos_p - by * sin_p, bx * sin_p + by * cos_p, bz], dim=-1)

        if "tilt" in lens.components:
            bearings = skew_rotate(lens.omega, bearings)

        z_profile = None
        if "z" in lens.components:
            z_basis = self.spline_basis(base, lens.z_weights.shape[-1])
            profile = bspline_eval_basis(lens.z_weights, z_basis, base["theta01"].shape)[0]
            gauge = bspline_eval(lens.z_weights, base["theta01"].new_zeros(()))[0]
            z_profile = profile - gauge

        raxel_offset = None
        if sampled is not None:
            # * Kept verbatim from the original expression (multiply, then add), so the
            # * cached and un-cached paths agree bit for bit. NOTE this offset is added to
            # * the WORLD-space direction -- see IMPLEMENTATION.md limitation 8; it is a
            # * per-pixel world bias, not a camera model. Caching does not change that.
            raxel_offset = sampled[2:].permute(1, 2, 0) * base["valid"].unsqueeze(-1)

        return {"bearings": bearings, "z_profile": z_profile, "raxel_offset": raxel_offset}

    def apply_pose(self, cam_info, base: dict, frame: dict, height: int, width: int):
        """The PER-IMAGE half: rotate the camera-frame field into the world by this pose.

        This is the only part that may ever depend on the view. The optional per-view SE(3)
        residual lives here too, and so does the world-space optical axis the non-central
        origin offset is laid along -- `z(theta)` is a scalar in the camera frame, but the
        direction it displaces the ray origin along rotates with the camera.
        """
        bearings = frame["bearings"]
        rotation = base["rotation"]  # [3, 3] c2w, OpenCV convention
        origin = base["origin"]  # [3]
        if self.pose is not None:
            key = PoseResidual._key(cam_info.image_name)
            self.register_pose_block(key)
            delta_rotation = skew_rotate(
                self.pose.rotation[key], torch.eye(3, device=bearings.device)
            )
            rotation = rotation @ delta_rotation.transpose(0, 1)
            origin = origin + rotation @ self.pose.translation[key]

        direction = bearings @ rotation.transpose(0, 1)

        if frame["raxel_offset"] is not None:
            direction = direction + frame["raxel_offset"]

        # * Renormalize (a no-op for the pure-rotation rungs, required once raxel offsets
        # * enter) while keeping the zero sentinel outside the lens disk exactly zero.
        norm = direction.norm(dim=-1, keepdim=True)
        direction = direction / norm.clamp_min(1e-20)
        direction = direction * base["valid"].unsqueeze(-1)

        ray_origin = origin.view(1, 1, 3).expand(height, width, 3)
        if frame["z_profile"] is not None:
            axis = rotation[:, 2]  # * OpenCV +z: the optical axis, in world space
            ray_origin = ray_origin + frame["z_profile"].unsqueeze(-1) * axis

        return ray_origin.contiguous(), direction.contiguous()

    def forward(self, cam_info, base: dict, height: int, width: int):
        """Return world-space (origin, direction), both [H, W, 3] float32 on CUDA.

        Splits into the pose-independent `camera_frame()` and the per-image `apply_pose()`,
        and reuses the former across views of the same camera whenever that is provably safe.

        **When the cache is used.** Only under `no_grad`, i.e. render / eval / FPS. Under
        autograd the camera-frame tensors carry a graph that the optimizer step consumes and
        the parameters move every iteration, so training always recomputes -- caching there
        would silently freeze the residual at its first value.

        **What invalidates it.** `step()` (a parameter moved), `set_frozen()` (phase A/B
        boundary), `materialize_from_state_dict()` and `_load_from_state_dict()` (a checkpoint
        arrived). Anything that pokes a parameter tensor by hand must call
        `invalidate_ray_cache()`; there is no cheap way to detect that without a device sync.
        """
        cacheable = (
            self.ray_cache_enabled
            and not torch.is_grad_enabled()
            and base.get("cache_key") is not None
        )
        if not cacheable:
            frame = self.camera_frame(cam_info, base, height, width)
            return self.apply_pose(cam_info, base, frame, height, width)

        key = (base["cache_key"], cam_info.uid, height, width)
        frame = self._ray_cache.get(key)
        if frame is None:
            frame = self.camera_frame(cam_info, base, height, width)
            self._ray_cache[key] = frame
        return self.apply_pose(cam_info, base, frame, height, width)

    # ------------------------------------------------------------------ optimization

    def regularization(self) -> torch.Tensor:
        loss = torch.zeros((), device=self.device)
        for lens in self.lenses.values():
            loss = loss + lens.regularization(
                self.cfg.camera_opt_reg_l2, self.cfg.camera_opt_reg_curvature
            )
        return loss

    def set_frozen(self, frozen: bool):
        self.frozen = frozen
        self.invalidate_ray_cache()

    def _load_from_state_dict(self, *args, **kwargs):
        # * A checkpoint just replaced every residual parameter; anything cached from the
        # * previous values would render the wrong rays.
        self.invalidate_ray_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def set_lr_scale(self, scale: float):
        "Scale every group's LR by a schedule factor, preserving the per-group ratios."
        for group in self.optimizer.param_groups:
            name = group.get("name", "")
            if name.startswith("tilt"):
                base_lr = self.cfg.camera_opt_lr_tilt
            elif name.startswith("angular"):
                base_lr = self.cfg.camera_opt_lr_angular
            elif name.startswith("z:"):
                base_lr = self.cfg.camera_opt_lr_z * self.scene_scale
            elif name.startswith("raxel"):
                base_lr = self.cfg.camera_opt_lr_raxel
            elif name.startswith("poseR"):
                base_lr = self.cfg.pose_opt_lr_rotation
            elif name.startswith("poseT"):
                base_lr = self.cfg.pose_opt_lr_translation * self.scene_scale
            else:
                continue
            group["lr"] = base_lr * scale

    def step(self):
        # * A true freeze must skip the step entirely: Adam still moves its moments at lr=0,
        # * so `lr = 0` alone would leak phase-A gradients into the first phase-B updates.
        if self.frozen:
            self.optimizer.zero_grad(set_to_none=True)
            return
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # * The parameters moved: every cached camera-frame table is now stale.
        self.invalidate_ray_cache()

    def monotonicity_margin(self) -> float:
        """min over theta of d(theta + dtheta_rad)/d(theta_base); < 0 means the map folded.

        The residual is expected to be ~1e-3 rad, so a violation would need d(delta)/d(theta)
        below -1. Cheap to check, and it turns an assumption into a logged number.
        """
        margin = 1.0
        samples = torch.linspace(0.0, 1.0, 512, device=self.device)
        for lens in self.lenses.values():
            if not lens.active_channels():
                continue
            with torch.no_grad():
                values = bspline_eval(lens.theta_weights, samples)[0]
                slope = (values[1:] - values[:-1]) / (self._theta_max / 511.0)
                margin = min(margin, float((1.0 + slope).min()))
        return margin
