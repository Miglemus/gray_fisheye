from gray.config import Config

import safetensors.torch
import torch
import torch.nn as nn


class Vignetting(nn.Module):
    "Global radial vignetting model, shared by all views and optimized during training."

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # * Radial polynomial coefficients: optional linear term followed by even powers (r^2, r^4, ...)
        self.coefficients = nn.Parameter(
            torch.zeros(cfg.num_vignetting_coefficients, device="cuda")
        )
        # * Vignetting center in NDC, normalized by the longest image axis
        self.principal_point = nn.Parameter(torch.zeros(2, device="cuda"))
        self._coord_cache = {}

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [self.coefficients],
                    "lr": cfg.vignetting_coeff_lr,
                    "name": "coefficients",
                },
                {
                    "params": [self.principal_point],
                    "lr": cfg.vignetting_pp_lr,
                    "name": "principal_point",
                },
            ]
        )

    def forward(self, image: torch.Tensor):
        height, width = image.shape[-2:]
        cache_key = (width, height, image.device, image.dtype)
        if cache_key not in self._coord_cache:
            normalizer = float(max(width, height))
            x = torch.arange(width, device=image.device, dtype=image.dtype) + 0.5
            y = torch.arange(height, device=image.device, dtype=image.dtype) + 0.5
            x = (2.0 * x - float(width)) / normalizer
            y = (2.0 * y - float(height)) / normalizer
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            self._coord_cache[cache_key] = (grid_x, grid_y)

        grid_x, grid_y = self._coord_cache[cache_key]
        principal_point = self.principal_point.to(dtype=image.dtype)
        coefficients = self.coefficients.to(dtype=image.dtype)
        squared_radius = (grid_x - principal_point[0]).square() + (
            grid_y - principal_point[1]
        ).square()
        radial_polynomial = torch.zeros_like(squared_radius)
        coefficient_offset = 0
        if self.cfg.vignetting_include_linear_term:
            radius = torch.sqrt(squared_radius.clamp_min(1e-12))
            radial_polynomial = radial_polynomial + coefficients[0] * radius
            coefficient_offset = 1
        radius_term = squared_radius
        for term_idx in range(coefficient_offset, coefficients.shape[0]):
            radial_polynomial = radial_polynomial + coefficients[term_idx] * radius_term
            radius_term = radius_term * squared_radius

        if self.cfg.vignetting_activation == "exp":
            vignette = torch.exp(-radial_polynomial)
        else:
            # * Clamp at zero while keeping the unclamped gradient
            linear_vignette = 1.0 - radial_polynomial
            vignette = linear_vignette + (linear_vignette.clamp_min(0.0) - linear_vignette).detach()

        if self.cfg.vignetting_srgb_comp:
            # * Apply the vignette in (approximate) linear space
            pseudo_srgb_floor = 1e-8
            pseudo_srgb_floor_encoded = pseudo_srgb_floor ** (1.0 / 2.4)
            image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).pow(2.4)
            image = image * vignette.unsqueeze(0)
            image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(pseudo_srgb_floor)
            image = image.pow(1.0 / 2.4)
            return (image - pseudo_srgb_floor_encoded) / (1.0 - pseudo_srgb_floor_encoded)
        return image * vignette.unsqueeze(0)

    def set_lrs(self, coefficient_lr: float, principal_point_lr: float):
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "coefficients":
                param_group["lr"] = coefficient_lr
            elif param_group.get("name") == "principal_point":
                param_group["lr"] = principal_point_lr

    def load_parameters(self, source: str | dict[str, torch.Tensor]):
        state_dict = safetensors.torch.load_file(source) if isinstance(source, str) else source
        with torch.no_grad():
            if "vignetting.coefficients" in state_dict:
                loaded = (
                    state_dict["vignetting.coefficients"]
                    .reshape(-1)
                    .to(device=self.coefficients.device, dtype=self.coefficients.dtype)
                )
                count = min(self.coefficients.numel(), loaded.numel())
                self.coefficients.zero_()
                self.coefficients[:count].copy_(loaded[:count])
            if "vignetting.principal_point" in state_dict:
                self.principal_point.copy_(
                    state_dict["vignetting.principal_point"].to(
                        device=self.principal_point.device, dtype=self.principal_point.dtype
                    )
                )

    def step(self):
        self.optimizer.step()
        with torch.no_grad():
            self.coefficients.clamp_(min=0.0)
            self.principal_point.clamp_(min=-1.0, max=1.0)
        self.optimizer.zero_grad()
