"""Validation of dL/d(ray_origin) and dL/d(ray_direction) produced by cuda/backward_pass.cu.

This is the test the whole learnable-camera-model effort rests on: if these gradients are
wrong, every downstream number is quietly wrong too, and nothing else would catch it
(training would still "work", it would just optimize the wrong thing).

Two independent checks:
  * an analytic torch reference of the single-gaussian forward, differentiated by autograd;
  * a black-box central finite difference of the image loss w.r.t. individual ray
    components on a multi-gaussian scene, which also covers the alpha-compositing chain.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gray.prelude import RaytracerConfig, Raytracer
from tests.utils import build_rotation


def make_rays(num_pixels, device="cuda", seed=0):
    "A mildly non-central, mildly irregular ray bundle -- nothing a central model could emit."
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ys, xs = torch.meshgrid(
        torch.linspace(-0.4, 0.4, num_pixels), torch.linspace(-0.4, 0.4, num_pixels), indexing="ij"
    )
    direction = torch.stack([xs, ys, -torch.ones_like(xs)], dim=-1)
    direction = F.normalize(direction, dim=-1)
    # * Per-pixel origin offsets: exactly what a non-central model produces.
    offset = 0.05 * torch.rand((num_pixels, num_pixels, 3), generator=generator)
    origin = torch.tensor([0.0, 0.0, 1.0]) + offset
    return origin.to(device).float(), direction.to(device).float()


def build_raytracer(num_gaussians, num_pixels, alpha_threshold, seed=0):
    cfg = RaytracerConfig(sh=False, alpha_threshold=alpha_threshold)
    raytracer = Raytracer(cfg, num_gaussians, num_pixels, num_pixels)

    config = raytracer.cuda_module.get_config()
    config.zero_grads.fill_(False)
    config.jitter_primary_rays.fill_(False)
    config.rays_from_python.fill_(True)

    background = torch.tensor([0.1, 0.1, 0.1], device="cuda")
    config.background_channels.copy_(background)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussians = raytracer.cuda_module.get_gaussians()
    means = (torch.rand((num_gaussians, 3), generator=generator) - 0.5) * 0.6
    scales = (0.10 + 0.15 * torch.rand((num_gaussians, 3), generator=generator)).log()
    quats = torch.randn((num_gaussians, 4), generator=generator)
    colors = torch.rand((num_gaussians, 3), generator=generator)
    opacities = (0.4 + 0.4 * torch.rand((num_gaussians, 1), generator=generator)).logit()
    gaussians.mean.copy_(means.cuda())
    gaussians.scale.copy_(scales.cuda())
    gaussians.rotation.copy_(quats.cuda())
    gaussians.channels.copy_(colors.cuda())
    gaussians.opacity.copy_(opacities.cuda())
    torch.cuda.synchronize()
    raytracer.cuda_module.rebuild_bvh()

    camera = raytracer.cuda_module.get_camera()
    camera.znear.fill_(0.0)
    camera.zfar.fill_(99999.9)
    camera.set_pinhole()
    camera.set_pose(torch.zeros(3, device="cuda"), torch.eye(3, device="cuda"))
    return raytracer, background


def render_with_rays(raytracer, origin, direction):
    "Drive the CUDA module directly with Python-supplied rays; returns [3, H, W]."
    height, width = origin.shape[0], origin.shape[1]
    framebuffer = raytracer.cuda_module.get_framebuffer()
    with torch.no_grad():
        framebuffer.ray_origin[:height, :width].copy_(origin)
        framebuffer.ray_direction[:height, :width].copy_(direction)
    raytracer.cuda_module.forward_pass()
    return framebuffer.output_channels[:height, :width].clone().moveaxis(-1, 0)


def backward_with_rays(raytracer, grad_output, height, width):
    "Run the CUDA backward for a given dL/d(output) and read the ray gradients back."
    framebuffer = raytracer.cuda_module.get_framebuffer()
    with torch.no_grad():
        framebuffer.grad_ray_origin.zero_()
        framebuffer.grad_ray_direction.zero_()
        framebuffer.grad_output_channels[:height, :width].copy_(grad_output.moveaxis(0, -1))
    raytracer.cuda_module.backward_pass()
    return (
        framebuffer.grad_ray_origin[:height, :width].clone(),
        framebuffer.grad_ray_direction[:height, :width].clone(),
    )


def test_ray_gradients_match_analytic_reference():
    """Single gaussian: compare against a closed-form torch forward, differentiated by autograd.

    The reference is the one already validated in tests/test_single_gaussian.py, rewritten
    as a function of (ray_origin, ray_direction) so autograd produces the quantity under test.
    """
    num_pixels = 32
    alpha_threshold = 0.01
    raytracer, background = build_raytracer(1, num_pixels, alpha_threshold)

    config = raytracer.cuda_module.get_config()
    exp_power = config.exp_power.item()
    max_alpha = raytracer.cuda_module.get_max_alpha()

    mean = torch.tensor([-0.05, -0.05, 0.0], device="cuda")
    log_scale = torch.tensor([0.2, 0.3, 0.25], device="cuda").log()
    quaternion = torch.tensor([-0.5, 0.0, 0.0, 1.0], device="cuda")
    logit_opacity = torch.tensor([0.7], device="cuda").logit()
    color = torch.tensor([1.0, 0.3, 0.1], device="cuda")

    gaussians = raytracer.cuda_module.get_gaussians()
    gaussians.mean.copy_(mean)
    gaussians.scale.copy_(log_scale)
    gaussians.rotation.copy_(quaternion)
    gaussians.opacity.copy_(logit_opacity)
    gaussians.channels.copy_(color)
    torch.cuda.synchronize()
    raytracer.cuda_module.rebuild_bvh()

    origin, direction = make_rays(num_pixels)
    render = render_with_rays(raytracer, origin, direction)

    # * A non-symmetric upstream gradient so no component can cancel by accident.
    torch.manual_seed(3)
    grad_output = torch.randn_like(render)
    cuda_grad_origin, cuda_grad_direction = backward_with_rays(
        raytracer, grad_output, num_pixels, num_pixels
    )

    # * Analytic reference, differentiable in the rays.
    origin_ref = origin.clone().requires_grad_()
    direction_ref = direction.clone().requires_grad_()
    # * Built locally rather than via tests.utils.build_scaling_rotation, whose retain_grad()
    # * requires the gaussian parameters to be leaves; here only the rays carry gradients.
    rotation = build_rotation(F.normalize(quaternion[None], dim=-1))
    scaling = torch.diag(log_scale.exp())
    world_to_local = torch.linalg.inv(rotation[0] @ scaling)
    local_origin = (origin_ref - mean) @ world_to_local.T
    local_direction_unnormalized = direction_ref @ world_to_local.T
    local_direction = local_direction_unnormalized / local_direction_unnormalized.norm(
        dim=-1, keepdim=True
    )
    distance = -(local_origin * local_direction).sum(-1, keepdim=True)
    local_hit = local_origin + distance * local_direction
    squared = (local_hit * local_hit).sum(-1)
    gaussian_value = torch.exp(-(squared**exp_power) / (2.0 * exp_power))
    alpha = logit_opacity[0].sigmoid() * gaussian_value * max_alpha
    alpha = torch.where(alpha < alpha_threshold, torch.zeros_like(alpha), alpha)
    reference = color * alpha[..., None] + background * (1.0 - alpha[..., None])
    reference = reference.moveaxis(-1, 0)

    # * float32 accumulation plus the hard alpha-threshold cut put the floor around 3e-5;
    # * anything structurally wrong would be orders of magnitude above this.
    assert F.l1_loss(reference, render).item() < 1e-4, "forward disagrees; gradients are moot"

    reference.backward(grad_output)

    scale = cuda_grad_origin.abs().amax().item()
    assert scale > 1e-3, "degenerate test: the ray gradients are essentially zero"
    assert (origin_ref.grad - cuda_grad_origin).abs().amax().item() < 2e-3 * scale + 1e-6
    assert (direction_ref.grad - cuda_grad_direction).abs().amax().item() < 2e-3 * scale + 1e-6


@pytest.mark.parametrize("component", ["origin", "direction"])
def test_ray_gradients_match_finite_differences(component):
    """Multi-gaussian black-box check, which also exercises the alpha-compositing chain.

    Central differences of the scalar image loss w.r.t. one ray component at a time.
    """
    num_pixels = 24
    # * 8 gaussians keeps the per-pixel hit count at 2-8, well under forward_pass.cu's
    # * BUFFER_SIZE = 32. Above that gray falls back to `remaining_channels_estimate`, an
    # * approximation of the truncated tail whose ray derivative is deliberately not
    # * modelled (exactly as for grad_mean) -- and the truncation boundary itself is a
    # * discontinuity that a finite difference reads as an enormous derivative. Measured:
    # * 48 gaussians here gives max 45 hits per pixel and the check becomes meaningless.
    raytracer, _ = build_raytracer(8, num_pixels, alpha_threshold=0.005, seed=7)
    origin, direction = make_rays(num_pixels, seed=11)

    target = torch.rand((3, num_pixels, num_pixels), device="cuda")

    def pixel_loss_of(origin_in, direction_in, row, col):
        """Loss restricted to one pixel.

        Each pixel's output depends only on its own ray, so the per-pixel loss has exactly
        the same ray derivative as the full-image loss -- but differencing the full sum over
        1728 values in float32 would drown a ~4e-4 signal in ~1e-5 of accumulation noise.
        """
        render = render_with_rays(raytracer, origin_in, direction_in)
        residual = (render[:, row, col] - target[:, row, col]).double()
        return (residual**2).sum().item() * 0.5

    render = render_with_rays(raytracer, origin, direction)
    loss = ((render - target) ** 2).sum() * 0.5
    grad_output = render - target  # * dL/d(output) for the above loss
    cuda_grad_origin, cuda_grad_direction = backward_with_rays(
        raytracer, grad_output, num_pixels, num_pixels
    )
    analytic = cuda_grad_origin if component == "origin" else cuda_grad_direction

    # * Probe the pixels with the largest gradient: they carry the signal, and staying away
    # * from near-zero entries keeps the relative comparison meaningful.
    magnitude = analytic.abs().sum(-1)
    flat = torch.argsort(magnitude.flatten(), descending=True)[:12]
    # * Epsilon has to be *large*. Measured error vs step on this scene (grad scale 2.58):
    # *     1e-2 -> 2.4e-3   3e-3 -> 7.7e-3   1e-3 -> 2.1e-2   3e-4 -> 4.5e-2   1e-4 -> 2.0e-1
    # * The error grows as 1/eps, i.e. it is float32 round-off in the render, not truncation
    # * error and not a gradient bug (a wrong gradient would show an eps-independent floor).
    epsilon = 1e-2
    errors = []
    for flat_index in flat.tolist():
        row, col = divmod(flat_index, num_pixels)
        for axis in range(3):
            plus_origin, plus_direction = origin.clone(), direction.clone()
            minus_origin, minus_direction = origin.clone(), direction.clone()
            if component == "origin":
                plus_origin[row, col, axis] += epsilon
                minus_origin[row, col, axis] -= epsilon
            else:
                plus_direction[row, col, axis] += epsilon
                minus_direction[row, col, axis] -= epsilon
            numeric = (
                pixel_loss_of(plus_origin, plus_direction, row, col)
                - pixel_loss_of(minus_origin, minus_direction, row, col)
            ) / (2.0 * epsilon)
            errors.append(abs(numeric - analytic[row, col, axis].item()))

    scale = analytic.abs().amax().item()
    assert scale > 1e-3, "degenerate test: the ray gradients are essentially zero"
    # * Median and an upper quantile rather than the max: gray's renderer is genuinely
    # * discontinuous where a gaussian enters or leaves the alpha-threshold clip
    # * (`sq_dist > 1` in __intersection__is), and a probe that straddles such a boundary
    # * makes the central difference read an arbitrarily large slope. A systematically wrong
    # * gradient would move the median, which is what this actually guards.
    assert float(np.median(errors)) < 3e-3 * scale, f"median FD error {np.median(errors):.3e}"
    assert float(np.quantile(errors, 0.8)) < 1.5e-2 * scale, (
        f"p80 FD error {np.quantile(errors, 0.8):.3e}"
    )
    assert loss.item() > 0.0
