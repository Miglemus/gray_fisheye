"""Tests for the learnable residual camera model (gray/camera_model.py).

The load-bearing one is `test_passthrough_reproduces_native_path`: the whole ablation ladder
compares rungs against each other, so it is only meaningful if the zero-residual rung is
indistinguishable from gray's native ray generation.
"""

import dataclasses

import numpy as np
import pytest
import torch

from gray.camera import CameraInfo
from gray.camera_model import bspline_eval, bspline_subdivide
from gray.prelude import RaytracerConfig, Raytracer

# * myscenes `tunnel` RAD_TAN_THIN_PRISM_FISHEYE calibration at 5472x3648, downscaled by 57
# * to 96x64. Distortion coefficients act on normalized coordinates and do not rescale.
FULL_RESOLUTION_INTRINSICS = np.array(
    [
        1240.524225527915, 1240.6390654335623, 2736.0, 1824.0,
        -0.034022403953697364, -0.00088328961651482, -0.00058149395765803,
        -0.00059997214095231, 0.00018427433009331, -2.9170232358608713e-05,
        0.00084438233242992, -0.0022367414456503, -0.0027090969528328,
        -0.00017409987835018, 0.0074191683774425, 0.001317083052115925,
    ],
    dtype=np.float64,
)
WIDTH, HEIGHT, DOWNSCALE = 96, 64, 57


def fisheye_camera():
    intrinsics = FULL_RESOLUTION_INTRINSICS.copy()
    intrinsics[:4] /= DOWNSCALE
    axis = np.array([0.3, -0.7, 0.5])
    axis /= np.linalg.norm(axis)
    angle = 0.6
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    return CameraInfo(
        uid=0,
        R=rotation,
        T=np.zeros(3),
        origin=np.array([0.2, -0.1, 0.4]),
        fov_y=1.0,
        fov_x=1.0,
        image_path="",
        image_name="probe.png",
        image_width=WIDTH,
        image_height=HEIGHT,
        is_test=False,
        model="rad_tan_thin_prism_fisheye",
        intrinsics=intrinsics,
    )


def build_scene(rung):
    cfg = RaytracerConfig(sh=False, camera_opt=rung)
    raytracer = Raytracer(cfg, 6, WIDTH, HEIGHT)
    generator = torch.Generator(device="cpu").manual_seed(5)
    gaussians = raytracer.cuda_module.get_gaussians()
    gaussians.mean.copy_(((torch.rand((6, 3), generator=generator) - 0.5) * 2.0).cuda())
    gaussians.scale.copy_((0.3 + 0.4 * torch.rand((6, 3), generator=generator)).log().cuda())
    gaussians.rotation.copy_(torch.randn((6, 4), generator=generator).cuda())
    gaussians.channels.copy_(torch.rand((6, 3), generator=generator).cuda())
    gaussians.opacity.copy_((0.6 * torch.ones((6, 1))).logit().cuda())
    torch.cuda.synchronize()
    raytracer.cuda_module.rebuild_bvh()
    config = raytracer.cuda_module.get_config()
    config.jitter_primary_rays.fill_(False)
    config.needs_ray_output.fill_(True)
    return raytracer


def test_passthrough_reproduces_native_path():
    """Rays from Python with a zero residual must reproduce gray's own raygen.

    Only one Raytracer can live per process (hence pytest --forked), so the two paths are
    exercised on the same instance by detaching and reattaching the camera model.
    """
    raytracer = build_scene("passthrough")
    camera = fisheye_camera()
    framebuffer = raytracer.cuda_module.get_framebuffer()

    model = raytracer.camera_model
    raytracer.camera_model = None
    with torch.no_grad():
        native_render = raytracer(camera).clone()
    native_directions = framebuffer.ray_direction[:HEIGHT, :WIDTH].clone()

    raytracer.camera_model = model
    with torch.no_grad():
        modelled_render = raytracer(camera).clone()
    modelled_directions = framebuffer.ray_direction[:HEIGHT, :WIDTH].clone()

    valid = native_directions.norm(dim=-1) > 0.5
    assert valid.sum() > 0.3 * valid.numel(), "degenerate test: almost no valid pixels"

    # * The invalid sentinel outside the lens disk must survive the round trip exactly.
    assert torch.equal(modelled_directions.norm(dim=-1) > 0.5, valid)

    # * Chord length, not arccos(dot). For unit vectors the chord equals the angle to second
    # * order, and it stays well conditioned: arccos near 1 amplifies a float32 dot-product
    # * error of 1e-7 into an apparent 5e-4 rad, which is pure measurement artefact.
    chord = (native_directions - modelled_directions).norm(dim=-1)[valid].max().item()
    assert chord < 1e-6, f"bearing mismatch {chord:.3e} rad"
    assert (native_render - modelled_render).abs().max().item() < 1e-5


def test_noncentral_origin_is_gauged_at_zero_and_moves_off_axis():
    "z(0) = 0 exactly, and z(theta) is otherwise free -- the anti-degeneracy gauge."
    raytracer = build_scene("noncentral")
    camera = fisheye_camera()
    raytracer.camera_model.scene_scale = 1.0

    base = dict(raytracer.base_bearings(camera))
    base["rotation"] = Raytracer._rotation_c2w_cuda(camera)
    base["origin"] = camera.origin_cuda()

    lens = raytracer.camera_model.lens(camera.uid)
    with torch.no_grad():
        lens.z_weights.normal_(0.0, 0.1, generator=None)
        # * Force one pixel to sit at exactly theta = 0 so the gauge can be checked exactly.
        base["theta01"] = base["theta01"].clone()
        base["theta01"][0, 0] = 0.0

    ray_origin, _ = raytracer.camera_model(camera, base, HEIGHT, WIDTH)
    centre = camera.origin_cuda()

    on_axis_offset = (ray_origin[0, 0] - centre).abs().max().item()
    assert on_axis_offset < 1e-6, f"gauge violated: z(0) = {on_axis_offset:.3e}"

    off_axis = (ray_origin - centre).norm(dim=-1)
    assert off_axis.max().item() > 1e-3, "z(theta) is degenerate, the test proves nothing"


def test_ray_synthesis_does_not_depend_on_scene_scale():
    """forward() must be free of scene_scale: it is a training-time LR factor, not state.

    Regression test for a bug that produced *correct training metrics and wrong renders*.
    When the scale multiplied z(theta) inside forward(), train.py set it from the scene
    radius but render.py had no reason to and left it at 1.0, so every re-rendered image for
    the non-central rung was wrong by ~14x on tunnel. The in-training PSNR said +0.17 dB
    while the re-rendered PNGs said -0.19; only the disagreement between the two exposed it.
    """
    raytracer = build_scene("noncentral")
    camera = fisheye_camera()
    base = dict(raytracer.base_bearings(camera))
    base["rotation"] = Raytracer._rotation_c2w_cuda(camera)
    base["origin"] = camera.origin_cuda()
    lens = raytracer.camera_model.lens(camera.uid)
    with torch.no_grad():
        lens.z_weights.normal_(0.0, 0.1)

    raytracer.camera_model.scene_scale = 1.0
    origin_a, direction_a = raytracer.camera_model(camera, base, HEIGHT, WIDTH)
    raytracer.camera_model.scene_scale = 37.0
    origin_b, direction_b = raytracer.camera_model(camera, base, HEIGHT, WIDTH)

    assert torch.equal(origin_a, origin_b), "forward() still depends on scene_scale"
    assert torch.equal(direction_a, direction_b)
    offset = (origin_a - camera.origin_cuda()).norm(dim=-1).max().item()
    assert offset > 1e-3, "z(theta) is degenerate here, so the test proves nothing"


def test_bspline_subdivision_preserves_the_function():
    """Lane-Riesenfeld refinement must represent exactly the same curve.

    Checked on the interior only: the outermost half-segment of a uniform (unclamped)
    B-spline depends on control points that refinement does not reconstruct.
    """
    torch.manual_seed(0)
    weights = torch.randn(3, 9, device="cuda")
    refined = bspline_subdivide(weights)
    assert refined.shape[-1] == 2 * weights.shape[-1] - 3

    samples = torch.linspace(0.15, 0.85, 257, device="cuda")
    coarse = bspline_eval(weights, samples)
    fine = bspline_eval(refined, samples)
    assert (coarse - fine).abs().max().item() < 1e-5

    # * And a zero-initialized spline stays exactly zero, which is what makes the residual
    # * parameterization reproduce COLMAP at init.
    zeros = torch.zeros(3, 9, device="cuda")
    assert bspline_eval(zeros, samples).abs().max().item() == 0.0


@pytest.mark.parametrize(
    "rung,expected",
    [
        ("passthrough", None),
        ("tilt", "omega"),
        ("radial", "theta_weights"),
        ("ana", "phi_weights"),
        ("noncentral", "z_weights"),
        ("central_matched", "theta_weights"),
        ("raxel", "raxel"),
    ],
)
def test_every_rung_completes_a_training_step(rung, expected):
    """One full step per rung, asserting the gradient actually reaches the parameters.

    This closes the loop end to end: torch rays -> framebuffer -> OptiX forward -> OptiX
    backward -> grad_ray_* -> autograd -> residual parameters. It is also a regression test
    for a real bug: the third backward stage used to hand both the ray origin and the ray
    direction to autograd.backward unconditionally, which raises "does not require grad" for
    every rung that leaves the origin a plain broadcast of the camera centre (all of them
    except `noncentral` and `pose_opt`).
    """
    raytracer = build_scene(rung)
    camera = fisheye_camera()
    raytracer.camera_model.scene_scale = 1.0
    raytracer.camera_model.set_frozen(False)

    target = torch.rand((3, HEIGHT, WIDTH), device="cuda")
    render = raytracer(camera)
    loss = torch.nn.functional.l1_loss(render, target)
    raytracer.backward(loss)

    if expected == "raxel":
        gradients = [p.grad for p in raytracer.camera_model.raxels.values()]
    elif expected is not None:
        gradients = [getattr(raytracer.camera_model.lens(camera.uid), expected).grad]
    else:
        gradients = []

    for gradient in gradients:
        assert gradient is not None, f"{rung}: no gradient reached {expected}"
        assert torch.isfinite(gradient).all(), f"{rung}: non-finite gradient on {expected}"
        assert gradient.abs().sum().item() > 0.0, f"{rung}: gradient on {expected} is all zero"

    raytracer.step()  # * must not raise, frozen or not


def test_monotonicity_margin_detects_a_folded_map():
    "The residual must not fold theta(theta_base); the margin turns that into a number."
    raytracer = build_scene("radial")
    camera = fisheye_camera()
    lens = raytracer.camera_model.lens(camera.uid)

    assert raytracer.camera_model.monotonicity_margin() == 1.0  # * zero residual

    with torch.no_grad():
        # * A steeply decreasing radial residual: d(delta)/d(theta) well below -1.
        ramp = torch.linspace(0.0, -5.0, lens.theta_weights.shape[-1], device="cuda")
        lens.theta_weights[0].copy_(ramp)
    assert raytracer.camera_model.monotonicity_margin() < 0.0


def test_base_bearings_do_not_collide_across_camera_models():
    """Two camera models, same uid and same render resolution -> different bearings.

    `render.py --eval-models pinhole rad_tan_thin_prism_fisheye` probes both models for the
    same camera in one process. The cache used to key on (uid, height, width) alone, so the
    second model silently received the first one's bearings. That is invisible whenever the
    two happen to render at different sizes (myscenes: 400x266 vs 1368x912) and catastrophic
    when they do not (`workshop_immervision`, both 1440x1080): the fisheye pass re-rendered
    at 13.31 dB against the 25.37 the same checkpoint reached live.

    Order matters -- probing the fisheye first hides the bug -- so assert BOTH directions.
    """
    raytracer = build_scene("passthrough")
    fisheye = fisheye_camera()
    pinhole = dataclasses.replace(fisheye, model="pinhole", intrinsics=None)

    # * Same uid and same resolution: the two differ only by the camera model.
    assert pinhole.uid == fisheye.uid
    assert (pinhole.image_width, pinhole.image_height) == (fisheye.image_width, fisheye.image_height)

    pinhole_first = raytracer.base_bearings(pinhole)["bearings"].clone()
    fisheye_second = raytracer.base_bearings(fisheye)["bearings"].clone()
    assert not torch.allclose(pinhole_first, fisheye_second)

    # * Same again on a fresh raytracer, fisheye first: the cache must not leak either way.
    raytracer = build_scene("passthrough")
    fisheye_first = raytracer.base_bearings(fisheye)["bearings"].clone()
    pinhole_second = raytracer.base_bearings(pinhole)["bearings"].clone()
    assert not torch.allclose(fisheye_first, pinhole_second)

    # * And the model each camera gets is order-independent.
    assert torch.equal(fisheye_first, fisheye_second)
    assert torch.equal(pinhole_first, pinhole_second)

    # * The camera model caches theta-derived tables on `cache_key`, so it must separate the
    # * two as well -- otherwise the collision simply moves one level up.
    assert (raytracer.base_bearings(pinhole)["cache_key"]
            != raytracer.base_bearings(fisheye)["cache_key"])
