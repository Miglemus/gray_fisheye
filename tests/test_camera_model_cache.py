"""The pose-independent ray cache: bit-exactness, and the correctness line it must not cross.

Why this file exists
--------------------
`--camera_opt noncentral` costs roughly 2x render FPS against gray's native ray generation,
and the controls say that cost is **not** the non-centrality: on `tunnel`, `noncentral`
(112.08 FPS), `central_matched` (115.98) and `ana` (117.44) sit within 5 % of each other and
all ~1.4x below native. The tax is the Python ray-synthesis path. So `CameraModel.forward()`
now caches the half of that synthesis which does not depend on the pose, exactly as the
`masked-efficiency` worktree caches per-camera bearings for the native path
(`gray/raytracer.py::_bind_ray_tables` there, bit-exact, 1.03-1.05x).

The one way this can be wrong
-----------------------------
The non-central model emits `(origin, direction)` in the CAMERA frame; the rotation into the
world depends on the pose. Caching one pixel too far -- anything downstream of `rotation` --
would freeze the first view's pose into every subsequent render, and it would look like a
plausible image, not like a crash. Hence `test_cache_does_not_freeze_the_pose`, which does
not merely check "the two poses differ" but reconstructs pose B's field from pose A's by the
relative rotation, and `test_camera_frame_ignores_the_pose`.

Running it
----------
Everything named `*_cpu_*` runs on CPU with no CUDA context and no render:

    python -m pytest tests/test_camera_model_cache.py -q -k "not gpu"

The `gpu_` tests exercise the real `Raytracer.base_bearings()` probe and therefore *render*.
They are skipped unless `GRAY_RUN_GPU_TESTS=1` is set, deliberately: this work was done under
an instruction to queue no GPU task at all, and a test suite that silently takes a card is a
trap for the next person too.

    GRAY_RUN_GPU_TESTS=1 python -m pytest tests/test_camera_model_cache.py -q
"""

import math
import os

import pytest
import torch

from gray.camera_model import (
    PoseResidual,
    CameraModel,
    bspline_eval,
    bspline_eval_basis,
    skew_rotate,
)
from gray.config import RaytracerConfig

RUN_GPU = os.environ.get("GRAY_RUN_GPU_TESTS", "") == "1"
gpu_only = pytest.mark.skipif(
    not (RUN_GPU and torch.cuda.is_available()),
    reason="renders on a GPU; set GRAY_RUN_GPU_TESTS=1 to enable",
)

# * Rungs worth covering: one per distinct code path through the synthesis.
#   passthrough = no residual at all, radial = angular only, ana = + harmonics,
#   z_only = the origin moves but nothing else, noncentral = everything.
RUNGS = ["passthrough", "tilt", "radial", "ana", "z_only", "noncentral_no_ana", "noncentral"]


# --------------------------------------------------------------------------- fixtures


def synthetic_base(height: int, width: int, tag: str, device: str = "cpu") -> dict:
    """A stand-in for `Raytracer.base_bearings()`, built without a GPU or a raygen.

    It reproduces the *contract* of that dict, which is all the ray synthesis reads: unit
    bearings in the OpenCV camera frame, zero outside the lens disk; a meridional axis
    orthogonal to each bearing; `theta01` as the (detached) spline coordinate; `cos_phi` /
    `sin_phi` zeroed where invalid; a float `valid`; and a `cache_key` identifying the
    (camera, resolution). The actual angles come from an equidistant fisheye, which is not
    the calibration of any real lens here and does not need to be -- the synthesis never
    inverts the projection, it only rotates what it is handed.
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    u = (xs + 0.5) / width * 2.0 - 1.0
    v = (ys + 0.5) / height * 2.0 - 1.0
    radius = torch.sqrt(u * u + v * v) / 0.9  # * > 1 near the corners: an invalid annulus
    valid = radius <= 1.0
    theta = (radius * (math.pi / 2.0)).clamp(0.0, math.pi / 2.0)
    phi = torch.atan2(v, u)
    zeros = torch.zeros_like(theta)
    cos_phi = torch.where(valid, torch.cos(phi), zeros)
    sin_phi = torch.where(valid, torch.sin(phi), zeros)

    bearings = torch.stack(
        [torch.sin(theta) * cos_phi, torch.sin(theta) * sin_phi, torch.cos(theta)], dim=-1
    )
    bearings = bearings * valid.unsqueeze(-1).to(bearings.dtype)

    return {
        "bearings": bearings,
        "meridian": torch.stack([-sin_phi, cos_phi, zeros], dim=-1),
        "theta01": (theta / (math.pi / 2.0)).clamp(0.0, 1.0),
        "cos_phi": cos_phi,
        "sin_phi": sin_phi,
        "valid": valid.to(bearings.dtype),
        "cache_key": ("synthetic", tag, height, width),
    }


def rotation_matrix(axis, angle, device="cpu") -> torch.Tensor:
    axis = torch.tensor(axis, dtype=torch.float32, device=device)
    axis = axis / axis.norm()
    cross = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        device=device,
    )
    eye = torch.eye(3, device=device)
    return eye + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


# * Three genuinely different poses: distinct rotations AND distinct centres.
POSES = [
    ((0.3, -0.7, 0.5), 0.6, (0.2, -0.1, 0.4)),
    ((-0.9, 0.2, 0.35), 1.9, (-3.0, 2.5, 0.75)),
    ((0.0, 1.0, 0.0), 2.7, (11.0, -0.25, -6.5)),
]


def posed(base: dict, index: int, device="cpu") -> dict:
    axis, angle, origin = POSES[index]
    posed_base = dict(base)
    posed_base["rotation"] = rotation_matrix(axis, angle, device)
    posed_base["origin"] = torch.tensor(origin, dtype=torch.float32, device=device)
    return posed_base


class _Cam:
    "The three attributes the ray synthesis reads off a CameraInfo."

    def __init__(self, uid=0, image_name="probe.png"):
        self.uid = uid
        self.image_name = image_name


def randomized_model(rung: str, device: str = "cpu", seed: int = 7) -> CameraModel:
    """A model with a *non-zero* residual, which is the only setting where the test bites.

    Magnitudes are the measured ones (IMPLEMENTATION.md): `theta_weights` ~4e-3 rad,
    `z_weights` ~6e-3 scene units, `omega` ~1e-3 rad. A zero residual would make every rung
    reproduce the base field and every assertion below pass vacuously.
    """
    cfg = RaytracerConfig(camera_opt=rung)
    model = CameraModel(cfg, device=device)
    lens = model.lens(0)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def fill(parameter, scale):
        values = torch.randn(parameter.shape, generator=generator) * scale
        with torch.no_grad():
            parameter.copy_(values.to(device))

    fill(lens.omega, 1e-3)
    fill(lens.theta_weights, 4e-3)
    fill(lens.phi_weights, 3e-3)
    fill(lens.z_weights, 6e-3)
    model.invalidate_ray_cache()
    return model


# --------------------------------------------------------------- the reference synthesis


def reference_forward(model: CameraModel, cam_info, base: dict, height: int, width: int):
    """The monolithic `forward()` exactly as it stood before the cache split.

    Copied verbatim on purpose. The split is a refactor of code that carries a measured
    +0.315 dB result, and this branch has twice shipped a silent multi-dB regression
    (`scene_scale` inside forward, then the stale-bearing collision). A refactor that is
    only checked against itself proves nothing about the refactor.
    """
    bearings = base["bearings"]
    lens = model.lens(cam_info.uid)

    delta_theta = None
    delta_phi = None
    if lens.active_channels():
        shape = base["theta01"].shape
        basis = model.spline_basis(base, lens.num_knots)
        spline_theta = bspline_eval_basis(lens.theta_weights, basis, shape)
        spline_phi = bspline_eval_basis(lens.phi_weights, basis, shape)
        delta_theta = lens.harmonics(spline_theta, base["cos_phi"], base["sin_phi"])
        delta_phi = lens.harmonics(spline_phi, base["cos_phi"], base["sin_phi"])

    if "raxel" in lens.components:
        field = model.raxel_field(cam_info.uid, height, width)
        sampled = torch.nn.functional.interpolate(
            field.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )[0]
        delta_theta = sampled[0] if delta_theta is None else delta_theta + sampled[0]
        delta_phi = sampled[1] if delta_phi is None else delta_phi + sampled[1]

    if delta_theta is not None:
        cos_d = torch.cos(delta_theta).unsqueeze(-1)
        sin_d = torch.sin(delta_theta).unsqueeze(-1)
        bearings = bearings * cos_d + torch.cross(base["meridian"], bearings, dim=-1) * sin_d

    if delta_phi is not None:
        cos_p = torch.cos(delta_phi)
        sin_p = torch.sin(delta_phi)
        bx, by, bz = bearings.unbind(-1)
        bearings = torch.stack([bx * cos_p - by * sin_p, bx * sin_p + by * cos_p, bz], dim=-1)

    if "tilt" in lens.components:
        bearings = skew_rotate(lens.omega, bearings)

    rotation = base["rotation"]
    origin = base["origin"]
    if model.pose is not None:
        key = PoseResidual._key(cam_info.image_name)
        model.register_pose_block(key)
        delta_rotation = skew_rotate(
            model.pose.rotation[key], torch.eye(3, device=bearings.device)
        )
        rotation = rotation @ delta_rotation.transpose(0, 1)
        origin = origin + rotation @ model.pose.translation[key]

    direction = bearings @ rotation.transpose(0, 1)

    if "raxel" in lens.components:
        direction = direction + sampled[2:].permute(1, 2, 0) * base["valid"].unsqueeze(-1)

    norm = direction.norm(dim=-1, keepdim=True)
    direction = direction / norm.clamp_min(1e-20)
    direction = direction * base["valid"].unsqueeze(-1)

    ray_origin = origin.view(1, 1, 3).expand(height, width, 3)
    if "z" in lens.components:
        z_basis = model.spline_basis(base, lens.z_weights.shape[-1])
        profile = bspline_eval_basis(lens.z_weights, z_basis, base["theta01"].shape)[0]
        gauge = bspline_eval(lens.z_weights, base["theta01"].new_zeros(()))[0]
        axis = rotation[:, 2]
        ray_origin = ray_origin + (profile - gauge).unsqueeze(-1) * axis

    return ray_origin.contiguous(), direction.contiguous()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


# ------------------------------------------------------------------------------- tests


@pytest.mark.parametrize("rung", RUNGS)
@pytest.mark.parametrize("size", [(64, 96), (37, 41)])
def test_cpu_cache_is_bit_exact_over_poses_and_resolutions(rung, size):
    """The headline requirement: the cached ray field must equal the un-cached one.

    Two resolutions x three poses x seven rungs, and the tolerance is `torch.equal`, not a
    threshold -- the cache reuses tensors rather than recomputing, so any difference at all
    would mean the split moved an operation, not that it lost precision.
    """
    height, width = size
    base = synthetic_base(height, width, rung)
    cam = _Cam()

    plain = randomized_model(rung)
    plain.ray_cache_enabled = False
    cached = randomized_model(rung)
    cached.ray_cache_enabled = True

    for index in range(len(POSES)):
        view = posed(base, index)
        with torch.no_grad():
            want_origin, want_direction = plain(cam, view, height, width)
            got_origin, got_direction = cached(cam, view, height, width)
        assert torch.equal(want_origin, got_origin), (
            f"{rung} pose {index} {size}: origin differs by "
            f"{max_abs(want_origin, got_origin):.3e}"
        )
        assert torch.equal(want_direction, got_direction), (
            f"{rung} pose {index} {size}: direction differs by "
            f"{max_abs(want_direction, got_direction):.3e}"
        )

    # * And the cache really was populated -- otherwise this test proves nothing.
    assert len(cached._ray_cache) == 1, cached._ray_cache.keys()
    assert len(plain._ray_cache) == 0


@pytest.mark.parametrize("rung", RUNGS)
def test_cpu_split_reproduces_the_pre_split_forward(rung):
    "Non-regression on the refactor itself, against a verbatim copy of the old forward()."
    height, width = 48, 61
    base = synthetic_base(height, width, rung)
    cam = _Cam()
    model = randomized_model(rung)

    for index in range(len(POSES)):
        view = posed(base, index)
        want_origin, want_direction = reference_forward(model, cam, view, height, width)
        model.invalidate_ray_cache()
        for enabled in (False, True):
            model.ray_cache_enabled = enabled
            with torch.no_grad():
                got_origin, got_direction = model(cam, view, height, width)
            assert torch.equal(want_origin, got_origin), f"{rung} cache={enabled} origin"
            assert torch.equal(want_direction, got_direction), f"{rung} cache={enabled} dir"


@pytest.mark.parametrize("rung", ["noncentral", "z_only", "ana"])
def test_cpu_cache_does_not_freeze_the_pose(rung):
    """The correctness line: what is cached is the CAMERA frame, not the world frame.

    Checked two ways, because "the two fields differ" is far too weak -- a cache that froze
    the pose would still differ between rungs and would still look like an image.

    1. The world direction under pose B must be **exactly** pose A's direction carried by
       the relative rotation `R_B R_A^T`. That is only true if the cached tensor lives in
       the camera frame. (Up to float32 renormalization, hence a 1e-6 chord tolerance.)
    2. The non-central origin offset must lie along **pose B's** optical axis, not pose A's.
       This is the one that a naive cache gets wrong: `z(theta)` is a camera-frame scalar,
       but the axis it displaces the origin along rotates with the camera.
    """
    height, width = 52, 71
    base = synthetic_base(height, width, rung)
    cam = _Cam()
    model = randomized_model(rung)
    model.ray_cache_enabled = True

    view_a, view_b = posed(base, 0), posed(base, 1)
    with torch.no_grad():
        origin_a, direction_a = model(cam, view_a, height, width)
        origin_b, direction_b = model(cam, view_b, height, width)

    valid = base["valid"] > 0.5
    assert valid.sum() > 0.3 * valid.numel(), "degenerate: almost no valid pixels"

    relative = view_b["rotation"] @ view_a["rotation"].transpose(0, 1)
    carried = direction_a @ relative.transpose(0, 1)
    chord = (carried - direction_b).norm(dim=-1)[valid].max().item()
    assert chord < 1e-6, f"{rung}: cached field is not in the camera frame ({chord:.3e})"

    # * Sanity: the two poses are genuinely different, so a frozen cache would be caught.
    assert (direction_a - direction_b).norm(dim=-1)[valid].max().item() > 0.5

    if "z" in model.components:
        # * Rebuilt from the cached camera-frame profile and pose B's own axis. Written as
        # * an exact reconstruction rather than "project out the axis and check the residue
        # * is small": `origin_b` is ~4 units from the world origin while the offset is
        # * ~1e-2, so the subtraction cancels and any absolute tolerance would be measuring
        # * float32 round-off on the camera centre, not the axis.
        with torch.no_grad():
            frame = model.camera_frame(cam, view_b, height, width)
        axis_b = view_b["rotation"][:, 2]
        expected = view_b["origin"].view(1, 1, 3).expand(height, width, 3)
        expected = (expected + frame["z_profile"].unsqueeze(-1) * axis_b).contiguous()
        assert torch.equal(origin_b, expected), "origin offset did not use pose B's axis"

        # * ... and pose A's axis would have been visibly wrong, so this is not vacuous.
        axis_a = view_a["rotation"][:, 2]
        wrong = view_b["origin"].view(1, 1, 3) + frame["z_profile"].unsqueeze(-1) * axis_a
        assert (origin_b - wrong).norm(dim=-1).max().item() > 1e-3
    else:
        # * Central rungs keep one centre of projection, per view.
        assert torch.equal(origin_b, view_b["origin"].view(1, 1, 3).expand(height, width, 3))


def test_cpu_camera_frame_ignores_the_pose():
    "`camera_frame()` must not read `rotation` / `origin` at all -- pinned, not assumed."
    height, width = 40, 44
    base = synthetic_base(height, width, "nc")
    cam = _Cam()
    model = randomized_model("noncentral")

    with torch.no_grad():
        frame_a = model.camera_frame(cam, posed(base, 0), height, width)
        frame_b = model.camera_frame(cam, posed(base, 2), height, width)
    assert torch.equal(frame_a["bearings"], frame_b["bearings"])
    assert torch.equal(frame_a["z_profile"], frame_b["z_profile"])

    # * A base with no pose keys at all must still work: that is the proof it never reads them.
    bare = dict(base)
    with torch.no_grad():
        frame_c = model.camera_frame(cam, bare, height, width)
    assert torch.equal(frame_a["bearings"], frame_c["bearings"])


def test_cpu_two_cameras_do_not_share_a_cache_entry():
    """Different lens uid -> different residual -> different entry.

    A rig renders alternating cameras, so a uid-blind cache would hand camera 2 camera 1's
    residual on every second view. That is the same failure the base-bearing cache already
    paid for once (`workshop_immervision`, 12 dB), one level up.
    """
    height, width = 36, 48
    model = randomized_model("noncentral")
    model.ray_cache_enabled = True

    lens_two = model.lens(1)
    with torch.no_grad():
        lens_two.theta_weights.copy_(model.lens(0).theta_weights * -3.0)
        lens_two.z_weights.copy_(model.lens(0).z_weights * -3.0)
    model.invalidate_ray_cache()

    base = synthetic_base(height, width, "rig")
    view = posed(base, 0)
    with torch.no_grad():
        origin_one, direction_one = model(_Cam(uid=0), view, height, width)
        origin_two, direction_two = model(_Cam(uid=1), view, height, width)
    assert len(model._ray_cache) == 2
    assert not torch.equal(direction_one, direction_two)
    assert not torch.equal(origin_one, origin_two)

    # * ... and each still matches its own un-cached synthesis.
    model.ray_cache_enabled = False
    with torch.no_grad():
        assert torch.equal(direction_one, model(_Cam(uid=0), view, height, width)[1])
        assert torch.equal(direction_two, model(_Cam(uid=1), view, height, width)[1])


def test_cpu_two_eval_models_do_not_share_a_cache_entry():
    """Same uid, same resolution, DIFFERENT base bearings -- the 12 dB bug, one level down.

    `render.py --eval-models pinhole rad_tan_thin_prism_fisheye` renders two camera models
    for one camera uid in a single process, and on `workshop_immervision` both are
    1440x1080. `Raytracer.base_bearings()` was fixed to key on the model and the intrinsics
    (IMPLEMENTATION.md, "the stale-bearing collision"); the ray cache added on top of it must
    inherit that separation rather than re-introduce the collision one level further down.
    Here the two bases differ only in their `cache_key`, exactly as two eval models do.
    """
    height, width = 40, 52
    model = randomized_model("noncentral")
    model.ray_cache_enabled = True
    cam = _Cam()

    fisheye = synthetic_base(height, width, "rttpf")
    pinhole = synthetic_base(height, width, "pinhole")
    # * A different lens really does give different bearings for the same pixel grid.
    with torch.no_grad():
        pinhole["bearings"] = pinhole["bearings"] * 0.0 + torch.tensor([0.0, 0.0, 1.0])
        pinhole["bearings"] *= pinhole["valid"].unsqueeze(-1)

    with torch.no_grad():
        _, direction_fisheye = model(cam, posed(fisheye, 0), height, width)
        _, direction_pinhole = model(cam, posed(pinhole, 0), height, width)
    assert len(model._ray_cache) == 2, "the two eval models collided on one cache entry"

    valid = fisheye["valid"] > 0.5
    assert (direction_fisheye - direction_pinhole).norm(dim=-1)[valid].max().item() > 0.5

    # * Each still equals its own un-cached synthesis, in either render order.
    model.ray_cache_enabled = False
    model.invalidate_ray_cache()
    with torch.no_grad():
        assert torch.equal(direction_pinhole, model(cam, posed(pinhole, 0), height, width)[1])
        assert torch.equal(direction_fisheye, model(cam, posed(fisheye, 0), height, width)[1])


def test_cpu_two_resolutions_do_not_share_a_cache_entry():
    "The warmup renders at half resolution; the tables are resolution-shaped."
    model = randomized_model("noncentral")
    model.ray_cache_enabled = True
    cam = _Cam()
    for height, width in [(32, 40), (64, 80)]:
        base = synthetic_base(height, width, "res")
        with torch.no_grad():
            origin, direction = model(cam, posed(base, 0), height, width)
        assert direction.shape == (height, width, 3)
        assert origin.shape == (height, width, 3)
    assert len(model._ray_cache) == 2


def test_cpu_optimizer_step_invalidates_the_cache():
    """A moved parameter must not be able to hide behind the cache.

    This is the failure mode that would be worst: training previews run under `no_grad`, so
    they populate the cache between optimizer steps. If `step()` did not clear it, every
    preview after the first would render the residual as it was at unfreeze time -- a slowly
    growing, entirely silent error in exactly the logged numbers used to judge convergence.
    """
    height, width = 40, 52
    base = synthetic_base(height, width, "step")
    view = posed(base, 0)
    cam = _Cam()
    model = randomized_model("noncentral")
    model.ray_cache_enabled = True

    with torch.no_grad():
        _, before = model(cam, view, height, width)
    assert len(model._ray_cache) == 1

    lens = model.lens(0)
    model.set_frozen(False)
    with torch.no_grad():
        lens.theta_weights.add_(2e-3)
        lens.z_weights.add_(5e-3)
    # * Populate the cache again *after* the mutation but *before* the step, so that the
    # * step is the only thing that can clear it.
    with torch.no_grad():
        model(cam, view, height, width)
    model.step()
    assert len(model._ray_cache) == 0, "step() did not invalidate the ray cache"

    with torch.no_grad():
        _, after = model(cam, view, height, width)
    assert not torch.equal(before, after), "the mutation did not reach the rays"

    model.ray_cache_enabled = False
    model.invalidate_ray_cache()
    with torch.no_grad():
        _, uncached = model(cam, view, height, width)
    assert torch.equal(after, uncached)


def test_cpu_load_state_dict_invalidates_the_cache():
    "render.py builds the model, then loads a checkpoint into it."
    height, width = 40, 52
    base = synthetic_base(height, width, "load")
    view = posed(base, 0)
    cam = _Cam()

    trained = randomized_model("noncentral", seed=11)
    fresh = randomized_model("noncentral", seed=3)
    fresh.ray_cache_enabled = True

    with torch.no_grad():
        _, stale = fresh(cam, view, height, width)
    assert len(fresh._ray_cache) == 1

    fresh.load_state_dict(trained.state_dict())
    assert len(fresh._ray_cache) == 0, "load_state_dict did not invalidate the ray cache"

    with torch.no_grad():
        _, loaded = fresh(cam, view, height, width)
        want = trained(cam, view, height, width)[1]
    assert not torch.equal(stale, loaded)
    assert torch.equal(loaded, want)


def test_cpu_training_path_is_never_cached():
    """Under autograd the synthesis must be recomputed, graph and all.

    Two reasons, both fatal if got wrong: the cached tensors carry a graph that a previous
    iteration's backward has already freed, and the parameters move every step.
    """
    height, width = 40, 52
    base = synthetic_base(height, width, "train")
    view = posed(base, 0)
    cam = _Cam()
    model = randomized_model("noncentral")
    model.ray_cache_enabled = True

    origin, direction = model(cam, view, height, width)
    assert len(model._ray_cache) == 0, "the training path populated the ray cache"
    assert direction.requires_grad and origin.requires_grad

    # * A gradient still reaches the parameters through the split.
    (direction.square().sum() + origin.square().sum()).backward()
    lens = model.lens(0)
    for name in ("omega", "theta_weights", "phi_weights", "z_weights"):
        gradient = getattr(lens, name).grad
        assert gradient is not None, f"no gradient reached {name}"
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0.0, f"gradient on {name} is all zero"


def test_cpu_invalid_pixels_stay_exactly_zero_through_the_cache():
    "The zero sentinel outside the lens disk is what tells the raygen to skip the pixel."
    height, width = 44, 60
    base = synthetic_base(height, width, "sentinel")
    invalid = base["valid"] < 0.5
    assert invalid.sum() > 0, "degenerate: the synthetic disk fills the frame"

    for rung in RUNGS:
        model = randomized_model(rung)
        model.ray_cache_enabled = True
        with torch.no_grad():
            for index in range(len(POSES)):
                _, direction = model(_Cam(), posed(base, index), height, width)
                assert direction[invalid].abs().max().item() == 0.0, rung


def test_cpu_pose_residual_stays_per_view_under_the_cache():
    """`pose_opt` is per image; it must survive on the per-image side of the split.

    Same camera, same resolution -- so one cache entry -- but two image names, hence two
    SE(3) blocks. A cache that swallowed the pose residual would give both views the first
    one's correction and would look like `pose_opt` simply not working.
    """
    height, width = 36, 48
    base = synthetic_base(height, width, "pose")
    view = posed(base, 0)

    cfg = RaytracerConfig(camera_opt="noncentral", pose_opt=True)
    model = CameraModel(cfg, device="cpu")
    model.ray_cache_enabled = True
    lens = model.lens(0)
    with torch.no_grad():
        lens.theta_weights.add_(3e-3)
        lens.z_weights.add_(6e-3)
    model.invalidate_ray_cache()

    cam_a, cam_b = _Cam(image_name="a.png"), _Cam(image_name="b.png")
    with torch.no_grad():
        model(cam_a, view, height, width)
        model(cam_b, view, height, width)
        # * Give view B a real pose correction, then re-render both.
        model.pose.rotation[PoseResidual._key("b.png")].add_(0.05)
        model.pose.translation[PoseResidual._key("b.png")].add_(0.5)
        _, direction_a = model(cam_a, view, height, width)
        origin_a, _ = model(cam_a, view, height, width)
        origin_b, direction_b = model(cam_b, view, height, width)

    assert len(model._ray_cache) == 1, "the pose residual leaked into the camera-frame cache"
    valid = base["valid"] > 0.5
    assert (direction_a - direction_b).norm(dim=-1)[valid].max().item() > 1e-3
    assert (origin_a - origin_b).norm(dim=-1).max().item() > 1e-3


# ------------------------------------------------------------------------- GPU coverage


@gpu_only
@pytest.mark.parametrize("rung", ["noncentral", "z_only"])
def test_gpu_cache_is_bit_exact_through_the_real_probe(rung):
    """The same bit-exactness claim, but on the bearings the raygen actually emits.

    The CPU tests feed a synthetic base dict; this one goes through
    `Raytracer.base_bearings()`, i.e. a real probe render with the real rttpf calibration,
    at two render resolutions and three poses.
    """
    import dataclasses

    from tests.test_camera_model import HEIGHT, WIDTH, build_scene, fisheye_camera

    raytracer = build_scene(rung)
    model = raytracer.camera_model
    lens = model.lens(0)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        lens.theta_weights.copy_((torch.randn(lens.theta_weights.shape, generator=generator) * 4e-3).cuda())
        lens.phi_weights.copy_((torch.randn(lens.phi_weights.shape, generator=generator) * 3e-3).cuda())
        lens.z_weights.copy_((torch.randn(lens.z_weights.shape, generator=generator) * 6e-3).cuda())
        lens.omega.copy_((torch.randn(3, generator=generator) * 1e-3).cuda())
    model.invalidate_ray_cache()

    camera = fisheye_camera()
    for height, width in [(HEIGHT, WIDTH), (HEIGHT // 2, WIDTH // 2)]:
        raytracer.set_render_resolution(width, height)
        base = dict(raytracer.base_bearings(camera))
        for index in range(3):
            axis, angle, origin = POSES[index]
            view = dict(base)
            view["rotation"] = rotation_matrix(axis, angle, "cuda")
            view["origin"] = torch.tensor(origin, device="cuda")

            model.ray_cache_enabled = False
            model.invalidate_ray_cache()
            with torch.no_grad():
                want = model(camera, view, height, width)
            model.ray_cache_enabled = True
            with torch.no_grad():
                got = model(camera, view, height, width)
            assert torch.equal(want[0], got[0]), f"{rung} {height}x{width} pose {index} origin"
            assert torch.equal(want[1], got[1]), f"{rung} {height}x{width} pose {index} dir"
        assert len(model._ray_cache) == 1
        model.invalidate_ray_cache()


@gpu_only
def test_gpu_rendered_image_is_unchanged_by_the_cache():
    "End to end: same checkpoint, same view, cache on and off -> identical pixels."
    from tests.test_camera_model import HEIGHT, WIDTH, build_scene, fisheye_camera

    raytracer = build_scene("noncentral")
    model = raytracer.camera_model
    lens = model.lens(0)
    with torch.no_grad():
        lens.z_weights.normal_(0.0, 5e-3)
        lens.theta_weights.normal_(0.0, 4e-3)
    model.invalidate_ray_cache()

    camera = fisheye_camera()
    model.ray_cache_enabled = False
    with torch.no_grad():
        first = raytracer(camera).clone()
        # * Render twice with the cache on: the second render is the one that reads it.
        # * The warm-up MUST go through `raytracer(...)`, not through `model(...)` on a raw
        # * `base_bearings()` dict. `base_bearings` returns the pose-INDEPENDENT half only;
        # * `Raytracer.set_camera` is what injects `base["rotation"]` and `base["origin"]`
        # * before calling the model, so calling the model directly raises KeyError on the
        # * first of those. Going through the raytracer also means the cache is populated by
        # * exactly the production path, which is the thing under test.
        model.ray_cache_enabled = True
        raytracer(camera)
        second = raytracer(camera).clone()
    assert torch.equal(first, second)
