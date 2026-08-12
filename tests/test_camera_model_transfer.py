"""Tests for `--seed`, `--camera_model_init` and `--camera_model_freeze`.

Everything here except the last test runs on CPU: `gray.config` is deliberately torch-free at
import time and the transfer planner is a pure function over tensors and shapes, so the whole
validation surface (uid correspondence, knot counts, rung equality, z units) is testable
without touching a GPU.

The one test that genuinely needs CUDA -- "a run initialized from its own checkpoint and
frozen synthesizes the same ray field" -- is marked `skipif` and compares the ray field
itself, never PSNR.
"""

import json
import os
import re

import numpy as np
import pytest
import torch

from gray.config import (
    Config,
    apply_camera_model_transfer,
    freeze_camera_model,
    parse_uid_map,
    plan_camera_model_transfer,
    read_source_lens_blocks,
    resolve_checkpoint_path,
    scene_radius_from_cameras_json,
    source_rung_of,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOTS, KNOTS_Z, CHANNELS = 10, 8, 5


def lens_block(uid: int, knots: int = KNOTS, knots_z: int = KNOTS_Z, seed: int = 0):
    "One `LensResidual` worth of tensors, with the exact names a checkpoint carries."
    generator = torch.Generator().manual_seed(seed + uid)
    return {
        "omega": torch.randn(3, generator=generator),
        "theta_weights": torch.randn(CHANNELS, knots, generator=generator),
        "phi_weights": torch.randn(CHANNELS, knots, generator=generator),
        "z_weights": torch.randn(1, knots_z, generator=generator),
    }


def shapes_of(blocks):
    return {uid: {name: tuple(t.shape) for name, t in block.items()} for uid, block in blocks.items()}


def write_checkpoint(directory, blocks, rung, iteration=15000, cameras=None):
    "A minimal run directory: gaussians_*.safetensors + config.json (+ cameras.json)."
    import safetensors.torch

    os.makedirs(directory, exist_ok=True)
    state = {"mean": torch.zeros(1, 3)}
    for uid, block in blocks.items():
        for name, tensor in block.items():
            state[f"camera_model.lenses.{uid}.{name}"] = tensor
    path = os.path.join(directory, f"gaussians_{iteration:05d}.safetensors")
    safetensors.torch.save_file(state, path)
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump({"camera_opt": rung, "camera_opt_knots": KNOTS}, handle)
    if cameras is not None:
        with open(os.path.join(directory, "cameras.json"), "w") as handle:
            json.dump(cameras, handle)
    return path


# ====================================================================== P1.a -- the seed


def test_seed_is_a_config_field_read_from_the_cli():
    "`--seed` must exist, parse as an int, and default to the historical hardcoded 0."
    import tyro

    default = tyro.cli(Config, args=["-s", "/tmp/s", "-m", "/tmp/m"])
    assert default.seed == 0, "the default must stay the value train.py used to hardcode"

    parsed = tyro.cli(Config, args=["-s", "/tmp/s", "-m", "/tmp/m", "--seed", "1234"])
    assert parsed.seed == 1234
    assert isinstance(parsed.seed, int)


def test_seed_lands_in_the_run_config_json(tmp_path):
    """The exact expression train.py writes must carry the seed.

    train.py does `json.dump(vars(cfg), f)` into `<model_path>/config.json`; this reproduces
    that literally rather than asserting on a substring of it.
    """
    import tyro

    cfg = tyro.cli(Config, args=["-s", "/tmp/s", "-m", str(tmp_path), "--seed", "4321"])
    path = tmp_path / "config.json"
    with open(path, "w") as handle:
        json.dump(vars(cfg), handle, indent=4)

    written = json.load(open(path))
    assert written["seed"] == 4321
    # * And the transfer flags are recorded too, so a run's provenance is complete.
    for key in ("camera_model_init", "camera_model_freeze", "camera_model_init_z_scale"):
        assert key in written


def test_train_py_uses_cfg_seed_and_dumps_the_whole_config():
    """train.py must seed FROM the config, and dump the config that contains it.

    A source-level check, because executing train.py needs a GPU. Together with the two
    tests above and `test_set_seeds_controls_the_draws` below, this closes the chain
    cli -> cfg.seed -> set_seeds -> config.json.
    """
    source = open(os.path.join(REPO_ROOT, "train.py")).read()
    assert "set_seeds(cfg.seed)" in source, "train.py no longer seeds from the config"
    assert not re.search(r"set_seeds\(\s*\d", source), "train.py still has a hardcoded seed"
    assert 'cli_json_path = os.path.join(cfg.model_path, "config.json")' in source
    assert "json.dump(vars(cfg), f, indent=4)" in source
    # * The seeding must happen before the scene / point cloud / raytracer are built.
    assert source.index("set_seeds(cfg.seed)") < source.index("SceneInfo.from_colmap")
    assert source.index("set_seeds(cfg.seed)") < source.index("Raytracer.from_point_cloud")


def test_set_seeds_controls_the_draws():
    "`set_seeds` must reach python `random`, numpy AND torch -- all three are used."
    import random

    from gray.utils import set_seeds

    def draw():
        return (random.random(), float(np.random.rand()), float(torch.rand(1)))

    set_seeds(7)
    first = draw()
    set_seeds(7)
    assert draw() == first, "same seed, different draws: the seeding is not wired"
    set_seeds(8)
    second = draw()
    assert second != first, "different seeds gave identical draws"
    # * Every one of the three generators must actually move, not just one of them.
    assert all(a != b for a, b in zip(first, second))


def test_config_json_round_trips_both_ways(tmp_path):
    """render.py / measure_fps.py rebuild a `Config` from a run's config.json.

    Two directions, both of which a new field can break:
      * forward  -- a config.json written by the NEW train.py must reload (no unexpected
        keyword), or every re-render of a fresh run dies at load;
      * backward -- a config.json written BEFORE these fields existed must still reload, or
        the entire back catalogue (`tmp/final`, `tmp/mipnerf360`, `out/`) becomes
        un-rerenderable. That is what pins `seed = 0` as the default: it is the value
        train.py hardcoded, so an old run reloads as the run it actually was.
    """
    import tyro

    cfg = tyro.cli(
        Config, args=["-s", "/tmp/s", "-m", "/tmp/m", "--seed", "7", "--camera_opt", "noncentral"]
    )
    written = json.loads(json.dumps(vars(cfg)))
    reloaded = Config(**written)
    assert reloaded.seed == 7
    assert reloaded.camera_model_init is None and reloaded.camera_model_freeze is False

    legacy = {key: value for key, value in written.items() if key not in ("seed",)}
    for key in (
        "camera_model_init",
        "camera_model_init_uid_map",
        "camera_model_init_rung",
        "camera_model_init_z_scale",
        "camera_model_freeze",
    ):
        legacy.pop(key, None)
    old = Config(**legacy)
    assert old.seed == 0, "the default must reproduce what pre-flag runs actually did"
    assert old.camera_model_init is None and old.camera_model_freeze is False


# ============================================== P1.b -- uid correspondence and shape checks


def test_self_transfer_is_bit_exact():
    "The base case: a checkpoint loaded into a model of the same shape copies verbatim."
    blocks = {1: lens_block(1)}
    plan = plan_camera_model_transfer(
        source_blocks=blocks,
        target_shapes=shapes_of(blocks),
        source_rung="noncentral",
        target_rung="noncentral",
    )
    assert set(plan) == {(1, name) for name in ("omega", "theta_weights", "phi_weights", "z_weights")}
    for (uid, name), tensor in plan.items():
        assert torch.equal(tensor, blocks[uid][name]), f"{name} was not copied verbatim"


def test_uid_mismatch_fails_loudly_instead_of_pairing_by_position():
    "A 2-lens rig into a mono-lens scene must raise, not quietly take the first block."
    source = {1: lens_block(1), 2: lens_block(2)}  # * FullCircle's back-to-back rig
    target = shapes_of({1: lens_block(1)})  # * a myscenes-style single lens
    with pytest.raises(ValueError, match="uid mismatch"):
        plan_camera_model_transfer(source, target, "noncentral", "noncentral")

    # * The other direction fails too: a target lens would be left at zero.
    with pytest.raises(ValueError, match="uid mismatch"):
        plan_camera_model_transfer(
            {1: lens_block(1)}, shapes_of({1: lens_block(1), 2: lens_block(2)}),
            "noncentral", "noncentral",
        )


def test_disjoint_uid_labels_still_need_an_explicit_map():
    "Same number of lenses, different labels: still refused without a map, honoured with one."
    source = {7: lens_block(7)}
    target_blocks = {1: lens_block(1)}
    with pytest.raises(ValueError, match="uid mismatch"):
        plan_camera_model_transfer(source, shapes_of(target_blocks), "noncentral", "noncentral")

    plan = plan_camera_model_transfer(
        source, shapes_of(target_blocks), "noncentral", "noncentral", uid_map={7: 1}
    )
    assert torch.equal(plan[(1, "z_weights")], source[7]["z_weights"])


def test_uid_map_must_cover_every_target_lens():
    source = {1: lens_block(1), 2: lens_block(2)}
    target = shapes_of({1: lens_block(1), 2: lens_block(2)})
    with pytest.raises(ValueError, match="would be left at zero"):
        plan_camera_model_transfer(source, target, "noncentral", "noncentral", uid_map={1: 1})

    with pytest.raises(ValueError, match="not in the checkpoint"):
        plan_camera_model_transfer(source, target, "noncentral", "noncentral", uid_map={3: 1, 2: 2})

    with pytest.raises(ValueError, match="not in this scene"):
        plan_camera_model_transfer(source, target, "noncentral", "noncentral", uid_map={1: 9, 2: 2})

    # * A swap is legal and must be applied in the direction it was written.
    plan = plan_camera_model_transfer(
        source, target, "noncentral", "noncentral", uid_map={1: 2, 2: 1}
    )
    assert torch.equal(plan[(2, "omega")], source[1]["omega"])
    assert torch.equal(plan[(1, "omega")], source[2]["omega"])


def test_knot_count_mismatch_fails_instead_of_truncating():
    """`central_matched` allocates 18 knots, every other rung 10. 18 into 10 must raise.

    Truncation would silently change the function the spline represents.
    """
    source = {1: lens_block(1, knots=KNOTS + KNOTS_Z)}  # * central_matched: [5, 18]
    target = shapes_of({1: lens_block(1, knots=KNOTS)})  # * [5, 10]
    with pytest.raises(ValueError, match="shape mismatch"):
        plan_camera_model_transfer(source, target, "ana", "ana")

    with pytest.raises(ValueError, match="knot count"):
        plan_camera_model_transfer(source, target, "ana", "ana")

    # * and the z spline's own knot count is checked separately
    source_z = {1: lens_block(1, knots_z=KNOTS_Z + 4)}
    with pytest.raises(ValueError, match="camera_opt_knots_z"):
        plan_camera_model_transfer(
            source_z, shapes_of({1: lens_block(1)}), "noncentral", "noncentral"
        )


def test_rung_mismatch_fails():
    """The rung is NOT in the checkpoint, so it has to be checked against the config.

    `noncentral` and `z_only` write byte-identical schemas; loading one under the other
    would leave the central channels uninitialized (or silently drop trained ones).
    """
    blocks = {1: lens_block(1)}
    with pytest.raises(ValueError, match="rung mismatch"):
        plan_camera_model_transfer(blocks, shapes_of(blocks), "noncentral", "z_only")
    with pytest.raises(ValueError, match="rung mismatch"):
        plan_camera_model_transfer(blocks, shapes_of(blocks), "z_only", "noncentral")
    with pytest.raises(ValueError, match="rung mismatch"):
        plan_camera_model_transfer(blocks, shapes_of(blocks), "noncentral", "noncentral_no_ana")
    with pytest.raises(ValueError, match="no camera model to transfer"):
        plan_camera_model_transfer(blocks, shapes_of(blocks), "off", "noncentral")

    # * Rungs that differ in NAME but not in components are still refused only if the
    # * component sets differ; identical sets pass.
    plan = plan_camera_model_transfer(blocks, shapes_of(blocks), "noncentral", "noncentral")
    assert len(plan) == 4


def test_empty_source_fails():
    with pytest.raises(ValueError, match="no `camera_model.lenses"):
        plan_camera_model_transfer({}, shapes_of({1: lens_block(1)}), "noncentral", "noncentral")


# ===================================================================== z units and scaling


def test_z_scale_only_touches_z_weights():
    """z(theta) is in RAW COLMAP world units, so a cross-scene transfer has to rescale it.

    The angular channels are in radians and must NEVER be rescaled -- an angle is an angle
    in any reconstruction.
    """
    blocks = {1: lens_block(1)}
    plan = plan_camera_model_transfer(
        blocks, shapes_of(blocks), "noncentral", "noncentral", z_scale=3.0
    )
    assert torch.allclose(plan[(1, "z_weights")], blocks[1]["z_weights"] * 3.0)
    for name in ("omega", "theta_weights", "phi_weights"):
        assert torch.equal(plan[(1, name)], blocks[1][name]), f"{name} must not be rescaled"


def test_scene_radius_from_cameras_json_matches_gray_own_formula():
    """The z rescaling factor is derived from cameras.json; it must equal what train.py uses.

    train.py feeds `scene.point_cloud.radius` = `get_nerf_pp_norm(train_cameras)["radius"]`
    to `camera_model.scene_scale`. Recomputing it from cameras.json (rather than storing it)
    is what makes the conversion available for the runs already on disk.
    """
    from gray.utils import get_nerf_pp_norm, get_world2view

    generator = np.random.default_rng(0)
    cameras = []
    for index in range(12):
        axis = generator.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(generator.uniform(0, 2))
        cross = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
        origin = generator.normal(size=3) * 5.0
        translation = -rotation.T @ origin
        world2view = get_world2view(rotation, translation)
        recovered = np.linalg.inv(world2view)[:3, 3]
        assert np.allclose(recovered, origin, atol=1e-9)
        cameras.append(
            {
                "R": rotation.tolist(),
                "T": translation.tolist(),
                "origin": origin.tolist(),
                "is_test": index % 8 == 0,
            }
        )

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "cameras.json")
        json.dump(cameras, open(path, "w"))
        measured = scene_radius_from_cameras_json(path)

    class _Cam:
        pass

    train = []
    for camera in cameras:
        if camera["is_test"]:
            continue
        cam = _Cam()
        cam.R = np.array(camera["R"])
        cam.T = np.array(camera["T"])
        train.append(cam)
    expected = get_nerf_pp_norm(train)["radius"]
    assert abs(measured - expected) < 1e-9 * max(1.0, expected)


# ================================================================ file-level plumbing


def test_resolve_checkpoint_path_and_source_rung(tmp_path):
    run = tmp_path / "run"
    write_checkpoint(str(run), {1: lens_block(1)}, "noncentral", iteration=7500)
    write_checkpoint(str(run), {1: lens_block(1)}, "noncentral", iteration=15000)

    resolved = resolve_checkpoint_path(str(run))
    assert resolved.endswith("gaussians_15000.safetensors"), "must pick the highest iteration"
    assert resolve_checkpoint_path(resolved) == resolved
    assert source_rung_of(resolved) == "noncentral"

    blocks = read_source_lens_blocks(resolved)
    assert set(blocks) == {1}
    assert set(blocks[1]) == {"omega", "theta_weights", "phi_weights", "z_weights"}

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no gaussians_"):
        resolve_checkpoint_path(str(empty))

    # * No config.json next to the checkpoint -> loud, with the workaround named.
    orphan = tmp_path / "orphan"
    write_checkpoint(str(orphan), {1: lens_block(1)}, "noncentral")
    os.remove(orphan / "config.json")
    with pytest.raises(FileNotFoundError, match="camera_model_init_rung"):
        source_rung_of(str(orphan / "gaussians_15000.safetensors"))


def test_parse_uid_map():
    assert parse_uid_map(None) is None
    assert parse_uid_map("1:1") == {1: 1}
    assert parse_uid_map("1:2, 2:1") == {1: 2, 2: 1}
    for bad in ("", "1", "1:2:3", "a:1", "1:1,1:2", "1:1,2:1"):
        with pytest.raises(ValueError):
            parse_uid_map(bad)


@pytest.mark.parametrize(
    "pattern,expect_uids",
    [("tmp/mipnerf360/*_noncentral", {1}), ("out/fullcircle_rttpf/*_refit_rttpf", {1, 2})],
)
def test_real_checkpoints_on_disk_have_the_expected_lens_layout(pattern, expect_uids):
    """Grounding check against runs actually on disk: uid sets differ between datasets.

    This is the situation `--camera_model_init_uid_map` exists for; it is not hypothetical.
    Skipped when the runs are not present.
    """
    import glob

    runs = sorted(glob.glob(os.path.join(REPO_ROOT, pattern)))
    runs = [run for run in runs if os.path.exists(os.path.join(run, "gaussians_15000.safetensors"))]
    if not runs:
        pytest.skip(f"no runs matching {pattern}")
    blocks = read_source_lens_blocks(os.path.join(runs[0], "gaussians_15000.safetensors"))
    assert set(blocks) == expect_uids
    assert source_rung_of(os.path.join(runs[0], "gaussians_15000.safetensors")) != "off"


def test_real_central_matched_checkpoint_cannot_be_loaded_into_a_ten_knot_model():
    "The 18-vs-10 knot trap, on a real `central_matched` checkpoint if one is on disk."
    import glob

    runs = sorted(glob.glob(os.path.join(REPO_ROOT, "tmp/*/*central_matched*")))
    runs = [run for run in runs if os.path.exists(os.path.join(run, "gaussians_15000.safetensors"))]
    if not runs:
        pytest.skip("no central_matched run on disk")
    checkpoint = os.path.join(runs[0], "gaussians_15000.safetensors")
    source = read_source_lens_blocks(checkpoint)
    uid = next(iter(source))
    assert source[uid]["theta_weights"].shape[-1] == 18, "fixture assumption changed"
    target = shapes_of({uid: lens_block(uid, knots=10)})
    with pytest.raises(ValueError, match="knot count"):
        plan_camera_model_transfer(source, target, "central_matched", "central_matched")


# ============================================ the ray cache the loader has to invalidate


class _StubLens:
    "The four tensors `apply_camera_model_transfer` writes, and nothing else."

    def __init__(self, knots=KNOTS, knots_z=KNOTS_Z):
        self.omega = torch.zeros(3)
        self.theta_weights = torch.zeros(CHANNELS, knots)
        self.phi_weights = torch.zeros(CHANNELS, knots)
        self.z_weights = torch.zeros(1, knots_z)

    def parameters(self):
        return [self.omega, self.theta_weights, self.phi_weights, self.z_weights]


class _StubCameraModel:
    """A CPU stand-in with `CameraModel`'s surface: `lens()`, `lenses`, `optimizer`.

    Exists so the two side-effecting helpers are testable without OptiX. It records
    `invalidate_ray_cache()` calls, which is the whole point: `CameraModel.forward()` caches
    the pose-independent camera frame and only drops it on `step()` / `set_frozen()` / a
    state_dict load, so a loader that writes parameters by hand and stays silent would leave
    every subsequent no-grad render on the PREVIOUS camera.
    """

    def __init__(self, uids=(1,), knots=KNOTS, knots_z=KNOTS_Z):
        self.lenses = {str(uid): _StubLens(knots, knots_z) for uid in uids}
        self.invalidations = 0

        class _Optimizer:
            param_groups = [
                {"params": [], "lr": 0.0, "name": "placeholder"},
                {"params": [], "lr": 1e-4, "name": "tilt:1"},
                {"params": [], "lr": 1e-4, "name": "angular:1"},
                {"params": [], "lr": 1e-4, "name": "z:1"},
                {"params": [], "lr": 1e-5, "name": "poseR:img0"},
                {"params": [], "lr": 1e-5, "name": "poseT:img0"},
            ]

        self.optimizer = _Optimizer()

    def lens(self, uid):
        return self.lenses[str(uid)]

    def invalidate_ray_cache(self):
        self.invalidations += 1


def _transfer_cfg(path, rung="noncentral"):
    class _Cfg:
        camera_model_init = path
        camera_model_init_rung = None
        camera_model_init_uid_map = None
        camera_model_init_z_scale = "none"
        camera_opt = rung

    return _Cfg()


def test_transfer_invalidates_the_camera_frame_cache(tmp_path):
    """Writing parameters by hand MUST drop `CameraModel`'s ray cache.

    Regression guard for a silent-render bug: the cache is keyed on (camera, resolution) and
    is blind to a parameter that changed underneath it. train.py happens to load before the
    first render, so nothing renders stale there today -- this test is what keeps that a
    property of the loader rather than an accident of call order.
    """
    blocks = {1: lens_block(1)}
    checkpoint = write_checkpoint(str(tmp_path / "src"), blocks, "noncentral")
    model = _StubCameraModel(uids=(1,))

    summary = apply_camera_model_transfer(
        model, _transfer_cfg(checkpoint), [1], target_scene_radius=1.0
    )

    assert summary["tensors_loaded"] == 4
    assert model.invalidations >= 1, (
        "apply_camera_model_transfer wrote the lens parameters without dropping the "
        "camera-frame cache; every no-grad render until the next step() would use the OLD "
        "camera and no assertion downstream would see it"
    )
    for name, tensor in blocks[1].items():
        assert torch.equal(getattr(model.lens(1), name), tensor), f"{name} not installed"


def test_transfer_survives_a_camera_model_without_a_ray_cache():
    "The invalidation is best-effort by design: an older model object must still load."

    class _NoCache(_StubCameraModel):
        invalidate_ray_cache = None  # * attribute exists but is not callable

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = write_checkpoint(directory, {1: lens_block(1)}, "noncentral")
        model = _NoCache(uids=(1,))
        summary = apply_camera_model_transfer(
            model, _transfer_cfg(checkpoint), [1], target_scene_radius=1.0
        )
    assert summary["tensors_loaded"] == 4


def test_freeze_drops_only_the_lens_groups(tmp_path):
    """`--camera_model_freeze` takes the lens groups out of Adam and leaves `--pose_opt` in.

    The pose residual is selected by its own flag; freezing the lens must not freeze it, or
    a `--camera_model_freeze --pose_opt` control would silently be measuring nothing.
    """
    model = _StubCameraModel(uids=(1,))
    summary = freeze_camera_model(model, [1])

    names = [group["name"] for group in model.optimizer.param_groups]
    assert names == ["placeholder", "poseR:img0", "poseT:img0"], names
    assert summary["dropped_param_groups"] == 3
    assert summary["frozen_parameters"] == 3 + 2 * CHANNELS * KNOTS + KNOTS_Z
    for parameter in model.lens(1).parameters():
        assert parameter.requires_grad is False


def test_transfer_then_freeze_is_the_headline_combination(tmp_path):
    "The main use case: init from another run AND hold it fixed. Both effects, one model."
    blocks = {1: lens_block(1)}
    checkpoint = write_checkpoint(str(tmp_path / "src"), blocks, "noncentral")
    model = _StubCameraModel(uids=(1,))

    apply_camera_model_transfer(model, _transfer_cfg(checkpoint), [1], target_scene_radius=1.0)
    freeze_camera_model(model, [1])

    for name, tensor in blocks[1].items():
        assert torch.equal(getattr(model.lens(1), name), tensor)
    assert all(
        not str(group["name"]).startswith(("tilt:", "angular:", "z:", "raxel:"))
        for group in model.optimizer.param_groups
    )


# ========================================================= the CUDA end-to-end proof


# * The guard MUST short-circuit before `torch.cuda.is_available()`, and the env var MUST
# * come first. Decorators are evaluated at COLLECTION time, in the pytest PARENT, and this
# * repo runs under `--forked` (pyproject `addopts`). `torch.cuda.is_available()` initialises
# * the CUDA driver in whatever process calls it; once the parent holds a context, every
# * forked child dies with `cudaErrorInitializationError` when it reaches
# * `torch.classes.gray.Raytracer(...)`. That is not hypothetical: with a bare
# * `skipif(not torch.cuda.is_available())` here, this module took the GPU suite from 6
# * pre-existing failures to 35 -- it poisoned every module collected after it, including
# * pure-CPU tests. `test_camera_model_cache.py` already had the right shape; copy it, do not
# * invent a new one.
RUN_GPU = os.environ.get("GRAY_RUN_GPU_TESTS", "") == "1"


@pytest.mark.skipif(
    not (RUN_GPU and torch.cuda.is_available()),
    reason="renders on a GPU (OptiX raytracer); set GRAY_RUN_GPU_TESTS=1 to enable",
)
def test_self_init_and_freeze_reproduce_the_source_ray_field(tmp_path):
    """`--camera_model_init <itself> --camera_model_freeze` must reproduce the SOURCE RAYS.

    Compared on the synthesized ray field -- (origin, direction) per pixel -- not on PSNR:
    PSNR would confound the camera model with the gaussians, and would not detect a
    half-loaded model.

    Two properties in one test:
      1. the loader installs the source weights bit-exactly (the field matches before any
         optimizer step);
      2. `--camera_model_freeze` keeps it there THROUGH a real training step, which is the
         part a shape check cannot prove.
    """
    from tests.test_camera_model import HEIGHT, WIDTH, build_scene, fisheye_camera

    from gray.prelude import Raytracer

    raytracer = build_scene("noncentral")
    camera = fisheye_camera()
    model = raytracer.camera_model
    model.scene_scale = 1.0

    # * A non-trivial residual, so an unloaded / half-loaded model cannot coincide with it.
    lens = model.lens(camera.uid)
    with torch.no_grad():
        lens.omega.normal_(0.0, 1e-3)
        lens.theta_weights.normal_(0.0, 2e-3)
        lens.phi_weights.normal_(0.0, 2e-3)
        lens.z_weights.normal_(0.0, 5e-3)

    def ray_field():
        """Synthesize (origin, direction) the way a render does -- CACHE INCLUDED.

        Deliberately no `invalidate_ray_cache()` in here. `CameraModel.forward()` reuses the
        pose-independent camera frame under `no_grad`, so this helper goes through the exact
        path an eval pass takes, and the test therefore also proves that the loader drops
        that cache. A helper that invalidated first would hide precisely that bug.
        """
        base = dict(raytracer.base_bearings(camera))
        base["rotation"] = Raytracer._rotation_c2w_cuda(camera)
        base["origin"] = camera.origin_cuda()
        with torch.no_grad():
            origin, direction = model(camera, base, HEIGHT, WIDTH)
        return origin.clone(), direction.clone()

    reference_origin, reference_direction = ray_field()
    assert (reference_origin - camera.origin_cuda()).norm(dim=-1).max() > 1e-4, (
        "degenerate fixture: z(theta) is flat, the test would prove nothing"
    )

    run = tmp_path / "source_run"
    os.makedirs(run, exist_ok=True)
    raytracer.save_safetensors(str(run), 15000)
    json.dump({"camera_opt": "noncentral"}, open(run / "config.json", "w"))

    # * Wipe the model: this is the state a fresh run starts from. The explicit invalidation
    # * is the contract `CameraModel.forward()` documents for a by-hand parameter poke -- the
    # * test does it here so that the CACHE IS PRIMED WITH THE ZEROED CAMERA below, which is
    # * the state that makes the loader's own invalidation observable.
    with torch.no_grad():
        for parameter in lens.parameters():
            parameter.zero_()
    model.invalidate_ray_cache()
    zeroed_origin, zeroed_direction = ray_field()
    assert not torch.equal(zeroed_origin, reference_origin)
    assert not torch.equal(zeroed_direction, reference_direction)

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.camera_model_init = str(run)
    cfg.camera_model_init_rung = None
    cfg.camera_model_init_uid_map = None
    cfg.camera_model_init_z_scale = "none"
    cfg.camera_opt = "noncentral"

    summary = apply_camera_model_transfer(model, cfg, [camera.uid], target_scene_radius=1.0)
    assert summary["tensors_loaded"] == 4
    assert summary["source_rung"] == "noncentral"

    # * No manual invalidation here: the cache currently holds the ZEROED camera frame, so a
    # * loader that forgot to drop it would return `zeroed_*` and fail on the next two lines.
    loaded_origin, loaded_direction = ray_field()
    assert torch.equal(loaded_origin, reference_origin), "ray origins differ after the load"
    assert torch.equal(loaded_direction, reference_direction), "bearings differ after the load"

    # * Now freeze and take a real training step: the field must not move at all.
    freeze_summary = freeze_camera_model(model, [camera.uid])
    assert freeze_summary["dropped_param_groups"] >= 1
    assert freeze_summary["frozen_parameters"] == sum(p.numel() for p in lens.parameters())

    model.set_frozen(False)  # * what train.py does once past camera_opt_from_iter
    target = torch.rand((3, HEIGHT, WIDTH), device="cuda")
    render = raytracer(camera)
    loss = torch.nn.functional.l1_loss(render, target)
    raytracer.backward(loss)
    raytracer.step()

    stepped_origin, stepped_direction = ray_field()
    assert torch.equal(stepped_origin, reference_origin), "freeze leaked: origins moved"
    assert torch.equal(stepped_direction, reference_direction), "freeze leaked: bearings moved"
    for parameter in lens.parameters():
        assert parameter.grad is None or parameter.grad.abs().sum().item() == 0.0
