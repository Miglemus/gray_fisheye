from pathlib import Path

import pytest

from gray.camera_models import GrayCameraModelClass
from gray.config import Config
from gray.eval import (
    load_eval_views,
    source_mode_for_eval,
    validate_eval_modes,
)


def test_resolved_eval_modes_defaults_and_override(tmp_path):
    pinhole_cfg = Config(source_path="data/scene", model_path=str(tmp_path / "pinhole"))
    assert pinhole_cfg.resolved_eval_modes() == ["pinhole"]

    fisheye_cfg = Config(
        source_path="data/scene",
        model_path=str(tmp_path / "fisheye"),
        camera_model="opencv_fisheye",
    )
    assert fisheye_cfg.resolved_eval_modes() == ["opencv_fisheye"]
    assert fisheye_cfg.images_dir == "input_4"

    tpf_cfg = Config(
        source_path="data/scene",
        model_path=str(tmp_path / "tpf"),
        camera_model="thin_prism_fisheye",
    )
    assert tpf_cfg.resolved_eval_modes() == ["thin_prism_fisheye"]
    assert tpf_cfg.images_dir == "input_4"

    rtpf_cfg = Config(
        source_path="data/scene",
        model_path=str(tmp_path / "rtpf"),
        camera_model="rad_tan_thin_prism_fisheye",
    )
    assert rtpf_cfg.resolved_eval_modes() == ["rad_tan_thin_prism_fisheye"]
    assert rtpf_cfg.images_dir == "input_4"

    dual_cfg = Config(
        source_path="data/scene",
        model_path=str(tmp_path / "dual"),
        camera_model="opencv_fisheye",
        eval_modes=["pinhole", "opencv_fisheye"],
    )
    assert dual_cfg.resolved_eval_modes() == ["pinhole", "opencv_fisheye"]


def test_validate_eval_modes_allows_pinhole_and_colmap_model():
    validate_eval_modes(["pinhole", "thin_prism_fisheye"], "thin_prism_fisheye")
    validate_eval_modes(["pinhole", "rad_tan_thin_prism_fisheye"], "rad_tan_thin_prism_fisheye")
    validate_eval_modes(["pinhole"], "opencv_fisheye")


def test_validate_eval_modes_rejects_unknown_model_without_intrinsics():
    with pytest.raises(ValueError, match="opencv_fisheye"):
        validate_eval_modes(["opencv_fisheye"], "thin_prism_fisheye")


def test_validate_eval_modes_allows_custom_model_with_intrinsics():
    validate_eval_modes(
        ["opencv_fisheye"],
        "thin_prism_fisheye",
        intrinsics_model="opencv_fisheye",
    )


def test_source_mode_for_eval_custom_intrinsics_uses_colmap_model():
    assert (
        source_mode_for_eval(
            "opencv_fisheye",
            "thin_prism_fisheye",
            intrinsics_model="opencv_fisheye",
        )
        == "thin_prism_fisheye"
    )


def test_load_eval_views_rejects_mismatched_sparse_model(tmp_path):
    project_dir = Path(__file__).resolve().parents[1]
    scene_path = project_dir.parents[1] / "data" / "myscenes" / "transmission_fe"
    if not (scene_path / "distorted" / "sparse" / "0").exists():
        pytest.skip("transmission_fe fisheye fixture data is not available")

    cfg = Config(
        source_path=str(scene_path),
        model_path=str(tmp_path / "model"),
        downsampling=4,
        camera_model="thin_prism_fisheye",
    )

    with pytest.raises(ValueError, match="opencv_fisheye"):
        load_eval_views(cfg, GrayCameraModelClass("opencv_fisheye"), load_images=False)


def test_load_eval_views_selects_expected_camera_models_and_masks(tmp_path):
    project_dir = Path(__file__).resolve().parents[1]
    scene_path = project_dir.parents[1] / "data" / "myscenes" / "transmission_fe"
    if not (scene_path / "sparse" / "0").exists() or not (scene_path / "images_4").exists():
        pytest.skip("transmission_fe fixture data is not available")

    cfg = Config(
        source_path=str(scene_path),
        model_path=str(tmp_path / "model"),
        downsampling=4,
    )

    pinhole_views = load_eval_views(cfg, GrayCameraModelClass("pinhole"), load_images=False)
    assert pinhole_views.train_cameras
    assert pinhole_views.test_cameras
    assert pinhole_views.valid_mask is None
    assert pinhole_views.train_cameras[0].model == "pinhole"
    assert "/images_4/" in pinhole_views.train_cameras[0].image_path

    fisheye_views = load_eval_views(cfg, GrayCameraModelClass("opencv_fisheye"), load_images=False)
    assert fisheye_views.train_cameras
    assert fisheye_views.test_cameras
    assert fisheye_views.valid_mask is not None
    assert fisheye_views.train_cameras[0].model == "opencv_fisheye"
    assert "/input_4/" in fisheye_views.train_cameras[0].image_path
