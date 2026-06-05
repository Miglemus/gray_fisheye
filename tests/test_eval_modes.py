from pathlib import Path

import pytest

from gray.config import Config
from gray.eval import load_eval_views


def test_resolved_eval_modes_defaults_and_override(tmp_path):
    pinhole_cfg = Config(source_path="data/scene", model_path=str(tmp_path / "pinhole"))
    assert pinhole_cfg.resolved_eval_modes() == ["pinhole"]

    fisheye_cfg = Config(source_path="data/scene", model_path=str(tmp_path / "fisheye"), fisheye=True)
    assert fisheye_cfg.resolved_eval_modes() == ["fisheye"]
    assert fisheye_cfg.images_dir == "input_4"

    dual_cfg = Config(
        source_path="data/scene",
        model_path=str(tmp_path / "dual"),
        eval_modes=["pinhole", "fisheye"],
    )
    assert dual_cfg.resolved_eval_modes() == ["pinhole", "fisheye"]


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

    pinhole_views = load_eval_views(cfg, "pinhole", load_images=False)
    assert pinhole_views.train_cameras
    assert pinhole_views.test_cameras
    assert pinhole_views.valid_mask is None
    assert pinhole_views.train_cameras[0].model == "pinhole"
    assert "/images_4/" in pinhole_views.train_cameras[0].image_path

    fisheye_views = load_eval_views(cfg, "fisheye", load_images=False)
    assert fisheye_views.train_cameras
    assert fisheye_views.test_cameras
    assert fisheye_views.valid_mask is not None
    assert fisheye_views.train_cameras[0].model == "opencv_fisheye"
    assert "/input_4/" in fisheye_views.train_cameras[0].image_path
