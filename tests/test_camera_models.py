import json
from pathlib import Path

import pytest

from gray.camera_models import (
    CAMERA_PARAM_KEYS,
    gray_models_equal,
    gray_model_from_colmap,
    normalize_gray_model,
    normalize_param_key,
    param_key_to_colmap_model,
)
from gray.config import Config
from run_colmap_fixed import load_config


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("pinhole", "pinhole"),
        ("PINHOLE", "pinhole"),
        ("opencv_fisheye", "opencv_fisheye"),
        ("OPENCV_FISHEYE", "opencv_fisheye"),
        ("thin_prism_fisheye", "thin_prism_fisheye"),
        ("THIN_PRISM_FISHEYE", "thin_prism_fisheye"),
    ],
)
def test_normalize_gray_model_accepts_aliases(raw, expected):
    assert normalize_gray_model(raw) == expected
    assert gray_model_from_colmap(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("opencv_fisheye", "opencv_fisheye"),
        ("OPENCV_FISHEYE", "opencv_fisheye"),
        ("opencv", "opencv"),
        ("OPENCV", "opencv"),
    ],
)
def test_normalize_param_key_accepts_aliases(raw, expected):
    assert normalize_param_key(raw) == expected
    assert param_key_to_colmap_model(raw) in {"OPENCV_FISHEYE", "OPENCV"}


def test_load_config_accepts_uppercase_model_in_params_json(tmp_path):
    params = {
        "fx": 1000.0,
        "fy": 1000.0,
        "cx": 500.0,
        "cy": 500.0,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "model": "OPENCV_FISHEYE",
    }
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))

    camera = load_config(path)
    assert camera.model == "opencv_fisheye"
    assert len(camera.params) == len(CAMERA_PARAM_KEYS["opencv_fisheye"])


def test_config_normalizes_camera_model_case(tmp_path):
    cfg = Config(
        source_path="data/scene",
        model_path=str(tmp_path / "model"),
        camera_model="THIN_PRISM_FISHEYE",
        eval_modes=["PINHOLE", "OPENCV_FISHEYE"],
    )
    assert cfg.camera_model == "thin_prism_fisheye"
    assert cfg.eval_modes == ["pinhole", "opencv_fisheye"]


def test_gray_models_equal_is_case_insensitive():
    assert gray_models_equal("OPENCV_FISHEYE", "opencv_fisheye")
