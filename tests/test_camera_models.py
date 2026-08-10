import json

import numpy as np
import pytest

from gray.camera_models import (
    CAMERA_PARAM_KEYS,
    GrayCameraModelClass,
    gray_models_equal,
    gray_model_from_colmap,
    is_fisheye_gray_model,
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
        ("rad_tan_thin_prism_fisheye", "rad_tan_thin_prism_fisheye"),
        ("RAD_TAN_THIN_PRISM_FISHEYE", "rad_tan_thin_prism_fisheye"),
        ("equirectangular", "equirectangular"),
        ("EQUIRECTANGULAR", "equirectangular"),
    ],
)
def test_normalize_gray_model_accepts_aliases(raw, expected):
    assert normalize_gray_model(raw) == expected
    assert gray_model_from_colmap(raw) == expected


def test_equirectangular_is_not_a_fisheye_and_has_no_colmap_model():
    """ERP is a first-class render model but COLMAP cannot express it."""
    model = GrayCameraModelClass("equirectangular")
    assert not model.is_fisheye(), "ERP must not take the fisheye sparse/images path or disk mask"
    assert not is_fisheye_gray_model("equirectangular")
    assert CAMERA_PARAM_KEYS["equirectangular"] == (), "ERP is defined by the render resolution"
    with pytest.raises(ValueError, match="no COLMAP counterpart"):
        param_key_to_colmap_model("equirectangular")


def test_equirectangular_intrinsics_json_needs_no_parameters(tmp_path):
    # * gray.camera_config is the loader render.py uses via --intrinsics. run_colmap_fixed keeps
    # * its own copy for COLMAP runs, which an ERP camera can never come out of.
    from gray.camera_config import load_config as load_camera_config

    path = tmp_path / "erp.json"
    path.write_text(json.dumps({"model": "equirectangular"}))

    camera = load_camera_config(path)
    assert camera.model == "equirectangular"
    assert camera.intrinsics == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("opencv_fisheye", "opencv_fisheye"),
        ("OPENCV_FISHEYE", "opencv_fisheye"),
        ("RAD_TAN_THIN_PRISM_FISHEYE", "rad_tan_thin_prism_fisheye"),
        ("opencv", "opencv"),
        ("OPENCV", "opencv"),
    ],
)
def test_normalize_param_key_accepts_aliases(raw, expected):
    assert normalize_param_key(raw) == expected
    assert param_key_to_colmap_model(raw) in {"OPENCV_FISHEYE", "OPENCV", "RAD_TAN_THIN_PRISM_FISHEYE"}


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
    assert len(camera.intrinsics) == len(CAMERA_PARAM_KEYS["opencv_fisheye"])


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


def test_intrinsics_cuda_refreshes_after_override():
    from gray.camera import CameraInfo

    cam = CameraInfo(
        uid=0,
        R=np.eye(3),
        T=np.zeros(3),
        origin=np.zeros(3),
        fov_y=1.0,
        fov_x=1.0,
        image_path="",
        image_name="test",
        image_width=100,
        image_height=100,
        is_test=False,
        model="thin_prism_fisheye",
        intrinsics=np.ones(12, dtype=np.float64),
    )
    assert cam.intrinsics_cuda().numel() == 12

    cam.intrinsics = np.ones(8, dtype=np.float64)
    cam.model = "opencv_fisheye"
    assert cam.intrinsics_cuda().numel() == 8

    cam.intrinsics = np.ones(16, dtype=np.float64)
    cam.model = "rad_tan_thin_prism_fisheye"
    assert cam.intrinsics_cuda().numel() == 16
