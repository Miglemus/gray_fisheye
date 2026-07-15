from gray.config import Config
from gray.vignetting import should_apply_vignetting


def test_should_apply_vignetting_only_for_training_camera_model(tmp_path):
    fisheye_cfg = Config(
        source_path=str(tmp_path),
        model_path=str(tmp_path / "model"),
        vignetting_comp=True,
        camera_model="rad_tan_thin_prism_fisheye",
    )
    pinhole_cfg = Config(
        source_path=str(tmp_path),
        model_path=str(tmp_path / "model"),
        vignetting_comp=True,
        camera_model="pinhole",
    )
    disabled_cfg = Config(
        source_path=str(tmp_path),
        model_path=str(tmp_path / "model"),
        vignetting_comp=False,
        camera_model="rad_tan_thin_prism_fisheye",
    )

    assert should_apply_vignetting(
        fisheye_cfg, "rad_tan_thin_prism_fisheye"
    )
    assert not should_apply_vignetting(fisheye_cfg, "pinhole")
    assert not should_apply_vignetting(fisheye_cfg, None)

    assert should_apply_vignetting(pinhole_cfg, "pinhole")
    assert not should_apply_vignetting(pinhole_cfg, "opencv_fisheye")

    assert not should_apply_vignetting(disabled_cfg, "rad_tan_thin_prism_fisheye")
