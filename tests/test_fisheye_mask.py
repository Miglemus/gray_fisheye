import numpy as np

from gray.fisheye_mask import geometric_valid_mask_opencv_fisheye


# * transmission_fe intrinsics scaled to downsample 4 (1296 x 864)
W, H = 1296, 864
INTRINSICS = np.array(
    [
        1897.7229661849062 / 4,  # fx
        1904.1002799142821 / 4,  # fy
        2592 / 4,  # cx
        1728 / 4,  # cy
        -0.034688,
        0.002964,
        -0.003554,
        0.000424,
    ],
    dtype=np.float64,
)


def test_geometric_mask_center_valid_corners_invalid():
    mask = geometric_valid_mask_opencv_fisheye(INTRINSICS, H, W, device="cpu")
    assert mask.shape == (H, W)

    # * Image center is well within the lens disk
    assert bool(mask[H // 2, W // 2])

    # * The four corners invert to theta >= 90 deg for this circular fisheye
    assert not bool(mask[0, 0])
    assert not bool(mask[0, W - 1])
    assert not bool(mask[H - 1, 0])
    assert not bool(mask[H - 1, W - 1])


def test_radius_scale_shrinks_valid_region():
    baseline = geometric_valid_mask_opencv_fisheye(INTRINSICS, H, W, device="cpu", radius_scale=1.0)
    aggressive = geometric_valid_mask_opencv_fisheye(INTRINSICS, H, W, device="cpu", radius_scale=0.9)
    larger = geometric_valid_mask_opencv_fisheye(INTRINSICS, H, W, device="cpu", radius_scale=1.1)

    # * Smaller radius_scale masks more pixels; larger keeps more
    assert aggressive.sum() < baseline.sum()
    assert larger.sum() > baseline.sum()

    # * Aggressive mask is a strict subset of the baseline disk
    assert bool((baseline | aggressive).eq(baseline).all())
