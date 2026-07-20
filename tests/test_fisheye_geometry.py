import numpy as np

from gray.fisheye_geometry import (
    opencv_fisheye_project,
    opencv_fisheye_unproject,
    rad_tan_thin_prism_project,
    rad_tan_thin_prism_unproject,
    thin_prism_project,
    thin_prism_unproject,
)


def assert_roundtrip(project, unproject, params):
    pix_u = np.array([2736.0, 2200.0, 3200.0, 2736.0])
    pix_v = np.array([1824.0, 1700.0, 2100.0, 1400.0])

    dirs, valid = unproject(pix_u, pix_v, params)
    reproj_u, reproj_v = project(dirs, params)
    err = np.sqrt((reproj_u - pix_u) ** 2 + (reproj_v - pix_v) ** 2)

    assert valid.all()
    assert float(err.max()) < 1e-5


def test_opencv_fisheye_roundtrip():
    params = np.array(
        [
            1241.0,
            1243.0,
            2736.0,
            1824.0,
            -0.034688,
            0.002964,
            -0.003554,
            0.000424,
        ],
        dtype=np.float64,
    )

    assert_roundtrip(opencv_fisheye_project, opencv_fisheye_unproject, params)


def test_thin_prism_fisheye_roundtrip():
    params = np.array(
        [
            1241.0,
            1243.0,
            2736.0,
            1824.0,
            -0.034688,
            0.002964,
            1e-4,
            -2e-4,
            -0.003554,
            0.000424,
            3e-4,
            -1e-4,
        ],
        dtype=np.float64,
    )

    assert_roundtrip(thin_prism_project, thin_prism_unproject, params)


def test_rad_tan_thin_prism_fisheye_roundtrip():
    params = np.array(
        [
            1241.3125623286894,
            1243.6962226027811,
            2736.0,
            1824.0,
            -0.033308519261349132,
            -0.0028984836239434328,
            0.00047213393519698443,
            -0.00072136713298229813,
            0.0001154597536915651,
            -1.2650572586498068e-05,
            0.00069127200937642373,
            -0.0032152956069817014,
            -0.0021713573624256993,
            -0.00021356467113650191,
            0.010695842998483789,
            0.0018589853872376905,
        ],
        dtype=np.float64,
    )

    assert_roundtrip(rad_tan_thin_prism_project, rad_tan_thin_prism_unproject, params)
