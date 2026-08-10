#pragma once

#include "../utils/vec_math.h"

// * Equirectangular (360 panorama) unprojection.
// *
// * The mapping is fixed by the image dimensions alone -- an ERP camera has no intrinsics:
// *   u = (x + 0.5) / width,  v = (y + 0.5) / height
// *   lon = (u - 0.5) * 2pi   -- u = 0.5 is the camera's forward axis, lon grows towards +x
// *   lat = (0.5 - v) * pi    -- v = 0 is the zenith, v = 1 the nadir
// * and returns a unit bearing in the OpenCV camera frame (x right, y down, z forward), which
// * is the frame every other model in this file returns so the caller can convert uniformly.
// *
// * This is the same convention as the panoramas the repo already produces
// * (scripts/make_fiord_panos.py, scripts/theta/make_erp_pano.py) and the one RaRPano/SPaGS
// * trains on, so an ERP render lands pixel-for-pixel in those panoramas' frame.
// *
// * *** Unlike the fisheye models there is no invalid region and no FOV cut: every pixel maps
// * *** to a bearing, longitude is periodic so u=0 and u=1 give the same direction (no seam),
// * *** and the poles are an oversampling artefact, not a singularity -- the returned vector is
// * *** exactly unit-length at lat = +-pi/2.
__device__ inline float3 equirectangular_unproject(const float u, const float v) {
    constexpr float kPi = 3.14159265358979323846f;
    const float lon = (u - 0.5f) * 2.0f * kPi;
    const float lat = (0.5f - v) * kPi;

    float sin_lon, cos_lon, sin_lat, cos_lat;
    sincosf(lon, &sin_lon, &cos_lon);
    sincosf(lat, &sin_lat, &cos_lat);

    return make_float3(cos_lat * sin_lon, -sin_lat, cos_lat * cos_lon);
}
