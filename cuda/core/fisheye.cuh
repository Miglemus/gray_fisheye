#pragma once

#ifdef __CUDACC__
#include "../utils/vec_math.h"

// * Inverse of the COLMAP/OpenCV OPENCV_FISHEYE projection.
// * Given a pixel (u, v) and intrinsics (fx, fy, cx, cy, k1..k4), returns the unit
// * bearing in the OpenCV camera frame (x right, y down, z forward).
// * Returns (0, 0, 0) for pixels outside the valid field of view (theta >= 90 deg),
// * i.e. the black vignette corners of a circular fisheye; the caller treats a
// * zero-length direction as an inactive pixel that renders the background.
__device__ __forceinline__ float3 fisheye_unproject(const float *params, float u, float v) {
    const float fx = params[0];
    const float fy = params[1];
    const float cx = params[2];
    const float cy = params[3];
    const float k1 = params[4];
    const float k2 = params[5];
    const float k3 = params[6];
    const float k4 = params[7];

    // * Distorted normalized coordinates; their radius is the distorted angle theta_d
    float xd = (u - cx) / fx;
    float yd = (v - cy) / fy;
    float theta_d = sqrtf(xd * xd + yd * yd);

    if (theta_d < 1e-8f) {
        return make_float3(0.0f, 0.0f, 1.0f);
    }
    constexpr float FISHEYE_MAX_THETA = 1.5707963f; // 90 degrees

    // * Newton iterations to invert theta_d = theta + k1*t^3 + k2*t^5 + k3*t^7 + k4*t^9
    float f = 0.0f;
    float theta = theta_d;

    // * first evaluate which values of theta_d have no solution
    float t2 = FISHEYE_MAX_THETA * FISHEYE_MAX_THETA;
    float t4 = t2 * t2;
    float t6 = t4 * t2;
    float t8 = t4 * t4;
    float theta_d_max = FISHEYE_MAX_THETA * (1.0f + k1 * t2 + k2 * t4 + k3 * t2 * t6 + k4 * t2 * t8);

    // * if theta_d is too large, return 0
    if (theta_d > theta_d_max) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    for (int i = 0; i < 10; i++) {
        t2 = theta * theta;
        t4 = t2 * t2;
        t6 = t4 * t2;
        t8 = t4 * t4;
        f = theta * (1.0f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d;
        float fp = 1.0f + 3.0f * k1 * t2 + 5.0f * k2 * t4 + 7.0f * k3 * t6 + 9.0f * k4 * t8;
        theta -= f / fp;
        if (theta < 0.0f || theta > FISHEYE_MAX_THETA) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
    }

    // *** Reject pixels beyond the lens' imaged circle (the black vignette corners). theta = atan(r)
    // *** is bounded to [0, pi/2) in the OpenCV fisheye model, so pixels inverting to theta >= 90 deg
    // *** lie outside the valid field of view. This matches the imaged disk of this lens almost
    // *** exactly (the GT is >99% non-black inside 90 deg and transitions to vignette beyond it).
    // *** A zero-length direction is treated by the caller as an inactive (background) pixel.
    if (theta >= FISHEYE_MAX_THETA || fabsf(f) > 1e-6f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // * Unit bearing (sin(theta)*dir, cos(theta)); stable for all valid theta
    float sin_theta = sinf(theta);
    float inv_theta_d = 1.0f / theta_d;
    return make_float3(sin_theta * xd * inv_theta_d, sin_theta * yd * inv_theta_d, cosf(theta));
}
#endif
