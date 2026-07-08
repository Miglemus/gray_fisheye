#pragma once

#ifdef __CUDACC__
#include "../utils/vec_math.h"

// * COLMAP RadTanThinPrismFisheyeModel
// * Params: fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3.
// * CamFromImg: FisheyeFromImg -> IterativeUndistortion -> NormalFromFisheye.
// * Returns a unit bearing in the OpenCV camera frame (x right, y down, z forward).
// * Returns (0, 0, 0) for invalid pixels (theta >= 90 deg or failed undistortion).

__device__ __forceinline__ void rad_tan_thin_prism_fisheye_distortion(const float *extra_params, const float u,
                                                                      const float v, float *du, float *dv) {
    const float k0 = extra_params[0];
    const float k1 = extra_params[1];
    const float k2 = extra_params[2];
    const float k3 = extra_params[3];
    const float k4 = extra_params[4];
    const float k5 = extra_params[5];
    const float p0 = extra_params[6];
    const float p1 = extra_params[7];
    const float s0 = extra_params[8];
    const float s1 = extra_params[9];
    const float s2 = extra_params[10];
    const float s3 = extra_params[11];

    const float theta2 = u * u + v * v;
    const float theta4 = theta2 * theta2;
    const float theta6 = theta4 * theta2;
    const float theta8 = theta4 * theta4;
    const float theta10 = theta8 * theta2;
    const float theta12 = theta6 * theta6;
    const float th_radial =
        1.0f + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8 + k4 * theta10 + k5 * theta12;

    const float x = th_radial * u;
    const float y = th_radial * v;

    const float x2 = x * x;
    const float y2 = y * y;
    const float xy = x * y;
    const float r2 = x2 + y2;
    const float r4 = r2 * r2;

    const float dx_tang = 2.0f * p1 * xy + p0 * (r2 + 2.0f * x2);
    const float dy_tang = 2.0f * p0 * xy + p1 * (r2 + 2.0f * y2);

    const float dx_tp = s0 * r2 + s1 * r4;
    const float dy_tp = s2 * r2 + s3 * r4;

    const float x_distorted = x + dx_tang + dx_tp;
    const float y_distorted = y + dy_tang + dy_tp;

    *du = x_distorted - u;
    *dv = y_distorted - v;
}

__device__ __forceinline__ void rad_tan_thin_prism_normal_from_fisheye(const float uu, const float vv, float *u,
                                                                      float *v) {
    *u = uu;
    *v = vv;
    const float theta = sqrtf(uu * uu + vv * vv);
    const float theta_cos_theta = theta * cosf(theta);
    if (theta_cos_theta > 1e-8f) {
        const float scale = sinf(theta) / theta_cos_theta;
        *u *= scale;
        *v *= scale;
    }
}

__device__ __forceinline__ float3 rad_tan_thin_prism_fisheye_unproject(const float *params, float x, float y) {
    const float fx = params[0];
    const float fy = params[1];
    const float cx = params[2];
    const float cy = params[3];
    const float *extra_params = params + 4;

    // * FisheyeFromImg
    float uu = (x - cx) / fx;
    float vv = (y - cy) / fy;
    const float uu0 = uu;
    const float vv0 = vv;

    // * IterativeUndistortion: solve uu + Distortion(uu, vv) = (uu0, vv0)
    constexpr int kMaxIterations = 100;
    constexpr float kMinStepSq = 1e-10f;
    bool converged = false;
    for (int i = 0; i < kMaxIterations; ++i) {
        float du, dv;
        rad_tan_thin_prism_fisheye_distortion(extra_params, uu, vv, &du, &dv);
        const float next_uu = uu0 - du;
        const float next_vv = vv0 - dv;
        const float step_u = next_uu - uu;
        const float step_v = next_vv - vv;
        uu = next_uu;
        vv = next_vv;
        if (step_u * step_u + step_v * step_v < kMinStepSq) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float theta = sqrtf(uu * uu + vv * vv);
    constexpr float kMaxTheta = 1.5707963f; // pi / 2
    if (theta >= kMaxTheta) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float u, v;
    rad_tan_thin_prism_normal_from_fisheye(uu, vv, &u, &v);
    const float inv_norm = rsqrtf(fmaxf(u * u + v * v + 1.0f, 1e-20f));
    return make_float3(u * inv_norm, v * inv_norm, inv_norm);
}
#endif
