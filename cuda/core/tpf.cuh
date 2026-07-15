#pragma once

#ifdef __CUDACC__
#include "../utils/vec_math.h"

// * COLMAP ThinPrismFisheyeCameraModel (params: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1).
// * CamFromImg: FisheyeFromImg -> IterativeUndistortion -> NormalFromFisheye.
// * Returns a unit bearing in the OpenCV camera frame (x right, y down, z forward).
// * Returns (0, 0, 0) for invalid pixels (theta >= 90 deg or failed undistortion).

__device__ __forceinline__ void thin_prism_fisheye_distortion(const float *extra_params, const float u,
                                                              const float v, float *du, float *dv) {
    const float k1 = extra_params[0];
    const float k2 = extra_params[1];
    const float p1 = extra_params[2];
    const float p2 = extra_params[3];
    const float k3 = extra_params[4];
    const float k4 = extra_params[5];
    const float sx1 = extra_params[6];
    const float sy1 = extra_params[7];

    const float u2 = u * u;
    const float uv = u * v;
    const float v2 = v * v;
    const float r2 = u2 + v2;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float r8 = r4 * r4;
    const float radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    *du = u * radial + 2.0f * p1 * uv + p2 * (r2 + 2.0f * u2) + sx1 * r2;
    *dv = v * radial + 2.0f * p2 * uv + p1 * (r2 + 2.0f * v2) + sy1 * r2;
}

__device__ __forceinline__ void thin_prism_normal_from_fisheye(const float uu, const float vv, float *u, float *v) {
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

__device__ __forceinline__ float3 thin_prism_fisheye_unproject(const float *params, float x, float y) {
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
        thin_prism_fisheye_distortion(extra_params, uu, vv, &du, &dv);
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

    // --- Verify if the convergence is valid ---
    float final_du, final_dv;
    thin_prism_fisheye_distortion(extra_params, uu, vv, &final_du, &final_dv);
    const float reprojected_u = uu + final_du;
    const float reprojected_v = vv + final_dv;
    const float err_u = reprojected_u - uu0;
    const float err_v = reprojected_v - vv0;
    constexpr float kMaxReprojectionErrorSq = 1e-5f;

    if ((err_u * err_u + err_v * err_v) > kMaxReprojectionErrorSq) {
        return make_float3(0.0f, 0.0f, 0.0f); // Reject false convergence
    }

    // * Equidistant fisheye radius theta = ||(uu, vv)||
    const float theta = sqrtf(uu * uu + vv * vv);
    constexpr float kMaxTheta = 1.5707963f; // pi / 2
    if (theta >= kMaxTheta) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // * NormalFromFisheye -> pinhole-normalized coords, then unit bearing
    float u, v;
    thin_prism_normal_from_fisheye(uu, vv, &u, &v);
    const float inv_norm = rsqrtf(fmaxf(u * u + v * v + 1.0f, 1e-20f));
    return make_float3(u * inv_norm, v * inv_norm, inv_norm);
}
#endif
