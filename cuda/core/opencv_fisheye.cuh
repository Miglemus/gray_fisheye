#pragma once

#ifdef __CUDACC__
#include "../utils/vec_math.h"

// * Inverse of the COLMAP/OpenCV OPENCV_FISHEYE projection.
// * Given a pixel (u, v) and intrinsics (fx, fy, cx, cy, k1..k4), returns the unit
// * bearing in the OpenCV camera frame (x right, y down, z forward).
// * Returns (0, 0, 0) for pixels outside the valid field of view (theta >= 90 deg),
// * i.e. the black vignette corners of a circular fisheye; the caller treats a
// * zero-length direction as an inactive pixel that renders the background.
__device__ __forceinline__ float3 opencv_fisheye_unproject(const float *params, float u, float v) {
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

    constexpr float FISHEYE_MAX_THETA = 1.5707963f; // 90 degrés

    // * Step 1: Mathematical existence of a root check ***
    // f(0) = -theta_d is ALWAYS negative.
    // For a root to exist in [0, pi/2], f(pi/2) MUST be positive. (Assuming monotonic function over [0, pi/2])
    float t2 = FISHEYE_MAX_THETA * FISHEYE_MAX_THETA;
    float t4 = t2 * t2;
    float t6 = t4 * t2;
    float t8 = t4 * t4;
    float f_high = FISHEYE_MAX_THETA * (1.0f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d;

    // If f(pi/2) <= 0, there is no valid root in the optical dome (pixel outside field of view / vignette)
    if (f_high <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // *** ÉTAPE 2 : Algorithme de Bissection (Dichotomie) ***
    float low = 0.0f;
    float high = FISHEYE_MAX_THETA;
    float theta = 0.0f;

    // 20 itérations donnent une précision de pi / (2^21) ~= 7.5e-7 radians.
    // Un nombre d'itérations fixe empêche la divergence de warp sur le GPU.
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        theta = 0.5f * (low + high);
        
        t2 = theta * theta;
        t4 = t2 * t2;
        t6 = t4 * t2;
        t8 = t4 * t4;
        float f = theta * (1.0f + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d;

        if (f > 0.0f) {
            high = theta; // La racine est dans la moitié inférieure
        } else {
            low = theta;  // La racine est dans la moitié supérieure
        }
    }
    
    // Valeur finale convergée
    theta = 0.5f * (low + high);

    // * Calcul du vecteur unitaire directionnel (stable et garanti sans oscillation)
    float sin_theta = sinf(theta);
    float inv_theta_d = 1.0f / theta_d;
    return make_float3(sin_theta * xd * inv_theta_d, sin_theta * yd * inv_theta_d, cosf(theta));
}
#endif